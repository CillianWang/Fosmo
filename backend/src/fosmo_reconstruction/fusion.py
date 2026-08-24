"""All-frame Depth Pro inference, TSDF fusion, and Blender mesh export."""

from __future__ import annotations

import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fosmo_scanbundle.validator import validate_bundle

from .geometry import point_cloud_bounds, scale_intrinsics
from .pipeline import (
    ReconstructionError,
    _clockwise_rotation_for_upright,
    _infer_with_loaded_model,
    _load_depth_model,
    _restore_sensor_depth,
    _rotate_image_clockwise,
    _save_point_cloud_preview,
)


FUSION_PIPELINE_VERSION = "fosmo-all-frame-tsdf-1"
CV_FROM_ARKIT_CAMERA = np.diag((1.0, -1.0, -1.0, 1.0))


def world_to_cv_camera_extrinsic(
    world_from_camera: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """Return the Open3D world-to-CV-camera extrinsic for an ARKit pose."""

    pose = np.asarray(world_from_camera, dtype=np.float64)
    if pose.size != 16:
        raise ValueError("world_from_camera must contain 16 row-major values")
    pose = pose.reshape(4, 4)
    return CV_FROM_ARKIT_CAMERA @ np.linalg.inv(pose)


def _depth_cache_path(cache_directory: Path, frame_id: int) -> Path:
    return cache_directory / f"{frame_id:06d}.npy"


def _load_cached_depth(path: Path, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        depth = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    if depth.shape != expected_shape or depth.dtype not in (np.float16, np.float32):
        return None
    values = depth.astype(np.float32)
    if not np.isfinite(values).all() or np.any(values <= 0):
        return None
    return values


def _infer_all_depths(
    bundle: Path,
    frames: list[dict[str, Any]],
    cache_directory: Path,
    checkpoint: Path,
    requested_device: str,
) -> tuple[list[dict[str, Any]], str, str, float]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    frame_results: list[dict[str, Any]] = []
    missing_frames: list[dict[str, Any]] = []
    for frame in frames:
        expected_shape = (frame["image_height"], frame["image_width"])
        cached = _load_cached_depth(_depth_cache_path(cache_directory, frame["id"]), expected_shape)
        if cached is None:
            missing_frames.append(frame)

    model = transform = torch_device = None
    torch_version = "not loaded; all depths came from cache"
    inference_device = "cache"
    model_started = time.perf_counter()
    if missing_frames:
        model, transform, torch_device, torch_version = _load_depth_model(
            checkpoint,
            requested_device,
        )
        inference_device = str(torch_device)

    try:
        for index, frame in enumerate(frames, start=1):
            cache_path = _depth_cache_path(cache_directory, frame["id"])
            expected_shape = (frame["image_height"], frame["image_width"])
            depth = _load_cached_depth(cache_path, expected_shape)
            used_cache = depth is not None
            frame_started = time.perf_counter()
            rotation = _clockwise_rotation_for_upright(frame)
            if depth is None:
                image_path = bundle / frame["image"]
                with Image.open(image_path) as source:
                    sensor_image = source.convert("RGB")
                upright_image = _rotate_image_clockwise(sensor_image, rotation)
                intrinsics = scale_intrinsics(
                    frame["intrinsics"],
                    (frame["calibration_width"], frame["calibration_height"]),
                    sensor_image.size,
                )
                focal_length = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
                upright_depth = _infer_with_loaded_model(
                    upright_image,
                    focal_length,
                    model,
                    transform,
                    torch_device,
                )
                depth = _restore_sensor_depth(upright_depth, rotation)
                if depth.shape != expected_shape:
                    raise ReconstructionError(
                        f"frame {frame['id']} depth shape {depth.shape} does not match {expected_shape}"
                    )
                np.save(cache_path, depth.astype(np.float16))

            elapsed = time.perf_counter() - frame_started
            valid = depth[np.isfinite(depth) & (depth > 0)]
            result = {
                "id": frame["id"],
                "depth_cache": str(cache_path),
                "cached": used_cache,
                "inference_rotation_clockwise_degrees": rotation,
                "depth_median_meters": round(float(np.median(valid)), 5),
                "seconds": round(elapsed, 3),
            }
            frame_results.append(result)
            source = "cache" if used_cache else "Depth Pro"
            print(
                f"[{index:02d}/{len(frames):02d}] frame {frame['id']:06d}: "
                f"{source}, median={result['depth_median_meters']:.3f}m, {elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )
    finally:
        if model is not None:
            del model, transform
            gc.collect()
            try:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except (ImportError, RuntimeError):
                pass

    return frame_results, inference_device, torch_version, time.perf_counter() - model_started


def _integration_images(
    image_path: Path,
    depth: np.ndarray,
    integration_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as source:
        color = source.convert("RGB")
    if integration_width <= 0 or integration_width > color.width:
        raise ValueError("integration_width must be positive and no larger than source width")
    integration_height = round(color.height * integration_width / color.width)
    size = (integration_width, integration_height)
    color_array = np.asarray(color.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
    depth_image = Image.fromarray(depth.astype(np.float32), mode="F")
    depth_array = np.asarray(
        depth_image.resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return np.ascontiguousarray(color_array), np.ascontiguousarray(depth_array)


def _remove_small_mesh_components(mesh, minimum_triangles: int):
    if len(mesh.triangles) == 0:
        return mesh
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters_array = np.asarray(clusters)
    counts_array = np.asarray(counts)
    remove_mask = counts_array[clusters_array] < minimum_triangles
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def _export_blender_glb(mesh, output_path: Path) -> None:
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    colors = np.asarray(mesh.vertex_colors)
    if colors.shape != vertices.shape:
        colors = np.full(vertices.shape, 0.7, dtype=np.float64)
    rgba = np.column_stack(
        (
            np.clip(colors * 255.0, 0, 255).astype(np.uint8),
            np.full(len(colors), 255, dtype=np.uint8),
        )
    )
    scene_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        vertex_colors=rgba,
        process=False,
    )
    scene_mesh.export(output_path, file_type="glb")


def reconstruct_all_frames(
    bundle_path: str | Path,
    output_directory: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    voxel_length: float = 0.025,
    sdf_truncation: float = 0.10,
    integration_width: int = 960,
    maximum_depth: float = 6.0,
    blender_triangle_target: int = 300000,
    inference_only: bool = False,
) -> dict[str, Any]:
    """Infer every frame and fuse them into Blender-compatible mesh artifacts."""

    total_started = time.perf_counter()
    bundle = Path(bundle_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    validation = validate_bundle(bundle)
    if not validation.valid:
        messages = "; ".join(f"{issue.path}: {issue.message}" for issue in validation.issues)
        raise ReconstructionError(f"ScanBundle validation failed: {messages}")
    if not checkpoint.is_file():
        raise ReconstructionError(f"Depth Pro checkpoint not found: {checkpoint}")
    if voxel_length <= 0 or sdf_truncation <= voxel_length:
        raise ReconstructionError("sdf_truncation must be greater than positive voxel_length")

    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    cache_directory = output / "depth_frames"
    cache_manifest_path = cache_directory / "manifest.json"
    if inference_only:
        existing_cache_manifest = None
        if cache_manifest_path.is_file():
            try:
                existing_cache_manifest = json.loads(
                    cache_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                pass
        frame_results, inference_device, torch_version, inference_seconds = _infer_all_depths(
            bundle,
            frames,
            cache_directory,
            checkpoint,
            device,
        )
        if (
            inference_device == "cache"
            and existing_cache_manifest is not None
            and existing_cache_manifest.get("pipeline_version") == FUSION_PIPELINE_VERSION
            and existing_cache_manifest.get("scan_id") == manifest["scan_id"]
        ):
            cache_manifest = existing_cache_manifest
        else:
            cache_manifest = {
                "pipeline_version": FUSION_PIPELINE_VERSION,
                "scan_id": manifest["scan_id"],
                "frames": frame_results,
                "model": {
                    "name": "Apple Depth Pro",
                    "checkpoint": str(checkpoint),
                    "torch_version": torch_version,
                    "device": inference_device,
                },
                "inference_seconds": round(inference_seconds, 3),
            }
        cache_manifest_path.write_text(
            json.dumps(cache_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "depths_cached",
            "frame_count": len(frame_results),
            "cache_manifest": str(cache_manifest_path),
        }

    if not cache_manifest_path.is_file():
        raise ReconstructionError("depth cache manifest is missing; run the inference phase first")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if (
        cache_manifest.get("pipeline_version") != FUSION_PIPELINE_VERSION
        or cache_manifest.get("scan_id") != manifest["scan_id"]
    ):
        raise ReconstructionError("depth cache belongs to a different scan or pipeline version")
    for frame in frames:
        expected_shape = (frame["image_height"], frame["image_width"])
        if _load_cached_depth(
            _depth_cache_path(cache_directory, frame["id"]),
            expected_shape,
        ) is None:
            raise ReconstructionError(f"depth cache for frame {frame['id']} is missing or invalid")
    frame_results = cache_manifest["frames"]
    model_information = cache_manifest.get("model", {})
    inference_device = model_information.get("device", "unknown")
    torch_version = model_information.get("torch_version", "unknown")
    inference_seconds = float(cache_manifest.get("inference_seconds", 0.0))

    import open3d as o3d

    fusion_started = time.perf_counter()
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_truncation,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for index, frame in enumerate(frames, start=1):
        depth = np.load(_depth_cache_path(cache_directory, frame["id"]), allow_pickle=False).astype(np.float32)
        color, integration_depth = _integration_images(
            bundle / frame["image"],
            depth,
            integration_width,
        )
        integration_height = integration_depth.shape[0]
        intrinsics = scale_intrinsics(
            frame["intrinsics"],
            (frame["calibration_width"], frame["calibration_height"]),
            (integration_width, integration_height),
        )
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            integration_width,
            integration_height,
            float(intrinsics[0, 0]),
            float(intrinsics[1, 1]),
            float(intrinsics[0, 2]),
            float(intrinsics[1, 2]),
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(integration_depth),
            depth_scale=1.0,
            depth_trunc=maximum_depth,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(
            rgbd,
            pinhole,
            world_to_cv_camera_extrinsic(frame["world_from_camera"]),
        )
        print(
            f"[{index:02d}/{len(frames):02d}] TSDF integrated frame {frame['id']:06d}",
            file=sys.stderr,
            flush=True,
        )

    point_cloud = volume.extract_point_cloud()
    mesh = volume.extract_triangle_mesh()
    if len(point_cloud.points) == 0 or len(mesh.triangles) == 0:
        raise ReconstructionError("TSDF fusion produced empty geometry")
    point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_length)
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(point_cloud.colors), 0.0, 1.0)
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh = _remove_small_mesh_components(mesh, minimum_triangles=100)
    mesh.compute_vertex_normals()
    if len(mesh.triangles) == 0:
        raise ReconstructionError("mesh cleanup removed every triangle")

    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0)
    )
    full_mesh_path = output / "room_mesh_full.ply"
    point_cloud_path = output / "room_fused_pointcloud.ply"
    o3d.io.write_point_cloud(str(point_cloud_path), point_cloud, write_ascii=False)
    o3d.io.write_triangle_mesh(
        str(full_mesh_path),
        mesh,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )

    blender_mesh = mesh
    if len(mesh.triangles) > blender_triangle_target:
        blender_mesh = mesh.simplify_quadric_decimation(blender_triangle_target)
        blender_mesh.compute_vertex_normals()
    blender_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.clip(np.asarray(blender_mesh.vertex_colors), 0.0, 1.0)
    )
    blender_ply_path = output / "room_blender_mesh.ply"
    blender_glb_path = output / "room_blender_mesh.glb"
    o3d.io.write_triangle_mesh(
        str(blender_ply_path),
        blender_mesh,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )
    _export_blender_glb(blender_mesh, blender_glb_path)

    preview_path = output / "room_model_preview.png"
    mesh_vertices = np.asarray(blender_mesh.vertices, dtype=np.float32)
    mesh_colors = np.asarray(blender_mesh.vertex_colors)
    if mesh_colors.shape != mesh_vertices.shape:
        mesh_colors = np.full(mesh_vertices.shape, 0.7)
    _save_point_cloud_preview(
        preview_path,
        mesh_vertices,
        np.clip(mesh_colors * 255.0, 0, 255).astype(np.uint8),
    )
    fusion_seconds = time.perf_counter() - fusion_started

    point_values = np.asarray(point_cloud.points, dtype=np.float32)
    report_path = output / "fusion_report.json"
    report: dict[str, Any] = {
        "status": "completed",
        "pipeline_version": FUSION_PIPELINE_VERSION,
        "scan_id": manifest["scan_id"],
        "input_bundle": str(bundle),
        "frame_count": len(frames),
        "frames": frame_results,
        "model": {
            "name": "Apple Depth Pro",
            "checkpoint": str(checkpoint),
            "torch_version": torch_version,
            "device": inference_device,
        },
        "fusion": {
            "method": "Open3D scalable TSDF",
            "voxel_length_meters": voxel_length,
            "sdf_truncation_meters": sdf_truncation,
            "integration_resolution": [integration_width, round(frames[0]["image_height"] * integration_width / frames[0]["image_width"])],
            "maximum_depth_meters": maximum_depth,
            "point_count": int(len(point_values)),
            "point_cloud_bounds": point_cloud_bounds(point_values),
            "full_mesh_vertices": int(len(mesh.vertices)),
            "full_mesh_triangles": int(len(mesh.triangles)),
            "blender_mesh_vertices": int(len(blender_mesh.vertices)),
            "blender_mesh_triangles": int(len(blender_mesh.triangles)),
        },
        "artifacts": {
            "depth_cache_directory": str(cache_directory),
            "fused_pointcloud": str(point_cloud_path),
            "full_mesh_ply": str(full_mesh_path),
            "blender_mesh_ply": str(blender_ply_path),
            "blender_mesh_glb": str(blender_glb_path),
            "preview": str(preview_path),
            "report": str(report_path),
        },
        "timings_seconds": {
            "all_frame_model_and_inference": round(inference_seconds, 3),
            "fusion_and_export": round(fusion_seconds, 3),
            "total": round(inference_seconds + fusion_seconds, 3),
        },
        "runtime": {
            "python": platform.python_version(),
            "open3d": o3d.__version__,
        },
        "blender_import": "File > Import > glTF 2.0, then select room_blender_mesh.glb",
        "limitations": [
            "each frame uses independent monocular metric depth; overlap can contain thickness or ghosting",
            "the result includes visible furniture and is not a watertight architectural floor plan",
            "dimensions have not been calibrated against tape measurements",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
