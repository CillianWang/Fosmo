"""Single-frame Depth Pro reconstruction pipeline."""

from __future__ import annotations

import json
import math
import os
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fosmo_scanbundle.validator import validate_bundle

from .geometry import (
    backproject_to_world,
    point_cloud_bounds,
    scale_intrinsics,
    write_binary_ply,
)


class ReconstructionError(RuntimeError):
    """Raised when a reconstruction cannot produce trustworthy artifacts."""


def _sharpness_score(image_path: Path) -> float:
    with Image.open(image_path) as source:
        grayscale = source.convert("L")
        grayscale.thumbnail((512, 512), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
    horizontal = np.diff(pixels, axis=1)
    vertical = np.diff(pixels, axis=0)
    return float(horizontal.var() + vertical.var())


def _pitch_degrees(frame: dict[str, Any]) -> float:
    matrix = frame["world_from_camera"]
    forward_y = -float(matrix[6])
    return math.degrees(math.asin(max(-1.0, min(1.0, forward_y))))


def _select_frame(
    bundle: Path,
    frames: list[dict[str, Any]],
) -> tuple[dict[str, Any], float, float]:
    candidates = [
        (frame, _sharpness_score(bundle / frame["image"]), _pitch_degrees(frame))
        for frame in frames
        if frame["tracking_state"] == "normal" and abs(_pitch_degrees(frame)) <= 45.0
    ]
    if not candidates:
        raise ReconstructionError("bundle contains no normally tracked frame within 45 degrees of level")
    return max(candidates, key=lambda candidate: candidate[1])


def _clockwise_rotation_for_upright(frame: dict[str, Any]) -> int:
    """Use gravity-aligned ARKit pose to orient sensor pixels for inference."""

    matrix = frame["world_from_camera"]
    world_up_along_image_u = float(matrix[4])
    world_up_along_image_v = -float(matrix[5])
    if abs(world_up_along_image_u) >= abs(world_up_along_image_v):
        return 90 if world_up_along_image_u < 0 else 270
    return 0 if world_up_along_image_v < 0 else 180


def _rotate_image_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    transpose = {
        0: None,
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }[degrees]
    return image.copy() if transpose is None else image.transpose(transpose)


def _restore_sensor_depth(depth: np.ndarray, clockwise_degrees: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(depth, k=clockwise_degrees // 90))


def _device(requested: str):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ReconstructionError("MPS was requested but is unavailable")
    if requested not in {"mps", "cpu"}:
        raise ReconstructionError(f"unsupported inference device: {requested}")
    return torch.device(requested)


def _infer_depth(
    image: Image.Image,
    focal_length_pixels: float,
    checkpoint_path: Path,
    requested_device: str,
) -> tuple[np.ndarray, str, str, float]:
    started = time.perf_counter()
    model, transform, device, torch_version = _load_depth_model(
        checkpoint_path,
        requested_device,
    )
    depth = _infer_with_loaded_model(
        image,
        focal_length_pixels,
        model,
        transform,
        device,
    )
    elapsed = time.perf_counter() - started
    return depth, str(device), torch_version, elapsed


def _load_depth_model(checkpoint_path: Path, requested_device: str):
    import depth_pro
    import torch
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

    device = _device(requested_device)
    config = replace(
        DEFAULT_MONODEPTH_CONFIG_DICT,
        checkpoint_uri=str(checkpoint_path),
    )
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model.eval()
    return model, transform, device, torch.__version__


def _infer_with_loaded_model(
    image: Image.Image,
    focal_length_pixels: float,
    model,
    transform,
    device,
) -> np.ndarray:
    import torch

    transformed = transform(image)
    focal = torch.tensor(focal_length_pixels, device=device, dtype=torch.float32)
    with torch.inference_mode():
        prediction = model.infer(transformed, f_px=focal)
    return prediction["depth"].detach().to("cpu").numpy().astype(np.float32)


def _save_depth_preview(path: Path, depth: np.ndarray, minimum: float, maximum: float) -> None:
    valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
    if not valid.any():
        raise ReconstructionError("Depth Pro returned no finite depth in the accepted range")
    near, far = np.percentile(depth[valid], (2.0, 98.0))
    if far <= near:
        far = near + 1e-6
    normalized = np.clip((far - depth) / (far - near), 0.0, 1.0)
    normalized[~valid] = 0.0
    Image.fromarray((normalized * 255).astype(np.uint8), mode="L").save(path)


def _save_point_cloud_preview(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_step = max(1, len(points) // 30000)
    sampled = points[::sample_step]
    sampled_colors = colors[::sample_step].astype(np.float32) / 255.0
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    views = (
        (0, 2, "World top: X / Z", "X (m)", "Z (m)"),
        (0, 1, "World front: X / Y", "X (m)", "Y (m)"),
        (2, 1, "World side: Z / Y", "Z (m)", "Y (m)"),
    )
    for axis, (horizontal, vertical, title, xlabel, ylabel) in zip(axes, views):
        axis.scatter(
            sampled[:, horizontal],
            sampled[:, vertical],
            c=sampled_colors,
            s=0.25,
            linewidths=0,
        )
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def reconstruct_single_frame(
    bundle_path: str | Path,
    output_directory: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    stride: int = 4,
    minimum_depth: float = 0.1,
    maximum_depth: float = 20.0,
) -> dict[str, Any]:
    """Run a validated ScanBundle through Depth Pro and export metric artifacts."""

    total_started = time.perf_counter()
    bundle = Path(bundle_path).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()

    validation_started = time.perf_counter()
    validation = validate_bundle(bundle)
    validation_seconds = time.perf_counter() - validation_started
    if not validation.valid:
        messages = "; ".join(f"{issue.path}: {issue.message}" for issue in validation.issues)
        raise ReconstructionError(f"ScanBundle validation failed: {messages}")
    if not checkpoint.is_file():
        raise ReconstructionError(f"Depth Pro checkpoint not found: {checkpoint}")

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    frame, sharpness, pitch_degrees = _select_frame(bundle, manifest["frames"])
    image_path = bundle / frame["image"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    inference_rotation = _clockwise_rotation_for_upright(frame)
    inference_image = _rotate_image_clockwise(image, inference_rotation)

    intrinsics = scale_intrinsics(
        frame["intrinsics"],
        (frame["calibration_width"], frame["calibration_height"]),
        image.size,
    )
    focal_length = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
    inference_depth, inference_device, torch_version, inference_seconds = _infer_depth(
        inference_image,
        focal_length,
        checkpoint,
        device,
    )
    depth = _restore_sensor_depth(inference_depth, inference_rotation)
    if depth.shape != rgb.shape[:2]:
        raise ReconstructionError(
            f"Depth Pro returned {depth.shape}, expected image shape {rgb.shape[:2]}"
        )

    projection_started = time.perf_counter()
    points, colors = backproject_to_world(
        depth,
        rgb,
        intrinsics,
        frame["world_from_camera"],
        stride=stride,
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
    )
    projection_seconds = time.perf_counter() - projection_started

    output.mkdir(parents=True, exist_ok=True)
    depth_path = output / "depth.npy"
    preview_path = output / "depth_preview.png"
    cloud_path = output / "pointcloud.ply"
    cloud_preview_path = output / "pointcloud_preview.png"
    report_path = output / "report.json"
    np.save(depth_path, depth)
    _save_depth_preview(preview_path, inference_depth, minimum_depth, maximum_depth)
    write_binary_ply(cloud_path, points, colors)
    _save_point_cloud_preview(cloud_preview_path, points, colors)

    valid_depth = depth[
        np.isfinite(depth) & (depth >= minimum_depth) & (depth <= maximum_depth)
    ]
    report: dict[str, Any] = {
        "status": "completed",
        "pipeline_version": "fosmo-single-frame-depth-pro-1",
        "scan_id": manifest["scan_id"],
        "input_bundle": str(bundle),
        "source_frame": {
            "id": frame["id"],
            "image": frame["image"],
            "width": image.width,
            "height": image.height,
            "sharpness_score": round(sharpness, 8),
            "pitch_degrees": round(pitch_degrees, 3),
            "focal_length_pixels": round(focal_length, 6),
            "inference_rotation_clockwise_degrees": inference_rotation,
        },
        "model": {
            "name": "Apple Depth Pro",
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "torch_version": torch_version,
            "device": inference_device,
        },
        "depth_meters": {
            "minimum": round(float(valid_depth.min()), 6),
            "median": round(float(np.median(valid_depth)), 6),
            "maximum": round(float(valid_depth.max()), 6),
            "valid_pixel_ratio": round(float(valid_depth.size / depth.size), 6),
        },
        "point_cloud": {
            "coordinate_system": "ARKit world coordinates",
            "units": "meters",
            "stride": stride,
            "point_count": int(len(points)),
            "bounds": point_cloud_bounds(points),
        },
        "timings_seconds": {
            "validation": round(validation_seconds, 3),
            "model_load_and_inference": round(inference_seconds, 3),
            "projection": round(projection_seconds, 3),
            "total": round(time.perf_counter() - total_started, 3),
        },
        "artifacts": {
            "depth": str(depth_path),
            "depth_preview": str(preview_path),
            "pointcloud": str(cloud_path),
            "pointcloud_preview": str(cloud_preview_path),
            "report": str(report_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "limitations": [
            "single-frame reconstruction; other scan frames are not fused",
            "metric depth is model-predicted and has not been compared with tape measurements",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
