"""World-top-driven finite wall and observed-floor reconstruction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_closing, label

from .fusion import _export_blender_glb
from .pipeline import ReconstructionError


@dataclass(frozen=True)
class WallSegment:
    start_xz: np.ndarray
    end_xz: np.ndarray
    orientation: int
    support: int

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end_xz - self.start_xz))

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_xz_meters": np.round(self.start_xz, 4).tolist(),
            "end_xz_meters": np.round(self.end_xz, 4).tolist(),
            "length_meters": round(self.length, 4),
            "axis": self.orientation,
            "supporting_cells": self.support,
        }


@dataclass(frozen=True)
class GridMask:
    mask: np.ndarray
    minimum_xz: np.ndarray
    resolution: float

    @property
    def extent(self) -> list[float]:
        rows, columns = self.mask.shape
        return [
            float(self.minimum_xz[0]),
            float(self.minimum_xz[0] + columns * self.resolution),
            float(self.minimum_xz[1]),
            float(self.minimum_xz[1] + rows * self.resolution),
        ]


@dataclass(frozen=True)
class TopologyResult:
    floor_y: float
    ceiling_y: float
    axis_angle_radians: float
    wall_segments: tuple[WallSegment, ...]
    wall_evidence: GridMask
    observed_floor: GridMask


def _histogram_peaks(
    values: np.ndarray,
    weights: np.ndarray,
    bin_width: float,
    minimum_weight: float,
) -> list[tuple[float, float]]:
    low = math.floor(float(values.min()) / bin_width) * bin_width
    high = math.ceil(float(values.max()) / bin_width) * bin_width
    edges = np.arange(low, high + bin_width * 1.01, bin_width)
    histogram, _ = np.histogram(values, edges, weights=weights)
    smoothed = np.convolve(histogram, np.array([1, 2, 3, 2, 1]) / 9.0, mode="same")
    peaks = []
    for index in range(len(smoothed)):
        neighborhood = smoothed[max(0, index - 2):min(len(smoothed), index + 3)]
        if smoothed[index] >= max(neighborhood) and smoothed[index] >= minimum_weight:
            peaks.append(((edges[index] + edges[index + 1]) / 2.0, float(smoothed[index])))
    return peaks


def estimate_floor_and_ceiling(points: np.ndarray, normals: np.ndarray) -> tuple[float, float]:
    horizontal = np.abs(normals[:, 1]) >= 0.86
    if int(horizontal.sum()) < 200:
        raise ReconstructionError("not enough horizontal evidence to estimate the floor")
    peaks = _histogram_peaks(
        points[horizontal, 1],
        np.abs(normals[horizontal, 1]) ** 2,
        bin_width=0.04,
        minimum_weight=max(20.0, float(horizontal.sum()) * 0.002),
    )
    pairs = []
    for lower in peaks:
        for upper in peaks:
            height = upper[0] - lower[0]
            if 2.0 <= height <= 4.2:
                pairs.append((lower[1] + upper[1], lower[0], upper[0]))
    if not pairs:
        raise ReconstructionError("could not find a plausible floor/ceiling pair")
    _, floor_y, ceiling_y = max(pairs)
    return floor_y, ceiling_y


def _point_grid(points_xz: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum = np.floor(points_xz.min(axis=0) / resolution) * resolution
    maximum = np.ceil(points_xz.max(axis=0) / resolution) * resolution
    shape_xz = np.round((maximum - minimum) / resolution).astype(int) + 1
    indices_xz = np.clip(((points_xz - minimum) / resolution).astype(int), 0, shape_xz - 1)
    linear = indices_xz[:, 1] * shape_xz[0] + indices_xz[:, 0]
    return minimum, shape_xz, linear


def _wall_evidence_grid(
    points: np.ndarray,
    floor_y: float,
    ceiling_y: float,
    resolution: float,
) -> tuple[GridMask, np.ndarray]:
    minimum, shape_xz, cells = _point_grid(points[:, (0, 2)], resolution)
    cell_count = int(np.prod(shape_xz))
    counts = np.bincount(cells, minlength=cell_count)
    vertical = (points[:, 1] > floor_y + 0.08) & (points[:, 1] < ceiling_y + 0.08)
    level_count = 18
    y_bins = np.clip(
        ((points[vertical, 1] - floor_y - 0.08) / (ceiling_y - floor_y) * level_count).astype(int),
        0,
        level_count - 1,
    )
    occupied_level_keys = np.unique(cells[vertical] * level_count + y_bins)
    occupied_levels = np.bincount(
        occupied_level_keys // level_count,
        minlength=cell_count,
    )
    minimum_y = np.full(cell_count, np.inf)
    maximum_y = np.full(cell_count, -np.inf)
    np.minimum.at(minimum_y, cells[vertical], points[vertical, 1])
    np.maximum.at(maximum_y, cells[vertical], points[vertical, 1])
    span = maximum_y - minimum_y
    span[~np.isfinite(span)] = 0
    evidence = (occupied_levels >= 7) & (span >= 1.25) & (counts >= 5)
    mask = evidence.reshape(shape_xz[1], shape_xz[0])
    return GridMask(mask, minimum, resolution), evidence[cells]


def _axis_angle(normals: np.ndarray, evidence_points: np.ndarray) -> float:
    usable = evidence_points & (np.abs(normals[:, 1]) < 0.40)
    if int(usable.sum()) < 200:
        raise ReconstructionError("not enough vertical wall evidence to estimate dominant axes")
    angles = np.mod(np.arctan2(normals[usable, 2], normals[usable, 0]), np.pi / 2.0)
    weights = (1.0 - np.abs(normals[usable, 1])) ** 2
    edges = np.linspace(0.0, np.pi / 2.0, 181)
    histogram, _ = np.histogram(angles, edges, weights=weights)
    wrapped = np.concatenate((histogram[-5:], histogram, histogram[:5]))
    smoothed = np.convolve(wrapped, np.ones(11) / 11.0, mode="same")[5:-5]
    index = int(np.argmax(smoothed))
    return float((edges[index] + edges[index + 1]) / 2.0)


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.concatenate(([False], values, [False])).astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _scan_axis_candidates(
    image: np.ndarray,
    orientation: int,
    minimum_q: np.ndarray,
    resolution: float,
) -> list[dict[str, float | int]]:
    candidates: list[dict[str, float | int]] = []
    band_radius = 2
    for normal_index in range(image.shape[0]):
        band_start = max(0, normal_index - band_radius)
        band_end = min(image.shape[0], normal_index + band_radius + 1)
        profile = image[band_start:band_end].any(axis=0)
        profile = binary_closing(profile, structure=np.ones(3, dtype=bool))
        for start, end in _true_runs(profile):
            length = (end - start) * resolution
            block = image[band_start:band_end, start:end]
            support = int(block.sum())
            if length < 0.70 or support < max(7, int((end - start) * 0.22)):
                continue
            occupied = np.argwhere(block)
            normal_position = (
                (band_start + occupied[:, 0] + 0.5).mean() * resolution
                + minimum_q[orientation]
            )
            candidates.append(
                {
                    "orientation": orientation,
                    "normal": float(normal_position),
                    "start": float((start + 0.5) * resolution + minimum_q[1 - orientation]),
                    "end": float((end - 0.5) * resolution + minimum_q[1 - orientation]),
                    "support": support,
                    "length": float(length),
                    "score": float(support * math.sqrt(length)),
                }
            )
    return candidates


def _finite_wall_segments(
    evidence: GridMask,
    axis_angle: float,
) -> tuple[WallSegment, ...]:
    row_indices, column_indices = np.nonzero(evidence.mask)
    centers_xz = np.column_stack(
        (
            evidence.minimum_xz[0] + (column_indices + 0.5) * evidence.resolution,
            evidence.minimum_xz[1] + (row_indices + 0.5) * evidence.resolution,
        )
    )
    axis0 = np.array([math.cos(axis_angle), math.sin(axis_angle)])
    axis1 = np.array([-math.sin(axis_angle), math.cos(axis_angle)])
    q = np.column_stack((centers_xz @ axis0, centers_xz @ axis1))
    minimum_q = np.floor(q.min(axis=0) / evidence.resolution) * evidence.resolution
    maximum_q = np.ceil(q.max(axis=0) / evidence.resolution) * evidence.resolution
    shape_q = np.round((maximum_q - minimum_q) / evidence.resolution).astype(int) + 1
    indices_q = np.clip(((q - minimum_q) / evidence.resolution).astype(int), 0, shape_q - 1)
    occupancy = np.zeros((shape_q[0], shape_q[1]), dtype=bool)
    occupancy[indices_q[:, 0], indices_q[:, 1]] = True

    candidates = _scan_axis_candidates(occupancy, 0, minimum_q, evidence.resolution)
    candidates.extend(_scan_axis_candidates(occupancy.T, 1, minimum_q, evidence.resolution))
    kept: list[dict[str, float | int]] = []
    for candidate in sorted(candidates, key=lambda item: float(item["score"]), reverse=True):
        duplicate = False
        for previous in kept:
            if candidate["orientation"] != previous["orientation"]:
                continue
            if abs(float(candidate["normal"]) - float(previous["normal"])) > 0.45:
                continue
            overlap = min(float(candidate["end"]), float(previous["end"])) - max(
                float(candidate["start"]), float(previous["start"])
            )
            if overlap > 0.30 * min(float(candidate["length"]), float(previous["length"])):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    segments = []
    for candidate in kept:
        orientation = int(candidate["orientation"])
        if orientation == 0:
            start_q = np.array([candidate["normal"], candidate["start"]], dtype=np.float64)
            end_q = np.array([candidate["normal"], candidate["end"]], dtype=np.float64)
        else:
            start_q = np.array([candidate["start"], candidate["normal"]], dtype=np.float64)
            end_q = np.array([candidate["end"], candidate["normal"]], dtype=np.float64)
        start_xz = axis0 * start_q[0] + axis1 * start_q[1]
        end_xz = axis0 * end_q[0] + axis1 * end_q[1]
        segments.append(
            WallSegment(start_xz, end_xz, orientation, int(candidate["support"]))
        )
    return tuple(segments)


def _observed_floor_grid(
    points: np.ndarray,
    normals: np.ndarray,
    floor_y: float,
    resolution: float = 0.12,
) -> GridMask:
    floor_points = (np.abs(points[:, 1] - floor_y) <= 0.13) & (np.abs(normals[:, 1]) >= 0.72)
    if int(floor_points.sum()) < 100:
        raise ReconstructionError("not enough horizontal floor samples for an observed-floor mesh")
    minimum, shape_xz, cells = _point_grid(points[:, (0, 2)], resolution)
    counts = np.bincount(cells[floor_points], minlength=int(np.prod(shape_xz)))
    mask = (counts.reshape(shape_xz[1], shape_xz[0]) >= 2)
    mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    components, component_count = label(mask)
    for component in range(1, component_count + 1):
        component_mask = components == component
        if int(component_mask.sum()) < 10:
            mask[component_mask] = False
    return GridMask(mask, minimum, resolution)


def extract_world_top_topology(points: np.ndarray, normals: np.ndarray) -> TopologyResult:
    """Extract finite wall segments without forcing a closed room envelope."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape (N, 3)")
    floor_y, ceiling_y = estimate_floor_and_ceiling(points, normals)
    wall_evidence, evidence_points = _wall_evidence_grid(
        points, floor_y, ceiling_y, resolution=0.06
    )
    angle = _axis_angle(normals, evidence_points)
    segments = _finite_wall_segments(wall_evidence, angle)
    if len(segments) < 2:
        raise ReconstructionError("World top did not contain enough finite wall segments")
    observed_floor = _observed_floor_grid(points, normals, floor_y)
    return TopologyResult(
        floor_y=floor_y,
        ceiling_y=ceiling_y,
        axis_angle_radians=angle,
        wall_segments=segments,
        wall_evidence=wall_evidence,
        observed_floor=observed_floor,
    )


def _add_wall_box(
    vertices: list[list[float]],
    triangles: list[list[int]],
    colors: list[list[float]],
    segment: WallSegment,
    floor_y: float,
    ceiling_y: float,
    thickness: float,
) -> None:
    tangent = segment.end_xz - segment.start_xz
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]]) * thickness / 2.0
    footprint = [
        segment.start_xz - normal,
        segment.end_xz - normal,
        segment.end_xz + normal,
        segment.start_xz + normal,
    ]
    start = len(vertices)
    for y in (floor_y, ceiling_y):
        vertices.extend([[float(point[0]), y, float(point[1])] for point in footprint])
    triangles.extend(
        [
            [start, start + 2, start + 1], [start, start + 3, start + 2],
            [start + 4, start + 5, start + 6], [start + 4, start + 6, start + 7],
            [start, start + 1, start + 5], [start, start + 5, start + 4],
            [start + 1, start + 2, start + 6], [start + 1, start + 6, start + 5],
            [start + 2, start + 3, start + 7], [start + 2, start + 7, start + 6],
            [start + 3, start, start + 4], [start + 3, start + 4, start + 7],
        ]
    )
    wall_color = [0.68, 0.76, 0.82]
    colors.extend([wall_color] * 8)


def _topology_mesh(result: TopologyResult):
    import open3d as o3d

    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    colors: list[list[float]] = []
    floor_rows, floor_columns = np.nonzero(result.observed_floor.mask)
    resolution = result.observed_floor.resolution
    for row, column in zip(floor_rows, floor_columns):
        x0 = result.observed_floor.minimum_xz[0] + column * resolution
        z0 = result.observed_floor.minimum_xz[1] + row * resolution
        x1, z1 = x0 + resolution, z0 + resolution
        start = len(vertices)
        y = result.floor_y + 0.005
        vertices.extend([[x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0]])
        triangles.extend(([start, start + 1, start + 2], [start, start + 2, start + 3]))
        colors.extend([[0.30, 0.33, 0.36]] * 4)
    for segment in result.wall_segments:
        _add_wall_box(
            vertices,
            triangles,
            colors,
            segment,
            result.floor_y,
            result.ceiling_y,
            thickness=0.08,
        )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors))
    mesh.compute_vertex_normals()
    return mesh


def _save_preview(path: Path, points: np.ndarray, result: TopologyResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    sample = points[::3]
    axes[0].scatter(sample[:, 0], sample[:, 2], s=0.12, c="0.75", alpha=0.30)
    evidence_rows, evidence_columns = np.nonzero(result.wall_evidence.mask)
    evidence_x = result.wall_evidence.minimum_xz[0] + (evidence_columns + 0.5) * result.wall_evidence.resolution
    evidence_z = result.wall_evidence.minimum_xz[1] + (evidence_rows + 0.5) * result.wall_evidence.resolution
    axes[0].scatter(evidence_x, evidence_z, s=3, c="#c83e3e", alpha=0.38)
    for segment in result.wall_segments:
        color = "#0066dd" if segment.orientation == 0 else "#00a060"
        axes[0].plot(
            [segment.start_xz[0], segment.end_xz[0]],
            [segment.start_xz[1], segment.end_xz[1]],
            color=color,
            linewidth=3,
        )
    axes[0].set_title("World top evidence + finite snapped wall segments")

    axes[1].imshow(
        result.observed_floor.mask,
        origin="lower",
        extent=result.observed_floor.extent,
        cmap="Greys",
        alpha=0.55,
    )
    for segment in result.wall_segments:
        axes[1].plot(
            [segment.start_xz[0], segment.end_xz[0]],
            [segment.start_xz[1], segment.end_xz[1]],
            color="#315a74",
            linewidth=5,
            solid_capstyle="butt",
        )
    axes[1].set_title("Clean output: observed floor only; gaps stay open")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Z (m)")
        axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def reconstruct_world_top(
    fusion_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Build a phone-viewable topology mesh from the fused World top."""

    import open3d as o3d

    source = Path(fusion_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    point_cloud_path = source / "room_fused_pointcloud.ply"
    source_report_path = source / "fusion_report.json"
    if not point_cloud_path.is_file() or not source_report_path.is_file():
        raise ReconstructionError("fusion directory is missing its point cloud or report")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path)).voxel_down_sample(0.025)
    if len(point_cloud.points) < 1000:
        raise ReconstructionError("fused point cloud is too small for World top extraction")
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=35)
    )
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    result = extract_world_top_topology(points, normals)
    mesh = _topology_mesh(result)

    output.mkdir(parents=True, exist_ok=True)
    ply_path = output / "room_blender_mesh.ply"
    glb_path = output / "room_blender_mesh.glb"
    preview_path = output / "room_world_top_preview.png"
    report_path = output / "fusion_report.json"
    o3d.io.write_triangle_mesh(
        str(ply_path),
        mesh,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )
    _export_blender_glb(mesh, glb_path)
    _save_preview(preview_path, points, result)

    report: dict[str, Any] = {
        "status": "completed",
        "pipeline_version": "fosmo-world-top-segments-1",
        "scan_id": source_report["scan_id"],
        "frame_count": source_report["frame_count"],
        "input_fusion_directory": str(source),
        "topology": {
            "floor_y_meters": round(result.floor_y, 4),
            "ceiling_y_meters": round(result.ceiling_y, 4),
            "axis_angle_degrees": round(math.degrees(result.axis_angle_radians), 4),
            "wall_segment_count": len(result.wall_segments),
            "wall_length_meters": round(sum(segment.length for segment in result.wall_segments), 4),
            "observed_floor_tile_count": int(result.observed_floor.mask.sum()),
            "wall_segments": [segment.to_dict() for segment in result.wall_segments],
        },
        "fusion": {
            "method": "World top vertical-evidence finite wall segments",
            "model_kind": "world_top",
            "floor_mesh_triangles": int(result.observed_floor.mask.sum()) * 2,
            "wall_segment_count": len(result.wall_segments),
            "blender_mesh_vertices": int(len(mesh.vertices)),
            "blender_mesh_triangles": int(len(mesh.triangles)),
        },
        "artifacts": {
            "blender_mesh_ply": str(ply_path),
            "blender_mesh_glb": str(glb_path),
            "preview": str(preview_path),
            "report": str(report_path),
        },
        "limitations": [
            "wall orientation is snapped, but positions and finite observed extents are retained",
            "unknown gaps are not bridged and the output is intentionally not a closed room",
            "floor tiles are emitted only where horizontal floor evidence exists",
            "monocular per-frame depth drift remains visible in the underlying World top",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
