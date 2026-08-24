"""Extract a clean gravity-aligned Manhattan room from fused geometry."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fusion import _export_blender_glb
from .pipeline import ReconstructionError


@dataclass(frozen=True)
class WallEvidence:
    position: float
    point_count: int
    tangent_span: float
    vertical_span: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_meters": round(self.position, 4),
            "supporting_point_count": self.point_count,
            "observed_tangent_span_meters": round(self.tangent_span, 4),
            "observed_vertical_span_meters": round(self.vertical_span, 4),
            "score": round(self.score, 3),
        }


@dataclass(frozen=True)
class ManhattanLayout:
    floor_y: float
    ceiling_y: float
    axis_angle_radians: float
    axis0: np.ndarray
    axis1: np.ndarray
    axis0_walls: tuple[WallEvidence, WallEvidence]
    axis1_walls: tuple[WallEvidence, WallEvidence]

    @property
    def width_axis0(self) -> float:
        return self.axis0_walls[1].position - self.axis0_walls[0].position

    @property
    def width_axis1(self) -> float:
        return self.axis1_walls[1].position - self.axis1_walls[0].position

    @property
    def height(self) -> float:
        return self.ceiling_y - self.floor_y


def _smoothed_histogram_peaks(
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


def _estimate_floor_and_ceiling(points: np.ndarray, normals: np.ndarray) -> tuple[float, float]:
    horizontal = np.abs(normals[:, 1]) >= 0.86
    if int(horizontal.sum()) < 200:
        raise ReconstructionError("not enough horizontal evidence to estimate a floor")
    peaks = _smoothed_histogram_peaks(
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


def _estimate_axis_angle(
    points: np.ndarray,
    normals: np.ndarray,
    floor_y: float,
    ceiling_y: float,
) -> float:
    vertical = (
        (np.abs(normals[:, 1]) <= 0.35)
        & (points[:, 1] >= floor_y + 0.12)
        & (points[:, 1] <= ceiling_y - 0.08)
    )
    if int(vertical.sum()) < 300:
        raise ReconstructionError("not enough vertical evidence to estimate Manhattan axes")
    angles = np.mod(np.arctan2(normals[vertical, 2], normals[vertical, 0]), np.pi / 2.0)
    weights = (1.0 - np.abs(normals[vertical, 1])) ** 2
    edges = np.linspace(0.0, np.pi / 2.0, 181)
    histogram, _ = np.histogram(angles, edges, weights=weights)
    wrapped = np.concatenate((histogram[-5:], histogram, histogram[:5]))
    smoothed = np.convolve(wrapped, np.ones(11) / 11.0, mode="same")[5:-5]
    index = int(np.argmax(smoothed))
    return float((edges[index] + edges[index + 1]) / 2.0)


def _wall_evidence(
    points: np.ndarray,
    normals: np.ndarray,
    normal_xz: np.ndarray,
    floor_y: float,
    ceiling_y: float,
) -> list[WallEvidence]:
    alignment = np.abs(normals[:, 0] * normal_xz[0] + normals[:, 2] * normal_xz[1])
    usable = (
        (alignment >= 0.88)
        & (np.abs(normals[:, 1]) <= 0.40)
        & (points[:, 1] >= floor_y + 0.12)
        & (points[:, 1] <= ceiling_y + 0.05)
    )
    if int(usable.sum()) < 200:
        return []
    projected = points[:, (0, 2)] @ normal_xz
    peaks = _smoothed_histogram_peaks(
        projected[usable],
        alignment[usable] ** 2,
        bin_width=0.05,
        minimum_weight=max(25.0, float(usable.sum()) * 0.001),
    )
    tangent = np.array([-normal_xz[1], normal_xz[0]])
    candidates = []
    for position, _ in peaks:
        supported = usable & (np.abs(projected - position) <= 0.12)
        sample = points[supported]
        if len(sample) < max(200, int(len(points) * 0.0025)):
            continue
        tangent_values = sample[:, (0, 2)] @ tangent
        tangent_span = float(np.percentile(tangent_values, 95) - np.percentile(tangent_values, 5))
        vertical_span = float(np.percentile(sample[:, 1], 95) - np.percentile(sample[:, 1], 5))
        if tangent_span < 0.8 or vertical_span < 1.0:
            continue
        score = len(sample) * min(tangent_span, 4.0) * min(vertical_span, 2.5)
        candidates.append(
            WallEvidence(position, len(sample), tangent_span, vertical_span, float(score))
        )
    candidates.sort(key=lambda evidence: evidence.score, reverse=True)
    separated = []
    for candidate in candidates:
        if all(abs(candidate.position - kept.position) > 0.30 for kept in separated):
            separated.append(candidate)
    return separated


def _choose_wall_pair(candidates: list[WallEvidence]) -> tuple[WallEvidence, WallEvidence]:
    if not candidates:
        raise ReconstructionError("no supported wall plane was found")
    strongest = candidates[0].score
    credible = [candidate for candidate in candidates if candidate.score >= strongest * 0.10]
    pairs = []
    for first_index, first in enumerate(credible):
        for second in credible[first_index + 1:]:
            separation = abs(first.position - second.position)
            if 1.5 <= separation <= 15.0:
                objective = math.sqrt(first.score * second.score) * min(separation, 8.0)
                pairs.append((objective, first, second))
    if not pairs:
        raise ReconstructionError("only one supported wall was found for a Manhattan axis")
    _, first, second = max(pairs, key=lambda item: item[0])
    return tuple(sorted((first, second), key=lambda evidence: evidence.position))


def estimate_manhattan_layout(points: np.ndarray, normals: np.ndarray) -> ManhattanLayout:
    """Estimate a main orthogonal room envelope from oriented surface samples."""

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
        raise ValueError("points and normals must both have shape (N, 3)")
    floor_y, ceiling_y = _estimate_floor_and_ceiling(points, normals)
    angle = _estimate_axis_angle(points, normals, floor_y, ceiling_y)
    axis0 = np.array([math.cos(angle), math.sin(angle)])
    axis1 = np.array([-math.sin(angle), math.cos(angle)])
    axis0_walls = _choose_wall_pair(_wall_evidence(points, normals, axis0, floor_y, ceiling_y))
    axis1_walls = _choose_wall_pair(_wall_evidence(points, normals, axis1, floor_y, ceiling_y))
    return ManhattanLayout(
        floor_y=floor_y,
        ceiling_y=ceiling_y,
        axis_angle_radians=angle,
        axis0=axis0,
        axis1=axis1,
        axis0_walls=axis0_walls,
        axis1_walls=axis1_walls,
    )


def _world_point(layout: ManhattanLayout, axis0_position: float, axis1_position: float, y: float) -> list[float]:
    xz = layout.axis0 * axis0_position + layout.axis1 * axis1_position
    return [float(xz[0]), y, float(xz[1])]


def _layout_mesh(layout: ManhattanLayout):
    import open3d as o3d

    axis0_low, axis0_high = (wall.position for wall in layout.axis0_walls)
    axis1_low, axis1_high = (wall.position for wall in layout.axis1_walls)
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    colors: list[list[float]] = []

    def add_quad(corners: list[list[float]], color: tuple[int, int, int]) -> None:
        start = len(vertices)
        vertices.extend(corners)
        triangles.extend(([start, start + 1, start + 2], [start, start + 2, start + 3]))
        colors.extend([[channel / 255.0 for channel in color]] * 4)

    floor = layout.floor_y
    ceiling = layout.ceiling_y
    add_quad(
        [
            _world_point(layout, axis0_low, axis1_low, floor),
            _world_point(layout, axis0_low, axis1_high, floor),
            _world_point(layout, axis0_high, axis1_high, floor),
            _world_point(layout, axis0_high, axis1_low, floor),
        ],
        (170, 178, 187),
    )
    for axis0 in (axis0_low, axis0_high):
        add_quad(
            [
                _world_point(layout, axis0, axis1_low, floor),
                _world_point(layout, axis0, axis1_high, floor),
                _world_point(layout, axis0, axis1_high, ceiling),
                _world_point(layout, axis0, axis1_low, ceiling),
            ],
            (223, 228, 233),
        )
    for axis1 in (axis1_low, axis1_high):
        add_quad(
            [
                _world_point(layout, axis0_low, axis1, floor),
                _world_point(layout, axis0_high, axis1, floor),
                _world_point(layout, axis0_high, axis1, ceiling),
                _world_point(layout, axis0_low, axis1, ceiling),
            ],
            (205, 216, 226),
        )

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors))
    mesh.compute_vertex_normals()
    return mesh


def _save_layout_preview(path: Path, layout: ManhattanLayout) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    axis0_low, axis0_high = (wall.position for wall in layout.axis0_walls)
    axis1_low, axis1_high = (wall.position for wall in layout.axis1_walls)
    floor = layout.floor_y
    ceiling = layout.ceiling_y
    floor_corners = [
        _world_point(layout, axis0_low, axis1_low, floor),
        _world_point(layout, axis0_high, axis1_low, floor),
        _world_point(layout, axis0_high, axis1_high, floor),
        _world_point(layout, axis0_low, axis1_high, floor),
    ]
    wall_polygons = []
    for index in range(4):
        first = floor_corners[index]
        second = floor_corners[(index + 1) % 4]
        wall_polygons.append([first, second, [second[0], ceiling, second[2]], [first[0], ceiling, first[2]]])

    figure = plt.figure(figsize=(14, 6))
    view = figure.add_subplot(1, 2, 1, projection="3d")
    view.add_collection3d(Poly3DCollection([floor_corners], facecolors="#aab2bb", alpha=0.9))
    view.add_collection3d(Poly3DCollection(wall_polygons, facecolors="#d8e0e7", edgecolors="#52606d", alpha=0.55))
    all_points = np.asarray(floor_corners + [point for wall in wall_polygons for point in wall])
    view.set_xlim(all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5)
    view.set_ylim(all_points[:, 2].min() - 0.5, all_points[:, 2].max() + 0.5)
    view.set_zlim(floor - 0.2, ceiling + 0.4)
    view.set_xlabel("X (m)")
    view.set_ylabel("Z (m)")
    view.set_zlabel("Y (m)")
    view.set_title("Clean Manhattan room")
    view.view_init(elev=28, azim=-58)

    plan = figure.add_subplot(1, 2, 2)
    polygon = np.asarray([[point[0], point[2]] for point in floor_corners + [floor_corners[0]]])
    plan.fill(polygon[:, 0], polygon[:, 1], color="#aab2bb", alpha=0.45)
    plan.plot(polygon[:, 0], polygon[:, 1], color="#334e68", linewidth=4)
    plan.set_aspect("equal")
    plan.grid(alpha=0.2)
    plan.set_xlabel("X (m)")
    plan.set_ylabel("Z (m)")
    plan.set_title(f"Top view: {layout.width_axis0:.2f} × {layout.width_axis1:.2f} m")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def reconstruct_manhattan_space(
    fusion_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Create a clean floor-and-wall model from an existing fused result."""

    import open3d as o3d

    source = Path(fusion_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    point_cloud_path = source / "room_fused_pointcloud.ply"
    fusion_report_path = source / "fusion_report.json"
    if not point_cloud_path.is_file() or not fusion_report_path.is_file():
        raise ReconstructionError("fusion directory is missing its point cloud or report")
    fusion_report = json.loads(fusion_report_path.read_text(encoding="utf-8"))
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path)).voxel_down_sample(0.04)
    if len(point_cloud.points) < 1000:
        raise ReconstructionError("fused point cloud is too small for Manhattan extraction")
    point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.14, max_nn=40)
    )
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    layout = estimate_manhattan_layout(points, normals)

    output.mkdir(parents=True, exist_ok=True)
    mesh = _layout_mesh(layout)
    ply_path = output / "room_manhattan_mesh.ply"
    glb_path = output / "room_manhattan_mesh.glb"
    preview_path = output / "room_manhattan_preview.png"
    report_path = output / "manhattan_report.json"
    o3d.io.write_triangle_mesh(
        str(ply_path),
        mesh,
        write_ascii=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )
    _export_blender_glb(mesh, glb_path)
    _save_layout_preview(preview_path, layout)

    report: dict[str, Any] = {
        "status": "completed",
        "pipeline_version": "fosmo-manhattan-room-1",
        "scan_id": fusion_report["scan_id"],
        "frame_count": fusion_report["frame_count"],
        "input_fusion_directory": str(source),
        "layout": {
            "floor_y_meters": round(layout.floor_y, 4),
            "ceiling_y_meters": round(layout.ceiling_y, 4),
            "height_meters": round(layout.height, 4),
            "axis_angle_degrees": round(math.degrees(layout.axis_angle_radians), 4),
            "dimensions_meters": [round(layout.width_axis0, 4), round(layout.width_axis1, 4)],
            "axis0_walls": [wall.to_dict() for wall in layout.axis0_walls],
            "axis1_walls": [wall.to_dict() for wall in layout.axis1_walls],
        },
        "mesh": {
            "vertex_count": len(mesh.vertices),
            "triangle_count": len(mesh.triangles),
            "contains_floor": True,
            "wall_count": 4,
            "contains_ceiling": False,
        },
        "artifacts": {
            "manhattan_mesh_ply": str(ply_path),
            "manhattan_mesh_glb": str(glb_path),
            "preview": str(preview_path),
            "report": str(report_path),
        },
        "limitations": [
            "the output is the strongest supported orthogonal main-room envelope, not every scanned recess",
            "doors, windows, openings, and wall thickness are not inferred yet",
            "the ceiling plane sets wall height but is intentionally omitted from the visualization",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview_manifest = {
        "scan_id": report["scan_id"],
        "frame_count": report["frame_count"],
        "vertex_count": report["mesh"]["vertex_count"],
        "triangle_count": report["mesh"]["triangle_count"],
        "model_file": ply_path.name,
        "model_kind": "manhattan",
        "dimensions_meters": report["layout"]["dimensions_meters"],
        "height_meters": report["layout"]["height_meters"],
    }
    (output / "preview_manifest.json").write_text(
        json.dumps(preview_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
