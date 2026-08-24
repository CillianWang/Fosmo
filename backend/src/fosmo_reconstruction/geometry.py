"""Coordinate-safe depth backprojection and PLY export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def scale_intrinsics(
    intrinsics: list[float] | tuple[float, ...],
    calibration_size: tuple[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Scale row-major ARKit intrinsics from calibration to image pixels."""

    if len(intrinsics) != 9:
        raise ValueError("intrinsics must contain 9 row-major values")
    calibration_width, calibration_height = calibration_size
    image_width, image_height = image_size
    if min(calibration_width, calibration_height, image_width, image_height) <= 0:
        raise ValueError("calibration and image dimensions must be positive")

    matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3).copy()
    matrix[0, :] *= image_width / calibration_width
    matrix[1, :] *= image_height / calibration_height
    matrix[2, :] = (0.0, 0.0, 1.0)
    return matrix


def backproject_to_world(
    depth_meters: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    world_from_camera: list[float] | tuple[float, ...] | np.ndarray,
    *,
    stride: int = 4,
    minimum_depth: float = 0.1,
    maximum_depth: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Backproject CV image pixels into the ARKit world coordinate system.

    Depth pixels use the conventional CV camera frame: +X right, +Y down,
    +Z forward. ARKit cameras use +X right, +Y up, and look along -Z.
    """

    depth = np.asarray(depth_meters, dtype=np.float32)
    colors = np.asarray(rgb)
    if depth.ndim != 2:
        raise ValueError("depth must be a two-dimensional array")
    if colors.shape != (*depth.shape, 3):
        raise ValueError("rgb must have shape (depth_height, depth_width, 3)")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if not 0 < minimum_depth < maximum_depth:
        raise ValueError("depth limits must satisfy 0 < minimum < maximum")

    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError("intrinsics must be a 3x3 matrix")
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    if not np.isfinite((fx, fy, cx, cy)).all() or fx <= 0 or fy <= 0:
        raise ValueError("intrinsics contain invalid focal lengths or principal point")

    pose = np.asarray(world_from_camera, dtype=np.float64)
    if pose.size != 16:
        raise ValueError("world_from_camera must contain 16 row-major values")
    pose = pose.reshape(4, 4)

    rows = np.arange(0, depth.shape[0], stride)
    columns = np.arange(0, depth.shape[1], stride)
    u, v = np.meshgrid(columns, rows)
    sampled_depth = depth[np.ix_(rows, columns)]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= minimum_depth)
        & (sampled_depth <= maximum_depth)
    )
    if not valid.any():
        raise ValueError("depth map has no valid samples inside the configured range")

    z_cv = sampled_depth[valid].astype(np.float64)
    x_cv = (u[valid] - cx) * z_cv / fx
    y_cv = (v[valid] - cy) * z_cv / fy

    camera_arkit = np.stack(
        (x_cv, -y_cv, -z_cv, np.ones_like(z_cv)),
        axis=0,
    )
    world_homogeneous = pose @ camera_arkit
    world_points = (world_homogeneous[:3] / world_homogeneous[3]).T.astype(np.float32)
    world_colors = colors[np.ix_(rows, columns)][valid].astype(np.uint8)
    return world_points, world_colors


def point_cloud_bounds(points: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("points must be a non-empty Nx3 array")
    return {
        "minimum": [round(float(value), 6) for value in values.min(axis=0)],
        "maximum": [round(float(value), 6) for value in values.max(axis=0)],
        "extent": [round(float(value), 6) for value in np.ptp(values, axis=0)],
    }


def write_binary_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a compact binary little-endian colored point cloud."""

    vertices = np.asarray(points, dtype=np.float32)
    vertex_colors = np.asarray(colors, dtype=np.uint8)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if vertex_colors.shape != vertices.shape:
        raise ValueError("colors must have shape (N, 3)")

    record_type = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    records = np.empty(len(vertices), dtype=record_type)
    records["x"], records["y"], records["z"] = vertices.T
    records["red"], records["green"], records["blue"] = vertex_colors.T

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated by Fosmo metric reconstruction\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)
