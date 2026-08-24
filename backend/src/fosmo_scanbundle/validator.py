"""Fail-closed validation for Fosmo ScanBundle 1.0."""

from __future__ import annotations

import json
import math
import re
import struct
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


SUPPORTED_SCHEMA_VERSION = "1.0"
IMAGE_PATTERN = re.compile(r"^frames/(?P<id>[0-9]{6})\.jpg$")
SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    bundle: str
    schema_version: str | None
    scan_id: str | None
    frame_count: int
    valid: bool
    issues: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["issues"] = [asdict(issue) for issue in self.issues]
        return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _matrix_is_rigid(values: list[float], tolerance: float = 1e-3) -> bool:
    rotation = [values[0:3], values[4:7], values[8:11]]
    for row in rotation:
        if abs(sum(component * component for component in row) - 1.0) > tolerance:
            return False
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if abs(sum(rotation[left][i] * rotation[right][i] for i in range(3))) > tolerance:
            return False
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    return abs(determinant - 1.0) <= tolerance


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions without adding an image-library dependency."""

    with path.open("rb") as image:
        if image.read(2) != b"\xff\xd8":
            raise ValueError("file does not start with the JPEG SOI marker")

        while True:
            prefix = image.read(1)
            if not prefix:
                break
            if prefix != b"\xff":
                continue

            marker_byte = image.read(1)
            while marker_byte == b"\xff":
                marker_byte = image.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]

            if marker in {0x01, 0xD8, 0xD9}:
                continue
            length_bytes = image.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                raise ValueError("JPEG contains an invalid segment length")

            if marker in SOF_MARKERS:
                header = image.read(5)
                if len(header) != 5:
                    break
                height, width = struct.unpack(">HH", header[1:])
                if width <= 0 or height <= 0:
                    raise ValueError("JPEG dimensions must be positive")
                return width, height

            image.seek(segment_length - 2, 1)

    raise ValueError("JPEG has no supported start-of-frame marker")


class _Collector:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def add(self, code: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(code=code, path=path, message=message))

    def exact_keys(self, value: Any, expected: set[str], path: str) -> bool:
        if not isinstance(value, dict):
            self.add("invalid_type", path, "must be an object")
            return False
        missing = expected - value.keys()
        unknown = value.keys() - expected
        for key in sorted(missing):
            self.add("missing_field", f"{path}.{key}", "is required")
        for key in sorted(unknown):
            self.add("unknown_field", f"{path}.{key}", "is not part of ScanBundle 1.0")
        return not missing


def validate_bundle(bundle_path: str | Path) -> ValidationReport:
    bundle = Path(bundle_path).expanduser()
    collector = _Collector()
    manifest_path = bundle / "manifest.json"
    manifest: dict[str, Any] | None = None

    if not bundle.is_dir():
        collector.add("bundle_not_found", "$", "bundle path must be an existing directory")
    elif not manifest_path.is_file():
        collector.add("manifest_missing", "$.manifest", "manifest.json is required")
    else:
        try:
            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                manifest = parsed
            else:
                collector.add("invalid_type", "$", "manifest root must be an object")
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            collector.add("manifest_invalid", "$.manifest", str(error))

    required_root = {"schema_version", "scan_id", "created_at", "device", "coordinate_system", "frames"}
    if manifest is not None and collector.exact_keys(manifest, required_root, "$"):
        _validate_header(manifest, collector)
        _validate_frames(bundle, manifest.get("frames"), collector)

    manifest_values = manifest or {}
    frames = manifest_values.get("frames")
    return ValidationReport(
        bundle=str(bundle.resolve()),
        schema_version=(
            manifest_values.get("schema_version")
            if isinstance(manifest_values.get("schema_version"), str)
            else None
        ),
        scan_id=manifest_values.get("scan_id") if isinstance(manifest_values.get("scan_id"), str) else None,
        frame_count=len(frames) if isinstance(frames, list) else 0,
        valid=not collector.issues,
        issues=tuple(collector.issues),
    )


def _validate_header(manifest: dict[str, Any], collector: _Collector) -> None:
    if manifest["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        collector.add(
            "unsupported_schema_version",
            "$.schema_version",
            f"expected {SUPPORTED_SCHEMA_VERSION!r}",
        )

    try:
        uuid.UUID(manifest["scan_id"])
    except (AttributeError, TypeError, ValueError):
        collector.add("invalid_uuid", "$.scan_id", "must be a UUID string")

    created_at = manifest["created_at"]
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError("timezone is required")
    except (AttributeError, TypeError, ValueError):
        collector.add("invalid_datetime", "$.created_at", "must be ISO-8601 with a timezone")

    device = manifest["device"]
    if collector.exact_keys(device, {"model", "os_version"}, "$.device"):
        for key in ("model", "os_version"):
            if not isinstance(device[key], str) or not device[key].strip():
                collector.add("invalid_value", f"$.device.{key}", "must be a non-empty string")

    coordinate = manifest["coordinate_system"]
    expected_coordinate = {
        "pose": "ARKit worldFromCamera",
        "matrix_layout": "row-major",
        "units": "meters",
    }
    if collector.exact_keys(coordinate, set(expected_coordinate), "$.coordinate_system"):
        for key, expected in expected_coordinate.items():
            if coordinate[key] != expected:
                collector.add("invalid_coordinate_system", f"$.coordinate_system.{key}", f"must equal {expected!r}")


def _validate_frames(bundle: Path, frames: Any, collector: _Collector) -> None:
    if not isinstance(frames, list) or not frames:
        collector.add("invalid_frames", "$.frames", "must be a non-empty array")
        return

    expected_keys = {
        "id",
        "image",
        "timestamp",
        "image_width",
        "image_height",
        "calibration_width",
        "calibration_height",
        "intrinsics",
        "world_from_camera",
        "tracking_state",
        "tracking_reason",
        "image_orientation",
        "image_format_version",
    }
    seen_ids: set[int] = set()
    previous_timestamp = -1.0

    for index, frame in enumerate(frames):
        path = f"$.frames[{index}]"
        if not collector.exact_keys(frame, expected_keys, path):
            continue

        frame_id = frame["id"]
        if not _is_positive_int(frame_id):
            collector.add("invalid_frame_id", f"{path}.id", "must be a positive integer")
        elif frame_id in seen_ids:
            collector.add("duplicate_frame_id", f"{path}.id", "must be unique")
        else:
            seen_ids.add(frame_id)

        timestamp = frame["timestamp"]
        if not _is_number(timestamp) or timestamp < 0:
            collector.add("invalid_timestamp", f"{path}.timestamp", "must be finite and non-negative")
        elif timestamp <= previous_timestamp:
            collector.add("timestamp_order", f"{path}.timestamp", "must be strictly increasing")
        else:
            previous_timestamp = float(timestamp)

        dimensions_valid = True
        for key in ("image_width", "image_height", "calibration_width", "calibration_height"):
            if not _is_positive_int(frame[key]):
                dimensions_valid = False
                collector.add("invalid_dimension", f"{path}.{key}", "must be a positive integer")

        _validate_intrinsics(frame["intrinsics"], f"{path}.intrinsics", collector)
        _validate_pose(frame["world_from_camera"], f"{path}.world_from_camera", collector)

        expected_values = {
            "tracking_state": "normal",
            "tracking_reason": None,
            "image_orientation": "sensor-native",
            "image_format_version": "jpeg-v1",
        }
        for key, expected in expected_values.items():
            if frame[key] != expected:
                collector.add("invalid_value", f"{path}.{key}", f"must equal {expected!r}")

        image_path = _validate_image_path(bundle, frame["image"], frame_id, path, collector)
        if image_path is not None and dimensions_valid:
            try:
                actual_width, actual_height = _jpeg_dimensions(image_path)
                declared = (frame["image_width"], frame["image_height"])
                if declared != (actual_width, actual_height):
                    collector.add(
                        "image_dimension_mismatch",
                        f"{path}.image",
                        f"manifest declares {declared[0]}x{declared[1]}, JPEG is {actual_width}x{actual_height}",
                    )
            except (OSError, ValueError) as error:
                collector.add("invalid_jpeg", f"{path}.image", str(error))


def _validate_image_path(
    bundle: Path,
    image: Any,
    frame_id: Any,
    frame_path: str,
    collector: _Collector,
) -> Path | None:
    if not isinstance(image, str):
        collector.add("invalid_image_path", f"{frame_path}.image", "must be a relative string path")
        return None

    pure_path = PurePosixPath(image)
    match = IMAGE_PATTERN.fullmatch(image)
    if pure_path.is_absolute() or ".." in pure_path.parts or not match:
        collector.add(
            "invalid_image_path",
            f"{frame_path}.image",
            "must match frames/000001.jpg and remain inside the bundle",
        )
        return None
    if _is_positive_int(frame_id) and int(match.group("id")) != frame_id:
        collector.add("image_id_mismatch", f"{frame_path}.image", "filename must match the frame id")

    image_path = bundle / pure_path
    if image_path.is_symlink():
        collector.add("unsafe_image_path", f"{frame_path}.image", "symbolic links are not allowed")
        return None
    if not image_path.is_file():
        collector.add("image_missing", f"{frame_path}.image", "referenced JPEG does not exist")
        return None
    try:
        image_path.resolve().relative_to(bundle.resolve())
    except ValueError:
        collector.add("unsafe_image_path", f"{frame_path}.image", "resolved path leaves the bundle")
        return None
    return image_path


def _validate_intrinsics(values: Any, path: str, collector: _Collector) -> None:
    if not isinstance(values, list) or len(values) != 9 or not all(_is_number(value) for value in values):
        collector.add("invalid_intrinsics", path, "must contain exactly 9 finite numbers")
        return
    if values[0] <= 0 or values[4] <= 0:
        collector.add("invalid_intrinsics", path, "fx and fy must be positive")
    if any(abs(values[index] - expected) > 1e-6 for index, expected in ((6, 0), (7, 0), (8, 1))):
        collector.add("invalid_intrinsics", path, "last row must be [0, 0, 1] in row-major order")


def _validate_pose(values: Any, path: str, collector: _Collector) -> None:
    if not isinstance(values, list) or len(values) != 16 or not all(_is_number(value) for value in values):
        collector.add("invalid_pose", path, "must contain exactly 16 finite numbers")
        return
    if any(abs(values[index] - expected) > 1e-6 for index, expected in ((12, 0), (13, 0), (14, 0), (15, 1))):
        collector.add("invalid_pose", path, "last row must be [0, 0, 0, 1] in row-major order")
    if not _matrix_is_rigid(values):
        collector.add("invalid_pose", path, "rotation block must be a right-handed orthonormal matrix")
