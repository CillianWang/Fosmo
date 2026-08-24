"""Evidence-based view coverage tracking for an indoor scan session."""

from __future__ import annotations

import base64
import binascii
import io
import math
import struct
from dataclasses import asdict, dataclass
from typing import Any


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


class CoverageFrameError(ValueError):
    """Raised when the backend cannot accept a coverage observation."""


@dataclass(frozen=True)
class CoveragePolicy:
    yaw_sector_count: int = 12
    minimum_duration_seconds: float = 8.0
    maximum_absolute_pitch_degrees: float = 50.0
    minimum_sharpness: float = 0.015
    minimum_luminance: float = 0.08
    maximum_luminance: float = 0.92
    minimum_thumbnail_width: int = 160
    minimum_thumbnail_height: int = 120


@dataclass(frozen=True)
class FrameObservation:
    frame_id: int
    timestamp: float
    world_from_camera: tuple[float, ...]
    sharpness: float
    mean_luminance: float
    thumbnail_base64: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FrameObservation:
        required = {
            "frame_id",
            "timestamp",
            "world_from_camera",
            "sharpness",
            "mean_luminance",
            "thumbnail_base64",
        }
        missing = required - payload.keys()
        if missing:
            raise CoverageFrameError(f"missing field(s): {', '.join(sorted(missing))}")
        matrix = payload["world_from_camera"]
        if not isinstance(matrix, list):
            raise CoverageFrameError("world_from_camera must be an array")
        return cls(
            frame_id=payload["frame_id"],
            timestamp=payload["timestamp"],
            world_from_camera=tuple(matrix),
            sharpness=payload["sharpness"],
            mean_luminance=payload["mean_luminance"],
            thumbnail_base64=payload["thumbnail_base64"],
        )


@dataclass(frozen=True)
class CoverageStatus:
    complete: bool
    accepted_frame_count: int
    rejected_frame_count: int
    sectors_covered: int
    sectors_required: int
    elapsed_seconds: float
    missing_sector_centers_degrees: tuple[float, ...]
    message: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["missing_sector_centers_degrees"] = list(self.missing_sector_centers_degrees)
        return result


class CoverageTracker:
    """Confirms a full yaw loop only from JPEGs actually received by the backend."""

    def __init__(self, policy: CoveragePolicy | None = None) -> None:
        self.policy = policy or CoveragePolicy()
        if self.policy.yaw_sector_count < 4:
            raise ValueError("yaw_sector_count must be at least 4")
        self._covered_sectors: set[int] = set()
        self._accepted_ids: set[int] = set()
        self._accepted_frame_count = 0
        self._rejected_frame_count = 0
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None

    def ingest(self, observation: FrameObservation) -> CoverageStatus:
        try:
            sector = self._validate_and_sector(observation)
        except CoverageFrameError:
            self._rejected_frame_count += 1
            raise

        if observation.frame_id not in self._accepted_ids:
            self._accepted_ids.add(observation.frame_id)
            self._accepted_frame_count += 1
            self._covered_sectors.add(sector)
            if self._first_timestamp is None:
                self._first_timestamp = observation.timestamp
            self._last_timestamp = observation.timestamp
        return self.status()

    def status(self) -> CoverageStatus:
        elapsed = 0.0
        if self._first_timestamp is not None and self._last_timestamp is not None:
            elapsed = max(0.0, self._last_timestamp - self._first_timestamp)
        missing = tuple(
            round((index + 0.5) * 360.0 / self.policy.yaw_sector_count, 1)
            for index in range(self.policy.yaw_sector_count)
            if index not in self._covered_sectors
        )
        complete = not missing and elapsed >= self.policy.minimum_duration_seconds
        if complete:
            message = "backend confirmed a complete 360-degree image loop"
        elif missing:
            message = (
                f"backend received {len(self._covered_sectors)}/{self.policy.yaw_sector_count} "
                "view sectors; continue the same full-circle recording"
            )
        else:
            message = (
                f"all view sectors received; continue steadily for "
                f"{max(0.0, self.policy.minimum_duration_seconds - elapsed):.1f}s"
            )
        return CoverageStatus(
            complete=complete,
            accepted_frame_count=self._accepted_frame_count,
            rejected_frame_count=self._rejected_frame_count,
            sectors_covered=len(self._covered_sectors),
            sectors_required=self.policy.yaw_sector_count,
            elapsed_seconds=round(elapsed, 3),
            missing_sector_centers_degrees=missing,
            message=message,
        )

    def _validate_and_sector(self, observation: FrameObservation) -> int:
        if not isinstance(observation.frame_id, int) or isinstance(observation.frame_id, bool) or observation.frame_id < 1:
            raise CoverageFrameError("frame_id must be a positive integer")
        numeric_values = (
            observation.timestamp,
            observation.sharpness,
            observation.mean_luminance,
            *observation.world_from_camera,
        )
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in numeric_values):
            raise CoverageFrameError("numeric fields must contain finite numbers")
        if observation.timestamp < 0:
            raise CoverageFrameError("timestamp must be non-negative")
        if len(observation.world_from_camera) != 16:
            raise CoverageFrameError("world_from_camera must contain 16 row-major values")
        if observation.sharpness < self.policy.minimum_sharpness:
            raise CoverageFrameError("thumbnail is too blurry for coverage evidence")
        if not self.policy.minimum_luminance <= observation.mean_luminance <= self.policy.maximum_luminance:
            raise CoverageFrameError("thumbnail luminance is outside the accepted range")

        if not isinstance(observation.thumbnail_base64, str):
            raise CoverageFrameError("thumbnail_base64 must be a string")
        try:
            thumbnail = base64.b64decode(observation.thumbnail_base64, validate=True)
        except (binascii.Error, ValueError, TypeError) as error:
            raise CoverageFrameError("thumbnail_base64 is not valid base64") from error
        width, height = jpeg_dimensions(thumbnail)
        if width < self.policy.minimum_thumbnail_width or height < self.policy.minimum_thumbnail_height:
            raise CoverageFrameError(
                f"thumbnail must be at least {self.policy.minimum_thumbnail_width}x{self.policy.minimum_thumbnail_height}"
            )

        matrix = observation.world_from_camera
        forward = (-matrix[2], -matrix[6], -matrix[10])
        norm = math.sqrt(sum(component * component for component in forward))
        if norm < 0.9 or norm > 1.1:
            raise CoverageFrameError("camera forward vector is not unit length")
        normalized = tuple(component / norm for component in forward)
        pitch_degrees = math.degrees(math.asin(max(-1.0, min(1.0, normalized[1]))))
        if abs(pitch_degrees) > self.policy.maximum_absolute_pitch_degrees:
            raise CoverageFrameError("camera pitch does not provide wall-loop coverage")

        yaw = math.atan2(normalized[0], normalized[2]) % (2 * math.pi)
        sector_width = 2 * math.pi / self.policy.yaw_sector_count
        return min(self.policy.yaw_sector_count - 1, int(yaw / sector_width))


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    image = io.BytesIO(data)
    if image.read(2) != b"\xff\xd8":
        raise CoverageFrameError("thumbnail is not a JPEG")
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
            raise CoverageFrameError("thumbnail contains an invalid JPEG segment")
        if marker in SOF_MARKERS:
            header = image.read(5)
            if len(header) != 5:
                break
            height, width = struct.unpack(">HH", header[1:])
            return width, height
        image.seek(segment_length - 2, 1)
    raise CoverageFrameError("thumbnail JPEG has no supported start-of-frame marker")
