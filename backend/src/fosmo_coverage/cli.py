"""Replay a ScanBundle through the backend coverage policy."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from .tracker import CoverageFrameError, CoverageTracker, FrameObservation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate full-circle coverage in an existing ScanBundle")
    parser.add_argument("bundle")
    args = parser.parse_args()
    bundle = Path(args.bundle).expanduser()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    tracker = CoverageTracker()
    errors: list[dict[str, object]] = []
    for frame in manifest["frames"]:
        image = (bundle / frame["image"]).read_bytes()
        observation = FrameObservation(
            frame_id=frame["id"],
            timestamp=frame["timestamp"],
            world_from_camera=tuple(frame["world_from_camera"]),
            sharpness=1.0,
            mean_luminance=0.5,
            thumbnail_base64=base64.b64encode(image).decode("ascii"),
        )
        try:
            tracker.ingest(observation)
        except CoverageFrameError as error:
            errors.append({"frame_id": frame["id"], "error": str(error)})
    print(json.dumps({**tracker.status().to_dict(), "frame_errors": errors}, indent=2))
    return 0 if tracker.status().complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
