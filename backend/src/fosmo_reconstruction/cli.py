"""Command-line entry point for Fosmo metric reconstruction."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import ReconstructionError, reconstruct_single_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct one metric colored point cloud from a Fosmo ScanBundle"
    )
    parser.add_argument("bundle", help="path to a validated .scanbundle directory")
    parser.add_argument("--output", required=True, help="artifact output directory")
    parser.add_argument("--checkpoint", required=True, help="Depth Pro checkpoint path")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--stride", type=int, default=4, help="point-cloud pixel stride")
    parser.add_argument("--minimum-depth", type=float, default=0.1)
    parser.add_argument("--maximum-depth", type=float, default=20.0)
    args = parser.parse_args()
    try:
        report = reconstruct_single_frame(
            args.bundle,
            args.output,
            args.checkpoint,
            device=args.device,
            stride=args.stride,
            minimum_depth=args.minimum_depth,
            maximum_depth=args.maximum_depth,
        )
    except (ReconstructionError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
