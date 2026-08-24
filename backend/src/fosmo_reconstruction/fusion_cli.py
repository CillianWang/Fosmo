"""Command-line entry point for all-frame TSDF fusion."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .fusion import reconstruct_all_frames
from .pipeline import ReconstructionError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse every ScanBundle frame into a Blender-compatible room mesh"
    )
    parser.add_argument("bundle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="infer 8 evenly spaced full-turn frames and use a lighter fusion mesh",
    )
    parser.add_argument("--max-inference-frames", type=int)
    parser.add_argument("--voxel-length", type=float)
    parser.add_argument("--sdf-truncation", type=float)
    parser.add_argument("--integration-width", type=int)
    parser.add_argument("--maximum-depth", type=float, default=6.0)
    parser.add_argument("--blender-triangle-target", type=int)
    parser.add_argument("--fusion-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        report = reconstruct_all_frames(
            args.bundle,
            args.output,
            args.checkpoint,
            device=args.device,
            voxel_length=args.voxel_length if args.voxel_length is not None else (0.04 if args.fast else 0.025),
            sdf_truncation=args.sdf_truncation if args.sdf_truncation is not None else (0.16 if args.fast else 0.10),
            integration_width=args.integration_width if args.integration_width is not None else (640 if args.fast else 960),
            maximum_depth=args.maximum_depth,
            blender_triangle_target=(
                args.blender_triangle_target
                if args.blender_triangle_target is not None
                else (150000 if args.fast else 300000)
            ),
            maximum_inference_frames=(
                args.max_inference_frames
                if args.max_inference_frames is not None
                else (8 if args.fast else None)
            ),
            inference_only=not args.fusion_only,
        )
    except (ReconstructionError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from error
    if not args.fusion_only:
        os.execv(
            sys.executable,
            [sys.executable, "-m", "fosmo_reconstruction.fusion_cli", *sys.argv[1:], "--fusion-only"],
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
