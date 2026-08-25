"""Command-line entry point for World-top-driven topology reconstruction."""

from __future__ import annotations

import argparse
import json

from .pipeline import ReconstructionError
from .topology import reconstruct_world_top


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract finite wall segments and observed floor from a fused World top"
    )
    parser.add_argument("fusion_directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = reconstruct_world_top(args.fusion_directory, args.output)
    except (ReconstructionError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

