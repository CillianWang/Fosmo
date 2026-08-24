"""Command-line entry point for clean Manhattan room extraction."""

from __future__ import annotations

import argparse
import json

from .manhattan import reconstruct_manhattan_space
from .pipeline import ReconstructionError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a clean orthogonal floor-and-wall room from a Fosmo fusion"
    )
    parser.add_argument("fusion_directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = reconstruct_manhattan_space(args.fusion_directory, args.output)
    except (ReconstructionError, OSError, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

