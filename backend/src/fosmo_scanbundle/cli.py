"""Command-line entry point for ScanBundle validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .validator import validate_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Fosmo ScanBundle 1.0 directory")
    parser.add_argument("bundle", help="path to a .scanbundle directory")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit a machine-readable report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_bundle(args.bundle)
    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    elif report.valid:
        print(f"valid ScanBundle {report.schema_version}: {report.scan_id} ({report.frame_count} frames)")
    else:
        print(f"invalid ScanBundle: {len(report.issues)} issue(s)")
        for issue in report.issues:
            print(f"- [{issue.code}] {issue.path}: {issue.message}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
