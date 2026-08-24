from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fosmo_scanbundle.validator import validate_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_BUNDLE = REPOSITORY_ROOT / "contracts" / "examples" / "golden.scanbundle"


class ScanBundleValidatorTests(unittest.TestCase):
    def copy_golden_bundle(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        destination = Path(temporary.name) / "copy.scanbundle"
        shutil.copytree(GOLDEN_BUNDLE, destination)
        return temporary, destination

    def test_golden_bundle_is_valid(self) -> None:
        report = validate_bundle(GOLDEN_BUNDLE)
        self.assertTrue(report.valid, report.issues)
        self.assertEqual(report.schema_version, "1.0")
        self.assertEqual(report.frame_count, 3)

    def test_unknown_schema_version_fails_closed(self) -> None:
        temporary, bundle = self.copy_golden_bundle()
        self.addCleanup(temporary.cleanup)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = "2.0"
        manifest_path.write_text(json.dumps(manifest))

        report = validate_bundle(bundle)

        self.assertFalse(report.valid)
        self.assertIn("unsupported_schema_version", {issue.code for issue in report.issues})

    def test_empty_manifest_is_not_valid(self) -> None:
        temporary, bundle = self.copy_golden_bundle()
        self.addCleanup(temporary.cleanup)
        (bundle / "manifest.json").write_text("{}")

        report = validate_bundle(bundle)

        self.assertFalse(report.valid)
        self.assertIn("missing_field", {issue.code for issue in report.issues})

    def test_image_dimensions_must_match_jpeg(self) -> None:
        temporary, bundle = self.copy_golden_bundle()
        self.addCleanup(temporary.cleanup)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["frames"][0]["image_width"] = 999
        manifest_path.write_text(json.dumps(manifest))

        report = validate_bundle(bundle)

        self.assertFalse(report.valid)
        self.assertIn("image_dimension_mismatch", {issue.code for issue in report.issues})

    def test_matrix_layout_errors_are_reported(self) -> None:
        temporary, bundle = self.copy_golden_bundle()
        self.addCleanup(temporary.cleanup)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["frames"][0]["world_from_camera"][12] = 1.0
        manifest_path.write_text(json.dumps(manifest))

        report = validate_bundle(bundle)

        self.assertFalse(report.valid)
        self.assertIn("invalid_pose", {issue.code for issue in report.issues})

    def test_frame_paths_cannot_escape_bundle(self) -> None:
        temporary, bundle = self.copy_golden_bundle()
        self.addCleanup(temporary.cleanup)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["frames"][0]["image"] = "../secret.jpg"
        manifest_path.write_text(json.dumps(manifest))

        report = validate_bundle(bundle)

        self.assertFalse(report.valid)
        self.assertIn("invalid_image_path", {issue.code for issue in report.issues})

    def test_cli_emits_machine_readable_report(self) -> None:
        process = subprocess.run(
            [sys.executable, "-m", "fosmo_scanbundle.cli", str(GOLDEN_BUNDLE), "--json"],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT / "backend" / "src")},
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(json.loads(process.stdout)["valid"])


if __name__ == "__main__":
    unittest.main()
