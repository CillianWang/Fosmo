from __future__ import annotations

import base64
import math
import json
import sys
import threading
import unittest
from pathlib import Path
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fosmo_coverage.tracker import (  # noqa: E402
    CoverageFrameError,
    CoveragePolicy,
    CoverageTracker,
    FrameObservation,
)
from fosmo_coverage.server import CoverageRequestHandler, CoverageSessionStore  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
JPEG = (REPOSITORY_ROOT / "contracts/examples/golden.scanbundle/frames/000001.jpg").read_bytes()


def pose_at_yaw(yaw: float) -> tuple[float, ...]:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine, 0.0, sine, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -sine, 0.0, cosine, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def observation(frame_id: int, yaw: float, timestamp: float, *, sharpness: float = 0.2) -> FrameObservation:
    return FrameObservation(
        frame_id=frame_id,
        timestamp=timestamp,
        world_from_camera=pose_at_yaw(yaw),
        sharpness=sharpness,
        mean_luminance=0.5,
        thumbnail_base64=base64.b64encode(JPEG).decode("ascii"),
    )


class CoverageTrackerTests(unittest.TestCase):
    def test_full_circle_of_received_images_completes(self) -> None:
        policy = CoveragePolicy(
            yaw_sector_count=12,
            minimum_duration_seconds=8,
            minimum_thumbnail_width=4,
            minimum_thumbnail_height=3,
        )
        tracker = CoverageTracker(policy)

        for index in range(12):
            tracker.ingest(observation(index + 1, (index + 0.5) * 2 * math.pi / 12, float(index)))

        status = tracker.status()
        self.assertTrue(status.complete)
        self.assertEqual(status.sectors_covered, 12)
        self.assertEqual(status.accepted_frame_count, 12)

    def test_repeated_direction_does_not_fake_coverage(self) -> None:
        policy = CoveragePolicy(minimum_thumbnail_width=4, minimum_thumbnail_height=3)
        tracker = CoverageTracker(policy)

        for index in range(20):
            tracker.ingest(observation(index + 1, 0, float(index)))

        self.assertFalse(tracker.status().complete)
        self.assertEqual(tracker.status().sectors_covered, 1)

    def test_backend_rejects_blurry_evidence(self) -> None:
        policy = CoveragePolicy(minimum_thumbnail_width=4, minimum_thumbnail_height=3)
        tracker = CoverageTracker(policy)

        with self.assertRaisesRegex(CoverageFrameError, "too blurry"):
            tracker.ingest(observation(1, 0, 0, sharpness=0.001))

        self.assertEqual(tracker.status().rejected_frame_count, 1)

    def test_all_sectors_still_require_minimum_duration(self) -> None:
        policy = CoveragePolicy(
            yaw_sector_count=4,
            minimum_duration_seconds=8,
            minimum_thumbnail_width=4,
            minimum_thumbnail_height=3,
        )
        tracker = CoverageTracker(policy)
        for index in range(4):
            tracker.ingest(observation(index + 1, (index + 0.5) * math.pi / 2, float(index)))

        status = tracker.status()
        self.assertFalse(status.complete)
        self.assertEqual(status.sectors_covered, 4)
        self.assertIn("continue steadily", status.message)


class QuietCoverageRequestHandler(CoverageRequestHandler):
    store = CoverageSessionStore()

    def log_message(self, format: str, *args: object) -> None:
        pass


class CoverageServerTests(unittest.TestCase):
    def setUp(self) -> None:
        QuietCoverageRequestHandler.store = CoverageSessionStore()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietCoverageRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_health_and_session_creation(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        connection.request("GET", "/health")
        response = connection.getresponse()
        health = json.loads(response.read())
        self.assertEqual(response.status, 200)
        self.assertEqual(health["status"], "ok")

        connection.request(
            "POST",
            "/coverage/sessions",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        session = json.loads(response.read())
        connection.close()
        self.assertEqual(response.status, 201)
        self.assertEqual(session["sectors_covered"], 0)
        self.assertEqual(session["sectors_required"], 12)
        self.assertFalse(session["complete"])
        self.assertIn("session_id", session)


if __name__ == "__main__":
    unittest.main()
