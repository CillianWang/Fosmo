from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "reconstruction dependencies are not installed")
class ReconstructionGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        from fosmo_reconstruction.geometry import (
            backproject_to_world,
            scale_intrinsics,
            write_binary_ply,
        )
        from fosmo_reconstruction.pipeline import (
            _clockwise_rotation_for_upright,
            _restore_sensor_depth,
        )
        from fosmo_reconstruction.fusion import (
            select_evenly_spaced_frames,
            world_to_cv_camera_extrinsic,
        )

        self.backproject_to_world = backproject_to_world
        self.scale_intrinsics = scale_intrinsics
        self.write_binary_ply = write_binary_ply
        self.clockwise_rotation_for_upright = _clockwise_rotation_for_upright
        self.restore_sensor_depth = _restore_sensor_depth
        self.select_evenly_spaced_frames = select_evenly_spaced_frames
        self.world_to_cv_camera_extrinsic = world_to_cv_camera_extrinsic

    def test_intrinsics_scale_with_image_resolution(self) -> None:
        scaled = self.scale_intrinsics(
            [100, 0, 50, 0, 120, 40, 0, 0, 1],
            (100, 80),
            (200, 160),
        )
        np.testing.assert_allclose(scaled, [[200, 0, 100], [0, 240, 80], [0, 0, 1]])

    def test_center_pixel_maps_to_negative_arkit_camera_z(self) -> None:
        depth = np.array([[2.0]], dtype=np.float32)
        rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
        intrinsics = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]], dtype=np.float64)
        points, colors = self.backproject_to_world(
            depth,
            rgb,
            intrinsics,
            np.eye(4),
            stride=1,
        )
        np.testing.assert_allclose(points, [[0, 0, -2]])
        np.testing.assert_array_equal(colors, rgb.reshape(1, 3))

    def test_world_pose_translation_is_applied(self) -> None:
        pose = np.eye(4)
        pose[:3, 3] = (1, 2, 3)
        points, _ = self.backproject_to_world(
            np.array([[2.0]], dtype=np.float32),
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]], dtype=np.float64),
            pose,
            stride=1,
        )
        np.testing.assert_allclose(points, [[1, 2, 1]])

    def test_binary_ply_contains_declared_vertex(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cloud.ply"
            self.write_binary_ply(
                output,
                np.array([[1, 2, 3]], dtype=np.float32),
                np.array([[4, 5, 6]], dtype=np.uint8),
            )
            data = output.read_bytes()
            header, payload = data.split(b"end_header\n", 1)
            self.assertIn(b"element vertex 1", header)
            self.assertEqual(len(payload), 15)
            self.assertEqual(struct.unpack("<fffBBB", payload), (1.0, 2.0, 3.0, 4, 5, 6))

    def test_gravity_pose_rotates_portrait_capture_clockwise(self) -> None:
        frame = {"world_from_camera": [
            0, 0, 0, 0,
            -1, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 1,
        ]}
        self.assertEqual(self.clockwise_rotation_for_upright(frame), 90)

    def test_inference_depth_rotation_is_reversed_for_sensor_geometry(self) -> None:
        upright = np.array([[1, 2], [3, 4], [5, 6]])
        restored = self.restore_sensor_depth(upright, 90)
        np.testing.assert_array_equal(restored, [[2, 4, 6], [1, 3, 5]])

    def test_open3d_extrinsic_converts_arkit_camera_axes(self) -> None:
        extrinsic = self.world_to_cv_camera_extrinsic(np.eye(4))
        np.testing.assert_array_equal(extrinsic, np.diag([1, -1, -1, 1]))

    def test_open3d_extrinsic_inverts_world_from_camera_translation(self) -> None:
        pose = np.eye(4)
        pose[0, 3] = 1
        extrinsic = self.world_to_cv_camera_extrinsic(pose)
        np.testing.assert_allclose(extrinsic[:3, 3], [-1, 0, 0])

    def test_fast_frame_selection_spans_the_full_sequence(self) -> None:
        frames = [{"id": index + 1} for index in range(21)]
        selected = self.select_evenly_spaced_frames(frames, 8)
        self.assertEqual([frame["id"] for frame in selected], [1, 3, 6, 8, 11, 14, 16, 19])

    def test_frame_selection_keeps_short_sequences_unchanged(self) -> None:
        frames = [{"id": index + 1} for index in range(5)]
        self.assertIs(self.select_evenly_spaced_frames(frames, 8), frames)


if __name__ == "__main__":
    unittest.main()
