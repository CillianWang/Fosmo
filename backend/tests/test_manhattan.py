from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "reconstruction dependencies are not installed")
class ManhattanLayoutTests(unittest.TestCase):
    def test_recovers_rotated_floor_and_four_walls(self) -> None:
        from fosmo_reconstruction.manhattan import estimate_manhattan_layout

        angle = math.radians(17)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        points = []
        normals = []

        def world_xz(axis0: float, axis1: float) -> np.ndarray:
            return rotation @ np.array([axis0, axis1])

        for axis0 in np.linspace(-2, 2, 45):
            for axis1 in np.linspace(-3, 3, 65):
                x, z = world_xz(axis0, axis1)
                points.append([x, -1.5, z])
                normals.append([0, 1, 0])
        for axis0 in np.linspace(-2, 2, 30):
            for axis1 in np.linspace(-3, 3, 42):
                x, z = world_xz(axis0, axis1)
                points.append([x, 1.2, z])
                normals.append([0, -1, 0])
        for boundary, normal_sign in ((-2, -1), (2, 1)):
            normal_xz = rotation @ np.array([normal_sign, 0])
            for axis1 in np.linspace(-3, 3, 70):
                for y in np.linspace(-1.5, 1.2, 34):
                    x, z = world_xz(boundary, axis1)
                    points.append([x, y, z])
                    normals.append([normal_xz[0], 0, normal_xz[1]])
        for boundary, normal_sign in ((-3, -1), (3, 1)):
            normal_xz = rotation @ np.array([0, normal_sign])
            for axis0 in np.linspace(-2, 2, 50):
                for y in np.linspace(-1.5, 1.2, 34):
                    x, z = world_xz(axis0, boundary)
                    points.append([x, y, z])
                    normals.append([normal_xz[0], 0, normal_xz[1]])

        layout = estimate_manhattan_layout(np.asarray(points), np.asarray(normals))

        self.assertAlmostEqual(layout.floor_y, -1.5, delta=0.05)
        self.assertAlmostEqual(layout.ceiling_y, 1.2, delta=0.05)
        dimensions = sorted((layout.width_axis0, layout.width_axis1))
        np.testing.assert_allclose(dimensions, [4, 6], atol=0.12)
        self.assertAlmostEqual(layout.height, 2.7, delta=0.08)


if __name__ == "__main__":
    unittest.main()

