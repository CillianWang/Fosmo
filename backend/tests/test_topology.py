from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(np is not None, "reconstruction dependencies are not installed")
class WorldTopTopologyTests(unittest.TestCase):
    def test_surface_colors_are_sampled_from_matching_source_layers(self) -> None:
        from fosmo_reconstruction.topology import _sample_source_colors

        floor_points = np.asarray(
            [[offset * 0.01, -1.5, 0.0] for offset in range(-5, 5)]
        )
        wall_points = np.asarray(
            [[1.0, -0.2 + offset * 0.04, 0.0] for offset in range(10)]
        )
        source_points = np.vstack((floor_points, wall_points))
        source_normals = np.vstack(
            (np.tile([0, 1, 0], (10, 1)), np.tile([1, 0, 0], (10, 1)))
        )
        source_colors = np.vstack(
            (np.tile([0.9, 0.1, 0.1], (10, 1)), np.tile([0.1, 0.2, 0.9], (10, 1)))
        )
        sampled = _sample_source_colors(
            np.asarray([[0, -1.5, 0], [1, 0, 0]]),
            np.asarray([0, 1]),
            source_points,
            source_normals,
            source_colors,
            -1.5,
            1.2,
        )
        self.assertGreater(sampled[0, 0], 0.85)
        self.assertLess(sampled[0, 2], 0.2)
        self.assertGreater(sampled[1, 2], 0.85)
        self.assertLess(sampled[1, 0], 0.2)

    def test_floor_fills_outer_boundary_from_floor_and_walls(self) -> None:
        from fosmo_reconstruction.topology import WallSegment, _observed_floor_grid

        points = []
        normals = []
        for x in np.linspace(-0.5, 0.5, 21):
            for z in np.linspace(-0.5, 0.5, 21):
                points.append([x, -1.5, z])
                normals.append([0, 1, 0])

        corners = [(-2, -1.5), (2, -1.5), (2, 1.5), (-2, 1.5)]
        segments = []
        for index, start in enumerate(corners):
            end = corners[(index + 1) % len(corners)]
            points.extend(([start[0], 0, start[1]], [end[0], 0, end[1]]))
            normals.extend(([1, 0, 0], [1, 0, 0]))
            segments.append(
                WallSegment(np.asarray(start), np.asarray(end), index % 2, 20)
            )

        grid = _observed_floor_grid(
            np.asarray(points),
            np.asarray(normals),
            -1.5,
            tuple(segments),
        )
        center = np.floor((np.asarray([0.0, 0.0]) - grid.minimum_xz) / grid.resolution).astype(int)
        near_corner = np.floor(
            (np.asarray([1.8, 1.3]) - grid.minimum_xz) / grid.resolution
        ).astype(int)
        self.assertTrue(grid.mask[center[1], center[0]])
        self.assertTrue(grid.mask[near_corner[1], near_corner[0]])
        self.assertGreater(int(grid.mask.sum()), 700)

    def test_finite_l_shape_does_not_become_a_bounding_box(self) -> None:
        from fosmo_reconstruction.topology import extract_world_top_topology

        points = []
        normals = []
        for x in np.linspace(0, 3, 55):
            for z in np.linspace(0, 2, 40):
                points.extend(([x, -1.5, z], [x, 1.2, z]))
                normals.extend(([0, 1, 0], [0, -1, 0]))

        def add_wall(start: tuple[float, float], end: tuple[float, float], normal: tuple[float, float]) -> None:
            for amount in np.linspace(0, 1, 65):
                x = start[0] + amount * (end[0] - start[0])
                z = start[1] + amount * (end[1] - start[1])
                for y in np.linspace(-1.5, 1.2, 20):
                    points.append([x, y, z])
                    normals.append([normal[0], 0, normal[1]])

        add_wall((0, 0), (3, 0), (0, 1))
        add_wall((3, 0), (3, 2), (1, 0))
        add_wall((3, 2), (2, 2), (0, 1))
        result = extract_world_top_topology(np.asarray(points), np.asarray(normals))

        self.assertGreaterEqual(len(result.wall_segments), 3)
        self.assertTrue(any(segment.length > 2.5 for segment in result.wall_segments))
        # No evidence exists for the missing left edge x=0, z=0..2. A bounding-box
        # implementation would invent that wall; the finite-segment extractor must not.
        invented_left_edges = []
        for segment in result.wall_segments:
            midpoint = (segment.start_xz + segment.end_xz) / 2
            direction = segment.end_xz - segment.start_xz
            if abs(midpoint[0]) < 0.25 and abs(direction[1]) > 1.0:
                invented_left_edges.append(segment)
        self.assertEqual(invented_left_edges, [])


if __name__ == "__main__":
    unittest.main()
