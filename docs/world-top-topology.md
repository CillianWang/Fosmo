# World-top topology reconstruction

This stage turns an existing fused point cloud into a lightweight model for the
iPhone viewer. The X/Z **World top** projection is the primary spatial evidence.
It does not infer a rectangular room envelope or require the walls to form a
closed polygon.

## Run it

First produce a fused output directory with `fusion_cli`. Then run:

```bash
PYTHONPATH=backend/src backend/.venv-reconstruction/bin/python \
  -m fosmo_reconstruction.topology_cli /path/to/fused-output \
  --output /path/to/world-top-output
```

The output directory contains:

- `room_blender_mesh.ply`, served directly to the native iPhone viewer;
- `room_blender_mesh.glb`, for other compatible 3D viewers;
- `room_world_top_preview.png`, comparing raw evidence with the clean result;
- `fusion_report.json`, including finite wall segments and viewer metadata.

Point the coverage server at the new output directory to preview it on iPhone:

```bash
PYTHONPATH=backend/src python3 -m fosmo_coverage.server \
  --host 0.0.0.0 --port 8765 \
  --preview-directory /path/to/world-top-output
```

## What is preserved

The extractor looks for X/Z cells that contain vertical evidence across a
substantial part of the estimated floor-to-ceiling height. It estimates the two
dominant orthogonal directions, snaps wall orientation, and retains multiple
independent finite wall runs. It closes only one-cell sampling gaps. It never
adds a missing side merely to close a room.

The floor uses the outermost X/Z boundary supported by horizontal floor samples
and the finite wall runs, then fills that convex polygon completely. This is an
intentional visualization choice: unlike the walls, the floor now spans
unobserved interior regions. The boundary remains an irregular polygon rather
than an axis-aligned room box.

The fusion report also carries the median ARKit capture position. The iPhone
viewer starts at that eye point, not outside the model. A one-finger drag turns
the camera in place through an unrestricted 360 degrees; pinching changes the
field of view and a double tap restores the captured viewpoint. Warm floor and
cool wall PBR materials, directional shadows, and a local fill light provide
depth cues.

## Current limitation

This output preserves the topology visible in the current World top, but it is
not yet a metrically verified architectural floor plan. The source uses
independent monocular depth inference per frame; residual scale and depth drift
can duplicate, bend, or offset surfaces. Axis snapping makes walls vertical and
orthogonal without proving which nearby runs are the same physical wall. Since
the requested floor uses the outermost evidence boundary, an isolated depth
outlier can also enlarge the filled floor footprint.
