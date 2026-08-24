# Single-frame metric reconstruction smoke test

This is Fosmo milestone M1: one suitable frame from a validated ScanBundle is
processed by Apple Depth Pro, backprojected with ARKit intrinsics, transformed
by `worldFromCamera`, and exported as a colored metric point cloud.

It is a real reconstruction, but it is intentionally not yet a fused 360-degree
room model. The all-frame command below adds initial TSDF fusion; wall/room
measurement and structural cleanup remain later M2 work.

## Set up the isolated environment

On the supported Apple Silicon Mac, run:

```bash
./scripts/setup-reconstruction.sh
```

The script uses macOS Python 3.9 by default, creates
`backend/.venv-reconstruction`, checks out the pinned official Depth Pro source
under `.vendor/`, installs pinned PyTorch dependencies, downloads the official
1.8 GB checkpoint, verifies its SHA-256 digest, and reports MPS availability.
None of these generated files are committed.

## Reconstruct a ScanBundle

```bash
PYTHONPATH=backend/src backend/.venv-reconstruction/bin/python \
  -m fosmo_reconstruction.cli \
  /path/to/scan.scanbundle \
  --output /path/to/output \
  --checkpoint .vendor/ml-depth-pro/checkpoints/depth_pro.pt \
  --device mps \
  --stride 4
```

The pipeline validates the entire bundle, rejects frames pitched more than 45
degrees from level, and selects the sharpest remaining frame. It uses the ARKit
gravity-aligned pose to rotate sensor-native pixels upright for Depth Pro, then
rotates the predicted depth back before geometric projection.

## Artifacts

- `depth.npy` — float32 metric depth aligned to sensor-native image pixels.
- `depth_preview.png` — upright grayscale visualization; nearer pixels are brighter.
- `pointcloud.ply` — binary little-endian RGB points in ARKit world meters.
- `pointcloud_preview.png` — top, front, and side world-coordinate projections.
- `report.json` — source frame, model/runtime, depth range, point count, bounds,
  timings, artifact paths, and explicit limitations.

Run geometry regression tests inside the reconstruction environment:

```bash
PYTHONPATH=backend/src backend/.venv-reconstruction/bin/python \
  -m unittest backend/tests/test_reconstruction_geometry.py -v
```

## Fuse every frame for Blender

```bash
PYTHONPATH=backend/src backend/.venv-reconstruction/bin/python \
  -m fosmo_reconstruction.fusion_cli \
  /path/to/scan.scanbundle \
  --output /path/to/fused-output \
  --checkpoint .vendor/ml-depth-pro/checkpoints/depth_pro.pt \
  --device mps
```

The command caches float16 depth for every frame, then automatically starts a
fresh process for Open3D TSDF fusion. The process boundary avoids loading the
large PyTorch/MPS model and Open3D in the same address space. Interrupted model
inference resumes from the last complete depth file.

Fusion outputs include:

- `room_fused_pointcloud.ply` — voxel-downsampled colored world point cloud.
- `room_mesh_full.ply` — full TSDF triangle mesh.
- `room_blender_mesh.ply` — simplified colored mesh.
- `room_blender_mesh.glb` — preferred Blender import artifact.
- `room_model_preview.png` and `fusion_report.json`.

In Blender, choose **File → Import → glTF 2.0** and select
`room_blender_mesh.glb`. glTF uses meters and Y-up; Blender performs its normal
axis conversion during import. This first fused model may contain furniture,
surface thickness, and ghosting because each frame's depth is independently
predicted. It must not yet be treated as a measured architectural shell.
