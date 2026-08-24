# Fosmo

Fosmo is an experimental, local-first indoor reconstruction pipeline for
non-LiDAR iPhones and Apple Silicon Macs.

The first milestone freezes `ScanBundle 1.0`: an iPhone records unmodified RGB
keyframes with ARKit intrinsics and poses, then the backend validates the bundle
before any depth inference runs.

## Repository layout

- `contracts/` — versioned ScanBundle schema and golden fixture.
- `backend/` — dependency-free Python validator and CLI.
- `ios/FosmoCapture/` — installable SwiftUI + ARKit iPhone app.
- `ios/ScanBundleKit/` — Swift contract types and bundle writer.
- `docs/` — implementation decisions and protocol documentation.

## Validate a ScanBundle

```bash
cd backend
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m fosmo_scanbundle.cli ../contracts/examples/golden.scanbundle
```

## Test the Swift contract

```bash
cd ios/ScanBundleKit
swift test
```

## Reconstruct a metric point cloud

Set up the pinned Apple Silicon Depth Pro environment and checkpoint:

```bash
./scripts/setup-reconstruction.sh
```

Then run the single-frame M1 reconstruction:

```bash
PYTHONPATH=backend/src backend/.venv-reconstruction/bin/python \
  -m fosmo_reconstruction.cli /path/to/scan.scanbundle \
  --output /path/to/output \
  --checkpoint .vendor/ml-depth-pro/checkpoints/depth_pro.pt \
  --device mps
```

This produces metric `depth.npy`, an upright depth preview, a colored
ARKit-world `pointcloud.ply`, a point-cloud preview, and `report.json`. See
[`docs/reconstruction-smoke-test.md`](docs/reconstruction-smoke-test.md) for
the coordinate conventions and current single-frame limitation.

Fuse the full scan and export a Blender-ready colored mesh:

```bash
PYTHONPATH=backend/src backend/.venv-reconstruction/bin/python \
  -m fosmo_reconstruction.fusion_cli /path/to/scan.scanbundle \
  --output /path/to/fused-output \
  --checkpoint .vendor/ml-depth-pro/checkpoints/depth_pro.pt \
  --device mps
```

The preferred Blender artifact is `room_blender_mesh.glb`; the full-resolution
PLY mesh and fused point cloud are retained alongside it.

For a faster preview, add `--fast`. It samples 8 frames evenly across the full
turn, uses a 640-pixel TSDF integration width, and targets 150,000 triangles.
It preserves 360-degree coverage while trading local detail for substantially
less Depth Pro inference and a smaller phone preview.

## Run the full-circle coverage backend

The capture app only declares a scan complete after this lightweight local
service has received acceptable JPEG evidence for all 12 yaw sectors:

```bash
cd backend
PYTHONPATH=src python3 -m fosmo_coverage.server --host 0.0.0.0 --port 8765 \
  --preview-directory /path/to/fused-output
```

The service keeps coverage session state in memory and stores no captured
images. When a preview directory is supplied, it also serves the existing
`room_blender_mesh.ply` through `/preview/manifest` and `/preview/model.ply`.
Check it with
`curl http://127.0.0.1:8765/health`, then enter the Mac's LAN URL in the iPhone
app (for example, `http://192.168.1.20:8765`). This milestone confirms image
view coverage, not semantic wall or room-plane detection.

## Run the iPhone capture app

Open [`ios/FosmoCapture/FosmoCapture.xcodeproj`](ios/FosmoCapture/FosmoCapture.xcodeproj)
in Xcode, select a physical iPhone, and run the `FosmoCapture` scheme. ARKit
capture is intentionally unavailable in the Simulator.

The app records automatically selected RGB keyframes throughout a complete
360-degree turn. Its progress and export button are driven by JPEGs actually
accepted by the coverage backend; there is no fixed five-frame completion.
After confirmation it writes a ScanBundle 1.0 directory and presents the
system share sheet for AirDrop or Files export.
From the idle screen, **查看最近重建** downloads the configured PLY from the Mac
and opens an interactive native SceneKit viewer; Blender is not involved.
See [`docs/iphone-smoke-test.md`](docs/iphone-smoke-test.md) for the end-to-end
test procedure.

See [minimum_closed_loop_plan.md](minimum_closed_loop_plan.md) for the staged
implementation plan.
