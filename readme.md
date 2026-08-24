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

## Run the iPhone capture app

Open [`ios/FosmoCapture/FosmoCapture.xcodeproj`](ios/FosmoCapture/FosmoCapture.xcodeproj)
in Xcode, select a physical iPhone, and run the `FosmoCapture` scheme. ARKit
capture is intentionally unavailable in the Simulator.

The app records 3–5 automatically selected RGB keyframes, writes a ScanBundle
1.0 directory, and presents the system share sheet for AirDrop or Files export.
See [`docs/iphone-smoke-test.md`](docs/iphone-smoke-test.md) for the end-to-end
test procedure.

See [minimum_closed_loop_plan.md](minimum_closed_loop_plan.md) for the staged
implementation plan.
