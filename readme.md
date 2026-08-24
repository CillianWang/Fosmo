# Fosmo

Fosmo is an experimental, local-first indoor reconstruction pipeline for
non-LiDAR iPhones and Apple Silicon Macs.

The first milestone freezes `ScanBundle 1.0`: an iPhone records unmodified RGB
keyframes with ARKit intrinsics and poses, then the backend validates the bundle
before any depth inference runs.

## Repository layout

- `contracts/` — versioned ScanBundle schema and golden fixture.
- `backend/` — dependency-free Python validator and CLI.
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

See [minimum_closed_loop_plan.md](minimum_closed_loop_plan.md) for the staged
implementation plan.
