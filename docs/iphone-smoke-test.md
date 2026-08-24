# iPhone ScanBundle smoke test

This test proves the first real device loop: ARKit capture on an iPhone,
ScanBundle export, and validation on the Mac.

## On the iPhone

1. Build and install the `FosmoCapture` scheme on a physical iPhone.
2. If iOS blocks the first launch, trust the development profile for the
   installed app in Settings, then open FosmoCapture again.
3. Allow camera access.
4. Hold the phone upright and wait until the status says `跟踪正常`.
5. Tap `开始扫描` and move slowly around a small room. Translate the phone as
   well as rotating it; the app rejects frames that are too close together,
   blurry, badly exposed, or captured while ARKit tracking is limited.
6. After three frames, either tap `结束并导出` or continue until the app
   automatically stops at five frames.
7. Tap `分享 ScanBundle` and send the `.scanbundle` item to the Mac using
   AirDrop or Save to Files.

## On the Mac

Run the validator against the received directory:

```bash
cd backend
PYTHONPATH=src python3 -m fosmo_scanbundle.cli /path/to/<scan-id>.scanbundle --json
```

The command must exit with code `0`, report schema version `1.0`, and report
between three and five frames. If it fails, preserve the exported bundle: its
error report is the input for the next capture/contract fix.
