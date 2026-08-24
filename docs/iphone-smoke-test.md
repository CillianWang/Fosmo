# iPhone ScanBundle smoke test

This test proves the first real device loop: ARKit capture on an iPhone,
backend-confirmed full-circle view coverage, ScanBundle export, and validation
on the Mac.

## Start the coverage backend on the Mac

Make sure the Mac and iPhone are on the same local network, then run:

```bash
cd backend
PYTHONPATH=src python3 -m fosmo_coverage.server --host 0.0.0.0 --port 8765
```

Verify it locally with `curl http://127.0.0.1:8765/health`. Find the Mac's LAN
address with `ipconfig getifaddr en0`; the app URL is
`http://<that-address>:8765`.

## On the iPhone

1. Build and install the `FosmoCapture` scheme on a physical iPhone.
2. If iOS blocks the first launch, trust the development profile for the
   installed app in Settings, then open FosmoCapture again.
3. Allow camera and local-network access.
4. Hold the phone upright and wait until the status says `跟踪正常`.
5. Enter the Mac backend URL and tap `连接后端并开始`.
6. Turn slowly through one complete 360-degree loop. The ring reports only
   view sectors for which the backend has actually received and accepted JPEG
   evidence. Local keyframe rejections do not produce speculative guidance.
7. Wait for `后端已确认完整一圈`; only then is `结束并导出` enabled.
8. Tap `分享 ScanBundle` and send the `.scanbundle` item to the Mac using
   AirDrop or Save to Files.

## On the Mac

Run the validator against the received directory:

```bash
cd backend
PYTHONPATH=src python3 -m fosmo_scanbundle.cli /path/to/<scan-id>.scanbundle --json
```

The command must exit with code `0` and report schema version `1.0`. The frame
count varies with motion and image quality; it is no longer capped at five. If
validation fails, preserve the exported bundle: its error report is the input
for the next capture/contract fix.

The coverage backend intentionally confirms horizontal viewing-direction
coverage, not whether a semantic model has recognized every physical wall.
Plane/point-cloud confirmation belongs to a later reconstruction milestone.
