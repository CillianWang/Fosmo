# Manhattan-space room extraction

The Manhattan stage converts a noisy fused RGB-depth surface into a deliberately
minimal architectural visualization. Gravity already defines world Y through
ARKit. The extractor then:

1. finds the strongest plausible horizontal floor/ceiling plane pair;
2. estimates the dominant wall-normal angle modulo 90 degrees;
3. clusters vertical surface evidence along both orthogonal axes;
4. selects two well-supported, separated boundary planes per axis; and
5. exports a horizontal floor and four exactly vertical wall panels.

Run it after full-frame or fast fusion:

```bash
PYTHONPATH=backend/src backend/.venv-reconstruction/bin/python \
  -m fosmo_reconstruction.manhattan_cli \
  /path/to/fused-output \
  --output /path/to/manhattan-output
```

Outputs:

- `room_manhattan_mesh.ply` — binary colored mesh for the native iPhone viewer.
- `room_manhattan_mesh.glb` — portable copy for other 3D tools.
- `room_manhattan_preview.png` — static 3D and top-view QA rendering.
- `manhattan_report.json` — plane evidence, dimensions, height, and limitations.
- `preview_manifest.json` — tells the lightweight backend which mesh to serve.

The result is intentionally conservative about semantics. It represents the
strongest supported orthogonal main-room envelope. It does not currently infer
doors, windows, wall thickness, or every recess. The detected ceiling plane is
used to set wall height but the ceiling mesh is omitted, leaving a readable
dollhouse view on iPhone.
