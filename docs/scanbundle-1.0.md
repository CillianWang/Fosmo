# ScanBundle 1.0 contract

`ScanBundle 1.0` is the boundary between Fosmo's iOS capture app and the local
reconstruction service. The backend must reject an invalid bundle before depth
inference starts.

## Layout

```text
<scan-id>.scanbundle/
├── manifest.json
└── frames/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

## Frozen conventions

- Images are JPEGs in the camera sensor's native pixel orientation. They are
  not cropped, resized, mirrored, or rotated after ARKit reports intrinsics.
- Matrices are flattened in row-major order. Intrinsics contain 9 finite
  values; `world_from_camera` contains 16 finite values.
- Poses are ARKit `worldFromCamera` transforms in meters.
- `calibration_width` and `calibration_height` identify the pixel coordinate
  system used by `intrinsics`. `image_width` and `image_height` must match the
  actual JPEG dimensions.
- Version 1.0 contains keyframes only. Every saved frame therefore has normal
  ARKit tracking and a null tracking reason.
- Frame IDs are unique positive integers, filenames use six zero-padded digits,
  and timestamps are finite, non-negative, and strictly increasing.
- `created_at` is an ISO-8601 timestamp with a timezone and `scan_id` is a UUID.

The machine-readable source of truth is
[`contracts/scanbundle-1.0.schema.json`](../contracts/scanbundle-1.0.schema.json).

## Validation behavior

Validation is fail-closed. Unknown schema versions, unsafe paths, missing
images, malformed matrices, inconsistent dimensions, and invalid camera
transforms are errors. The CLI returns exit code `0` for a valid bundle and `1`
for an invalid bundle.
