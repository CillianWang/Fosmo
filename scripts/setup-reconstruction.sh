#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIRECTORY}/.." && pwd)"
PYTHON_BINARY="${FOSMO_RECONSTRUCTION_PYTHON:-/usr/bin/python3}"
VIRTUAL_ENVIRONMENT="${REPOSITORY_ROOT}/backend/.venv-reconstruction"
DEPTH_PRO_SOURCE="${REPOSITORY_ROOT}/.vendor/ml-depth-pro"
DEPTH_PRO_COMMIT="9efe5c1def37a26c5367a71df664b18e1306c708"
CHECKPOINT="${DEPTH_PRO_SOURCE}/checkpoints/depth_pro.pt"
CHECKPOINT_SHA256="3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"

"${PYTHON_BINARY}" - <<'PY'
import sys
if not ((3, 9) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit("Fosmo reconstruction requires Python 3.9 through 3.12")
PY

"${PYTHON_BINARY}" -m venv "${VIRTUAL_ENVIRONMENT}"
"${VIRTUAL_ENVIRONMENT}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VIRTUAL_ENVIRONMENT}/bin/python" -m pip install \
    -r "${REPOSITORY_ROOT}/backend/requirements-reconstruction.txt"

if [[ ! -d "${DEPTH_PRO_SOURCE}/.git" ]]; then
    mkdir -p "${REPOSITORY_ROOT}/.vendor"
    git clone git@github.com:apple/ml-depth-pro.git "${DEPTH_PRO_SOURCE}"
fi
git -C "${DEPTH_PRO_SOURCE}" fetch origin "${DEPTH_PRO_COMMIT}"
git -C "${DEPTH_PRO_SOURCE}" checkout --detach "${DEPTH_PRO_COMMIT}"
"${VIRTUAL_ENVIRONMENT}/bin/python" -m pip install --no-deps -e "${DEPTH_PRO_SOURCE}"

mkdir -p "$(dirname "${CHECKPOINT}")"
if [[ ! -f "${CHECKPOINT}" ]] || \
   [[ "$(shasum -a 256 "${CHECKPOINT}" | awk '{print $1}')" != "${CHECKPOINT_SHA256}" ]]; then
    TEMPORARY_CHECKPOINT="${CHECKPOINT}.download"
    curl -L --fail --show-error --retry 3 \
        --output "${TEMPORARY_CHECKPOINT}" \
        https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
    printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${TEMPORARY_CHECKPOINT}" | shasum -a 256 -c -
    mv "${TEMPORARY_CHECKPOINT}" "${CHECKPOINT}"
fi

"${VIRTUAL_ENVIRONMENT}/bin/python" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"mps_available={torch.backends.mps.is_available()}")
PY
