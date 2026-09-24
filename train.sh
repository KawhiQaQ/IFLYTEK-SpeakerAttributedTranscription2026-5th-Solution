#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

for argument in "$@"; do
  if [[ "${argument}" == "--dry-run" ]]; then
    "${PYTHON_BIN}" scripts/train_models.py . "$@"
    exit 0
  fi
done

"${PYTHON_BIN}" scripts/prepare_official_data.py .
if [[ "${CAST_SKIP_PUBLIC_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/download_public_models.py --root . --skip-existing
fi
"${PYTHON_BIN}" scripts/prepare_voxconverse_fixed.py .
"${PYTHON_BIN}" scripts/train_models.py . --overwrite "$@"

echo "Training completed. Run 'bash test.sh --allow-retrained-checkpoints' to infer with the new weights."
