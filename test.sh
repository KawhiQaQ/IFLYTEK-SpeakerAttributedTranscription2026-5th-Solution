#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

for argument in "$@"; do
  if [[ "${argument}" == "--dry-run" ]]; then
    "${PYTHON_BIN}" scripts/run_inference.py . --skip-preflight "$@"
    exit 0
  fi
done

"${PYTHON_BIN}" scripts/prepare_official_data.py .
"${PYTHON_BIN}" scripts/normalize_checkpoint_layout.py .
if [[ "${CAST_SKIP_PUBLIC_MODEL_DOWNLOAD:-0}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/download_public_models.py --root . --skip-existing
fi

"${PYTHON_BIN}" scripts/run_inference.py . "$@"

result_path="${PROJECT_ROOT}/submissions/final_solution.seglst.json"
if [[ ! -f "${result_path}" ]]; then
  echo "Inference finished without producing ${result_path}" >&2
  exit 1
fi
echo "Prediction written to ${result_path}"
