#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOSS_VENV="${PROJECT_ROOT}/.venv-moss"
cd "$PROJECT_ROOT"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate the CAST conda environment before running this script." >&2
  exit 2
fi

# Avoid loading Ubuntu's older C++/FFmpeg libraries ahead of the conda build.
install -d "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
install -m 0644 configs/conda/activate.d/cast-libs.sh \
  "$CONDA_PREFIX/etc/conda/activate.d/cast-libs.sh"
install -m 0644 configs/conda/deactivate.d/cast-libs.sh \
  "$CONDA_PREFIX/etc/conda/deactivate.d/cast-libs.sh"
export CAST_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python -m pip install --upgrade "pip<27" "setuptools<82" wheel
python -m pip install \
  --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
  "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128"
python -m pip install -r configs/runtime_requirements.txt

# These exact source snapshots are included in third_party.  Their upstream
# dependency ranges target newer, mutually incompatible stacks, so install the
# package itself without letting pip replace the pinned runtime above.
python -m pip install --no-deps "nemo_toolkit==2.6.0rc0"
python -m pip install --no-deps "qwen-asr==0.0.6"

# MOSS was used with Transformers 5.16.1, while Qwen-ASR 0.0.6 requires
# Transformers 4.57.6.  A small venv isolates only that Python dependency and
# reuses PyTorch/CUDA and the remaining packages from the conda environment.
mkdir -p "$(dirname "${MOSS_VENV}")"
rm -rf "${MOSS_VENV}"
python -m venv --system-site-packages "${MOSS_VENV}"
"${MOSS_VENV}/bin/python" -m pip install --upgrade "pip<27"
"${MOSS_VENV}/bin/python" -m pip install "transformers==5.16.1"
"${MOSS_VENV}/bin/python" -m pip install --no-deps -e third_party/MOSS-Transcribe-Diarize
python - <<'PY'
import importlib.util
import sys

required = [
    "torch",
    "torchaudio",
    "transformers",
    "qwen_asr",
    "modelscope",
    "nemo",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"runtime installation incomplete: {missing}")
print(f"runtime import check passed with Python {sys.version.split()[0]}")
PY

python - <<'PY'
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "third_party/FireRedASR2S"))
sys.path.insert(0, str(root / "third_party/3D-Speaker"))

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from nemo.collections.asr.models import SortformerEncLabelModel
from qwen_asr import Qwen3ASRModel
from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2

print("ASR, Sortformer, and speaker-encoder import checks passed")
PY

"${MOSS_VENV}/bin/python" - <<'PY'
import torch
import transformers
from moss_transcribe_diarize import parse_transcript

assert transformers.__version__ == "5.16.1", transformers.__version__
print(f"MOSS runtime import check passed with Transformers {transformers.__version__}")
PY
