#!/usr/bin/env bash
# One-time environment setup on NYU HPC Cloud Bursting (run from an OOD
# terminal; course work must not use the Torch cluster).
#
#   bash scripts/setup_hpc.sh
#
# Creates a venv on /scratch (home quota is small), installs the pinned
# requirements, and pre-downloads the model so GPU jobs don't spend their
# time allocation downloading. If any step fails, paste the full output back.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${SCRATCH:-/scratch/$USER}"
VENV="${KVT_VENV:-$SCRATCH_DIR/kvt-venv}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf_cache}"
MODEL="${KVT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

echo "repo:    $REPO_DIR"
echo "venv:    $VENV"
echo "HF_HOME: $HF_HOME"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  # System Python is too old (transformers 5.x needs >= 3.10). Instead of
  # depending on cluster-specific modules, fetch a standalone Python 3.11
  # with uv and keep it on scratch.
  echo "$("$PY" --version 2>&1) is too old; installing Python 3.11 via uv"
  export UV_PYTHON_INSTALL_DIR="$SCRATCH_DIR/uv-python"
  export UV_CACHE_DIR="$SCRATCH_DIR/uv-cache"
  if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    "$PY" -m pip install --user --quiet uv
  fi
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  "$UV" python install 3.11
  PY="$("$UV" python find 3.11)"
fi
"$PY" -c 'import sys; print("python", sys.version.split()[0], "OK:", sys.executable)'

if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements.txt"
pip install -e "$REPO_DIR"

mkdir -p "$HF_HOME"
python - <<EOF
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL")
print("model cached at", p)
EOF

python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| CUDA build', torch.version.cuda)"
echo
echo "Setup done. Next: sbatch scripts/verify.sbatch"
