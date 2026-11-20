#!/usr/bin/env bash
# One-time environment setup on the NYU Torch cluster (run on a login node).
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
"$PY" - <<'EOF'
import sys
assert sys.version_info >= (3, 10), f"need Python >= 3.10, found {sys.version}. Load a newer Python module first."
print("python", sys.version.split()[0], "OK")
EOF

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
