#!/usr/bin/env bash
# Run the verification suite on the real model on one GPU.
#
#   sbatch scripts/verify.sbatch
#   tail -f logs/verify-<jobid>.out
#
#SBATCH --job-name=kvt-verify
#SBATCH --account=torch_pr_1071_general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify-%j.out
set -euo pipefail

SCRATCH_DIR="${SCRATCH:-/scratch/$USER}"
VENV="${KVT_VENV:-$SCRATCH_DIR/kvt-venv}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf_cache}"
export HF_HUB_OFFLINE=1  # model was pre-downloaded by setup_hpc.sh
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:-.}"

echo "== node $(hostname) | job ${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda available', torch.cuda.is_available())"

echo "== 1/3 tests on tiny random model (CPU logic)"
KVT_REAL_MODEL=0 python -m pytest -q tests

echo "== 2/3 tests on real model (float32, GPU)"
KVT_REAL_MODEL=1 python -m pytest -q tests

echo "== 3/3 smoke run on real model (bf16)"
python scripts/smoke.py
