#!/usr/bin/env bash
# Needle-in-a-haystack sweep on one A100, then the summary table.
#
#   sbatch scripts/needle.sbatch experiments/configs/needle_quick.yaml
#   tail -f logs/needle-<jobid>.out
#
# Resubmitting with the same config resumes where it stopped.
#SBATCH --job-name=kvt-needle
#SBATCH --account=csci_ga_3033_91-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/needle-%j.out
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/needle.sbatch <config.yaml>}"
SCRATCH_DIR="${SCRATCH:-/scratch/$USER}"
VENV="${KVT_VENV:-$SCRATCH_DIR/kvt-venv}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1  # model + dataset pre-downloaded by setup_hpc.sh
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:-.}"

echo "== node $(hostname) | job ${SLURM_JOB_ID:-local} | config $CONFIG | commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CONFIG")
python experiments/needle.py --config "$CONFIG"
python experiments/summarize.py "$OUT"
