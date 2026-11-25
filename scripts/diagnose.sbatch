#!/usr/bin/env bash
# Score diagnostic on one A100: needle rank under window vs cumulative scores,
# where H2O's heavy hitters sit. Extra arguments go to diagnose_scores.py.
#
#   sbatch scripts/diagnose.sbatch --lengths 8192 --depths 0.1 0.5 0.9 --budget 0.2
#
#SBATCH --job-name=kvt-diagnose
#SBATCH --account=csci_ga_3033_91-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diagnose-%j.out
set -euo pipefail
SCRATCH_DIR="${SCRATCH:-/scratch/$USER}"
VENV="${KVT_VENV:-$SCRATCH_DIR/kvt-venv}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:-.}"
echo "== node $(hostname) | job ${SLURM_JOB_ID:-local} | commit $(git rev-parse --short HEAD)"
python experiments/diagnose_scores.py --config experiments/configs/needle_main.yaml "$@"
