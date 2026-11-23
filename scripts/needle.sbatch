#!/usr/bin/env bash
# Needle-in-a-haystack sweep on one A100, then the summary table.
#
#   sbatch scripts/needle.sbatch experiments/configs/needle_quick.yaml
#   tail -f logs/needle-<jobid>.out
#
# Auto-resubmit: 5 minutes before the time limit Slurm sends USR1. The script
# stops the sweep, waits for it to exit (so two jobs never write the same
# results file), and submits itself again with the same config. The new job
# resumes from the results file. It only resubmits on timeout (a crash fails
# normally) and at most KVT_MAX_RESUBMITS times (default 4).
#SBATCH --job-name=kvt-needle
#SBATCH --account=csci_ga_3033_91-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/needle-%j.out
set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/needle.sbatch <config.yaml>}"
RESUBMIT_COUNT="${KVT_RESUBMIT_COUNT:-0}"
MAX_RESUBMITS="${KVT_MAX_RESUBMITS:-4}"
SCRATCH_DIR="${SCRATCH:-/scratch/$USER}"
VENV="${KVT_VENV:-$SCRATCH_DIR/kvt-venv}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1  # model + dataset pre-downloaded by setup_hpc.sh
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:-.}"

echo "== node $(hostname) | job ${SLURM_JOB_ID:-local} | config $CONFIG | commit $(git rev-parse --short HEAD) | resubmit #$RESUBMIT_COUNT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SWEEP_PID=""
on_timeout() {
  echo "== time limit approaching: stopping sweep (pid $SWEEP_PID)"
  kill -TERM "$SWEEP_PID" 2>/dev/null || true
  wait "$SWEEP_PID" 2>/dev/null || true
  if [ "$RESUBMIT_COUNT" -ge "$MAX_RESUBMITS" ]; then
    echo "== reached KVT_MAX_RESUBMITS=$MAX_RESUBMITS; not resubmitting. Resubmit by hand to continue."
    exit 1
  fi
  NEXT=$((RESUBMIT_COUNT + 1))
  NEW_JOB=$(sbatch --parsable --export=ALL,KVT_RESUBMIT_COUNT="$NEXT" scripts/needle.sbatch "$CONFIG")
  echo "== resubmitted as job $NEW_JOB (resubmit #$NEXT); it resumes from the results file"
  exit 0
}
trap on_timeout USR1

OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CONFIG")

# Run in the background and `wait`, so bash can handle USR1 while the sweep
# runs (a foreground child would block the trap until it finished).
python experiments/needle.py --config "$CONFIG" &
SWEEP_PID=$!
wait "$SWEEP_PID"
trap - USR1

python experiments/summarize.py "$OUT"
