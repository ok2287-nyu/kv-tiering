#!/usr/bin/env bash
# Generic sweep on one A100: needle or perplexity, then summary (+ plots).
#
#   sbatch scripts/sweep.sbatch needle     experiments/configs/needle_ablation.yaml
#   sbatch scripts/sweep.sbatch perplexity experiments/configs/ppl_main.yaml
#
# Same auto-resubmit as needle.sbatch: USR1 five minutes before the limit ->
# stop the sweep, wait for it (releasing the results lock), resubmit with the
# same arguments; resume picks up from the results file. Never after a crash;
# at most KVT_MAX_RESUBMITS (default 4) times.
#SBATCH --job-name=kvt-sweep
#SBATCH --account=csci_ga_3033_91-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/sweep-%j.out
set -euo pipefail

EXP="${1:?usage: sbatch scripts/sweep.sbatch <needle|perplexity> <config.yaml>}"
CONFIG="${2:?usage: sbatch scripts/sweep.sbatch <needle|perplexity> <config.yaml>}"
case "$EXP" in needle|perplexity) ;; *) echo "unknown experiment $EXP"; exit 2;; esac
RESUBMIT_COUNT="${KVT_RESUBMIT_COUNT:-0}"
MAX_RESUBMITS="${KVT_MAX_RESUBMITS:-4}"
SCRATCH_DIR="${SCRATCH:-/scratch/$USER}"
VENV="${KVT_VENV:-$SCRATCH_DIR/kvt-venv}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf_cache}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:-.}"

echo "== node $(hostname) | job ${SLURM_JOB_ID:-local} | $EXP $CONFIG | commit $(git rev-parse --short HEAD) | resubmit #$RESUBMIT_COUNT"
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
  NEW_JOB=$(sbatch --parsable --export=ALL,KVT_RESUBMIT_COUNT="$NEXT" scripts/sweep.sbatch "$EXP" "$CONFIG")
  echo "== resubmitted as job $NEW_JOB (resubmit #$NEXT); it resumes from the results file"
  exit 0
}
trap on_timeout USR1

OUT=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out'])" "$CONFIG")
python "experiments/$EXP.py" --config "$CONFIG" &
SWEEP_PID=$!
wait "$SWEEP_PID"
trap - USR1

if [ "$EXP" = needle ]; then
  python experiments/summarize.py "$OUT"
  python experiments/plot_needle.py "$OUT"
else
  python experiments/summarize_ppl.py "$OUT"
fi
