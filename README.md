# kv-tiering

Importance-aware mixed-precision KV-cache compression. Each cached token
gets a precision tier per KV head: full precision, 4-bit, 2-bit, or evicted.
Tiers are allocated from prompt attention under a fixed memory budget. The
method is compared against StreamingLLM, H2O and uniform quantization on
Qwen2.5-1.5B-Instruct.

NYU CSCI-GA 3033-091 (Fall 2026) final project.

> **Status (Nov 21):** verification tests pass on Qwen2.5-1.5B (A100).
> Importance scoring, baselines (StreamingLLM, H2O, SnapKV-like, uniform
> int4/int2), the tiered allocator and the needle harness are implemented.
> Next: first curves, rate–distortion plots, ablations, demo.

**What is measured vs. accounted.** Quantization is *simulated*
(quantize → dequantize, tensors stay bf16), so its memory savings are
computed, not measured. Eviction physically removes tokens, so its savings are
real. No speed or latency claims are made.

## Layout

```
src/kvtier/     cache access, quantizer, tier plans, memory accounting
tests/          verification tests (run before any reported result)
scripts/        HPC setup, Slurm jobs, smoke run
docs/           architecture & design decisions, TA feedback
experiments/    needle sweep + summary (bootstrap CIs), YAML configs
results/        sweep outputs (JSONL) and summaries
demo/           (next) side-by-side demo
```

## Quick start (any machine, CPU, ~1 min)

The tests default to a tiny randomly initialised Qwen2 model, so they need
no GPU and no download.

```bash
git clone https://github.com/ok2287-nyu/kv-tiering.git
cd kv-tiering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q          # expect: 49 passed
```

## NYU HPC Cloud Bursting

Course work runs on Cloud Bursting (not Torch), via a terminal from
https://ood.burst.hpc.nyu.edu/ (NYU VPN off campus). Jobs use account
`csci_ga_3033_91-2026fa` and partition `c12m85-a100-1` (1× A100 40 GB);
the course allocation is 300 GPU hours. If the system Python is older than
3.10, `setup_hpc.sh` installs a standalone Python 3.11 on /scratch with `uv`.

```bash
# once
git clone https://github.com/ok2287-nyu/kv-tiering.git
cd kv-tiering
bash scripts/setup_hpc.sh        # venv on /scratch, pinned deps, model download

# each time
git pull
sbatch scripts/verify.sbatch     # tests on tiny model, tests on real model (fp32), smoke run (bf16)
tail -f logs/verify-<jobid>.out
```

Expected at the end of the log: two `passed` lines and `smoke OK`.

Needle-in-a-haystack sweep (rerun `setup_hpc.sh` once first: it now also
caches WikiText-103 for the haystack):

```bash
sbatch scripts/needle.sbatch experiments/configs/needle_quick.yaml   # ~10 trials, minutes
sbatch scripts/needle.sbatch experiments/configs/needle_main.yaml    # 135 trials
```

The log ends with the summary: accuracy per method × memory budget with
95% Wilson intervals, paired comparisons of tiered vs each baseline (paired
bootstrap CI of the difference, exact McNemar test), and accuracy by length
and depth. It's also written to `results/<name>.summary.md`; figures go to
`results/figures/` (rate–distortion curve, depth heatmaps). Raw rows, each
tagged with Slurm job id and git commit, go to `results/*.jsonl`.

Perplexity and ablations use the generic sweep script (same auto-resubmit,
results lock and resume):

```bash
sbatch scripts/sweep.sbatch perplexity experiments/configs/ppl_main.yaml        # 60 texts, no question at the end
sbatch scripts/sweep.sbatch needle     experiments/configs/needle_ablation.yaml # tiered variants + key-vs-value bits
```

Perplexity: each trial compresses the cache of a WikiText-103 prefix, then
scores the next 256 tokens through it (every token, including the first,
depends on the compressed cache). The summary reports ΔNLL vs the full cache,
paired per text. Ablation variants are tagged methods in the config
(`tag:`); the summary compares each with default tiered on the same prompts.

Analysis tools (no GPU needed unless noted):

```bash
python experiments/clean_results.py --config <cfg>          # audit: corrupt/self-contradictory rows, completeness
python experiments/compare_runs.py <a.jsonl> <b.jsonl>        # row-by-row agreement between two runs
python experiments/diagnose_scores.py --config <cfg> --lengths 8192 --depths 0.1 0.5 0.9   # GPU: why scores keep/lose the needle
```

Long sweeps resubmit themselves: 5 minutes before the 3-hour limit the job
stops the sweep, submits a continuation job (same config), and exits. The
continuation resumes from the results file. This happens at most 4 times
(`KVT_MAX_RESUBMITS`), and never after a crash. Follow the chain with
`squeue -u $USER` and the newest `logs/needle-*.out`.

Configuration lives in environment variables (see `.env.example`): `KVT_MODEL`,
`HF_HOME`, `KVT_REAL_MODEL`. No secrets are needed.

## Versions

`torch==2.9.1`, `transformers==5.23.0` (pinned). The cache API changes
between transformers releases, and `src/kvtier/cache.py` is written against
5.23 exactly. See `docs/architecture.md` for why positions are passed
explicitly after eviction.
