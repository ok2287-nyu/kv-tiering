# kv-tiering

Importance-aware mixed-precision KV-cache compression. Each cached token
gets a precision tier per KV head: full precision, 4-bit, 2-bit, or evicted.
Tiers are allocated from prompt attention under a fixed memory budget. The
method is compared against StreamingLLM, H2O and uniform quantization on
Qwen2.5-1.5B-Instruct.

NYU CSCI-GA 3033-091 (Fall 2026) final project.

> **Status (Nov 20):** cache instrumentation, simulated quantizer, memory
> accounting, tier application and the four verification tests are done.
> Importance scoring, baselines, the needle harness and the demo are next.

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
experiments/    (next) needle harness, perplexity, sweep configs
results/        (next) metrics and figures
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
python -m pytest -q          # expect: 26 passed
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

Configuration lives in environment variables (see `.env.example`): `KVT_MODEL`,
`HF_HOME`, `KVT_REAL_MODEL`. No secrets are needed.

## Versions

`torch==2.9.1`, `transformers==5.23.0` (pinned). The cache API changes
between transformers releases, and `src/kvtier/cache.py` is written against
5.23 exactly. See `docs/architecture.md` for why positions are passed
explicitly after eviction.
