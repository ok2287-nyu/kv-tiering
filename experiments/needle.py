"""Needle-in-a-haystack sweep: lengths x depths x repetitions, every method.

    python experiments/needle.py --config experiments/configs/needle_quick.yaml

Writes one JSON line per (trial, method, budget) to the config's ``out`` path.
Re-running with the same config skips trials that are already complete, so an
interrupted job can simply be resubmitted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
import yaml

from kvtier.model import load_model
from kvtier.needle import build_prompt, expand_methods, load_haystack_text, make_needle, run_trial
from kvtier.tiers import QuantConfig


def eos_ids(model, tok) -> set[int]:
    ids = {tok.eos_token_id}
    g = getattr(model.generation_config, "eos_token_id", None)
    ids.update(g if isinstance(g, (list, tuple, set)) else [g])
    return {i for i in ids if i is not None}


def trial_id(length: int, depth: float, rep: int) -> str:
    return f"L{length}-d{depth:.2f}-r{rep}"


def done_trials(path: str, n_specs: int) -> set[str]:
    if not os.path.exists(path):
        return set()
    counts: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            tid = json.loads(line)["trial"]
            counts[tid] = counts.get(tid, 0) + 1
    return {t for t, c in counts.items() if c >= n_specs}


def main(argv=None, model=None, tok=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="override the config's output path")
    ap.add_argument("--limit", type=int, help="stop after this many trials (smoke testing)")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_path = args.out or cfg["out"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if model is None:
        model, tok = load_model(cfg.get("model"))
    specs = expand_methods(cfg["methods"], cfg["budgets"])
    qcfg = QuantConfig(**cfg.get("quant", {}))
    window = cfg.get("window", 64)
    eos = eos_ids(model, tok)

    t0 = time.time()
    text = load_haystack_text(cfg.get("haystack", "wikitext"), seed=cfg.get("seed", 0))
    hay_ids = tok(text, add_special_tokens=False).input_ids
    print(f"haystack: {len(hay_ids)} tokens ({time.time() - t0:.1f}s); {len(specs)} method/budget runs per trial", flush=True)

    skip = done_trials(out_path, len(specs))
    grid = [(L, d, r) for L in cfg["lengths"] for d in cfg["depths"] for r in range(cfg["reps"])]
    todo = [g for g in grid if trial_id(*g) not in skip]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(grid)} trials in grid, {len(skip)} already done, running {len(todo)}", flush=True)

    with open(out_path, "a") as fout:
        for i, (L, d, r) in enumerate(todo):
            tid = trial_id(L, d, r)
            rng = random.Random(f"{cfg.get('seed', 0)}-{tid}")
            needle = make_needle(rng)
            prompt = build_prompt(tok, hay_ids, L, d, needle, rng)
            ts = time.time()
            rows = run_trial(model, tok, prompt, specs, window, cfg.get("max_new_tokens", 16), qcfg, eos)
            for row in rows:
                row.update(trial=tid, length=L, depth=d, rep=r, needle=needle.value, prompt_tokens=prompt.length)
                fout.write(json.dumps(row) + "\n")
            fout.flush()
            acc = {}
            for row in rows:
                acc.setdefault(row["method"], []).append(int(row["correct"]))
            summary = " ".join(f"{m}={sum(v)}/{len(v)}" for m, v in acc.items())
            print(f"[{i + 1}/{len(todo)}] {tid} ({time.time() - ts:.1f}s) {summary}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
