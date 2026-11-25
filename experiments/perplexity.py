"""Perplexity sweep: lengths x repetitions, every method (see src/kvtier/perplexity.py).

    python experiments/perplexity.py --config experiments/configs/ppl_main.yaml

Each trial takes a window of ``L`` tokens of WikiText-103 at a seeded random
offset: the first ``L - cont_len`` are the prefix (compressed), the last
``cont_len`` are scored. Same resumable JSONL, results lock and provenance as
the needle sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle import _provenance, done_trials, results_lock  # noqa: E402

from kvtier.model import load_model  # noqa: E402
from kvtier.needle import expand_methods, load_haystack_text  # noqa: E402
from kvtier.perplexity import run_ppl_trial  # noqa: E402
from kvtier.tiers import QuantConfig  # noqa: E402


def trial_id(length: int, rep: int) -> str:
    return f"L{length}-r{rep}"


def main(argv=None, model=None, tok=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    out_path = args.out or cfg["out"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with results_lock(out_path):
        _sweep(args, cfg, out_path, model, tok)


def _sweep(args, cfg, out_path, model, tok):
    prov = _provenance()
    if model is None:
        model, tok = load_model(cfg.get("model"))
    specs = expand_methods(cfg["methods"], cfg["budgets"])
    qcfg = QuantConfig(**cfg.get("quant", {}))
    window = cfg.get("window", 64)
    cont_len = cfg.get("cont_len", 256)
    t0 = time.time()
    text = load_haystack_text(cfg.get("haystack", "wikitext"), seed=cfg.get("seed", 0))
    corpus = tok(text, add_special_tokens=False).input_ids
    print(f"corpus: {len(corpus)} tokens; {len(specs)} method/budget runs per trial", flush=True)

    skip = done_trials(out_path, len(specs))
    grid = [(L, r) for L in cfg["lengths"] for r in range(cfg["reps"])]
    todo = [g for g in grid if trial_id(*g) not in skip]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(grid)} trials in grid, {len(skip)} already done, running {len(todo)}", flush=True)

    with open(out_path, "a") as fout:
        for i, (L, r) in enumerate(todo):
            tid = trial_id(L, r)
            rng = random.Random(f"{cfg.get('seed', 0)}-{tid}")
            off = rng.randint(0, len(corpus) - L)
            ids = torch.tensor([corpus[off : off + L]])
            ts = time.time()
            rows = run_ppl_trial(model, ids, cont_len, specs, window, qcfg)
            for row in rows:
                row.update(trial=tid, length=L, rep=r, offset=off, **prov)
            fout.write("".join(json.dumps(row) + "\n" for row in rows))
            fout.flush()
            os.fsync(fout.fileno())
            full = next((x["nll"] for x in rows if x["method"] == "full"), float("nan"))
            show = " ".join(f"{x['method']}@{x['budget']}={x['nll'] - full:+.3f}" for x in rows if x["method"] in ("tiered", "streaming"))
            print(f"[{i + 1}/{len(todo)}] {tid} ({time.time() - ts:.1f}s) full nll={full:.3f} | ΔNLL {show}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
