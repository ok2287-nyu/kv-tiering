"""Compare two needle result files row by row.

    python experiments/compare_runs.py results/needle_main.damaged-2026-11-23.jsonl results/needle_main.jsonl --seed 0

For every (trial, method, budget) present in both files, it checks whether the
two runs gave the same answer. Use cases:

* **Reproducibility across GPUs**: two clean runs of the same config. Greedy
  decoding with fixed seeds should agree except for rare flips where bf16
  kernels break a near-tie differently.
* **Damage assessment**: a damaged file vs a clean re-run. Rows that fail
  the consistency check (see clean_results.row_problems) are counted
  separately, and so is how many *consistent-looking* rows still disagree with
  the clean run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_results import row_problems  # noqa: E402


def read(path: str, seed: int):
    rows: dict[tuple, list[dict]] = defaultdict(list)
    n_corrupt = n_bad = 0
    with open(path, errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line.strip("\x00 \n"))
                key = (r["trial"], r["method"], r["budget"])
            except (json.JSONDecodeError, KeyError, TypeError):
                n_corrupt += 1
                continue
            if row_problems(r, seed):
                n_bad += 1
                continue
            rows[key].append(r)
    return rows, n_corrupt, n_bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seed", type=int, default=0, help="the sweep's config seed (for the needle check)")
    args = ap.parse_args(argv)

    A, ca, ba = read(args.a, args.seed)
    B, cb, bb = read(args.b, args.seed)
    print(f"A: {args.a}: {len(A)} keys, {ca} corrupt lines, {ba} inconsistent rows")
    print(f"B: {args.b}: {len(B)} keys, {cb} corrupt lines, {bb} inconsistent rows")
    common = sorted(set(A) & set(B))
    print(f"keys in both: {len(common)}; only in A: {len(set(A) - set(B))}; only in B: {len(set(B) - set(A))}")

    same = 0
    flips_by_method: Counter = Counter()
    correctness_flips: Counter = Counter()
    examples = []
    for k in common:
        a, b = A[k][-1], B[k][-1]
        if a["answer"] == b["answer"]:
            same += 1
            continue
        flips_by_method[k[1]] += 1
        if bool(a["correct"]) != bool(b["correct"]):
            correctness_flips[k[1]] += 1
        if len(examples) < 10:
            examples.append(f"{k}: A={a['answer']!r} ({a['correct']}) B={b['answer']!r} ({b['correct']})")
    n = len(common)
    print(f"\nidentical answers: {same}/{n} ({same / max(n, 1):.2%})")
    if flips_by_method:
        print("different answers by method:", dict(flips_by_method))
        print("of which correctness changed:", dict(correctness_flips))
        print("examples:")
        for e in examples:
            print("  ", e)


if __name__ == "__main__":
    main()
