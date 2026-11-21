"""Summarise needle results: accuracy per method x budget with 95% bootstrap CIs.

    python experiments/summarize.py results/needle_quick.jsonl

The CI is a percentile bootstrap over trials: resample the trials of a
(method, budget) cell with replacement, recompute accuracy, repeat
``--boot`` times, and take the 2.5th and 97.5th percentiles. Two methods
differ at a budget only where their intervals do not overlap. That's
conservative but easy to defend.

Writes ``<input>.summary.csv`` and prints a Markdown table, plus accuracy by
depth (the depth dimension is where eviction methods fail).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np


def bootstrap_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(args.path)]
    rng = np.random.default_rng(args.seed)
    cells = defaultdict(list)
    by_depth = defaultdict(list)
    for r in rows:
        key = (r["method"], r["budget"])
        cells[key].append(r)
        by_depth[(key, r["depth"])].append(int(r["correct"]))

    def sort_key(k):
        m, b = k
        return (b is not None, b if b is not None else 0, m)

    out = []
    for key in sorted(cells, key=sort_key):
        rs = cells[key]
        c = np.array([int(r["correct"]) for r in rs], dtype=float)
        lo, hi = bootstrap_ci(c, args.boot, rng)
        out.append(
            {
                "method": key[0],
                "budget": "" if key[1] is None else key[1],
                "ratio": float(np.mean([r["ratio"] for r in rs])),
                "n": len(rs),
                "accuracy": float(c.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "needle_kept": float(np.mean([r["needle_kept_frac"] for r in rs])),
            }
        )

    csv_path = args.path.rsplit(".", 1)[0] + ".summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print("| method | budget | accounted memory | n | accuracy | 95% CI | needle kept |")
    print("|---|---|---|---|---|---|---|")
    for o in out:
        b = "—" if o["budget"] == "" else f"{o['budget']:.2f}"
        print(
            f"| {o['method']} | {b} | {o['ratio']:.1%} | {o['n']} | {o['accuracy']:.2f} | "
            f"[{o['ci_low']:.2f}, {o['ci_high']:.2f}] | {o['needle_kept']:.2f} |"
        )

    depths = sorted({d for (_, d) in by_depth})
    print("\nAccuracy by needle depth:")
    print("| method | budget | " + " | ".join(f"{d:.1f}" for d in depths) + " |")
    print("|---|---|" + "---|" * len(depths))
    for key in sorted(cells, key=sort_key):
        b = "—" if key[1] is None else f"{key[1]:.2f}"
        accs = [np.mean(by_depth[(key, d)]) if by_depth[(key, d)] else float("nan") for d in depths]
        print(f"| {key[0]} | {b} | " + " | ".join(f"{a:.2f}" for a in accs) + " |")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
