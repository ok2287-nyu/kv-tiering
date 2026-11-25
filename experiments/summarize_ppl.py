"""Summarise a perplexity sweep.

    python experiments/summarize_ppl.py results/ppl_main.jsonl

For each method x budget: the mean increase in NLL over the full cache,
paired per trial (ΔNLL, nats/token; 0 = no damage), with a paired-bootstrap
95% CI, and the corresponding perplexity ratio exp(ΔNLL). Reported for all
continuation tokens and for the first 32 (the most cache-sensitive). Then
tiered vs each baseline at the same budget: difference in ΔNLL, paired CI.
Writes ``<input>.summary.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import paired_bootstrap_diff  # noqa: E402

OURS = "tiered"


def load(path):
    seen = {}
    for line in open(path, errors="replace"):
        try:
            r = json.loads(line.strip("\x00 \n"))
        except json.JSONDecodeError:
            continue
        seen[(r["trial"], r["method"], r["budget"])] = r
    return list(seen.values())


def _key(k):
    m, b = k
    return (b is not None, b or 0.0, m)


def _fb(b):
    return "—" if b is None else f"{b:.0%}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args(argv)
    rows = load(args.path)
    full = {r["trial"]: r for r in rows if r["method"] == "full"}
    cells = defaultdict(dict)
    for r in rows:
        if r["trial"] in full:
            cells[(r["method"], r["budget"])][r["trial"]] = r

    out = [f"# Perplexity summary\n\nSource: `{args.path}`. ΔNLL = NLL − full-cache NLL on the same text "
           "(nats/token, paired per trial; 95% paired-bootstrap CI). PPL ratio = exp(ΔNLL).\n"]
    base = np.mean([r["nll"] for r in full.values()])
    out.append(f"Full-cache NLL {base:.3f} (PPL {math.exp(base):.2f}) over {len(full)} trials.\n")
    out.append("| method | budget | memory | n | ΔNLL (all) | 95% CI | PPL ratio | ΔNLL (first 32) | 95% CI |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(cells, key=_key):
        if key[0] == "full":
            continue
        ts = sorted(cells[key])
        d_all = paired_bootstrap_diff(np.array([cells[key][t]["nll"] for t in ts]), np.array([full[t]["nll"] for t in ts]))
        d_head = paired_bootstrap_diff(np.array([cells[key][t]["nll_head"] for t in ts]), np.array([full[t]["nll_head"] for t in ts]))
        ratio = np.mean([cells[key][t]["ratio"] for t in ts])
        out.append(
            f"| {key[0]} | {_fb(key[1])} | {ratio:.1%} | {len(ts)} | {d_all[0]:+.3f} | [{d_all[1]:+.3f}, {d_all[2]:+.3f}] | "
            f"{math.exp(d_all[0]):.3f} | {d_head[0]:+.3f} | [{d_head[1]:+.3f}, {d_head[2]:+.3f}] |"
        )

    out.append("\n## Tiered vs each baseline at the same budget (ΔNLL difference; negative = tiered better)\n")
    out.append("| baseline | budget | n | tiered − baseline | 95% CI |")
    out.append("|---|---|---|---|---|")
    for key in sorted(cells, key=_key):
        m, b = key
        if m in (OURS, "full") or b is None or (OURS, b) not in cells:
            continue
        ts = sorted(set(cells[key]) & set(cells[(OURS, b)]))
        d = paired_bootstrap_diff(np.array([cells[(OURS, b)][t]["nll"] for t in ts]), np.array([cells[key][t]["nll"] for t in ts]))
        out.append(f"| {m} | {_fb(b)} | {len(ts)} | {d[0]:+.3f} | [{d[1]:+.3f}, {d[2]:+.3f}] |")

    lengths = sorted({r["length"] for r in rows})
    out.append("\n## ΔNLL by context length\n")
    out.append("| method | budget | " + " | ".join(str(L) for L in lengths) + " |")
    out.append("|---|---|" + "---|" * len(lengths))
    for key in sorted(cells, key=_key):
        if key[0] == "full":
            continue
        vals = []
        for L in lengths:
            ds = [cells[key][t]["nll"] - full[t]["nll"] for t in cells[key] if full[t]["length"] == L]
            vals.append(f"{np.mean(ds):+.3f}" if ds else "—")
        out.append(f"| {key[0]} | {_fb(key[1])} | " + " | ".join(vals) + " |")
    md = "\n".join(out) + "\n"
    print(md)
    p = args.path.rsplit(".", 1)[0] + ".summary.md"
    Path(p).write_text(md)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
