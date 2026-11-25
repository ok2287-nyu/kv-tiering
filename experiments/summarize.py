"""Summarise needle results.

    python experiments/summarize.py results/needle_main.jsonl

Prints, and writes to ``<input>.summary.md`` (plus a CSV of the main table):

1. **Accuracy per method × memory budget** with 95% Wilson intervals
   (see stats.py for why not a bootstrap of one proportion).
2. **Paired comparison: tiered vs each baseline at the same budget.** Same
   prompts, so: accuracy difference with a paired-bootstrap 95% CI, the
   discordant counts (only-tiered-right / only-baseline-right) and an exact
   McNemar p-value. Uniform int2/int4 are compared with tiered at the largest
   budget not exceeding their memory, so tiered never gets extra memory.
3. Accuracy by context length and by needle depth.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import mcnemar_exact, paired_bootstrap_diff, wilson  # noqa: E402

OURS = "tiered"


def load_rows(path: str) -> list[dict]:
    """Read JSONL rows, skipping unparseable lines and de-duplicating.

    Keep one row per (trial, method, budget): the last one. (For files damaged
    by concurrent writers, audit with clean_results.py first; de-duplication
    cannot detect spliced rows.)
    """
    seen: dict[tuple, dict] = {}
    with open(path, errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line.strip("\x00 \n"))
            except json.JSONDecodeError:
                continue
            seen[(r["trial"], r["method"], r["budget"])] = r
    return list(seen.values())


def _cell_key(k):
    m, b = k
    return (b is not None, b if b is not None else 0.0, m)


def _fmt_b(b):
    return "—" if b is None else f"{b:.0%}"


def summarize(rows: list[dict]) -> dict:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["method"], r["budget"])].append(r)

    main = []
    for key in sorted(cells, key=_cell_key):
        rs = cells[key]
        k = sum(int(r["correct"]) for r in rs)
        lo, hi = wilson(k, len(rs))
        main.append(
            {
                "method": key[0],
                "budget": key[1],
                "ratio": float(np.mean([r["ratio"] for r in rs])),
                "n": len(rs),
                "correct": k,
                "accuracy": k / len(rs),
                "wilson_low": lo,
                "wilson_high": hi,
                "needle_kept": float(np.mean([r["needle_kept_frac"] for r in rs])),
            }
        )

    by_trial = {key: {r["trial"]: bool(r["correct"]) for r in rs} for key, rs in cells.items()}
    ours_budgets = sorted(b for (m, b) in cells if m == OURS)
    paired = []
    for key in sorted(cells, key=_cell_key):
        m, b = key
        if m == OURS or m == "full":
            continue
        if b is not None:
            ob = b if b in ours_budgets else None
        else:  # unbudgeted (uniform): tiered at the largest budget <= its memory ratio
            ratio = float(np.mean([r["ratio"] for r in cells[key]]))
            cands = [x for x in ours_budgets if x <= ratio + 1e-9]
            ob = max(cands) if cands else None
        if ob is None:
            continue
        ours, theirs = by_trial[(OURS, ob)], by_trial[key]
        common = sorted(set(ours) & set(theirs))
        a = np.array([ours[t] for t in common])
        c = np.array([theirs[t] for t in common])
        diff, dlo, dhi = paired_bootstrap_diff(a, c)
        n10, n01, p = mcnemar_exact(a, c)
        paired.append(
            {
                "baseline": m, "baseline_budget": b, "tiered_budget": ob, "n": len(common),
                "diff": diff, "diff_low": dlo, "diff_high": dhi,
                "only_tiered": n10, "only_baseline": n01, "p": p,
            }
        )

    def grouped(field):
        g = defaultdict(list)
        for r in rows:
            g[((r["method"], r["budget"]), r[field])].append(int(r["correct"]))
        return g

    return {"main": main, "paired": paired, "by_length": grouped("length"), "by_depth": grouped("depth"), "cells": cells}


def to_markdown(s: dict, source: str) -> str:
    out = [f"# Needle-in-a-haystack summary\n\nSource: `{source}`\n"]
    out.append("## Accuracy by method and memory budget (95% Wilson interval)\n")
    out.append("| method | budget | accounted memory | n | accuracy | 95% CI | needle kept |")
    out.append("|---|---|---|---|---|---|---|")
    for r in s["main"]:
        out.append(
            f"| {r['method']} | {_fmt_b(r['budget'])} | {r['ratio']:.1%} | {r['n']} | {r['accuracy']:.2f} | "
            f"[{r['wilson_low']:.2f}, {r['wilson_high']:.2f}] | {r['needle_kept']:.2f} |"
        )
    out.append("\n## Tiered vs each baseline, paired on the same trials\n")
    out.append(
        "Difference = tiered accuracy − baseline accuracy (paired bootstrap 95% CI). "
        "Discordant = trials only tiered / only the baseline answered. p = exact McNemar. "
        "Uniform baselines are compared with tiered at the largest budget not exceeding their memory.\n"
    )
    out.append("| baseline | baseline memory | tiered budget | n | difference | 95% CI | only tiered | only baseline | p |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for p in s["paired"]:
        bm = _fmt_b(p["baseline_budget"]) if p["baseline_budget"] is not None else "fixed"
        pv = "<1e-4" if p["p"] < 1e-4 else f"{p['p']:.3g}"
        out.append(
            f"| {p['baseline']} | {bm} | {p['tiered_budget']:.0%} | {p['n']} | {p['diff']:+.2f} | "
            f"[{p['diff_low']:+.2f}, {p['diff_high']:+.2f}] | {p['only_tiered']} | {p['only_baseline']} | {pv} |"
        )
    for title, key in (("context length (tokens)", "by_length"), ("needle depth", "by_depth")):
        vals = sorted({v for (_, v) in s[key]})
        out.append(f"\n## Accuracy by {title}\n")
        hdr = " | ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in vals)
        out.append(f"| method | budget | {hdr} |")
        out.append("|---|---|" + "---|" * len(vals))
        for cell in sorted(s["cells"], key=_cell_key):
            accs = []
            for v in vals:
                xs = s[key].get((cell, v), [])
                accs.append(f"{np.mean(xs):.2f}" if xs else "—")
            out.append(f"| {cell[0]} | {_fmt_b(cell[1])} | " + " | ".join(accs) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args(argv)
    rows = load_rows(args.path)
    s = summarize(rows)
    md = to_markdown(s, args.path)
    print(md)
    stem = args.path.rsplit(".", 1)[0]
    Path(stem + ".summary.md").write_text(md)
    with open(stem + ".summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s["main"][0].keys()))
        w.writeheader()
        w.writerows(s["main"])
    print(f"wrote {stem}.summary.md and {stem}.summary.csv")


if __name__ == "__main__":
    main()
