"""Audit and repair a needle results file.

    python experiments/clean_results.py --config experiments/configs/needle_main.yaml           # report only
    python experiments/clean_results.py --config experiments/configs/needle_main.yaml --write   # repair

Why this exists: on Nov 23 two sweeps (jobs 40660 and 40667) appended to
``results/needle_main.jsonl`` at the same time. On a network filesystem
concurrent appends can overwrite each other, leaving unparseable lines,
duplicate rows and trials with rows missing. (``needle.py`` now takes a lock,
so this can't recur.)

The report counts:
  * corrupt lines: not valid JSON, or missing required fields;
  * duplicate rows: the same (trial, method, budget) more than once, and how
    many of those **disagree** on the answer. Trials are deterministic (fixed
    seeds, greedy decoding), so this should be 0. A nonzero count would mean
    the duplicates aren't trustworthy;
  * trials complete / incomplete / absent relative to the config's grid.

``--write`` keeps one row per (trial, method, budget) for complete trials only,
backs up the original to ``<file>.raw-<timestamp>``, and rewrites the file. Then
re-running the sweep with the same config re-runs just the dropped trials.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle import results_lock, trial_id  # noqa: E402

from kvtier.needle import expand_methods  # noqa: E402

REQUIRED = ("trial", "method", "budget", "correct", "ratio", "answer")


def audit(path: str, cfg: dict) -> dict:
    specs = expand_methods(cfg["methods"], cfg["budgets"])
    expected = {(s.label, s.budget) for s in specs}
    grid = [trial_id(L, d, r) for L in cfg["lengths"] for d in cfg["depths"] for r in range(cfg["reps"])]

    n_lines = n_corrupt = 0
    rows: dict[tuple, list[dict]] = defaultdict(list)
    with open(path, errors="replace") as f:
        for line in f:
            n_lines += 1
            try:
                r = json.loads(line)
                if not all(k in r for k in REQUIRED):
                    raise KeyError
            except (json.JSONDecodeError, KeyError, TypeError):
                n_corrupt += 1
                continue
            rows[(r["trial"], r["method"], r["budget"])].append(r)

    dup_keys = [k for k, v in rows.items() if len(v) > 1]
    disagree = [k for k in dup_keys if len({(x["correct"], x["answer"]) for x in rows[k]}) > 1]

    per_trial: dict[str, set] = defaultdict(set)
    for t, m, b in rows:
        per_trial[t].add((m, b))
    complete = [t for t in grid if expected <= per_trial.get(t, set())]
    incomplete = [t for t in grid if t in per_trial and not expected <= per_trial[t]]
    absent = [t for t in grid if t not in per_trial]
    return {
        "rows": rows,
        "expected": expected,
        "complete": complete,
        "report": {
            "lines": n_lines,
            "corrupt_lines": n_corrupt,
            "distinct_rows": len(rows),
            "duplicated_keys": len(dup_keys),
            "duplicates_disagreeing": len(disagree),
            "trials_in_grid": len(grid),
            "trials_complete": len(complete),
            "trials_incomplete": len(incomplete),
            "trials_absent": len(absent),
            "incomplete_ids": incomplete[:20],
            "disagreeing_keys": [list(k) for k in disagree[:20]],
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--path", help="results file (default: the config's out)")
    ap.add_argument("--write", action="store_true", help="rewrite the file with complete trials only")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    path = args.path or cfg["out"]

    with results_lock(path):  # never audit or rewrite while a sweep is writing
        a = audit(path, cfg)
        for k, v in a["report"].items():
            print(f"{k:>24}: {v}")
        if a["report"]["duplicates_disagreeing"]:
            print("\nWARNING: duplicate rows disagree. Trials are not reproducible; inspect before trusting results.")
        if not args.write:
            print("\n(report only; pass --write to keep complete trials and back up the original)")
            return
        backup = f"{path}.raw-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, backup)
        keep = set(a["complete"])
        tmp = path + ".tmp"
        n = 0
        with open(tmp, "w") as f:
            for (t, m, b), rs in sorted(a["rows"].items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
                if t in keep and (m, b) in a["expected"]:
                    f.write(json.dumps(rs[-1]) + "\n")
                    n += 1
        os.replace(tmp, path)
        print(f"\nwrote {n} rows ({len(keep)} complete trials) to {path}; original saved as {backup}")
        print("re-run the sweep with the same config to fill in the dropped trials.")


if __name__ == "__main__":
    main()
