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
  * **inconsistent rows**: valid JSON that contradicts itself. Overlapping
    writes can splice the front of one row onto the back of another and still
    produce valid JSON (found on Nov 25: trial r0's row carried trial r1's
    answer). Each row is checked against facts derivable from the row alone:
    its needle must equal the needle regenerated from the trial's seed,
    ``correct`` must equal "needle appears in answer", the trial id must match
    its length/depth/rep fields, and a budgeted row's memory ratio must not
    exceed its budget;
  * duplicate rows: the same (trial, method, budget) more than once, and how
    many consistent duplicates give a different answer. Trials use fixed seeds
    and greedy decoding, but bf16 kernels on different GPUs are not bit-exact,
    so an occasional flip near a tie is expected; it's reported, not fatal;
  * trials complete / incomplete / absent relative to the config's grid. A
    trial with any inconsistent row counts as incomplete.

``--write`` keeps one row per (trial, method, budget) for complete trials only,
backs up the original to ``<file>.raw-<timestamp>``, and rewrites the file. Then
re-running the sweep with the same config re-runs just the dropped trials.
"""

from __future__ import annotations

import argparse
import json
import random
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle import results_lock, trial_id  # noqa: E402

from kvtier.needle import expand_methods, make_needle  # noqa: E402

REQUIRED = ("trial", "method", "budget", "correct", "ratio", "answer")


def row_problems(r: dict, seed: int) -> list[str]:
    """Reasons this row contradicts itself (empty list = consistent)."""
    probs = []
    tid = r["trial"]
    needle = r.get("needle")
    if needle is not None:
        expected = make_needle(random.Random(f"{seed}-{tid}")).value
        if needle != expected:
            probs.append(f"needle {needle} != seeded {expected}")
        if bool(r["correct"]) != (str(needle) in str(r["answer"])):
            probs.append("correct flag contradicts answer")
    if all(k in r for k in ("length", "depth", "rep")):
        if trial_id(r["length"], r["depth"], r["rep"]) != tid:
            probs.append("trial id contradicts length/depth/rep")
    if r["budget"] is not None and r["ratio"] > r["budget"] + 1e-9:
        probs.append(f"ratio {r['ratio']:.4f} > budget {r['budget']}")
    return probs


def audit(path: str, cfg: dict) -> dict:
    specs = expand_methods(cfg["methods"], cfg["budgets"])
    expected = {(s.label, s.budget) for s in specs}
    grid = [trial_id(L, d, r) for L in cfg["lengths"] for d in cfg["depths"] for r in range(cfg["reps"])]
    seed = cfg.get("seed", 0)

    n_lines = n_corrupt = 0
    rows: dict[tuple, list[dict]] = defaultdict(list)
    bad_rows: list[tuple[dict, list[str]]] = []
    bad_trials: set[str] = set()
    with open(path, errors="replace") as f:
        for line in f:
            n_lines += 1
            try:
                r = json.loads(line.strip("\x00 \n"))
                if not all(k in r for k in REQUIRED):
                    raise KeyError
            except (json.JSONDecodeError, KeyError, TypeError):
                n_corrupt += 1
                continue
            probs = row_problems(r, seed)
            if probs:
                bad_rows.append((r, probs))
                bad_trials.add(r["trial"])
                continue
            rows[(r["trial"], r["method"], r["budget"])].append(r)

    dup_keys = [k for k, v in rows.items() if len(v) > 1]
    flips = [k for k in dup_keys if len({(x["correct"], x["answer"]) for x in rows[k]}) > 1]

    per_trial: dict[str, set] = defaultdict(set)
    for t, m, b in rows:
        per_trial[t].add((m, b))
    complete = [t for t in grid if t not in bad_trials and expected <= per_trial.get(t, set())]
    incomplete = [t for t in grid if t not in complete and (t in per_trial or t in bad_trials)]
    absent = [t for t in grid if t not in per_trial and t not in bad_trials]
    return {
        "rows": rows,
        "expected": expected,
        "complete": complete,
        "report": {
            "lines": n_lines,
            "corrupt_lines": n_corrupt,
            "inconsistent_rows": len(bad_rows),
            "distinct_rows": len(rows),
            "duplicated_keys": len(dup_keys),
            "duplicate_answer_flips": len(flips),
            "trials_in_grid": len(grid),
            "trials_complete": len(complete),
            "trials_incomplete": len(incomplete),
            "trials_absent": len(absent),
            "incomplete_ids": incomplete[:20],
            "inconsistent_examples": [
                f"{r['trial']} {r['method']} {r['budget']}: {'; '.join(p)}" for r, p in bad_rows[:10]
            ],
            "flipped_keys": [list(k) for k in flips[:20]],
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
        rep = a["report"]
        if rep["inconsistent_rows"]:
            print(
                f"\nWARNING: {rep['inconsistent_rows']} rows contradict themselves (spliced or corrupted). "
                "Their trials are treated as incomplete. If there are many, re-run the sweep into a fresh file "
                "rather than repairing this one."
            )
        if rep["duplicate_answer_flips"]:
            print(
                f"\nnote: {rep['duplicate_answer_flips']} of {rep['duplicated_keys']} results computed twice gave "
                "different answers (expected occasionally across GPUs: bf16 kernels aren't bit-exact)."
            )
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
