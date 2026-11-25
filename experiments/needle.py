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
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

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


def _job_alive(job_id: str) -> bool:
    """True if Slurm still lists ``job_id`` (or if we can't tell)."""
    if not job_id or job_id == "local":
        return True
    try:
        out = subprocess.run(["squeue", "-h", "-j", job_id], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return True  # can't check: assume alive, refuse to run
    return out.returncode == 0 and out.stdout.strip() != ""


@contextmanager
def results_lock(out_path: str):
    """Allow only one sweep to write a results file at a time.

    Two jobs appending to the same file on a network filesystem can overwrite
    each other's lines (this happened: jobs 40660 and 40667, Nov 23). The lock
    is a file created atomically (O_EXCL) holding the owner's Slurm job id. A
    lock whose job is no longer in the queue is stale and is taken over.
    """
    lock = out_path + ".lock"
    me = os.environ.get("SLURM_JOB_ID", "local")
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{me} {socket.gethostname()} {os.getpid()}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            owner = open(lock).read().split()
            owner_job = owner[0] if owner else ""
            if owner_job != "local" and not _job_alive(owner_job):
                print(f"removing stale lock from finished job {owner_job}", flush=True)
                os.remove(lock)
                continue
            raise SystemExit(
                f"{out_path} is being written by job {owner_job or '?'} ({lock}). "
                "Wait for it to finish (squeue -u $USER). If no such job is running, delete the lock file."
            )
    try:
        yield
    finally:
        try:
            os.remove(lock)
        except FileNotFoundError:
            pass


def done_trials(path: str, n_specs: int) -> set[str]:
    if not os.path.exists(path):
        return set()
    seen: dict[str, set] = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                key = (r["method"], r["budget"])
                tid = r["trial"]
            except (json.JSONDecodeError, KeyError, TypeError):  # truncated or clobbered line
                continue
            seen.setdefault(tid, set()).add(key)
    # distinct (method, budget) pairs, so duplicate rows can't make a trial look complete
    return {t for t, keys in seen.items() if len(keys) >= n_specs}


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

    # SIGTERM (sent by needle.sbatch before the time limit) -> normal exit, so
    # the lock is released before the continuation job starts.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    # Lock first: a second job fails immediately, before loading the model or
    # reading a file that another job is still writing.
    with results_lock(out_path):
        _sweep(args, cfg, out_path, model, tok)


def _provenance() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        commit = ""
    return {"job": os.environ.get("SLURM_JOB_ID", "local"), "commit": commit or "unknown", "host": socket.gethostname()}


def _sweep(args, cfg, out_path, model, tok):
    prov = _provenance()
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
                row.update(trial=tid, length=L, depth=d, rep=r, needle=needle.value, prompt_tokens=prompt.length, **prov)
            # One write per trial: if the job is killed, a trial is either
            # fully recorded or absent, never half-written (resume relies on it).
            fout.write("".join(json.dumps(row) + "\n" for row in rows))
            fout.flush()
            os.fsync(fout.fileno())
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
