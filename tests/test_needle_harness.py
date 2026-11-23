"""Integration test: the needle sweep runs end to end (tiny model, fake
character-level tokenizer, synthetic haystack) and writes sane rows."""

import json
import random

import pytest

from kvtier.needle import Needle, build_prompt, expand_methods, load_haystack_text, run_trial


class CharTokenizer:
    """Just enough tokenizer for the harness: one token per character."""

    eos_token_id = 1

    def __init__(self, vocab_size: int):
        self.v = vocab_size

    def _enc(self, text):
        return [2 + (ord(c) % (self.v - 2)) for c in text]

    def __call__(self, text, add_special_tokens=False):
        class R:
            pass

        r = R()
        r.input_ids = self._enc(text)
        return r

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 2) if 32 <= i - 2 < 127 else "?" for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "<u>" + messages[0]["content"] + "</u><a>"


@pytest.fixture(scope="module")
def tok(model):
    return CharTokenizer(model.config.vocab_size)


def test_prompt_has_exact_length_and_needle(tok):
    hay = tok(load_haystack_text("synthetic", n_chars=20_000)).input_ids
    needle = Needle("Oslo", "4812337")
    p = build_prompt(tok, hay, 600, 0.5, needle, random.Random(0))
    assert p.length == p.ids.shape[1]
    assert abs(p.length - 600) <= 0  # exact: haystack is sized to fill the rest
    s, e = p.needle_span
    assert tok.decode(p.ids[0, s:e].tolist()) == needle.text
    assert 0.4 * 600 < s < 0.7 * 600


def test_sweep_end_to_end(model, tok, tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import needle as needle_cli  # experiments/needle.py
    import summarize

    cfg = {
        "haystack": "synthetic",
        "out": str(tmp_path / "r.jsonl"),
        "lengths": [400],
        "depths": [0.3, 0.7],
        "reps": 1,
        "window": 16,
        "max_new_tokens": 4,
        "budgets": [0.2, 0.5],
        "methods": [{"name": n} for n in ("full", "streaming", "h2o", "window_evict", "tiered")]
        + [{"name": "uniform", "bits": 4}],
    }
    cfg_path = tmp_path / "c.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg))
    needle_cli.main(["--config", str(cfg_path)], model=model, tok=tok)
    rows = [json.loads(line) for line in open(cfg["out"])]
    n_specs = len(expand_methods(cfg["methods"], cfg["budgets"]))
    assert len(rows) == 2 * n_specs
    for r in rows:
        assert r["ratio"] <= (r["budget"] or 1.0) + 1e-9
        if r["method"] == "full":
            assert r["needle_kept_frac"] == 1.0 and r["needle_mean_key_bits"] == 16
    # resume: a second run adds nothing
    needle_cli.main(["--config", str(cfg_path)], model=model, tok=tok)
    assert len(open(cfg["out"]).readlines()) == len(rows)
    summarize.main([cfg["out"], "--boot", "200"])
    assert (tmp_path / "r.summary.csv").exists()


def test_run_trial_shares_one_prefill(model, tok):
    hay = tok(load_haystack_text("synthetic", n_chars=20_000)).input_ids
    p = build_prompt(tok, hay, 300, 0.5, Needle("Lima", "1234567"), random.Random(1))
    specs = expand_methods([{"name": "full"}, {"name": "streaming"}], [0.1])
    rows = run_trial(model, tok, p, specs, window=16, max_new_tokens=3)
    assert [r["method"] for r in rows] == ["full", "streaming"]
    assert rows[1]["needle_kept_frac"] == 0.0  # mid-document needle evicted by sink+recent


def test_resume_tolerates_killed_job(tmp_path):
    # A job killed at the time limit can leave a truncated last line, and the
    # resubmitted job re-runs that trial. Resume must skip only complete
    # trials, and the summary must count each (trial, method, budget) once.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import needle as needle_cli
    import summarize

    def row(trial, method, correct):
        return json.dumps(
            {"trial": trial, "method": method, "budget": 0.1, "ratio": 0.1, "correct": correct,
             "depth": 0.5, "needle_kept_frac": 1.0}
        )

    p = tmp_path / "r.jsonl"
    p.write_text(
        row("A", "m1", True) + "\n" + row("A", "m2", True) + "\n"   # complete trial
        + row("B", "m1", False) + "\n" + '{"trial": "B", "meth'      # killed mid-write
    )
    assert needle_cli.done_trials(str(p), n_specs=2) == {"A"}
    # the resubmitted job re-runs B completely
    with open(p, "a") as f:
        f.write("\n" + row("B", "m1", True) + "\n" + row("B", "m2", True) + "\n")
    rows = summarize.load_rows(str(p))
    assert len(rows) == 4
    assert {(r["trial"], r["method"]): r["correct"] for r in rows}[("B", "m1")] is True
    assert needle_cli.done_trials(str(p), n_specs=2) == {"A", "B"}


def _exp_modules():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import clean_results
    import needle as needle_cli

    return needle_cli, clean_results


def test_lock_blocks_second_writer(tmp_path, monkeypatch):
    needle_cli, _ = _exp_modules()
    out = str(tmp_path / "r.jsonl")
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    monkeypatch.setattr(needle_cli, "_job_alive", lambda j: True)
    with needle_cli.results_lock(out):
        with pytest.raises(SystemExit, match="being written by job 111"):
            with needle_cli.results_lock(out):
                pass
    assert not (tmp_path / "r.jsonl.lock").exists()  # released on exit


def test_stale_lock_is_taken_over(tmp_path, monkeypatch):
    needle_cli, _ = _exp_modules()
    out = str(tmp_path / "r.jsonl")
    (tmp_path / "r.jsonl.lock").write_text("999 somehost 1\n")
    monkeypatch.setattr(needle_cli, "_job_alive", lambda j: False)
    with needle_cli.results_lock(out):
        assert (tmp_path / "r.jsonl.lock").read_text().split()[0] != "999"


def test_clean_results_repairs_clobbered_file(tmp_path):
    # Simulates the Nov 23 concurrent-writer damage: duplicates, a clobbered
    # line, and a trial with a row missing.
    import yaml

    needle_cli, clean_results = _exp_modules()
    cfg = {
        "out": str(tmp_path / "r.jsonl"),
        "lengths": [100], "depths": [0.5], "reps": 3, "budgets": [0.1],
        "methods": [{"name": "full"}, {"name": "tiered"}],
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))

    def row(t, m, b, ok=True):
        return json.dumps({"trial": t, "method": m, "budget": b, "correct": ok, "ratio": 0.1, "answer": "x",
                           "depth": 0.5, "needle_kept_frac": 1.0})

    t0, t1, t2 = (needle_cli.trial_id(100, 0.5, r) for r in range(3))
    lines = [
        row(t0, "full", None), row(t0, "tiered", 0.1),
        row(t0, "full", None), row(t0, "tiered", 0.1),   # duplicate trial (second writer)
        row(t1, "full", None), '{"trial": "L100-d0.50-r1", "meth{"trial":',   # clobbered
        row(t2, "full", None), row(t2, "tiered", 0.1),
    ]
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(lines) + "\n")

    rep = clean_results.audit(str(p), cfg)["report"]
    assert rep["corrupt_lines"] == 1
    assert rep["duplicated_keys"] == 2 and rep["duplicates_disagreeing"] == 0
    assert (rep["trials_complete"], rep["trials_incomplete"], rep["trials_absent"]) == (2, 1, 0)

    clean_results.main(["--config", str(tmp_path / "c.yaml"), "--write"])
    clean = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(clean) == 4 and {r["trial"] for r in clean} == {t0, t2}
    assert list(tmp_path.glob("r.jsonl.raw-*"))  # backup kept
    # resume now re-runs only the dropped trial
    assert needle_cli.done_trials(str(p), n_specs=2) == {t0, t2}
