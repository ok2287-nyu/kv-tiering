"""The perplexity path: a multi-token chunk read through a (compressed) cache
must agree with Hugging Face's own computation."""

import torch

from kvtier.cache import cache_len, clone_cache, prefill_split
from kvtier.needle import expand_methods
from kvtier.perplexity import chunk_nll, run_ppl_trial
from kvtier.tiers import TierPlan, apply_plan


def test_chunk_nll_matches_single_forward(model, prompt_ids, atol):
    # No compression: split prefill + one chunk == one ordinary forward pass.
    ids = prompt_ids
    C = 12
    prefix, cont = ids[:, :-C], ids[:, -C:]
    sp = prefill_split(model, prefix)
    chunk = torch.cat([prefix[:, -1:], cont[:, :-1]], dim=1)
    got = chunk_nll(model, clone_cache(sp.cache), chunk, sp.last_position, cont)
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits[-C - 1 : -1], dim=-1)
    ref = -logp.gather(-1, cont[0][:, None]).squeeze(-1).cpu()
    assert torch.allclose(got, ref, atol=10 * atol), (got - ref).abs().max()


def test_chunk_after_eviction_matches_masked_attention(model, dims, prompt_ids, atol):
    # Evicting tokens then reading a chunk == keeping them and masking them,
    # at the same true positions. Covers the multi-token causal mask with an
    # offset (physical cache shorter than true positions).
    C = 10
    prefix, cont = prompt_ids[:, :-C], prompt_ids[:, -C:]
    sp = prefill_split(model, prefix)
    T = cache_len(sp.cache)
    ev = list(range(T // 3, T // 3 + T // 4))
    bits = torch.full((dims.num_layers, dims.num_kv_heads, T), 16)
    bits[..., ev] = 0
    chunk = torch.cat([prefix[:, -1:], cont[:, :-1]], dim=1)

    cache = apply_plan(clone_cache(sp.cache), TierPlan.from_bits(bits))
    got = chunk_nll(model, cache, chunk, sp.last_position, cont)

    mask = torch.ones(1, T + C, dtype=torch.long, device=model.device)
    mask[0, ev] = 0
    pos = torch.arange(sp.last_position, sp.last_position + C, device=model.device)[None]
    with torch.no_grad():
        out = model(input_ids=chunk.to(model.device), past_key_values=clone_cache(sp.cache),
                    attention_mask=mask, position_ids=pos, use_cache=True)
    logp = torch.log_softmax(out.logits[0].float(), dim=-1)
    ref = -logp.gather(-1, cont[0].to(model.device)[:, None]).squeeze(-1).cpu()
    assert torch.allclose(got, ref, atol=10 * atol), (got - ref).abs().max()
    # sensitivity: eviction really changed the predictions
    full = chunk_nll(model, clone_cache(sp.cache), chunk, sp.last_position, cont)
    assert (full - got).abs().max() > 1e-3


def test_run_ppl_trial(model, prompt_ids):
    specs = expand_methods(
        [{"name": "full"}, {"name": "streaming"}, {"name": "tiered"}, {"name": "tiered", "tag": "tiered-no2bit", "tiers": [4, 16]}],
        [0.5],
    )
    rows = run_ppl_trial(model, prompt_ids, cont_len=8, specs=specs, window=8)
    by = {r["method"]: r for r in rows}
    assert set(by) == {"full", "streaming", "tiered", "tiered-no2bit"}
    assert by["full"]["ratio"] == 1.0 and by["tiered"]["ratio"] <= 0.5 + 1e-9
    assert all(r["n_tokens"] == 8 and r["nll"] > 0 for r in rows)


def test_ppl_sweep_end_to_end(model, tmp_path):
    import json
    import sys
    from pathlib import Path

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import perplexity as ppl_cli
    import summarize_ppl
    from test_needle_harness import CharTokenizer

    tok = CharTokenizer(model.config.vocab_size)
    cfg = {
        "haystack": "synthetic", "out": str(tmp_path / "p.jsonl"), "seed": 1,
        "lengths": [200], "reps": 2, "cont_len": 16, "window": 8, "budgets": [0.3],
        "methods": [{"name": n} for n in ("full", "streaming", "h2o", "window_evict", "tiered")]
        + [{"name": "tiered", "tag": "tiered-protect4", "protect": 4}, {"name": "uniform", "bits": 2}],
    }
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    ppl_cli.main(["--config", str(tmp_path / "c.yaml")], model=model, tok=tok)
    rows = [json.loads(x) for x in open(cfg["out"])]
    assert len(rows) == 2 * 7 and all("job" in r and "commit" in r for r in rows)
    ppl_cli.main(["--config", str(tmp_path / "c.yaml")], model=model, tok=tok)  # resume adds nothing
    assert len(open(cfg["out"]).readlines()) == len(rows)
    summarize_ppl.main([cfg["out"]])
    assert (tmp_path / "p.summary.md").read_text().count("| tiered |") >= 1
