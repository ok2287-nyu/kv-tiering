"""Verification test 1: a compressed cache with every token at full precision
reproduces the original logits exactly."""

import torch

from kvtier.cache import cache_len, clone_cache, decode_step, greedy_generate, prefill, prefill_split
from kvtier.tiers import TierPlan, apply_plan


def _decode_logits(model, cache, token, pos, steps):
    out = []
    for _ in range(steps):
        logits = decode_step(model, token, cache, pos)
        out.append(logits)
        token, pos = int(torch.argmax(logits)), pos + 1
    return torch.stack(out)


def test_full_precision_plan_is_bit_exact(model, dims, prompt_ids):
    sp = prefill_split(model, prompt_ids)
    ref_cache = clone_cache(sp.cache)
    comp_cache = clone_cache(sp.cache)
    apply_plan(comp_cache, TierPlan.uniform(dims, cache_len(comp_cache), bits=16))

    ref = _decode_logits(model, ref_cache, sp.last_token, sp.last_position, steps=8)
    got = _decode_logits(model, comp_cache, sp.last_token, sp.last_position, steps=8)
    assert torch.equal(ref, got), f"max |diff| = {(ref - got).abs().max().item()}"


def test_decode_loop_matches_single_forward(model, prompt_ids, atol):
    # Our split-prefill + decode loop must agree with one ordinary forward
    # pass over the same tokens (different kernels, so allclose not equal).
    sp = prefill_split(model, prompt_ids)
    gen = greedy_generate(model, sp.cache, sp.last_token, sp.last_position, max_new_tokens=5)
    full_ids = torch.cat([prompt_ids, torch.tensor([gen[:-1]], device=prompt_ids.device)], dim=1)
    with torch.no_grad():
        full_logits = model(full_ids).logits[0].float()
    n_prompt = prompt_ids.shape[1]
    ref_tokens = torch.argmax(full_logits[n_prompt - 1 :], dim=-1).tolist()
    assert ref_tokens == gen


def test_quantized_plan_changes_logits(model, dims, prompt_ids):
    # Sensitivity check: if 2-bit quantization left logits unchanged, the
    # identity test above would prove nothing.
    sp = prefill_split(model, prompt_ids)
    c = clone_cache(sp.cache)
    apply_plan(c, TierPlan.uniform(dims, cache_len(c), bits=2))
    ref = decode_step(model, sp.last_token, clone_cache(sp.cache), sp.last_position)
    got = decode_step(model, sp.last_token, c, sp.last_position)
    assert (ref - got).abs().max() > 1e-3


def test_prefill_returns_cache_of_prompt_length(model, dims, prompt_ids):
    res = prefill(model, prompt_ids)
    assert cache_len(res.cache) == prompt_ids.shape[1]
    k = res.cache.layers[0].keys
    assert k.shape == (1, dims.num_kv_heads, prompt_ids.shape[1], dims.head_dim)
    assert len(res.cache.layers) == dims.num_layers
