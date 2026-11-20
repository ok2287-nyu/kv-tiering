"""Verification test 4: eviction keeps generation working and positions correct.

Reference for eviction: Hugging Face's own attention mask. Evicting token j
from the cache must give the same logits as keeping it and masking it out
(attention_mask[j] = 0), with the same true positions. The masked run never
touches our cache surgery, so agreement checks both the gather and the
position handling.
"""

import pytest
import torch

from kvtier.cache import cache_len, clone_cache, decode_step, greedy_generate, prefill_split
from kvtier.tiers import TierPlan, apply_plan


def _evict_plan(dims, T, evict_idx):
    bits = torch.full((dims.num_layers, dims.num_kv_heads, T), 16)
    bits[..., evict_idx] = 0
    return TierPlan.from_bits(bits)


def _middle_tokens(T):
    # Evict a contiguous middle block, the hard case for StreamingLLM, plus a
    # few scattered tokens. Keep the first token (attention sink).
    return sorted(set(range(T // 3, T // 3 + T // 4)) | {5, T - 3})


@torch.no_grad()
def _masked_reference(model, sp, evict_idx, steps):
    """Decode with the full cache, masking evicted positions."""
    cache = clone_cache(sp.cache)
    T = cache_len(cache)
    mask = torch.ones(1, T, dtype=torch.long, device=model.device)
    mask[0, evict_idx] = 0
    tok, pos, out = sp.last_token, sp.last_position, []
    for _ in range(steps):
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=model.device)], dim=1)
        logits = model(
            input_ids=torch.tensor([[tok]], device=model.device),
            past_key_values=cache,
            attention_mask=mask,
            position_ids=torch.tensor([[pos]], device=model.device),
            use_cache=True,
        ).logits[0, -1].float()
        out.append(logits)
        tok, pos = int(torch.argmax(logits)), pos + 1
    return torch.stack(out)


def _evicted_decode(model, sp, plan, steps):
    cache = clone_cache(sp.cache)
    apply_plan(cache, plan)
    tok, pos, out = sp.last_token, sp.last_position, []
    for _ in range(steps):
        logits = decode_step(model, tok, cache, pos)
        out.append(logits)
        tok, pos = int(torch.argmax(logits)), pos + 1
    return torch.stack(out), cache


def test_eviction_matches_masked_attention(model, dims, prompt_ids, atol):
    sp = prefill_split(model, prompt_ids)
    T = cache_len(sp.cache)
    ev = _middle_tokens(T)
    ref = _masked_reference(model, sp, ev, steps=6)
    got, cache = _evicted_decode(model, sp, _evict_plan(dims, T, ev), steps=6)
    assert cache_len(cache) == T - len(ev) + 6
    assert torch.allclose(ref, got, atol=atol), f"max |diff| = {(ref - got).abs().max().item()}"


def test_eviction_actually_matters(model, dims, prompt_ids):
    # Sensitivity: evicted logits must differ from the unevicted ones,
    # otherwise the test above could pass with a no-op eviction.
    sp = prefill_split(model, prompt_ids)
    T = cache_len(sp.cache)
    got, _ = _evicted_decode(model, sp, _evict_plan(dims, T, _middle_tokens(T)), steps=1)
    full = decode_step(model, sp.last_token, clone_cache(sp.cache), sp.last_position)
    assert (got[0] - full).abs().max() > 1e-3


def test_wrong_position_is_detected(model, dims, prompt_ids, atol):
    # If we let HF infer the position from the shrunken cache (the bug this
    # code exists to avoid), logits must differ from the masked reference.
    sp = prefill_split(model, prompt_ids)
    T = cache_len(sp.cache)
    ev = _middle_tokens(T)
    ref = _masked_reference(model, sp, ev, steps=1)[0]
    cache = clone_cache(sp.cache)
    apply_plan(cache, _evict_plan(dims, T, ev))
    wrong = decode_step(model, sp.last_token, cache, position=cache_len(cache))
    assert (ref - wrong).abs().max() > 10 * atol


def test_per_head_eviction_gathers_correct_tokens(model, dims, prompt_ids):
    # Different heads evict different tokens (same count per head).
    sp = prefill_split(model, prompt_ids)
    T = cache_len(sp.cache)
    bits = torch.full((dims.num_layers, dims.num_kv_heads, T), 16)
    g = torch.Generator().manual_seed(0)
    n_evict = T // 3
    kept = {}
    for layer in range(dims.num_layers):
        for h in range(dims.num_kv_heads):
            ev = torch.randperm(T - 1, generator=g)[:n_evict] + 1  # never evict token 0
            bits[layer, h, ev] = 0
            kept[(layer, h)] = torch.nonzero(bits[layer, h] > 0).squeeze(-1)
    cache = clone_cache(sp.cache)
    apply_plan(cache, TierPlan.from_bits(bits))
    for layer in range(dims.num_layers):
        k0, v0 = sp.cache.layers[layer].keys, sp.cache.layers[layer].values
        k1, v1 = cache.layers[layer].keys, cache.layers[layer].values
        assert k1.shape[2] == T - n_evict
        for h in range(dims.num_kv_heads):
            idx = kept[(layer, h)].to(k0.device)
            assert torch.equal(k1[0, h], k0[0, h, idx])
            assert torch.equal(v1[0, h], v0[0, h, idx])


def test_token_order_within_cache_is_irrelevant(model, dims, prompt_ids, atol):
    # Per-head eviction stores each head's tokens in its own order. That is
    # only valid if attention is invariant to the order of cached entries,
    # which holds because keys already carry their RoPE positions.
    sp = prefill_split(model, prompt_ids)
    ref = decode_step(model, sp.last_token, clone_cache(sp.cache), sp.last_position)
    cache = clone_cache(sp.cache)
    T = cache_len(cache)
    g = torch.Generator().manual_seed(0)
    for layer in cache.layers:
        for h in range(layer.keys.shape[1]):
            perm = torch.randperm(T, generator=g).to(layer.keys.device)
            layer.keys[0, h] = layer.keys[0, h, perm]
            layer.values[0, h] = layer.values[0, h, perm]
    got = decode_step(model, sp.last_token, cache, sp.last_position)
    assert torch.allclose(ref, got, atol=atol)


def test_generation_runs_after_heavy_eviction(model, dims, prompt_ids):
    sp = prefill_split(model, prompt_ids)
    T = cache_len(sp.cache)
    keep = 8
    bits = torch.zeros((dims.num_layers, dims.num_kv_heads, T), dtype=torch.long)
    bits[..., :4] = 16  # sink tokens
    bits[..., T - (keep - 4) :] = 16  # recent window
    cache = clone_cache(sp.cache)
    apply_plan(cache, TierPlan.from_bits(bits))
    assert cache_len(cache) == keep
    toks = greedy_generate(model, cache, sp.last_token, sp.last_position, max_new_tokens=10)
    assert len(toks) == 10
    assert cache_len(cache) == keep + 10


def test_plan_validation_rejects_unequal_head_counts(dims):
    T = 20
    bits = torch.full((dims.num_layers, dims.num_kv_heads, T), 16)
    bits[0, 0, 3] = 0  # only one head evicts
    with pytest.raises(ValueError, match="same number of tokens"):
        TierPlan.from_bits(bits).validate()
