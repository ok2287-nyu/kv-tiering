"""Importance scores equal the attention weights computed by the reference
(eager) implementation, and recording does not change model outputs."""

import torch

from kvtier.cache import prefill_split
from kvtier.importance import ATTN_IMPL, record_attention


def _eager_attentions(model, ids):
    model.set_attn_implementation("eager")
    try:
        with torch.no_grad():
            out = model(ids, output_attentions=True)
    finally:
        model.set_attn_implementation(ATTN_IMPL)
    return torch.stack([a[0].float() for a in out.attentions])  # [L, Hq, T, T]


def test_recording_does_not_change_outputs(model, prompt_ids):
    with torch.no_grad():
        ref = model(prompt_ids).logits
        with record_attention(model, window=8, cumulative=True):
            got = model(prompt_ids).logits
    assert torch.equal(ref, got)


def test_scores_match_eager_attention(model, dims, prompt_ids):
    W = 8
    with torch.no_grad(), record_attention(model, window=W, cumulative=True) as rec:
        model(prompt_ids)
    s = rec.result()
    A = _eager_attentions(model, prompt_ids)
    L, Hq, T, _ = A.shape
    g = dims.group_size
    # Query head h reads KV head h // g, so group consecutive query heads.
    exp_window = A[:, :, -W:].sum(2).view(L, dims.num_kv_heads, g, T).sum(2).cpu()
    exp_cum = A.sum(2).view(L, dims.num_kv_heads, g, T).sum(2).cpu()
    assert torch.allclose(s.window, exp_window, rtol=1e-4, atol=1e-3)
    assert torch.allclose(s.cumulative, exp_cum, rtol=1e-4, atol=1e-3)


def test_window_mass_and_causality(model, dims, prompt_ids):
    W = 8
    with torch.no_grad(), record_attention(model, window=W) as rec:
        sp = prefill_split(model, prompt_ids)
    s = rec.result()
    T = prompt_ids.shape[1] - 1
    assert s.window.shape == (dims.num_layers, dims.num_kv_heads, T)
    # Each of the W window queries spreads mass 1 per query head; g heads per KV head.
    assert torch.allclose(s.window.sum(-1), torch.full_like(s.window[..., 0], W * dims.group_size), atol=1e-3)
    assert s.cumulative is None
    # Cumulative mass is biased toward early tokens (H2O's known bias):
    # token j is seen by T - j queries. Check with the cumulative score.
    with torch.no_grad(), record_attention(model, window=W, cumulative=True) as rec:
        prefill_split(model, prompt_ids)
    c = rec.result().cumulative
    first_half = c[..., : T // 2].mean()
    second_half = c[..., T // 2 :].mean()
    assert first_half > second_half


def test_decode_steps_are_not_recorded(model, prompt_ids):
    from kvtier.cache import decode_step

    with torch.no_grad(), record_attention(model, window=4) as rec:
        sp = prefill_split(model, prompt_ids)
        before = [x.clone() for x in rec.win]
        decode_step(model, sp.last_token, sp.cache, sp.last_position)
    assert all(torch.equal(a, b) for a, b in zip(before, rec.win))
