"""Policies respect the memory budget and keep what they claim to keep."""

import pytest
import torch

from kvtier.memory import plan_bytes
from kvtier.model import ModelDims
from kvtier.policies import (
    epsilon,
    plan_h2o,
    plan_streaming,
    plan_tiered,
    plan_uniform,
    plan_window_evict,
    pool_scores,
)

DIMS = ModelDims(num_layers=4, num_q_heads=12, num_kv_heads=2, head_dim=128)
T = 1000
W = 32


@pytest.fixture(scope="module")
def scores():
    g = torch.Generator().manual_seed(0)
    s = torch.rand(DIMS.num_layers, DIMS.num_kv_heads, T, generator=g) ** 6
    s[..., 500:508] += 3.0  # a "needle" every head attends to
    return s


@pytest.mark.parametrize("ratio", [0.05, 0.1, 0.3, 0.6])
def test_eviction_policies_hit_budget(scores, ratio):
    for plan in (
        plan_streaming(DIMS, T, ratio),
        plan_h2o(DIMS, T, ratio, scores),
        plan_window_evict(DIMS, T, ratio, scores, W),
    ):
        plan.validate()
        assert plan.kept_per_head() == int(ratio * T)
        assert plan_bytes(plan, DIMS.head_dim).ratio == pytest.approx(int(ratio * T) / T)


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3, 0.5])
def test_tiered_never_exceeds_budget(scores, ratio):
    plan, info = plan_tiered(DIMS, T, ratio, scores, W)
    plan.validate()
    exact = plan_bytes(plan, DIMS.head_dim).ratio
    assert exact <= ratio + 1e-9
    assert exact > ratio * 0.95  # and does not waste much of it
    assert info.ratio == pytest.approx(exact)


def test_streaming_keeps_sinks_and_recent():
    plan = plan_streaming(DIMS, T, 0.1)
    kept = plan.key_bits[0, 0] > 0
    assert kept[:4].all() and kept[T - 96 :].all() and not kept[4 : T - 96].any()


def test_importance_methods_keep_the_needle(scores):
    for plan in (plan_window_evict(DIMS, T, 0.1, scores, W), plan_tiered(DIMS, T, 0.1, scores, W)[0]):
        assert (plan.key_bits[..., 500:508] > 0).all()
    # ...and StreamingLLM does not: that's the failure the project targets.
    assert not (plan_streaming(DIMS, T, 0.1).key_bits[..., 500:508] > 0).any()


def test_protected_tokens_full_precision(scores):
    plan, _ = plan_tiered(DIMS, T, 0.1, scores, W)
    assert (plan.key_bits[..., :4] == 16).all()
    assert (plan.key_bits[..., T - W :] == 16).all()


def test_tiered_more_budget_means_less_distortion(scores):
    d = [plan_tiered(DIMS, T, r, scores, W)[1].modelled_distortion for r in (0.1, 0.2, 0.3, 0.5)]
    assert all(a > b for a, b in zip(d, d[1:]))


def test_tiered_bits_follow_importance(scores):
    plan, _ = plan_tiered(DIMS, T, 0.2, pool_scores(scores, 1), W, pool_kernel=1)
    kb = plan.key_bits.float()
    s = scores
    # Among non-protected tokens, mean importance increases with tier.
    free = torch.ones(T, dtype=torch.bool)
    free[:4] = False
    free[T - W :] = False
    means = []
    for b in (0, 2, 4, 16):
        m = (kb == b) & free
        if m.any():
            means.append(float(s[m].mean()))
    assert means == sorted(means)


def test_uniform_ratios_match_hand_numbers():
    # head_dim 128, groups of 32: int4 = 160/512, int2 = 96/512 (docs/architecture.md)
    assert plan_bytes(plan_uniform(DIMS, 256, 4), 128).ratio == pytest.approx(160 / 512)
    assert plan_bytes(plan_uniform(DIMS, 256, 2), 128).ratio == pytest.approx(96 / 512)


def test_epsilon_ordering():
    assert epsilon(0) > epsilon(2) > epsilon(4) > epsilon(16) == 0


def test_budget_too_small_raises(scores):
    with pytest.raises(ValueError):
        plan_tiered(DIMS, T, 0.01, scores, W)


def _score_cases():
    g = torch.Generator().manual_seed(123)
    yield "heavy-tail", torch.rand(DIMS.num_layers, DIMS.num_kv_heads, T, generator=g) ** 12
    yield "mostly-zero", torch.relu(torch.randn(DIMS.num_layers, DIMS.num_kv_heads, T, generator=g) - 1.5) ** 6
    yield "sparse-spikes", (torch.rand(DIMS.num_layers, DIMS.num_kv_heads, T, generator=g) > 0.99).float()
    yield "ties", torch.ones(DIMS.num_layers, DIMS.num_kv_heads, T)
    z = torch.rand(DIMS.num_layers, DIMS.num_kv_heads, T, generator=g)
    z[:, 0] = 0
    yield "dead-head", z


@pytest.mark.parametrize("ratio", [0.07, 0.1, 0.15, 0.2, 0.3, 0.45])
def test_tiered_budget_is_a_guarantee(ratio):
    # Regression: the old re-tightening loop could fail to converge on real
    # attention scores (RuntimeError in needle job 39825). The budget must now
    # hold for any score distribution, in one pass.
    for name, s in _score_cases():
        s = s.clone()
        s[..., 0] += 10.0  # attention sink, as in real models
        plan, info = plan_tiered(DIMS, T, ratio, s, W)
        exact = plan_bytes(plan, DIMS.head_dim).ratio
        assert exact <= ratio + 1e-9, name
