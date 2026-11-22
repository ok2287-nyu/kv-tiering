"""Compression policies: each turns (model dims, prompt length, scores, budget)
into a :class:`TierPlan`.

The budget is always a **memory ratio** ``r``: accounted bytes of the
compressed cache divided by the bytes of the full bf16 cache (``memory.py``).
All methods are compared at equal ``r``.

| policy           | precision        | which tokens survive                         |
|------------------|------------------|----------------------------------------------|
| ``full``         | bf16             | all                                          |
| ``uniform``      | b bits (KIVI)    | all (``r`` is fixed by ``b``: 31% / 19%)     |
| ``streaming``    | bf16             | 4 sink tokens + most recent                  |
| ``h2o``          | bf16             | half heavy hitters (cumulative attn), half recent |
| ``window_evict`` | bf16             | top by observation-window attention (SnapKV-like) |
| ``tiered``       | 16 / 4 / 2 / 0   | allocator below (ours)                       |

``window_evict`` vs ``tiered`` is the key comparison. They share the same
scores, so any difference comes from mixed precision, not from better scoring.

Every eviction policy keeps the same number of tokens in every (layer, KV
head), as the rectangular cache requires (see ``tiers.py``). H2O and
window_evict still pick *which* tokens per head.

The tiered allocator
--------------------
Distortion model (docs/architecture.md and the derivation in the project
notes): entry ``j`` with importance ``s_j`` stored at ``b`` bits adds expected
squared output error ``s_j^p · ε(b)`` with

* ``ε(b) = 1 / (12 (2^b − 1)²)``: uniform-quantization noise variance,
  relative to the squared group range;
* ``ε(0) = 1/12``: eviction removes ``w_j (v_j − o)``, and ``|v_j − o|²`` is
  modelled as the variance of a value spread uniformly over its range;
* ``ε(16) = 0``.

With ``p = 2`` this is the value-error model ``Σ w_j² E‖ε_j‖²``. Given a
kept count ``K`` (equal for every head), each head keeps its top-``K`` tokens.
Kept tokens start at the lowest tier and are upgraded greedily by
distortion removed per byte. The ratios shrink for each further upgrade of a
token (2→4 beats 4→16), so a global sort respects the order of upgrades. ``K``
is chosen by a grid search over the modelled distortion. Sink tokens and the
observation window are always kept at full precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .memory import entry_cost, plan_bytes
from .model import ModelDims
from .tiers import FULL, QuantConfig, TierPlan

N_SINK = 4


def _shape(dims: ModelDims, T: int) -> tuple[int, int, int]:
    return (dims.num_layers, dims.num_kv_heads, T)


def _keep_count(ratio: float, T: int) -> int:
    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    return max(1, int(ratio * T))


def pool_scores(scores: torch.Tensor, kernel: int) -> torch.Tensor:
    """Max-pool along tokens so neighbours of an important token score high too.

    SnapKV's observation: keeping isolated tokens fragments facts. A needle
    like "number for Oslo is 4812337" spans several tokens and they should
    survive together. Applied to window_evict and tiered alike.
    """
    if kernel <= 1:
        return scores
    L, H, T = scores.shape
    x = scores.reshape(L * H, 1, T)
    return F.max_pool1d(x, kernel, stride=1, padding=kernel // 2).reshape(L, H, T)


def protected_mask(T: int, window: int, n_sink: int = N_SINK) -> torch.Tensor:
    m = torch.zeros(T, dtype=torch.bool)
    m[: min(n_sink, T)] = True
    m[max(0, T - window) :] = True
    return m


# --------------------------------------------------------------------------- baselines


def plan_full(dims: ModelDims, T: int) -> TierPlan:
    return TierPlan.uniform(dims, T, FULL)


def plan_uniform(dims: ModelDims, T: int, bits: int, value_bits: int | None = None) -> TierPlan:
    return TierPlan.uniform(dims, T, bits, value_bits)


def plan_streaming(dims: ModelDims, T: int, ratio: float, n_sink: int = N_SINK) -> TierPlan:
    """StreamingLLM (Xiao et al., 2024): attention sinks + recent window."""
    K = _keep_count(ratio, T)
    bits = torch.zeros(_shape(dims, T), dtype=torch.long)
    sink = min(n_sink, K)
    bits[..., :sink] = FULL
    if K > sink:
        bits[..., T - (K - sink) :] = FULL
    return TierPlan.from_bits(bits)


def _topk_keep(scores: torch.Tensor, K: int, forced: torch.Tensor) -> torch.Tensor:
    """Bool [L, H, T]: per head, the K highest-scoring tokens, forced ones first."""
    s = scores.clone()
    s[..., forced] = float("inf")
    idx = torch.topk(s, K, dim=-1).indices
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep.scatter_(-1, idx, True)
    return keep


def plan_h2o(dims: ModelDims, T: int, ratio: float, cumulative: torch.Tensor, recent_frac: float = 0.5) -> TierPlan:
    """H2O (Zhang et al., 2023): heavy hitters by cumulative attention + recent tokens.

    One-shot variant: scores come from the prompt only, then eviction happens
    once. The original also updates scores during generation.
    """
    K = _keep_count(ratio, T)
    n_recent = min(K, max(1, int(round(K * recent_frac))))
    forced = torch.zeros(T, dtype=torch.bool)
    forced[T - n_recent :] = True
    keep = _topk_keep(cumulative.float(), K, forced)
    return TierPlan.from_bits(keep.long() * FULL)


def plan_window_evict(
    dims: ModelDims, T: int, ratio: float, window_scores: torch.Tensor, window: int, pool_kernel: int = 7
) -> TierPlan:
    """SnapKV-style (Li et al., 2024): keep top tokens by observation-window attention."""
    K = _keep_count(ratio, T)
    forced = protected_mask(T, window)
    if int(forced.sum()) > K:
        raise ValueError(f"budget keeps {K} tokens, fewer than the {int(forced.sum())} protected ones")
    keep = _topk_keep(pool_scores(window_scores.float(), pool_kernel), K, forced)
    return TierPlan.from_bits(keep.long() * FULL)


# --------------------------------------------------------------------------- ours


def epsilon(bits: int) -> float:
    if bits == FULL:
        return 0.0
    if bits == 0:
        return 1.0 / 12.0
    return 1.0 / (12.0 * (2**bits - 1) ** 2)


@dataclass
class AllocInfo:
    kept: int  # tokens kept per head
    modelled_distortion: float
    ratio: float  # exact accounted memory ratio
    tier_fractions: dict[int, float]  # fraction of all entries at each bit width


def _allocate_for_K(s_pow, forced, K, budget_bytes, tiers, costs):
    """Greedy allocation for one kept count. Returns (bits [L,H,T], distortion) or None."""
    lo = tiers[0]
    keep = _topk_keep(s_pow, K, forced)
    free = keep & ~forced  # kept entries the allocator may quantize
    n_forced = int((keep & forced).sum())
    base = n_forced * costs[FULL] + int(free.sum()) * costs[lo]
    if base > budget_bytes:
        return None
    bits = torch.zeros_like(s_pow, dtype=torch.long)
    bits[keep & forced] = FULL
    bits[free] = lo

    # Candidate upgrades for every free entry: lo->t1, t1->t2, ...
    sv = s_pow[free]
    flat_idx = torch.nonzero(free.flatten()).squeeze(-1)
    gains, cst, targets, owners = [], [], [], []
    for a, b in zip(tiers[:-1], tiers[1:]):
        gains.append(sv * (epsilon(a) - epsilon(b)))
        cst.append(torch.full_like(sv, costs[b] - costs[a]))
        targets.append(torch.full_like(flat_idx, b))
        owners.append(flat_idx)
    gains, cst = torch.cat(gains), torch.cat(cst)
    targets, owners = torch.cat(targets), torch.cat(owners)
    order = torch.argsort(gains / cst, descending=True, stable=True)
    csum = torch.cumsum(cst[order].double(), 0)
    n_take = int((csum <= budget_bytes - base).sum())
    take = order[:n_take]
    # Within one owner, later tiers have lower ratios, so the prefix is
    # prerequisite-closed and the max target is the final tier.
    flat_bits = bits.flatten()
    final = torch.zeros_like(flat_bits)
    final.scatter_reduce_(0, owners[take], targets[take], reduce="amax")
    upgraded = final > 0
    flat_bits[upgraded] = final[upgraded]
    bits = flat_bits.view_as(bits)

    eps = torch.zeros_like(s_pow)
    for t in (0, *tiers):
        eps[bits == t] = epsilon(t)
    distortion = float((s_pow * eps).sum())
    return bits, distortion


def plan_tiered(
    dims: ModelDims,
    T: int,
    ratio: float,
    window_scores: torch.Tensor,
    window: int,
    qcfg: QuantConfig | None = None,
    tiers: tuple[int, ...] = (2, 4, FULL),
    power: float = 2.0,
    pool_kernel: int = 7,
    k_grid: int = 24,
) -> tuple[TierPlan, AllocInfo]:
    """Importance-aware mixed precision under a memory ratio (ours)."""
    qcfg = qcfg or QuantConfig()
    tiers = tuple(sorted(tiers))
    costs = {b: entry_cost(b, dims.head_dim, qcfg) for b in (0, *tiers)}
    full_bytes = T * dims.num_layers * dims.num_kv_heads * costs[FULL]
    forced = protected_mask(T, window)
    n_forced = int(forced.sum())

    s = pool_scores(window_scores.float(), pool_kernel)
    s = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # each head's mass sums to 1
    s_pow = s.pow(power)

    # The allocator prices key statistics amortised per token (D·2·stat/Gk),
    # but exact accounting rounds each (layer, head, quantized tier) up to whole
    # groups of Gk tokens. Reserve that worst case up front:
    #   per (layer, head, quantized tier) at most (Gk − 1)/Gk of one group's stats.
    # This *guarantees* exact bytes ≤ budget in one pass. (An earlier version
    # re-allocated after measuring the overshoot; the rounding changes with the
    # allocation, so it could fail to converge — it crashed on a real trial.)
    n_quant_tiers = sum(1 for b in tiers if b != FULL)
    key_group_stats = dims.head_dim * 2 * qcfg.stat_bytes
    rounding_reserve = (
        dims.num_layers * dims.num_kv_heads * n_quant_tiers * key_group_stats * (qcfg.key_group_size - 1) / qcfg.key_group_size
    )
    budget = ratio * full_bytes
    target = budget - rounding_reserve
    best = None
    for K in torch.linspace(n_forced, T, k_grid).round().long().unique().tolist():
        r = _allocate_for_K(s_pow, forced, int(K), target, tiers, costs)
        if r is not None and (best is None or r[1] < best[1]):
            best = (r[0], r[1], int(K))
    if best is None:
        raise ValueError(f"ratio {ratio} too small: even protected tokens at full precision don't fit")
    plan = TierPlan.from_bits(best[0])
    exact = plan_bytes(plan, dims.head_dim, qcfg).total_bytes
    if exact > budget * (1 + 1e-9):  # cannot happen if the reserve is right; fail loudly if it isn't
        raise AssertionError(f"tiered plan uses {exact:.0f} B > budget {budget:.0f} B (reserve {rounding_reserve:.0f} B)")

    total = best[0].numel()
    fr = {b: float((best[0] == b).sum()) / total for b in (0, *tiers)}
    info = AllocInfo(kept=best[2], modelled_distortion=best[1], ratio=exact / full_bytes, tier_fractions=fr)
    return plan, info
