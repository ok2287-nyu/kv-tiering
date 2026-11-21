"""Memory accounting for a tier plan: the "rate" axis of the rate-distortion curves.

These are the bytes a real implementation with packed low-bit storage would
use. Our quantization is simulated, so for quantized tiers this is an
*accounted* number, not a measured one. Evicted tokens are physically removed,
so that part of the saving is real. Reports must keep that distinction.

Per stored (layer, kv_head, token) entry, with head dimension D:

* data: ``D * key_bits / 8 + D * value_bits / 8`` bytes
* value statistics (per token): ``ceil(D / Gv)`` groups x (scale + zero)
* key statistics (per channel): each group of ``Gk`` same-tier tokens in one
  head carries ``D`` (scale, zero) pairs, so a tier holding ``n`` tokens costs
  ``ceil(n / Gk) * D * 2 * stat_bytes``.

Full-precision entries carry no statistics. No per-token tier index is
counted: since keys carry their RoPE position, a real store can keep each
tier's tokens contiguously and needs only a few counts per head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import ModelDims
from .tiers import FULL, QuantConfig, TierPlan

DTYPE_BYTES = 2  # bf16 / fp16


def full_bytes_per_token(dims: ModelDims, dtype_bytes: int = DTYPE_BYTES) -> int:
    """Bytes the uncompressed cache spends per token: K and V, all layers and KV heads."""
    return 2 * dims.num_layers * dims.num_kv_heads * dims.head_dim * dtype_bytes


@dataclass
class MemoryReport:
    data_bytes: float
    stat_bytes: float
    full_bytes: float  # the uncompressed cache for the same token count
    tokens: int

    @property
    def total_bytes(self) -> float:
        return self.data_bytes + self.stat_bytes

    @property
    def ratio(self) -> float:
        """Compressed size / uncompressed size (1.0 = no compression)."""
        return self.total_bytes / self.full_bytes

    @property
    def bytes_per_token(self) -> float:
        return self.total_bytes / self.tokens


def entry_cost(bits: int, head_dim: int, qcfg: QuantConfig | None = None, dtype_bytes: int = DTYPE_BYTES) -> float:
    """Approximate bytes for one (layer, head, token) entry, K and V together.

    Key statistics are amortised over a full group (``D·2·stat/Gk`` per token).
    ``plan_bytes`` rounds partial groups up, so it can be slightly higher; the
    allocator checks the exact number afterwards.
    """
    qcfg = qcfg or QuantConfig()
    D = head_dim
    if bits == 0:
        return 0.0
    if bits == FULL:
        return 2 * D * dtype_bytes
    data = 2 * D * bits / 8
    vstats = math.ceil(D / qcfg.value_group_size) * 2 * qcfg.stat_bytes
    kstats = D * 2 * qcfg.stat_bytes / qcfg.key_group_size
    return data + vstats + kstats


def plan_bytes(plan: TierPlan, head_dim: int, qcfg: QuantConfig | None = None, dtype_bytes: int = DTYPE_BYTES) -> MemoryReport:
    qcfg = qcfg or QuantConfig()
    L, H, T = plan.shape
    D = head_dim
    data = 0.0
    stats = 0.0
    value_groups_per_token = math.ceil(D / qcfg.value_group_size)
    for layer in range(L):
        for h in range(H):
            kb = plan.key_bits[layer, h]
            vb = plan.value_bits[layer, h]
            for bits_t, is_key in ((kb, True), (vb, False)):
                for bits in bits_t.unique().tolist():
                    if bits == 0:
                        continue
                    n = int((bits_t == bits).sum())
                    if bits == FULL:
                        data += n * D * dtype_bytes
                        continue
                    data += n * D * bits / 8
                    if is_key:
                        stats += math.ceil(n / qcfg.key_group_size) * D * 2 * qcfg.stat_bytes
                    else:
                        stats += n * value_groups_per_token * 2 * qcfg.stat_bytes
    full = T * 2 * L * H * D * dtype_bytes
    return MemoryReport(data_bytes=data, stat_bytes=stats, full_bytes=full, tokens=T)
