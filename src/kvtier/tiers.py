"""Tier plans and their application to a KV cache.

A *tier plan* assigns a bit width to every cached entry, indexed by
``(layer, kv_head, token)``:

* 16 = keep in the model dtype (bf16 on GPU), no error
* 8 / 4 / 2 = simulated quantization at that width
* 0 = evict

Keys and values get separate bit tensors so the key-vs-value ablation can
give them different precisions. An entry is evicted only if both its key and
value bits are 0.

Rectangular-cache constraint
----------------------------
Cache tensors are ``[batch, kv_heads, tokens, head_dim]``: every head in every
layer must hold the same number of tokens. Different heads may evict
*different* tokens (that is the point of per-head importance), but they must
evict the *same number*. Because keys carry their RoPE position, the order of
tokens inside a head does not matter to attention, so each head can keep its
own token set as long as the count matches. ``TierPlan.validate`` enforces
this; allocators must respect it (per-head budgets were cut from scope).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache import get_layer_kv, num_layers, set_layer_kv
from .model import ModelDims
from .quant import fake_quant_keys, fake_quant_values

ALLOWED_BITS = (0, 2, 4, 8, 16)
FULL = 16


@dataclass
class QuantConfig:
    key_group_size: int = 32  # tokens sharing one per-channel (scale, zero) — KIVI uses 32
    value_group_size: int = 32  # channels sharing one per-token (scale, zero)
    stat_bytes: int = 2  # bytes per stored scale or zero (fp16)


@dataclass
class TierPlan:
    key_bits: torch.Tensor  # int64 [L, H_kv, T]
    value_bits: torch.Tensor  # int64 [L, H_kv, T]

    @classmethod
    def uniform(cls, dims: ModelDims, seq_len: int, bits: int, value_bits: int | None = None) -> "TierPlan":
        shape = (dims.num_layers, dims.num_kv_heads, seq_len)
        vb = bits if value_bits is None else value_bits
        return cls(torch.full(shape, bits, dtype=torch.long), torch.full(shape, vb, dtype=torch.long))

    @classmethod
    def from_bits(cls, bits: torch.Tensor) -> "TierPlan":
        """Same bit width for key and value of each entry."""
        bits = bits.long()
        return cls(bits.clone(), bits.clone())

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.key_bits.shape)  # type: ignore[return-value]

    @property
    def evicted(self) -> torch.Tensor:
        return self.key_bits == 0

    def kept_per_head(self) -> int:
        kept = (~self.evicted).sum(dim=-1)
        return int(kept.flatten()[0])

    def validate(self, dims: ModelDims | None = None, seq_len: int | None = None) -> None:
        if self.key_bits.shape != self.value_bits.shape:
            raise ValueError("key_bits and value_bits shapes differ")
        if self.key_bits.dim() != 3:
            raise ValueError("plan must be [layers, kv_heads, tokens]")
        if dims is not None and self.shape[:2] != (dims.num_layers, dims.num_kv_heads):
            raise ValueError(f"plan shape {self.shape} does not match model dims {dims}")
        if seq_len is not None and self.shape[2] != seq_len:
            raise ValueError(f"plan covers {self.shape[2]} tokens, cache has {seq_len}")
        for name, b in (("key_bits", self.key_bits), ("value_bits", self.value_bits)):
            bad = ~torch.isin(b, torch.tensor(ALLOWED_BITS))
            if bad.any():
                raise ValueError(f"{name} has values outside {ALLOWED_BITS}: {b[bad].unique().tolist()}")
        if not torch.equal(self.key_bits == 0, self.value_bits == 0):
            raise ValueError("eviction must be joint: key_bits==0 exactly where value_bits==0")
        kept = (~self.evicted).sum(dim=-1)
        if not torch.all(kept == kept.flatten()[0]):
            raise ValueError(
                "every (layer, kv_head) must keep the same number of tokens "
                f"(got min {int(kept.min())}, max {int(kept.max())})"
            )
        if int(kept.flatten()[0]) == 0:
            raise ValueError("plan evicts every token")


@torch.no_grad()
def apply_plan(cache, plan: TierPlan, qcfg: QuantConfig | None = None):
    """Apply ``plan`` to ``cache`` in place and return it.

    Order: quantize first (within each tier, tokens are taken in their
    original order, so key groups are runs of same-tier tokens), then evict
    by gathering the kept tokens of each head.
    """
    qcfg = qcfg or QuantConfig()
    L = num_layers(cache)
    k0, _ = get_layer_kv(cache, 0)
    if k0.shape[0] != 1:
        raise ValueError("batch size 1 only")
    plan.validate(seq_len=k0.shape[-2])
    if plan.shape[0] != L or plan.shape[1] != k0.shape[1]:
        raise ValueError(f"plan shape {plan.shape} vs cache layers={L}, kv_heads={k0.shape[1]}")

    kb_all = plan.key_bits.to(k0.device)
    vb_all = plan.value_bits.to(k0.device)
    any_evict = bool(plan.evicted.any())

    for layer in range(L):
        k, v = get_layer_kv(cache, layer)
        k = k.clone()
        v = v.clone()
        for h in range(k.shape[1]):
            for bits_t, tensor, fq, gs in (
                (kb_all[layer, h], k, fake_quant_keys, qcfg.key_group_size),
                (vb_all[layer, h], v, fake_quant_values, qcfg.value_group_size),
            ):
                for bits in bits_t.unique().tolist():
                    if bits in (0, FULL):
                        continue
                    idx = torch.nonzero(bits_t == bits, as_tuple=False).squeeze(-1)
                    tensor[0, h, idx] = fq(tensor[0, h, idx], int(bits), gs)
        if any_evict:
            keep = ~(kb_all[layer] == 0)  # [H, T]
            n_keep = int(keep[0].sum())
            # Stable sort puts kept tokens first, in original order.
            order = torch.argsort((~keep).to(torch.int8), dim=-1, stable=True)[:, :n_keep]  # [H, n_keep]
            gather_idx = order[None, :, :, None].expand(1, -1, -1, k.shape[-1])
            k = torch.gather(k, 2, gather_idx)
            v = torch.gather(v, 2, gather_idx)
        set_layer_kv(cache, layer, k, v)
    return cache
