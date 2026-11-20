"""Verification test 3: memory accounting matches a hand computation."""

import torch

from kvtier.memory import full_bytes_per_token, plan_bytes
from kvtier.model import ModelDims
from kvtier.tiers import QuantConfig, TierPlan


def test_qwen25_15b_full_bytes_per_token():
    # Qwen2.5-1.5B: 28 layers, 12 query heads, 2 KV heads, head_dim 128.
    # Hand: 2 (K and V) * 28 * 2 * 128 * 2 bytes = 28,672 B = 28 KiB per token.
    dims = ModelDims(num_layers=28, num_q_heads=12, num_kv_heads=2, head_dim=128)
    assert full_bytes_per_token(dims) == 28_672
    # At 32k tokens: 28 KiB * 32768 = 896 MiB.
    assert full_bytes_per_token(dims) * 32_768 == 896 * 2**20


def test_full_plan_equals_uncompressed():
    dims = ModelDims(num_layers=3, num_q_heads=4, num_kv_heads=2, head_dim=16)
    plan = TierPlan.uniform(dims, seq_len=50, bits=16)
    r = plan_bytes(plan, dims.head_dim)
    assert r.total_bytes == 50 * full_bytes_per_token(dims)
    assert r.stat_bytes == 0
    assert r.ratio == 1.0


def test_uniform_int4_hand_computed():
    # One layer, one head, D=64, T=64, groups of 32.
    #   data:        64 tok * 64 ch * 4 bit / 8 * 2 (K,V)          = 4096
    #   key stats:   ceil(64/32)=2 groups * 64 ch * 2 (scale,zero) * 2 B = 512
    #   value stats: 64 tok * ceil(64/32)=2 groups * 2 * 2 B          = 512
    #   total 5120; full = 64 * 64 * 2 B * 2 = 16384; ratio 0.3125
    dims = ModelDims(num_layers=1, num_q_heads=1, num_kv_heads=1, head_dim=64)
    plan = TierPlan.uniform(dims, seq_len=64, bits=4)
    r = plan_bytes(plan, 64, QuantConfig(key_group_size=32, value_group_size=32))
    assert r.data_bytes == 4096
    assert r.stat_bytes == 1024
    assert r.full_bytes == 16384
    assert r.ratio == 5120 / 16384


def test_evicting_half_halves_memory():
    dims = ModelDims(num_layers=2, num_q_heads=4, num_kv_heads=2, head_dim=16)
    bits = torch.full((2, 2, 40), 16)
    bits[..., ::2] = 0
    r = plan_bytes(TierPlan.from_bits(bits), 16)
    assert r.ratio == 0.5


def test_mixed_plan_hand_computed():
    # One layer, one head, D=8, T=10, groups of 4.
    # Tiers: 3 tokens fp16, 4 tokens 4-bit, 2 tokens 2-bit, 1 evicted.
    #   fp16:  3 * 8 * 2 B * 2          = 96
    #   4-bit: 4 * 8 * 4/8 * 2          = 32 ; key stats ceil(4/4)=1*8*2*2 = 32 ; value stats 4*2*2*2 = 32
    #   2-bit: 2 * 8 * 2/8 * 2          = 8  ; key stats ceil(2/4)=1*8*2*2 = 32 ; value stats 2*2*2*2 = 16
    #   total = 96 + 32 + 32 + 32 + 8 + 32 + 16 = 248
    bits = torch.tensor([[[16, 16, 16, 4, 4, 4, 4, 2, 2, 0]]])
    r = plan_bytes(TierPlan.from_bits(bits), 8, QuantConfig(key_group_size=4, value_group_size=4))
    assert r.data_bytes == 96 + 32 + 8
    assert r.stat_bytes == 32 + 32 + 32 + 16
    assert r.total_bytes == 248
    assert r.full_bytes == 10 * 8 * 2 * 2
