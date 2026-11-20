"""Verification test 2: quantize/dequantize on known inputs gives known
outputs, and error shrinks as bits increase."""

import pytest
import torch

from kvtier.quant import dequantize, fake_quant, fake_quant_keys, fake_quant_values, quantize


def test_exact_grid_roundtrip():
    # Values already on the 2-bit grid of [0, 3] come back exactly.
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    q = quantize(x, bits=2, axis=1, group_size=4)
    assert q.codes.tolist() == [[0, 1, 2, 3]]
    assert torch.equal(dequantize(q), x)


def test_known_rounding():
    # Range [0, 10] with 2 bits: scale = 10/3, grid {0, 3.33, 6.67, 10}.
    x = torch.tensor([[0.0, 3.4, 6.6, 10.0]])
    q = quantize(x, bits=2, axis=1, group_size=4)
    assert q.codes.tolist() == [[0, 1, 2, 3]]
    expected = torch.tensor([[0.0, 10 / 3, 20 / 3, 10.0]])
    assert torch.allclose(dequantize(q), expected, atol=1e-6)


def test_constant_group_is_exact():
    x = torch.full((4, 8), 2.5)
    assert torch.equal(fake_quant(x, 2, axis=0, group_size=4), x)


def test_partial_last_group():
    # 33 tokens with group 32: one full group and one group of 1 (exact).
    torch.manual_seed(0)
    x = torch.randn(33, 8)
    y = fake_quant_keys(x, 4, group_size=32)
    assert y.shape == x.shape
    assert torch.equal(y[32], x[32])


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_error_bounded_by_half_step(bits):
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    q = quantize(x, bits, axis=0, group_size=16)
    err = (dequantize(q) - x).abs()
    # scale has shape [n_groups, 16]; expand back to one row per token.
    half_step = q.scale.repeat_interleave(16, dim=0)[: x.shape[0]] / 2
    assert torch.all(err <= half_step + 1e-6)


def test_error_shrinks_with_bits():
    torch.manual_seed(0)
    x = torch.randn(256, 64)
    errs = [(fake_quant(x, b, axis=0, group_size=32) - x).pow(2).mean().item() for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2] > 0
    # Each extra bit roughly halves the step, so 2 extra bits ~ 16x less MSE.
    assert errs[0] / errs[1] > 8
    assert torch.equal(fake_quant(x, 16, axis=0, group_size=32), x)


def test_per_channel_key_grouping_isolates_outlier_channel():
    # Why KIVI quantizes keys per channel: one outlier channel should not
    # destroy the resolution of the others.
    torch.manual_seed(0)
    k = torch.randn(64, 16)
    k[:, 0] *= 100.0
    per_channel = fake_quant_keys(k, 2, group_size=32)
    per_token = fake_quant_values(k, 2, group_size=16)  # stats shared across channels
    err_ch = (per_channel - k)[:, 1:].abs().mean()
    err_tok = (per_token - k)[:, 1:].abs().mean()
    assert err_ch * 3 < err_tok  # ~4.9x on this seed


def test_dtype_preserved():
    x = torch.randn(32, 16, dtype=torch.bfloat16)
    assert fake_quant(x, 4, axis=0, group_size=32).dtype == torch.bfloat16
