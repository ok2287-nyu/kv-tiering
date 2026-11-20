"""Simulated (fake) asymmetric uniform quantization for KV-cache tensors.

"Simulated" means we quantize to integer codes and immediately dequantize back
to floating point. The numerical error is exactly what real low-bit storage
would introduce, but the tensors stay in the model dtype, so no memory is
actually saved. Memory is *accounted* separately in ``memory.py``. This is the
labelling the proposal commits to: quantization savings are computed, eviction
savings are real.

Grouping follows KIVI (Liu et al., 2024):

* keys are quantized **per channel**: one (scale, zero) pair per channel,
  shared by a group of ``group_size`` consecutive tokens. Key outliers live in
  a few fixed channels, so per-channel statistics keep one large channel from
  wrecking the resolution of all the others.
* values are quantized **per token**: one (scale, zero) pair per token, shared
  by a group of ``group_size`` channels. Value errors enter the attention
  output linearly, weighted by that token's attention, so per-token error
  control is what matters.

Both are the same operation: share statistics along one axis in groups. The
``axis`` argument names the axis along which statistics are shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Quantized:
    """Integer codes plus the per-group statistics needed to dequantize."""

    codes: torch.Tensor  # same shape as the input, integer values in [0, 2^bits - 1]
    scale: torch.Tensor  # one entry per group (broadcast shape, see quantize())
    zero: torch.Tensor  # group minimum; dequant = codes * scale + zero
    bits: int
    axis: int
    group_size: int


def _split_groups(x: torch.Tensor, axis: int, group_size: int) -> list[torch.Tensor]:
    """Split ``x`` along ``axis`` into chunks of ``group_size`` (last may be shorter)."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return list(torch.split(x, group_size, dim=axis))


def quantize(x: torch.Tensor, bits: int, axis: int, group_size: int) -> Quantized:
    """Asymmetric uniform quantization with statistics shared along ``axis``.

    For each group (``group_size`` consecutive entries along ``axis``, for every
    index of the other axes) we take min and max, set
    ``scale = (max - min) / (2^bits - 1)`` and ``code = round((x - min) / scale)``.
    The worst-case absolute error per element is ``scale / 2``.

    Computation is done in float32 regardless of input dtype, so the only
    error is quantization error, not bf16 arithmetic.
    """
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    levels = 2**bits - 1
    xf = x.float()
    codes, scales, zeros = [], [], []
    for chunk in _split_groups(xf, axis, group_size):
        mn = chunk.amin(dim=axis, keepdim=True)
        mx = chunk.amax(dim=axis, keepdim=True)
        scale = (mx - mn) / levels
        # A constant group has zero range; any scale reconstructs it exactly
        # because every code is 0 and dequant returns mn. Use 1 to avoid 0/0.
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.clamp(torch.round((chunk - mn) / safe), 0, levels)
        codes.append(q)
        scales.append(safe)
        zeros.append(mn)
    return Quantized(
        codes=torch.cat(codes, dim=axis).to(torch.int16),
        scale=torch.cat(scales, dim=axis),
        zero=torch.cat(zeros, dim=axis),
        bits=bits,
        axis=axis,
        group_size=group_size,
    )


def dequantize(q: Quantized, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of :func:`quantize`: ``codes * scale + zero`` per group."""
    out = []
    code_chunks = _split_groups(q.codes.float(), q.axis, q.group_size)
    # scale/zero have size 1 per group along the axis, so chunk them by 1.
    scale_chunks = torch.split(q.scale, 1, dim=q.axis)
    zero_chunks = torch.split(q.zero, 1, dim=q.axis)
    for c, s, z in zip(code_chunks, scale_chunks, zero_chunks):
        out.append(c * s + z)
    return torch.cat(out, dim=q.axis).to(dtype)


def fake_quant(x: torch.Tensor, bits: int, axis: int, group_size: int) -> torch.Tensor:
    """Quantize then dequantize; returns a tensor of ``x``'s dtype and shape."""
    if bits >= 16:
        return x
    return dequantize(quantize(x, bits, axis, group_size), dtype=x.dtype)


def fake_quant_keys(k: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """KIVI-style key quantization. ``k`` is ``[n_tokens, head_dim]``.

    Statistics are shared along the token axis (axis 0), i.e. per channel,
    in groups of ``group_size`` tokens.
    """
    return fake_quant(k, bits, axis=0, group_size=group_size)


def fake_quant_values(v: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """KIVI-style value quantization. ``v`` is ``[n_tokens, head_dim]``.

    Statistics are shared along the channel axis (axis 1), i.e. per token,
    in groups of ``group_size`` channels.
    """
    return fake_quant(v, bits, axis=1, group_size=group_size)
