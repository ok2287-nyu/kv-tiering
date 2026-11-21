"""Token importance from prompt attention.

SDPA/FlashAttention never materialise the attention weights, so we register
our own attention function, ``"kvt_sdpa"``. It always computes the layer
output with the stock SDPA kernel, so outputs are unchanged. While a recorder
is active and the call is a prefill (more than one query), it *additionally*
recomputes softmax(q·Kᵀ·scale) for the queries we care about and accumulates
importance per (layer, KV head, token).

Two scores are recorded:

* **window** (ours, SnapKV-style): attention received from the last ``W``
  prompt queries. Those queries include the user's question, so they attend
  to whatever the question is about. Every token is scored by the same ``W``
  queries, so position doesn't bias the score.
* **cumulative** (H2O): attention received summed over *all* prompt queries.
  Under causal masking, token ``j`` is seen by ``T - j`` queries, so early
  tokens collect more mass just for being early. That's faithful to H2O and
  is one of the things the comparison tests.

GQA: query head ``h`` reads KV head ``h // g`` (``g`` = group size), matching
``repeat_kv``. A KV entry's importance is the **sum** over its ``g`` query
heads, because every one of them reads the same stored entry.

The recomputation runs in float32. The window score costs O(W·T) per head;
the cumulative score costs O(T²) per head and is done in query chunks to
bound memory.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

ATTN_IMPL = "kvt_sdpa"

_ACTIVE: "_Recorder | None" = None


@dataclass
class ImportanceScores:
    window: torch.Tensor  # float32 [L, H_kv, T]
    cumulative: torch.Tensor | None  # float32 [L, H_kv, T] or None
    window_size: int


class _Recorder:
    def __init__(self, num_layers: int, window: int, cumulative: bool, chunk_elems: int):
        self.window = window
        self.cumulative = cumulative
        self.chunk_elems = chunk_elems
        self.win: list[torch.Tensor | None] = [None] * num_layers
        self.cum: list[torch.Tensor | None] = [None] * num_layers

    @torch.no_grad()
    def observe(self, layer: int, query: torch.Tensor, key: torch.Tensor, scaling: float) -> None:
        # query [1, Hq, Tq, D] (post-RoPE), key [1, Hkv, Tk, D] (post-RoPE, incl. any past)
        _, hq, tq, d = query.shape
        hkv, tk = key.shape[1], key.shape[2]
        g = hq // hkv
        offset = tk - tq  # absolute position of query row 0
        k = key[0].float()  # [Hkv, Tk, D]
        q = query[0].float().view(hkv, g, tq, d)  # query head h -> kv head h // g

        def mass(rows: slice) -> torch.Tensor:
            qs = q[:, :, rows]  # [Hkv, g, R, D]
            r = qs.shape[2]
            logits = torch.einsum("hgrd,htd->hgrt", qs, k) * scaling
            qpos = offset + torch.arange(rows.start, rows.start + r, device=k.device)
            kpos = torch.arange(tk, device=k.device)
            logits.masked_fill_(kpos[None, :] > qpos[:, None], float("-inf"))
            w = torch.softmax(logits, dim=-1)
            return w.sum(dim=(1, 2))  # [Hkv, Tk]

        w_rows = min(self.window, tq)
        self.win[layer] = mass(slice(tq - w_rows, tq))
        if self.cumulative:
            chunk = max(1, self.chunk_elems // (hq * tk))
            acc = torch.zeros(hkv, tk, device=k.device)
            for s in range(0, tq, chunk):
                acc += mass(slice(s, min(s + chunk, tq)))
            self.cum[layer] = acc

    def result(self) -> ImportanceScores:
        if any(x is None for x in self.win):
            raise RuntimeError("no attention recorded for some layers; was the model loaded with attn_implementation='kvt_sdpa'?")
        win = torch.stack([x.cpu() for x in self.win])
        cum = torch.stack([x.cpu() for x in self.cum]) if self.cumulative else None
        return ImportanceScores(window=win, cumulative=cum, window_size=self.window)


def _kvt_sdpa(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    out = sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    rec = _ACTIVE
    if rec is not None and query.shape[2] > 1:
        scaling = kwargs.get("scaling")
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        rec.observe(module.layer_idx, query, key, scaling)
    return out


def register() -> None:
    """Register ``kvt_sdpa`` (attention + SDPA-style mask). Idempotent."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register(ATTN_IMPL, _kvt_sdpa)
    AttentionMaskInterface.register(ATTN_IMPL, sdpa_mask)


register()


@contextlib.contextmanager
def record_attention(model, window: int = 64, cumulative: bool = False, chunk_elems: int = 2**27):
    """Record importance during the forward passes inside the block.

    Usage::

        with record_attention(model, window=64, cumulative=True) as rec:
            sp = prefill_split(model, ids)
        scores = rec.result()

    Only prefill calls (more than one query token) are recorded; decode steps
    inside the block are ignored.
    """
    global _ACTIVE
    if model.config._attn_implementation != ATTN_IMPL:
        raise RuntimeError(f"model must use attn_implementation={ATTN_IMPL!r} (got {model.config._attn_implementation!r})")
    if _ACTIVE is not None:
        raise RuntimeError("record_attention is not reentrant")
    rec = _Recorder(model.config.num_hidden_layers, window, cumulative, chunk_elems)
    _ACTIVE = rec
    try:
        yield rec
    finally:
        _ACTIVE = None
