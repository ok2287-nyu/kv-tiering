"""Direct access to the Hugging Face KV cache, and a decode loop that is
correct after tokens have been removed from the cache.

Design choice (see docs/architecture.md): we compress **between prefill and
decoding** by editing cache tensors. We never patch the attention forward
pass for compression, which keeps us insulated from transformers internals.

Pinned against transformers 5.23.0, where ``DynamicCache.layers[i]`` is a
``DynamicLayer`` holding ``.keys`` and ``.values`` of shape
``[batch, num_kv_heads, seq_len, head_dim]``. Keys are stored **after** RoPE.

Positions after eviction
------------------------
Each cached key already has its rotary position baked in, so removing tokens
needs no recomputation. But Hugging Face infers the position of the next token
from the cache length when ``position_ids`` is not given. After evicting ``e``
tokens from a prompt of length ``T`` the cache length is ``T - e``, and the
default would rotate the next query/key as position ``T - e`` instead of
``T``. So every decode step passes ``position_ids`` = the true position.

In transformers 5.23 that is sufficient: ``position_ids`` drives only RoPE,
while the attention mask is sized from the *physical* cache
(``Cache.get_mask_sizes``), so the new token attends to every cached entry.
(Earlier 4.x versions also needed a separate ``cache_position`` argument;
5.x removed it.) Verification test 4 checks this, including that the wrong
position produces different logits.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


def num_layers(cache) -> int:
    return len(cache.layers)


def get_layer_kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    lyr = cache.layers[layer]
    return lyr.keys, lyr.values


def set_layer_kv(cache, layer: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    if keys.shape != values.shape:
        raise ValueError(f"keys {tuple(keys.shape)} and values {tuple(values.shape)} differ")
    lyr = cache.layers[layer]
    lyr.keys = keys.contiguous()
    lyr.values = values.contiguous()


def cache_len(cache) -> int:
    """Physical number of cached tokens (layer 0; all layers are kept equal)."""
    return cache.layers[0].keys.shape[-2]


def clone_cache(cache):
    return copy.deepcopy(cache)


@dataclass
class PrefillResult:
    cache: object
    last_logits: torch.Tensor  # [vocab], float32, logits for the token after the prompt
    prompt_len: int


@torch.no_grad()
def prefill(model, input_ids: torch.Tensor) -> PrefillResult:
    """Run the prompt once and return the cache plus next-token logits."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[0] != 1:
        raise ValueError("batch size 1 only (batched serving is out of scope)")
    out = model(input_ids=input_ids, use_cache=True)
    return PrefillResult(
        cache=out.past_key_values,
        last_logits=out.logits[0, -1].float(),
        prompt_len=input_ids.shape[1],
    )


@torch.no_grad()
def decode_step(model, token_id: int, cache, position: int) -> torch.Tensor:
    """Feed one token at true sequence position ``position``; return its logits.

    The cache is updated in place (the new token's K/V are appended).
    """
    device = model.device
    out = model(
        input_ids=torch.tensor([[token_id]], device=device),
        past_key_values=cache,
        position_ids=torch.tensor([[position]], device=device),
        use_cache=True,
    )
    return out.logits[0, -1].float()


@dataclass
class SplitPrefill:
    """Prefill of all prompt tokens but the last.

    Evaluation protocol: if we prefilled the whole prompt and compressed
    afterwards, the first answer token would come from logits computed with
    the *full* cache, and compression could not affect it. For a needle test
    whose answer is a single token that would make every method look perfect.
    So we prefill ``prompt[:-1]``, compress that cache, and then feed the last
    prompt token through the compressed cache. Every generated token,
    including the first, reads the compressed cache.
    """

    cache: object
    last_token: int  # the held-back final prompt token
    last_position: int  # its true position (= prompt_len - 1)


@torch.no_grad()
def prefill_split(model, input_ids: torch.Tensor) -> SplitPrefill:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[1] < 2:
        raise ValueError("prompt must have at least 2 tokens")
    res = prefill(model, input_ids[:, :-1])
    return SplitPrefill(
        cache=res.cache,
        last_token=int(input_ids[0, -1]),
        last_position=input_ids.shape[1] - 1,
    )


@torch.no_grad()
def greedy_generate(
    model,
    cache,
    next_token: int,
    next_position: int,
    max_new_tokens: int,
    eos_token_ids: set[int] | None = None,
) -> list[int]:
    """Greedy decoding from a (possibly compressed) cache.

    Starts by feeding ``next_token`` (normally the held-back last prompt
    token from :func:`prefill_split`) at true position ``next_position``.
    Returns the generated token ids (not including ``next_token``).
    """
    eos_token_ids = eos_token_ids or set()
    tokens: list[int] = []
    tok, pos = next_token, next_position
    for _ in range(max_new_tokens):
        logits = decode_step(model, tok, cache, pos)
        tok = int(torch.argmax(logits))
        pos += 1
        tokens.append(tok)
        if tok in eos_token_ids:
            break
    return tokens
