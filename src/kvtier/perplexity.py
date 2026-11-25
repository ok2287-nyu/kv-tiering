"""Perplexity of a continuation, read through a compressed cache.

Why this evaluation exists: in the needle task the question sits at the end
of the prompt, so attention from the last prompt tokens points straight at
the answer. That favours every method that scores tokens by that attention
(ours and window_evict). Here there is no question: the model reads natural
text and must predict what comes next. If prompt-attention importance only
works because the question points at the needle, it shows up here.

Protocol (one trial):

    text = prefix (P tokens) + continuation (C tokens)
    1. prefill prefix[:-1] with importance recording
    2. compress a copy of the cache with each method
    3. feed [prefix[-1]] + continuation[:-1] as ONE chunk through the
       compressed cache, at their true positions, and take the NLL of each
       continuation token.

So every continuation token, including the first, is predicted from the
compressed prefix cache. Within the chunk, tokens also attend to earlier
chunk tokens at full precision (they're appended uncompressed), as in real
decoding after compression. The NLL of the first continuation tokens is
therefore the most sensitive to compression; we report it separately.
"""

from __future__ import annotations

import torch

from .cache import cache_len, clone_cache, prefill_split
from .importance import record_attention
from .memory import plan_bytes
from .model import ModelDims
from .needle import MethodSpec, make_plan
from .tiers import QuantConfig, apply_plan


@torch.no_grad()
def chunk_nll(model, cache, tokens: torch.Tensor, start_position: int, targets: torch.Tensor) -> torch.Tensor:
    """Per-token NLL (float32, CPU) of ``targets`` given ``tokens`` fed after ``cache``.

    ``tokens`` and ``targets`` are [1, C]; ``targets[i]`` is the token that
    follows ``tokens[i]``. Positions are passed explicitly (true positions),
    so this is correct after eviction; the attention mask is sized from the
    physical cache, as in decode_step.
    """
    C = tokens.shape[1]
    pos = torch.arange(start_position, start_position + C, device=model.device)[None]
    out = model(input_ids=tokens.to(model.device), past_key_values=cache, position_ids=pos, use_cache=True)
    logp = torch.log_softmax(out.logits[0].float(), dim=-1)
    return -logp.gather(-1, targets[0].to(model.device)[:, None]).squeeze(-1).cpu()


@torch.no_grad()
def run_ppl_trial(
    model,
    ids: torch.Tensor,
    cont_len: int,
    specs: list[MethodSpec],
    window: int = 64,
    qcfg: QuantConfig | None = None,
    head_tokens: int = 32,
) -> list[dict]:
    """Compress the prefix cache with every method and score the continuation."""
    qcfg = qcfg or QuantConfig()
    dims = ModelDims.from_config(model.config)
    ids = ids.to(model.device)
    prefix, cont = ids[:, :-cont_len], ids[:, -cont_len:]
    need_cum = any(s.name == "h2o" or s.params.get("scores") == "cumulative" for s in specs)
    with record_attention(model, window=window, cumulative=need_cum) as rec:
        sp = prefill_split(model, prefix)
    scores = rec.result()
    T = cache_len(sp.cache)
    chunk = torch.cat([prefix[:, -1:], cont[:, :-1]], dim=1)
    rows = []
    for spec in specs:
        plan, extra = make_plan(spec, dims, T, scores, window, qcfg)
        cache = apply_plan(clone_cache(sp.cache), plan, qcfg)
        nll = chunk_nll(model, cache, chunk, sp.last_position, cont)
        del cache
        rows.append(
            {
                "method": spec.label,
                "budget": spec.budget,
                "ratio": plan_bytes(plan, dims.head_dim, qcfg).ratio,
                "nll": float(nll.mean()),
                "nll_head": float(nll[:head_tokens].mean()),
                "n_tokens": int(nll.numel()),
                **extra,
            }
        )
    return rows
