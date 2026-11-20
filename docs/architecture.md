# Architecture and design decisions

## Pipeline

```
prompt ─► prefill(prompt[:-1]) ─► KV cache ─► [importance scores] ─► allocator ─► TierPlan
                                     │                                               │
                                     └──────────────► apply_plan ◄───────────────────┘
                                                          │
                     last prompt token ─► decode loop (true positions) ─► answer
```

| Module | Role |
|---|---|
| `src/kvtier/model.py` | load model; read layer/head dims from config |
| `src/kvtier/cache.py` | read/write cache tensors; split prefill; decode with true positions |
| `src/kvtier/quant.py` | simulated asymmetric uniform quantization, KIVI grouping |
| `src/kvtier/tiers.py` | `TierPlan` (bits per layer × KV head × token) and `apply_plan` |
| `src/kvtier/memory.py` | accounted bytes for a plan (the rate axis) |
| *(next)* importance, allocator, baselines | scoring and plan construction |

## Decisions worth being able to defend

**Compress between prefill and decode, by editing tensors.** The cache is a
list of `[batch, kv_heads, tokens, head_dim]` tensors per layer. Editing them
directly touches only the cache's public `.keys/.values`, not the attention
forward pass, so a transformers upgrade is less likely to silently break us.
The cost: compression happens once, after the prompt. Tokens generated later
stay full precision. That is fine for the needle task, where the answer is
short.

**Split prefill (evaluation protocol).** We prefill `prompt[:-1]`, compress,
then feed the last prompt token through the compressed cache. If we
prefilled the whole prompt, the first answer token would be predicted from
the uncompressed cache. For single-token answers, every method would then
score perfectly.

**Positions after eviction.** Keys are cached after RoPE, so each key
already encodes its position, and evicting a token needs no recomputation.
Only the *new* token's position must be right. Hugging Face infers it from
the cache length, which shrinks after eviction. So we always pass
`position_ids` explicitly. In transformers 5.23 the attention mask is sized
from the physical cache, so nothing else is needed. Test:
`tests/test_eviction.py::test_wrong_position_is_detected`.

**Per-head eviction with equal counts.** Different KV heads may evict
different tokens, but every head in every layer must keep the same number,
because the cache tensor is rectangular. Attention is invariant to the order
of cached entries (positions live inside the keys), so each head can hold its
own token set. Tests: `test_per_head_eviction_gathers_correct_tokens`,
`test_token_order_within_cache_is_irrelevant`.

**Simulated quantization.** Quantize then dequantize in place, so tensors
stay bf16. The error is the real error; the memory saving is *accounted*
(`memory.py`), not measured. Eviction savings are real. Packed low-bit
storage was cut for time.

**Quantization grouping (KIVI).** Keys: per channel, statistics shared across
groups of 32 same-tier tokens in one head. Values: per token, in groups of 32
channels. Within a tier, tokens keep their original order, so a key group is
a run of same-tier tokens. That is not a contiguous run of the sequence.

**Memory accounting.** Counted: packed data bits, plus fp16 scale and zero
per group. Not counted: a per-token tier index. Since order doesn't matter, a
real store can keep each tier contiguous and needs only a few counts per head.

**bf16, not fp16.** Qwen2.5 activations can overflow fp16. "Full precision"
therefore means "model dtype". Accounting uses 2 bytes/element either way.
Real-model *tests* run in float32 so they can use tight tolerances. The smoke
run and experiments use bf16.

## Verification tests (all must pass before any result is reported)

| # | Claim | Test file |
|---|---|---|
| 1 | All-16-bit plan reproduces original logits exactly (`torch.equal`) | `tests/test_cache_identity.py` |
| 2 | Quantize/dequantize known inputs → known outputs; error ≤ step/2; error shrinks with bits | `tests/test_quant.py` |
| 3 | Memory accounting matches hand computation (incl. Qwen2.5-1.5B = 28 KiB/token) | `tests/test_memory.py` |
| 4 | Eviction ≡ masking those tokens in HF attention; wrong position is detected; generation continues | `tests/test_eviction.py` |

Each equivalence test has a paired *sensitivity* test showing that the
compared quantity does change when it should. On the tiny model, eviction
matches the mask reference to ~1e-7, while the sensitivity tests move logits
by 0.04–0.06.
