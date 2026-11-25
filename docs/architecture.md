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
| `src/kvtier/importance.py` | `kvt_sdpa` attention: stock SDPA output + recorded window / cumulative attention mass |
| `src/kvtier/policies.py` | baselines (full, uniform, StreamingLLM, H2O, SnapKV-like) and the tiered allocator (ours) |
| `src/kvtier/needle.py` | needle prompts, one shared prefill per trial, every method on a fresh cache copy |
| `src/kvtier/perplexity.py` | continuation NLL through a compressed prefix cache (one chunk, true positions) |
| `experiments/needle.py`, `summarize.py` | sweep CLI (resumable JSONL) and bootstrap-CI tables |

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

**Importance without touching the attention code path.** We register an
attention function `kvt_sdpa` through transformers' public
`AttentionInterface`. It returns exactly what SDPA returns (a test checks
bit-identical logits). During a recorded prefill it also recomputes
softmax(q·Kᵀ) in float32 for the last W queries (window score) and,
for H2O, for all queries in chunks (cumulative score). For a KV head, the
score is summed over its GQA group. Tests compare against the eager
implementation's attention weights (~1e-7 on the tiny model).

**Baselines at equal memory.** The budget is the accounted memory ratio.
Eviction methods keep ⌊r·T⌋ tokens per head. Uniform int4/int2 have fixed
ratios (31.25% / 18.75%). `window_evict` uses the same scores and pooling
as `tiered`, so window_evict vs tiered isolates the effect of mixed
precision from the effect of better scoring.

**Tiered allocator.** Distortion model s^p·ε(b) with
ε(b) = 1/(12(2^b−1)²) for quantization, ε(0) = 1/12 for eviction, p = 2.
Each head's scores are normalised to sum to 1. Upgrades are sorted by
distortion removed per byte (the ratio shrinks for each further upgrade, so
a global sort is valid). The kept count K is chosen by grid search, and the
exact byte count is re-checked and tightened. Sinks and the observation
window stay at full precision. The ε constants are a modelling assumption,
so p and the tier set are ablation knobs.

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
