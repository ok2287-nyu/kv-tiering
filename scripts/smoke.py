"""End-to-end smoke run on the real model (bf16): load, prefill a document
with a fact buried in the middle, then answer with

  (a) the full cache,
  (b) the full cache passed through an all-16-bit plan (must match (a)),
  (c) a StreamingLLM-style cache (4 sink tokens + recent window, middle evicted),
  (d) uniform simulated int4.

This is a plumbing check, not a result: one prompt, no statistics.

    python scripts/smoke.py [--context-tokens 4000]
"""

from __future__ import annotations

import argparse
import time

import torch

from kvtier import ModelDims, TierPlan, apply_plan, clone_cache, full_bytes_per_token, greedy_generate, load_model, plan_bytes, prefill_split
from kvtier.cache import cache_len

FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. "
)
NEEDLE = "The secret passcode for the blue door is 7392. "
QUESTION = "What is the secret passcode for the blue door? Answer with the number only."


def build_prompt(tok, context_tokens: int, depth: float = 0.5) -> torch.Tensor:
    filler_ids = tok(FILLER, add_special_tokens=False).input_ids
    reps = max(1, context_tokens // len(filler_ids))
    chunks = [FILLER] * reps
    chunks.insert(int(reps * depth), NEEDLE)
    messages = [{"role": "user", "content": "".join(chunks) + "\n\n" + QUESTION}]
    # Render to text first, then tokenize: stable across transformers versions.
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids


def eos_ids(model, tok) -> set[int]:
    ids = {tok.eos_token_id}
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(gen_eos)
    return {i for i in ids if i is not None}


def run(model, tok, sp, cache, label, plan=None, eos=None):
    t0 = time.time()
    out = greedy_generate(model, cache, sp.last_token, sp.last_position, max_new_tokens=12, eos_token_ids=eos if eos is not None else {tok.eos_token_id})
    text = tok.decode(out, skip_special_tokens=True).strip()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem = ""
    if plan is not None:
        r = plan_bytes(plan, ModelDims.from_config(model.config).head_dim)
        mem = f"accounted {r.ratio:6.1%} of full"
    print(f"{label:<28} cache_len={cache_len(cache):>6}  {mem:<24} -> {text!r}  ({time.time() - t0:.2f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-tokens", type=int, default=4000)
    ap.add_argument("--window", type=int, default=512, help="recent tokens kept by the StreamingLLM-style plan")
    args = ap.parse_args()

    model, tok = load_model()
    dims = ModelDims.from_config(model.config)
    print(f"model={model.config._name_or_path} dtype={model.dtype} device={model.device}")
    print(f"dims: {dims}  (GQA group size {dims.group_size})")
    print(f"full KV cache: {full_bytes_per_token(dims)} bytes/token = {full_bytes_per_token(dims) / 1024:.1f} KiB/token")

    eos = eos_ids(model, tok)
    ids = build_prompt(tok, args.context_tokens).to(model.device)
    sp = prefill_split(model, ids)
    T = cache_len(sp.cache)
    print(f"prompt tokens: {ids.shape[1]} (cache after split prefill: {T})")
    print()

    ref = run(model, tok, sp, clone_cache(sp.cache), "full cache", eos=eos)

    plan16 = TierPlan.uniform(dims, T, 16)
    c = apply_plan(clone_cache(sp.cache), plan16)
    same = run(model, tok, sp, c, "all-16-bit plan", plan16, eos=eos)
    assert same == ref, "all-16-bit plan changed the output: cache surgery is broken"

    bits = torch.zeros((dims.num_layers, dims.num_kv_heads, T), dtype=torch.long)
    bits[..., :4] = 16
    bits[..., max(4, T - args.window) :] = 16
    plan_s = TierPlan.from_bits(bits)
    run(model, tok, sp, apply_plan(clone_cache(sp.cache), plan_s), f"sink4+recent{args.window}", plan_s, eos=eos)

    plan4 = TierPlan.uniform(dims, T, 4)
    run(model, tok, sp, apply_plan(clone_cache(sp.cache), plan4), "uniform int4 (simulated)", plan4, eos=eos)

    if torch.cuda.is_available():
        print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("smoke OK")


if __name__ == "__main__":
    main()
