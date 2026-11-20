"""Shared fixtures.

By default the tests use a tiny randomly initialised Qwen2 model (2 layers,
4 query heads, 2 KV heads, head_dim 16) so they run on a laptop CPU in
seconds. The code paths are identical to the real model: same architecture
class, same cache class, same attention implementation.

Set ``KVT_REAL_MODEL=1`` to run the same tests against the real model
(``KVT_MODEL``, default Qwen/Qwen2.5-1.5B-Instruct). Real-model tests load it
in **float32** so equivalence checks can use tight tolerances; bf16 numerics
are a separate question from whether the cache surgery is correct.
"""

from __future__ import annotations

import os

import pytest
import torch

from kvtier.model import DEFAULT_MODEL, ModelDims, pick_device

REAL = os.environ.get("KVT_REAL_MODEL", "0") == "1"


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    # Random init gives near-uniform attention, which makes eviction tests
    # weak (removing a token barely matters). Scale up q/k projections so
    # attention is peaked and every test below is sensitive to mistakes.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.q_proj.weight.mul_(4.0)
            layer.self_attn.k_proj.weight.mul_(4.0)
    return model.eval()


@pytest.fixture(scope="session")
def model():
    if REAL:
        from transformers import AutoModelForCausalLM

        name = os.environ.get("KVT_MODEL", DEFAULT_MODEL)
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa")
        return m.to(pick_device()).eval()
    return _tiny_model()


@pytest.fixture(scope="session")
def dims(model) -> ModelDims:
    return ModelDims.from_config(model.config)


@pytest.fixture(scope="session")
def prompt_ids(model) -> torch.Tensor:
    n = 512 if REAL else 48
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(10, min(model.config.vocab_size, 30000), (1, n), generator=g)
    return ids.to(model.device)


@pytest.fixture(scope="session")
def atol() -> float:
    # float32 everywhere. SDPA with a mask vs. SDPA on a physically smaller
    # cache take different kernel paths, so allow tiny float differences.
    return 1e-3 if REAL else 1e-4
