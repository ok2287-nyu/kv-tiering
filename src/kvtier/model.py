"""Model loading and the cache-relevant dimensions read from the model config.

Nothing about the architecture is hardcoded: layer count, KV-head count and
head dimension all come from ``model.config``, so the same code runs on the
tiny random test model and on Qwen2.5-1.5B/7B.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class ModelDims:
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def group_size(self) -> int:
        """Query heads per KV head (grouped-query attention)."""
        return self.num_q_heads // self.num_kv_heads

    @classmethod
    def from_config(cls, config) -> "ModelDims":
        n_q = config.num_attention_heads
        n_kv = getattr(config, "num_key_value_heads", None) or n_q
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // n_q
        return cls(
            num_layers=config.num_hidden_layers,
            num_q_heads=n_q,
            num_kv_heads=n_kv,
            head_dim=head_dim,
        )


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # bf16, not fp16: Qwen2.5 activations can overflow fp16's range. The
    # "full-precision" tier therefore means "stored in the model dtype" (bf16
    # on GPU). Memory accounting uses 2 bytes/element either way.
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model(name: str | None = None, device: str | None = None, dtype: torch.dtype | None = None):
    """Load a causal LM and its tokenizer in eval mode.

    The attention implementation is ``kvt_sdpa``: stock SDPA for the output
    (so results are identical to ``sdpa``), plus optional recording of
    attention mass for importance scores (see ``importance.py``).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .importance import ATTN_IMPL

    name = name or os.environ.get("KVT_MODEL", DEFAULT_MODEL)
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=ATTN_IMPL)
    model.to(device).eval()
    return model, tok
