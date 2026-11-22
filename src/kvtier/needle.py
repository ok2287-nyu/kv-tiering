"""Needle-in-a-haystack: build prompts, run every method on one shared prefill.

Task (RULER-style key-value retrieval): a sentence "The special magic number
for <city> is <7 digits>." is inserted at relative depth ``d`` into natural
text of ``L`` tokens. The model is asked for the number. A trial is correct
if the exact 7-digit string appears in the answer. Random keys and values
rule out answering from memory.

One trial = one prefill (with importance recorded) and then, for every
method x budget, a fresh copy of the cache, compressed and decoded. All
methods therefore see the identical prompt and the identical uncompressed
cache.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import torch

from .cache import cache_len, clone_cache, greedy_generate, prefill_split
from .importance import record_attention
from .memory import plan_bytes
from .model import ModelDims
from .policies import plan_full, plan_h2o, plan_streaming, plan_tiered, plan_uniform, plan_window_evict
from .tiers import QuantConfig, TierPlan, apply_plan

KEYS = [
    "Oslo", "Lima", "Hanoi", "Nairobi", "Quito", "Tbilisi", "Reykjavik", "Accra", "Riga", "Bogota",
    "Dakar", "Manila", "Tallinn", "Havana", "Kyoto", "Muscat", "Porto", "Zagreb", "Lusaka", "Hobart",
]

PLACEHOLDER = "<<HAYSTACK>>"


@dataclass
class Needle:
    key: str
    value: str

    @property
    def text(self) -> str:
        return f" The special magic number for {self.key} is {self.value}. "

    @property
    def question(self) -> str:
        return f"What is the special magic number for {self.key} mentioned in the text above? Answer with the number only."


def make_needle(rng: random.Random) -> Needle:
    return Needle(key=rng.choice(KEYS), value=str(rng.randint(1_000_000, 9_999_999)))


# --------------------------------------------------------------------------- haystack

SYNTHETIC_SENTENCES = [
    "The river bends twice before reaching the old mill.",
    "Several committees met to discuss the annual budget.",
    "A light rain fell across the northern valleys overnight.",
    "The library extended its opening hours during the exam period.",
    "Engineers inspected the bridge after the spring floods.",
    "The orchestra rehearsed the second movement until late evening.",
    "Farmers reported an early harvest in the eastern provinces.",
    "The museum acquired a collection of early printed maps.",
]


def _clean_wikitext(text: str) -> str:
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r"\n\s*=+[^=\n]*=+\s*\n", "\n", text)  # section headings
    return re.sub(r"[ \t]+", " ", text)


def load_haystack_text(source: str, seed: int = 0, n_chars: int = 2_000_000) -> str:
    """``wikitext`` (WikiText-103 test split), ``synthetic``, or a path to a .txt file."""
    if source == "synthetic":
        rng = random.Random(seed)
        out, n = [], 0
        while n < n_chars:
            s = rng.choice(SYNTHETIC_SENTENCES)
            out.append(s)
            n += len(s) + 1
        return " ".join(out)
    if source == "wikitext":
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        return _clean_wikitext("".join(ds["text"]))[:n_chars]
    with open(source, encoding="utf-8") as f:
        return f.read()[:n_chars]


# --------------------------------------------------------------------------- prompts


@dataclass
class Prompt:
    ids: torch.Tensor  # [1, N]
    needle: Needle
    needle_span: tuple[int, int]  # token indices [start, end) in ids
    length: int
    depth: float


def _sentence_end_after(tok, hay: list[int], pos: int, search: int = 200) -> int:
    """First index after ``pos`` that follows a token containing a period."""
    for i in range(pos, min(len(hay), pos + search)):
        if "." in tok.decode([hay[i]]):
            return i + 1
    return pos


def build_prompt(tok, hay_ids: list[int], length: int, depth: float, needle: Needle, rng: random.Random) -> Prompt:
    """Chat-formatted prompt of exactly ``length`` tokens with the needle at ``depth``."""
    msg = [{"role": "user", "content": PLACEHOLDER + "\n\n" + needle.question}]
    text = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    pre, post = text.split(PLACEHOLDER)
    pre_ids = tok(pre, add_special_tokens=False).input_ids
    post_ids = tok(post, add_special_tokens=False).input_ids
    nd_ids = tok(needle.text, add_special_tokens=False).input_ids
    n_hay = length - len(pre_ids) - len(post_ids) - len(nd_ids)
    if n_hay <= 0:
        raise ValueError(f"length {length} too short for the template")
    if len(hay_ids) < n_hay:
        raise ValueError(f"haystack has {len(hay_ids)} tokens, need {n_hay}")
    off = rng.randint(0, len(hay_ids) - n_hay)
    hay = hay_ids[off : off + n_hay]
    pos = _sentence_end_after(tok, hay, int(depth * n_hay))
    ids = pre_ids + hay[:pos] + nd_ids + hay[pos:] + post_ids
    start = len(pre_ids) + pos
    return Prompt(
        ids=torch.tensor([ids]),
        needle=needle,
        needle_span=(start, start + len(nd_ids)),
        length=len(ids),
        depth=depth,
    )


# --------------------------------------------------------------------------- methods


@dataclass
class MethodSpec:
    name: str  # full | uniform | streaming | h2o | window_evict | tiered
    budget: float | None = None  # memory ratio for budgeted methods
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.name == "uniform":
            kb = self.params.get("bits", 4)
            vb = self.params.get("value_bits", kb)
            return f"uniform-k{kb}v{vb}"
        return self.name


BUDGETED = {"streaming", "h2o", "window_evict", "tiered"}


def expand_methods(methods: list[dict], budgets: list[float]) -> list[MethodSpec]:
    """Config entries -> specs. Budgeted methods are crossed with ``budgets``."""
    out = []
    for m in methods:
        m = dict(m)
        name = m.pop("name")
        if name in BUDGETED:
            out.extend(MethodSpec(name, b, dict(m)) for b in budgets)
        else:
            out.append(MethodSpec(name, None, m))
    return out


def make_plan(spec: MethodSpec, dims: ModelDims, T: int, scores, window: int, qcfg: QuantConfig) -> tuple[TierPlan, dict]:
    p = spec.params
    if spec.name == "full":
        return plan_full(dims, T), {}
    if spec.name == "uniform":
        return plan_uniform(dims, T, p.get("bits", 4), p.get("value_bits")), {}
    if spec.name == "streaming":
        return plan_streaming(dims, T, spec.budget), {}
    if spec.name == "h2o":
        return plan_h2o(dims, T, spec.budget, scores.cumulative, p.get("recent_frac", 0.5)), {}
    if spec.name == "window_evict":
        return plan_window_evict(dims, T, spec.budget, scores.window, window, p.get("pool_kernel", 7)), {}
    if spec.name == "tiered":
        plan, info = plan_tiered(
            dims, T, spec.budget, scores.window, window, qcfg,
            tiers=tuple(p.get("tiers", (2, 4, 16))), power=p.get("power", 2.0), pool_kernel=p.get("pool_kernel", 7),
        )
        return plan, {"kept_per_head": info.kept, "tier_fractions": {str(k): v for k, v in info.tier_fractions.items()}}
    raise ValueError(f"unknown method {spec.name!r}")


def needle_stats(plan: TierPlan, span: tuple[int, int]) -> dict:
    """How the plan treated the needle tokens, averaged over layers and heads."""
    kb = plan.key_bits[..., span[0] : span[1]]
    kept = kb > 0
    frac = float(kept.float().mean())
    mean_bits = float(kb[kept].float().mean()) if kept.any() else 0.0
    return {"needle_kept_frac": frac, "needle_mean_key_bits": mean_bits}


@torch.no_grad()
def run_trial(
    model,
    tok,
    prompt: Prompt,
    specs: list[MethodSpec],
    window: int = 64,
    max_new_tokens: int = 16,
    qcfg: QuantConfig | None = None,
    eos_ids: set[int] | None = None,
) -> list[dict]:
    qcfg = qcfg or QuantConfig()
    dims = ModelDims.from_config(model.config)
    need_cum = any(s.name == "h2o" for s in specs)
    ids = prompt.ids.to(model.device)
    with record_attention(model, window=window, cumulative=need_cum) as rec:
        sp = prefill_split(model, ids)
    scores = rec.result()
    T = cache_len(sp.cache)
    rows = []
    for spec in specs:
        plan, extra = make_plan(spec, dims, T, scores, window, qcfg)
        cache = apply_plan(clone_cache(sp.cache), plan, qcfg)
        out = greedy_generate(model, cache, sp.last_token, sp.last_position, max_new_tokens, eos_ids)
        del cache
        answer = tok.decode(out, skip_special_tokens=True).strip()
        rows.append(
            {
                "method": spec.label,
                "budget": spec.budget,
                "ratio": plan_bytes(plan, dims.head_dim, qcfg).ratio,
                "correct": prompt.needle.value in answer,
                "answer": answer,
                **needle_stats(plan, prompt.needle_span),
                **extra,
            }
        )
    return rows
