"""Why do the scores keep (or lose) the needle? A per-prompt diagnostic.

    python experiments/diagnose_scores.py --config experiments/configs/needle_main.yaml \
        --lengths 8192 --depths 0.1 0.5 0.9 --budget 0.2

For each prompt it records both importance scores (window = ours/SnapKV-like,
cumulative = H2O) and reports, averaged over layers and KV heads:

* **needle percentile**: fraction of prompt tokens scoring below the needle's
  best token (1.0 = the needle is the top-scoring token). High under window
  and low under cumulative means the scoring, not the budget, explains H2O.
* **where H2O's heavy hitters sit**: share of the heavy-hitter half of its
  budget (recent tokens excluded) in each tenth of the prompt. Mass piled into
  the first decile is the causal-mask bias: early tokens are seen by more
  queries.
* **needle in heavy hitters**: share of heads whose heavy hitters include a
  needle token.

Writes ``results/diagnose_scores.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from needle import trial_id  # noqa: E402

from kvtier.cache import cache_len, prefill_split  # noqa: E402
from kvtier.importance import record_attention  # noqa: E402
from kvtier.model import ModelDims, load_model  # noqa: E402
from kvtier.needle import build_prompt, load_haystack_text, make_needle  # noqa: E402
from kvtier.policies import plan_h2o  # noqa: E402


def needle_percentile(s: torch.Tensor, span: tuple[int, int]) -> float:
    """Mean over (layer, head) of the fraction of tokens scoring below the needle's best token."""
    best = s[..., span[0] : span[1]].max(dim=-1, keepdim=True).values
    return float((s < best).float().mean())


def analyze(scores, dims: ModelDims, T: int, span: tuple[int, int], budget: float, recent_frac: float = 0.5) -> dict:
    plan = plan_h2o(dims, T, budget, scores.cumulative, recent_frac)
    K = plan.kept_per_head()
    n_recent = min(K, max(1, int(round(K * recent_frac))))
    hh = plan.key_bits > 0
    hh[..., T - n_recent :] = False  # heavy hitters only
    pos = torch.nonzero(hh)[:, -1].float() / T
    deciles = torch.histc(pos, bins=10, min=0, max=1)
    deciles = (deciles / deciles.sum().clamp_min(1)).tolist()
    return {
        "needle_pct_window": needle_percentile(scores.window, span),
        "needle_pct_cumulative": needle_percentile(scores.cumulative, span),
        "h2o_heavy_hitter_deciles": [round(x, 3) for x in deciles],
        "needle_in_h2o_heavy_hitters": float(hh[..., span[0] : span[1]].any(dim=-1).float().mean()),
        "needle_in_recent_window": span[0] >= T - n_recent,
    }


def main(argv=None, model=None, tok=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lengths", type=int, nargs="+")
    ap.add_argument("--depths", type=float, nargs="+")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--out", default="results/diagnose_scores.json")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    if model is None:
        model, tok = load_model(cfg.get("model"))
    dims = ModelDims.from_config(model.config)
    hay = tok(load_haystack_text(cfg.get("haystack", "wikitext"), seed=cfg.get("seed", 0)), add_special_tokens=False).input_ids
    window = cfg.get("window", 64)

    results = []
    for L in args.lengths or cfg["lengths"]:
        for d in args.depths or cfg["depths"]:
            tid = trial_id(L, d, args.rep)
            rng = random.Random(f"{cfg.get('seed', 0)}-{tid}")  # same prompt as the sweep's trial
            needle = make_needle(rng)
            prompt = build_prompt(tok, hay, L, d, needle, rng)
            with torch.no_grad(), record_attention(model, window=window, cumulative=True) as rec:
                sp = prefill_split(model, prompt.ids.to(model.device))
            scores = rec.result()
            r = {"trial": tid, "budget": args.budget, **analyze(scores, dims, cache_len(sp.cache), prompt.needle_span, args.budget)}
            results.append(r)
            dec = " ".join(f"{x:.2f}" for x in r["h2o_heavy_hitter_deciles"])
            print(
                f"{tid}: needle pct window={r['needle_pct_window']:.3f} cumulative={r['needle_pct_cumulative']:.3f} | "
                f"needle in H2O heavy hitters={r['needle_in_h2o_heavy_hitters']:.2f} | HH by decile: {dec}",
                flush=True,
            )
            del sp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")
    return results


if __name__ == "__main__":
    main()
