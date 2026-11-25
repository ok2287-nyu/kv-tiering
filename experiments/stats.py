"""Small, dependency-free statistics for binary (correct / incorrect) outcomes.

Why these and not a plain bootstrap:

* **Wilson score interval** for one accuracy. A percentile bootstrap of a
  proportion collapses to [1, 1] when every trial is correct (all resamples
  are identical), which wrongly suggests certainty. Wilson behaves at 0 and 1:
  10/10 gives about [0.72, 1.00], 135/135 about [0.97, 1.00].
* **Exact McNemar test** for comparing two methods. Every method runs on the
  *same* prompts, so outcomes are paired. Only discordant trials (one method
  right, the other wrong) carry information about which is better. Under H0
  they split 50/50, so the two-sided p-value is a binomial tail.
* **Paired bootstrap** for a CI on the accuracy *difference*: resample trials
  (keeping each trial's pair together) and recompute the difference.
"""

from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # exact endpoints at the boundaries (floating point leaves 0.9999999999999999)
    return (0.0 if k == 0 else lo), (1.0 if k == n else hi)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired binary outcomes.

    Returns (n_a_only, n_b_only, p): trials only ``a`` got right, trials only
    ``b`` got right, and the p-value for H0 "a and b are equally accurate".
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return n10, n01, min(1.0, 2 * tail)


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean(a) − mean(b) with a 95% percentile CI from resampling paired trials."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
