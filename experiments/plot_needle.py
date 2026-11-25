"""Figures for the report from a needle results file.

    python experiments/plot_needle.py results/needle_main.jsonl

Writes to ``results/figures/`` (PNG for slides, PDF for the paper):

* ``rate_distortion``: accuracy vs accounted memory, one line per
  method with 95% Wilson bands; uniform int2/int4 as single points (their
  memory is fixed by the bit width); the full cache as a reference line.
* ``depth_heatmaps``: accuracy by needle depth (columns) × memory budget
  (rows), one panel per method. This is where eviction methods fail.

Style follows the project's dataviz rules: fixed categorical order, thin
lines, ≥8px markers, recessive grid, ink-coloured text (never series-coloured),
a legend plus direct labels, and distinct marker shapes so identity never rests
on colour alone. The summary tables (summarize.py) are the table view.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import wilson  # noqa: E402
from summarize import load_rows  # noqa: E402

INK = "#191919"
INK2 = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"
SURFACE = "#ffffff"

# Fixed categorical order (validated palette, light mode). Colour follows the method.
SERIES = {
    "tiered": ("#257de0", "o", "tiered (ours)"),
    "window_evict": ("#f47230", "s", "SnapKV-style eviction"),
    "streaming": ("#00b894", "^", "StreamingLLM"),
    "h2o": ("#f2a600", "D", "H2O"),
}
UNIFORM = ("#ee7fac", "P")
SEQ = ["#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"]

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
    }
)


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _cells(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["method"], r["budget"])].append(r)
    return cells


def _spread(ys, min_gap):
    """Nudge label y positions apart (keeps order) so direct labels don't collide."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def plot_rate_distortion(rows, path: Path, title_suffix: str = ""):
    cells = _cells(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    _style(ax)
    ends = []
    for m, (color, marker, label) in SERIES.items():
        pts = sorted((b, rs) for (mm, b), rs in cells.items() if mm == m)
        if not pts:
            continue
        x = np.array([np.mean([r["ratio"] for r in rs]) for _, rs in pts]) * 100
        k = np.array([sum(int(r["correct"]) for r in rs) for _, rs in pts])
        n = np.array([len(rs) for _, rs in pts])
        acc = k / n
        ci = np.array([wilson(int(a), int(b)) for a, b in zip(k, n)])
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, linewidth=0)
        ax.plot(x, acc, color=color, linewidth=1.5, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
        ends.append((x[-1], acc[-1], label))
    for m, label in (("uniform-k2v2", "uniform int2"), ("uniform-k4v4", "uniform int4")):
        rs = cells.get((m, None))
        if not rs:
            continue
        x = np.mean([r["ratio"] for r in rs]) * 100
        k = sum(int(r["correct"]) for r in rs)
        lo, hi = wilson(k, len(rs))
        acc = k / len(rs)
        ax.errorbar(x, acc, yerr=[[acc - lo], [hi - acc]], color=UNIFORM[0], marker=UNIFORM[1], markersize=8,
                    markeredgecolor=SURFACE, linewidth=1.2, capsize=0, zorder=4,
                    label="uniform (KIVI-style)" if m == "uniform-k2v2" else None)
        ax.annotate(label, (x, acc), xytext=(6, -12), textcoords="offset points", color=INK2, fontsize=8)
    full = cells.get(("full", None))
    if full:
        acc = np.mean([int(r["correct"]) for r in full])
        ax.axhline(acc, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate("full cache (100% memory)", (1, acc), xytext=(2, 4), textcoords="offset points",
                    color=INK2, fontsize=8)
    if ends:
        ys = _spread([e[1] for e in ends], 0.055)
        for (x, _, label), y in zip(ends, ys):
            ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=8)
    ax.set_xlim(0, max(55, ax.get_xlim()[1]))
    ax.set_xlim(right=ax.get_xlim()[1] + 18)  # room for direct labels
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("accounted KV-cache memory (% of full bf16 cache)")
    ax.set_ylabel("needle retrieval accuracy")
    ax.set_title("Accuracy vs memory" + title_suffix, loc="left")
    # Legend below the plot so it never covers data or the direct labels.
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def plot_depth_heatmaps(rows, path: Path):
    cells = _cells(rows)
    methods = [m for m in SERIES if any(mm == m for (mm, _) in cells)]
    budgets = sorted({b for (m, b) in cells if b is not None})
    depths = sorted({r["depth"] for r in rows})
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, axes = plt.subplots(1, len(methods), figsize=(2.9 * len(methods), 0.42 * len(budgets) + 1.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods):
        grid = np.full((len(budgets), len(depths)), np.nan)
        for i, b in enumerate(budgets):
            for j, d in enumerate(depths):
                xs = [int(r["correct"]) for r in cells.get((m, b), []) if r["depth"] == d]
                if xs:
                    grid[i, j] = np.mean(xs)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
        for i in range(len(budgets)):
            for j in range(len(depths)):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                label = "0" if v < 0.05 else ("1" if v > 0.95 else f"{v:.1f}".lstrip("0"))
                ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                        color=SURFACE if v > 0.55 else INK)
        ax.set_title(SERIES[m][2], loc="left")
        ax.set_xticks(range(len(depths)), [f"{d:g}" for d in depths], fontsize=7)
        ax.set_yticks(range(len(budgets)), [f"{b:.0%}" for b in budgets], fontsize=7)
        ax.set_xlabel("needle depth")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("memory budget")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args(argv)
    rows = load_rows(args.path)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.path).stem
    lengths = sorted({r["length"] for r in rows})
    suffix = f" · Qwen2.5-1.5B, {len({r['trial'] for r in rows})} prompts, {'/'.join(f'{L // 1024}k' for L in lengths)} tokens"
    plot_rate_distortion(rows, out / f"{stem}_rate_distortion", suffix)
    plot_depth_heatmaps(rows, out / f"{stem}_depth_heatmaps")
    print(f"wrote figures to {out}/")


if __name__ == "__main__":
    main()
