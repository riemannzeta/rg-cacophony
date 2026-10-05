"""Static figures (matplotlib, light theme) using the validated reference palette.

Series colours are assigned in fixed slot order; text uses ink tokens, never series colours.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 150,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def m1_depth(summary: dict, path: Path) -> None:
    """Δd′ (primary) with CI; d′ target vs control-token with CIs; per-head entropy. Shared x."""
    d = summary["dprime"]
    S = len(d["delta"]["pooled"])
    x = np.arange(S)
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(7, 8), sharex=True, height_ratios=[3, 2.2, 2],
                                        constrained_layout=True)
    if summary.get("band"):
        b = summary["band"]
        for ax in (ax0, ax1):
            ax.axvspan(b["start"], b["end"] + 1, color=GRID, alpha=0.55, zorder=0, linewidth=0)
    ax0.fill_between(x, d["delta"]["ci_lo"], d["delta"]["ci_hi"], color=SERIES[0], alpha=0.18, linewidth=0)
    ax0.plot(x, d["delta"]["pooled"], color=SERIES[0], marker="o", markersize=4, zorder=3)
    ax0.axhline(0, color=AXIS, linewidth=1)
    ax0.set_ylabel("Δd′ = d′ target − d′ control")
    band_txt = ""
    if summary.get("band"):
        band_txt = f"  (shaded: band, blocks {summary['band']['start']}–{summary['band']['end']})"
    ax0.set_title(f"Sense separation across depth — {summary['variant']}{band_txt}", loc="left")

    for slot, (key, label) in ((1, ("target", "d′ at ambiguous target")), (2, ("control", "d′ at control token"))):
        c = SERIES[slot]
        ax1.fill_between(x, d[key]["ci_lo"], d[key]["ci_hi"], color=c, alpha=0.14, linewidth=0)
        ax1.plot(x, d[key]["pooled"], color=c, marker="o", markersize=4, label=label, zorder=3)
    ax1.axhline(0, color=AXIS, linewidth=1)
    ax1.set_ylabel("cross-validated d′")
    ax1.legend(loc="upper left", fontsize=8)

    H = np.array(summary["attention"]["H"]["test"])  # [L, H]
    sel = {tuple(h) for h in summary["selected_heads"]}
    for L in range(H.shape[0]):
        jitter = (np.arange(H.shape[1]) - (H.shape[1] - 1) / 2) * 0.03
        xs = L + 0.5 + jitter  # block L sits between slices L and L+1
        ax2.scatter(xs, H[L], s=14, color=SERIES[0], alpha=0.55, linewidths=0, zorder=2)
        for h in range(H.shape[1]):
            if (L, h) in sel:
                ax2.scatter(xs[h], H[L, h], s=46, facecolors="none", edgecolors=SERIES[1], linewidths=1.6, zorder=3)
    ax2.scatter([], [], s=14, color=SERIES[0], label="head (block L, plotted at L + ½)")
    ax2.scatter([], [], s=46, facecolors="none", edgecolors=SERIES[1], label="top-6 cue-mass heads (train)")
    ax2.set_ylabel("attention entropy\nat target (normalised)")
    ax2.set_xlabel("residual slice (0 = embedding; last = input to final LayerNorm)")
    ax2.set_ylim(0, 1.02)
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_xticks(x)
    _save(fig, path)


def m1_heatmaps(summary: dict, path: Path) -> None:
    """Cue mass, cue mass / uniform, entropy and head contribution per (layer, head)."""
    panels = [
        ("cue mass (test)", np.array(summary["attention"]["M"]["test"]), SEQ_BLUE, None),
        ("cue mass ÷ uniform baseline", np.array(summary["cue_mass_rel_uniform_test"]), SEQ_BLUE, None),
        ("normalised entropy (test)", np.array(summary["attention"]["H"]["test"]), SEQ_BLUE, None),
        ("head direct share of d′(L+1)", np.array(summary["head_contrib"]), "RdBu_r", "div"),
    ]
    sel = [tuple(h) for h in summary["selected_heads"]]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), constrained_layout=True)
    for ax, (title, A, cmap, kind) in zip(axes, panels):
        if kind == "div":
            lim = np.abs(A).max() or 1.0
            im = ax.imshow(A, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        else:
            im = ax.imshow(A, cmap=cmap, aspect="auto")
        for L, h in sel:
            ax.add_patch(plt.Rectangle((h - 0.5, L - 0.5), 1, 1, fill=False, edgecolor=SERIES[1], linewidth=1.8))
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xticks(range(A.shape[1]))
        ax.set_yticks(range(A.shape[0]))
        ax.set_xlabel("head")
        ax.set_ylabel("block (layer)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85).outline.set_visible(False)
    fig.suptitle(f"Per-head attention at the target — {summary['variant']}  (orange boxes: selected heads)",
                 x=0.0, ha="left", fontweight="bold", color=INK)
    _save(fig, path)


def m1_ablation(summary: dict, path: Path) -> None:
    a = summary["ablation"]
    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    bins = np.linspace(min(min(a["null_random"]), min(a["null_layer_matched"]), a["effect"]) - 0.05,
                       max(max(a["null_random"]), max(a["null_layer_matched"]), a["effect"]) + 0.05, 40)
    ax.hist(a["null_random"], bins=bins, color=SERIES[0], alpha=0.6, label="random 6-head sets")
    ax.hist(a["null_layer_matched"], bins=bins, color=SERIES[2], alpha=0.6, label="layer-matched random sets")
    ax.axvline(a["effect"], color=SERIES[1], linewidth=2.4, label="top-6 cue-mass heads")
    ax.set_xlabel("ablation effect E = 1 − Δd′_ablated(final) / Δd′(final)")
    ax.set_ylabel("count")
    ax.set_title(f"Mean-ablation effect vs nulls — {summary['variant']}", loc="left")
    ax.legend(fontsize=8)
    _save(fig, path)


def m1_all(summary_path: Path, fig_dir: Path) -> None:
    s = json.loads(Path(summary_path).read_text())
    m1_depth(s, fig_dir / "m1_dprime_entropy_depth.png")
    m1_heatmaps(s, fig_dir / "m1_head_heatmaps.png")
    m1_ablation(s, fig_dir / "m1_ablation_nulls.png")
