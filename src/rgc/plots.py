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


# ---------------------------------------------------------------------------------------------
# Measurement 2
# ---------------------------------------------------------------------------------------------

def m2_calibration(calib: dict, path: Path) -> None:
    rows = calib["rows"]
    eps = [r["eps"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    ax.plot(eps, [r["median_rel_dev"] for r in rows], color=SERIES[0], marker="o", markersize=4, label="median")
    ax.plot(eps, [r["p95_rel_dev"] for r in rows], color=SERIES[1], marker="o", markersize=4, label="95th percentile")
    ax.axhline(0.05, color=MUTED, linestyle="--", linewidth=1)
    ax.text(eps[0], 0.05, " 5% tolerance", va="bottom", color=INK2, fontsize=8)
    if calib.get("eps_star"):
        ax.axvline(calib["eps_star"], color=SERIES[2], linewidth=1.6)
        ax.text(calib["eps_star"], ax.get_ylim()[1] if False else 1.0, f" ε* = {calib['eps_star']:.0e}", color=INK2,
                fontsize=8, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ε (relative perturbation size)")
    ax.set_ylabel("|G_float32 − G_float64ref| / G_ref")
    ax.set_title(f"Linear-regime calibration — {calib['variant']}" + ("" if calib.get("manip", "none") == "none" else f" [{calib['manip']}]"), loc="left")
    ax.legend(fontsize=8)
    _save(fig, path)


def m2_lattices(G_by_k: dict, p: int, theta: float, title: str, path: Path, ks=(0, 4, 8)) -> None:
    """Mean gain lattices log10 G(L, q) for injection at (k, p), averaged over passages and directions."""
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 3.6), constrained_layout=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        G = G_by_k[k]  # [S_k, T] mean lattice
        A = np.log10(np.clip(G, 1e-4, None))
        im = ax.imshow(A, aspect="auto", cmap=SEQ_BLUE, vmin=-4, vmax=max(0.5, float(A.max())), origin="lower",
                       extent=[-0.5, G.shape[1] - 0.5, k - 0.5, k + G.shape[0] - 0.5])
        ax.contour(np.arange(G.shape[1]), np.arange(k, k + G.shape[0]), G, levels=[theta], colors=[SERIES[1]], linewidths=0.9)
        ax.axvline(p, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xlim(max(0, p - 8), G.shape[1] - 1)
        ax.set_title(f"inject at slice {k}, position {p}", loc="left", fontsize=9)
        ax.set_xlabel("position q")
        ax.set_ylabel("residual slice L")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85, label="log10 gain").outline.set_visible(False)
    fig.suptitle(f"{title}  (orange contour: θ = {theta}; dotted: injected column)", x=0.0, ha="left",
                 fontweight="bold", color=INK)
    _save(fig, path)


def m2_pci_theta(mains: dict, path: Path) -> None:
    """PCI vs threshold with 95% CIs; mains: label -> main.json dict (slot order = dict order)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    for slot, (label, m) in enumerate(mains.items()):
        th = [r["theta"] for r in m["pci_by_theta"]]
        ax.fill_between(th, [r["ci"][0] for r in m["pci_by_theta"]], [r["ci"][1] for r in m["pci_by_theta"]],
                        color=SERIES[slot % 8], alpha=0.15, linewidth=0)
        ax.plot(th, [r["pci"] for r in m["pci_by_theta"]], color=SERIES[slot % 8], marker="o", markersize=4, label=label)
    ax.axvline(0.1, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("binarisation threshold θ on gain")
    ax.set_ylabel("PCI (normalised LZ76)")
    ax.set_title("PCI across thresholds (dashed: pre-registered θ = 0.1)", loc="left")
    ax.legend(fontsize=8)
    _save(fig, path)


def m2_spread(main: dict, path: Path, cell: str = "k0_p64") -> None:
    sp = main["spread"][cell]
    S = np.array(sp["S_mean"]); Seff = np.array(sp["S_eff_mean"])
    t = np.arange(1, len(S) + 1)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax.plot(t, S, color=SERIES[0], marker="o", markersize=5, linewidth=0, label="S(t), mean over passages")
    f = sp["pooled_fit"]
    if f.get("fittable"):
        ok = (S > 0) & (S < 0.9 * sp["n_available"])
        tt = t[ok]
        a_p = np.mean(np.log(S[ok]) - f["b"] * np.log(tt)); a_e = np.mean(np.log(S[ok]) - f["inv_tau"] * tt)
        ax.plot(tt, np.exp(a_p) * tt ** f["b"], color=SERIES[1], label=f"power law, z = {f['z']:.2f}" if f.get("z") else "power law")
        ax.plot(tt, np.exp(a_e + f["inv_tau"] * tt), color=SERIES[2], linestyle="--", label="exponential")
        ax.set_title(f"Spread vs depth ({cell}); ΔAIC(exp − power) = {f['delta_aic_exp_minus_power']:.1f}", loc="left", fontsize=9)
    ax.axhline(0.9 * sp["n_available"], color=MUTED, linestyle=":", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("depth after injection t = L − k"); ax.set_ylabel("positions with gain ≥ θ")
    ax.legend(fontsize=8)
    ax2.plot(t, Seff, color=SERIES[0], marker="o", markersize=4)
    ax2.set_xlabel("depth after injection t"); ax2.set_ylabel("participation spread (threshold-free)")
    ax2.set_title("Threshold-free spread", loc="left", fontsize=9)
    _save(fig, path)
