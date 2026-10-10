"""Stage 5: consolidated figures for REPORT.md and the symmetrybroken.com page (results/figures/).

All numbers come from the saved per-run outputs; nothing is recomputed except bootstrap-free means.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rgc import plots as P
from rgc.determinism import REPO

R = REPO / "results"
OUT = R / "figures"
TR = "pythia-160m_step143000"


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fig_m1_delta(path):
    """Fig 1: control-corrected sense separation across depth: trained vs controls and early checkpoints."""
    runs = [(TR, "trained (step 143k)", 0, 2.4), ("pythia-160m_step4000", "step 4000", 1, 1.6),
            ("pythia-160m_step512", "step 512", 2, 1.6), ("pythia-160m_step0", "untrained (step 0)", 3, 1.6),
            (f"{TR}-shuffled1001", "shuffled weights (3 seeds)", 4, 1.2)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for name, label, slot, lw in runs:
        s = load(R / "m1" / name / "summary.json")
        d = s["dprime"]["delta"]
        x = np.arange(len(d["pooled"]))
        ax.fill_between(x, d["ci_lo"], d["ci_hi"], color=P.SERIES[slot], alpha=0.12, linewidth=0)
        ax.plot(x, d["pooled"], color=P.SERIES[slot], linewidth=lw, marker="o", markersize=3.5, label=label)
    for seed in (1002, 1003):
        d = load(R / "m1" / f"{TR}-shuffled{seed}" / "summary.json")["dprime"]["delta"]
        ax.plot(np.arange(len(d["pooled"])), d["pooled"], color=P.SERIES[4], linewidth=1.2, marker="o", markersize=3.5)
    ax.axhline(0, color=P.AXIS, linewidth=1)
    ax.set_xlabel("depth (residual slice; 0 = embedding)")
    ax.set_ylabel("Δd′ = sense separation at the word − at a neutral token")
    ax.set_title("Word-sense separation beyond context leakage (95% CIs)", loc="left")
    ax.legend(fontsize=8, ncol=2)
    P._save(fig, path)


def fig_plane(path):
    """Fig 2: the integration-differentiation plane at theta = 0.01: reach vs PCI for every condition."""
    pts = []
    def add(root, name, label, slot, marker="o"):
        m = load(R / root / name / "main.json")
        if m:
            r = [x for x in m["pci_by_theta"] if x["theta"] == 0.01][0]
            pts.append((r["mean_p1"], r["pci"], label, slot, marker))
    add("m2", TR, "Pythia-160M trained", 0, "o")
    for b in ("0.5", "0.25", "0"):
        add("m2", f"{TR}__beta={b}", f"flattened attention β={b}", 1, "v")
    for w in ("64", "16", "4"):
        add("m2", f"{TR}__window={w}", f"local window {w}", 2, "s")
    add("m2", "pythia-160m_step0", "untrained (step 0)", 3, "D")
    for sd in (1001, 1002, 1003):
        add("m2", f"{TR}-shuffled{sd}", "Pythia shuffled", 4, "X")
    add("m2", "mamba-130m-hf", "Mamba-130M trained", 6, "o")
    for sd in (1001, 1002, 1003):
        add("m2", f"mamba-130m-hf-shuffled{sd}", "Mamba shuffled", 7, "X")
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    seen = set()
    for x, y, label, slot, mk in pts:
        ax.scatter(x, y, s=70, color=P.SERIES[slot], marker=mk, edgecolors=P.SURFACE, linewidths=1.2, zorder=3,
                   label=None if label in seen else label)
        seen.add(label)
    ax.set_xlabel("reach: fraction of sites the poke moves by ≥ 1% of its size (θ = 0.01)")
    ax.set_ylabel("PCI (normalised LZ complexity of the echo)")
    ax.set_title("Integration vs differentiation: where each condition sits", loc="left")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    P._save(fig, path)


def fig_replication(path):
    """Fig 3: PREREG-3 fresh-data replication: PCI, reach and threshold-free spread vs attention temperature."""
    betas = ["1", "0.5", "0.25", "0"]
    rows = []
    for b in betas:
        name = TR if b == "1" else f"{TR}__beta={b}"
        m = load(R / "m2_fresh" / name / "main.json")
        rows.append((float(b), np.mean(m["per_passage_dir_pci_by_theta"]["0.01"]),
                     np.mean(m["per_passage_dir_reach_by_theta"]["0.01"]), np.mean(m["per_passage_dir_seff"]),
                     np.mean(m["per_passage_dir_pci_dm"])))
    rows = np.array(rows)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    for ax, (j, title, slot) in zip(axes, [(1, "PCI (θ = 0.01) falls", 0), (2, "reach barely moves", 2),
                                           (3, "threshold-free spread grows", 1)]):
        ax.plot(rows[:, 0], rows[:, j], color=P.SERIES[slot], marker="o", markersize=5)
        if j == 1:
            ax.plot(rows[:, 0], rows[:, 4], color=P.SERIES[6], marker="s", markersize=4, linestyle="--",
                    label="density-matched PCI")
            ax.legend(fontsize=7.5)
        ax.invert_xaxis()
        ax.set_xlabel("β  (1 = trained, 0 = uniform)")
        ax.set_title(title, loc="left", fontsize=9.5)
        if j == 2:
            ax.set_ylim(0, 0.4)  # honest scale: reach changes by only a few hundredths
    fig.suptitle("Fresh-data replication: flattening attention lowers PCI without lowering reach (pre-registered)",
                 x=0.0, ha="left", fontweight="bold", color=P.INK)
    P._save(fig, path)


def fig_lambda(path):
    """Fig 4: PREREG-4 growth exponent lambda: trained vs controls, and vs temperature (exploratory)."""
    root = R / "m2_fresh2"
    names = [(TR, "trained"), ("pythia-160m_step4000", "step 4000"), ("pythia-160m_step512", "step 512"),
             ("pythia-160m_step0", "step 0"), (f"{TR}-shuffled1001", "shuffled 1"), (f"{TR}-shuffled1002", "shuffled 2"),
             (f"{TR}-shuffled1003", "shuffled 3")]
    vals = [(lab, load(root / n / "main.json")) for n, lab in names]
    vals = [(lab, m) for lab, m in vals if m]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True, width_ratios=[1.1, 1])
    lam = [np.nanmean(m["per_passage_dir_lambda"]) for _, m in vals]
    ys = np.arange(len(vals))[::-1]
    ax.axvline(0, color=P.AXIS, linewidth=1)
    rng = np.random.default_rng(0)
    for y, (lab, m), l in zip(ys, vals, lam):
        arr = np.nanmean(np.array(m["per_passage_dir_lambda"], float), axis=1)  # per passage
        boots = [np.mean(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])  # 95% CI of the mean (bootstrap over passages)
        ax.plot([lo, hi], [y, y], color=P.SERIES[0 if lab == "trained" else 4], linewidth=2)
        ax.scatter(l, y, s=50, color=P.SERIES[0 if lab == "trained" else 4], zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for lab, _ in vals])
    ax.set_xlabel("growth exponent λ per layer, mean with 95% CI\n(<0 dies out · 0 poised · >0 grows)")
    ax.set_title("Trained model sits at λ ≈ 0 (pre-registered)", loc="left")
    betas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
    for slot, base, lab in ((0, TR, "trained"), (4, f"{TR}-shuffled1001", "shuffled 1")):
        xs, ls = [], []
        for b in betas:
            m = load(root / (base if b == 1.0 else f"{base}__beta={b:g}") / "main.json")
            if m:
                xs.append(b)
                ls.append(np.nanmean(m["per_passage_dir_lambda"]))
        ax2.plot(xs, ls, color=P.SERIES[slot], marker="o", markersize=4, label=lab)
    ax2.axhline(0, color=P.AXIS, linewidth=1)
    ax2.axvline(1, color=P.MUTED, linestyle=":", linewidth=1)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("attention inverse temperature β (log scale; 1 = natural)")
    ax2.set_ylabel("λ")
    ax2.set_title("λ across temperature (exploratory)", loc="left")
    ax2.legend(fontsize=8)
    P._save(fig, path)


def fig_temperature(path):
    """Fig 5: PCI and chimera index across temperature (PREREG-4; beta=4 PCI is post hoc, D23)."""
    root, post = R / "m2_fresh2", R / "m2_fresh2_posthoc"
    betas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    for slot, base, lab in ((0, TR, "trained"), (4, f"{TR}-shuffled1001", "shuffled 1")):
        xs, ys, xh, yh, xc, yc = [], [], [], [], [], []
        for b in betas:
            d = root / (base if b == 1.0 else f"{base}__beta={b:g}")
            m = load(d / "main.json")
            if m:
                xs.append(b); ys.append(np.mean(m["per_passage_dir_pci_by_theta"]["0.01"]))
                xc.append(b); yc.append(m["chimera"]["blocks16"]["chimera_mean"])
            else:
                mp = load(post / d.name / "main.json")
                if mp:
                    xh.append(b); yh.append(np.mean(mp["per_passage_dir_pci_by_theta"]["0.01"]))
                ch = load(d / "chimera.json")
                if ch:
                    xc.append(b); yc.append(ch["chimera"]["blocks16"]["chimera_mean"])
        ax.plot(xs, ys, color=P.SERIES[slot], marker="o", markersize=4, label=lab)
        if xh:
            ax.scatter(xh, yh, facecolors="none", edgecolors=P.SERIES[slot], s=45, linewidths=1.6,
                       label=f"{lab}: β=4 post hoc (ε=0.01)")
        order = np.argsort(xc)
        ax2.plot(np.array(xc)[order], np.array(yc)[order], color=P.SERIES[slot], marker="o", markersize=4, label=lab)
    for a in (ax, ax2):
        a.set_xscale("log", base=2)
        a.axvline(1, color=P.MUTED, linestyle=":", linewidth=1)
        a.set_xlabel("attention inverse temperature β (log scale)")
        a.legend(fontsize=7.5)
    ax.set_ylabel("PCI (θ = 0.01)")
    ax.set_title("PCI across temperature (T1/T2)", loc="left")
    ax2.set_ylabel("chimera index χ")
    ax2.set_title("Chimera index across temperature (C2)", loc="left")
    P._save(fig, path)


def fig_growth(path):
    """Fig 6: growth curves behind lambda (exploratory). Each curve is labelled with its OWN fitted
    slope (0.5*log E over all t, and over t >= 3); the aggregate lambda averages over all injections."""
    gc = load(R / "m2_fresh2" / "growth_curves.json")
    if not gc:
        return
    show = [(TR, "trained", 0), ("pythia-160m_step0", "step 0", 3), (f"{TR}-shuffled1001", "shuffled 1", 4)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True, sharey=True)
    for ax, cell, title in ((axes[0], "k0_p64", "poke at the embedding (k = 0)"), (axes[1], "k4_p64", "poke mid-depth (k = 4)")):
        for name, lab, slot in show:
            c = gc["conditions"].get(name)
            if not c or cell not in c["curves"]:
                continue
            E = np.array(c["curves"][cell])
            t = np.arange(1, len(E) + 1)
            y = 0.5 * np.log(E)
            sl_all = np.polyfit(t, y, 1)[0]
            sl_late = np.polyfit(t[t >= 3], y[t >= 3], 1)[0] if (t >= 3).sum() >= 3 else np.nan
            ax.plot(t, np.sqrt(E), color=P.SERIES[slot], marker="o", markersize=3.5,
                    label=f"{lab}: slope {sl_all:+.3f} (late {sl_late:+.3f})")
        ax.set_yscale("log")
        ax.set_xlabel("layers after the poke (position 64)")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("RMS cross-word response (gain)")
    fig.suptitle("How a poke's footprint evolves with depth (exploratory; aggregate λ averages all injection layers)",
                 x=0.0, ha="left", fontweight="bold", color=P.INK, fontsize=10)
    P._save(fig, path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (R / "figures" / "zap_examples.npz").exists():
        fig_zap_examples(OUT / "fig0_zap_examples.png")
    fig_m1_delta(OUT / "fig1_m1_sense_separation.png")
    fig_plane(OUT / "fig2_integration_differentiation_plane.png")
    fig_replication(OUT / "fig3_anaesthesia_replication.png")
    fig_lambda(OUT / "fig4_growth_exponent.png")
    fig_temperature(OUT / "fig5_temperature_sweep.png")
    fig_growth(OUT / "fig6_growth_curves.png")
    print("figures ->", sorted(p.name for p in OUT.glob("*.png")))



def fig_zap_examples(path):
    """Fig 0 (explainer): one zap, four systems. Binarised echo maps at theta = 0.01."""
    from matplotlib.colors import ListedColormap

    from rgc.pci import lattice_bits, normalised_lz

    z = np.load(R / "figures" / "zap_examples.npz")
    panels = [("pythia-160m_step143000", "Trained language model"),
              ("pythia-160m_step143000__beta=0", "Same model, attention flattened (“anaesthesia”)"),
              ("pythia-160m_step0", "Untrained network"),
              ("mamba-130m-hf-shuffled1001", "Scrambled weights (Mamba): “seizure-like”")]
    p, th = 64, 0.01
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.4), constrained_layout=True)
    cmap = ListedColormap([P.SURFACE, P.SERIES[0]])
    for ax, (name, title) in zip(axes.ravel(), panels):
        G = z[name]
        bits = lattice_bits(G, p, th)
        B = bits.reshape(G.shape[0] - 1, -1)
        ax.imshow(B, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"{title}\nPCI {normalised_lz(bits):.2f} · reach {100 * bits.mean():.0f}% of sites",
                     loc="left", fontsize=9)
        ax.set_xlabel("later words (after the zapped word)")
        ax.set_ylabel("layers after the zap")
        ax.set_yticks([0, B.shape[0] - 1])
        ax.set_yticklabels(["1", str(B.shape[0])])
        ax.grid(False)
    fig.suptitle("One typical zap, four systems (blue = moved by at least 1% of the zap)",
                 x=0.0, ha="left", fontweight="bold", color=P.INK, fontsize=10.5)
    P._save(fig, path)


if __name__ == "__main__":
    main()
