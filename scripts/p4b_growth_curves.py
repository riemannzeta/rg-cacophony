"""EXPLORATORY (PREREG-4 follow-up): per-layer growth curves behind the exponent lambda.

Question: is the controls' positive lambda steady amplification (constant slope over depth) or an
initial build-up of cross-position influence that then plateaus? For every condition in
results/m2_fresh2 with raw lattices, compute the cross-position energy E(t) = mean_{q>p} G(k+t, q)^2
(the quantity lambda is fitted to), averaged over passages and directions, per injection layer k <= 9
and position p, and report:
  - lambda_all  : slope of 0.5*log E over all t (as pre-registered)
  - lambda_late : slope over t >= 3 only (excludes the build-up)
  - E curves for the figure.
Runs where the raw lattices live (results/raw/m2_fresh2_*.npz). Writes results/m2_fresh2/growth_curves.json.
"""

import json
from pathlib import Path

import numpy as np

from rgc.config import CFG
from rgc.determinism import REPO

RAW = REPO / "results" / "raw"
OUT = REPO / "results" / "m2_fresh2" / "growth_curves.json"
KMAX = CFG.min_downstream_layers  # k <= n_layers - 3


def slope(t, y):
    ok = np.isfinite(y)
    return float(np.polyfit(t[ok], y[ok], 1)[0]) if ok.sum() >= 3 else float("nan")


def main() -> None:
    out = {"label": "EXPLORATORY follow-up to PREREG-4 S1/T3 (not pre-registered)", "conditions": {}}
    for f in sorted(RAW.glob("m2_fresh2_*.npz")):
        name = f.stem.replace("m2_fresh2_", "")
        z = np.load(f)
        ks = sorted(int(k[1:]) for k in z.files)
        nL = max(ks) + 1
        curves, lam_all, lam_late = {}, [], []
        for k in [k for k in ks if k <= nL - KMAX]:
            G = z[f"k{k}"].astype(np.float64)  # [N, nP, nD, S_k, T]
            for ip, p in enumerate(CFG.p_grid):
                E = (G[:, ip, :, 1:, p + 1 :] ** 2).mean(-1).mean(axis=(0, 1))  # [S_k - 1]
                t = np.arange(1, len(E) + 1, dtype=float)
                y = 0.5 * np.log(np.where(E > 0, E, np.nan))
                lam_all.append(slope(t, y))
                lam_late.append(slope(t[t >= 3], y[t >= 3]))
                curves[f"k{k}_p{p}"] = E.tolist()
        out["conditions"][name] = {"lambda_all_mean": float(np.nanmean(lam_all)),
                                   "lambda_late_mean": float(np.nanmean(lam_late)), "curves": curves}
        print(f"{name:52s} lambda_all {np.nanmean(lam_all):+.4f}  lambda_late(t>=3) {np.nanmean(lam_late):+.4f}")
    OUT.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
