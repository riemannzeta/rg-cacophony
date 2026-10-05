"""Stage 3 figures from main.json / calib.json and the raw lattices (run where results/raw exists).

  uv run python scripts/s3b_m2_figs.py results/m2/<name> [results/m2/<other> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

from rgc import plots
from rgc.config import CFG
from rgc.determinism import REPO


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    mains = {}
    for d in dirs:
        calib = json.loads((d / "calib.json").read_text())
        main = json.loads((d / "main.json").read_text())
        label = main["variant"] + ("" if main["manip"] == "none" else f" [{main['manip']}]")
        mains[label] = main
        figs = d / "figs"
        plots.m2_calibration(calib, figs / "m2_calibration.png")
        plots.m2_spread(main, figs / "m2_spread_vs_depth.png")
        raw = REPO / "results" / "raw" / f"m2_{d.name}.npz"
        if raw.exists():
            z = np.load(raw)
            ip = CFG.p_grid.index(CFG.z_primary_p)
            mean_lat = {int(k[1:]): z[k][:, ip].mean(axis=(0, 1)) for k in z.files}
            plots.m2_lattices(mean_lat, CFG.z_primary_p, CFG.theta_primary, f"Mean gain lattice — {label}",
                              figs / "m2_lattices_mean.png")
            one = {int(k[1:]): z[k][0, ip, 0] for k in z.files}
            plots.m2_lattices(one, CFG.z_primary_p, CFG.theta_primary, f"Single run (passage 0, direction 0) — {label}",
                              figs / "m2_lattices_single.png")
    plots.m2_pci_theta(mains, dirs[0] / "figs" / "m2_pci_by_theta.png")


if __name__ == "__main__":
    main()
