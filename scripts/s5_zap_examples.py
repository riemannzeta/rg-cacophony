"""Pick one representative zap per condition for the explainer figure (fig0).

Runs where the raw lattices live (results/raw/m2_<name>.npz, i.e. on the GPU box or after
downloading the release assets). For each condition, scans every (k <= 3 blocks, passage,
direction) zap at p = 64 and keeps the one whose (PCI, reach) at theta = 0.01 is nearest the
condition's mean from results/m2/<name>/main.json. Mamba-130M has 24 layers; as in the primary
analysis (PREREG-2 C), its lattices are strided by 2 to match Pythia-160M's 12-layer depth.

Writes results/figures/zap_examples.npz (the chosen gain maps) and zap_examples.json (which zap
was chosen, its PCI and reach, and the condition mean it was matched to).
"""

import json
from pathlib import Path

import numpy as np

from rgc.pci import lattice_bits, normalised_lz

R = Path("results")
THETA = 0.01
P_INDEX, P = 1, 64
CONDS = [("pythia-160m_step143000", 1), ("pythia-160m_step143000__beta=0", 1),
         ("pythia-160m_step0", 1), ("mamba-130m-hf-shuffled1001", 2)]


def main() -> None:
    out, info = {}, {}
    for name, stride in CONDS:
        m = json.load(open(R / "m2" / name / "main.json"))
        r = [x for x in m["pci_by_theta"] if x["theta"] == THETA][0]
        target = (r["pci"], r["mean_p1"])
        z = np.load(R / "raw" / f"m2_{name}.npz")
        best = None
        for kname in [k for k in z.files if int(k[1:]) <= 3 * stride and int(k[1:]) % stride == 0]:
            L = z[kname]
            for n in range(L.shape[0]):
                for d in range(L.shape[2]):
                    G = L[n, P_INDEX, d][::stride]
                    b = lattice_bits(G, P, THETA)
                    pci, reach = normalised_lz(b), b.mean()
                    dist = ((pci - target[0]) / 0.1) ** 2 + ((reach - target[1]) / 0.1) ** 2
                    if best is None or dist < best[0]:
                        best = (dist, kname, n, d, pci, reach, G)
        out[name] = best[6].astype(np.float32)
        info[name] = {"k": best[1], "passage": best[2], "dir": best[3], "pci": float(best[4]),
                      "reach": float(best[5]), "target": target}
        print(name, info[name])
    (R / "figures").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(R / "figures" / "zap_examples.npz", **out)
    json.dump(info, open(R / "figures" / "zap_examples.json", "w"))


if __name__ == "__main__":
    main()
