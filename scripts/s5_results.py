"""Stage 5: assemble results/results.json — every summary number (with CIs) plus the scorecard.

Reads the per-run summaries already in results/ (no recomputation) and scripts/s4_compare.py's
scorecard. Run s4_compare.py first.
"""

import json
import subprocess
from pathlib import Path

from rgc.config import CFG, config_hash
from rgc.determinism import REPO

R = REPO / "results"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def m2_summary(d: Path) -> dict | None:
    m, c = load(d / "main.json"), load(d / "calib.json")
    if m is None:
        return None
    by = {str(r["theta"]): {"pci": r["pci"], "ci": r["ci"], "reach_p1": r["mean_p1"],
                            "degenerate_frac": r["degenerate_frac"]} for r in m["pci_by_theta"]}
    out = {"commit": m["env"]["git_sha"], "eps_star": c["eps_star"] if c else m.get("eps_star"),
           "eps_override_post_hoc": m.get("eps_override_post_hoc"), "pci_by_theta": by,
           "pci_auc_logtheta": m["pci_auc_logtheta"], "z": {k: v for k, v in m["z"].items() if k != "boot_z"},
           "reach_primary_theta0.1": m.get("reach_primary"), "local_response": m.get("local_response"),
           "next_token_ce": m.get("ce_mean")}
    if "lambda_mean" in m:
        out["lambda_mean"] = m["lambda_mean"]
    if "per_passage_dir_pci_dm" in m:
        import numpy as np
        out["pci_density_matched_mean"] = float(np.mean(m["per_passage_dir_pci_dm"]))
    if "chimera" in m:
        ch = m["chimera"]["blocks16"]
        out["chimera"] = {"chi_mean": ch["chimera_mean"], "metastability_mean": ch["metastability_mean"],
                          "eff_dim_by_slice": ch["eff_dim_mean"]}
    return out


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    tags = subprocess.check_output(["git", "tag", "-l", "prereg*", "--format=%(refname:short) %(*objectname:short) %(taggerdate:iso)"],
                                   cwd=REPO, text=True).strip().splitlines()
    out = {
        "meta": {"repo": "https://github.com/riemannzeta/rg-cacophony", "built_at_commit": sha,
                 "config_hash": config_hash(), "preregistration_tags": tags,
                 "note": "Every number here is copied from the per-run outputs in results/; see REPORT.md."},
        "scorecard": load(R / "scorecard.json"),
        "M1": {}, "M2": {}, "M2_fresh_PREREG3": {}, "M2_fresh2_PREREG4": {}, "conflict_PREREG2_B": {},
    }
    for d in sorted((R / "m1").iterdir()):
        s = load(d / "summary.json")
        if s:
            out["M1"][d.name] = {
                "commit": s["env"]["git_sha"], "dprime": s["dprime"], "peak": s["peak"], "L_star": s["L_star"],
                "band": s["band"], "selected_heads": s["selected_heads"], "alignment": s["alignment"],
                "ablation": {k: v for k, v in s["ablation"].items() if not k.startswith("null_") or k.endswith("_pct")},
                "neutral": s["neutral"], "post_hoc": load(d / "posthoc.json") and {
                    k: v for k, v in load(d / "posthoc.json").items() if k in ("label", "ablation", "entropy_detrended", "neutral_lean")}}
    for root, key in (("m2", "M2"), ("m2_fresh", "M2_fresh_PREREG3"), ("m2_fresh2", "M2_fresh2_PREREG4"),
                      ("m2_fresh2_posthoc", "M2_fresh2_PREREG4_posthoc"), ("m2_fresh3", "M2_fresh3_PREREG5")):
        if (R / root).exists():
            out.setdefault(key, {})
            for d in sorted(p for p in (R / root).iterdir() if p.is_dir()):
                s = m2_summary(d)
                if s:
                    out[key][d.name] = s
                elif (d / "chimera.json").exists():
                    ch = load(d / "chimera.json")["chimera"]["blocks16"]
                    out[key][d.name] = {"chimera_only": True, "chimera": {"chi_mean": ch["chimera_mean"],
                                        "metastability_mean": ch["metastability_mean"], "eff_dim_by_slice": ch["eff_dim_mean"]}}
    g = load(R / "m2" / f"{CFG.model_name}_step{CFG.final_step}" / "gating.json")
    if g:
        out["M2_gating_exploratory"] = {k: v for k, v in g.items() if k.startswith("contrast") or k in ("L_star_trained", "k_before", "k_after", "window", "clipped")}
    gc = load(R / "m2_fresh2" / "growth_curves.json")
    if gc:
        out["PREREG4_growth_curves_exploratory"] = {k: {kk: vv for kk, vv in v.items() if kk != "curves"} for k, v in gc["conditions"].items()}
    gc5 = load(R / "m2_fresh3" / "growth_curves.json")
    if gc5:
        out["PREREG5_growth_curves_secondary"] = {k: {kk: vv for kk, vv in v.items() if kk != "curves"} for k, v in gc5["conditions"].items()}
    for d in sorted((R / "conflict").iterdir()) if (R / "conflict").exists() else []:
        s = load(d / "summary.json")
        if s:
            out["conflict_PREREG2_B"][d.name] = {k: s[k] for k in ("B1", "B2", "B3_exploratory")} | {
                w: {m: s[w][m]["pooled"] for m in ("R", "D", "rho", "sep")} for w in ("target", "control")}
    (R / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote results/results.json:", {k: (len(v) if isinstance(v, dict) else "") for k, v in out.items()})


if __name__ == "__main__":
    main()
