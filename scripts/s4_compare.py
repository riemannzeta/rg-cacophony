"""Stage 4: mechanical scoring of every pre-registered criterion (PREDICTIONS.md, PREREG-2.md).

Reads results/m1/*/summary.json, results/m2/*/main.json, results/conflict/*/summary.json and writes
results/scorecard.json and results/SCORECARD.md. Criteria whose inputs do not exist yet are reported
as "pending". Nothing here chooses thresholds: every constant comes from config.py / config_prereg2.py.
"""

import json
from pathlib import Path

import numpy as np

from rgc.config import CFG
from rgc.config_prereg2 import CFG2
from rgc.determinism import REPO, env_info
from rgc.stats import ci

R = REPO / "results"
TRAINED = f"{CFG.model_name}_step{CFG.final_step}"
CONTROLS = [f"{CFG.model_name}_step0"] + [f"{TRAINED}-shuffled{s}" for s in CFG.shuffle_seeds]
EARLY = [f"{CFG.model_name}_step{s}" for s in CFG.checkpoints if s not in (0, CFG.final_step)]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def paired_boot(A: np.ndarray, B: np.ndarray, fn=lambda a, b: a.mean() - b.mean(), seed=CFG.boot_seed):
    """Crossed paired bootstrap over passages (rows) and directions (cols); same draws for A and B."""
    rng = np.random.default_rng(seed)
    N, D = A.shape
    out = np.empty(CFG.n_boot)
    for b in range(CFG.n_boot):
        i, j = rng.integers(0, N, N), rng.integers(0, D, D)
        out[b] = fn(A[i][:, j], B[i][:, j])
    lo, hi = ci(out)
    return {"point": float(fn(A, B)), "ci": [float(lo), float(hi)]}


def score_a(s):
    al, ab = s["alignment"], s["ablation"]
    a1 = s["band"] is not None and s["band"]["width"] <= CFG.band_max_width and s["peak"]["ci"][0] > 0
    return {
        "a1": a1,
        "a2-i": al["p_hypergeom"] is not None and al["p_hypergeom"] < CFG.alpha,
        "a2-ii": al["spearman_contrib_cuemass"]["rho"] > 0 and al["spearman_contrib_cuemass"]["p_greater"] < CFG.alpha,
        "a2-iii": al["spearman_contrib_entropy"]["rho"] < 0 and al["spearman_contrib_entropy"]["p_less"] < CFG.alpha,
        "a3": ab["effect"] >= CFG.ablation_min_effect and ab["effect"] > ab["null_random_pct"]
              and ab["effect"] > ab["null_layer_matched_pct"],
    }


def score_b(s, trained_peak):
    pk = s["peak"]
    b1 = (pk["ci"][0] <= 0 <= pk["ci"][1]) or pk["value"] < CFG.b1_peak_ratio * trained_peak
    b2 = not (s["ablation"]["effect"] > s["ablation"]["null_random_pct"])
    return {"b1": b1, "b2": b2, "pass": b1 and b2}


def theta_fallback(mains: dict) -> dict:
    """Label-blind fallback (PREDICTIONS.md): pooled degenerate fraction over ALL models at theta=0.1."""
    thetas = list(CFG.theta_grid)
    frac = {th: float(np.mean([[r["degenerate_frac"] for r in m["pci_by_theta"] if r["theta"] == th][0]
                               for m in mains.values()])) for th in thetas}
    primary = CFG.theta_primary
    if frac[primary] >= CFG.degenerate_frac:
        ok = [th for th in thetas if frac[th] < CFG.degenerate_frac]
        primary = min(ok, key=lambda th: abs(np.log10(th) - np.log10(CFG.theta_primary))) if ok else None
    return {"pooled_degenerate_frac": frac, "primary_theta": primary, "moved": primary != CFG.theta_primary}


def pci_matrix(m, theta):
    if theta == CFG.theta_primary:
        return np.array(m["per_passage_dir_pci_primary"])
    if "per_passage_dir_pci_by_theta" in m:
        return np.array(m["per_passage_dir_pci_by_theta"][str(theta)])
    raise RuntimeError("fallback theta needs per-theta matrices: recompute from raw lattices")


def main() -> None:
    card: dict = {"env": env_info(), "PREDICTIONS": {}, "PREREG-2": {}}
    m1 = {d.name: load(d / "summary.json") for d in (R / "m1").iterdir() if (d / "summary.json").exists()}
    tr = m1.get(TRAINED)
    P = card["PREDICTIONS"]
    if tr:
        P["a"] = score_a(tr)
        P["a"]["pass"] = all(P["a"].values())
        P["b"] = {c: score_b(m1[c], tr["peak"]["value"]) if c in m1 else "pending" for c in CONTROLS}
        if all(isinstance(v, dict) for v in P["b"].values()):
            P["b"]["pass"] = all(v["pass"] for v in P["b"].values() if isinstance(v, dict))

    m2 = {d.name: load(d / "main.json") for d in (R / "m2").iterdir() if (d / "main.json").exists()} if (R / "m2").exists() else {}
    intact = {k: v for k, v in m2.items() if "__" not in k}
    needed = [TRAINED] + CONTROLS + EARLY
    if all(k in intact for k in needed):
        fb = theta_fallback({k: intact[k] for k in needed})
        P["theta_fallback"] = fb
        th = fb["primary_theta"]
        M = {k: pci_matrix(intact[k], th) for k in needed}
        c1 = {f"{TRAINED} - {c}": paired_boot(M[TRAINED], M[c]) for c in CONTROLS}
        c2 = {}
        for e in EARLY:
            c2[f"{e} - step0"] = paired_boot(M[e], M[CONTROLS[0]])
            c2[f"{TRAINED} - {e}"] = paired_boot(M[TRAINED], M[e])
        P["c"] = {"c1": c1, "c2": c2,
                  "c1_pass": all(v["ci"][0] > 0 for v in c1.values()),
                  "c2_pass": all(v["ci"][0] > 0 for v in c2.values())}
    else:
        P["c"] = {"pending": [k for k in needed if k not in intact]}
    if TRAINED in intact:
        z = intact[TRAINED]["z"]
        half = (z["pooled_z_ci"][1] - z["pooled_z_ci"][0]) / 2 if z["pooled_z"] else np.inf
        P["d"] = {
            "fittable_frac_ok": z["fittable_with_b_pos_frac"] >= CFG.d_min_fittable,
            "cv_ok": z["cv_z"] is not None and z["cv_z"] <= CFG.d_max_cv,
            "ci_halfwidth_ok": bool(z["pooled_z"]) and half <= CFG.d_max_ci_halfwidth * z["pooled_z"],
            "r2_ok": z["median_r2_power"] is not None and z["median_r2_power"] >= CFG.d_min_median_r2,
            "exp_wins_decisively_frac": z["frac_exp_wins_decisively"], "z": z,
        }
        P["d"]["pass"] = all(P["d"][k] for k in ("fittable_frac_ok", "cv_ok", "ci_halfwidth_ok", "r2_ok"))

    # ---- PREREG-2 A ----------------------------------------------------------------------------
    A = card["PREREG-2"]["A"] = {}
    base = intact.get(TRAINED)
    cond = lambda spec: m2.get(f"{TRAINED}__{spec}")
    if base and all(cond(f"window={w}") for w in CFG2.windows if w):
        seq = [base] + [cond(f"window={w}") for w in CFG2.windows if w]
        mats = [np.array(m["per_passage_dir_pci_primary"]) for m in seq]
        steps = [paired_boot(mats[i], mats[i + 1]) for i in range(len(mats) - 1)]
        A["A1"] = {"adjacent_differences": steps, "pass": all(s["ci"][0] > 0 for s in steps)}
    if base and all(cond(f"beta={b}") for b in CFG2.betas if b != 1.0):
        b0 = cond("beta=0")
        pci1, pci0 = np.array(base["per_passage_dir_pci_primary"]), np.array(b0["per_passage_dir_pci_primary"])
        r1, r0 = np.array(base["per_passage_dir_reach_primary"]), np.array(b0["per_passage_dir_reach_primary"])
        drop = paired_boot(pci1, pci0)
        stacked1, stacked0 = np.stack([pci1, r1], -1), np.stack([pci0, r0], -1)
        ratio = paired_boot(stacked0, stacked1, fn=lambda a, b: a[..., 0].mean() / b[..., 0].mean() - a[..., 1].mean() / b[..., 1].mean())
        A["A2"] = {"pci_intact_minus_beta0": drop, "pci_ratio_minus_reach_ratio": ratio,
                   "pass": drop["ci"][0] > 0 and ratio["ci"][1] < 0}
        pts = [base["pci_primary"]["pci"]] + [cond(f"beta={b}")["pci_primary"]["pci"] for b in CFG2.betas if b != 1.0]
        d25 = paired_boot(pci1, np.array(cond("beta=0.25")["per_passage_dir_pci_primary"]))
        A["A3"] = {"pci_by_beta": dict(zip([str(b) for b in CFG2.betas], pts)), "pci1_minus_pci025": d25,
                   "pass": all(pts[i] >= pts[i + 1] for i in range(len(pts) - 1)) and d25["ci"][0] > 0}
        lr = b0["local_response"] / base["local_response"]
        A["A4"] = {"local_response_ratio": lr, "pass": lr >= CFG2.local_response_floor}

    # ---- PREREG-2 B ----------------------------------------------------------------------------
    B = card["PREREG-2"]["B"] = {}
    cf = {d.name: load(d / "summary.json") for d in (R / "conflict").iterdir()} if (R / "conflict").exists() else {}
    if TRAINED in cf:
        t = cf[TRAINED]
        B["B1"] = {**t["B1"], "pass": t["B1"].get("testable", False) and t["B1"]["ci"][0] > 0}
        B["B2"] = {**t["B2"], "pass": t["B2"]["ci"][0] > 0}
        B["B3_exploratory"] = t["B3_exploratory"]
        b4 = {}
        for c in CONTROLS:
            if c in cf:
                cc = cf[c]
                ok1 = (not cc["B1"].get("testable", False)) or cc["B1"]["ci"][0] <= 0 <= cc["B1"]["ci"][1]
                ok2 = cc["B2"]["ci"][0] <= 0 <= cc["B2"]["ci"][1]
                b4[c] = {"B1_null": ok1, "B2_null": ok2, "pass": ok1 and ok2}
        B["B4"] = b4 or "pending"

    (R / "scorecard.json").write_text(json.dumps(card, indent=1, default=float))
    lines = ["# Scorecard (mechanical; see scripts/s4_compare.py)", ""]
    def emit(name, d):
        if isinstance(d, dict) and "pass" in d:
            lines.append(f"- **{name}**: {'PASS' if d['pass'] else 'FAIL'}")
    for k, v in P.items():
        if k == "a":
            for sub in ("a1", "a2-i", "a2-ii", "a2-iii", "a3"):
                lines.append(f"- a / {sub}: {'pass' if v[sub] else 'fail'}")
        emit(f"PREDICTIONS ({k})", v)
    for part in ("A", "B"):
        for k, v in card["PREREG-2"][part].items():
            emit(f"PREREG-2 {k}", v)
    (R / "SCORECARD.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
