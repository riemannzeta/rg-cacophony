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
from rgc.config_prereg3 import CFG3
from rgc.config_prereg4 import CFG4
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
    N, D = A.shape[:2]
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
    bspec = lambda b: f"beta={b:g}"  # directory names: beta=0.5, beta=0.25, beta=0
    if base and all(cond(bspec(b)) for b in CFG2.betas if b != 1.0):
        b0 = cond("beta=0")
        pci1, pci0 = np.array(base["per_passage_dir_pci_primary"]), np.array(b0["per_passage_dir_pci_primary"])
        r1, r0 = np.array(base["per_passage_dir_reach_primary"]), np.array(b0["per_passage_dir_reach_primary"])
        drop = paired_boot(pci1, pci0)
        stacked1, stacked0 = np.stack([pci1, r1], -1), np.stack([pci0, r0], -1)
        ratio = paired_boot(stacked0, stacked1, fn=lambda a, b: a[..., 0].mean() / b[..., 0].mean() - a[..., 1].mean() / b[..., 1].mean())
        A["A2"] = {"pci_intact_minus_beta0": drop, "pci_ratio_minus_reach_ratio": ratio,
                   "pass": drop["ci"][0] > 0 and ratio["ci"][1] < 0}
        pts = [base["pci_primary"]["pci"]] + [cond(bspec(b))["pci_primary"]["pci"] for b in CFG2.betas if b != 1.0]
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

    # ---- PREREG-2 C and D (exploratory: reported with paired CIs, not scored) -------------------
    C = card["PREREG-2"]["C_exploratory"] = {}
    for mamba_repo, pythia in CFG2.mamba_pairs:
        mk, pk = mamba_repo.split("/")[-1], f"{pythia}_step{CFG.final_step}"
        if mk in m2 and pk in m2:
            C[f"{mk} - {pythia}"] = paired_boot(np.array(m2[mk]["per_passage_dir_pci_primary"]),
                                                np.array(m2[pk]["per_passage_dir_pci_primary"]))
    mk = CFG2.mamba_pairs[0][0].split("/")[-1]
    for sd in CFG2.mamba_shuffle_seeds:
        if mk in m2 and f"{mk}-shuffled{sd}" in m2:
            C[f"{mk} - shuffled{sd}"] = paired_boot(np.array(m2[mk]["per_passage_dir_pci_primary"]),
                                                    np.array(m2[f"{mk}-shuffled{sd}"]["per_passage_dir_pci_primary"]))
    D = card["PREREG-2"]["D_exploratory"] = {}
    zs = {m: m2.get(f"{m}_step{CFG.final_step}") for m in CFG2.scale_models}
    for m, mm in zs.items():
        if mm:
            D[m] = {"z": mm["z"]["pooled_z"], "ci": mm["z"]["pooled_z_ci"], "cv": mm["z"]["cv_z"],
                    "median_r2": mm["z"]["median_r2_power"]}
    names = [m for m, mm in zs.items() if mm and "boot_z" in mm["z"]]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.array([np.nan if v is None else v for v in zs[names[i]]["z"]["boot_z"]], float)
            b = np.array([np.nan if v is None else v for v in zs[names[j]]["z"]["boot_z"]], float)
            d = (a - b)[np.isfinite(a - b)]
            if d.size:
                lo, hi = ci(d)
                D[f"z {names[i]} - {names[j]}"] = {"ci": [float(lo), float(hi)], "n_finite": int(d.size)}

    # ---- PREREG-3 (fresh-data replication at theta = 0.01) -------------------------------------
    F = card["PREREG-3"] = {}
    fr = REPO / CFG3.out_root
    fm = {d.name: load(d / "main.json") for d in fr.iterdir() if (d / "main.json").exists()} if fr.exists() else {}
    th = str(CFG3.theta_primary)
    get = lambda spec: fm.get(TRAINED if spec is None else f"{TRAINED}__{spec}")
    names = [None] + [f"beta={b:g}" for b in CFG3.betas if b != 1.0] + [f"window={w}" for w in CFG3.windows_run]
    if all(get(n) for n in names):
        pci = {n: np.array(get(n)["per_passage_dir_pci_by_theta"][th]) for n in names}
        reach = {n: np.array(get(n)["per_passage_dir_reach_by_theta"][th]) for n in names}
        seff = {n: np.array(get(n)["per_passage_dir_seff"]) for n in names}
        r1 = paired_boot(pci[None], pci["beta=0"])
        st1, st0 = np.stack([pci[None], reach[None]], -1), np.stack([pci["beta=0"], reach["beta=0"]], -1)
        ratio = paired_boot(st0, st1, fn=lambda a, b: a[..., 0].mean() / b[..., 0].mean() - a[..., 1].mean() / b[..., 1].mean())
        sdiff = paired_boot(seff["beta=0"], seff[None])
        pts = [float(pci[n].mean()) for n in (None, "beta=0.5", "beta=0.25", "beta=0")]
        r3 = paired_boot(pci[None], pci["beta=0"])
        steps = [paired_boot(pci[a], pci[b]) for a, b in ((None, "window=64"), ("window=64", "window=16"))]
        F["R1"] = {"pci_intact_minus_beta0": r1, "pass": r1["ci"][0] > 0}
        F["R2"] = {"pci_ratio_minus_reach_ratio": ratio, "seff_beta0_minus_intact": sdiff,
                   "pass": ratio["ci"][1] < 0 and sdiff["ci"][0] > 0}
        F["R3"] = {"pci_by_beta": dict(zip(["1", "0.5", "0.25", "0"], pts)), "pci1_minus_pci0": r3,
                   "pass": all(pts[i] > pts[i + 1] for i in range(3)) and r3["ci"][0] > 0}
        F["R4"] = {"adjacent": steps, "pass": all(s_["ci"][0] > 0 for s_ in steps)}
        dm = {n: np.array(get(n)["per_passage_dir_pci_dm"]) for n in (None, "beta=0")}
        F["secondary_density_matched"] = {"pci_dm_intact_minus_beta0": paired_boot(dm[None], dm["beta=0"])}
        F["levels"] = {str(n): {"pci": float(pci[n].mean()), "reach": float(reach[n].mean()), "seff": float(seff[n].mean())}
                       for n in names}
    else:
        F["pending"] = [str(n) for n in names if not get(n)]

    # ---- PREREG-4 (criticality and chimera states, second fresh set) ---------------------------
    G4 = card["PREREG-4"] = {}
    f2 = REPO / CFG4.out_root
    m4 = {d.name: load(d / "main.json") for d in f2.iterdir() if (d / "main.json").exists()} if f2.exists() else {}
    th4 = str(CFG4.theta)
    cond4 = lambda base, b: m4.get(base if b == 1.0 else f"{base}__beta={b:g}")
    tr4 = TRAINED
    sh4 = f"{TRAINED}-shuffled1001"
    sweep_ok = lambda base: all(cond4(base, b) for b in CFG4.betas)
    pci4 = lambda m: np.array(m["per_passage_dir_pci_by_theta"][th4])
    lam4 = lambda m: np.array(m["per_passage_dir_lambda"])
    chi4 = lambda m: np.array(m["chimera"]["blocks16"]["chimera"])[:, None]
    if sweep_ok(tr4):
        P_ = {b: pci4(cond4(tr4, b)) for b in CFG4.betas}
        L_ = {b: lam4(cond4(tr4, b)) for b in CFG4.betas}
        C_ = {b: chi4(cond4(tr4, b)) for b in CFG4.betas}
        d1, d2 = paired_boot(P_[1.0], P_[0.25]), paired_boot(P_[1.0], P_[4.0])
        G4["T1"] = {"pci1_minus_pci025": d1, "pci1_minus_pci4": d2, "pass": d1["ci"][0] > 0 and d2["ci"][0] > 0}
        means = {b: float(P_[b].mean()) for b in CFG4.betas}
        G4["T2"] = {"pci_by_beta": means, "argmax": max(means, key=means.get),
                    "pass": max(means, key=means.get) in CFG4.near_natural}
        lm = {b: float(np.nanmean(L_[b])) for b in CFG4.betas}
        mono = all(lm[a] < lm[b] for a, b in zip(CFG4.betas, CFG4.betas[1:]))
        amin = min(lm, key=lambda b: abs(lm[b]))
        G4["T3_exploratory"] = {"lambda_by_beta": lm, "monotone_increasing": mono, "argmin_abs": amin,
                                "would_meet_draft_criterion": mono and amin in CFG4.near_natural}
        c1a, c1b = paired_boot(C_[1.0], C_[0.25]), paired_boot(C_[1.0], C_[4.0])
        G4["C2"] = {"chi1_minus_chi025": c1a, "chi1_minus_chi4": c1b, "pass": c1a["ci"][0] > 0 and c1b["ci"][0] > 0}
        from scipy import stats as _st
        G4["exploratory_spearman_pci_chi_over_beta"] = float(_st.spearmanr([means[b] for b in CFG4.betas],
                                                                            [float(C_[b].mean()) for b in CFG4.betas])[0])
        G4["exploratory_pci_dm_by_beta"] = {b: float(np.mean(cond4(tr4, b)["per_passage_dir_pci_dm"])) for b in CFG4.betas}
        G4["exploratory_chi_by_beta"] = {b: float(C_[b].mean()) for b in CFG4.betas}
    ctrl4 = [f"{CFG.model_name}_step0"] + [f"{TRAINED}-shuffled{sd}" for sd in CFG.shuffle_seeds]
    if m4.get(tr4) and all(m4.get(c) for c in ctrl4):
        absdiff = lambda a, b: abs(np.nanmean(a)) - abs(np.nanmean(b))
        s1 = {c: paired_boot(np.nan_to_num(lam4(m4[c]), nan=np.nanmean(lam4(m4[c]))),
                             np.nan_to_num(lam4(m4[tr4]), nan=np.nanmean(lam4(m4[tr4]))), fn=absdiff) for c in ctrl4}
        G4["S1"] = {"abs_lambda_control_minus_trained": s1, "lambda": {c: float(np.nanmean(lam4(m4[c]))) for c in [tr4] + ctrl4},
                    "pass": all(v["ci"][0] > 0 for v in s1.values())}
        c1 = {c: paired_boot(chi4(m4[tr4]), chi4(m4[c])) for c in ctrl4}
        G4["C1"] = {"chi_trained_minus_control": c1, "pass": all(v["ci"][0] > 0 for v in c1.values())}
        lo, hi = CFG4.waist_middle
        waist = lambda m: 1 + int(np.argmin(np.array(m["chimera"]["blocks16"]["eff_dim_mean"])[1:12]))
        wt, w0 = waist(m4[tr4]), waist(m4[ctrl4[0]])
        G4["C3"] = {"waist_trained": wt, "waist_step0": w0, "pass": lo <= wt <= hi and not (lo <= w0 <= hi)}
    if sweep_ok(sh4):
        sm = {b: float(pci4(cond4(sh4, b)).mean()) for b in CFG4.betas}
        G4["S2"] = {"shuffled_pci_by_beta": sm, "argmax": max(sm, key=sm.get),
                    "pass": max(sm, key=sm.get) not in CFG4.near_natural}
    if not G4:
        G4["pending"] = True

    (R / "scorecard.json").write_text(json.dumps(card, indent=1, default=float))
    lines = ["# Scorecard (mechanical; see scripts/s4_compare.py)", ""]
    def emit(name, d):
        if isinstance(d, dict) and "pass" in d:
            lines.append(f"- **{name}**: {'PASS' if d['pass'] else 'FAIL'}")
    if isinstance(P.get("c"), dict) and "c1_pass" in P["c"]:
        lines.append(f"- c / c1 (trained > step 0 and > each shuffle): {'pass' if P['c']['c1_pass'] else 'fail'}")
        lines.append(f"- c / c2 (early checkpoints in between): {'pass' if P['c']['c2_pass'] else 'fail'}")
        P["c"]["pass"] = P["c"]["c1_pass"] and P["c"]["c2_pass"]
    for k, v in P.items():
        if k == "a":
            for sub in ("a1", "a2-i", "a2-ii", "a2-iii", "a3"):
                lines.append(f"- a / {sub}: {'pass' if v[sub] else 'fail'}")
        emit(f"PREDICTIONS ({k})", v)
    for part in ("A", "B"):
        for k, v in card["PREREG-2"][part].items():
            emit(f"PREREG-2 {k}", v)
    b4 = card["PREREG-2"]["B"].get("B4")
    if isinstance(b4, dict) and b4:
        lines.append(f"- **PREREG-2 B4**: {'PASS' if all(v['pass'] for v in b4.values()) else 'FAIL'}")
    if card["PREREG-2"].get("C_exploratory"):
        lines.append("- PREREG-2 C: exploratory (no prediction); see scorecard.json")
    if card["PREREG-2"].get("D_exploratory"):
        lines.append("- PREREG-2 D: exploratory; not testable because (d) failed (z undefined)")
    for k, v in card["PREREG-3"].items():
        emit(f"PREREG-3 {k}", v)
    for k, v in card["PREREG-4"].items():
        emit(f"PREREG-4 {k}", v)
    (R / "SCORECARD.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
