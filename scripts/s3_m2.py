"""Measurement 2 (PCI analogue) for one model variant, optionally under an attention manipulation.

  uv run python scripts/s3_m2.py --step 143000 [--shuffle-seed S] [--manip none|beta=x|window=w]
                                 --phase calib|main|gating|all --device cuda

Phases (definitions in PREDICTIONS.md):
  calib   linear-regime epsilon* from float32 vs float64 finite differences (passages 0-7, 4 dirs)
  main    gain lattices for 64 passages x 12 k x 3 p x 8 dirs at epsilon*; PCI over the theta grid;
          spread S(t), participation spread, power/exponential fits, z; crossed bootstrap CIs
  gating  exploratory: injection at the ambiguous target before/after the trained model's L*
Outputs: results/m2/<name>/{calib,main,gating}.json ; raw lattices in results/raw/.
"""

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import torch

from rgc import stimuli
from rgc.attention_manip import manipulation_hooks
from rgc.config import CFG
from rgc.determinism import REPO, default_device, env_info, set_determinism
from rgc.m1_order import word_index
from rgc.m2_perturb import center_unit, perturb_response, unit_directions
from rgc.models import Variant, load
from rgc.pci import lattice_bits, normalised_lz, pci_task
from rgc.spread import fit_spread, participation_spread, spread_counts
from rgc.stats import ci
from rgc.wikitext import load_passages

TRAINED_M1 = REPO / "results" / "m1" / f"{CFG.model_name}_step{CFG.final_step}" / "summary.json"


def seq(ids: list[int], bos: int) -> torch.Tensor:
    return torch.tensor([bos] + list(ids), dtype=torch.long)


# ---------------------------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------------------------

def calibrate(model, model64, hooks, passages, dirs) -> dict:
    nL, eps_grid = model.cfg.n_layers, np.array(CFG.eps_grid)
    devs = {e: [] for e in range(len(eps_grid))}
    agree = {e: [0, 0] for e in range(len(eps_grid))}
    bos = model.tokenizer.bos_token_id
    for ids in passages:
        tokens = seq(ids, bos)
        for k in range(nL):
            P, U, E = [], [], []
            for p in CFG.p_grid:
                for u in dirs:
                    for e in eps_grid:
                        P.append(p); U.append(u); E.append(e)
            P, U, E = torch.tensor(P), torch.stack(U), torch.tensor(E, dtype=torch.float32)
            G32 = (perturb_response(model, tokens, k, P, U, E, extra_hooks=hooks) / E.to(model.cfg.device)[:, None, None]).cpu().numpy()
            P0 = torch.tensor([p for p in CFG.p_grid for _ in dirs])
            U0 = torch.stack([u for _ in CFG.p_grid for u in dirs])
            E0 = torch.full((len(P0),), CFG.eps_ref_f64, dtype=torch.float32)
            Gref = (perturb_response(model64, tokens, k, P0, U0.double(), E0, extra_hooks=hooks)
                    / CFG.eps_ref_f64).cpu().numpy()
            G32 = G32.reshape(len(CFG.p_grid), len(dirs), len(eps_grid), *G32.shape[1:])
            Gref = Gref.reshape(len(CFG.p_grid), len(dirs), *Gref.shape[1:])
            for ip, p in enumerate(CFG.p_grid):
                ref = Gref[ip][:, 1:, p + 1 :]  # [dirs, slices>k, q>p]
                sel = ref >= CFG.linear_site_min_gain
                for ie in range(len(eps_grid)):
                    g = G32[ip, :, ie][:, 1:, p + 1 :]
                    devs[ie].append((np.abs(g - ref) / ref)[sel].astype(np.float32))
                    b32, bref = g >= CFG.theta_primary, ref >= CFG.theta_primary
                    agree[ie][0] += int((b32 == bref).sum()); agree[ie][1] += b32.size
    rows = []
    for ie, e in enumerate(eps_grid):
        d = np.concatenate(devs[ie]) if devs[ie] else np.array([np.nan])
        rows.append({"eps": float(e), "median_rel_dev": float(np.median(d)), "p95_rel_dev": float(np.percentile(d, 95)),
                     "n_sites": int(d.size), "binary_agreement_theta_primary": agree[ie][0] / max(agree[ie][1], 1)})
    ok = [r["eps"] for r in rows if r["median_rel_dev"] <= CFG.linear_tol]
    return {"rows": rows, "eps_star": max(ok) if ok else None}


# ---------------------------------------------------------------------------------------------
# main grid
# ---------------------------------------------------------------------------------------------

def main_grid(model, hooks, passages, dirs, eps: float) -> dict:
    nL, bos = model.cfg.n_layers, model.tokenizer.bos_token_id
    lattices = {k: [] for k in range(nL)}  # k -> [n, p, d, S_k, T]
    for ids in passages:
        tokens = seq(ids, bos)
        for k in range(nL):
            P = torch.tensor([p for p in CFG.p_grid for _ in dirs])
            U = torch.stack([u for _ in CFG.p_grid for u in dirs])
            E = torch.full((len(P),), eps, dtype=torch.float32)
            G = (perturb_response(model, tokens, k, P, U, E, extra_hooks=hooks) / eps).cpu().numpy()
            lattices[k].append(G.reshape(len(CFG.p_grid), len(dirs), *G.shape[1:]))
    return {k: np.stack(v) for k, v in lattices.items()}


def analyse_main(lat: dict, nL: int, workers: int) -> dict:
    N, nP, nD = lat[0].shape[:3]
    thetas = list(CFG.theta_grid)
    tasks, index = [], []
    for k in range(nL):
        for n in range(N):
            for ip, p in enumerate(CFG.p_grid):
                for d in range(nD):
                    tasks.append((lat[k][n, ip, d], p, thetas)); index.append((k, n, ip, d))
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(pci_task, tasks, chunksize=64)
    else:
        res = [pci_task(t) for t in tasks]
    PCI = np.zeros((nL, N, nP, nD, len(thetas)))
    P1 = np.zeros_like(PCI)
    for (k, n, ip, d), r in zip(index, res):
        PCI[k, n, ip, d] = [x[0] for x in r]
        P1[k, n, ip, d] = [x[1] for x in r]

    ti = thetas.index(CFG.theta_primary)
    kmax = nL - CFG.min_downstream_layers  # k <= n_layers - 3
    rng = np.random.default_rng(CFG.boot_seed)
    idx_n = rng.integers(0, N, size=(CFG.n_boot, N))
    idx_d = rng.integers(0, nD, size=(CFG.n_boot, nD))

    def crossed_boot(M):  # M: [N, nD] -> bootstrap means
        return np.array([M[idx_n[b]][:, idx_d[b]].mean() for b in range(CFG.n_boot)])

    per_nd = PCI[: kmax + 1, :, :, :, :].mean(axis=(0, 2))  # [N, nD, theta]
    # PREREG-2 (A): reach = mean p1 over the primary lattices; local response = self-column gain at
    # the final slice (the perturbed position's own response), both per (passage, direction).
    reach_nd = P1[: kmax + 1, ..., ti].mean(axis=(0, 2))  # [N, nD]
    local_nd = np.mean([[lat[k][:, ip, :, -1, p] for ip, p in enumerate(CFG.p_grid)] for k in range(kmax + 1)],
                       axis=(0, 1))  # [N, nD]
    pci_theta = []
    for j, th in enumerate(thetas):
        boot = crossed_boot(per_nd[:, :, j])
        lo, hi = ci(boot)
        pci_theta.append({"theta": th, "pci": float(per_nd[:, :, j].mean()), "ci": [float(lo), float(hi)],
                          "degenerate_frac": float(((P1[..., j] < CFG.degenerate_p1_lo) | (P1[..., j] > CFG.degenerate_p1_hi)).mean()),
                          "mean_p1": float(P1[..., j].mean())})
    logt = np.log10(thetas)
    auc_nd = np.trapezoid(per_nd, logt, axis=-1) / (logt[-1] - logt[0])
    auc_boot = crossed_boot(auc_nd)

    # spread at every cell, primary theta
    p_idx = {p: i for i, p in enumerate(CFG.p_grid)}
    spread = {}
    for k in range(nL):
        for p in CFG.p_grid:
            G = lat[k][:, p_idx[p]]  # [N, nD, S_k, T]
            S = spread_counts(G, p, CFG.theta_primary).mean(1)  # [N, t]
            Seff = participation_spread(G, p).mean(1)
            n_avail = G.shape[-1] - 1 - p
            fits = [fit_spread(S[n], n_avail, CFG.saturation_frac, CFG.fit_min_points) for n in range(N)]
            pooled = fit_spread(S.mean(0), n_avail, CFG.saturation_frac, CFG.fit_min_points)
            spread[f"k{k}_p{p}"] = {"S_mean": S.mean(0).tolist(), "S_eff_mean": Seff.mean(0).tolist(),
                                    "n_available": n_avail, "pooled_fit": pooled, "per_passage": fits}
    prim = spread[f"k{CFG.z_primary_k}_p{CFG.z_primary_p}"]
    G = lat[CFG.z_primary_k][:, p_idx[CFG.z_primary_p]]
    n_avail = prim["n_available"]
    zs = [f.get("z") for f in prim["per_passage"]]
    z_ok = np.array([z for z in zs if z is not None])
    boot_z = []
    S_nd = spread_counts(G, CFG.z_primary_p, CFG.theta_primary)  # [N, nD, t]
    for b in range(CFG.n_boot):
        Sb = S_nd[idx_n[b]][:, idx_d[b]].mean(axis=(0, 1))
        f = fit_spread(Sb, n_avail, CFG.saturation_frac, CFG.fit_min_points)
        boot_z.append(f.get("z") if f.get("z") is not None else np.nan)
    boot_z = np.array(boot_z)
    zlo, zhi = (ci(boot_z[np.isfinite(boot_z)]) if np.isfinite(boot_z).any() else (np.nan, np.nan))
    r2 = [f["r2_power"] for f in prim["per_passage"] if f.get("fittable")]
    daic = [f["delta_aic_exp_minus_power"] for f in prim["per_passage"] if f.get("fittable")]
    z_summary = {
        "pooled_z": prim["pooled_fit"].get("z"), "pooled_z_ci": [float(zlo), float(zhi)],
        "boot_z_finite_frac": float(np.isfinite(boot_z).mean()),
        "fittable_with_b_pos_frac": float(len(z_ok) / N),
        "cv_z": float(z_ok.std(ddof=1) / z_ok.mean()) if len(z_ok) > 1 else None,
        "median_r2_power": float(np.median(r2)) if r2 else None,
        "frac_exp_wins_decisively": float(np.mean(np.array(daic) < -CFG.aic_decisive)) if daic else None,
        "frac_power_wins_decisively": float(np.mean(np.array(daic) > CFG.aic_decisive)) if daic else None,
    }
    ip, pz = p_idx[CFG.z_primary_p], CFG.z_primary_p
    self_col = {k: [float(lat[k][:, ip, :, s, pz].mean()) for s in range(lat[k].shape[3])] for k in range(nL)}
    return {
        "pci_by_theta": pci_theta,
        "pci_primary": pci_theta[ti],
        "pci_auc_logtheta": {"value": float(auc_nd.mean()), "ci": [float(x) for x in ci(auc_boot)]},
        "pci_by_k_primary_theta": PCI[..., ti].mean(axis=(1, 2, 3)).tolist(),
        "pci_by_p_primary_theta": PCI[: kmax + 1, ..., ti].mean(axis=(0, 1, 3)).tolist(),
        "per_passage_dir_pci_primary": per_nd[:, :, ti].tolist(),  # for paired comparisons in s4
        "per_passage_dir_pci_by_theta": {str(th): per_nd[:, :, j].tolist() for j, th in enumerate(thetas)},
        "per_passage_dir_reach_primary": reach_nd.tolist(),
        "per_passage_dir_local_response": local_nd.tolist(),
        "reach_primary": float(reach_nd.mean()),
        "local_response": float(local_nd.mean()),
        "per_passage_dir_auc": auc_nd.tolist(),
        "z": z_summary,
        "spread": spread,
        "self_column_gain_k_p64": self_col,
    }


@torch.no_grad()
def next_token_ce(model, hooks, passages) -> list[float]:
    """Mean next-token cross-entropy per passage (descriptive manipulation check, PREREG-2 A)."""
    bos, out = model.tokenizer.bos_token_id, []
    for i in range(0, len(passages), 8):
        toks = torch.stack([seq(ids, bos) for ids in passages[i : i + 8]]).to(model.cfg.device)
        logits = model.run_with_hooks(toks, fwd_hooks=list(hooks))
        ce = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2).float(), toks[:, 1:], reduction="none")
        out += ce.mean(1).tolist()
    return out


# ---------------------------------------------------------------------------------------------
# gating (exploratory)
# ---------------------------------------------------------------------------------------------

def gating(model, hooks, dirs, eps: float, xt_path) -> dict:
    s_tr = json.loads(TRAINED_M1.read_text())
    L_star = s_tr["L_star"]
    nL, bos_dev = model.cfg.n_layers, model.cfg.device
    W = CFG.gating_window
    k_b = max(0, L_star + CFG.gating_before)
    k_a = L_star + CFG.gating_after
    if k_a + W > nL:
        W = nL - k_a
    prompts = [p for p in stimuli.load() if p.split in ("train", "test")]
    Xt = np.load(xt_path)["Xt"] if xt_path.exists() else None
    widx = {w.word: w for w in word_index(stimuli.load())}
    out = {"L_star_trained": L_star, "k_before": k_b, "k_after": k_a, "window": W,
           "clipped": L_star + CFG.gating_before < 0, "conditions": {}}
    T = max(p.n_tokens for p in prompts)
    toks = torch.zeros(len(prompts), T, dtype=torch.long)
    for i, p in enumerate(prompts):
        toks[i, : p.n_tokens] = torch.tensor(p.token_ids)
    for name, k in (("before", k_b), ("after", k_a)):
        res = {}
        dir_sets = [("random", d, u) for d, u in enumerate(dirs)]
        if Xt is not None and k > 0:
            dir_sets.append(("sense", None, None))
        for kind, d, u in dir_sets:
            pci_vals, s_vals = [], []
            for b0 in range(0, len(prompts), 64):
                chunk = prompts[b0 : b0 + 64]
                R = len(chunk)
                P = torch.tensor([p.target_pos for p in chunk])
                if kind == "random":
                    U = u[None].expand(R, -1)
                else:
                    U = []
                    for p in chunk:
                        w = widx[p.word]
                        v = Xt[w.trA, k].mean(0) - Xt[w.trB, k].mean(0)
                        U.append(center_unit(torch.tensor(v)))
                    U = torch.stack(U)
                E = torch.full((R,), eps, dtype=torch.float32)
                G = (perturb_response(model, toks[b0 : b0 + R], k, P, U, E, extra_hooks=hooks) / eps).cpu().numpy()
                for i, p in enumerate(chunk):
                    g = G[i, : W + 1, : p.n_tokens]  # injection slice + W downstream slices, real tokens only
                    pci_vals.append(normalised_lz(lattice_bits(g, p.target_pos, CFG.theta_primary)))
                    s_vals.append(float(spread_counts(g, p.target_pos, CFG.theta_primary).mean()))
            res[f"{kind}{'' if d is None else d}"] = {"pci": pci_vals, "spread": s_vals}
        out["conditions"][name] = res
    ids = [p.id for p in prompts]
    out["prompt_ids"] = ids
    # contrasts after - before (random dirs pooled), bootstrap over minimal pairs x dirs
    rnd = [k for k in out["conditions"]["before"] if k.startswith("random")]
    for metric in ("pci", "spread"):
        B = np.array([out["conditions"]["before"][k][metric] for k in rnd])  # [dirs, prompts]
        A = np.array([out["conditions"]["after"][k][metric] for k in rnd])
        D = A - B
        pair_key = [(p.word, p.split, p.pair) for p in prompts]
        pairs = sorted(set(pair_key))
        pid = np.array([pairs.index(x) for x in pair_key])
        members = [np.where(pid == q)[0] for q in range(len(pairs))]
        rng = np.random.default_rng(CFG.boot_seed + 3)
        boots = []
        for _ in range(CFG.n_boot):
            ps = rng.integers(0, len(pairs), len(pairs))
            ds = rng.integers(0, D.shape[0], D.shape[0])
            mask = np.concatenate([members[q] for q in ps])
            boots.append(D[ds][:, mask].mean())
        out[f"contrast_after_minus_before_{metric}_random"] = {"mean": float(D.mean()), "ci": [float(x) for x in ci(np.array(boots))]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=CFG.final_step)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--manip", default="none")
    ap.add_argument("--phase", default="all")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--n-passages", type=int, default=CFG.n_passages)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    set_determinism(CFG.master_seed)
    v = Variant(args.step, args.shuffle_seed)
    name = v.label.replace("@", "_") + ("" if args.manip == "none" else f"__{args.manip}")
    out_dir = REPO / "results" / "m2" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = REPO / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    model = load(v, device=args.device)
    hooks = manipulation_hooks(args.manip, model.cfg.n_layers)
    dirs = list(unit_directions(model.cfg.d_model, CFG.n_dirs, CFG.dir_seed))
    passages = load_passages()[: args.n_passages]
    phases = ["calib", "main", "gating"] if args.phase == "all" else [args.phase]
    meta = {"variant": v.label, "manip": args.manip, "env": env_info()}

    if "calib" in phases:
        t0 = time.time()
        model64 = load(v, device=args.device).to(torch.float64)
        model64.cfg.dtype = torch.float64
        probe = torch.tensor([model.tokenizer.bos_token_id] + passages[0][:64])[None].to(args.device)
        with torch.no_grad():
            l32, l64 = model(probe).double(), model64(probe)
        f64_check = float(((l32 - l64).abs().max() / l64.abs().max()).item())
        assert l64.dtype == torch.float64 and f64_check < 1e-4, f64_check
        cal = calibrate(model, model64, hooks, passages[: CFG.n_calib_passages], dirs[: CFG.n_calib_dirs])
        cal["f32_vs_f64_logit_rel_maxdiff"] = f64_check
        del model64
        (out_dir / "calib.json").write_text(json.dumps({**meta, **cal, "seconds": time.time() - t0}, indent=1))
        print(f"calib done eps*={cal['eps_star']}")
    eps = json.loads((out_dir / "calib.json").read_text())["eps_star"]
    if "main" in phases:
        t0 = time.time()
        lat = main_grid(model, hooks, passages, dirs, eps)
        np.savez_compressed(raw / f"m2_{name}.npz", **{f"k{k}": v for k, v in lat.items()})
        res = analyse_main(lat, model.cfg.n_layers, args.workers)
        ce = next_token_ce(model, hooks, passages)
        res.update({"ce_mean": float(np.mean(ce)), "ce_per_passage": ce})
        (out_dir / "main.json").write_text(json.dumps({**meta, "eps_star": eps, **res, "seconds": time.time() - t0}, indent=1))
        print("main done")
    if "gating" in phases:
        t0 = time.time()
        g = gating(model, hooks, dirs, eps, raw / f"m1_{v.label.replace('@', '_')}.npz")
        (out_dir / "gating.json").write_text(json.dumps({**meta, "eps_star": eps, **g, "seconds": time.time() - t0}, indent=1))
        print("gating done")


if __name__ == "__main__":
    main()
