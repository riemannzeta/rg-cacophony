"""Measurement 1 for one model variant: Δd′ across depth, attention, head contributions, ablation.

  uv run python scripts/s2_m1.py --step 143000 [--shuffle-seed 1001] [--device cuda]

Writes results/m1/<variant>/summary.json and results/raw/m1_<variant>.npz.
Pass/fail scoring against PREDICTIONS.md happens in s4_compare.py (needs all variants).
"""

import argparse
import json
import time

import numpy as np
import torch

from rgc import stimuli
from rgc.config import CFG
from rgc.determinism import REPO, default_device, env_info, set_determinism
from rgc.m1_ablation import final_delta, layer_matched_sets, random_sets, select_heads
from rgc.m1_order import (collect, directions, dprime_boot, dprime_curve, head_contributions,
                          head_means, resample_counts, word_index)
from rgc.models import Variant, load
from rgc.stats import ci, hypergeom_tail, shortest_band, spearman_perm


def run(model, variant_label: str, n_boot: int, n_null: int, batch_size: int, out_dir, raw_dir) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(CFG.boot_seed)
    prompts = stimuli.load()
    widx = word_index(prompts)
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    S = nL + 1

    T = collect(model, prompts, control=False, attention=True, batch_size=batch_size)
    C = collect(model, prompts, control=True, attention=False, batch_size=batch_size)
    Xt, Xc = T["X"], C["X"]

    # ---- d′ curves (point) ------------------------------------------------------------------
    dt = np.stack([dprime_curve(Xt, w) for w in widx])  # [W, S]
    dc = np.stack([dprime_curve(Xc, w) for w in widx])
    dd = dt - dc
    pooled = {"target": dt.mean(0), "control": dc.mean(0), "delta": dd.mean(0)}

    # ---- bootstrap (pairs resampled within word; same counts for target and control) ---------
    bt = np.zeros((n_boot, len(widx), S))
    bc = np.zeros_like(bt)
    neu_t_boot = np.zeros((n_boot, len(widx)))
    for wi, w in enumerate(widx):
        c_tr = resample_counts(len(w.trA), n_boot, rng)
        c_te = resample_counts(len(w.teA), n_boot, rng)
        c_ne = resample_counts(len(w.neu), n_boot, rng)
        bt[:, wi] = dprime_boot(Xt, w, c_tr, c_te)
        bc[:, wi] = dprime_boot(Xc, w, c_tr, c_te)
        # neutral position at the final slice, same resamples
        s = nL
        V = (c_tr @ Xt[w.trA, s] - c_tr @ Xt[w.trB, s]) / len(w.trA)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        mA = (c_te * (Xt[w.teA, s] @ V.T).T).sum(1) / len(w.teA)
        mB = (c_te * (Xt[w.teB, s] @ V.T).T).sum(1) / len(w.teB)
        mN = (c_ne * (Xt[w.neu, s] @ V.T).T).sum(1) / len(w.neu)
        neu_t_boot[:, wi] = (mN - mB) / (mA - mB)
    bdelta = (bt - bc).mean(1)  # [B, S] pooled over words
    boot_pooled = {"target": bt.mean(1), "control": bc.mean(1), "delta": bdelta}
    lo = {k: ci(v)[0] for k, v in boot_pooled.items()}
    hi = {k: ci(v)[1] for k, v in boot_pooled.items()}

    peak_slice = int(np.argmax(pooled["delta"]))
    peak = {"slice": peak_slice, "value": float(pooled["delta"][peak_slice]),
            "ci": [float(lo["delta"][peak_slice]), float(hi["delta"][peak_slice])]}
    jumps = np.diff(pooled["delta"])  # J(L) for blocks L = 0..nL-1
    band = shortest_band(jumps, CFG.band_frac)
    L_star = int(np.argmax(jumps))

    # ---- attention and head contributions ----------------------------------------------------
    split = np.array([p.split for p in prompts])
    tr, te = split == "train", split == "test"
    att = {k: {"train": T[k][tr].mean(0), "test": T[k][te].mean(0)} for k in ("H", "H_nobos", "M")}
    n_keys = np.array([p.target_pos + 1 for p in prompts], dtype=float)
    n_cue = np.array([len(p.cue_pos) for p in prompts], dtype=float)
    M_rel_test = (T["M"][te] / (n_cue[te] / n_keys[te])[:, None, None]).mean(0)
    W_O = model.W_O.detach().double().cpu().numpy()  # [L, H, dh, d]
    contrib = head_contributions(T["Z"], T["scale"], W_O, Xt, widx)  # [L, H]

    selected = select_heads(att["M"]["train"], CFG.k_heads)
    perm_rng = np.random.default_rng(CFG.boot_seed + 1)
    rho_M, p_M = spearman_perm(contrib.ravel(), att["M"]["test"].ravel(), CFG.n_perm, perm_rng, "greater")
    rho_H, p_H = spearman_perm(contrib.ravel(), att["H"]["test"].ravel(), CFG.n_perm, perm_rng, "less")
    rho_Hn, p_Hn = spearman_perm(contrib.ravel(), att["H_nobos"]["test"].ravel(), CFG.n_perm, perm_rng, "less")
    if band is not None:
        in_band = sum(band[0] <= L <= band[1] for L, _ in selected)
        width = band[1] - band[0] + 1
        p_band = hypergeom_tail(in_band, nL * nH, nH * width, CFG.k_heads)
    else:
        in_band, width, p_band = None, None, None

    # ---- causal check: mean ablation ---------------------------------------------------------
    sense_train = [p for p in prompts if p.split == "train"]
    means = head_means(model, sense_train, batch_size=batch_size)
    d_clean = final_delta(model, prompts, widx, [], means, batch_size)
    assert abs(d_clean - pooled["delta"][nL]) < 1e-9, (d_clean, pooled["delta"][nL])
    d_abl = final_delta(model, prompts, widx, selected, means, batch_size)
    E = 1 - d_abl / d_clean
    null_rng = np.random.default_rng(CFG.null_seed)
    E_rand = [1 - final_delta(model, prompts, widx, s, means, batch_size) / d_clean
              for s in random_sets(null_rng, n_null, CFG.k_heads, nL, nH)]
    E_layer = [1 - final_delta(model, prompts, widx, s, means, batch_size) / d_clean
               for s in layer_matched_sets(null_rng, n_null, selected, nH)]
    q = CFG.null_percentile

    # ---- neutral prompts ---------------------------------------------------------------------
    neu_t = []
    for w in widx:
        v = directions(Xt, w)[nL]
        mA, mB = (Xt[w.teA, nL] @ v).mean(), (Xt[w.teB, nL] @ v).mean()
        neu_t.append((Xt[w.neu, nL] @ v - mB) / (mA - mB))
    neu_t = np.array(neu_t)  # [W, n_neutral]
    neu_lo, neu_hi = ci(neu_t_boot.mean(1))

    summary = {
        "variant": variant_label,
        "env": env_info(),
        "n_boot": n_boot,
        "n_null": n_null,
        "words": [w.word for w in widx],
        "senses": {w.word: list(w.senses) for w in widx},
        "dprime": {
            k: {"pooled": pooled[k].tolist(), "ci_lo": lo[k].tolist(), "ci_hi": hi[k].tolist()}
            for k in pooled
        },
        "dprime_per_word": {w.word: {"target": dt[i].tolist(), "control": dc[i].tolist(),
                                     "delta": dd[i].tolist()} for i, w in enumerate(widx)},
        # Slice 0 is the target token's embedding: identical for every prompt of a word.
        "slice0_identical_within_word": all(
            bool(np.all(Xt[np.r_[w.trA, w.trB, w.teA, w.teB, w.neu], 0] == Xt[w.trA[0], 0])) for w in widx
        ),
        "peak": peak,
        "jumps": jumps.tolist(),
        "L_star": L_star,
        "band": None if band is None else {"start": band[0], "end": band[1], "width": width},
        "attention": {k: {s: v.tolist() for s, v in d.items()} for k, d in att.items()},
        "cue_mass_rel_uniform_test": M_rel_test.tolist(),
        "head_contrib": contrib.tolist(),
        "selected_heads": selected,
        "alignment": {
            "selected_in_band": in_band, "p_hypergeom": p_band,
            "spearman_contrib_cuemass": {"rho": rho_M, "p_greater": p_M},
            "spearman_contrib_entropy": {"rho": rho_H, "p_less": p_H},
            "spearman_contrib_entropy_nobos": {"rho": rho_Hn, "p_less": p_Hn},
        },
        "ablation": {
            "delta_final_clean": d_clean, "delta_final_ablated": d_abl, "effect": E,
            "null_random": E_rand, "null_layer_matched": E_layer,
            "null_random_pct": float(np.percentile(E_rand, q)),
            "null_layer_matched_pct": float(np.percentile(E_layer, q)),
            "p_random": float((1 + np.sum(np.array(E_rand) >= E)) / (1 + len(E_rand))),
            "p_layer_matched": float((1 + np.sum(np.array(E_layer) >= E)) / (1 + len(E_layer))),
        },
        "neutral": {
            "t_per_word_mean": neu_t.mean(1).tolist(),
            "t_mean": float(neu_t.mean()), "t_ci": [float(neu_lo), float(neu_hi)],
            "frac_between": float(((neu_t > 0) & (neu_t < 1)).mean()),
        },
        "seconds": time.time() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir / f"m1_{out_dir.name}.npz", Xt=Xt.astype(np.float32), Xc=Xc.astype(np.float32),
                        H=T["H"], H_nobos=T["H_nobos"], M=T["M"], scale=T["scale"],
                        Z=T["Z"].astype(np.float32), boot_delta=bdelta)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=CFG.final_step)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--n-boot", type=int, default=CFG.n_boot)
    ap.add_argument("--n-null", type=int, default=CFG.n_null_sets)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    set_determinism(CFG.master_seed)
    v = Variant(args.step, args.shuffle_seed)
    model = load(v, device=args.device)
    name = v.label.replace("@", "_")
    s = run(model, v.label, args.n_boot, args.n_null, args.batch_size,
            REPO / "results" / "m1" / name, REPO / "results" / "raw")
    print(f"done {v.label} in {s['seconds']:.0f}s -> results/m1/{name}/summary.json")


if __name__ == "__main__":
    main()
