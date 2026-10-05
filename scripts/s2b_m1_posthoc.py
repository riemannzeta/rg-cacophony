"""POST HOC / EXPLORATORY analyses of Measurement 1 (D14). Not part of the PREDICTIONS.md scoring.

Motivated by the Stage 2 results:
  1. Ablation effect measured at the model's own peak slice (and on d′_target alone), instead of
     the final slice where trained Δd′ turned out smallest. Same selected heads, same head means,
     and the SAME null head sets as the pre-registered run (identical RNG stream).
  2. Entropy-contribution correlation with the depth trend removed (entropy ranked within layer),
     plus a partial Spearman controlling for layer.
  3. Neutral-prompt lean per word: |t - 0.5| (pooled t hides per-word leans of opposite sign).

  uv run python scripts/s2b_m1_posthoc.py --step 143000 [--shuffle-seed S] --device cuda
Writes results/m1/<variant>/posthoc.json
"""

import argparse
import json

import numpy as np
from scipy import stats

from rgc import stimuli
from rgc.config import CFG
from rgc.determinism import REPO, default_device, env_info, set_determinism
from rgc.m1_ablation import layer_matched_sets, random_sets, spec
from rgc.m1_order import collect, dprime_curve, head_means, word_index
from rgc.models import Variant, load


def ablated_scores(model, prompts, widx, heads, means, slices):
    """Pooled Δd′ and d′_target at each requested slice, heads mean-ablated, directions refit."""
    ab = spec(heads, means) if heads else None
    Xt = collect(model, prompts, control=False, attention=False, slices=slices, ablate=ab)["X"]
    Xc = collect(model, prompts, control=True, attention=False, slices=slices, ablate=ab)["X"]
    dt = np.mean([dprime_curve(Xt, w) for w in widx], axis=0)
    dc = np.mean([dprime_curve(Xc, w) for w in widx], axis=0)
    return {"delta": (dt - dc).tolist(), "target": dt.tolist()}


def run(model, d, n_null: int = CFG.n_null_sets) -> dict:
    """d: the variant's results/m1/<variant> directory (must contain summary.json)."""
    s = json.loads((d / "summary.json").read_text())
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    prompts = stimuli.load()
    widx = word_index(prompts)
    selected = [tuple(h) for h in s["selected_heads"]]
    L_peak = s["peak"]["slice"]
    slices = sorted({L_peak, nL})
    means = head_means(model, [p for p in prompts if p.split == "train"])

    clean = ablated_scores(model, prompts, widx, [], means, slices)
    abl = ablated_scores(model, prompts, widx, selected, means, slices)
    null_rng = np.random.default_rng(CFG.null_seed)  # same stream as the pre-registered run
    sets_r = random_sets(null_rng, n_null, CFG.k_heads, nL, nH)
    sets_l = layer_matched_sets(null_rng, n_null, selected, nH)
    assert len(sets_r) == len(s["ablation"]["null_random"])
    nulls = {"random": [ablated_scores(model, prompts, widx, h, means, slices) for h in sets_r],
             "layer_matched": [ablated_scores(model, prompts, widx, h, means, slices) for h in sets_l]}

    def effect(score, key, i):
        return 1 - score[key][i] / clean[key][i]

    abl_out = {}
    for key in ("delta", "target"):
        for i, sl in enumerate(slices):
            name = f"{key}@slice{sl}"
            E = effect(abl, key, i)
            row = {"clean": clean[key][i], "ablated": abl[key][i], "effect": E}
            for nk, sets in nulls.items():
                En = np.array([effect(x, key, i) for x in sets])
                row[f"null_{nk}_p95"] = float(np.percentile(En, 95))
                row[f"p_{nk}"] = float((1 + np.sum(En >= E)) / (1 + len(En)))
            abl_out[name] = row
    # consistency with the pre-registered run (final-slice Δ effect must reproduce exactly)
    assert abs(abl_out[f"delta@slice{nL}"]["effect"] - s["ablation"]["effect"]) < 1e-9

    # ---- entropy with depth trend removed --------------------------------------------------
    c = np.array(s["head_contrib"])
    H = np.array(s["attention"]["H"]["test"])
    Hn = np.array(s["attention"]["H_nobos"]["test"])
    M = np.array(s["attention"]["M"]["test"])
    layer = np.repeat(np.arange(nL), nH).astype(float)

    def within_layer_rank(A):
        return np.concatenate([stats.rankdata(row) for row in A])

    def partial_spearman(x, y, z):
        rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
        res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
        return float(stats.pearsonr(res(rx), res(ry))[0])

    rng = np.random.default_rng(CFG.boot_seed + 7)
    def perm_p(x, y, rho, n=CFG.n_perm, alt="less"):
        null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n)])
        return float((1 + np.sum(null <= rho if alt == "less" else null >= rho)) / (1 + n))

    ent = {}
    for nm, A in (("H", H), ("H_nobos", Hn)):
        r_wl = float(stats.spearmanr(c.ravel(), within_layer_rank(A))[0])
        ent[nm] = {
            "spearman_contrib_vs_within_layer_rank": r_wl,
            "p_less": perm_p(c.ravel(), within_layer_rank(A), r_wl),
            "partial_spearman_given_layer": partial_spearman(c.ravel(), A.ravel(), layer),
        }
    ent["cue_mass_partial_spearman_given_layer"] = partial_spearman(c.ravel(), M.ravel(), layer)
    ent["entropy_by_layer_mean"] = H.mean(1).tolist()

    # ---- neutral lean ------------------------------------------------------------------------
    t = np.array(s["neutral"]["t_per_word_mean"])
    neutral = {"abs_t_minus_half_per_word": dict(zip(s["words"], np.abs(t - 0.5).round(4).tolist())),
               "mean_abs_t_minus_half": float(np.abs(t - 0.5).mean()),
               "leans_toward": {w: (s["senses"][w][0] if ti > 0.5 else s["senses"][w][1])
                                for w, ti in zip(s["words"], t)}}

    out = {"label": "POST HOC / EXPLORATORY (D14) - not scored against PREDICTIONS.md",
           "variant": s["variant"], "env": env_info(), "peak_slice": L_peak, "slices": slices,
           "ablation": abl_out, "entropy_detrended": ent, "neutral_lean": neutral}
    (d / "posthoc.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=CFG.final_step)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    set_determinism(CFG.master_seed)
    v = Variant(args.step, args.shuffle_seed)
    d = REPO / "results" / "m1" / v.label.replace("@", "_")
    run(load(v, device=args.device), d)
    print(f"done {v.label} -> {d / 'posthoc.json'}")


if __name__ == "__main__":
    main()
