"""PREREG-2 part B: symmetry breaking under graded and conflicting cues (new prompts).

  uv run python scripts/s2c_conflict.py --step 143000 [--shuffle-seed S] --device cuda
Writes results/conflict/<variant>/summary.json

Per word and slice L >= 1: direction v = unit(mean train-AA - mean train-BB); test projections
s = X v; t = (s - m_BB) / (m_AA - m_BB) with test-AA/BB means (so mean t(AA) = 1, t(BB) = 0).
  R(L) = t(+1) - t(-1)                       [the ratio's denominator t(+2) - t(-2) = 1 by construction;
                                               "denominator CI excludes 0" is checked on raw m_AA - m_BB]
  D(L) = mean|t - 1/2| over conflict (AB, BA) - mean|t - 1/2| over none (NN)
  BC(L) = bimodality coefficient of per-word-standardised t over conflict prompts (pooled)
  rho(L) = t(AB) - t(BA)   (negative: the later cue wins)
Bootstrap: train and test FRAMES resampled within word (a frame carries all its conditions),
same resamples for target and control prompts. Pooled = mean over words.
"""

import argparse
import json
from types import SimpleNamespace

import numpy as np
from scipy import stats

from rgc import stimuli_v2
from rgc.config import CFG
from rgc.config_prereg2 import CFG2, config2_hash
from rgc.determinism import REPO, default_device, env_info, set_determinism
from rgc.m1_order import collect, resample_counts
from rgc.models import Variant, load
from rgc.stats import ci

GROUPS = {"p2": ["AA"], "m2": ["BB"], "p1": ["AN", "NA"], "m1": ["BN", "NB"], "conf": ["AB", "BA"],
          "none": ["NN"], "AB": ["AB"], "BA": ["BA"]}


def bimodality(x: np.ndarray) -> float:
    n = len(x)
    if n < 4 or np.std(x) == 0:
        return np.nan
    g, k = stats.skew(x, bias=False), stats.kurtosis(x, bias=False)
    return float((g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def analyse(X: np.ndarray, prompts, n_boot: int, rng) -> dict:
    """X: [N, S, d] for this prompt set (target or control). Returns per-word arrays + bootstraps."""
    words = list(dict.fromkeys(p.word for p in prompts))
    S = X.shape[1]
    idx = {(p.word, p.split, p.cond, p.frame): i for i, p in enumerate(prompts)}
    tr_frames, te_frames = list(range(8)), list(range(8, 16))
    out = {k: np.full((len(words), S), np.nan) for k in ("R", "D", "rho", "sep")}
    boot = {k: np.full((n_boot, len(words), S), np.nan) for k in ("R", "D", "rho", "sep")}
    conf_t = {L: [] for L in range(S)}
    counts = {}
    for wi, w in enumerate(words):
        c_tr = resample_counts(8, n_boot, rng)
        c_te = resample_counts(8, n_boot, rng)
        counts[w] = (c_tr, c_te)
        for L in range(1, S):
            A = X[[idx[(w, "train", "AA", f)] for f in tr_frames], L]
            B = X[[idx[(w, "train", "BB", f)] for f in tr_frames], L]
            v = A.mean(0) - B.mean(0)
            if np.linalg.norm(v) == 0:
                continue
            v /= np.linalg.norm(v)
            Vb = c_tr @ A / 8 - c_tr @ B / 8
            Vb /= np.linalg.norm(Vb, axis=1, keepdims=True)
            # projections per condition: [frames=8] point, [8, B] bootstrap
            proj = {c: X[[idx[(w, "test", c, f)] for f in te_frames], L] for c in stimuli_v2.CONDITIONS}
            sp = {c: x @ v for c, x in proj.items()}
            sb = {c: x @ Vb.T for c, x in proj.items()}

            def stat(s, wts):  # s: [8] or [8, B]; wts: None or [B, 8]
                if wts is None:
                    mean = lambda cs: np.mean([s[c].mean() for c in cs])
                    mAA, mBB = mean(["AA"]), mean(["BB"])
                    t = {c: (s[c] - mBB) / (mAA - mBB) for c in s}
                    absd = lambda cs: np.mean([np.abs(t[c] - 0.5).mean() for c in cs])
                    tm = lambda cs: np.mean([t[c].mean() for c in cs])
                    return {"sep": mAA - mBB, "R": tm(GROUPS["p1"]) - tm(GROUPS["m1"]),
                            "D": absd(GROUPS["conf"]) - absd(GROUPS["none"]), "rho": tm(["AB"]) - tm(["BA"]), "t": t}
                wm = lambda c: (wts * s[c].T).sum(1) / 8  # [B]
                mAA, mBB = wm("AA"), wm("BB")
                den = np.where(np.abs(mAA - mBB) > 1e-12, mAA - mBB, np.nan)
                tm = lambda cs: np.mean([(wm(c) - mBB) / den for c in cs], axis=0)
                absd = lambda cs: np.mean([(wts * np.abs((s[c] - mBB) / den - 0.5).T).sum(1) / 8 for c in cs], axis=0)
                return {"sep": mAA - mBB, "R": tm(GROUPS["p1"]) - tm(GROUPS["m1"]),
                        "D": absd(GROUPS["conf"]) - absd(GROUPS["none"]), "rho": tm(["AB"]) - tm(["BA"])}

            pt = stat(sp, None)
            for k in ("R", "D", "rho", "sep"):
                out[k][wi, L] = pt[k]
            bt = stat(sb, c_te)
            for k in ("R", "D", "rho", "sep"):
                boot[k][:, wi, L] = bt[k]
            tc = np.concatenate([pt["t"]["AB"], pt["t"]["BA"]])
            conf_t[L].append((tc - tc.mean()) / tc.std() if tc.std() > 0 else tc * 0)
    BC = [bimodality(np.concatenate(conf_t[L])) if conf_t[L] else np.nan for L in range(S)]
    return {"words": words, "point": out, "boot": boot, "BC": BC}


def run(model, out_dir, n_boot: int = CFG.n_boot) -> dict:
    prompts = stimuli_v2.load()
    adapt = [SimpleNamespace(token_ids=p.token_ids, control_token_ids=p.control_token_ids, target_pos=p.target_pos,
                             cue_pos=p.cueA_pos + p.cueB_pos) for p in prompts]
    Xt = collect(model, adapt, control=False, attention=False)["X"]
    Xc = collect(model, adapt, control=True, attention=False)["X"]
    res = {}
    for name, X in (("target", Xt), ("control", Xc)):
        res[name] = analyse(X, prompts, n_boot, np.random.default_rng(CFG.boot_seed + 11))  # same resamples
    S = Xt.shape[1]

    def pooled(r, k):
        return np.nanmean(r["point"][k], 0), np.nanmean(r["boot"][k], 1)  # [S], [B, S]

    summary = {"variant": None, "env": env_info(), "config2_hash": config2_hash(), "n_boot": n_boot,
               "words": res["target"]["words"]}
    for name in ("target", "control"):
        summary[name] = {}
        for k in ("R", "D", "rho", "sep"):
            p, b = pooled(res[name], k)
            lo, hi = ci(b)
            summary[name][k] = {"pooled": p.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
                                "per_word": res[name]["point"][k].tolist()}
        summary[name]["BC"] = res[name]["BC"]
    # B1 / B2: target - control at the argmax slice of the pooled point estimate; CI at that slice
    sep_ok = [L for L in range(1, S)
              if summary["target"]["sep"]["ci_lo"][L] > 0 and summary["control"]["sep"]["ci_lo"][L] > 0]
    for crit, k in (("B1", "R"), ("B2", "D")):
        pt_diff = np.array(summary["target"][k]["pooled"]) - np.array(summary["control"][k]["pooled"])
        b_diff = np.nanmean(res["target"]["boot"][k], 1) - np.nanmean(res["control"]["boot"][k], 1)
        cand = sep_ok if k == "R" else list(range(1, S))
        if not cand:
            summary[crit] = {"testable": False, "reason": "no slice where both raw separations' CIs exclude 0"}
            continue
        L = max(cand, key=lambda x: pt_diff[x])
        lo, hi = ci(b_diff[:, L][np.isfinite(b_diff[:, L])])
        summary[crit] = {"testable": True, "slice": int(L), "diff": float(pt_diff[L]), "ci": [float(lo), float(hi)],
                         "diff_by_slice": pt_diff.tolist()}
    bc = np.array(summary["target"]["BC"], dtype=float)
    summary["B3_exploratory"] = {"max_BC_target": float(np.nanmax(bc)) if np.isfinite(bc).any() else None,
                                 "slice": int(np.nanargmax(bc)) if np.isfinite(bc).any() else None,
                                 "threshold": CFG2.bc_threshold}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=CFG.final_step)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()
    set_determinism(CFG.master_seed)
    v = Variant(args.step, args.shuffle_seed)
    d = REPO / "results" / "conflict" / v.label.replace("@", "_")
    s = run(load(v, device=args.device), d)
    s["variant"] = v.label
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    print(f"done {v.label} -> {d / 'summary.json'}")


if __name__ == "__main__":
    main()
