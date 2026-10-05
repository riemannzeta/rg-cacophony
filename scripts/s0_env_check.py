"""Stage 0: environment and sanity checks for every model variant.

Writes results/stage0/sanity_<device>.json. Contains no measurement results — only checks that
the instruments work (hooks, TL/HF agreement, determinism, eps=0, causality, shuffle integrity).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

from rgc.config import CFG
from rgc.determinism import REPO, default_device, env_info, set_determinism
from rgc.m2_perturb import clean_stack, perturb_response, unit_directions
from rgc.models import Variant, load, load_hf, registry, tokenize

TEXT = (
    "The Normans were the people who in the 10th and 11th centuries gave their name to "
    "Normandy, a region in France. They were descended from Norse raiders and pirates."
)


def check_variant(v: Variant, device: str, original_hf=None) -> dict:
    out: dict = {"variant": v.label}
    hf = load_hf(v).to(device)
    model = load(v, device=device)
    tokens = tokenize(model, TEXT)
    x = tokens[None].to(device)
    T = tokens.shape[0]
    with torch.no_grad():
        lt, cache = model.run_with_cache(x)
        lh = hf(x).logits
        lt2 = model(x)
    out["tl_vs_hf_rel_maxdiff"] = ((lt - lh).abs().max() / lh.abs().max()).item()
    # Fingerprint of the weights TL actually runs, to prove variants differ from each other.
    out["W_E_fingerprint"] = float(model.W_E.detach()[:64].double().sum())
    out["logit_fingerprint"] = float(lt.detach()[0, -1, :8].double().sum())
    out["bitwise_reproducible"] = bool(torch.equal(lt, lt2))
    out["hook_shapes_ok"] = all(
        cache[f"blocks.{L}.hook_resid_pre"].shape == (1, T, model.cfg.d_model)
        and cache[f"blocks.{L}.attn.hook_pattern"].shape == (1, model.cfg.n_heads, T, T)
        and cache[f"blocks.{L}.ln1.hook_normalized"].shape == (1, T, model.cfg.d_model)
        for L in range(model.cfg.n_layers)
    )
    out["clean_stack_reproducible"] = bool(
        torch.equal(clean_stack(model, tokens, 0, 3), clean_stack(model, tokens, 0, 3))
    )

    p, eps = 12, torch.tensor([0.0, 1e-4, 1e-2, 0.0])
    u = unit_directions(model.cfg.d_model, 4, seed=3)
    r = perturb_response(model, tokens, 0, torch.full((4,), p), u, eps).cpu()
    out["eps0_exact_zero"] = bool(torch.all(r[0] == 0) and torch.all(r[3] == 0))
    out["causal_exact_zero_before_p"] = bool(torch.all(r[:, :, :p] == 0))
    out["injected_site_rel_err"] = float(((r[1:3, 0, p] - eps[1:3]).abs() / eps[1:3]).max())

    if original_hf is not None:  # shuffled control: every tensor a permutation of the original
        orig = dict(original_hf.named_parameters())
        perm_ok, changed = True, 0
        for name, prm in load_hf(v).named_parameters():
            a, b = prm.detach().reshape(-1), orig[name].detach().reshape(-1)
            perm_ok &= bool(torch.equal(a.sort().values, b.sort().values))
            changed += int(not torch.equal(a, b))
        out["shuffle_is_permutation"] = perm_ok
        out["shuffle_tensors_changed"] = f"{changed}/{len(orig)}"
    out["passed"] = (
        out["tl_vs_hf_rel_maxdiff"] < 1e-5
        and out["bitwise_reproducible"]
        and out["hook_shapes_ok"]
        and out["clean_stack_reproducible"]
        and out["eps0_exact_zero"]
        and out["causal_exact_zero_before_p"]
        and out["injected_site_rel_err"] < 1e-3
        and out.get("shuffle_is_permutation", True)
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--variants", default="all", help="'all' or comma-separated labels")
    args = ap.parse_args()
    set_determinism(CFG.master_seed)

    refs = {b.name for b in HfApi().list_repo_refs(CFG.hf_repo).branches}
    needed = {f"step{s}" for s in CFG.checkpoints}
    report: dict = {
        "env": env_info(),
        "checkpoints_available": {s: s in refs for s in sorted(needed)},
    }

    pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(REPO / "tests")], capture_output=True, text=True
    )
    report["pytest"] = pytest.stdout.strip().splitlines()[-1]
    report["pytest_ok"] = pytest.returncode == 0

    variants = registry()
    if args.variants != "all":
        keep = set(args.variants.split(","))
        variants = [v for v in variants if v.label in keep]
    original = load_hf(Variant(CFG.final_step))
    report["variants"] = []
    for v in variants:
        res = check_variant(v, args.device, original if v.shuffle_seed is not None else None)
        print(json.dumps(res), flush=True)
        report["variants"].append(res)
    report["all_passed"] = report["pytest_ok"] and all(r["passed"] for r in report["variants"])

    dev = "cuda" if args.device.startswith("cuda") else "cpu"
    path = REPO / "results" / "stage0" / f"sanity_{dev}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path}  all_passed={report['all_passed']}")


if __name__ == "__main__":
    main()
