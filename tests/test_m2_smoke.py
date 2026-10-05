"""Smoke test of the Stage 3 pipeline on a tiny random transformer (no real-model results)."""

import importlib.util

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rgc.attention_manip import manipulation_hooks
from rgc.m2_perturb import perturb_response, unit_directions
from rgc.wikitext import load_passages


@pytest.fixture(scope="module")
def tiny():
    cfg = HookedTransformerConfig(
        n_layers=3, d_model=32, n_heads=4, d_head=8, d_mlp=64, d_vocab=50304, n_ctx=300,
        act_fn="gelu", normalization_type="LN", positional_embedding_type="rotary", rotary_dim=4,
        parallel_attn_mlp=True, device="cpu", seed=0,
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision="step143000")
    return HookedTransformer(cfg, tokenizer=tok, default_padding_side="right")


@pytest.fixture(scope="module")
def s3():
    spec = importlib.util.spec_from_file_location("s3", "scripts/s3_m2.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_calibration_main_and_analysis(tiny, s3):
    import copy

    tiny64 = HookedTransformer(copy.deepcopy(tiny.cfg), tokenizer=tiny.tokenizer, default_padding_side="right")
    tiny64.load_state_dict(tiny.state_dict())
    tiny64 = tiny64.to(torch.float64)
    tiny64.cfg.dtype = torch.float64
    passages = load_passages()[:2]
    dirs = list(unit_directions(32, 3, 11))
    cal = s3.calibrate(tiny, tiny64, [], passages[:1], dirs[:2])
    assert len(cal["rows"]) == 13 and cal["eps_star"] is not None
    lat = s3.main_grid(tiny, [], passages, dirs, cal["eps_star"])
    assert lat[0].shape == (2, 3, 3, 4, 257)
    res = s3.analyse_main(lat, 3, workers=1)
    assert 0 <= res["pci_primary"]["pci"] and len(res["pci_by_theta"]) == 6
    assert "k0_p64" in res["spread"]


def test_flatten_beta1_is_identity_and_window_caps_reach(tiny):
    toks = torch.tensor([0] + load_passages()[0][:63])
    u = unit_directions(32, 1, 3)
    P, E = torch.tensor([10]), torch.tensor([1e-3])
    r0 = perturb_response(tiny, toks, 0, P, u, E)
    r1 = perturb_response(tiny, toks, 0, P, u, E, extra_hooks=manipulation_hooks("beta=1.0", 3))
    assert torch.equal(r0, r1)
    rw = perturb_response(tiny, toks, 0, P, u, E, extra_hooks=manipulation_hooks("window=2", 3))
    # with window 2 and 3 layers, a perturbation at position 10 can reach at most positions 10..13
    assert torch.all(rw[0, :, 14:] == 0) and torch.all(rw[0, -1, 11:14] > 0)
    rb0 = perturb_response(tiny, toks, 0, P, u, E, extra_hooks=manipulation_hooks("beta=0", 3))
    assert torch.isfinite(rb0).all() and (rb0[0, -1, 11:] > 0).all()


def test_gating_runs(tiny, s3, tmp_path):
    dirs = list(unit_directions(32, 2, 11))
    g = s3.gating(tiny, [], dirs, 1e-3, tmp_path / "missing.npz")
    assert g["k_before"] == max(0, g["L_star_trained"] - 2)
    assert set(g["conditions"]) == {"before", "after"}
    assert np.isfinite(g["contrast_after_minus_before_pci_random"]["mean"])


def test_main_outputs_reach_local_and_ce(tiny, s3):
    passages = load_passages()[:2]
    dirs = list(unit_directions(32, 2, 11))
    lat = s3.main_grid(tiny, [], passages, dirs, 1e-3)
    res = s3.analyse_main(lat, 3, workers=1)
    assert np.array(res["per_passage_dir_reach_primary"]).shape == (2, 2)
    assert np.array(res["per_passage_dir_local_response"]).shape == (2, 2)
    ce = s3.next_token_ce(tiny, manipulation_hooks("beta=0", 3), passages)
    assert len(ce) == 2 and all(np.isfinite(ce))


def test_mamba_backend_through_s3(s3):
    import copy

    from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM

    from rgc.mamba import MambaRunner

    torch.manual_seed(0)
    cfg = MambaConfig(vocab_size=50280, hidden_size=16, state_size=4, num_hidden_layers=8, expand=2, conv_kernel=4)
    hf = MambaForCausalLM(cfg)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision="step143000")
    m = MambaRunner(model=hf, tokenizer=tok)
    m64 = MambaRunner(model=copy.deepcopy(hf), tokenizer=tok).to_float64()
    passages = load_passages()[:1]
    dirs = list(unit_directions(16, 2, 11))
    cal = s3.calibrate(m, m64, [], passages, dirs[:1], chunk=20)
    assert cal["eps_star"] is not None
    lat = s3.main_grid(m, [], passages, dirs, cal["eps_star"])
    assert lat[0].shape[3] == 9  # slices 0..8
    sub = {k // 2: lat[k][..., ::2, :] for k in lat if k % 2 == 0}
    assert sub[0].shape[3] == 5 and sub[1].shape[3] == 4  # slices {0,2,4,6,8} and {2,4,6,8}
    res = s3.analyse_main(sub, 4, workers=1)
    assert len(res["z"]["boot_z"]) == 10_000
    assert "pci_primary" in res
    ce = s3.next_token_ce(m, [], passages)
    assert np.isfinite(ce).all()
