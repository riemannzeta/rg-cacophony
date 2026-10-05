"""End-to-end smoke test of Measurement 1 on a tiny random transformer (no real-model results)."""

import json

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from rgc.m1_order import WordIdx, dprime, dprime_boot, dprime_curve, resample_counts
from rgc.stats import shortest_band


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=32, n_heads=4, d_head=8, d_mlp=64, d_vocab=50304, n_ctx=64,
        act_fn="gelu", normalization_type="LN", positional_embedding_type="rotary", rotary_dim=4,
        parallel_attn_mlp=True, device="cpu", seed=0,
    )
    return HookedTransformer(cfg)


def test_s2_pipeline_runs(tiny, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("s2", "scripts/s2_m1.py")
    s2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s2)
    s = s2.run(tiny, "tiny", n_boot=50, n_null=3, batch_size=256, out_dir=tmp_path / "m1", raw_dir=tmp_path / "raw")
    assert s["slice0_identical_within_word"]
    assert len(s["dprime"]["delta"]["pooled"]) == 3
    assert s["dprime"]["delta"]["pooled"][0] == 0.0  # slice 0: zero variance -> d' := 0
    H = np.array(s["attention"]["H"]["test"])
    assert H.shape == (2, 4) and np.all((H > 0) & (H <= 1 + 1e-9))
    M = np.array(s["attention"]["M"]["test"])
    assert np.all((M >= 0) & (M <= 1))
    assert len(s["selected_heads"]) == 6 and len(s["ablation"]["null_random"]) == 3
    json.dumps(s)


def test_dprime_boot_matches_point_with_identity_counts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3, 16))
    X[:20, :, 0] += 1.0  # sense A shifted
    X[40:50, :, 0] += 1.0
    w = WordIdx("x", ("a", "b"), np.arange(20), np.arange(20, 40), np.arange(40, 50), np.arange(50, 60), np.array([]))
    point = dprime_curve(X, w)
    boot = dprime_boot(X, w, np.ones((1, 20)), np.ones((1, 10)))
    np.testing.assert_allclose(boot[0], point, rtol=1e-10)
    assert point[0] > 0


def test_dprime_zero_variance():
    assert dprime(np.ones(5), np.ones(5)) == 0.0


def test_shortest_band():
    assert shortest_band(np.array([0.1, 0.1, 2.0, 0.1, 0.0]), 0.5) == (2, 2)
    assert shortest_band(np.array([0.5, 0.5, 0.5, 0.5]), 0.5) == (0, 1)
    assert shortest_band(np.array([-1.0, -0.5]), 0.5) is None


def test_attention_stats_and_head_outputs_match_cache(tiny):
    from rgc import stimuli
    from rgc.m1_order import collect

    p = [q for q in stimuli.load() if q.split == "train"][:3]
    out = collect(tiny, p, attention=True, batch_size=8)
    for i, q in enumerate(p):
        toks = torch.tensor(q.token_ids[: q.target_pos + 1])[None]
        _, cache = tiny.run_with_cache(toks)
        t = q.target_pos
        for L in range(tiny.cfg.n_layers):
            a = cache[f"blocks.{L}.attn.hook_pattern"][0, :, t, : t + 1].double()
            H = -(a * a.log()).sum(-1) / np.log(t + 1)
            np.testing.assert_allclose(out["H"][i, L], H.numpy(), rtol=1e-5)
            np.testing.assert_allclose(out["M"][i, L], a[:, q.cue_pos].sum(-1).numpy(), rtol=1e-5)
            # head decomposition: sum_h z_h W_O[h] + b_O == attn_out at the target
            z = torch.tensor(out["Z"][i, L])
            o = (torch.einsum("hk,hkd->d", z, tiny.W_O[L].double()) + tiny.b_O[L].double()).detach()
            np.testing.assert_allclose(o.numpy(), cache[f"blocks.{L}.hook_attn_out"][0, t].double().numpy(), atol=1e-4)
        np.testing.assert_allclose(out["X"][i, -1], cache["ln_final.hook_normalized"][0, t].double().numpy(), atol=1e-5)


def test_mean_ablation_hook_leaves_bos(tiny):
    from rgc.m1_order import ablation_hooks

    means = torch.full((2, tiny.cfg.d_head), 7.0)
    seen = {}

    def grab(act, hook):
        seen["z"] = act.clone()

    toks = torch.randint(1, 1000, (2, 6))
    tiny.run_with_hooks(toks, return_type=None,
                        fwd_hooks=ablation_hooks({1: ([0, 2], means)}) + [("blocks.1.attn.hook_z", grab)])
    z = seen["z"]
    assert torch.all(z[:, 1:, [0, 2]] == 7.0)
    assert not torch.all(z[:, 0, [0, 2]] == 7.0)  # BOS untouched
    assert not torch.all(z[:, 1:, 1] == 7.0)  # other heads untouched


def test_s2b_posthoc_runs_and_reproduces_preregistered_effect(tiny, tmp_path):
    import importlib.util

    def mod(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    s2, s2b = mod("scripts/s2_m1.py", "s2"), mod("scripts/s2b_m1_posthoc.py", "s2b")
    d = tmp_path / "m1"
    s2.run(tiny, "tiny", n_boot=20, n_null=3, batch_size=256, out_dir=d, raw_dir=tmp_path / "raw")
    out = s2b.run(tiny, d, n_null=3)  # asserts the final-slice effect reproduces exactly
    assert "POST HOC" in out["label"]
    assert set(out["entropy_detrended"]) >= {"H", "H_nobos"}
