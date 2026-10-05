"""Smoke tests for PREREG-2 part B (tiny random transformer; no real-model results)."""

import importlib.util

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import AutoTokenizer

from rgc import stimuli_v2


@pytest.fixture(scope="module")
def tiny():
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=32, n_heads=4, d_head=8, d_mlp=64, d_vocab=50304, n_ctx=64,
        act_fn="gelu", normalization_type="LN", positional_embedding_type="rotary", rotary_dim=4,
        parallel_attn_mlp=True, device="cpu", seed=0,
    )
    return HookedTransformer(cfg)


def test_saved_v2_matches_source_and_is_valid():
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision="step143000")
    built, rep = stimuli_v2.build(tok)
    assert rep["errors"] == [] and rep["warnings"] == []
    assert [p.__dict__ for p in built] == [p.__dict__ for p in stimuli_v2.load()]
    for p in built:
        assert p.token_ids[p.target_pos] == tok(" " + p.word, add_special_tokens=False)["input_ids"][0]
        assert (len(p.cueA_pos) > 0) == ("A" in p.cond) and (len(p.cueB_pos) > 0) == ("B" in p.cond)


def test_conflict_pipeline_runs(tiny, tmp_path):
    spec = importlib.util.spec_from_file_location("s2c", "scripts/s2c_conflict.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    s = m.run(tiny, tmp_path / "c", n_boot=50)
    assert len(s["target"]["R"]["pooled"]) == 3
    assert "B1" in s and "B2" in s


def test_bimodality_coefficient():
    spec = importlib.util.spec_from_file_location("s2c", "scripts/s2c_conflict.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(-3, 0.5, 200), rng.normal(3, 0.5, 200)])
    assert m.bimodality(bimodal) > 0.555 > m.bimodality(rng.normal(0, 1, 400))


@pytest.mark.parametrize("shape,expected_R", [("linear", 0.5), ("step", 1.0)])
def test_saturation_ratio_on_synthetic_responses(shape, expected_R):
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location("s2c", "scripts/s2c_conflict.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    lin = {"AA": 2, "AN": 1, "NA": 1, "NN": 0, "AB": 0, "BA": 0, "BN": -1, "NB": -1, "BB": -2}
    step = {"AA": 2, "AN": 2, "NA": 2, "NN": 0, "AB": 0, "BA": 0, "BN": -2, "NB": -2, "BB": -2}
    val = lin if shape == "linear" else step
    rng = np.random.default_rng(0)
    prompts, rows = [], []
    for w in ("x", "y"):
        for f in range(16):
            for c in stimuli_v2.CONDITIONS:
                prompts.append(SimpleNamespace(word=w, split="train" if f < 8 else "test", cond=c, frame=f))
                x = np.zeros((2, 8))
                x[1, 0] = val[c]
                x[1, 1:] = rng.normal(0, 1e-3, 7)
                rows.append(x)
    X = np.stack(rows)
    r = m.analyse(X, prompts, n_boot=20, rng=np.random.default_rng(1))
    np.testing.assert_allclose(r["point"]["R"][:, 1], expected_R, atol=1e-2)
