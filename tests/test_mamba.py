"""Stage 0-style instrument checks for the Mamba runner (PREREG-2 part C), on mamba-130m (CPU)."""

import pytest
import torch

from rgc.determinism import set_determinism
from rgc.m2_perturb import unit_directions
from rgc.wikitext import load_passages

set_determinism(0)


@pytest.fixture(scope="module")
def mamba():
    from rgc.mamba import MambaRunner

    return MambaRunner("state-spaces/mamba-130m-hf", device="cpu")


@pytest.fixture(scope="module")
def tokens(mamba):
    return torch.tensor([mamba.tokenizer.bos_token_id] + load_passages()[0][:40])


def _run(m, tokens, k, p, eps):
    u = unit_directions(m.cfg.d_model, len(eps), 3)
    return m.perturb_response(tokens, k, torch.full((len(eps),), p), u, torch.tensor(eps))


def test_shapes_and_bitwise_reproducible(mamba, tokens):
    a = mamba.clean_stack(tokens, 0, 2)
    b = mamba.clean_stack(tokens, 0, 2)
    assert a.shape == (2, mamba.cfg.n_layers + 1, tokens.shape[0], mamba.cfg.d_model)
    assert torch.equal(a, b)


def test_eps_zero_and_causality(mamba, tokens):
    r = _run(mamba, tokens, 3, 10, [0.0, 1e-3, 0.0, 1e-1])
    assert torch.all(r[0] == 0) and torch.all(r[2] == 0)
    assert torch.all(r[:, :, :10] == 0)
    assert (r[1, -1, 11:] > 0).all()


def test_injected_site(mamba, tokens):
    eps = [1e-4, 1e-2]
    r = _run(mamba, tokens, 5, 7, eps)
    torch.testing.assert_close(r[:, 0, 7], torch.tensor(eps), rtol=1e-3, atol=0)
    assert torch.all(r[:, 0, 8:] == 0)


def test_shuffle_is_permutation():
    from rgc.mamba import MambaRunner

    a = MambaRunner("state-spaces/mamba-130m-hf", device="cpu")
    b = MambaRunner("state-spaces/mamba-130m-hf", device="cpu", shuffle_seed=1001)
    pa, pb = dict(a.model.named_parameters()), dict(b.model.named_parameters())
    for n in pa:
        assert torch.equal(pa[n].detach().reshape(-1).sort().values, pb[n].detach().reshape(-1).sort().values)
