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


def test_scan_patch_is_bitwise_noop_for_float32():
    from transformers.models.mamba import modeling_mamba as mm

    import rgc.mamba  # noqa: F401  (applies the patch)

    g = torch.Generator().manual_seed(0)
    b, d, n, T = 2, 8, 4, 30
    args = (torch.randn(b, d, T, generator=g), torch.randn(b, d, T, generator=g), -torch.rand(d, n, generator=g),
            torch.randn(b, n, T, generator=g), torch.randn(b, n, T, generator=g))
    kw = dict(D=torch.randn(d, generator=g), z=torch.randn(b, d, T, generator=g), delta_bias=torch.randn(d, generator=g),
              delta_softplus=True)
    assert torch.equal(mm.mamba_selective_scan(*args, **kw), mm._rgc_original_selective_scan(*args, **kw))
    out64 = mm.mamba_selective_scan(*[a.double() for a in args], **{k: (v.double() if torch.is_tensor(v) else v) for k, v in kw.items()})
    assert out64.dtype == torch.float64


def test_float64_reference_is_truly_float64():
    """The f64 copy must be linear down to eps = 1e-8 (no hidden float32 casts; D18)."""
    from rgc.mamba import MambaRunner

    m64 = MambaRunner("state-spaces/mamba-130m-hf").to_float64()
    x = torch.tensor([m64.tokenizer.bos_token_id] + load_passages()[0][:48])
    u = unit_directions(768, 1, 11).double()
    g6 = m64.perturb_response(x, 0, torch.tensor([20]), u, torch.tensor([1e-6]))[0] / 1e-6
    g8 = m64.perturb_response(x, 0, torch.tensor([20]), u, torch.tensor([1e-8]))[0] / 1e-8
    sel = g6[1:, 21:] >= 0.01
    assert ((g8 - g6).abs() / g6)[1:, 21:][sel].median() < 1e-3
