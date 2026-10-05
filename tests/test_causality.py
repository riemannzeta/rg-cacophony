import torch

from rgc.m2_perturb import perturb_response, unit_directions


def _run(model, tokens, k, p, eps):
    R = len(eps)
    u = unit_directions(model.cfg.d_model, R, seed=3)
    return perturb_response(
        model, tokens, k, torch.full((R,), p, dtype=torch.long), u, torch.tensor(eps)
    )


def test_eps_zero_gives_exact_zero(model, tokens):
    r = _run(model, tokens, k=2, p=10, eps=[0.0, 0.0, 0.0])
    assert torch.all(r == 0)


def test_mixed_batch_eps_zero_rows_exact_zero(model, tokens):
    r = _run(model, tokens, k=1, p=10, eps=[0.0, 1e-3, 0.0, 1e-1])
    assert torch.all(r[0] == 0) and torch.all(r[2] == 0)


def test_no_response_before_injection_position(model, tokens):
    p = 12
    r = _run(model, tokens, k=0, p=p, eps=[1e-3, 1e-2, 1e-1])
    assert torch.all(r[:, :, :p] == 0)


def test_injected_site_has_relative_size_eps(model, tokens):
    eps = [1e-4, 1e-2, 1e-1]
    r = _run(model, tokens, k=4, p=7, eps=eps)
    torch.testing.assert_close(r[:, 0, 7].cpu(), torch.tensor(eps), rtol=1e-3, atol=0)
    assert torch.all(r[:, 0, 8:] == 0)  # injection slice: only the injected site moves


def test_perturbation_spreads_forward(model, tokens):
    r = _run(model, tokens, k=0, p=5, eps=[1e-2])
    assert (r[0, -1, 6:] > 0).all()
