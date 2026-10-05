import torch

from rgc.m2_perturb import clean_stack


def test_forward_bitwise_reproducible(model, tokens):
    x = tokens[None].to(model.cfg.device)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert torch.equal(a, b)


def test_clean_stack_bitwise_reproducible(model, tokens):
    a = clean_stack(model, tokens, 0, 4)
    b = clean_stack(model, tokens, 0, 4)
    assert torch.equal(a, b)
