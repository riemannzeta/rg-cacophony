import torch

from rgc.models import shuffle_parameters_


def test_shuffle_is_per_tensor_permutation():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 4))
    with torch.no_grad():
        m[1].weight.normal_()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    shuffle_parameters_(m, seed=1001)
    for n, p in m.named_parameters():
        assert torch.equal(p.detach().reshape(-1).sort().values, before[n].reshape(-1).sort().values)
        if before[n].unique().numel() > 1:  # constant tensors (e.g. zero LN bias) cannot change
            assert not torch.equal(p.detach(), before[n])


def test_shuffle_seeded():
    torch.manual_seed(0)
    a, b = torch.nn.Linear(32, 32), torch.nn.Linear(32, 32)
    b.load_state_dict(a.state_dict())
    shuffle_parameters_(a, 5)
    shuffle_parameters_(b, 5)
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
