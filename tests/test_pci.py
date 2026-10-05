import numpy as np

from rgc.pci import lattice_bits, pci


def test_lattice_bits_excludes_injection_slice_and_self_column():
    g = np.zeros((4, 10))
    g[0, :] = 1.0  # injection slice: ignored
    g[:, 3] = 1.0  # self column p=3: ignored
    g[2, 5] = 1.0
    b = lattice_bits(g, p=3, theta=0.5)
    assert b.size == 3 * 6 and b.sum() == 1
    assert b[1 * 6 + (5 - 4)] == 1  # layer-major ordering


def test_pci_degenerate_lattices():
    assert pci(np.zeros((5, 20)), 2, 0.1) == 0.0
    assert pci(np.ones((5, 20)), 2, 0.1) == 0.0


def test_density_matched_bits_fix_p1_and_order():
    from rgc.pci import lattice_bits_density

    rng = np.random.default_rng(0)
    g = rng.random((5, 40))
    b = lattice_bits_density(g, p=9, frac=0.2)
    assert b.size == 4 * 30 and b.sum() == round(0.2 * 120)
    sub = g[1:, 10:].reshape(-1)
    assert sub[b == 1].min() >= sub[b == 0].max()  # exactly the top sites
