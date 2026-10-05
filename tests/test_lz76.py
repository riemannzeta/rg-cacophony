import numpy as np

from rgc.pci import lz76, normalised_lz


def bits(s):
    return np.array([int(ch) for ch in s], dtype=np.uint8)


def test_textbook_example():
    # Lempel & Ziv (1976) / Kaspar & Schuster (1987): 0·001·10·100·1000·101
    assert lz76(bits("0001101001000101")) == 6


def test_small_cases():
    assert lz76(bits("0")) == 1
    assert lz76(bits("01")) == 2
    assert lz76(bits("0000000000")) == 2
    assert lz76(bits("0101010101")) == 3


def test_random_strings_normalise_near_one():
    rng = np.random.default_rng(0)
    for p1 in (0.5, 0.2):
        x = (rng.random(20000) < p1).astype(np.uint8)
        assert 0.85 < normalised_lz(x) < 1.1


def test_degenerate_strings_are_zero():
    assert normalised_lz(np.zeros(500, np.uint8)) == 0.0
    assert normalised_lz(np.ones(500, np.uint8)) == 0.0


def test_periodic_string_is_low():
    x = np.tile(bits("0011"), 1000)
    assert normalised_lz(x) < 0.05


def test_fast_lz76_matches_kaspar_schuster_exactly():
    from rgc.pci import lz76_ks

    rng = np.random.default_rng(1)
    for n in (1, 2, 3, 5, 17, 100, 257, 1000, 2304):
        for p1 in (0.0, 0.02, 0.1, 0.5, 0.9, 1.0):
            for _ in range(10 if n < 1000 else 3):
                x = (rng.random(n) < p1).astype(np.uint8)
                assert lz76(x) == lz76_ks(x)
    for per in ("0011", "01", "000111", "0010110"):
        x = bits(per * 300)
        assert lz76(x) == lz76_ks(x)
