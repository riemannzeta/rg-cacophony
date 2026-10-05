"""PCI analogue: binarise a gain lattice, flatten in layer order, normalised LZ76 complexity.

Following Casali et al. (2013): PCI = c(n) * log2(n) / (n * H(p1)), where c(n) is the
Lempel-Ziv (1976) phrase count computed with the Kaspar-Schuster algorithm, n the string length
and H the binary source entropy of the string. A string with p1 in {0, 1} carries no
information and is assigned PCI = 0.
"""

import math

import numpy as np


def lz76(bits) -> int:
    """Lempel-Ziv 1976 complexity (number of phrases), exhaustive-history parsing.

    Vectorised equivalent of Kaspar & Schuster (1987): a new phrase starting at i is one symbol
    longer than the longest prefix of s[i:] that also starts at some j < i (overlap allowed). The
    phrase count is identical to `lz76_ks` (verified exactly in tests/test_lz76.py) and ~10x faster.
    """
    s = np.asarray(bits, dtype=np.uint8)
    n = s.size
    if n == 0:
        return 0
    c, i = 1, 1  # the first symbol is always its own phrase
    while i < n:
        cand = np.flatnonzero(s[:i] == s[i])  # earlier starts j whose symbol matches s[i]
        if cand.size == 0:
            i += 1  # unseen symbol: a phrase of length 1
        else:
            m = 0  # s[i : i + m + 1] is matched by some s[j : j + m + 1], j < i
            while i + m + 1 < n:
                nxt = cand[s[cand + m + 1] == s[i + m + 1]]
                if nxt.size == 0:
                    break
                cand, m = nxt, m + 1
            i += m + 2  # matched part plus one new symbol (the last phrase may be incomplete)
        c += 1
    return c


def lz76_ks(bits) -> int:
    """Lempel-Ziv 1976 complexity (number of phrases), Kaspar & Schuster (1987) reference algorithm."""
    s = bytes(np.asarray(bits, dtype=np.uint8))
    n = len(s)
    if n == 0:
        return 0
    if n == 1:
        return 1
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def binary_entropy(p1: float) -> float:
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    return -p1 * math.log2(p1) - (1 - p1) * math.log2(1 - p1)


def normalised_lz(bits) -> float:
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    h = binary_entropy(float(bits.mean())) if n else 0.0
    if n < 2 or h == 0.0:
        return 0.0
    return lz76(bits) * math.log2(n) / (n * h)


def lattice_bits(gain: np.ndarray, p: int, theta: float) -> np.ndarray:
    """Binarise a gain lattice and flatten in layer order.

    gain: [n_slices, T] gain G = r / eps for slices L = k..n_layers (row 0 is the injection slice).
    Uses slices L = k+1..n (row 0 only contains the injected site) and positions q > p
    (the self-column q = p is excluded by construction). Layer-major: each layer contributes its
    vector over q in natural order.
    """
    return (gain[1:, p + 1 :] >= theta).astype(np.uint8).reshape(-1)


def pci(gain: np.ndarray, p: int, theta: float) -> float:
    return normalised_lz(lattice_bits(gain, p, theta))


def lattice_bits_density(gain: np.ndarray, p: int, frac: float) -> np.ndarray:
    """Density-matched binarisation (PREREG-3 secondary): the top `frac` of downstream sites are on.

    Same site set and layer-major order as `lattice_bits`; exactly round(frac * n) sites are on, ties
    broken by position in the flattened order (stable sort), so p1 is fixed and PCI measures arrangement.
    """
    g = gain[1:, p + 1 :].reshape(-1)
    n_on = int(round(frac * g.size))
    bits = np.zeros(g.size, dtype=np.uint8)
    if n_on:
        bits[np.argsort(-g, kind="stable")[:n_on]] = 1
    return bits


def pci_task(args) -> list[tuple[float, float]]:
    """(gain lattice, p, thetas[, dm_frac]) -> [(PCI, p1) per theta] (+ [(PCI_dm, frac)] if dm_frac).

    Module-level so worker pools can pickle it."""
    gain, p, thetas = args[:3]
    out = []
    for th in thetas:
        bits = lattice_bits(gain, p, th)
        out.append((normalised_lz(bits), float(bits.mean()) if bits.size else 0.0))
    if len(args) > 3 and args[3]:
        out.append((normalised_lz(lattice_bits_density(gain, p, args[3])), float(args[3])))
    return out
