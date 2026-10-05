"""PCI analogue: binarise a gain lattice, flatten in layer order, normalised LZ76 complexity.

Following Casali et al. (2013): PCI = c(n) * log2(n) / (n * H(p1)), where c(n) is the
Lempel-Ziv (1976) phrase count computed with the Kaspar-Schuster algorithm, n the string length
and H the binary source entropy of the string. A string with p1 in {0, 1} carries no
information and is assigned PCI = 0.
"""

import math

import numpy as np


def lz76(bits) -> int:
    """Lempel-Ziv 1976 complexity (number of phrases), Kaspar & Schuster (1987) algorithm."""
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


def pci_task(args) -> list[tuple[float, float]]:
    """(gain lattice, p, thetas) -> [(PCI, p1) per theta]. Module-level so worker pools can pickle it."""
    gain, p, thetas = args
    out = []
    for th in thetas:
        bits = lattice_bits(gain, p, th)
        out.append((normalised_lz(bits), float(bits.mean()) if bits.size else 0.0))
    return out
