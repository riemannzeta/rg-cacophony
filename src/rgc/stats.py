"""Statistics helpers: percentile CIs, permutation Spearman, hypergeometric tail."""

import numpy as np
from scipy import stats


def ci(samples: np.ndarray, axis: int = 0, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2 * 100
    return np.percentile(samples, a, axis=axis), np.percentile(samples, 100 - a, axis=axis)


def spearman_perm(x, y, n_perm: int, rng: np.random.Generator, alternative: str) -> tuple[float, float]:
    """Spearman rho with a one-sided permutation p-value ('greater' or 'less')."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rx, ry = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    rho = float((rx * ry).mean())
    perms = np.stack([rng.permutation(ry) for _ in range(n_perm)])
    null = (perms * rx).mean(1)
    hits = (null >= rho) if alternative == "greater" else (null <= rho)
    return rho, float((1 + hits.sum()) / (1 + n_perm))


def hypergeom_tail(k: int, population: int, successes: int, draws: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(population, successes, draws)."""
    return float(stats.hypergeom.sf(k - 1, population, successes, draws))


def shortest_band(jumps: np.ndarray, frac: float) -> tuple[int, int] | None:
    """Shortest run [s, e] of consecutive entries whose sum is >= frac * total rise.

    total rise R = max(cumsum(jumps)) (the peak of the curve starting from 0). Ties -> earliest.
    Returns None if R <= 0.
    """
    R = float(np.max(np.cumsum(jumps)))
    if R <= 0:
        return None
    target, n = frac * R, len(jumps)
    for width in range(1, n + 1):
        for s in range(0, n - width + 1):
            if jumps[s : s + width].sum() >= target:
                return s, s + width - 1
    return None
