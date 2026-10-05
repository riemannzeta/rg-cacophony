"""Spread of a perturbation across positions vs depth, and the dynamic exponent z.

S(t) = #{q > p : G(k + t, q) >= theta} for t = 1..n_layers - k (G = gain lattice, row t = slice k+t).
Power law  log S = a + b log t   (z = 1/b);   exponential  log S = a + t / tau.
Fits use points with 0 < S < saturation_frac * n_available; AIC = n log(RSS/n) + 2 * 2.
"""

import numpy as np


def spread_counts(gain: np.ndarray, p: int, theta: float) -> np.ndarray:
    """gain: [..., n_slices, T] with row 0 = injection slice. Returns S[..., t] for t = 1..n_slices-1."""
    return (gain[..., 1:, p + 1 :] >= theta).sum(-1)


def participation_spread(gain: np.ndarray, p: int) -> np.ndarray:
    """Threshold-free effective number of reached positions: (sum G)^2 / sum G^2 over q > p."""
    g = gain[..., 1:, p + 1 :].astype(np.float64)
    num, den = g.sum(-1) ** 2, (g**2).sum(-1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


def _ols(x: np.ndarray, y: np.ndarray):
    A = np.stack([np.ones_like(x), x], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    return coef, rss, (1 - rss / tss) if tss > 0 else np.nan


def fit_spread(S: np.ndarray, n_available: int, saturation_frac: float, min_points: int) -> dict:
    """Fit power law and exponential to a spread curve S[t-1], t = 1..len(S)."""
    t = np.arange(1, len(S) + 1, dtype=float)
    ok = (S > 0) & (S < saturation_frac * n_available)
    n = int(ok.sum())
    out = {"n_points": n, "fittable": n >= min_points}
    if not out["fittable"]:
        return out
    y = np.log(S[ok].astype(float))
    (a_p, b), rss_p, r2_p = _ols(np.log(t[ok]), y)
    (a_e, inv_tau), rss_e, r2_e = _ols(t[ok], y)
    eps = 1e-300
    aic_p = n * np.log(max(rss_p, eps) / n) + 4
    aic_e = n * np.log(max(rss_e, eps) / n) + 4
    out.update({
        "b": float(b), "z": float(1 / b) if b > 0 else None, "r2_power": float(r2_p),
        "inv_tau": float(inv_tau), "r2_exp": float(r2_e),
        "aic_power": float(aic_p), "aic_exp": float(aic_e), "delta_aic_exp_minus_power": float(aic_e - aic_p),
    })
    return out
