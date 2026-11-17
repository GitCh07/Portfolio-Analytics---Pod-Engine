"""Covariance estimation built to survive being fed to an optimiser.

correlation = EWMA correlation of volatility-standardised returns
            -> random-matrix clipping (eigenvalues inside the Marchenko-Pastur
               noise band are flattened; the ones above carry real structure)
            -> light shrinkage to the identity for conditioning
            -> optional blend with the stress-regime correlation matrix,
               weighted by the probability of being in stress tomorrow.
covariance  = D C D with D = the volatility forecasts.
"""
import numpy as np


def ewma_corr(z, halflife):
    """EWMA correlation of a (days x assets) array; NaN treated as 0 (no info)."""
    z = np.nan_to_num(z, nan=0.0)
    n = z.shape[0]
    w = 0.5 ** (np.arange(n - 1, -1, -1) / halflife)
    w = w / w.sum()
    S = (z * w[:, None]).T @ z
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C, 1.0 / np.sum(w ** 2)          # correlation, effective sample size


def weighted_corr(z, w):
    z = np.nan_to_num(z, nan=0.0)
    sw = w.sum()
    if sw <= 0:
        return np.eye(z.shape[1])
    S = (z * (w / sw)[:, None]).T @ z
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def rmt_clip(C, n_eff):
    """Flatten eigenvalues below the Marchenko-Pastur upper edge (1+sqrt(q))^2,
    q = assets / effective observations, preserving the trace."""
    N = C.shape[0]
    if N < 3:
        return C
    q = N / max(n_eff, N + 1)
    edge = (1 + np.sqrt(q)) ** 2
    w, V = np.linalg.eigh(C)
    noise = w < edge
    if noise.sum() >= 2 and (~noise).any():
        w = w.copy()
        w[noise] = w[noise].mean()
    w = np.clip(w, 1e-8, None)
    C2 = (V * w) @ V.T
    d = np.sqrt(np.diag(C2))
    C2 = C2 / np.outer(d, d)
    np.fill_diagonal(C2, 1.0)
    return C2


def clean_corr(z, halflife=126, shrink=0.10):
    C, n_eff = ewma_corr(z, halflife)
    C = rmt_clip(C, n_eff)
    N = C.shape[0]
    return (1 - shrink) * C + shrink * np.eye(N)


def to_cov(C, vol_ann):
    d = np.asarray(vol_ann, dtype=float)
    return C * np.outer(d, d)


def risk_contributions(w, Sigma):
    """Euler decomposition: contributions sum exactly to portfolio volatility."""
    w = np.asarray(w, dtype=float)
    port = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    if port <= 0:
        return np.zeros_like(w), 0.0
    return w * (Sigma @ w) / port, port
