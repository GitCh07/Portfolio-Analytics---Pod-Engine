"""Market regimes and turbulence.

* Hidden Markov model (Hamilton 1989) on a benchmark's daily returns: states
  are calm / stress / crisis, ordered by volatility. Fitted by EM on an
  expanding window, re-fitted about once a year, and *filtered* forward so
  that the probability at day t uses data up to t only.
* Financial turbulence (Kritzman & Li 2010): Mahalanobis distance of today's
  cross-asset move from its own history - high when assets move unusually
  far or in unusual combinations.
"""
import numpy as np
import pandas as pd


def _emission(x, mu, sig):
    return np.exp(-0.5 * ((x[:, None] - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)) + 1e-300


def hmm_filter(x, params, return_loglik=False):
    """Forward (filtered) state probabilities P(state_t | x_1..x_t)."""
    mu, sig, A, pi = params["mu"], params["sigma"], params["A"], params["pi"]
    B = _emission(x, mu, sig)
    T, K = B.shape
    out = np.empty((T, K))
    p = pi
    ll = 0.0
    for t in range(T):
        a = (p @ A if t else p) * B[t]
        s = a.sum()
        p = a / s
        out[t] = p
        ll += np.log(s)
    return (out, ll) if return_loglik else out


def _em(x, K, init, max_iter=60, tol=1e-5):
    mu, sig, A, pi = (np.array(init[k], dtype=float) for k in ("mu", "sigma", "A", "pi"))
    T = len(x)
    prev = -np.inf
    for _ in range(max_iter):
        B = _emission(x, mu, sig)
        alpha = np.empty((T, K))
        c = np.empty(T)
        p = pi
        for t in range(T):
            a = (p @ A if t else p) * B[t]
            c[t] = a.sum()
            p = a / c[t]
            alpha[t] = p
        beta = np.empty((T, K))
        b = np.ones(K)
        beta[-1] = b
        for t in range(T - 2, -1, -1):
            b = A @ (B[t + 1] * b) / c[t + 1]
            beta[t] = b
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi = (alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :] / c[1:, None, None]).sum(0)
        A = xi / xi.sum(1, keepdims=True)
        A = 0.999 * A + 0.001 / K
        g = gamma.sum(0)
        mu = (gamma * x[:, None]).sum(0) / g
        sig = np.sqrt((gamma * (x[:, None] - mu) ** 2).sum(0) / g)
        sig = np.maximum(sig, 1e-5)
        pi = gamma[0]
        ll = np.log(c).sum()
        if abs(ll - prev) < tol * T:
            break
        prev = ll
    order = np.argsort(sig)
    return {"mu": mu[order], "sigma": sig[order], "A": A[np.ix_(order, order)], "pi": pi[order],
            "loglik": float(ll), "occupancy": (g / T)[order]}


def _init(x, K):
    roll = pd.Series(x).rolling(21, min_periods=5).std().bfill().to_numpy()
    qs = np.quantile(roll, np.linspace(0.25, 0.97, K)) if K > 1 else np.array([x.std()])
    A = np.full((K, K), 0.03 / max(K - 1, 1))
    np.fill_diagonal(A, 0.97)
    return {"mu": np.zeros(K), "sigma": qs, "A": A, "pi": np.full(K, 1.0 / K)}


def fit_hmm(x, K=3, warm=None):
    """Fit a K-state Gaussian HMM; fall back to fewer states if one is degenerate."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    while True:
        init = warm if (warm is not None and len(warm["mu"]) == K) else _init(x, K)
        p = _em(x, K, init)
        s = p["sigma"]
        degenerate = K > 1 and (p["occupancy"].min() < 0.01 or np.any(s[1:] / s[:-1] < 1.15))
        if not degenerate or K == 1:
            return p
        K, warm = K - 1, None


def walk_forward_regimes(ret, K=3, refit_days=252, min_days=756):
    """Causal regime probabilities for a return series.

    Returns (probs, fits). `probs` has K columns (0 = calm ... K-1 = worst).
    `fits` is a list of dicts {start, params, hist}: `hist` holds the filtered
    probabilities of every day up to the end of that fit's segment under that
    fit's parameters (used by the Monte Carlo engine to weight history)."""
    x_all = ret.to_numpy(dtype=float)
    idx = ret.index
    good = np.isfinite(x_all)
    pos = np.flatnonzero(good)                 # positions with data
    x = x_all[good]
    T = len(x)
    probs = np.zeros((len(idx), K))
    probs[:, 0] = 1.0                          # before the first fit: assume calm
    fits, warm = [], None
    for fp in range(min_days, T, refit_days):
        params = fit_hmm(x[:fp], K, warm)
        warm = params
        end = min(fp + refit_days, T)
        filt = hmm_filter(x[:end], params)
        k = filt.shape[1]
        seg = np.zeros((end - fp, K))
        seg[:, :k] = filt[fp:end]
        probs[pos[fp:end]] = seg
        hist = np.zeros((len(idx), K))
        hist[:, 0] = 1.0
        full = np.zeros((end, K))
        full[:, :k] = filt
        hist[pos[:end]] = full
        fits.append({"start": idx[pos[fp]], "start_pos": int(pos[fp]), "params": params, "hist": hist})
    # forward-fill across days where the benchmark had no return
    P = pd.DataFrame(probs, index=idx, columns=[f"s{i}" for i in range(K)])
    if (~good).any():
        P[~good] = np.nan
        P = P.ffill().fillna(pd.Series({c: float(i == 0) for i, c in enumerate(P.columns)}))
    return P, fits


def next_day_stress(prob_row, params, K):
    """Probability of being in a non-calm state tomorrow."""
    k = len(params["mu"])
    p = np.asarray(prob_row[:k]) @ params["A"]
    return float(1.0 - p[0])


def turbulence(composites, halflife=252, smooth=10, rank_window=1260, min_periods=252):
    """Kritzman-Li turbulence percentile.

    `composites`: DataFrame of standardised daily returns, one column per asset
    class. The distance at day t uses the covariance estimated through t-1.
    Returns the percentile (0-1) of the smoothed distance within its own
    trailing history."""
    Y = composites.to_numpy(dtype=float)
    T, M = Y.shape
    lam = 0.5 ** (1.0 / halflife)
    S = np.eye(M)
    seen = np.zeros(M)
    d = np.full(T, np.nan)
    for t in range(T):
        y = Y[t]
        ok = np.isfinite(y) & (seen >= 60)
        if ok.sum() >= 2:
            Ss = S[np.ix_(ok, ok)]
            try:
                d[t] = float(y[ok] @ np.linalg.solve(Ss + 1e-6 * np.eye(ok.sum()), y[ok])) / ok.sum()
            except np.linalg.LinAlgError:
                pass
        have = np.isfinite(y)
        yy = np.where(have, y, 0.0)
        upd = np.outer(yy, yy)
        mask = np.outer(have, have)
        a = np.maximum(1 - lam, 1.0 / (np.minimum.outer(seen, seen) + 1.0))   # expanding mean until warm
        S = np.where(mask, (1 - a) * S + a * upd, S)
        seen += have
    ds = pd.Series(d, index=composites.index).ewm(halflife=smooth, min_periods=5).mean()
    pct = ds.rolling(rank_window, min_periods=min_periods).rank(pct=True)
    return pct, ds
