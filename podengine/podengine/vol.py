"""Volatility forecasts.

Volatility is the most predictable quantity in markets, so this is where the
engine spends model complexity. Two forecasters compete:

* EWMA  - the RiskMetrics baseline.
* HAR   - Corsi's (2009) cascade model: tomorrow's variance explained by the
          last day, week, month and year of variance. This is the volatility
          analogue of the turbulent energy cascade (large scales feed small).

Both are scored out of sample with the QLIKE loss and blended by exponential
weights, so whichever has been forecasting better recently gets more say.
Everything at row t uses information up to the close of day t only.
"""
import warnings

import numpy as np
import pandas as pd

ANN = 252


def _rolling_mean(a, w):
    """Trailing mean over w rows of a 2-D array, NaN until w valid rows exist."""
    df = pd.DataFrame(a)
    return df.rolling(w, min_periods=w).mean().to_numpy()


def qlike(realized, forecast):
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1.0


def forecast_vol(md, horizon=5, ewma_halflife=21, refit_days=252, min_days=750, eta=50.0):
    """Return (vol, diag). `vol` is the annualised volatility forecast for the
    next `horizon` days, made at each close. `diag` holds the out-of-sample
    scorecard of HAR against EWMA."""
    proxy = md.var_proxy()
    cols, idx = proxy.columns, proxy.index
    v = proxy.to_numpy(dtype=float)
    r2 = (np.log(md.close / md.close.shift(1)) ** 2).to_numpy(dtype=float)
    T, N = v.shape

    # bias correction: range estimators miss jumps/discreteness, so rescale the
    # proxy to match close-to-close variance using past data only
    cs_r2 = pd.DataFrame(np.where(np.isnan(v), np.nan, r2)).expanding(min_periods=252).mean().to_numpy()
    cs_v = pd.DataFrame(v).expanding(min_periods=252).mean().to_numpy()
    scale = np.clip(cs_r2 / cs_v, 0.5, 2.0)
    scale = np.where(np.isnan(scale), 1.0, scale)
    v = v * scale

    ew = pd.DataFrame(v).ewm(halflife=ewma_halflife, min_periods=20).mean().to_numpy()

    m5, m22, m252 = _rolling_mean(v, 5), _rolling_mean(v, 22), _rolling_mean(v, 252)
    floor = np.maximum(0.05 * m22, 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        X = np.stack([np.log(np.maximum(v, floor)), np.log(np.maximum(m5, floor)),
                      np.log(m22), np.log(m252)], axis=2)              # T x N x 4
        fwd = pd.DataFrame(v[::-1]).rolling(horizon, min_periods=horizon).mean().to_numpy()[::-1]
        fwd = np.vstack([fwd[1:], np.full((1, N), np.nan)])            # mean var over t+1..t+h
        y = np.log(np.maximum(fwd, floor))
    valid = np.isfinite(X).all(axis=2) & np.isfinite(y)

    har = np.full((T, N), np.nan)
    coefs = []
    fit_points = list(range(min_days, T, refit_days))
    for k, fp in enumerate(fit_points):
        last = fp - horizon                       # targets fully observed by day fp
        m = valid[:last]
        if m.sum() < 2000:
            continue
        A = np.column_stack([np.ones(m.sum()), X[:last][m]])
        b, *_ = np.linalg.lstsq(A, y[:last][m], rcond=None)
        resid = y[:last][m] - A @ b
        s2 = resid.var()
        end = fit_points[k + 1] if k + 1 < len(fit_points) else T
        seg = X[fp:end]
        ok = np.isfinite(seg).all(axis=2)
        pred = b[0] + seg @ b[1:] + 0.5 * s2
        har[fp:end] = np.where(ok, np.exp(pred), np.nan)
        coefs.append((idx[fp], b))

    # out-of-sample scorecard (loss for the forecast made at t is known at t+h)
    with np.errstate(invalid="ignore", divide="ignore"):
        l_har = qlike(fwd, har)
        l_ew = qlike(fwd, ew)
    both = np.isfinite(l_har) & np.isfinite(l_ew)
    with np.errstate(invalid="ignore"):
        d = np.where(both, l_ew - l_har, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_day = np.nanmean(d, axis=1) if both.any() else np.full(T, np.nan)
    d_known = pd.Series(d_day).shift(horizon)                       # only losses already observed
    edge = d_known.rolling(504, min_periods=60).mean().to_numpy()
    w_har = 1.0 / (1.0 + np.exp(-eta * np.nan_to_num(edge, nan=0.0)))
    w_har = np.where(np.isnan(edge), 0.5, w_har)[:, None]

    blend = np.where(np.isfinite(har), w_har * har + (1 - w_har) * ew, ew)
    vol = pd.DataFrame(np.sqrt(blend * ANN), index=idx, columns=cols)
    vol = vol.where(md.close.notna())

    diag = {
        "qlike_har": float(np.nanmean(l_har[both])) if both.any() else float("nan"),
        "qlike_ewma": float(np.nanmean(l_ew[both])) if both.any() else float("nan"),
        "har_weight_now": float(w_har[-1, 0]),
        "har_coefs": [float(x) for x in coefs[-1][1]] if coefs else None,
        "uses_range_data": md.high is not None,
    }
    return vol, diag
