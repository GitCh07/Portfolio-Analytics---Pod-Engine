"""Precompute everything the animated viewer draws (viz/data/*.json).

The browser does no maths: every frame is computed here, from the engine's
own backtest result, using a separate random seed. Nothing in this module
writes to the engine's state, so engine outputs are identical with or without
an export (a test checks this).

Files written (schema in viz/data/schema.json):
  meta.json        what was run, when, on which data, and the labels
  fan.json         Monte Carlo percentile bands + realised path, one frame per forecast date
  calibration.json PIT histogram, cumulative as the replay advances, with tail flags
  regimes.json     regime-conditional return densities (ridgeline), per date
  frontier.json    rolling efficient frontier (Ledoit-Wolf covariance) + the portfolio's position
  exposure.json    long/short weights by asset class, gross and net, over time
"""
import datetime as _dt
import json
import os
import shutil

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .montecarlo import _fhs, _inputs, simulate_holdings

ANN = 252
SCHEMA_VERSION = 1
QS = (5, 25, 50, 75, 95)
REGIME_NAMES = {1: ["single"], 2: ["calm", "stress"], 3: ["calm", "stress", "crisis"]}
LABEL_REPLAY = "Simulated / historical replay · not a forecast"
VIZ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viz")


# ------------------------------------------------------------------ helpers

def _r(x, d=6):
    """Round for compact JSON; NaN/inf -> None."""
    a = np.asarray(x, dtype=float)
    out = np.round(a, d).tolist()
    if a.ndim == 0:
        return out if np.isfinite(a) else None
    def clean(v):
        if isinstance(v, list):
            return [clean(u) for u in v]
        return v if (v is not None and np.isfinite(v)) else None
    return clean(out)


def _date(prep, p):
    return str(prep.dates[p].date())


def _spaced(res, spacing, horizon=0, need_regime=False, max_frames=None):
    """Rebalance records at least `spacing` days apart that hold positions and
    leave `horizon` days of future data."""
    prep = res.prep
    out, last = [], -10 ** 9
    for rec in res.rebalances:
        p = rec["p"]
        if p - last < spacing or p + horizon >= prep.T or p < res.start - 1:
            continue
        if need_regime and prep.fit_at(p) < 0:
            continue
        if np.abs(rec["W"]).sum() <= 0:
            continue
        out.append(rec)
        last = p
    if max_frames and len(out) > max_frames:
        idx = np.unique(np.linspace(0, len(out) - 1, max_frames).round().astype(int))
        out = [out[i] for i in idx]
    return out


def pit_value(samples, x):
    """Probability integral transform: share of the forecast distribution at or
    below the outcome (mid-rank for ties). Uniform on [0,1] if the forecast
    distribution is correct."""
    s = np.asarray(samples, dtype=float)
    return float(((s < x).sum() + 0.5 * (s == x).sum()) / len(s))


def pit_summary(pits, bins=10, tail=0.1, min_n=30):
    """Histogram + uniformity checks for a list of PIT values.
    flag: 'thin'  = outcomes landed in the outer tails more often than forecast
                    (the model's tails are too thin, ranges too narrow);
          'fat'   = less often than forecast (tails too fat, ranges too wide);
          'ok'    = consistent with the forecast; 'insufficient' = under min_n."""
    u = np.asarray(pits, dtype=float)
    n = len(u)
    counts = np.histogram(u, bins=bins, range=(0.0, 1.0))[0]
    out = {"n": int(n), "counts": counts.tolist()}
    if n == 0:
        return dict(out, tail_share=None, tail_expected=2 * tail, tail_p=None, chi2_p=None, ks_p=None,
                    flag="insufficient")
    k = int(((u < tail) | (u > 1 - tail)).sum())
    tail_p = float(stats.binomtest(k, n, 2 * tail).pvalue)
    chi2_p = float(stats.chisquare(counts).pvalue) if n >= bins * 5 else None
    ks_p = float(stats.kstest(u, "uniform").pvalue)
    if n < min_n:
        flag = "insufficient"
    elif tail_p < 0.05:
        flag = "thin" if k / n > 2 * tail else "fat"
    else:
        flag = "ok"
    return dict(out, tail_share=k / n, tail_expected=2 * tail, tail_p=tail_p, chi2_p=chi2_p, ks_p=ks_p, flag=flag)


def ledoit_wolf(X):
    """Ledoit & Wolf (2004) shrinkage of the sample covariance toward a scaled
    identity. X: (n obs x p assets). Returns (Sigma, shrinkage intensity)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    delta = float(((S - F) ** 2).sum())
    beta = float(((Xc ** 2).sum(axis=1) ** 2).sum() / n ** 2 - (S ** 2).sum() / n)
    shrink = 0.0 if delta <= 0 else float(np.clip(beta / delta, 0.0, 1.0))
    return shrink * F + (1 - shrink) * S, shrink


def bayes_stein_means(X, S):
    """Jorion (1986) Bayes-Stein shrinkage of sample means toward the mean of
    the global minimum-variance portfolio. X: (T x N) returns, S: their (N x N)
    covariance in the same units. Returns (shrunk means, shrinkage weight phi).
    Uses the given (here Ledoit-Wolf) covariance in place of the sample one."""
    T, N = X.shape
    m = X.mean(axis=0)
    Si = np.linalg.inv(S)
    one = np.ones(N)
    mu0 = float(one @ Si @ m / (one @ Si @ one))
    d = m - mu0
    phi = float((N + 2) / ((N + 2) + T * (d @ Si @ d)))
    phi = min(max(phi, 0.0), 1.0)
    return (1 - phi) * m + phi * mu0, phi


def kde_on_grid(x, grid):
    """Gaussian KDE (Silverman bandwidth) evaluated on an even grid by binning
    the sample finely and convolving with the kernel. Integrates to 1 over the grid."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    sd = x.std()
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 0.9 * min(sd, iqr / 1.34 if iqr > 0 else sd) * n ** -0.2
    lo, hi = grid[0], grid[-1]
    fine = np.linspace(lo, hi, 8 * len(grid))
    dx = fine[1] - fine[0]
    counts = np.histogram(x, bins=len(fine), range=(lo - dx / 2, hi + dx / 2))[0].astype(float)
    m = int(np.ceil(4 * h / dx))
    ker = np.exp(-0.5 * (np.arange(-m, m + 1) * dx / h) ** 2)
    dens = np.convolve(counts, ker / ker.sum(), mode="same")
    dens = np.interp(grid, fine, dens)
    area = np.trapezoid(dens, grid) if hasattr(np, "trapezoid") else np.trapz(dens, grid)
    return dens / area if area > 0 else dens, float(h)


def stationary(A):
    w, v = np.linalg.eig(np.asarray(A).T)
    s = np.real(v[:, np.argmin(np.abs(w - 1))])
    s = np.abs(s)
    return s / s.sum()


# ------------------------------------------------------------------ 1 + 2: fan and calibration

def fan_and_calibration(res, cfg, horizon=21, n_paths=2000, seed_offset=101):
    """One frame per forecast date (non-overlapping horizons): Monte Carlo
    percentile bands of growth of 1.00 for the engine's target weights held
    for `horizon` days, the realised path of the same weights, the PIT of the
    realised outcome, and the cumulative PIT histogram up to that frame.
    Gross of trading costs and financing; returns over cash."""
    prep = res.prep
    rng = np.random.default_rng(cfg["seed"] + seed_offset)
    R, rf = prep.R, prep.rf
    frames, pits = [], []
    for rec in _spaced(res, horizon, horizon, need_regime=True):
        p = rec["p"]
        avail = np.flatnonzero(rec["W"] != 0)
        w = rec["W"][avail]
        drift = rec["sr_port"] * rec["ex_ante_vol"]
        daily = simulate_holdings(prep, p, avail, w, cfg, horizon, n_paths, rng, drift)["daily"]
        wealth = np.exp(np.cumsum(np.log1p(np.clip(daily, -0.99, None)), axis=1))
        bands = np.percentile(wealth, QS, axis=0)
        bands = np.hstack([np.ones((len(QS), 1)), bands])
        real_d = (np.nan_to_num(R[p + 1:p + 1 + horizon][:, avail], nan=0.0) - rf[p + 1:p + 1 + horizon, None]) @ w
        real = np.concatenate([[1.0], np.cumprod(1 + real_d)])
        u = pit_value(wealth[:, -1], real[-1])
        pits.append(u)
        frames.append({"date": _date(prep, p), "end": _date(prep, p + horizon),
                       "bands": {str(q): _r(b) for q, b in zip(QS, bands)},
                       "realized": _r(real), "pit": round(u, 6),
                       "ex_ante_vol": round(float(rec["ex_ante_vol"]), 6)})
    cal = []
    for i in range(len(pits)):
        s = pit_summary(pits[:i + 1])
        cal.append({"date": frames[i]["date"], "pit": frames[i]["pit"], **{k: _r(v) if isinstance(v, float) else v
                                                                          for k, v in s.items()}})
    fan = {"schema_version": SCHEMA_VERSION, "label": LABEL_REPLAY, "costs": "gross of trading costs and financing",
           "horizon_days": horizon, "n_paths": n_paths, "percentiles": list(QS),
           "units": "growth of 1.00, return over cash", "frames": frames}
    calj = {"schema_version": SCHEMA_VERSION, "label": LABEL_REPLAY, "bins": 10, "tail": 0.1,
            "horizon_days": horizon, "frames": cal}
    return fan, calj


# ------------------------------------------------------------------ 3: regimes

def regimes(res, cfg, horizon=21, n_paths=10000, max_frames=120, grid_n=160, seed_offset=202):
    """Regime-conditional distribution of the target portfolio's `horizon`-day
    return, one ridge per regime: the Monte Carlo starts in that regime and
    then follows the fitted transition probabilities. The unconditional
    reference starts from the chain's long-run (stationary) regime mix."""
    prep = res.prep
    rng = np.random.default_rng(cfg["seed"] + seed_offset)
    recs = _spaced(res, 21, 0, need_regime=True, max_frames=max_frames)
    raw = []
    for rec in recs:
        p = rec["p"]
        avail = np.flatnonzero(rec["W"] != 0)
        w = rec["W"][avail]
        Zh, G, A, p_next, v0, lr = _inputs(prep, p, avail, cfg)
        K = G.shape[1]
        drift = rec["sr_port"] * rec["ex_ante_vol"] / ANN
        starts = [np.eye(K)[j] for j in range(K)] + [stationary(A)]
        totals = []
        for s in starts:
            port, _ = _fhs(Zh, G, A, s, v0, lr, horizon, n_paths, cfg["mc_block"], cfg["garch_alpha"],
                           cfg["garch_beta"], rng, w=w)
            totals.append(np.expm1(np.log1p(np.clip(port + drift, -0.99, None)).sum(axis=1)))
        k = prep.fit_at(p)
        probs = prep.reg_np[p, :K]
        raw.append((rec, K, totals, probs, prep.reg_fits[k]["params"]))
    if not raw:
        return {"schema_version": SCHEMA_VERSION, "label": LABEL_REPLAY, "grid": [], "frames": []}
    pooled = np.concatenate([np.concatenate(t) for _, _, t, _, _ in raw])
    lo, hi = np.quantile(pooled, [0.002, 0.998])
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, grid_n)
    frames = []
    for rec, K, totals, probs, par in raw:
        ridges = []
        names = REGIME_NAMES.get(K, [f"state {i}" for i in range(K)])
        for j, t in enumerate(totals):
            dens, h = kde_on_grid(t, grid)
            ridges.append({"name": names[j] if j < K else "unconditional",
                           "density": _r(dens, 4), "bandwidth": round(h, 6),
                           "q": _r(np.quantile(t, [0.05, 0.5, 0.95])),
                           "regime_vol": round(float(par["sigma"][j] * np.sqrt(ANN)), 4) if j < K else None})
        frames.append({"date": _date(prep, rec["p"]), "probs": _r(probs, 4), "current": int(np.argmax(probs)),
                       "ridges": ridges[:K], "unconditional": ridges[K]})
    return {"schema_version": SCHEMA_VERSION, "label": LABEL_REPLAY,
            "costs": "gross of trading costs and financing", "horizon_days": horizon, "n_paths": n_paths,
            "grid": _r(grid), "units": f"{horizon}-day return over cash", "frames": frames}


# ------------------------------------------------------------------ 4: frontier

def _frontier_point(mu, S, target, cfg, x0=None):
    """max mu'w  s.t. w'Sw <= target^2, gross <= cap, per-asset caps, shorts per
    profile. Solved with w = a - b, a, b >= 0, so every constraint is smooth."""
    n = len(mu)
    cap, G, short = cfg["asset_cap"], cfg["gross_cap"], cfg["allow_short"]
    bounds = [(0.0, cap)] * n + [(0.0, cap if short else 0.0)] * n
    w = lambda x: x[:n] - x[n:]
    cons = [{"type": "ineq", "fun": lambda x: target ** 2 - w(x) @ S @ w(x),
             "jac": lambda x: np.concatenate([-2 * S @ w(x), 2 * S @ w(x)])},
            {"type": "ineq", "fun": lambda x: G - x.sum(), "jac": lambda x: -np.ones(2 * n)}]
    if x0 is None:
        x0 = np.zeros(2 * n)
    r = minimize(lambda x: -(mu @ w(x)), x0, jac=lambda x: np.concatenate([-mu, mu]), bounds=bounds,
                 constraints=cons, method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
    return w(r.x), r.x


def frontier_curve(mu, S, cfg, vol_grid):
    pts, x0 = [], None
    for v in vol_grid:
        wv, x0 = _frontier_point(mu, S, v, cfg, x0)
        pts.append((float(np.sqrt(max(wv @ S @ wv, 0.0))), float(mu @ wv), wv))
    return pts


def frontier(res, cfg, window=252, spacing=63, n_points=14, max_vol=None):
    """Rolling efficient frontier: for each date, Ledoit-Wolf covariance and
    Bayes-Stein-shrunk average excess returns over the trailing `window` days,
    then the highest return achievable at each volatility under the engine's
    constraints (gross leverage cap, per-asset cap, shorting allowed or not).
    Expected returns are backward-looking, so the frontier is an in-sample
    'best you could have done' line; a real portfolio is expected to sit below it."""
    prep = res.prep
    ex = prep.R - prep.rf[:, None]
    max_vol = max_vol or 2.0 * cfg["vol_target"]
    vol_grid = np.linspace(max_vol / n_points, max_vol, n_points)
    frames = []
    for rec in _spaced(res, spacing):
        p = rec["p"]
        if p - window < 0:
            continue
        avail = np.flatnonzero(prep.elig[p])
        X = ex[p - window + 1:p + 1][:, avail]
        ok = np.isfinite(X).all(axis=0)
        avail, X = avail[ok], X[:, ok]
        if len(avail) < 3:
            continue
        S_d, shrink = ledoit_wolf(X)
        mu_d, phi = bayes_stein_means(X, S_d)
        S, mu = S_d * ANN, mu_d * ANN
        pts = frontier_curve(mu, S, cfg, vol_grid)
        keep, best = [(0.0, 0.0)], -np.inf
        for v, m, _ in pts:                               # efficient part only: return must keep rising
            if m > best + 1e-9:
                keep.append((v, m))
                best = m
        held = res.held[min(p + 1, prep.T - 1), avail]
        dot = (float(np.sqrt(max(held @ S @ held, 0.0))), float(mu @ held))
        sh = np.array([m / v if v > 0 else -np.inf for v, m in keep])
        # with leverage allowed the frontier starts as a straight line from cash, so many points tie
        # for the best Sharpe ratio; mark the last of them (where the leverage or position caps start to bind)
        t = int(np.flatnonzero(sh >= sh.max() - 0.005 * abs(sh.max()))[-1])
        frames.append({"date": _date(prep, p), "n_assets": int(len(avail)), "shrinkage": round(shrink, 4), "mean_shrinkage": round(phi, 4),
                       "frontier": [[round(v, 6), round(m, 6)] for v, m in keep],
                       "tangency": [round(keep[t][0], 6), round(keep[t][1], 6)],
                       "portfolio": [round(dot[0], 6), round(dot[1], 6)]})
    return {"schema_version": SCHEMA_VERSION, "label": LABEL_REPLAY,
            "costs": "gross of trading costs and financing", "window_days": window,
            "covariance": "Ledoit-Wolf shrinkage to scaled identity",
            "expected_returns": "trailing-window average excess return, Bayes-Stein shrunk (Jorion 1986); "
                                "backward-looking and in-sample",
            "constraints": {"allow_short": bool(cfg["allow_short"]), "gross_cap": float(cfg["gross_cap"]),
                            "asset_cap": float(cfg["asset_cap"])},
            "frames": frames}


# ------------------------------------------------------------------ 5: exposure

def exposure(res, cfg, step=5):
    prep = res.prep
    idx = np.arange(res.start, prep.T, step)
    if len(idx) and idx[-1] != prep.T - 1:
        idx = np.append(idx, prep.T - 1)
    H = res.held[idx]
    classes = sorted({m["cls"] for m in prep.meta_list})
    cls_of = np.array([m["cls"] for m in prep.meta_list])
    long_, short_ = {}, {}
    for c in classes:
        sub = H[:, cls_of == c]
        long_[c] = _r(np.where(sub > 0, sub, 0).sum(axis=1), 5)
        short_[c] = _r(np.where(sub < 0, sub, 0).sum(axis=1), 5)
    return {"schema_version": SCHEMA_VERSION, "label": "Historical replay of the engine's backtest positions",
            "units": "share of account equity (1.0 = 100%)", "dates": [_date(prep, i) for i in idx],
            "classes": classes, "long": long_, "short": short_,
            "gross": _r(np.abs(H).sum(axis=1), 5), "net": _r(H.sum(axis=1), 5),
            "gross_cap": float(cfg["gross_cap"])}


# ------------------------------------------------------------------ export

def export_viz(res, cfg, viz_dir=None, reports=(), quick=False, verbose=True):
    """Write all viewer data. `reports`: paths of report.html / optimizer.html
    to copy into viz/reports/ so the tabbed page can show them."""
    say = print if verbose else (lambda *a, **k: None)
    viz_dir = viz_dir or VIZ_DIR
    data = os.path.join(viz_dir, "data")
    os.makedirs(data, exist_ok=True)
    if os.path.abspath(viz_dir) != os.path.abspath(VIZ_DIR):      # a separate folder: bring the viewer with it
        for item in ("index.html", "viz.css", "selftest.html", "js", os.path.join("data", "schema.json")):
            src, dst = os.path.join(VIZ_DIR, item), os.path.join(viz_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif os.path.exists(src):
                shutil.copy2(src, dst)
    md = res.prep.md
    say("  viz: forecast fan and calibration ...")
    fan, cal = fan_and_calibration(res, cfg, n_paths=800 if quick else 2000)
    say("  viz: regime ridgeline ...")
    reg = regimes(res, cfg, n_paths=3000 if quick else 10000, max_frames=40 if quick else 120)
    say("  viz: rolling efficient frontier ...")
    fr = frontier(res, cfg, spacing=126 if quick else 63)
    say("  viz: exposure ...")
    ex = exposure(res, cfg)
    out = {"fan.json": fan, "calibration.json": cal, "regimes.json": reg, "frontier.json": fr, "exposure.json": ex}
    meta = {"schema_version": SCHEMA_VERSION, "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "source": md.source, "synthetic": bool(md.synthetic), "profile": cfg["profile"],
            "asof": _date(res.prep, res.prep.T - 1), "label": LABEL_REPLAY,
            "files": sorted(out), "reports": []}
    rdir = os.path.join(viz_dir, "reports")
    os.makedirs(rdir, exist_ok=True)
    for path in reports:
        if path and os.path.exists(path):
            shutil.copy2(path, os.path.join(rdir, os.path.basename(path)))
            meta["reports"].append(os.path.basename(path))
    for name, obj in out.items():
        with open(os.path.join(data, name), "w") as fh:
            json.dump(obj, fh, separators=(",", ":"))
    with open(os.path.join(data, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    say(f"  viz data written to {data}")
    return out, meta
