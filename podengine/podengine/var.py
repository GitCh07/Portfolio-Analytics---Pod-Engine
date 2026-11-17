"""Value at Risk and Expected Shortfall: estimates, attribution and backtests.

Conventions: VaR and ES are reported as positive fractions of equity (a loss
of 0.021 = 2.1%). VaR at 99% = the loss exceeded on 1% of days. ES at 97.5% =
the average loss on the worst 2.5% of days (the Basel FRTB measure).

Three estimators, so their disagreement is visible rather than hidden:

* Normal (parametric)  - the engine's covariance matrix and a bell curve.
* Filtered historical  - the last ~4 years of real joint daily moves,
  each asset standardised by its own volatility forecast at the time, then
  rescaled to today's forecast (Barone-Adesi et al. 1999). Keeps fat tails
  and crash correlations; adapts to today's volatility.
* Regime Monte Carlo   - the engine's Monte Carlo (montecarlo.py), which adds
  regime switching and volatility clustering; used for the 10-day horizon,
  where scaling 1-day VaR by sqrt(10) ignores clustering.

Backtests (on a daily series of 1-day forecasts, each made with data up to
the previous close):

* Kupiec (1995) proportion-of-failures: are breaches as frequent as stated?
* Christoffersen (1998) independence: do breaches cluster (model too slow)?
  Conditional coverage = both together.
* Basel traffic light: breaches of 99% VaR in the last 250 days
  (green 0-4, yellow 5-9, red 10+).
* Acerbi-Szekely (2014) Z2 test of ES, with a Monte Carlo p-value computed
  from the model's own daily scenario sets.
"""
import numpy as np
from scipy import stats

from .vol import forecast_vol

ANN = 252
LEVELS = (0.95, 0.99)
ES_LEVEL = 0.975


# ------------------------------------------------------------- 1-day volatility

def daily_inputs(prep):
    """1-day-ahead volatility forecasts and the returns standardised by them.
    (The engine sizes positions on a 1-month forecast; VaR needs a 1-day one.)
    Cached on the Prep object."""
    if getattr(prep, "_vol1", None) is None:
        v1, _ = forecast_vol(prep.md, horizon=1, ewma_halflife=prep.base_cfg["ewma_halflife"])
        vol1 = v1.to_numpy(dtype=float)
        prev = np.vstack([np.full((1, vol1.shape[1]), np.nan), vol1[:-1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            Z1 = prep.R / (prev / np.sqrt(ANN))
        Z1[~np.isfinite(Z1)] = np.nan
        prep._vol1, prep._Z1 = vol1, np.clip(Z1, -15, 15)
    return prep._vol1, prep._Z1


def fhs_sample(prep, p, avail, w, window=1000, assets=False):
    """Filtered-historical scenarios for tomorrow's P&L of weights `w` (over
    `avail`), using standardised moves up to and including day p, rescaled by
    the 1-day forecast made at day p. Returns the P&L vector (and the asset
    scenario matrix if assets=True). Assets with no history on a scenario day
    contribute zero that day."""
    vol1, Z1 = daily_inputs(prep)
    lo = max(0, p - window + 1)
    Zw = np.nan_to_num(Z1[lo:p + 1][:, avail], nan=0.0)
    scale = np.nan_to_num(vol1[p, avail], nan=0.0) / np.sqrt(ANN)
    X = Zw * scale                                   # scenario returns per asset
    pnl = X @ np.asarray(w, dtype=float)
    return (pnl, X) if assets else pnl


def var_es(pnl, level=0.99, es_level=ES_LEVEL):
    """VaR at `level` and ES at `es_level` of a P&L sample, as positive losses."""
    pnl = np.asarray(pnl, dtype=float)
    var = -float(np.quantile(pnl, 1 - level))
    q = np.quantile(pnl, 1 - es_level)
    es = -float(pnl[pnl <= q].mean())
    return var, es


def normal_var(sigma_daily, level):
    return float(stats.norm.ppf(level) * sigma_daily)


def normal_es(sigma_daily, level=ES_LEVEL):
    a = 1 - level
    return float(sigma_daily * stats.norm.pdf(stats.norm.ppf(a)) / a)


# ------------------------------------------------------------- today's numbers

def today_table(prep, cfg, p, avail, w, Sigma, drift_ann=0.0, n_paths=None, rng=None):
    """VaR/ES of weights `w` at day p by all three methods."""
    from .montecarlo import simulate_holdings
    rng = rng or np.random.default_rng(cfg["seed"] + 5)
    w = np.asarray(w, dtype=float)
    out = {}
    sd = float(np.sqrt(max(w @ Sigma @ w, 0.0))) / np.sqrt(ANN)
    out["normal"] = {"var95": normal_var(sd, 0.95), "var99": normal_var(sd, 0.99), "es975": normal_es(sd),
                     "var99_10d": normal_var(sd * np.sqrt(10), 0.99), "es975_10d": normal_es(sd * np.sqrt(10))}
    pnl = fhs_sample(prep, p, avail, w)
    v95, _ = var_es(pnl, 0.95)
    v99, es = var_es(pnl, 0.99)
    out["fhs"] = {"var95": v95, "var99": v99, "es975": es, "n": len(pnl)}
    n_paths = n_paths or cfg["mc_paths"]
    s1 = simulate_holdings(prep, p, avail, w, cfg, 1, n_paths, rng, drift_ann)["total"]
    s10 = simulate_holdings(prep, p, avail, w, cfg, 10, n_paths, rng, drift_ann)["total"]
    a, b = var_es(s1, 0.95), var_es(s1, 0.99)
    c = var_es(s10, 0.99)
    out["mc"] = {"var95": a[0], "var99": b[0], "es975": b[1], "var99_10d": c[0], "es975_10d": c[1]}
    return out


def component_es(prep, p, avail, w, es_level=ES_LEVEL):
    """Euler allocation of filtered-historical ES to each asset:
    CES_i = -E[w_i x_i | portfolio P&L in its worst (1-es_level) tail].
    The components sum exactly to the ES of the scenario set."""
    w = np.asarray(w, dtype=float)
    pnl, X = fhs_sample(prep, p, avail, w, assets=True)
    q = np.quantile(pnl, 1 - es_level)
    tail = pnl <= q
    comp = -(X[tail] * w).mean(axis=0)
    return comp, float(-pnl[tail].mean())


def component_var_normal(w, Sigma, level=0.99):
    """Euler allocation of normal VaR: z * w_i (Sigma w)_i / sigma_p (daily)."""
    w = np.asarray(w, dtype=float)
    sp = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    if sp <= 0:
        return np.zeros_like(w)
    return stats.norm.ppf(level) * w * (Sigma @ w) / sp / np.sqrt(ANN)


def pod_component_es(prep, p, avail, w, U, es_level=ES_LEVEL):
    """ES split across pods: project w onto the pod vectors (w = c U + e) and
    allocate the tail loss to each pod's piece and to the residual e."""
    w = np.asarray(w, dtype=float)
    pnl, X = fhs_sample(prep, p, avail, w, assets=True)
    q = np.quantile(pnl, 1 - es_level)
    tail = pnl <= q
    c, *_ = np.linalg.lstsq(U.T, w, rcond=None)
    e = w - c @ U
    pods = -(X[tail] @ U.T).mean(axis=0) * c
    resid = -float((X[tail] @ e).mean())
    return pods, resid


# ------------------------------------------------------------- tests

def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(breaches, p):
    """Kupiec POF likelihood-ratio test. Returns (LR, p-value)."""
    b = np.asarray(breaches, dtype=bool)
    T, x = len(b), int(b.sum())
    if T == 0:
        return float("nan"), float("nan")
    ph = x / T
    ll0 = _xlogy(T - x, 1 - p) + _xlogy(x, p)
    ll1 = _xlogy(T - x, 1 - ph) + _xlogy(x, ph)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(breaches):
    """Christoffersen independence test (first-order Markov). Returns (LR, p-value)."""
    b = np.asarray(breaches, dtype=int)
    if len(b) < 3:
        return float("nan"), float("nan")
    prev, cur = b[:-1], b[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    if n01 + n11 == 0:
        return 0.0, 1.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def traffic_light(breaches_99_last250):
    n = int(np.sum(breaches_99_last250))
    return n, ("green" if n <= 4 else "yellow" if n <= 9 else "red")


def _summ(breach, level):
    lr_uc, p_uc = kupiec(breach, 1 - level)
    lr_ind, p_ind = christoffersen(breach)
    lr_cc = lr_uc + lr_ind
    return {"n": int(len(breach)), "breaches": int(breach.sum()), "expected": float(len(breach) * (1 - level)),
            "rate": float(breach.mean()) if len(breach) else float("nan"),
            "kupiec_p": p_uc, "christoffersen_p": p_ind, "cc_p": float(stats.chi2.sf(lr_cc, 2))}


def backtest(prep, cfg, W, pnl, start, window=1000, min_window=250, n_draws=400, gamma=0.01):
    """Daily 1-day VaR/ES backtest.

    W: (T x N) weights held on each day; pnl: realised P&L of those weights on
    each day (market P&L, before costs). The forecast for day d uses data up
    to the close of day d-1 only.

    Methods compared at 95% and 99%:
      'normal'  - bell curve with the same volatility as the scenario set
                  (isolates the effect of fat tails),
      'fhs'     - filtered historical simulation,
      'fhs_aci' - the same, with the quantile level corrected after every
                  breach (adaptive conformal inference)."""
    rng = np.random.default_rng(cfg["seed"] + 9)
    T = len(pnl)
    zs = {lv: stats.norm.ppf(lv) for lv in LEVELS}
    aci = {lv: 1 - lv for lv in LEVELS}
    rec = {m: {lv: [] for lv in LEVELS} for m in ("normal", "fhs", "fhs_aci")}
    dates, real, es_f, var975, draws, var99_series = [], [], [], [], [], []
    for d in range(max(start, min_window + 1), T):
        w = W[d]
        avail = np.flatnonzero(np.abs(w) > 1e-10)
        if len(avail) == 0:
            continue
        sample = fhs_sample(prep, d - 1, avail, w[avail], window)
        if len(sample) < min_window:
            continue
        x = float(pnl[d])
        sd = sample.std()
        for lv in LEVELS:
            v_f = -np.quantile(sample, 1 - lv)
            v_n = zs[lv] * sd
            v_a = -np.quantile(sample, float(np.clip(aci[lv], 1e-4, 0.5)))
            rec["fhs"][lv].append(x < -v_f)
            rec["normal"][lv].append(x < -v_n)
            miss = x < -v_a
            rec["fhs_aci"][lv].append(miss)
            aci[lv] += gamma * ((1 - lv) - float(miss))
            if lv == 0.99:
                var99_series.append(v_f)
        q = np.quantile(sample, 1 - ES_LEVEL)
        es_f.append(-sample[sample <= q].mean())
        var975.append(-q)
        draws.append(rng.choice(sample, n_draws))
        real.append(x)
        dates.append(prep.dates[d])
    if not real:
        return {"n": 0}
    real, es_f, var975 = np.array(real), np.array(es_f), np.array(var975)
    out = {"n": len(real), "start": dates[0], "end": dates[-1], "methods": {}}
    for m, by in rec.items():
        out["methods"][m] = {lv: _summ(np.array(v, dtype=bool), lv) for lv, v in by.items()}
        b99 = np.array(by[0.99], dtype=bool)[-250:]
        out["methods"][m]["traffic"] = traffic_light(b99)
    # Acerbi-Szekely Z2 for ES at 97.5%, Monte Carlo p-value under the model
    a = 1 - ES_LEVEL
    Tn = len(real)
    z2 = float(np.sum(real * (real < -var975) / (Tn * a * es_f)) + 1.0)
    D = np.array(draws)
    z2_sim = np.sum(D * (D < -var975[:, None]) / (Tn * a * es_f[:, None]), axis=0) + 1.0
    tail = real < -var975
    out["es_test"] = {"z2": z2, "p_value": float(np.mean(z2_sim <= z2)),
                      "avg_es": float(es_f[tail].mean()) if tail.any() else float("nan"),
                      "avg_loss_beyond_var975": float(-real[real < -var975].mean()) if (real < -var975).any() else float("nan"),
                      "n_tail": int((real < -var975).sum())}
    out["series"] = {"dates": dates, "pnl": real, "var99": np.array(var99_series)}
    return out


def engine_var(res, cfg):
    """Everything the engine report shows about VaR."""
    prep, f = res.prep, res.final
    p = prep.T - 1
    avail, w, Sigma = f["avail"], f["W"][f["avail"]], f["Sigma"]
    drift = f["sr_port"] * f["ex_ante_vol"]
    today = today_table(prep, cfg, p, avail, w, Sigma, drift)
    comp, es_tot = component_es(prep, p, avail, w)
    cvar_n = component_var_normal(w, Sigma)
    pods, resid = pod_component_es(prep, p, avail, w, f["U"][:, avail])
    ex = np.nan_to_num(prep.R) - prep.rf[:, None]
    pnl = (res.held * ex).sum(axis=1)                  # market P&L of what was actually held
    bt = backtest(prep, cfg, res.held, pnl, res.start)
    return {"today": today, "comp_es": comp, "es_total": es_tot, "comp_var_normal": cvar_n,
            "tickers": [prep.tickers[a] for a in avail], "pod_es": pods, "pod_resid": resid,
            "pod_names": res.pod_names, "backtest": bt}
