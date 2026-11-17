"""Regime-switching filtered historical simulation, with self-correcting ranges.

How a path is built (no bell curves anywhere):

1. Tomorrow's regime is drawn from today's filtered regime probabilities and
   the fitted transition matrix; it then evolves as a Markov chain.
2. Each simulated day re-plays the *joint* standardised move of all assets from
   a real historical day, drawn from days that belonged to the simulated
   regime. Cross-asset dependence, fat tails and "everything falls together"
   days are therefore whatever history contained, not a modelling assumption.
   Days are drawn in blocks so short-term dependence survives.
3. Volatility follows a GARCH-type recursion that starts at today's forecast
   and is pulled toward the simulated regime's long-run level, so calm and
   crisis paths differ in both level and clustering.
4. The historical sample mean is removed and replaced by a deliberately
   sceptical drift, because means are the least reliable thing in the data.

Then the ranges are audited. `calibrate` replays the forecast at every past
date, checks how often reality fell outside the stated range, and adaptive
conformal inference (Gibbs & Candes 2021) nudges the quantile levels after
every miss:  a_{t+1} = a_t + gamma * (target - miss_t).  The long-run miss
rate converges to the target whatever the data do - that is a theorem about
the procedure, not a hope about the model.
"""
import numpy as np

from .allocator import golden_section_max
from .universe import CLASS_PRIOR_SHARPE

ANN = 252


def _fhs(Zh, G, A, p_next, v0, lr, horizon, n_paths, block, alpha, beta, rng, w=None, keep_assets=False):
    """Core simulator.

    Zh (S x n) standardised history, G (S x K) regime weights of each day,
    A (K x K) transition matrix, p_next (K,) regime probabilities for day 1,
    v0 (n,) starting daily variance, lr (K x n) long-run daily variance per
    regime. Returns daily portfolio returns (paths x horizon) if `w` is given
    and/or cumulative asset returns (paths x n)."""
    S, n = Zh.shape
    K = G.shape[1]
    cdfs, scale = [], np.ones((K, n))
    pooled = np.linspace(1.0 / S, 1.0, S)
    for k in range(K):
        tot = G[:, k].sum()
        if tot >= 20:
            cdfs.append(np.cumsum(G[:, k]) / tot)
            # unit variance within the regime, so the regime's level comes only from `lr`
            scale[k] = np.sqrt(np.maximum((G[:, k] @ Zh ** 2) / tot, 1e-6))
        else:
            cdfs.append(pooled)
    Acdf = np.cumsum(A, axis=1)
    state = np.minimum(np.searchsorted(np.cumsum(p_next / p_next.sum()), rng.random(n_paths)), K - 1)
    idx = np.zeros(n_paths, dtype=int)
    v = np.tile(v0, (n_paths, 1))
    omega = (1.0 - alpha - beta) * lr
    port = np.empty((n_paths, horizon)) if w is not None else None
    cum = np.zeros((n_paths, n)) if keep_assets else None
    p_cont = 1.0 - 1.0 / max(block, 1)
    for h in range(horizon):
        if h:
            new = np.minimum((rng.random(n_paths)[:, None] > Acdf[state]).sum(axis=1), K - 1)
            changed = new != state
            state = new
        else:
            changed = np.ones(n_paths, dtype=bool)
        cont = (~changed) & (rng.random(n_paths) < p_cont) & (idx + 1 < S)
        u = rng.random(n_paths)
        fresh = np.empty(n_paths, dtype=int)
        for k in range(K):
            m = state == k
            if m.any():
                fresh[m] = np.minimum(np.searchsorted(cdfs[k], u[m]), S - 1)
        idx = np.where(cont, idx + 1, fresh)
        r = Zh[idx] / scale[state] * np.sqrt(v)
        r = np.clip(r, -0.5, 0.5)
        v = omega[state] + alpha * r * r + beta * v
        if port is not None:
            port[:, h] = r @ w
        if cum is not None:
            cum += np.log1p(r)
    return port, (np.expm1(cum) if cum is not None else None)


def _inputs(prep, p, avail, cfg):
    """History, regime weights and volatility state as known at day p."""
    Z, R = prep.Z, prep.R
    first = prep.first_valid[avail]
    s0 = int(first.max())
    if p - s0 < 250:                                   # not enough joint history: use what exists
        s0 = max(0, p - 500)
    Zh = np.nan_to_num(Z[s0:p + 1][:, avail], nan=0.0)
    Zh = Zh - Zh.mean(axis=0)                          # drop the sample mean (replaced by a prior drift)
    Zh = Zh / np.maximum(Zh.std(axis=0), 1e-9)         # unit variance: volatility comes from the forecast
    Rh = np.nan_to_num(R[s0:p + 1][:, avail], nan=0.0)
    k = prep.fit_at(p)
    if k >= 0:
        par = prep.reg_fits[k]["params"]
        K = len(par["mu"])
        G = prep.reg_fits[k]["hist"][s0:p + 1, :K]
        A = par["A"]
        p_next = prep.reg_np[p, :K] @ A
    else:
        G, A, p_next = np.ones((len(Zh), 1)), np.ones((1, 1)), np.ones(1)
    pooled = (Rh ** 2).mean(axis=0)
    lr = np.empty((G.shape[1], len(avail)))
    for j in range(G.shape[1]):
        tot = G[:, j].sum()
        lr[j] = (G[:, j] @ Rh ** 2) / tot if tot >= 20 else pooled
    lr = np.maximum(lr, 1e-10)
    v0 = prep.vol_np[p, avail] ** 2 / ANN
    return Zh, G, A, p_next, v0, lr


def simulate_holdings(prep, p, avail, w, cfg, horizon=21, n_paths=None, rng=None, drift_ann=0.0,
                      keep_assets=False):
    """Simulate holding weights `w` (over `avail`) for `horizon` days from day p.
    `drift_ann` = expected annual return over cash to centre the paths on.
    Returns dict with 'daily' (paths x horizon, excess of cash) and 'total'."""
    rng = rng or np.random.default_rng(cfg["seed"])
    n_paths = n_paths or cfg["mc_paths"]
    Zh, G, A, p_next, v0, lr = _inputs(prep, p, avail, cfg)
    port, assets = _fhs(Zh, G, A, p_next, v0, lr, horizon, n_paths, cfg["mc_block"],
                        cfg["garch_alpha"], cfg["garch_beta"], rng, w=np.asarray(w, float),
                        keep_assets=keep_assets)
    port = port + drift_ann / ANN                      # sceptical expected return, not the sample mean
    return {"daily": port, "total": np.expm1(np.log1p(np.clip(port, -0.99, None)).sum(axis=1)), "assets": assets}


def simulate_strategy(res, cfg, horizon=252, n_paths=None, rng=None, sharpe=None):
    """Simulate the strategy's own return stream forward (the positions keep
    changing, so this re-plays the engine's realised daily behaviour rather
    than today's holdings). Drift = the shrunk Sharpe estimate, not the
    backtest average."""
    prep = res.prep
    rng = rng or np.random.default_rng(cfg["seed"] + 1)
    n_paths = n_paths or cfg["mc_paths"]
    x = res.excess[res.start:]
    T = len(x)
    lam = 0.5 ** (1 / 42)
    ev = np.empty(T)
    acc = np.nanvar(x[:126]) if T > 126 else np.nanvar(x)
    for t in range(T):                                  # lagged EWMA variance of the strategy
        ev[t] = acc
        acc = lam * acc + (1 - lam) * x[t] ** 2
    z = x / np.sqrt(np.maximum(ev, 1e-12))
    z = z[126:] - z[126:].mean()
    p = prep.T - 1
    k = prep.fit_at(p)
    if k >= 0:
        par = prep.reg_fits[k]["params"]
        K = len(par["mu"])
        G = prep.reg_fits[k]["hist"][res.start + 126:, :K]
        A, p_next = par["A"], prep.reg_np[p, :K] @ par["A"]
    else:
        G, A, p_next = np.ones((len(z), 1)), np.ones((1, 1)), np.ones(1)
    vol_now = max(res.final.get("ex_ante_vol", 0.0), 1e-4)
    v0 = np.array([vol_now ** 2 / ANN])
    tot_var = (z ** 2).mean()
    lr = np.empty((G.shape[1], 1))
    for j in range(G.shape[1]):
        tot = G[:, j].sum()
        lr[j, 0] = v0[0] * ((G[:, j] @ z ** 2) / tot / tot_var if tot >= 20 else 1.0)
    port, _ = _fhs(z[:, None], G, A, p_next, v0, lr, horizon, n_paths, cfg["mc_block"],
                   cfg["garch_alpha"], cfg["garch_beta"], rng, w=np.ones(1))
    sr = res.final.get("sr_port", 0.0) if sharpe is None else sharpe
    daily = port + sr * vol_now / ANN
    wealth = np.exp(np.cumsum(np.log1p(np.clip(daily, -0.99, None)), axis=1))
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0), axis=1)
    return {"daily": daily, "wealth": wealth, "max_dd": (wealth / peak - 1.0).min(axis=1),
            "sharpe_used": sr, "vol_now": vol_now}


def kelly_leverage(daily_excess, rf_daily=0.0, max_lev=12.0):
    """Growth-optimal leverage on a simulated return distribution, found by
    golden-section search on expected log growth. Unlike mu/sigma^2 this
    accounts for fat tails and the simulated chance of ruin."""
    x = np.asarray(daily_excess, dtype=float).ravel()
    worst = x.min()
    hi = min(max_lev, 0.98 / abs(worst)) if worst < 0 else max_lev

    def growth(L):
        return float(np.mean(np.log1p(rf_daily + L * x)))

    L, g = golden_section_max(growth, 0.0, hi)
    return L, g * ANN


class ACI:
    """Adaptive conformal inference for one tail. The internal state is left
    unclipped (that is what the coverage guarantee needs); only the quantile
    level actually used is kept inside (0, 0.5)."""

    def __init__(self, target, gamma):
        self.target, self.gamma, self.state = target, gamma, target

    @property
    def level(self):
        return float(np.clip(self.state, 0.001, 0.499))

    def update(self, missed):
        self.state += self.gamma * (self.target - float(missed))


def calibrate(res, cfg, horizon=21, n_paths=1000, benchmark=False):
    """Replay the Monte Carlo at past dates and score its ranges out of sample.

    At every `horizon`-th rebalance date the engine's then-current portfolio
    (or the benchmark asset) is simulated forward and compared with what
    actually happened. Reports raw and self-corrected coverage of the 90% and
    50% ranges, plus the corrected quantile levels to use today."""
    prep = res.prep
    rng = np.random.default_rng(cfg["seed"] + 2)
    R, rf = prep.R, prep.rf
    gamma = cfg["aci_gamma"]
    targets = {"lo90": 0.05, "hi90": 0.05, "lo50": 0.25, "hi50": 0.25}
    aci = {k: ACI(v, gamma) for k, v in targets.items()}
    raw_miss = {k: [] for k in targets}
    adj_miss = {k: [] for k in targets}
    pit = []
    bench_col = prep.tickers.index(prep.bench)
    bench_sharpe = CLASS_PRIOR_SHARPE.get(prep.meta_list[bench_col]["cls"], 0.0)
    last = -10 ** 9
    for rec in res.rebalances:
        p = rec["p"]
        if p - last < horizon or p + horizon >= prep.T or prep.fit_at(p) < 0:
            continue
        if benchmark:
            if not prep.elig[p, bench_col]:
                continue
            avail, w = np.array([bench_col]), np.ones(1)
            drift = bench_sharpe * prep.vol_np[p, bench_col]
        else:
            avail = np.flatnonzero(rec["W"] != 0)
            if len(avail) == 0:
                continue
            w = rec["W"][avail]
            drift = rec["sr_port"] * rec["ex_ante_vol"]
        last = p
        sim = simulate_holdings(prep, p, avail, w, cfg, horizon, n_paths, rng, drift)["total"]
        path = np.nan_to_num(R[p + 1:p + 1 + horizon][:, avail], nan=0.0) - rf[p + 1:p + 1 + horizon, None]
        real = float(np.expm1(np.log1p(np.clip(path @ w, -0.99, None)).sum()))
        pit.append(float((sim <= real).mean()))
        for key, tgt in targets.items():
            lower = key.startswith("lo")
            q_raw = np.quantile(sim, tgt if lower else 1 - tgt)
            q_adj = np.quantile(sim, aci[key].level if lower else 1 - aci[key].level)
            raw_miss[key].append(real < q_raw if lower else real > q_raw)
            miss = real < q_adj if lower else real > q_adj
            adj_miss[key].append(miss)
            aci[key].update(miss)
    n = len(pit)
    if n == 0:
        return {"n": 0}
    cov = lambda d, a, b: 1.0 - float(np.mean(d[a])) - float(np.mean(d[b]))
    return {"n": n, "horizon": horizon,
            "raw90": cov(raw_miss, "lo90", "hi90"), "adj90": cov(adj_miss, "lo90", "hi90"),
            "raw50": cov(raw_miss, "lo50", "hi50"), "adj50": cov(adj_miss, "lo50", "hi50"),
            "levels": {k: a.level for k, a in aci.items()}, "pit": pit}


def corrected_range(sim_total, levels=None):
    """90% and 50% ranges of a simulated distribution, raw and self-corrected."""
    q = lambda a: float(np.quantile(sim_total, a))
    out = {"raw90": (q(0.05), q(0.95)), "raw50": (q(0.25), q(0.75)), "median": q(0.5)}
    if levels:
        out["adj90"] = (q(levels["lo90"]), q(1 - levels["hi90"]))
        out["adj50"] = (q(levels["lo50"]), q(1 - levels["hi50"]))
    return out
