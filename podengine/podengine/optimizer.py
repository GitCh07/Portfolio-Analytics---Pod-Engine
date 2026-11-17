"""Optimise a portfolio you already hold.

Inputs: your positions (shares, dollar values or weights), optional views,
and the account profile. Uses the engine's risk model (1-month volatility
forecasts, cleaned and regime-blended correlations) and its Monte Carlo.

Three alternatives are proposed next to the current portfolio:

1. Minimum variance  - lowest-risk mix of the assets you hold, at the same
                       net exposure. Needs no return forecasts at all.
2. Risk parity       - every holding contributes the same share of risk
                       (long only), scaled to the target volatility.
3. Black-Litterman   - maximises expected return at the target volatility,
                       net of trading costs. Expected returns start from
                       those *implied by your current portfolio* (reverse
                       optimisation, with a sceptical Sharpe ratio), so with
                       no views the answer is your current portfolio; views
                       move it only as far as their stated confidence allows.

Why not plain mean-variance? With estimated expected returns it is famously
unstable ("error maximisation", Michaud 1989). Black-Litterman fixes that by
anchoring to a neutral prior and blending in views with explicit uncertainty
(Black & Litterman 1992; He & Litterman 1999).
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import var as VR
from .allocator import risk_budget_weights
from .cov import risk_contributions
from .montecarlo import simulate_holdings
from .universe import CLASS_PRIOR_SHARPE

ANN = 252
EPS = 1e-9


# ------------------------------------------------------------------ inputs

def read_positions(path):
    """CSV with a `ticker` column and one of `shares`, `value` or `weight`.
    A row with ticker CASH holds cash in dollars (`value` or `shares` column).
    Negative numbers are short positions. Returns (DataFrame, kind)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    for kind in ("shares", "value", "weight"):
        if kind in df.columns:
            df[kind] = pd.to_numeric(df[kind], errors="coerce").fillna(0.0)
            return df[["ticker", kind]].groupby("ticker", as_index=False).sum(), kind
    if df.shape[1] >= 2:
        df = df.rename(columns={df.columns[1]: "shares"})
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
        return df[["ticker", "shares"]].groupby("ticker", as_index=False).sum(), "shares"
    raise ValueError("positions file needs a ticker column and a shares, value or weight column")


def positions_to_weights(pos, kind, prices, capital=None):
    """Turn positions into weights of account equity.
    Returns (weights {ticker: w}, equity, cash weight, notes)."""
    notes = []
    cash = float(pos.loc[pos["ticker"] == "CASH", kind].sum()) if (pos["ticker"] == "CASH").any() else 0.0
    rows = pos[pos["ticker"] != "CASH"]
    if kind == "weight":
        w = dict(zip(rows["ticker"], rows["weight"].astype(float)))
        equity = float(capital or 100000.0)
        cash_w = 1.0 - sum(w.values()) if cash == 0 else cash
        return w, equity, cash_w, notes
    values = {}
    for t, x in zip(rows["ticker"], rows[kind].astype(float)):
        if kind == "shares":
            if t not in prices:
                notes.append(f"{t}: no price found; left out")
                continue
            values[t] = x * prices[t]
        else:
            values[t] = x
    equity = sum(values.values()) + cash
    if capital:
        equity = float(capital)
        notes.append(f"Equity set to {capital:,.0f} by --capital (positions + cash = {sum(values.values()) + cash:,.0f})")
    if equity <= 0:
        raise ValueError("account equity (positions + cash) must be positive; add a CASH row or use --capital")
    w = {t: v / equity for t, v in values.items()}
    return w, equity, 1.0 - sum(w.values()), notes


def read_views(path):
    """JSON list of views. Each view:
       {"assets": {"QQQ": 1, "TLT": -1}, "return": 0.03, "confidence": 0.5, "label": "..."}
    "assets" are the view's weights (one asset = absolute view, +1/-1 = relative),
    "return" is the expected annual excess return of that combination,
    "confidence" in (0, 1): 0.5 = as much weight as the prior, 0.9 = strong."""
    with open(path) as f:
        views = json.load(f)
    if isinstance(views, dict):
        views = views.get("views", [])
    return views


# ------------------------------------------------------------------ Black-Litterman

def implied_returns(Sigma, w0, prior_sharpe=0.3, vol_ann=None, cls=None):
    """Reverse optimisation: expected excess returns that make w0 optimal,
    Pi = delta * Sigma * w0, with delta set so w0 has the given Sharpe ratio.
    Falls back to a per-asset prior (Sharpe x volatility) if w0 has no risk."""
    s0 = float(np.sqrt(max(w0 @ Sigma @ w0, 0.0)))
    if s0 > 1e-6 and abs(w0).sum() > 1e-6:
        delta = prior_sharpe / s0
        return delta * Sigma @ w0, "implied by your current portfolio"
    vol = np.sqrt(np.diag(Sigma)) if vol_ann is None else vol_ann
    sr = np.array([CLASS_PRIOR_SHARPE.get(c, 0.0) for c in cls]) if cls is not None else np.full(len(vol), prior_sharpe)
    return sr * vol, "asset-class prior (current portfolio has no risk)"


def build_views(views, tickers, Sigma, tau):
    """P (k x n), Q (k), Omega (k x k) from the view list. Omega uses the
    confidence parametrisation omega_k = (1/c - 1) * p_k (tau Sigma) p_k',
    so c = 0.5 weighs the view equally with the prior (a common heuristic,
    not a law). Views naming assets not in the portfolio are skipped."""
    idx = {t: i for i, t in enumerate(tickers)}
    P, Q, om, used, skipped = [], [], [], [], []
    for v in views:
        a = {str(k).upper(): float(x) for k, x in v.get("assets", {}).items()}
        missing = [k for k in a if k not in idx]
        if missing or not a:
            skipped.append((v.get("label", str(a)), f"not in portfolio: {missing}"))
            continue
        row = np.zeros(len(tickers))
        for k, x in a.items():
            row[idx[k]] = x
        c = float(np.clip(v.get("confidence", 0.5), 0.01, 0.99))
        P.append(row)
        Q.append(float(v["return"]))
        om.append((1.0 / c - 1.0) * float(row @ (tau * Sigma) @ row))
        used.append({"label": v.get("label") or " ".join(f"{x:+g} {k}" for k, x in a.items()),
                     "view": float(v["return"]), "confidence": c, "row": row})
    if not P:
        return None, None, None, used, skipped
    return np.array(P), np.array(Q), np.diag(np.maximum(om, 1e-12)), used, skipped


def black_litterman(Pi, Sigma, P, Q, Omega, tau=0.05):
    """Posterior mean and covariance of expected returns (He & Litterman 1999)."""
    if P is None:
        return Pi.copy(), tau * Sigma
    tS_inv = np.linalg.inv(tau * Sigma)
    Om_inv = np.linalg.inv(Omega)
    M = np.linalg.inv(tS_inv + P.T @ Om_inv @ P)
    mu = M @ (tS_inv @ Pi + P.T @ Om_inv @ Q)
    return mu, M


# ------------------------------------------------------------------ solvers

def _sabs(x):
    return np.sqrt(x * x + EPS)


def _bounds(n, cfg, allow_short=None):
    cap = cfg["asset_cap"]
    short = cfg["allow_short"] if allow_short is None else allow_short
    return [(-cap if short else 0.0, cap)] * n


def solve_bl(mu, Sigma, w0, target_vol, cost, cfg, max_turnover=None):
    """max mu'w - sum cost_i |w_i - w0_i|  s.t.  w'Sigma w <= target^2,
    gross <= cap, per-asset bounds, long only if the profile says so."""
    n = len(mu)
    G = cfg["gross_cap"]
    cons = [{"type": "ineq", "fun": lambda w: target_vol ** 2 - w @ Sigma @ w,
             "jac": lambda w: -2 * Sigma @ w},
            {"type": "ineq", "fun": lambda w: G - _sabs(w).sum(),
             "jac": lambda w: -w / _sabs(w)}]
    if max_turnover is not None:
        cons.append({"type": "ineq", "fun": lambda w: max_turnover - _sabs(w - w0).sum(),
                     "jac": lambda w: -(w - w0) / _sabs(w - w0)})

    def f(w):
        return -(mu @ w) + cost @ _sabs(w - w0)

    def g(w):
        return -mu + cost * (w - w0) / _sabs(w - w0)

    bnds = _bounds(n, cfg)
    x0 = np.clip(w0, [b[0] for b in bnds], [b[1] for b in bnds])
    s = float(np.sqrt(max(x0 @ Sigma @ x0, 0.0)))
    if s > target_vol and s > 0:
        x0 = x0 * target_vol / s * 0.999
    if _sabs(x0).sum() > G:
        x0 = x0 * G / _sabs(x0).sum() * 0.999
    r = minimize(f, x0, jac=g, bounds=bnds, constraints=cons, method="SLSQP",
                 options={"maxiter": 1000, "ftol": 1e-12})
    return r.x, r


def solve_min_var(Sigma, net, cfg):
    """min w'Sigma w  s.t.  sum(w) = net, gross <= cap, bounds."""
    n = Sigma.shape[0]
    G = cfg["gross_cap"]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - net, "jac": lambda w: np.ones(n)},
            {"type": "ineq", "fun": lambda w: G - _sabs(w).sum(), "jac": lambda w: -w / _sabs(w)}]
    bnds = _bounds(n, cfg)
    x0 = np.full(n, net / n)
    r = minimize(lambda w: w @ Sigma @ w, x0, jac=lambda w: 2 * Sigma @ w, bounds=bnds,
                 constraints=cons, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14})
    return r.x, r


def solve_risk_parity(Sigma, target_vol, cfg):
    """Long-only equal risk contributions, scaled to the target volatility,
    then cut back if the gross or per-asset cap would be breached."""
    x = risk_budget_weights(Sigma, np.ones(Sigma.shape[0]))
    s = float(np.sqrt(x @ Sigma @ x))
    w = x * target_vol / s
    note = ""
    k = min(1.0, cfg["gross_cap"] / w.sum(), cfg["asset_cap"] / w.max())
    if k < 1.0:
        w = w * k
        note = f"scaled to {k:.0%} of target to respect the account caps"
    return w, note


# ------------------------------------------------------------------ evaluation

def evaluate(name, w, w0, mu, Sigma, prep, cfg, p, avail, cost, rng, note=""):
    """Risk and return summary of one candidate. `cost` = one-way trading cost
    per unit of weight traded (a one-off charge, not annualised)."""
    w = np.asarray(w, dtype=float)
    vol = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    exp = float(mu @ w)
    rc, port = risk_contributions(w, Sigma)
    share = rc / port if port > 0 else np.zeros_like(w)
    pnl = VR.fhs_sample(prep, p, avail, w)
    v99, es = VR.var_es(pnl, 0.99)
    sim = simulate_holdings(prep, p, avail, w, cfg, 21, cfg["mc_paths"], rng, exp)["total"]
    asset_vol = np.sqrt(np.diag(Sigma))
    return {"name": name, "w": w, "exp": exp, "vol": vol, "sharpe": exp / vol if vol > 0 else float("nan"),
            "var99": v99, "es975": es, "range90": (float(np.quantile(sim, 0.05)), float(np.quantile(sim, 0.95))),
            "p_loss_1m": float((sim < 0).mean()), "risk_share": share,
            "max_share": float(share.max()) if len(share) else 0.0,
            # diversification ratio (Choueifaty & Coignard 2008): weighted average asset volatility
            # over portfolio volatility; 1 = no diversification benefit, higher = more
            "div_ratio": float(np.abs(w) @ asset_vol / vol) if vol > 0 else float("nan"),
            "gross": float(np.abs(w).sum()), "net": float(w.sum()),
            "turnover": float(np.abs(w - w0).sum()), "cost": float(cost @ np.abs(w - w0)), "note": note}


def optimize_holdings(md, cfg, prep, w_now, equity, views=None, target_vol=None, max_turnover=None,
                      tau=0.05, prior_sharpe=0.30, horizon_years=1.0, extra=()):
    """Run the optimiser for weights `w_now` ({ticker: weight of equity})."""
    rng = np.random.default_rng(cfg["seed"] + 21)
    p = prep.T - 1
    tick = prep.tickers
    want = [t for t in w_now if t in tick] + [t for t in extra if t in tick and t not in w_now]
    notes = []
    missing = [t for t in w_now if t not in tick]
    if missing:
        notes.append(f"No price history for {', '.join(missing)}: held as is and left out of the risk numbers.")
    elig = [t for t in want if prep.elig[p, tick.index(t)]]
    frozen = [t for t in want if t not in elig]
    if frozen:
        notes.append(f"Too little history (under {cfg['min_history']} days) for {', '.join(frozen)}: "
                     "kept at current weight and left out of the optimisation.")
    if len(elig) < 2:
        raise RuntimeError("Need at least two holdings with enough price history to optimise.")
    avail = np.array([tick.index(t) for t in elig])
    w0 = np.array([w_now.get(t, 0.0) for t in elig])
    Sigma = prep.sigma_at(p, avail, cfg)
    cls = [prep.meta_list[a]["cls"] for a in avail]
    trade_cost = prep.cost_rate[avail] * cfg["cost_mult"]          # one-way, per unit of weight
    cost = trade_cost / horizon_years                                # amortised into the annual objective

    if not cfg["allow_short"] and (w0 < 0).any():
        notes.append("You hold short positions but the profile is long only; proposals will not short.")
    Pi, prior_src = implied_returns(Sigma, w0, prior_sharpe, np.sqrt(np.diag(Sigma)), cls)
    P = Q = Om = None
    used, skipped = [], []
    if views:
        P, Q, Om, used, skipped = build_views(views, elig, Sigma, tau)
    mu, M = black_litterman(Pi, Sigma, P, Q, Om, tau)
    for u in used:
        u["prior"] = float(u["row"] @ Pi)
        u["posterior"] = float(u["row"] @ mu)

    s0 = float(np.sqrt(max(w0 @ Sigma @ w0, 0.0)))
    tv = float(target_vol) if target_vol else (s0 if s0 > 1e-4 else cfg["vol_target"])
    out = [evaluate("Current", w0, w0, mu, Sigma, prep, cfg, p, avail, trade_cost, rng)]

    net0 = float(w0.sum())
    if abs(net0) >= 0.05:
        wm, r = solve_min_var(Sigma, net0, cfg)
        out.append(evaluate("Minimum variance", wm, w0, mu, Sigma, prep, cfg, p, avail, trade_cost, rng,
                            f"same net exposure ({net0:.0%})" + ("" if r.success else f"; solver: {r.message}")))
    else:
        notes.append("Minimum variance skipped: current net exposure is near zero.")
    wr, nr = solve_risk_parity(Sigma, tv, cfg)
    out.append(evaluate("Risk parity", wr, w0, mu, Sigma, prep, cfg, p, avail, trade_cost, rng,
                        "long only" + (f"; {nr}" if nr else "")))
    wb, r = solve_bl(mu, Sigma, w0, tv, cost, cfg, max_turnover)
    out.append(evaluate("Black-Litterman", wb, w0, mu, Sigma, prep, cfg, p, avail, trade_cost, rng,
                        ("with your views" if used else "no views given, so it stays close to current")
                        + ("" if r.success else f"; solver: {r.message}")))

    # how the risk model would have done on these weights, held constant over history
    W = np.zeros((prep.T, prep.N))
    W[:, avail] = w0
    ex = np.nan_to_num(prep.R) - prep.rf[:, None]
    start = int(max(prep.first_valid[avail].max() + 260, cfg["min_history"]))
    bt = VR.backtest(prep, cfg, W, (W * ex).sum(axis=1), start) if np.abs(w0).sum() > 0 else {"n": 0}

    today = VR.today_table(prep, cfg, p, avail, w0, Sigma, float(mu @ w0)) if s0 > 0 else None
    comp, es_tot = VR.component_es(prep, p, avail, w0) if s0 > 0 else (np.zeros(len(w0)), 0.0)
    return {"tickers": elig, "avail": avail, "w0": w0, "Sigma": Sigma, "Pi": Pi, "mu": mu, "prior_src": prior_src,
            "views": used, "skipped_views": skipped, "target_vol": tv, "tau": tau, "prior_sharpe": prior_sharpe,
            "portfolios": out, "notes": notes, "frozen": frozen, "missing": missing, "equity": equity,
            "var_today": today, "comp_es": comp, "es_total": es_tot, "backtest": bt,
            "names": [prep.meta_list[a]["name"] for a in avail], "cls": cls,
            "prices": {t: float(md.close[t].dropna().iloc[-1]) for t in elig}, "asof": prep.dates[-1]}
