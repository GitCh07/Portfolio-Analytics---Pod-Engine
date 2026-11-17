"""Capital allocation across pods, position sizing and risk limits.

The pieces, and where each idea comes from:

* risk budgeting  - every pod contributes a chosen share of portfolio risk
                    (Bridgewater's risk balancing, applied to strategies).
* shrunk Sharpe   - a pod's share tilts toward its *out-of-sample* record,
                    shrunk hard toward a sceptical prior (Bayesian update).
* Kelly cap       - the volatility target can never exceed a fraction of the
                    growth-optimal level implied by the evidence (Thorp).
* drawdown stops  - a pod that loses too much is halved, then switched off,
                    mechanically (the multi-manager "pod" rule).
* golden-section  - the 1-D optimiser used wherever a single number is tuned.
"""
import numpy as np

ANN = 252
PHI = (np.sqrt(5.0) - 1.0) / 2.0            # 0.618..., the golden ratio conjugate


def golden_section_max(f, lo, hi, tol=1e-4, max_iter=200):
    """Maximise a unimodal function on [lo, hi]. Each step shrinks the bracket
    by the golden ratio and re-uses one earlier evaluation - provably the most
    efficient way to do this with function values alone (Kiefer 1953)."""
    a, b = float(lo), float(hi)
    c, d = b - PHI * (b - a), a + PHI * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - PHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + PHI * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)


def risk_budget_weights(Sigma, budgets, iters=500, tol=1e-10):
    """Long-only weights whose risk contributions are proportional to `budgets`.

    Solves  min 1/2 x'Sx - sum_i b_i ln x_i  by cyclical coordinate descent
    (Griveau-Billion, Richard & Roncalli 2013). Equal budgets give the
    equal-risk-contribution ("risk parity") portfolio. Returned weights sum to 1;
    zero-budget items get zero weight."""
    b = np.asarray(budgets, dtype=float)
    n = len(b)
    out = np.zeros(n)
    act = np.flatnonzero(b > 0)
    if len(act) == 0:
        return out
    S = np.asarray(Sigma, dtype=float)[np.ix_(act, act)]
    ba = b[act] / b[act].sum()
    d = np.clip(np.diag(S), 1e-16, None)
    x = ba / np.sqrt(d)
    for _ in range(iters):
        x_old = x.copy()
        for i in range(len(act)):
            c = S[i] @ x - S[i, i] * x[i]
            x[i] = (-c + np.sqrt(c * c + 4.0 * d[i] * ba[i])) / (2.0 * d[i])
        if np.max(np.abs(x - x_old)) < tol * np.max(np.abs(x)):
            break
    out[act] = x / x.sum()
    return out


def posterior_sharpe(n_days, mean_d, std_d, prior_mean, prior_sd):
    """Bayesian shrinkage of an annualised Sharpe ratio.

    Lo (2002): the per-period Sharpe estimated from n observations has variance
    (1 + SR_p^2 / 2) / n. Annualised, that is (1 + SR^2 / (2 * 252)) / years,
    so a short record barely moves the prior."""
    if n_days < 60 or not np.isfinite(std_d) or std_d <= 0:
        return prior_mean, prior_sd
    years = n_days / ANN
    sr = mean_d / std_d * np.sqrt(ANN)
    obs_var = (1.0 + sr * sr / (2.0 * ANN)) / years
    prec = 1.0 / prior_sd ** 2 + 1.0 / obs_var
    return (prior_mean / prior_sd ** 2 + sr / obs_var) / prec, np.sqrt(1.0 / prec)


class DrawdownGovernor:
    """Three-state risk switch driven by drawdown from a rolling high-water mark:
    full (1.0) -> halved (0.5) at `soft` -> off (0.0) at `hard`, with hysteresis
    on the way back so it does not flicker."""

    def __init__(self, soft, hard):
        self.soft, self.hard = soft, hard
        self.state = 1.0

    def update(self, drawdown):
        dd = max(float(drawdown), 0.0)
        if dd >= self.hard:
            self.state = 0.0
        elif self.state == 0.0:
            if dd < self.soft:
                self.state = 0.5
        elif self.state == 0.5:
            if dd < 0.5 * self.soft:
                self.state = 1.0
        elif dd >= self.soft:
            self.state = 0.5
        return self.state


def turbulence_scalar(pct, start=0.75, floor=0.6):
    if pct is None or not np.isfinite(pct) or pct <= start:
        return 1.0
    return float(1.0 - (1.0 - floor) * (pct - start) / (1.0 - start))


def apply_limits(w, Sigma, target_vol, cfg, z_hist=None, vol_now=None):
    """Scale raw weights to the volatility target, then enforce the account
    limits. Returns (weights, info) where info says which limit was binding."""
    info = {"binding": []}
    w = np.array(w, dtype=float)
    if not cfg["allow_short"]:
        w = np.maximum(w, 0.0)
    sig = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    if sig <= 1e-12 or target_vol <= 0:
        return np.zeros_like(w), {"binding": ["no risk"], "ex_ante_vol": 0.0}
    w = w * target_vol / sig
    cap = cfg["asset_cap"]
    if np.any(np.abs(w) > cap):
        w = np.clip(w, -cap, cap)
        info["binding"].append("per-asset cap")
    gross = np.abs(w).sum()
    if gross > cfg["gross_cap"]:
        w = w * cfg["gross_cap"] / gross
        info["binding"].append("gross leverage cap")
    if z_hist is not None and vol_now is not None and len(z_hist) >= 250:
        pnl = np.nan_to_num(z_hist, nan=0.0) @ (w * vol_now / np.sqrt(ANN))
        q = np.quantile(pnl, 0.025)
        tail = pnl[pnl <= q]
        es = -tail.mean() if len(tail) else 0.0
        limit = cfg["es_cap_mult"] * cfg["vol_target"] / np.sqrt(ANN)
        info["es_1d"] = float(es)
        if es > limit > 0:
            w = w * limit / es
            info["binding"].append("tail-loss cap")
    info["ex_ante_vol"] = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    return w, info
