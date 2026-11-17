"""Walk-forward engine.

One pass through history, day by day. On each rebalance day the engine uses
only what was known at that close to (1) estimate risk, (2) ask every pod for
positions, (3) decide each pod's share of risk from its out-of-sample record,
(4) apply the risk governor and account limits. Trades happen `exec_lag` days
later and pay costs. The same code path produces today's target portfolio, so
the backtest and the live recommendation cannot drift apart.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .allocator import (DrawdownGovernor, apply_limits, posterior_sharpe,
                        risk_budget_weights, turbulence_scalar)
from .cov import clean_corr, to_cov, weighted_corr
from .pods import REGISTRY
from .regime import next_day_stress, turbulence, walk_forward_regimes
from .vol import forecast_vol

ANN = 252


PREP_KEYS = ("corr_halflife", "corr_window", "corr_shrink", "min_history", "vol_horizon", "ewma_halflife",
             "regime_states", "regime_refit_days", "regime_min_days", "regime_asset")


def prep_key(cfg):
    return tuple(str(cfg.get(k)) for k in PREP_KEYS)


class Prep:
    """Everything that depends on the data but not on strategy settings, so a
    grid of settings can share it: volatility forecasts, regimes, turbulence,
    eligibility and cached covariance matrices."""

    def __init__(self, md, cfg, verbose=False):
        self.md, self.base_cfg = md, cfg
        say = print if verbose else (lambda *a, **k: None)
        self.dates = md.close.index
        self.T, self.N = md.close.shape
        self.tickers = md.tickers
        self.meta_list = [md.meta[t] for t in self.tickers]
        self.R = md.returns.to_numpy(dtype=float)
        self.rf = md.rf.to_numpy(dtype=float)

        say("  volatility forecasts ...")
        self.vol, self.vol_diag = forecast_vol(md, cfg["vol_horizon"], cfg["ewma_halflife"])
        self.vol_np = self.vol.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.Z = self.R / (self.vol.shift(1).to_numpy(dtype=float) / np.sqrt(ANN))
        self.Z[~np.isfinite(self.Z)] = np.nan
        self.Z = np.clip(self.Z, -12, 12)

        say("  regime model ...")
        self.bench = cfg.get("regime_asset") or self._pick_bench()
        self.K = cfg["regime_states"]
        self.reg_probs, self.reg_fits = walk_forward_regimes(
            md.returns[self.bench], self.K, cfg["regime_refit_days"], cfg["regime_min_days"])
        self.reg_np = self.reg_probs.to_numpy(dtype=float)
        self._fit_starts = np.array([f["start_pos"] for f in self.reg_fits], dtype=int)

        say("  turbulence index ...")
        classes = sorted({m["cls"] for m in self.meta_list})
        zdf = pd.DataFrame(self.Z, index=self.dates, columns=self.tickers)
        comp = pd.DataFrame({c: zdf[[t for t, m in zip(self.tickers, self.meta_list) if m["cls"] == c]].mean(axis=1)
                             for c in classes})
        self.turb, self.turb_raw = turbulence(comp)
        self.turb_pct = self.turb.to_numpy(dtype=float)

        have = md.close.notna()
        count = have.cumsum()
        recent = have.rolling(5, min_periods=5).sum() == 5
        self.elig = ((count >= cfg["min_history"]) & have & recent & self.vol.notna()
                     & (self.vol > 1e-4)).to_numpy()
        self.cost_rate = np.array([m["cost_bps"] for m in self.meta_list]) / 1e4

        fin = np.isfinite(self.Z)
        self.first_valid = np.where(fin.any(axis=0), fin.argmax(axis=0), self.T)
        self._sigma, self._stress = {}, {}
        self._sig_key = prep_key(cfg)

    def _pick_bench(self):
        if "SPY" in self.tickers:
            return "SPY"
        eq = [t for t, m in zip(self.tickers, self.meta_list) if m["cls"] == "equity"] or self.tickers
        return self.md.close[eq].notna().sum().idxmax()

    def fit_at(self, p):
        """Index of the regime fit in force at day p (or -1 before the first)."""
        return int(np.searchsorted(self._fit_starts, p, side="right") - 1) if len(self._fit_starts) else -1

    def stress_corr(self, k):
        if k not in self._stress:
            f = self.reg_fits[k]
            fp = f["start_pos"]
            w = f["hist"][:fp, 1:].sum(axis=1)
            C = weighted_corr(self.Z[:fp], w) if w.sum() > 30 else np.eye(self.N)
            C = np.where(np.isfinite(C), C, 0.0)
            np.fill_diagonal(C, 1.0)
            self._stress[k] = 0.9 * C + 0.1 * np.eye(self.N)
        return self._stress[k]

    def sigma_at(self, p, avail, cfg):
        """Covariance of `avail` assets using data up to day p."""
        key = (p, np.asarray(avail).tobytes())
        if key in self._sigma:
            return self._sigma[key]
        lo = max(0, p - cfg["corr_window"] + 1)
        C = clean_corr(self.Z[lo:p + 1][:, avail], cfg["corr_halflife"], cfg["corr_shrink"])
        k = self.fit_at(p)
        ps = 0.0
        if k >= 0:
            ps = next_day_stress(self.reg_np[p], self.reg_fits[k]["params"], self.K)
            C = (1 - ps) * C + ps * self.stress_corr(k)[np.ix_(avail, avail)]
        S = to_cov(C, self.vol_np[p, avail])
        self._sigma[key] = S
        return S

    def rebalance_flags(self, freq):
        d = self.dates
        if freq == "monthly":
            key = d.year * 12 + d.month
        else:
            iso = d.isocalendar()
            key = iso.year.to_numpy() * 53 + iso.week.to_numpy()
        key = np.asarray(key)
        flag = np.zeros(self.T, dtype=bool)
        flag[:-1] = key[1:] != key[:-1]
        flag[-1] = True                       # always produce a target for the latest date
        return flag


@dataclass
class Result:
    dates: pd.DatetimeIndex
    tickers: list
    pod_names: list
    cfg: dict
    ret: np.ndarray                 # net daily portfolio return (including cash yield)
    excess: np.ndarray              # net daily return over the risk-free rate
    cost: np.ndarray
    financing: np.ndarray
    pod_ret: np.ndarray             # T x P paper returns of each pod at its own vol target (net of its own costs)
    contrib: np.ndarray             # T x P contribution of each pod to portfolio return (before costs)
    resid: np.ndarray               # T: return from the gap between holdings and the pod mix (limits, band, drift)
    held: np.ndarray                # T x N weights actually held each day
    gross: np.ndarray
    rebalances: list = field(default_factory=list)
    start: int = 0
    final: dict = field(default_factory=dict)

    def series(self, arr):
        return pd.Series(arr[self.start:], index=self.dates[self.start:])

    @property
    def returns(self):
        return self.series(self.ret)

    @property
    def excess_returns(self):
        return self.series(self.excess)

    def pod_frame(self):
        return pd.DataFrame(self.pod_ret[self.start:], index=self.dates[self.start:], columns=self.pod_names)

    def contrib_frame(self):
        df = pd.DataFrame(self.contrib[self.start:], index=self.dates[self.start:], columns=self.pod_names)
        df["limits & band"] = self.resid[self.start:]
        return df


def _trade_to_band(held, target, buffer):
    """No-trade band: leave a position alone while it is within `buffer` (as a
    fraction of the target size) of its target, otherwise trade only to the
    edge of the band. With proportional costs this is the cost-efficient rule;
    it removes the churn from small week-to-week changes in the target."""
    if buffer <= 0:
        return target
    band = buffer * np.abs(target)
    gap = target - held
    return held + np.sign(gap) * np.maximum(np.abs(gap) - band, 0.0)


class Engine:
    def __init__(self, md, cfg, prep=None, verbose=False):
        self.md, self.cfg = md, cfg
        self.prep = prep or Prep(md, cfg, verbose)
        pr = self.prep
        if prep_key(cfg) != pr._sig_key:
            raise ValueError("risk-model settings differ from the shared Prep; build a new Prep")
        # expose what pods need
        self.vol, self.meta_list, self.turb_pct = pr.vol, pr.meta_list, pr.turb_pct
        self.pods = [REGISTRY[n]() for n in cfg["pods"]]
        for pod in self.pods:
            pod.prepare(self)

    def run(self):
        pr, cfg = self.prep, self.cfg
        T, N, P = pr.T, pr.N, len(self.pods)
        R, rf, Z = pr.R, pr.rf, pr.Z
        flags = pr.rebalance_flags(cfg["rebalance"])
        lag = int(cfg["exec_lag"])
        cost_rate = pr.cost_rate * cfg["cost_mult"]
        fin_spread, short_fee = cfg["financing_spread"] / ANN, cfg["short_fee"] / ANN
        no_rebate = 1.0 - cfg["short_rebate"]       # share of the cash rate NOT credited on short proceeds
        pod_vol = cfg["pod_vol"]
        priors = [cfg["pod_prior_sharpe"].get(pod.name, 0.0) for pod in self.pods]

        ret, excess = np.zeros(T), np.zeros(T)
        cost, fin, gross = np.zeros(T), np.zeros(T), np.zeros(T)
        pod_ret, contrib = np.zeros((T, P)), np.zeros((T, P))
        resid = np.zeros(T)                     # P&L from the gap between what is held and the pod mix
        held = np.zeros((T, N))
        H, HU = np.zeros(N), np.zeros((P, N))
        Hpv, Hvol = np.zeros(P), 0.0           # ex-ante vol of what is held (for risk-model calibration)
        lam = 0.5 ** (1.0 / 252)
        pod_b2, pod_bn, port_b2, port_bn = np.ones(P), np.zeros(P), 1.0, 0
        pending = None
        pod_n, pod_s, pod_ss = np.zeros(P), np.zeros(P), np.zeros(P)
        pod_eq = np.zeros((T, P))
        govs = [DrawdownGovernor(cfg["pod_soft_dd"] * pod_vol, cfg["pod_hard_dd"] * pod_vol) for _ in range(P)]
        port_gov = DrawdownGovernor(cfg["port_soft_dd"] * cfg["vol_target"], cfg["port_hard_dd"] * cfg["vol_target"])
        port_eq = np.zeros(T)
        rebalances, final = [], {}
        start = None

        for d in range(T):
            # --- trades scheduled for today take effect before today's return
            if pending is not None and pending["eff"] == d:
                W_exec = _trade_to_band(H, pending["W"], cfg["trade_buffer"])
                g_exec = np.abs(W_exec).sum()
                if g_exec > cfg["gross_cap"]:               # the band must never breach a hard account limit
                    W_exec = W_exec * cfg["gross_cap"] / g_exec
                U_exec = _trade_to_band(HU, pending["U"], cfg["trade_buffer"])
                cost[d] = float(np.abs(W_exec - H) @ cost_rate)
                pod_trade_cost = np.abs(U_exec - HU) @ cost_rate
                H, HU = W_exec, U_exec
                Hpv, Hvol = pending["pv"], pending["vol"]
                pending = None
            else:
                pod_trade_cost = np.zeros(P)
            r = np.nan_to_num(R[d], nan=0.0)
            ex = r - rf[d]

            def carry(Wv):
                """Financing on exposure above 1x, short-borrow fee, and the part of
                the cash rate not credited on short proceeds (per day)."""
                gr = np.abs(Wv).sum(axis=-1)
                sh = np.abs(np.minimum(Wv, 0.0)).sum(axis=-1)
                return fin_spread * np.maximum(gr - 1.0, 0.0) + (short_fee + no_rebate * rf[d]) * sh

            g = np.abs(H).sum()
            fin[d] = float(carry(H))
            excess[d] = H @ ex - cost[d] - fin[d]
            ret[d] = excess[d] + rf[d]
            gross[d] = g
            held[d] = H
            pg = HU @ ex
            pod_ret[d] = pg - pod_trade_cost - (carry(HU) if P else 0.0)
            # attribution of what is actually held: project H onto the pod vectors,
            # the remainder (from limits, the trade band, drift) is reported separately
            if P and np.abs(HU).sum() > 0:
                cw, *_ = np.linalg.lstsq(HU.T, H, rcond=None)
                contrib[d] = cw * pg
                resid[d] = (H - cw @ HU) @ ex
            else:
                resid[d] = H @ ex
            active = np.abs(HU).sum(axis=1) > 0
            pod_n += active
            pod_s += np.where(active, pod_ret[d], 0.0)
            pod_ss += np.where(active, pod_ret[d] ** 2, 0.0)
            pod_eq[d] = (pod_eq[d - 1] if d else 0.0) + pod_ret[d]
            # realised vs predicted variance (bias statistic), per pod and for the portfolio
            okp = Hpv > 0
            pod_b2 = np.where(okp, lam * pod_b2 + (1 - lam) * pg ** 2 * ANN / np.where(okp, Hpv, 1.0) ** 2, pod_b2)
            pod_bn += okp
            if Hvol > 0:
                port_b2 = lam * port_b2 + (1 - lam) * (H @ ex) ** 2 * ANN / Hvol ** 2
                port_bn += 1
            port_eq[d] = (port_eq[d - 1] if d else 0.0) + excess[d]
            # weights drift with prices until the next trade
            if g > 0:
                H = H * (1.0 + r) / max(1.0 + ret[d], 1e-6)
            if P:
                HU = HU * (1.0 + r) / np.maximum(1.0 + pg + rf[d], 1e-6)[:, None]

            if not flags[d]:
                continue
            avail = np.flatnonzero(pr.elig[d])
            if len(avail) < 3:
                continue
            if start is None:
                start = min(d + 1 + lag, T - 1)
            Sigma = pr.sigma_at(d, avail, cfg)
            vol_a = pr.vol_np[d, avail]

            # --- pods propose positions, each scaled to the same stand-alone volatility
            U = np.zeros((P, N))
            pv = np.zeros(P)
            for i, pod in enumerate(self.pods):
                w = np.nan_to_num(pod.weights(self, d, avail, Sigma, vol_a), nan=0.0)
                s = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
                if s < 1e-10:
                    continue
                u = w * pod_vol / s
                if cfg["risk_calibration"] and pod_bn[i] >= 252:
                    u = u / float(np.clip(np.sqrt(pod_b2[i]), 0.6, 1.5))
                gr = np.abs(u).sum()
                if gr > cfg["pod_gross_cap"]:
                    u *= cfg["pod_gross_cap"] / gr
                U[i, avail] = u
                pv[i] = float(np.sqrt(u @ Sigma @ u))
            Ua = U[:, avail]
            Omega = Ua @ Sigma @ Ua.T

            # --- evidence so far -> each pod's share of risk
            sr, sr_sd, gstate, dd = np.zeros(P), np.zeros(P), np.ones(P), np.zeros(P)
            lo = max(0, d - cfg["hwm_window"] + 1)
            for i in range(P):
                n = pod_n[i]
                mu = pod_s[i] / n if n else 0.0
                sd = np.sqrt(max(pod_ss[i] / n - mu * mu, 0.0)) if n > 1 else 0.0
                sr[i], sr_sd[i] = posterior_sharpe(n, mu, sd, priors[i], cfg["prior_sharpe_sd"])
                dd[i] = pod_eq[lo:d + 1, i].max() - pod_eq[d, i]
                gstate[i] = govs[i].update(dd[i]) if cfg["pod_stops"] else 1.0
            live = pv > 0
            b0 = np.where(live, 1.0 if cfg["tilt_power"] == 0 else np.maximum(sr, 0.0) ** cfg["tilt_power"], 0.0)
            b = b0 * gstate
            m = risk_budget_weights(Omega, b)
            W_raw = m @ Ua

            share = b.sum() / b0.sum() if b0.sum() > 0 else 0.0
            x = m * pv
            denom = float(np.sqrt(max(m @ Omega @ m, 0.0)))
            sr_port = float(x @ sr / denom) if denom > 0 else 0.0
            target = cfg["vol_target"] * np.sqrt(share)
            kelly_cap = cfg["kelly_fraction"] * max(sr_port, 0.0)
            scalars = {"pods_active": float(np.sqrt(share))}
            if kelly_cap < target:
                target = kelly_cap
                scalars["kelly_cap"] = kelly_cap
            ts = turbulence_scalar(pr.turb_pct[d], cfg["turbulence_start_pct"], cfg["turbulence_floor"]) \
                if cfg["turbulence_derisk"] else 1.0
            scalars["turbulence"] = ts
            pgs = 1.0
            if cfg["port_stop"]:
                pdd = port_eq[lo:d + 1].max() - port_eq[d]
                pgs = port_gov.update(pdd)
            scalars["portfolio_stop"] = pgs
            target *= ts * pgs
            risk_bias = float(np.clip(np.sqrt(port_b2), 0.7, 1.4)) if (cfg["risk_calibration"] and port_bn >= 252) else 1.0
            scalars["risk_model_bias"] = risk_bias

            zh = Z[max(0, d - 999):d + 1][:, avail]
            Wa, info = apply_limits(W_raw, Sigma, target / risk_bias, cfg, zh, vol_a)
            k = float(Wa @ W_raw / (W_raw @ W_raw)) if W_raw @ W_raw > 0 else 0.0
            W = np.zeros(N)
            W[avail] = Wa
            c = m * k
            pending = {"eff": d + 1 + lag, "W": W, "U": U, "pv": pv, "vol": info.get("ex_ante_vol", 0.0)}
            rec = {"p": d, "date": pr.dates[d], "W": W, "c": c, "m": m, "gov": gstate.copy(), "sr": sr.copy(),
                   "target": float(target), "ex_ante_vol": info.get("ex_ante_vol", 0.0),
                   "gross": float(np.abs(W).sum()), "net": float(W.sum()), "binding": info["binding"],
                   "turb": float(pr.turb_pct[d]) if np.isfinite(pr.turb_pct[d]) else None,
                   "sr_port": sr_port, "scalars": scalars,
                   "pod_bias": np.sqrt(pod_b2).copy()}
            rebalances.append(rec)
            if d == T - 1:
                final = dict(rec, avail=avail, Sigma=Sigma, U=U, pv=pv, Omega=Omega, sr_sd=sr_sd, dd=dd,
                             budgets=b, es_1d=info.get("es_1d"), W_raw=W_raw, H_now=H.copy(),
                             regime=pr.reg_np[d].copy(), vol=vol_a)

        return Result(pr.dates, pr.tickers, [p.name for p in self.pods], cfg, ret, excess, cost, fin,
                      pod_ret, contrib, resid, held, gross, rebalances, start or 0, final)


def run_backtest(md, cfg, prep=None, verbose=False):
    eng = Engine(md, cfg, prep, verbose)
    res = eng.run()
    res.prep = eng.prep
    res.pods = eng.pods
    return res
