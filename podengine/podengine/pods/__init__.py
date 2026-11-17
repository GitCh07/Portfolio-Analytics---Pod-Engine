"""Pods: independent return streams. Each one turns market data into a vector
of positions; the engine sizes every pod to the same stand-alone volatility,
tracks its paper P&L, and decides how much capital it deserves.

To add your own pod, subclass `Pod`, implement `prepare` and `weights`, and
register it in `REGISTRY`.
"""
import numpy as np
import pandas as pd

from ..allocator import risk_budget_weights
from ..universe import RISK_PARITY_CLASSES

ANN = 252


def centered_ranks(x):
    """Ranks mapped to [-1, 1] with mean zero; NaN stays NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    n = ok.sum()
    if n < 2:
        return out
    r = pd.Series(x[ok]).rank().to_numpy()
    out[ok] = (r - (n + 1) / 2.0) / ((n - 1) / 2.0)
    return out


def group_scores(eng, raw, min_n=4, only_cls=None):
    """Daily cross-sectional rank scores of `raw` (T x N) within each peer
    group, among assets eligible on that day, smoothed over a few days to cut
    turnover. Row t uses data up to day t only."""
    T, N = raw.shape
    out = np.full((T, N), np.nan)
    groups = {}
    for j, m in enumerate(eng.meta_list):
        if m["group"] is not None and (only_cls is None or m["cls"] in only_cls):
            groups.setdefault(m["group"], []).append(j)
    elig = eng.prep.elig
    for cols in groups.values():
        if len(cols) < min_n:
            continue
        x = pd.DataFrame(np.where(elig[:, cols], raw[:, cols], np.nan))
        n = x.notna().sum(axis=1)
        r = x.rank(axis=1)
        k = r.sub((n + 1) / 2.0, axis=0).div(((n - 1) / 2.0).where(n >= min_n), axis=0)
        out[:, cols] = k.to_numpy()
    hl = eng.cfg.get("score_halflife", 0)
    if hl and hl > 0:
        sm = pd.DataFrame(out).ewm(halflife=hl, min_periods=1, ignore_na=True).mean().to_numpy()
        out = np.where(np.isfinite(out), sm, np.nan)
    return out


class Pod:
    name = "pod"
    description = ""
    source = ""

    def prepare(self, eng):
        """Pre-compute signals. Row t may only use data up to the close of day t."""

    def weights(self, eng, p, avail, Sigma, vol):
        """Unscaled positions for the assets in `avail` (integer column indices)
        at day index p. `Sigma`/`vol` are for those assets, in that order."""
        raise NotImplementedError

    # helpers -------------------------------------------------------------
    @staticmethod
    def _groups(eng, avail, key="group", only_cls=None):
        out = {}
        for j, a in enumerate(avail):
            m = eng.meta_list[a]
            if only_cls and m["cls"] not in only_cls:
                continue
            g = m[key]
            if g is not None:
                out.setdefault(g, []).append(j)
        return out


class TrendPod(Pod):
    name = "trend"
    description = "Time-series momentum: long what has been rising, short what has been falling, across every asset class."
    source = "AQR / managed-futures funds (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017)"

    def prepare(self, eng):
        logp = np.log(eng.md.close)
        vol = eng.vol
        parts = []
        for L in eng.cfg["trend_lookbacks"]:
            z = (logp - logp.shift(L)) / (vol * np.sqrt(L / ANN))
            parts.append(z.clip(-2, 2) / 2.0)
        cnt = sum(p.notna().astype(float) for p in parts)
        tot = sum(p.fillna(0.0) for p in parts)
        self.sig = (tot / cnt.where(cnt >= 2)).to_numpy()

    def weights(self, eng, p, avail, Sigma, vol):
        s = np.nan_to_num(self.sig[p, avail], nan=0.0)
        w = s / vol
        for idx in self._groups(eng, avail, key="cls").values():     # equal risk per asset class
            w[idx] = w[idx] / len(idx)
        return w


class ValueMomentumPod(Pod):
    name = "value_momentum"
    description = "Within each peer group, long cheap and strong assets, short expensive and weak ones. The two signals offset each other's bad periods."
    source = "AQR (Asness, Moskowitz & Pedersen 2013, 'Value and Momentum Everywhere')"

    def prepare(self, eng):
        c = eng.md.close
        hi, lo = eng.cfg["mom_window"]
        mom = (c.shift(lo) / c.shift(hi) - 1.0).to_numpy()
        yrs = eng.cfg["value_years"]
        old = c.rolling(ANN, min_periods=200).mean().shift(int((yrs - 0.5) * ANN))
        val = (np.log(old) - np.log(c)).to_numpy()                   # high = cheap vs 5 years ago
        self.mom, self.val = group_scores(eng, mom), group_scores(eng, val)

    def weights(self, eng, p, avail, Sigma, vol):
        w = np.zeros(len(avail))
        groups = {g: i for g, i in self._groups(eng, avail).items() if len(i) >= 4}
        for idx in groups.values():
            a = avail[idx]
            m, v = self.mom[p, a], self.val[p, a]
            if np.isfinite(m).sum() < 4:
                continue
            score = np.nan_to_num(m, nan=0.0)
            if np.isfinite(v).sum() >= 4:
                score = 0.5 * score + 0.5 * np.nan_to_num(v, nan=0.0)
            w[idx] = score / vol[idx] / len(groups)
        return w


class DefensivePod(Pod):
    name = "defensive"
    description = "Betting against beta: long low-beta equity markets (levered up), short high-beta ones (scaled down), so the pair has no net market exposure."
    source = "AQR (Frazzini & Pedersen 2014)"

    def prepare(self, eng):
        r = eng.md.returns
        self.beta = np.full(r.shape, np.nan)
        cols = {t: i for i, t in enumerate(eng.md.tickers)}
        groups = {}
        for t in eng.md.tickers:
            m = eng.md.meta[t]
            if m["cls"] == "equity" and m["group"]:
                groups.setdefault(m["group"], []).append(t)
        for members in groups.values():
            if len(members) < 4:
                continue
            sub = r[members]
            mkt = sub.mean(axis=1)
            kw = dict(halflife=252, min_periods=252)
            ex, ey = sub.ewm(**kw).mean(), mkt.ewm(**kw).mean()
            cov = sub.mul(mkt, axis=0).ewm(**kw).mean() - ex.mul(ey, axis=0)
            var = (mkt ** 2).ewm(**kw).mean() - ey ** 2
            b = 0.6 * cov.div(var, axis=0) + 0.4                       # shrink toward 1
            for t in members:
                self.beta[:, cols[t]] = b[t].to_numpy()
        self.score = group_scores(eng, -self.beta, only_cls=("equity",))

    def weights(self, eng, p, avail, Sigma, vol):
        w = np.zeros(len(avail))
        groups = {g: i for g, i in self._groups(eng, avail, only_cls=("equity",)).items() if len(i) >= 4}
        for idx in groups.values():
            beta = self.beta[p, avail[idx]]
            k = self.score[p, avail[idx]]
            if np.isfinite(k).sum() < 4:
                continue
            k = np.nan_to_num(k, nan=0.0)
            b = np.nan_to_num(beta, nan=1.0)
            lo, hi = np.maximum(k, 0), np.maximum(-k, 0)
            if lo.sum() <= 0 or hi.sum() <= 0:
                continue
            lo, hi = lo / lo.sum(), hi / hi.sum()
            b_lo, b_hi = max(lo @ b, 0.2), max(hi @ b, 0.2)
            w[idx] = (lo / b_lo - hi / b_hi) / len(groups)
        return w


class RiskParityPod(Pod):
    name = "risk_parity"
    description = "Long-only core: equal risk from equities, government bonds, credit and real assets, so no single economic environment dominates."
    source = "Bridgewater All Weather (risk balanced across growth and inflation outcomes)"

    def prepare(self, eng):
        pass

    def weights(self, eng, p, avail, Sigma, vol):
        w = np.zeros(len(avail))
        classes = self._groups(eng, avail, key="cls", only_cls=RISK_PARITY_CLASSES)
        if not classes:
            return w
        names, vecs = [], []
        for c, idx in classes.items():
            sub = Sigma[np.ix_(idx, idx)]
            x = risk_budget_weights(sub, np.ones(len(idx)))
            v = np.zeros(len(avail))
            v[idx] = x / np.sqrt(max(x @ sub @ x, 1e-16))              # unit-vol class portfolio
            names.append(c)
            vecs.append(v)
        V = np.array(vecs)
        top = risk_budget_weights(V @ Sigma @ V.T, np.ones(len(names)))
        return top @ V


class ReversalPod(Pod):
    name = "reversal"
    experimental = True
    description = "EXPERIMENTAL. Buys last week's laggards and sells its leaders within equity groups, sized up when markets are turbulent (when liquidity is scarce and providing it pays more). Starts with a prior of zero edge, so it only gets capital if its own record earns it."
    source = "Liquidity provision (Lehmann 1990; Nagel 2012). Weak at ETF level and cost-sensitive - the validation gate decides."

    def prepare(self, eng):
        c = eng.md.close
        d = eng.cfg["reversal_days"]
        self.rev = (-(c / c.shift(d) - 1.0) / (eng.vol * np.sqrt(d / ANN))).to_numpy()

    def weights(self, eng, p, avail, Sigma, vol):
        w = np.zeros(len(avail))
        groups = {g: i for g, i in self._groups(eng, avail, only_cls=("equity",)).items() if len(i) >= 4}
        for idx in groups.values():
            k = centered_ranks(self.rev[p, avail[idx]])
            if np.isfinite(k).sum() < 4:
                continue
            w[idx] = np.nan_to_num(k, nan=0.0) / vol[idx] / len(groups)
        t = eng.turb_pct[p]
        return w * (0.5 + t if np.isfinite(t) else 1.0)


REGISTRY = {c.name: c for c in (TrendPod, ValueMomentumPod, DefensivePod, RiskParityPod, ReversalPod)}
