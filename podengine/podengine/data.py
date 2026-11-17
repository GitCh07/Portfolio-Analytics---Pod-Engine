"""Market data: Yahoo Finance loader with a local cache, CSV loader, and a
synthetic generator for the offline demo and the tests."""
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .universe import DEFAULT_UNIVERSE, FUTURES, build_meta

ANN = 252


@dataclass
class MarketData:
    close: pd.DataFrame                      # adjusted closes, dates x tickers
    open: pd.DataFrame | None = None
    high: pd.DataFrame | None = None
    low: pd.DataFrame | None = None
    rf: pd.Series | None = None              # daily risk-free rate (decimal per day)
    meta: dict = field(default_factory=dict)
    source: str = "unknown"
    synthetic: bool = False

    def __post_init__(self):
        self.close = self.close.sort_index()
        self.close = self.close.loc[~self.close.index.duplicated()]
        self.close = self.close.dropna(how="all")
        idx = self.close.index
        for name in ("open", "high", "low"):
            df = getattr(self, name)
            if df is not None:
                setattr(self, name, df.reindex(index=idx, columns=self.close.columns))
        if self.rf is None:
            self.rf = pd.Series(0.0, index=idx)
        else:
            self.rf = self.rf.reindex(idx).ffill().bfill().fillna(0.0)
        if not self.meta:
            self.meta = build_meta(list(self.close.columns))

    @property
    def tickers(self):
        return list(self.close.columns)

    @property
    def returns(self):
        """Simple daily returns; NaN before an asset is listed."""
        return self.close.pct_change(fill_method=None)

    def var_proxy(self):
        """Daily variance proxy. Uses the Garman-Klass range estimator plus the
        overnight gap when open/high/low exist (far less noisy than squared
        close-to-close returns); falls back to squared returns otherwise."""
        r2 = np.log(self.close / self.close.shift(1)) ** 2
        if self.open is None or self.high is None or self.low is None:
            return r2
        o, h, l, c = self.open, self.high, self.low, self.close
        ok = (o > 0) & (h > 0) & (l > 0) & (h >= l)
        overnight = np.log(o / c.shift(1)) ** 2
        gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
        v = (overnight + gk).where(ok & (gk >= 0))
        return v.where(v.notna(), r2)

    def truncate(self, end):
        """Copy of the data up to and including `end` (used by look-ahead tests)."""
        sl = lambda df: None if df is None else df.loc[:end]
        return MarketData(self.close.loc[:end], sl(self.open), sl(self.high), sl(self.low),
                          self.rf.loc[:end], dict(self.meta), self.source, self.synthetic)


# --------------------------------------------------------------------------- Yahoo

def _split_yahoo(df, tickers):
    """Turn a yfinance download (any column layout) into {field: wide frame}."""
    out = {}
    if not isinstance(df.columns, pd.MultiIndex):
        df = pd.concat({tickers[0]: df}, axis=1)
    lvl0 = set(df.columns.get_level_values(0))
    fields_first = bool(lvl0 & {"Close", "Open", "High", "Low", "Adj Close"})
    for fld in ("Open", "High", "Low", "Close"):
        try:
            sub = df.xs(fld, axis=1, level=0 if fields_first else 1)
        except KeyError:
            continue
        out[fld.lower()] = sub.reindex(columns=[t for t in tickers if t in sub.columns])
    return out


def fetch_yahoo(tickers=None, start="1999-01-01", cache_dir="data_cache", refresh=False):
    """Download adjusted daily OHLC from Yahoo Finance (free, no key) and cache
    it as CSV. With refresh=False a cache from today is reused; if the download
    fails, any existing cache is used so the engine still runs offline."""
    tickers = list(tickers or DEFAULT_UNIVERSE)
    os.makedirs(cache_dir, exist_ok=True)
    paths = {f: os.path.join(cache_dir, f"{f}.csv") for f in ("open", "high", "low", "close", "rf")}

    def read_cache():
        frames = {f: pd.read_csv(p, index_col=0, parse_dates=True) for f, p in paths.items()
                  if os.path.exists(p)}
        if "close" not in frames:
            return None
        rf = frames.pop("rf", None)
        rf = None if rf is None else rf.iloc[:, 0]
        cl = frames["close"]
        keep = [t for t in tickers if t in cl.columns]
        return MarketData(cl[keep], frames.get("open"), frames.get("high"), frames.get("low"),
                          rf, build_meta(keep), source="Yahoo Finance (cached)")

    cached = read_cache() if os.path.exists(paths["close"]) else None
    fresh = cached is not None and set(tickers) <= set(cached.tickers) and \
        pd.Timestamp(os.path.getmtime(paths["close"]), unit="s").date() == pd.Timestamp.now().date()
    if cached is not None and fresh and not refresh:
        return cached
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                          group_by="ticker", threads=True)
        if raw is None or raw.empty:
            raise RuntimeError("empty download")
        frames = _split_yahoo(raw, tickers)
        close = frames["close"].dropna(how="all", axis=1)
        if close.empty:
            raise RuntimeError("no prices returned")
        missing = sorted(set(tickers) - set(close.columns))
        if missing:
            print(f"  note: no data for {missing}; continuing without them")
        try:
            irx = yf.download("^IRX", start=start, auto_adjust=True, progress=False)
            irx = _split_yahoo(irx, ["^IRX"])["close"].iloc[:, 0]
            rf = (irx / 100.0 / ANN).clip(lower=0).rename("rf")
        except Exception:
            rf = pd.Series(0.0, index=close.index, name="rf")
        for f in ("open", "high", "low", "close"):
            if f in frames:
                frames[f].reindex(columns=close.columns).to_csv(paths[f])
        rf.to_frame().to_csv(paths["rf"])
        return MarketData(close, frames.get("open"), frames.get("high"), frames.get("low"),
                          rf, build_meta(list(close.columns)), source="Yahoo Finance")
    except Exception as e:  # network down, Yahoo changed something, ...
        if cached is not None:
            print(f"  download failed ({e}); using cached data")
            return cached
        raise RuntimeError(
            f"Could not download prices ({e}) and no cache exists. Check your connection, "
            "run `pip install -U yfinance`, or pass your own file with --prices.") from e


def fetch_futures(etfs, cache_dir="data_cache", period="2y"):
    """Daily closes of the continuous front-month futures mapped to `etfs`.
    Used only to price contracts and estimate hedge ratios. Returns {} on failure."""
    want = {e: FUTURES[e]["yahoo"] for e in etfs if e in FUTURES}
    if not want:
        return {}
    try:
        import yfinance as yf
        raw = yf.download(list(want.values()), period=period, auto_adjust=True, progress=False,
                          group_by="ticker", threads=True)
        close = _split_yahoo(raw, list(want.values()))["close"]
        return {e: close[y].dropna() for e, y in want.items() if y in close.columns}
    except Exception as e:
        print(f"  futures prices unavailable ({e}); order sheet will show ETFs only")
        return {}


def load_csv(path, meta_path=None, rf_annual=0.0):
    """Load your own prices: a CSV with a date column first, then one column of
    adjusted closes per ticker. Optional meta JSON: {ticker: {cls, group}}."""
    close = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
    meta = build_meta(list(close.columns))
    if meta_path:
        import json
        with open(meta_path) as f:
            for t, m in json.load(f).items():
                if t in meta:
                    meta[t].update(m)
    rf = pd.Series(rf_annual / ANN, index=close.index)
    return MarketData(close, rf=rf, meta=meta, source=os.path.basename(path))


# ----------------------------------------------------------------------- synthetic

_LISTING = {"EEM": "2003-04-14", "IEI": "2007-01-11", "IEF": "2002-07-26", "TLT": "2002-07-26",
            "TIP": "2003-12-05", "LQD": "2002-07-26", "HYG": "2007-04-11", "EMB": "2007-12-19",
            "GLD": "2004-11-18", "SLV": "2006-04-28", "DBC": "2006-02-06", "USO": "2006-04-10",
            "DBA": "2007-01-05", "VNQ": "2004-09-29", "FXE": "2005-12-12", "FXY": "2007-02-13",
            "FXB": "2006-06-26", "FXA": "2006-06-26", "FXC": "2006-06-26", "FXI": "2004-10-08",
            "EFA": "2001-08-27", "IWM": "2000-05-26", "EWY": "2000-05-12", "EWT": "2000-06-23"}

# loadings on [equity, duration, commodity, dollar, credit] and idiosyncratic vol
_LOAD = {
    "equity_index": ([1.0, 0.0, 0.05, -0.05, 0.1], 0.05), "sector": ([1.0, 0.0, 0.05, 0.0, 0.1], 0.10),
    "country": ([1.05, 0.0, 0.15, -0.6, 0.1], 0.12), "bonds": ([0.0, 1.0, 0.0, 0.0, 0.0], 0.01),
    "commodity": ([0.15, 0.0, 1.0, -0.3, 0.0], 0.12), "fx": ([0.05, 0.0, 0.1, -1.0, 0.0], 0.03),
    None: ([0.9, 0.3, 0.1, 0.0, 0.3], 0.10),
}
_SPECIAL = {"IEI": [0.0, 0.55, 0, 0, 0], "IEF": [0.0, 1.0, 0, 0, 0], "TLT": [0.0, 2.2, 0, 0, 0],
            "TIP": [0.0, 0.8, 0.12, 0, 0], "LQD": [0.05, 1.1, 0, 0, 0.6], "HYG": [0.25, 0.3, 0, 0, 1.4],
            "EMB": [0.2, 0.8, 0.05, -0.2, 1.2], "QQQ": [1.25, 0, 0, 0, 0], "IWM": [1.2, 0, 0.05, 0, 0.2],
            "XLU": [0.6, 0.35, 0, 0, 0], "XLP": [0.6, 0.1, 0, 0, 0], "XLE": [1.0, 0, 0.6, 0, 0.1],
            "XLK": [1.2, 0, 0, 0, 0], "XLF": [1.2, -0.1, 0, 0, 0.3], "GLD": [0.0, 0.4, 0.5, -0.5, 0],
            "USO": [0.2, 0, 2.0, -0.2, 0]}


def make_synthetic(start="2000-01-03", end="2025-12-31", seed=11, tickers=None):
    """Simulated multi-asset history for the offline demo and the tests.

    A regime-switching factor model with fat tails, volatility clustering,
    mild trends, slow mean reversion and staggered listing dates. It is built
    to exercise every code path; RESULTS ON IT SAY NOTHING ABOUT REAL MARKETS.
    """
    rng = np.random.default_rng(seed)
    tickers = list(tickers or DEFAULT_UNIVERSE)
    meta = build_meta(tickers)
    idx = pd.bdate_range(start, end)
    T, N, F = len(idx), len(tickers), 5

    # market regime: calm / stress / crisis
    P = np.array([[0.985, 0.014, 0.001], [0.040, 0.940, 0.020], [0.000, 0.080, 0.920]])
    reg = np.zeros(T, dtype=int)
    u = rng.random(T)
    for t in range(1, T):
        reg[t] = np.searchsorted(np.cumsum(P[reg[t - 1]]), u[t])
    reg = np.minimum(reg, 2)
    # inflation regime (stocks and bonds fall together)
    infl = np.zeros(T, dtype=int)
    u2 = rng.random(T)
    for t in range(1, T):
        stay = 0.998 if infl[t - 1] == 0 else 0.995
        infl[t] = infl[t - 1] if u2[t] < stay else 1 - infl[t - 1]

    vol_mult = np.array([0.8, 1.4, 3.0])[reg]
    logv = np.zeros(T)
    e = rng.normal(0, 0.08, T)
    for t in range(1, T):
        logv[t] = 0.97 * logv[t - 1] + e[t]
    vol_mult = vol_mult * np.exp(logv - 0.5 * logv.var())

    f_vol = np.array([0.15, 0.065, 0.15, 0.07, 0.05]) / np.sqrt(ANN)
    prem = np.array([0.40, 0.25, 0.10, 0.0, 0.30]) * f_vol * np.sqrt(ANN) / ANN
    reg_drift = np.zeros((3, F))
    reg_drift[:, 0] = np.array([0.13, -0.10, -0.55]) / ANN
    reg_drift[:, 4] = np.array([0.02, -0.03, -0.20]) / ANN
    reg_drift[:, 1] = np.array([0.0, 0.01, 0.06]) / ANN

    def corr_for(r, inf):
        C = np.eye(F)
        ed = [-0.2, -0.3, -0.5][r] if not inf else 0.45
        ec, eu, ecr = [0.15, 0.3, 0.5][r], [-0.15, -0.3, -0.5][r], [0.4, 0.6, 0.8][r]
        for (i, j, v) in ((0, 1, ed), (0, 2, ec), (0, 3, eu), (0, 4, ecr), (2, 3, -0.3), (1, 4, -0.1)):
            C[i, j] = C[j, i] = v
        w, V = np.linalg.eigh(C)
        return V @ np.diag(np.sqrt(np.clip(w, 1e-6, None)))

    chol = {(r, i): corr_for(r, i) for r in range(3) for i in range(2)}
    shocks = rng.standard_t(5, size=(T, F)) / np.sqrt(5 / 3)
    trend = np.zeros((T, F))
    te = rng.normal(0, 1, (T, F))
    phi = 0.5 ** (1 / 126)
    for t in range(1, T):
        trend[t] = phi * trend[t - 1] + np.sqrt(1 - phi ** 2) * te[t]
    fac = np.empty((T, F))
    for t in range(T):
        z = chol[(reg[t], infl[t])] @ shocks[t]
        drift = prem + reg_drift[reg[t]] + 0.9 * trend[t] * f_vol / np.sqrt(ANN)
        if infl[t]:
            drift = drift + np.array([-0.05, -0.04, 0.15, 0.0, 0.0]) / ANN
        fac[t] = drift + f_vol * vol_mult[t] * z

    B = np.zeros((N, F))
    ivol = np.zeros(N)
    for i, tk in enumerate(tickers):
        load, iv = _LOAD.get(meta[tk]["group"], _LOAD[None])
        B[i] = _SPECIAL.get(tk, load)
        ivol[i] = iv / np.sqrt(ANN)
    # idiosyncratic part: momentum (OU drift) plus slow mean reversion in level (value)
    idio_e = rng.standard_t(5, size=(T, N)) / np.sqrt(5 / 3)
    mom = np.zeros((T, N))
    me = rng.normal(0, 1, (T, N))
    phi_m = 0.5 ** (1 / 90)
    level = np.zeros(N)
    idio = np.empty((T, N))
    kappa = np.log(2) / (3 * ANN)
    for t in range(T):
        if t:
            mom[t] = phi_m * mom[t - 1] + np.sqrt(1 - phi_m ** 2) * me[t]
        idio[t] = ivol * (idio_e[t] * np.sqrt(vol_mult[t]) + 0.05 * mom[t]) - kappa * level
        level += idio[t]
    short = np.zeros(T)
    se = rng.normal(0, 0.0006, T)
    short[0] = 0.03
    for t in range(1, T):
        short[t] = np.clip(short[t - 1] + 0.001 * (0.025 - short[t - 1]) + se[t], 0.0, 0.07)
    r = fac @ B.T + idio + short[:, None] / ANN      # assets earn cash plus a premium
    r = np.clip(r, -0.35, 0.35)
    close = 50.0 * np.exp(np.cumsum(np.log1p(r), axis=0))

    # open / high / low from a coarse intraday walk (30% of variance overnight)
    sig = np.sqrt(((f_vol * vol_mult[:, None]) ** 2) @ (B.T ** 2) + (ivol ** 2) * vol_mult[:, None])
    on = rng.normal(0, 1, (T, N)) * sig * np.sqrt(0.3)
    prev = np.vstack([close[:1], close[:-1]])
    opn = prev * np.exp(on)
    steps = rng.normal(0, 1, (8, T, N)) * sig * np.sqrt(0.7 / 8)
    path = np.cumsum(steps, axis=0)
    path = path - path[-1] * (np.arange(1, 9)[:, None, None] / 8)       # bridge: start 0, end 0
    target = np.log(close / opn)
    path = path + target[None] * (np.arange(1, 9)[:, None, None] / 8)   # end at the close
    hi = opn * np.exp(np.maximum(path.max(0), 0))
    lo = opn * np.exp(np.minimum(path.min(0), 0))
    hi = np.maximum(hi, np.maximum(opn, close))
    lo = np.minimum(lo, np.minimum(opn, close))

    def frame(a):
        df = pd.DataFrame(a, index=idx, columns=tickers)
        for tk, d in _LISTING.items():
            if tk in df.columns:
                df.loc[df.index < pd.Timestamp(d), tk] = np.nan
        return df

    rf = pd.Series(short / ANN, index=idx)
    md = MarketData(frame(close), frame(opn), frame(hi), frame(lo), rf, meta,
                    source="SYNTHETIC DEMO DATA", synthetic=True)
    md.true_regime = pd.Series(reg, index=idx)
    return md
