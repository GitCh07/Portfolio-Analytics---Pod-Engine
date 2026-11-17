"""Performance, risk and attribution statistics."""
import numpy as np
import pandas as pd

from .cov import risk_contributions

ANN = 252

STRESS_WINDOWS = [
    ("Dot-com bust", "2000-03-24", "2002-10-09"),
    ("Global financial crisis", "2007-10-09", "2009-03-09"),
    ("Euro crisis / US downgrade", "2011-07-22", "2011-10-03"),
    ("Taper tantrum", "2013-05-22", "2013-06-24"),
    ("China devaluation", "2015-08-10", "2016-02-11"),
    ("Q4 2018 sell-off", "2018-09-20", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 inflation shock", "2022-01-03", "2022-10-14"),
    ("April 2025 tariff shock", "2025-04-02", "2025-04-08"),
]


def perf_stats(ret, rf=None):
    """Headline statistics for a daily return series (total returns). Sharpe
    and Sortino use returns over cash when `rf` is given."""
    r = pd.Series(ret).dropna()
    if len(r) < 20:
        return {}
    ex = r - rf.reindex(r.index).fillna(0.0) if rf is not None else r
    eq = (1 + r).cumprod()
    years = len(r) / ANN
    dd = eq / eq.cummax() - 1
    vol = r.std() * np.sqrt(ANN)
    down = ex[ex < 0]
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    q = r.quantile(0.025)
    return {
        "years": years,
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1),
        "vol": float(vol),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(ANN)) if ex.std() > 0 else 0.0,
        "sortino": float(ex.mean() * ANN / (np.sqrt((down ** 2).sum() / len(ex)) * np.sqrt(ANN))) if len(down) else 0.0,
        "max_dd": float(dd.min()),
        "calmar": float((eq.iloc[-1] ** (1 / years) - 1) / abs(dd.min())) if dd.min() < 0 else float("nan"),
        "skew": float(r.skew()),
        "kurt": float(r.kurt() + 3),
        "worst_day": float(r.min()),
        "worst_month": float(monthly.min()),
        "pct_pos_months": float((monthly > 0).mean()),
        "es_97_5": float(-r[r <= q].mean()),
    }


def drawdown(ret):
    eq = (1 + pd.Series(ret).fillna(0.0)).cumprod()
    return eq / eq.cummax() - 1


def yearly_returns(ret):
    r = pd.Series(ret).dropna()
    return (1 + r).groupby(r.index.year).prod() - 1


def benchmarks(md, index):
    """Simple reference portfolios over the same dates (total returns, no costs)."""
    R = md.returns.reindex(index)
    out = {}
    eq = "SPY" if "SPY" in R else None
    bd = "IEF" if "IEF" in R else ("TLT" if "TLT" in R else None)
    if eq:
        out["Stocks (SPY)"] = R[eq].fillna(0.0)
    if eq and bd:
        b = R[bd]
        out["60/40 (SPY/" + bd + ")"] = (0.6 * R[eq].fillna(0.0) + 0.4 * b.fillna(md.rf.reindex(index))).where(R[eq].notna(), 0.0)
    if not out:
        out["Equal weight"] = R.mean(axis=1).fillna(0.0)
    return out


def stress_table(series_dict, min_days=3):
    """Return over named historical episodes for each series in the dict."""
    rows = []
    for name, a, b in STRESS_WINDOWS:
        row = {"episode": name, "from": a, "to": b}
        ok = False
        for k, s in series_dict.items():
            w = s.loc[a:b]
            if len(w) >= min_days and s.index[0] <= pd.Timestamp(a) + pd.Timedelta(days=10):
                row[k] = float((1 + w).prod() - 1)
                ok = True
            else:
                row[k] = None
        if ok:
            rows.append(row)
    return rows


def worst_drawdowns(ret, n=5):
    r = pd.Series(ret).fillna(0.0)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    out, work = [], dd.copy()
    for _ in range(n):
        if work.min() >= -1e-6:
            break
        trough = work.idxmin()
        peak = eq.loc[:trough].idxmax()
        after = dd.loc[trough:]
        rec = after[after >= -1e-9]
        end = rec.index[0] if len(rec) else dd.index[-1]
        out.append({"peak": peak, "trough": trough, "recovered": end if len(rec) else None,
                    "depth": float(dd.loc[trough])})
        work.loc[peak:end] = 0.0
    return out


def exposure_summary(res):
    """Today's target: weights and risk contributions by asset, class and pod."""
    f, pr = res.final, res.prep
    if not f:
        return None
    avail, Sigma = f["avail"], f["Sigma"]
    w = f["W"][avail]
    rc, port = risk_contributions(w, Sigma)
    rows = []
    for j, a in enumerate(avail):
        m = pr.meta_list[a]
        rows.append({"ticker": pr.tickers[a], "name": m["name"], "cls": m["cls"], "weight": float(w[j]),
                     "vol": float(f["vol"][j]), "risk": float(rc[j]),
                     "risk_share": float(rc[j] / port) if port > 0 else 0.0})
    assets = pd.DataFrame(rows)
    by_cls = assets.groupby("cls").agg(long=("weight", lambda x: x[x > 0].sum()),
                                       short=("weight", lambda x: x[x < 0].sum()),
                                       net=("weight", "sum"), risk_share=("risk_share", "sum")).reset_index()
    # pod risk shares: project the actual target W onto the pod vectors, W = sum_i c_i u_i + e,
    # then Euler-decompose W's variance; e (from account limits) gets its own share. Shares sum to 1.
    Ua = f["U"][:, avail]
    var = float(w @ Sigma @ w)
    if var > 0 and np.abs(Ua).sum() > 0:
        c, *_ = np.linalg.lstsq(Ua.T, w, rcond=None)
        e = w - c @ Ua
        sw = Sigma @ w
        pod_share = c * (Ua @ sw) / var
        resid_share = float(e @ sw / var)
    else:
        pod_share, resid_share = np.zeros(Ua.shape[0]), 0.0
    return {"assets": assets, "by_cls": by_cls, "pod_share": pod_share, "resid_share": resid_share, "port_vol": port}


def pod_table(res):
    """Scoreboard: each pod's out-of-sample record and current standing."""
    f, cfg = res.final, res.cfg
    pf = res.pod_frame()
    rows = []
    for i, pod in enumerate(res.pods):
        x = pf[pod.name]
        x = x[x != 0]
        sr_raw = float(x.mean() / x.std() * np.sqrt(ANN)) if len(x) > 60 and x.std() > 0 else float("nan")
        g = f["gov"][i] if f else 1.0
        rows.append({
            "pod": pod.name, "idea": pod.description, "source": pod.source,
            "experimental": getattr(pod, "experimental", False),
            "years": len(x) / ANN, "sharpe_raw": sr_raw,
            "prior": cfg["pod_prior_sharpe"].get(pod.name, 0.0),
            "sharpe_shrunk": float(f["sr"][i]) if f else float("nan"),
            "sharpe_sd": float(f["sr_sd"][i]) if f else float("nan"),
            "drawdown": float(f["dd"][i]) if f else 0.0,
            "state": ("benched" if (f and f["sr"][i] <= 0 and g > 0) else
                      {1.0: "full", 0.5: "halved", 0.0: "off"}.get(float(g), str(g))),
            "max_dd": float(drawdown(x).min()) if len(x) else 0.0,
        })
    return pd.DataFrame(rows)
