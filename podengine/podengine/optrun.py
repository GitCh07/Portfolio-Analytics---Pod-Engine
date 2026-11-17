"""Load data, optimise a set of holdings and write the report. Shared by the
`optimize` command and the viewer's Optimize button (podengine/server.py)."""
import json
import os
import re
import shutil

import pandas as pd

from .backtest import Prep
from .config import load_config
from .data import fetch_yahoo, load_csv, make_synthetic
from .optimizer import optimize_holdings, positions_to_weights
from .report import build_opt_report

DEMO_POSITIONS = pd.DataFrame({"ticker": ["SPY", "QQQ", "XLK", "EFA", "TLT", "GLD", "CASH"],
                               "value": [42000, 22000, 10000, 8000, 9000, 4000, 5000]})
DEMO_VIEWS = [{"label": "Long bonds beat cash by 4%", "assets": {"TLT": 1}, "return": 0.04, "confidence": 0.6},
              {"label": "Nasdaq lags the S&P by 3%", "assets": {"QQQ": 1, "SPY": -1}, "return": -0.03, "confidence": 0.5}]

TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")
MAX_HOLDINGS = 60


def _num(text, pct_ok=True):
    t = str(text).strip().replace(",", "").replace("$", "")
    if pct_ok and t.endswith("%"):
        return float(t[:-1]) / 100.0
    return float(t)


def parse_holdings_text(text, kind):
    """Lines of 'TICKER, amount' (comma, tab or spaces). A CASH line holds dollars.
    Returns a positions DataFrame with columns ticker and `kind`."""
    if kind not in ("shares", "value", "weight"):
        raise ValueError("amounts must be shares, value or weight")
    rows = {}
    for ln, raw in enumerate(str(text).splitlines(), 1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([^\s,;\t]+)[\s,;\t]+(.+)$", line)     # split at the first separator only,
        if not m:                                                   # so "CASH, $5,000" keeps its thousands comma
            raise ValueError(f"line {ln}: expected 'TICKER, amount'")
        parts = [m.group(1), m.group(2).replace(" ", "")]
        if parts[0].lower() == "ticker":           # a pasted CSV header row
            continue
        t = parts[0].upper()
        if not TICKER.match(t):
            raise ValueError(f"line {ln}: '{parts[0]}' is not a ticker")
        try:
            amt = _num(parts[1], pct_ok=(kind == "weight" and t != "CASH"))
        except ValueError:
            raise ValueError(f"line {ln}: '{parts[1]}' is not a number") from None
        rows[t] = rows.get(t, 0.0) + amt
    if not [t for t in rows if t != "CASH"]:
        raise ValueError("enter at least two holdings")
    if len(rows) > MAX_HOLDINGS:
        raise ValueError(f"at most {MAX_HOLDINGS} lines")
    return pd.DataFrame({"ticker": list(rows), kind: list(rows.values())})


def parse_views_text(text):
    """Lines of 'ASSET, return, confidence' or 'LONG - SHORT, return, confidence',
    e.g. 'TLT, 4%, 60%' or 'QQQ - SPY, -3%, 50%'. Confidence defaults to 50%."""
    views = []
    for ln, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise ValueError(f"view line {ln}: expected 'ASSET, return, confidence'")
        expr = parts[0].upper().replace(" ", "")
        legs = expr.split("-") if "-" in expr.strip("-") else [expr]
        if len(legs) > 2 or not all(TICKER.match(x) for x in legs):
            raise ValueError(f"view line {ln}: use one ticker or 'LONG - SHORT'")
        assets = {legs[0]: 1.0} if len(legs) == 1 else {legs[0]: 1.0, legs[1]: -1.0}
        try:
            ret = _num(parts[1])
            conf = _num(parts[2]) if len(parts) > 2 and parts[2] else 0.5
        except ValueError:
            raise ValueError(f"view line {ln}: return and confidence must be numbers") from None
        if not 0 < conf < 1:
            raise ValueError(f"view line {ln}: confidence must be between 0% and 100%")
        label = (f"{legs[0]} beats {legs[1]} by {ret:.1%} a year" if len(legs) == 2
                 else f"{legs[0]} returns {ret:.1%} a year over cash")
        views.append({"label": label, "assets": assets, "return": ret, "confidence": conf})
    return views


def optimize_and_report(pos, kind, views=None, *, profile=None, vol_target=None, max_turnover=None, capital=None,
                        add=(), tau=0.05, prior_sharpe=0.30, demo=False, prices_csv=None, meta=None,
                        start="1999-01-01", cache="data_cache", refresh=False, config=None, out_dir="output",
                        viz_dir=None, copy_to_viz=False, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    cfg = load_config(config, profile=profile, vol_target=None)
    if not (config and "min_history" in json.load(open(config))):
        cfg["min_history"] = 252                    # one year of history is enough to hold an asset
    add = [t.strip().upper() for t in add if t.strip()]
    held = [t for t in pos["ticker"] if t != "CASH"]
    say("Loading data ...")
    if demo:
        md = make_synthetic(seed=11)
    elif prices_csv:
        md = load_csv(prices_csv, meta)
    else:
        want = list(dict.fromkeys(held + add + ["SPY"]))
        md = fetch_yahoo(want, start, os.path.join(cache, "holdings"), refresh)
    if "SPY" in md.tickers:
        cfg["regime_asset"] = "SPY"
    prices = {t: float(md.close[t].dropna().iloc[-1]) for t in md.tickers if md.close[t].notna().any()}
    w_now, equity, cash_w, notes = positions_to_weights(pos, kind, prices, capital)
    say(f"  {len(w_now)} holdings, equity ${equity:,.0f}, cash {cash_w:.1%}")
    prep = Prep(md, cfg, verbose=verbose)
    r = optimize_holdings(md, cfg, prep, w_now, equity, views, vol_target, max_turnover, tau, prior_sharpe, extra=add)
    r["notes"] = notes + r["notes"]

    rows = []
    for q in r["portfolios"]:
        for t, w0, w1 in zip(r["tickers"], r["w0"], q["w"]):
            px = r["prices"][t]
            rows.append({"portfolio": q["name"], "ticker": t, "w_now": w0, "w_new": w1, "price": px,
                         "shares_now": w0 * equity / px, "shares_new": w1 * equity / px,
                         "trade": round(w1 * equity / px) - round(w0 * equity / px)})
    trades = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    trades.to_csv(os.path.join(out_dir, "optimizer_trades.csv"), index=False)
    bl = trades[trades["portfolio"] == "Black-Litterman"].copy()
    bl = bl[(bl["trade"] != 0) | (bl["w_now"].abs() > 1e-6)]
    path = build_opt_report(r, cfg, os.path.join(out_dir, "optimizer.html"),
                            {"source": md.source, "synthetic": md.synthetic}, bl)
    if copy_to_viz:
        from .vizexport import VIZ_DIR
        rdir = os.path.join(viz_dir or VIZ_DIR, "reports")
        os.makedirs(rdir, exist_ok=True)
        shutil.copy2(path, os.path.join(rdir, "optimizer.html"))
    return {"path": path, "synthetic": md.synthetic, "equity": equity, "notes": r["notes"],
            "portfolios": [{k: q[k] for k in ("name", "vol", "var99", "es975", "sharpe", "turnover")}
                           for q in r["portfolios"]]}
