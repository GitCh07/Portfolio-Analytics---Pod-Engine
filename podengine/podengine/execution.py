"""Turn target weights into an order sheet: ETF shares, and the equivalent
futures contracts where a liquid contract exists."""
import numpy as np
import pandas as pd

from .universe import FUTURES


def hedge_ratio(etf_close, fut_close, default, window=250):
    """Futures notional needed per $1 of ETF exposure, estimated as the beta of
    ETF returns on futures returns over the last year (falls back to the table
    default when data are thin or the estimate is implausible)."""
    try:
        df = pd.concat([etf_close, fut_close], axis=1, join="inner").dropna().iloc[-window:]
        if len(df) < 120:
            return default, "default"
        r = df.pct_change().dropna()
        # 2-day returns reduce the effect of different closing times
        r2 = r.rolling(2).sum().dropna()
        b = float(np.cov(r2.iloc[:, 0], r2.iloc[:, 1])[0, 1] / r2.iloc[:, 1].var())
        if not np.isfinite(b) or not 0.3 <= b <= 3.0:
            return default, "default"
        return b, "estimated"
    except Exception:
        return default, "default"


def order_sheet(res, capital, futures_prices=None, current=None, use_futures=True):
    """Table of target positions for an account of `capital` dollars.

    `current`: optional {ticker: shares} already held, to show the trade.
    Futures alternative: the largest contract that still gives at least two
    contracts, otherwise the smallest (micro) contract; rounded to whole
    contracts, with the rounding error shown."""
    f, pr = res.final, res.prep
    md = pr.md
    futures_prices = futures_prices or {}
    current = current or {}
    rows = []
    for a in f["avail"]:
        t = pr.tickers[a]
        w = float(f["W"][a])
        px = float(md.close[t].dropna().iloc[-1])
        dollars = w * capital
        shares = int(np.round(dollars / px))
        held = float(current.get(t, 0.0))
        if abs(w) < 1e-4 and held == 0:
            continue
        row = {"ticker": t, "name": pr.meta_list[a]["name"], "class": pr.meta_list[a]["cls"],
               "weight": w, "dollars": dollars, "price": px, "shares": shares,
               "trade_shares": shares - held, "futures": "", "contracts": None, "fut_notional": None,
               "rounding_gap": None, "ratio_source": ""}
        spec = FUTURES.get(t)
        if use_futures and spec and t in futures_prices and len(futures_prices[t]):
            fpx = float(futures_prices[t].iloc[-1])
            ratio, src = hedge_ratio(md.close[t], futures_prices[t], spec["ratio"])
            want = dollars * ratio
            best = None
            for sym, mult in sorted(spec["contracts"].items(), key=lambda kv: -kv[1]):
                n = want / (fpx * mult)
                best = (sym, mult, n)
                if abs(n) >= 2:
                    break
            sym, mult, n = best
            n_round = int(np.round(n))
            row.update({"futures": sym, "contracts": n_round, "fut_notional": n_round * fpx * mult,
                        "rounding_gap": (n_round - n) * fpx * mult / capital, "ratio_source": f"{ratio:.2f} ({src})"})
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.reindex(df["weight"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df
