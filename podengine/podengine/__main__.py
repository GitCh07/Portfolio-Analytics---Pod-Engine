"""Command line.

  python -m podengine run --demo                 offline demo on simulated data
  python -m podengine run                        real prices from Yahoo Finance, full report
  python -m podengine today --capital 50000      fast daily update (skips the robustness grid)
  python -m podengine run --profile margin       2x gross, long/short (also: cash, unconstrained)
  python -m podengine run --prices my.csv        your own price file
  python -m podengine optimize --positions my_holdings.csv [--views my_views.json]
                                                 risk report and better weights for what you hold
  python -m podengine optimize --demo            optimiser demo on simulated data
  python -m podengine viz --demo                 export the animated views (viz/data/*.json)
  python -m podengine run --viz                  full run, then export the views
  python -m podengine serve                      open the single-page viewer at http://127.0.0.1:8765/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="podengine", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "today", "fetch", "optimize", "viz", "serve"], nargs="?", default="run")
    ap.add_argument("--demo", action="store_true", help="use simulated data (no internet needed)")
    ap.add_argument("--prices", help="CSV of adjusted closes: date column, then one column per ticker")
    ap.add_argument("--meta", help="JSON {ticker: {cls, group}} to classify tickers in --prices")
    ap.add_argument("--tickers", help="comma-separated Yahoo tickers to use instead of the default universe")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--profile", choices=["unconstrained", "margin", "cash"])
    ap.add_argument("--vol-target", type=float, help="annual volatility target, e.g. 0.12")
    ap.add_argument("--capital", type=float, help="account size (run: default 100,000; optimize: overrides positions total)")
    ap.add_argument("--positions", help="CSV: ticker plus shares, value or weight (a CASH row for cash)")
    ap.add_argument("--views", help="optimize: JSON file of views for Black-Litterman")
    ap.add_argument("--add", help="optimize: extra tickers the optimiser may buy, comma-separated")
    ap.add_argument("--max-turnover", type=float, help="optimize: cap on total weight traded, e.g. 0.3")
    ap.add_argument("--tau", type=float, default=0.05, help="optimize: Black-Litterman prior uncertainty")
    ap.add_argument("--prior-sharpe", type=float, default=0.30, help="optimize: Sharpe ratio assumed for your current mix")
    ap.add_argument("--config", help="JSON file overriding any default setting")
    ap.add_argument("--quick", action="store_true", help="skip the robustness grid")
    ap.add_argument("--no-futures", action="store_true", help="do not look up futures prices")
    ap.add_argument("--refresh", action="store_true", help="re-download prices even if cached today")
    ap.add_argument("--out", default="output")
    ap.add_argument("--viz", action="store_true", help="also export data for the animated viewer")
    ap.add_argument("--viz-dir", help="viewer folder (default: the viz/ folder next to the package)")
    ap.add_argument("--port", type=int, default=8765, help="serve: port")
    ap.add_argument("--no-browser", action="store_true", help="serve: don't open a browser")
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--state", default="state")
    a = ap.parse_args(argv)

    from .config import load_config
    from .data import fetch_futures, fetch_yahoo, load_csv, make_synthetic
    from .pipeline import full_run
    from .report import build_report, save_tables

    if a.command == "optimize":
        return run_optimize(a)
    if a.command == "serve":
        return run_serve(a)
    cfg = load_config(a.config, profile=a.profile, vol_target=a.vol_target)
    tickers = [t.strip().upper() for t in a.tickers.split(",")] if a.tickers else None
    print("Loading data ...")
    if a.demo:
        md = make_synthetic(seed=11)
    elif a.prices:
        md = load_csv(a.prices, a.meta)
    else:
        md = fetch_yahoo(tickers, a.start, a.cache, a.refresh)
    print(f"  {md.close.shape[1]} assets, {md.close.index[0].date()} to {md.close.index[-1].date()} ({md.source})")
    if a.command == "fetch":
        return 0
    if a.command == "viz":
        from .backtest import run_backtest
        from .vizexport import export_viz
        res = run_backtest(md, cfg, verbose=True)
        export_viz(res, cfg, a.viz_dir, reports=[os.path.join(a.out, "report.html"),
                                                 os.path.join(a.out, "optimizer.html")], quick=a.quick)
        print("Open it with:  python -m podengine serve")
        return 0

    fut = {}
    if not (a.demo or a.prices or a.no_futures):
        fut = fetch_futures(md.tickers, a.cache)
    current = None
    if a.positions:
        from .optimizer import read_positions
        pos, kind = read_positions(a.positions)
        if kind == "shares":
            current = dict(zip(pos["ticker"], pos["shares"].astype(float)))
        else:
            print("  note: --positions for `run` needs a shares column to show trades; ignoring it")

    state = a.state + ("_demo" if a.demo else "")
    b = full_run(md, cfg, a.capital or 100000.0, quick=(a.quick or a.command == "today"), state_dir=state,
                 futures_prices=fut, current=current)
    os.makedirs(a.out, exist_ok=True)
    path = build_report(b, os.path.join(a.out, "report.html"))
    save_tables(b, a.out)
    if a.viz:
        from .vizexport import export_viz
        export_viz(b["res"], cfg, a.viz_dir, reports=[path, os.path.join(a.out, "optimizer.html")], quick=a.quick)

    st, v, f = b["stats"]["Engine"], b["validation"], b["final"]
    print("\n" + "=" * 64)
    if md.synthetic:
        print(" DEMO ON SIMULATED DATA - numbers say nothing about real markets")
    print(f" Backtest {b['info']['years']:.1f} yrs: return {st['cagr']:.1%}/yr, vol {st['vol']:.1%}, "
          f"Sharpe {st['sharpe']:.2f}, max drawdown {st['max_dd']:.1%}")
    print(f" Deflated Sharpe {v['dsr']:.0%} over {v['n_trials']} logged trials; "
          f"Sharpe 90% interval {v['ci'][0]:.2f} to {v['ci'][1]:.2f}")
    r = b["mc"]["ranges"].get(21)
    if r:
        lo, hi = r.get("adj90", r["raw90"])
        print(f" Today: expected vol {f['ex_ante_vol']:.1%}, gross {f['gross']:.2f}x, net {f['net']:+.2f}x; "
              f"1-month 90% range {lo:.1%} to {hi:.1%}")
    print(f" Report: {os.path.abspath(path)}")
    print("=" * 64)
    return 0


def run_optimize(a):
    from .optimizer import read_positions, read_views
    from .optrun import DEMO_POSITIONS, DEMO_VIEWS, optimize_and_report

    if a.positions:
        pos, kind = read_positions(a.positions)
    elif a.demo:
        pos, kind = DEMO_POSITIONS.copy(), "value"
    else:
        raise SystemExit("optimize needs --positions your_holdings.csv (or --demo)")
    if a.views:
        views = read_views(a.views)
    else:
        views = DEMO_VIEWS if (a.demo and not a.positions) else None
    out = optimize_and_report(pos, kind, views, profile=a.profile, vol_target=a.vol_target,
                              max_turnover=a.max_turnover, capital=a.capital,
                              add=a.add.split(",") if a.add else (), tau=a.tau, prior_sharpe=a.prior_sharpe,
                              demo=a.demo, prices_csv=a.prices, meta=a.meta, start=a.start, cache=a.cache,
                              refresh=a.refresh, config=a.config, out_dir=a.out, viz_dir=a.viz_dir,
                              copy_to_viz=a.viz)
    print("\n" + "=" * 64)
    if out["synthetic"]:
        print(" DEMO ON SIMULATED DATA - numbers say nothing about real markets")
    for q in out["portfolios"]:
        print(f" {q['name']:17s} vol {q['vol']:6.1%}  VaR99 {q['var99']:5.2%}  ES {q['es975']:5.2%}  "
              f"model Sharpe {q['sharpe']:5.2f}  turnover {q['turnover']:5.0%}")
    print(f" Report: {os.path.abspath(out['path'])}")
    print("=" * 64)
    return 0


def run_serve(a):
    from .server import serve
    return serve(viz_dir=a.viz_dir, port=a.port, open_browser=not a.no_browser, demo=a.demo, cache=a.cache)


if __name__ == "__main__":
    sys.exit(main())
