"""Runs the whole thing and gathers every number the report needs."""
import copy
import os
import time

import numpy as np
import pandas as pd

from . import analytics as an
from . import montecarlo as mc
from . import validate as V
from . import var as VR
from .backtest import Prep, run_backtest
from .execution import order_sheet

ANN = 252


def _turnover(res):
    H = res.held[res.start:]
    yrs = max(len(H) / ANN, 1e-9)
    return float(np.abs(np.diff(H, axis=0)).sum() / yrs)


def _row(label, res, base=None):
    x = res.excess[res.start:]
    st = an.perf_stats(res.returns, res.prep.md.rf)
    row = {"variant": label, "sharpe": V.sharpe(x), "vol": st.get("vol"), "cagr": st.get("cagr"),
           "max_dd": st.get("max_dd"), "turnover": _turnover(res)}
    if base is not None:
        row["d_sharpe"] = row["sharpe"] - base["sharpe"]
    return row


def full_run(md, cfg, capital=100000.0, quick=False, state_dir="state", futures_prices=None,
             current=None, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    os.makedirs(state_dir, exist_ok=True)
    trial_path = os.path.join(state_dir, "trials.jsonl")

    say("[1/5] Preparing risk models")
    prep = Prep(md, cfg, verbose)
    say("[2/5] Walk-forward backtest")
    res = run_backtest(md, cfg, prep)
    if not res.final or res.start >= prep.T - 260:
        raise RuntimeError("Not enough history: need roughly 3+ years of daily prices for at least 3 assets.")
    x = res.excess[res.start:]
    idx = res.dates[res.start:]
    ret = res.returns
    base_row = _row("baseline", res)
    V.log_trial(trial_path, md, cfg, base_row["sharpe"], "baseline")

    # ---------------------------------------------------------- validation
    rows, pbo, cols = [base_row], {"pbo": float("nan"), "n_splits": 0}, {}
    if not quick:
        say("[3/5] Robustness grid (ablations, cost stress, overfitting test)")
        cols["baseline"] = pd.Series(x, index=idx)
        for label, over, selectable in V.variants(cfg)[1:]:
            c2 = copy.deepcopy(cfg)
            c2.update(over)
            r2 = run_backtest(md, c2, prep)
            rows.append(_row(label, r2, base_row))
            V.log_trial(trial_path, md, c2, rows[-1]["sharpe"], label)
            if selectable:
                cols[label] = pd.Series(r2.excess[r2.start:], index=r2.dates[r2.start:])
            say(f"      {label:32s} Sharpe {rows[-1]['sharpe']:+.2f}")
        M = pd.DataFrame(cols).dropna()
        pbo = V.pbo_cscv(M.to_numpy())
    else:
        say("[3/5] Robustness grid skipped (--quick)")
    trials = V.read_trials(trial_path, md)
    dsr, hurdle, dsr_detail = V.deflated_sharpe(x, len(trials), list(trials.values()))
    lo, hi = V.stationary_bootstrap_ci(x, seed=cfg["seed"])
    validation = {"sharpe": base_row["sharpe"], "psr": V.psr(x), "dsr": dsr, "hurdle": hurdle, "dsr_detail": dsr_detail,
                  "n_trials": len(trials), "ci": (lo, hi), "mtrl": V.min_track_record_years(x),
                  "pbo": pbo, "ablation": rows}

    # ---------------------------------------------------------- Monte Carlo
    say("[4/5] Monte Carlo and range calibration")
    f = res.final
    avail, w = f["avail"], f["W"][f["avail"]]
    drift = f["sr_port"] * f["ex_ante_vol"]
    cal = {"port21": mc.calibrate(res, cfg, 21, 1000), "port5": mc.calibrate(res, cfg, 5, 600),
           "bench21": mc.calibrate(res, cfg, 21, 1000, benchmark=True)}
    ranges = {}
    rng = np.random.default_rng(cfg["seed"])
    if np.abs(w).sum() > 0:
        for h, key in ((1, None), (5, "port5"), (21, "port21")):
            sim = mc.simulate_holdings(prep, prep.T - 1, avail, w, cfg, h, cfg["mc_paths"], rng, drift)
            lv = cal[key].get("levels") if key and cal[key].get("n", 0) >= 30 else None
            r = mc.corrected_range(sim["total"], lv)
            q = np.quantile(sim["total"], 0.025)
            r["es_97_5"] = float(-sim["total"][sim["total"] <= q].mean())
            r["p_loss"] = float((sim["total"] < 0).mean())
            ranges[h] = r
    strat = mc.simulate_strategy(res, cfg, 252)
    wl = strat["wealth"]
    steps = list(range(4, 252, 5)) + [251]
    fan = {"days": [s + 1 for s in steps],
           "q": {str(q): [float(v) for v in np.quantile(wl[:, steps], q / 100, axis=0)] for q in (5, 25, 50, 75, 95)}}
    L, g = mc.kelly_leverage(strat["daily"], float(md.rf.iloc[-1]))
    mcout = {"cal": cal, "ranges": ranges, "fan": fan,
             "p_loss_1y": float((wl[:, -1] < 1).mean()),
             "p_dd": {d: float((strat["max_dd"] < -d).mean()) for d in (0.10, 0.20, 0.30)},
             "kelly_full": L, "kelly_growth": g, "kelly_gauss": strat["sharpe_used"] / strat["vol_now"],
             "sharpe_used": strat["sharpe_used"], "vol_now": strat["vol_now"]}

    say("      Value at Risk and its backtest")
    var_out = VR.engine_var(res, cfg)

    # ---------------------------------------------------------- analytics
    say("[5/5] Analytics and order sheet")
    bench = an.benchmarks(md, idx)
    stats = {"Engine": an.perf_stats(ret, md.rf)}
    stats.update({k: an.perf_stats(v, md.rf) for k, v in bench.items()})
    series = {"Engine": ret, **bench}
    yearly = pd.DataFrame({k: an.yearly_returns(v) for k, v in series.items()})
    wk = slice(None, None, 5)
    curves = {k: (1 + v.fillna(0.0)).cumprod().iloc[wk] for k, v in series.items()}
    pods_cum = res.pod_frame().cumsum().iloc[wk]
    contrib = res.contrib_frame()
    reb = pd.DataFrame([{"date": r["date"], "target": r["target"], "ex_ante_vol": r["ex_ante_vol"],
                         "gross": r["gross"], "net": r["net"], "turb": r["turb"]} for r in res.rebalances])
    gov = pd.DataFrame([r["gov"] for r in res.rebalances], columns=res.pod_names,
                       index=[r["date"] for r in res.rebalances])
    k = prep.fit_at(prep.T - 1)
    par = prep.reg_fits[k]["params"] if k >= 0 else None
    regime = {"probs": [float(v) for v in f["regime"]],
              "vols": [float(s * np.sqrt(ANN)) for s in par["sigma"]] if par else [],
              "bench": prep.bench, "turb": f["turb"],
              "stay": [float(v) for v in np.diag(par["A"])] if par else []}
    costs = {"trading": float(res.cost[res.start:].mean() * ANN), "financing": float(res.financing[res.start:].mean() * ANN),
             "turnover": _turnover(res), "avg_gross": float(res.gross[res.start:].mean()),
             "avg_net": float(res.held[res.start:].sum(axis=1).mean())}
    orders = order_sheet(res, capital, futures_prices, current)

    # live journal: what the engine recommended on each run date
    jpath = os.path.join(state_dir, "journal.csv")
    jr = pd.DataFrame({"date": str(res.dates[-1].date()), "ticker": [prep.tickers[a] for a in avail], "weight": w})
    jr = jr[jr["weight"].abs() > 1e-4]
    if os.path.exists(jpath):
        old = pd.read_csv(jpath)
        jr = pd.concat([old[old["date"] != str(res.dates[-1].date())], jr], ignore_index=True)
    jr.to_csv(jpath, index=False)

    return {
        "res": res, "prep": prep, "cfg": cfg, "capital": capital, "quick": quick,
        "info": {"asof": res.dates[-1], "source": md.source, "synthetic": md.synthetic,
                 "n_assets": len(avail), "n_universe": prep.N, "first_trade": res.dates[res.start],
                 "years": len(x) / ANN, "elapsed": time.time() - t0},
        "stats": stats, "yearly": yearly, "curves": curves, "pods_cum": pods_cum,
        "pod_table": an.pod_table(res), "exposure": an.exposure_summary(res),
        "contrib_ann": (contrib.mean() * ANN).to_dict(),
        "stress": an.stress_table(series), "worst_dd": an.worst_drawdowns(ret),
        "dd": an.drawdown(ret).iloc[wk], "reb": reb, "gov_avg": gov.mean().to_dict(),
        "validation": validation, "mc": mcout, "regime": regime, "costs": costs,
        "vol_diag": prep.vol_diag, "orders": orders, "final": f, "var": var_out,
    }
