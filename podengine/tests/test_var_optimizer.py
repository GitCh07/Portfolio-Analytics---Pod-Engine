"""Tests for the VaR module and the holdings optimiser."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from podengine import optimizer as O
from podengine import var as VR
from podengine.backtest import Prep
from podengine.config import load_config
from podengine.cov import risk_contributions
from podengine.data import make_synthetic

TICK = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLU", "IEF", "TLT", "LQD", "GLD", "DBC"]


@pytest.fixture(scope="module")
def env():
    md = make_synthetic("2008-01-01", "2019-12-31", seed=4, tickers=TICK)
    cfg = load_config(profile="margin", min_history=252)
    cfg["regime_asset"] = "SPY"
    return md, cfg, Prep(md, cfg)


# ------------------------------------------------------------------ VaR statistics

def test_kupiec_matches_hand_calculation():
    b = np.zeros(1000, bool)
    b[:20] = True                                   # 20 breaches where 10 are expected
    lr, p = VR.kupiec(b, 0.01)
    hand = -2 * (980 * np.log(0.99) + 20 * np.log(0.01) - 980 * np.log(0.98) - 20 * np.log(0.02))
    assert abs(lr - hand) < 1e-9 and abs(p - stats.chi2.sf(hand, 1)) < 1e-12
    assert VR.kupiec(np.r_[np.zeros(990), np.ones(10)].astype(bool), 0.01)[1] > 0.99


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(0)
    iid = rng.random(5000) < 0.02
    clustered = np.zeros(5000, bool)
    for s in rng.choice(4990, 25, replace=False):
        clustered[s:s + 4] = True
    assert VR.christoffersen(iid)[1] > 0.01
    assert VR.christoffersen(clustered)[1] < 1e-6


def test_traffic_light_zones():
    assert VR.traffic_light(np.r_[np.ones(4), np.zeros(246)])[1] == "green"
    assert VR.traffic_light(np.r_[np.ones(5), np.zeros(245)])[1] == "yellow"
    assert VR.traffic_light(np.r_[np.ones(10), np.zeros(240)])[1] == "red"


def test_normal_formulas():
    assert abs(VR.normal_var(1.0, 0.99) - 2.3263478740) < 1e-8
    assert abs(VR.normal_es(1.0, 0.975) - 2.3378027922) < 1e-8      # phi(q)/(1-0.975)
    x = np.random.default_rng(1).normal(0, 1, 2_000_000)
    v, e = VR.var_es(x, 0.99)
    assert abs(v - 2.326) < 0.01 and abs(e - 2.338) < 0.01


def test_component_es_sums_to_total(env):
    md, cfg, prep = env
    p = prep.T - 1
    avail = np.flatnonzero(prep.elig[p])
    w = np.linspace(0.3, -0.1, len(avail))
    comp, tot = VR.component_es(prep, p, avail, w)
    assert abs(comp.sum() - tot) < 1e-12
    Sigma = prep.sigma_at(p, avail, cfg)
    cv = VR.component_var_normal(w, Sigma)
    sd = np.sqrt(w @ Sigma @ w / 252)
    assert abs(cv.sum() - VR.normal_var(sd, 0.99)) < 1e-12


def test_var_backtest_is_causal(env):
    md, cfg, prep = env
    avail = [prep.tickers.index(t) for t in ("SPY", "TLT", "GLD")]
    W = np.zeros((prep.T, prep.N))
    W[:, avail] = [0.5, 0.4, 0.1]
    ex = np.nan_to_num(prep.R) - prep.rf[:, None]
    full = VR.backtest(prep, cfg, W, (W * ex).sum(axis=1), 600)
    cut = md.close.index[-300]
    md2 = md.truncate(cut)
    prep2 = Prep(md2, cfg)
    W2 = W[:prep2.T]
    ex2 = np.nan_to_num(prep2.R) - prep2.rf[:, None]
    part = VR.backtest(prep2, cfg, W2, (W2 * ex2).sum(axis=1), 600)
    n = part["n"]
    np.testing.assert_allclose(full["series"]["var99"][:n], part["series"]["var99"], rtol=1e-12)


def test_fhs_var_is_well_calibrated_on_engine_data(env):
    md, cfg, prep = env
    avail = [prep.tickers.index(t) for t in ("SPY", "IEF", "GLD")]
    W = np.zeros((prep.T, prep.N))
    W[:, avail] = [0.6, 0.3, 0.1]
    ex = np.nan_to_num(prep.R) - prep.rf[:, None]
    bt = VR.backtest(prep, cfg, W, (W * ex).sum(axis=1), 600)
    assert bt["methods"]["fhs_aci"][0.99]["kupiec_p"] > 0.01
    assert 0.005 < bt["methods"]["fhs"][0.99]["rate"] < 0.02


# ------------------------------------------------------------------ optimiser

def _run(env, w_now, views=None, **kw):
    md, cfg, prep = env
    return O.optimize_holdings(md, cfg, prep, w_now, 100000, views, **kw)


W_NOW = {"SPY": 0.45, "QQQ": 0.2, "XLK": 0.1, "TLT": 0.15, "GLD": 0.05}


def test_no_views_returns_current_portfolio(env):
    r = _run(env, W_NOW)
    bl = [q for q in r["portfolios"] if q["name"] == "Black-Litterman"][0]
    np.testing.assert_allclose(bl["w"], r["w0"], atol=2e-3)
    np.testing.assert_allclose(r["mu"], r["Pi"], atol=1e-12)
    cur = r["portfolios"][0]
    assert abs(cur["sharpe"] - 0.30) < 1e-6                         # reverse optimisation hits the prior Sharpe


def test_confident_view_is_respected(env):
    md, cfg, prep = env
    tick = list(W_NOW)
    avail = np.array([prep.tickers.index(t) for t in tick])
    Sigma = prep.sigma_at(prep.T - 1, avail, cfg)
    w0 = np.array(list(W_NOW.values()))
    Pi, _ = O.implied_returns(Sigma, w0)
    P, Q, Om, used, skipped = O.build_views([{"assets": {"TLT": 1, "SPY": -1}, "return": 0.05, "confidence": 0.99},
                                             {"assets": {"ZZZ": 1}, "return": 0.1}], tick, Sigma, 0.05)
    mu, _ = O.black_litterman(Pi, Sigma, P, Q, Om, 0.05)
    assert abs(P[0] @ mu - 0.05) < 0.002 and len(skipped) == 1
    mu_half, _ = O.black_litterman(Pi, Sigma, P, Q, np.diag([(1 / 0.5 - 1) * P[0] @ (0.05 * Sigma) @ P[0]]), 0.05)
    mid = P[0] @ mu_half
    assert abs(mid - 0.5 * (0.05 + P[0] @ Pi)) < 1e-9              # 50% confidence = halfway between prior and view


def test_view_moves_weights_in_its_direction(env):
    base = _run(env, W_NOW)
    r = _run(env, W_NOW, [{"assets": {"GLD": 1}, "return": 0.06, "confidence": 0.7}])
    i = r["tickers"].index("GLD")
    w_bl = [q for q in r["portfolios"] if q["name"] == "Black-Litterman"][0]["w"]
    assert w_bl[i] > base["w0"][i] + 0.02
    vol = np.sqrt(w_bl @ r["Sigma"] @ w_bl)
    assert vol <= r["target_vol"] * 1.001


def test_min_variance_and_risk_parity_properties(env):
    r = _run(env, W_NOW)
    q = {x["name"]: x for x in r["portfolios"]}
    S = r["Sigma"]
    mv = q["Minimum variance"]["w"]
    assert abs(mv.sum() - r["w0"].sum()) < 1e-6
    for name in ("Current", "Risk parity"):
        w = q[name]["w"] * r["w0"].sum() / q[name]["w"].sum()         # same net exposure for a fair comparison
        assert mv @ S @ mv <= w @ S @ w + 1e-10
    rp = q["Risk parity"]["w"]
    rc, port = risk_contributions(rp, S)
    np.testing.assert_allclose(rc / port, 1 / len(rp), atol=1e-6)


def test_cash_profile_never_shorts_or_levers(env):
    md, _, prep = env
    cfg = load_config(profile="cash", min_history=252)
    cfg["regime_asset"] = "SPY"
    r = O.optimize_holdings(md, cfg, Prep(md, cfg), W_NOW, 100000,
                            [{"assets": {"QQQ": 1, "SPY": -1}, "return": -0.08, "confidence": 0.9}])
    for q in r["portfolios"][1:]:
        assert q["w"].min() >= -1e-6 and q["gross"] <= 1.0 + 1e-6 and q["w"].max() <= cfg["asset_cap"] + 1e-6


def test_turnover_cap(env):
    r = _run(env, W_NOW, [{"assets": {"GLD": 1}, "return": 0.08, "confidence": 0.9}], max_turnover=0.10)
    bl = [q for q in r["portfolios"] if q["name"] == "Black-Litterman"][0]
    assert bl["turnover"] <= 0.10 + 1e-3


def test_positions_parsing(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("Ticker,Shares\nspy,10\nTLT,-5\nCASH,1000\n")
    pos, kind = O.read_positions(f)
    w, eq, cash, notes = O.positions_to_weights(pos, kind, {"SPY": 100.0, "TLT": 50.0})
    assert kind == "shares" and abs(eq - (1000 - 250 + 1000)) < 1e-9
    assert abs(w["SPY"] - 1000 / 1750) < 1e-12 and abs(w["TLT"] + 250 / 1750) < 1e-12
    f.write_text("ticker,weight\nSPY,0.6\nIEF,0.3\n")
    pos, kind = O.read_positions(f)
    w, eq, cash, _ = O.positions_to_weights(pos, kind, {}, capital=50000)
    assert kind == "weight" and eq == 50000 and abs(cash - 0.1) < 1e-12


def test_views_file_and_cli_demo(tmp_path):
    from podengine.__main__ import main
    vf = tmp_path / "v.json"
    vf.write_text(json.dumps([{"assets": {"TLT": 1}, "return": 0.03, "confidence": 0.5}]))
    assert O.read_views(vf)[0]["return"] == 0.03
    assert main(["optimize", "--demo", "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "optimizer.html").exists() and (tmp_path / "o" / "optimizer_trades.csv").exists()
