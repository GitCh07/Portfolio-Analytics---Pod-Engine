"""Tests. Run with:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from podengine import montecarlo as mc
from podengine import validate as V
from podengine.allocator import (DrawdownGovernor, apply_limits, golden_section_max,
                                 posterior_sharpe, risk_budget_weights, turbulence_scalar)
from podengine.backtest import Prep, _trade_to_band, run_backtest
from podengine.config import load_config
from podengine.cov import clean_corr, risk_contributions, rmt_clip
from podengine.data import MarketData, _split_yahoo, make_synthetic
from podengine.regime import fit_hmm, hmm_filter
from podengine.vol import forecast_vol

SMALL = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLU", "IEF", "TLT", "LQD", "GLD", "DBC", "FXE"]


@pytest.fixture(scope="module")
def small():
    md = make_synthetic("2008-01-01", "2019-12-31", seed=3, tickers=SMALL)
    cfg = load_config()
    res = run_backtest(md, cfg)
    return md, cfg, res


# ------------------------------------------------------------------ no look-ahead

def test_no_lookahead(small):
    """Positions held on any past day must not change when future data is removed."""
    md, cfg, res = small
    cut = md.close.index[-400]
    res2 = run_backtest(md.truncate(cut), cfg)
    n = len(res2.dates) - 15
    assert n > 1500
    np.testing.assert_allclose(res.held[:n], res2.held[:n], atol=1e-9)
    np.testing.assert_allclose(res.ret[:n], res2.ret[:n], atol=1e-10)


def _target_mismatch(md, cfg, res, n_dates=4):
    """Re-run on data cut at several past rebalance dates; the target decided on
    that date (and the Monte Carlo inputs) must match the full run exactly."""
    from podengine.montecarlo import _inputs
    recs = [r for r in res.rebalances if r["p"] > 900]
    worst = 0.0
    for rec in recs[:: max(1, len(recs) // n_dates)][:n_dates]:
        p = rec["p"]
        r2 = run_backtest(md.truncate(res.dates[p]), cfg)
        worst = max(worst, float(np.abs(rec["W"] - r2.final["W"]).max()))
        avail = r2.final["avail"]
        a, b = _inputs(res.prep, p, avail, cfg), _inputs(r2.prep, p, avail, cfg)
        for x, y in zip(a, b):
            worst = max(worst, float(np.abs(np.asarray(x) - np.asarray(y)).max()))
    return worst


def test_targets_match_when_future_is_deleted(small):
    md, cfg, res = small
    assert _target_mismatch(md, cfg, res) < 1e-9


def test_lookahead_test_catches_a_two_day_peek(small, monkeypatch):
    """Negative control: a trend signal that peeks two days ahead must be caught."""
    from podengine.pods import TrendPod
    md, cfg, _ = small
    orig = TrendPod.prepare

    def peek(self, eng):
        orig(self, eng)
        self.sig = np.vstack([self.sig[2:], np.full((2, self.sig.shape[1]), np.nan)])

    monkeypatch.setattr(TrendPod, "prepare", peek)
    res = run_backtest(md, cfg)
    assert _target_mismatch(md, cfg, res) > 1e-3


def test_vol_forecast_is_causal():
    md = make_synthetic("2010-01-01", "2017-12-31", seed=5, tickers=SMALL[:6])
    v1, _ = forecast_vol(md)
    cut = md.close.index[-300]
    v2, _ = forecast_vol(md.truncate(cut))
    np.testing.assert_allclose(v1.loc[:cut].to_numpy(), v2.to_numpy(), rtol=1e-9, equal_nan=True)


def test_hmm_filter_is_causal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 1500)
    p = fit_hmm(x[:1000], 2)
    a, b = hmm_filter(x, p), hmm_filter(x[:1200], p)
    np.testing.assert_allclose(a[:1200], b, atol=1e-12)


# ------------------------------------------------------------------ maths

def test_risk_budget_weights_hit_budgets():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    S = A @ A.T + 0.5 * np.eye(6)
    b = np.array([1, 2, 3, 1, 0, 2.0])
    w = risk_budget_weights(S, b)
    rc, _ = risk_contributions(w, S)
    np.testing.assert_allclose(rc / rc.sum(), b / b.sum(), atol=1e-6)
    assert w[4] == 0 and abs(w.sum() - 1) < 1e-12 and (w >= 0).all()


def test_golden_section_and_kelly():
    x, fx = golden_section_max(lambda v: -(v - 1.2345) ** 2, 0, 5)
    assert abs(x - 1.2345) < 1e-3
    rng = np.random.default_rng(2)
    mu, sd = 0.0004, 0.01
    r = rng.normal(mu, sd, 400000)
    L, _ = mc.kelly_leverage(r)
    assert abs(L - r.mean() / r.var()) / (mu / sd ** 2) < 0.08     # Gaussian case: f* = mu / sigma^2


def test_posterior_sharpe_shrinks():
    m, s = posterior_sharpe(30, 0.01, 0.01, 0.3, 0.25)             # too little data: prior stands
    assert (m, s) == (0.3, 0.25)
    m2, s2 = posterior_sharpe(252 * 20, 0.0008, 0.01, 0.0, 0.35)   # 20 years at Sharpe ~1.27
    assert 0.6 < m2 < 1.27 and s2 < 0.35


def test_governor_states():
    g = DrawdownGovernor(0.15, 0.225)
    assert g.update(0.05) == 1.0
    assert g.update(0.16) == 0.5
    assert g.update(0.10) == 0.5          # hysteresis: stays halved until well recovered
    assert g.update(0.07) == 1.0
    assert g.update(0.30) == 0.0
    assert g.update(0.20) == 0.0          # stays off above the soft line
    assert g.update(0.14) == 0.5
    assert turbulence_scalar(0.5) == 1.0 and abs(turbulence_scalar(1.0, 0.75, 0.6) - 0.6) < 1e-12


def test_apply_limits():
    S = np.diag([0.04, 0.01, 0.0025])
    cfg = load_config(profile="cash")
    w, info = apply_limits(np.array([0.5, -0.3, 0.4]), S, 0.10, cfg)
    assert (w >= 0).all() and np.abs(w).sum() <= 1.0 + 1e-9 and (np.abs(w) <= cfg["asset_cap"] + 1e-9).all()
    cfg = load_config(profile="unconstrained")
    w, info = apply_limits(np.array([0.5, -0.3, 0.4]), S, 0.10, cfg)
    assert abs(info["ex_ante_vol"] - 0.10) < 1e-9 and w[1] < 0


def test_trade_band():
    held, target = np.array([1.0, 1.0, 0.0, -1.0]), np.array([1.1, 1.5, 0.5, 1.0])
    out = _trade_to_band(held, target, 0.2)
    np.testing.assert_allclose(out, [1.0, 1.2, 0.4, 0.8])
    np.testing.assert_allclose(_trade_to_band(held, np.zeros(4), 0.2), 0)        # exits are always completed


def test_rmt_clip_keeps_valid_correlation():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(300, 20))
    z[:, :10] += rng.normal(size=(300, 1))
    C = clean_corr(z, 126, 0.1)
    assert np.allclose(np.diag(C), 1) and np.linalg.eigvalsh(C).min() > 0
    assert np.allclose(rmt_clip(np.eye(5), 100), np.eye(5))


def test_hmm_recovers_two_regimes():
    rng = np.random.default_rng(5)
    s, out = 0, []
    for _ in range(4000):
        s = s if rng.random() < (0.98 if s == 0 else 0.95) else 1 - s
        out.append(rng.normal(0, 0.006 if s == 0 else 0.02))
    p = fit_hmm(np.array(out), 2)
    assert 0.004 < p["sigma"][0] < 0.008 and 0.015 < p["sigma"][1] < 0.026


# ------------------------------------------------------------------ validation statistics

def test_psr_dsr():
    rng = np.random.default_rng(6)
    noise = rng.normal(0, 0.01, 5000)
    assert 0.02 < V.psr(noise - noise.mean()) < 0.98 and abs(V.psr(noise - noise.mean()) - 0.5) < 0.02
    good = rng.normal(0.0006, 0.01, 5000)
    assert V.psr(good) > 0.99
    d1, h1, _ = V.deflated_sharpe(good, 1)
    d100, h100, det = V.deflated_sharpe(good, 100, list(np.linspace(0.1, 0.2, 100)))
    assert h100 == max(det["hurdle_trials"], det["hurdle_zero_skill"])
    assert h1 == 0 and h100 > 0.3 and d100 < d1
    assert V.expected_max_sharpe(1000, 0.2) > V.expected_max_sharpe(10, 0.2)


def test_pbo_noise_vs_skill():
    rng = np.random.default_rng(7)
    # pure noise: picking the in-sample best is a coin flip on average (one sample is noisy, so average a few)
    avg = np.mean([V.pbo_cscv(np.random.default_rng(s).normal(0, 0.01, (2500, 12)))["pbo"] for s in range(12)])
    assert 0.35 < avg < 0.65
    noise = rng.normal(0, 0.01, (2500, 12))
    skill = noise.copy()
    skill[:, 0] += 0.001
    assert V.pbo_cscv(skill)["pbo"] < 0.1


def test_bootstrap_ci_brackets_truth():
    rng = np.random.default_rng(8)
    x = rng.normal(0.0005, 0.01, 4000)
    lo, hi = V.stationary_bootstrap_ci(x, n_boot=500)
    assert lo < V.sharpe(x) < hi and hi - lo < 1.5


def test_adaptive_conformal_restores_coverage():
    """A forecaster whose ranges are far too narrow still ends up near 5% misses."""
    rng = np.random.default_rng(9)
    aci = mc.ACI(0.05, 0.02)
    sims = rng.normal(0, 0.7, 5000)                                # forecast ranges are 30% too narrow
    raw, adj = [], []
    for _ in range(4000):
        y = rng.normal(0, 1.0)
        raw.append(y < np.quantile(sims, 0.05))
        miss = y < np.quantile(sims, aci.level)
        adj.append(miss)
        aci.update(miss)
    assert np.mean(raw) > 0.10 and abs(np.mean(adj) - 0.05) < 0.01


# ------------------------------------------------------------------ data plumbing

def test_yahoo_frame_layouts():
    idx = pd.date_range("2024-01-01", periods=3)
    a = pd.DataFrame(1.0, index=idx, columns=pd.MultiIndex.from_product([["SPY", "TLT"], ["Open", "High", "Low", "Close"]]))
    b = pd.DataFrame(1.0, index=idx, columns=pd.MultiIndex.from_product([["Open", "High", "Low", "Close"], ["SPY", "TLT"]]))
    c = pd.DataFrame(1.0, index=idx, columns=["Open", "High", "Low", "Close"])
    for df, tk in ((a, ["SPY", "TLT"]), (b, ["SPY", "TLT"]), (c, ["SPY"])):
        out = _split_yahoo(df, tk)
        assert list(out["close"].columns) == tk and set(out) == {"open", "high", "low", "close"}


def test_closes_only_data_runs():
    md0 = make_synthetic("2010-01-01", "2018-12-31", seed=2, tickers=SMALL)
    md = MarketData(md0.close, rf=md0.rf, meta=md0.meta, source="closes only")
    res = run_backtest(md, load_config(profile="margin"))
    trade_days = res.cost > 0
    assert np.isfinite(res.ret).all() and res.gross[trade_days].max() <= 2.0 + 1e-6
    assert res.gross.max() <= 2.0 * 1.10            # prices can drift a little between trades
    assert res.final and abs(res.final["W"]).sum() > 0


# ------------------------------------------------------------------ engine behaviour

def test_accounting_and_limits(small):
    md, cfg, res = small
    assert np.isfinite(res.ret).all() and (res.cost >= 0).all() and (res.financing >= 0).all()
    assert res.gross[res.cost > 0].max() <= cfg["gross_cap"] + 1e-6
    x = res.excess[res.start:]
    assert 0.03 < x.std() * np.sqrt(252) < 0.20                      # in the neighbourhood of the 12% target
    # zero-cost run must not be worse than the costed one
    free = run_backtest(md, load_config(cost_mult=0.0, financing_spread=0.0, short_fee=0.0), res.prep)
    assert free.excess.sum() > res.excess.sum()


def test_attribution_adds_up(small):
    md, cfg, res = small
    ex = np.nan_to_num(res.prep.R) - res.prep.rf[:, None]
    gross_pnl = (res.held * ex).sum(axis=1)
    np.testing.assert_allclose(res.contrib.sum(axis=1) + res.resid, gross_pnl, atol=1e-12)


def test_weights_drift_between_trades(small):
    md, cfg, res = small
    H = res.held[res.start:]
    changed = np.abs(np.diff(H, axis=0)).sum(axis=1) > 0
    assert changed.mean() > 0.95          # weights move every day with prices, not only on trade days


def test_cash_profile_is_long_only_unlevered(small):
    md, cfg, res = small
    r = run_backtest(md, load_config(profile="cash"), res.prep)
    assert r.held.min() >= -1e-9 and r.gross[r.cost > 0].max() <= 1.0 + 1e-6 and r.gross.max() <= 1.02


def test_monte_carlo_matches_risk_model(small):
    md, cfg, res = small
    f = res.final
    sim = mc.simulate_holdings(res.prep, res.prep.T - 1, f["avail"], f["W"][f["avail"]], cfg, 21, 3000)
    ratio = sim["total"].std() / (f["ex_ante_vol"] * np.sqrt(21 / 252))
    assert 0.6 < ratio < 1.6
    st = mc.simulate_strategy(res, cfg, 126, 1000)
    assert st["wealth"].shape == (1000, 126) and (st["max_dd"] <= 0).all()


def test_full_pipeline_writes_report(small, tmp_path):
    from podengine.pipeline import full_run
    from podengine.report import build_report, save_tables
    md, cfg, _ = small
    b = full_run(md, cfg, 50000, quick=True, state_dir=str(tmp_path / "state"), verbose=False)
    path = build_report(b, str(tmp_path / "report.html"))
    save_tables(b, str(tmp_path))
    text = open(path).read()
    assert "Pod Engine report" in text and "Demo mode" in text and len(b["orders"]) > 0
    assert (tmp_path / "state" / "trials.jsonl").exists() and (tmp_path / "state" / "journal.csv").exists()
