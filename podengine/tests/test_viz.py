"""Tests for the animated viewer: export maths, schema, engine isolation, browser behaviour."""
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading

import numpy as np
import pytest
from scipy import stats

from podengine import vizexport as VX
from podengine.backtest import run_backtest
from podengine.config import load_config
from podengine.data import make_synthetic
from podengine.vizschema import validate_dir

sys.path.insert(0, os.path.dirname(__file__))
from browser import chrome_path, run_page  # noqa: E402

TICK = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLU", "EFA", "EEM", "IEF", "TLT", "LQD", "GLD", "DBC", "FXE"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Static viewer files + a fresh export into a temporary folder."""
    d = tmp_path_factory.mktemp("viz")
    shutil.copytree(VX.VIZ_DIR, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("reports", "*.json"))
    shutil.copy2(os.path.join(VX.VIZ_DIR, "data", "schema.json"), d / "data" / "schema.json")
    md = make_synthetic("2006-01-01", "2016-12-31", seed=8, tickers=TICK)
    cfg = load_config()
    res = run_backtest(md, cfg)
    before = (res.held.copy(), res.ret.copy(), [r["W"].copy() for r in res.rebalances])
    VX.export_viz(res, cfg, str(d), quick=True, verbose=False)
    return d, md, cfg, res, before


# ------------------------------------------------------------------ engine isolation

def test_export_leaves_engine_untouched(exported):
    d, md, cfg, res, (held, ret, W) = exported
    assert np.array_equal(held, res.held) and np.array_equal(ret, res.ret)
    assert all(np.array_equal(a, r["W"]) for a, r in zip(W, res.rebalances))
    fresh = run_backtest(md, cfg)                       # same answer as a run that never exported
    assert np.array_equal(fresh.held, res.held) and np.array_equal(fresh.ret, res.ret)


def test_engine_run_does_not_load_viz_code():
    code = ("import sys, podengine.__main__, podengine.pipeline, podengine.report, podengine.optimizer; "
            "print(any(m.startswith('podengine.viz') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(VX.VIZ_DIR)).stdout.strip()
    assert out == "False"


# ------------------------------------------------------------------ schema

def test_all_exports_validate(exported):
    d = exported[0]
    for use in (True, False):                           # jsonschema package and the built-in validator
        errs = validate_dir(str(d), use_jsonschema=use)
        assert set(errs) == {"meta.json", "fan.json", "calibration.json", "regimes.json", "frontier.json", "exposure.json"}
        assert all(v == [] for v in errs.values()), errs


def test_validator_catches_errors(exported, tmp_path):
    d = exported[0]
    shutil.copytree(d / "data", tmp_path / "data")
    p = tmp_path / "data" / "calibration.json"
    obj = json.loads(p.read_text())
    obj["frames"][0]["flag"] = "maybe"
    obj["frames"][1]["counts"][0] += 1                  # counts no longer sum to n
    p.write_text(json.dumps(obj))
    for use in (True, False):
        errs = validate_dir(str(tmp_path), use_jsonschema=use)["calibration.json"]
        assert any("maybe" in e for e in errs) and any("sum to n" in e for e in errs)


# ------------------------------------------------------------------ PIT maths

def _pits(scale, n=3000, m=2000, dist="normal", seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sig, mu = rng.uniform(0.5, 3.0), rng.normal(0, 1)
        draw = (lambda k, s: rng.standard_t(4, k) * s) if dist == "t" else (lambda k, s: rng.normal(0, s, k))
        forecast = mu + draw(m, sig * scale)
        outcome = mu + draw(1, sig)[0]
        out.append(VX.pit_value(forecast, outcome))
    return np.array(out)


@pytest.mark.parametrize("dist", ["normal", "t"])
def test_pit_uniform_when_model_is_correct(dist):
    u = _pits(1.0, dist=dist)
    assert stats.kstest(u, "uniform").pvalue > 0.01
    s = VX.pit_summary(u)
    assert s["flag"] == "ok" and abs(s["tail_share"] - 0.2) < 0.03
    assert max(s["counts"]) / min(s["counts"]) < 1.35


def test_pit_flags_wrong_tails():
    assert VX.pit_summary(_pits(0.6, n=800))["flag"] == "thin"    # forecast too narrow -> outcomes in the tails
    assert VX.pit_summary(_pits(1.6, n=800))["flag"] == "fat"     # forecast too wide -> tails rarely reached
    assert VX.pit_summary(_pits(1.0, n=10))["flag"] == "insufficient"


def test_exported_pit_matches_fan(exported):
    d = exported[0]
    fan = json.loads((d / "data" / "fan.json").read_text())
    cal = json.loads((d / "data" / "calibration.json").read_text())
    assert [f["pit"] for f in fan["frames"]] == [f["pit"] for f in cal["frames"]]
    last = cal["frames"][-1]
    ref = VX.pit_summary([f["pit"] for f in fan["frames"]])
    assert last["counts"] == ref["counts"] and last["flag"] == ref["flag"]


# ------------------------------------------------------------------ frontier

def test_ledoit_wolf_matches_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    X = np.random.default_rng(3).normal(size=(120, 25)) @ np.random.default_rng(4).normal(size=(25, 25)) * 0.01
    S, a = VX.ledoit_wolf(X)
    S2, a2 = sk.ledoit_wolf(X)
    assert abs(a - a2) < 1e-10 and np.allclose(S, S2, atol=1e-14)


def _cfg(short=True):
    return {"asset_cap": 1.0, "gross_cap": 3.0, "allow_short": short}


def test_frontier_respects_engine_constraints():
    rng = np.random.default_rng(5)
    X = rng.normal(0.0004, 0.01, (252, 12))
    S, _ = VX.ledoit_wolf(X)
    mu, _ = VX.bayes_stein_means(X, S)
    for short in (True, False):
        pts = VX.frontier_curve(mu * 252, S * 252, _cfg(short), np.linspace(0.02, 0.3, 8))
        for (v, m, w), target in zip(pts, np.linspace(0.02, 0.3, 8)):
            assert v <= target + 1e-6 and np.abs(w).sum() <= 3.0 + 1e-6 and np.abs(w).max() <= 1.0 + 1e-6
            assert short or w.min() >= -1e-8
        rets = [m for _, m, _ in pts]
        assert all(b >= a - 1e-9 for a, b in zip(rets, rets[1:]))


def _stability(N, W, trials=12, seed=6):
    """Average weight change of the 10%-volatility frontier portfolio when the window moves
    10 days, and how far that portfolio's TRUE volatility is from the 10% the estimate
    promised, for Ledoit-Wolf vs the sample covariance. True covariance is known."""
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(N, 3)) * 0.006
    true = (L @ L.T + 0.008 ** 2 * np.eye(N)) * 252
    out = {"lw": ([], []), "sample": ([], [])}
    for _ in range(trials):
        R = rng.normal(size=(W + 10, 3)) @ L.T + rng.normal(0, 0.008, (W + 10, N)) + 0.0002
        for name, est in (("lw", lambda X: VX.ledoit_wolf(X)[0]), ("sample", lambda X: np.cov(X.T, bias=True))):
            ws = []
            for start in (0, 10):
                X = R[start:start + W]
                S = est(X)
                mu, _ = VX.bayes_stein_means(X, S)
                (_, _, w), = VX.frontier_curve(mu * 252, S * 252, _cfg(), [0.10])
                ws.append(w)
            out[name][0].append(np.abs(ws[1] - ws[0]).sum())
            out[name][1].append(abs(np.sqrt(ws[0] @ true @ ws[0]) - 0.10))
    return {k: (np.mean(a), np.mean(b)) for k, (a, b) in out.items()}


def test_frontier_stable_under_shrinkage():
    low = _stability(30, 252)          # about the engine's setting: 1 year of data, dozens of assets
    high = _stability(40, 80)          # few observations per asset: where shrinkage matters most
    for r in (low, high):
        assert r["lw"][0] < r["sample"][0]          # the frontier portfolio moves less as the window rolls
        assert r["lw"][1] < r["sample"][1]          # and its true risk is closer to what was promised
    gain = lambda r: 1 - r["lw"][0] / r["sample"][0]
    assert gain(high) > gain(low)                   # the benefit grows with assets per observation


def test_bayes_stein_and_kde():
    X = np.random.default_rng(7).normal(0.001, 0.01, (252, 10))
    S, _ = VX.ledoit_wolf(X)
    mu, phi = VX.bayes_stein_means(X, S)
    assert 0 <= phi <= 1 and np.ptp(mu) <= np.ptp(X.mean(axis=0)) + 1e-15
    x = np.random.default_rng(8).normal(0, 1, 20000)
    grid = np.linspace(-5, 5, 201)
    dens, h = VX.kde_on_grid(x, grid)
    assert abs(np.trapezoid(dens, grid) - 1) < 1e-6
    assert np.abs(dens - stats.norm.pdf(grid)).max() < 0.03


def test_regime_export_is_consistent(exported):
    d = exported[0]
    reg = json.loads((d / "data" / "regimes.json").read_text())
    grid = np.array(reg["grid"])
    for f in reg["frames"]:
        assert abs(sum(f["probs"]) - 1) < 1e-3 and f["current"] == int(np.argmax(f["probs"]))
        for r in f["ridges"] + [f["unconditional"]]:
            assert abs(np.trapezoid(r["density"], grid) - 1) < 0.02


# ------------------------------------------------------------------ browser

@pytest.fixture(scope="module")
def server(exported):
    if not chrome_path():
        pytest.skip("Chrome/Chromium not installed")
    pytest.importorskip("websockets")
    d = exported[0]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


SELFTEST = "document.getElementById('result') && document.getElementById('result').textContent"
VIEW_IDS = ["fan", "calibration", "regimes", "frontier", "exposure"]


def test_browser_charts_draw_in_once(server):
    out = json.loads(run_page(server + "selftest.html", SELFTEST))
    assert "error" not in out, out.get("error")
    assert out["reduced"] is False and out["default_on"] is True
    # every chart is real SVG and draws in when its data first arrives ...
    assert all(out["svg_drawn"][k] > 3 for k in VIEW_IDS), out["svg_drawn"]
    assert all(out["draw_on_show"][k] > 0 for k in VIEW_IDS), out["draw_on_show"]
    assert all(400 <= out["longest"][k] <= 2600 for k in VIEW_IDS), out["longest"]   # quick, and done before freeze
    # ... then everything is still: no running animations, no animation-frame loop
    assert out["settled"] == 0 and out["idle_raf"] == 0
    # revisiting a tab or dragging the date shows the chart directly; picking a date replays once
    assert out["revisit_replays"] == 0 and out["drag_animates"] == 0
    assert out["change_replays"] > 0 and out["change_settles"] == 0
    # Motion off: nothing moves, the chart is identical, and the choice is saved and shown
    assert out["off_animates"] == 0 and out["same_chart"] is True
    assert out["pref_saved"] == "off" and out["attr_off"] == "off" and out["button_pressed"] == "false"
    assert out["globals_leaked"] == []


def test_browser_honours_reduced_motion(server):
    reduce = [{"name": "prefers-reduced-motion", "value": "reduce"}]
    out = json.loads(run_page(server + "selftest.html", SELFTEST, media=reduce,
                              done=lambda v: bool(v) and v != "pending" and '"reduced":true' in v))
    assert "error" not in out, out.get("error")
    assert out["reduced"] is True
    # every animation, delay included, collapses to ~0ms: charts appear complete at once
    assert all(0 < out["longest"][k] <= 1 for k in VIEW_IDS), out["longest"]
    assert out["settled"] == 0


REPORT_CHECK = r"""
<script>
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const running = (el) => el.getAnimations({subtree: true}).filter((a) => a.playState === "running").length;
  const o = {};
  try {
    await sleep(150);
    const first = document.querySelector(".subpanel:not([hidden])");
    o.first_draws = running(first);
    o.button_pressed = document.querySelector(".motion-btn").getAttribute("aria-pressed");
    await sleep(3000);
    o.first_settled = running(document.body);
    o.first_frozen = first.hasAttribute("data-drawn");
    const tiles = [...document.querySelectorAll("[data-count]")];
    o.tiles = tiles.length;
    o.tiles_final = tiles.every((t) => t.textContent.trim() === t.dataset.count.trim());
    const tabs = [...document.querySelectorAll(".subtab")];
    tabs[1].click(); await sleep(150);
    o.second_draws = running(document.getElementById(tabs[1].dataset.tab));
    tabs[0].click(); await sleep(150);
    o.revisit_draws = running(first);
    document.querySelector(".motion-btn").click();
    o.attr_off = document.documentElement.dataset.motion;
    o.pref_saved = localStorage.getItem("motion");
    tabs[2].click(); await sleep(150);
    o.off_draws = running(document.body);
  } catch (e) { o.error = String(e && e.stack || e); }
  window.__result = JSON.stringify(o);
})();
</script>"""


@pytest.fixture(scope="module")
def report_url(server, exported):
    from podengine.pipeline import full_run
    from podengine.report import build_report
    d, md, cfg = exported[0], exported[1], exported[2]
    b = full_run(md, cfg, 50000, quick=True, state_dir=str(d / "_state"), verbose=False)
    path = build_report(b, str(d / "motion_report.html"))
    text = open(path).read()
    open(path, "w").write(text.replace("</body>", REPORT_CHECK + "</body>"))
    return server + "motion_report.html"


def test_report_charts_draw_in_once(report_url):
    out = json.loads(run_page(report_url, "window.__result || ''", done=bool))
    assert "error" not in out, out.get("error")
    assert out["button_pressed"] == "true" and out["first_draws"] > 0
    assert out["first_settled"] == 0 and out["first_frozen"] is True
    assert out["tiles"] > 0 and out["tiles_final"] is True        # count-up ends on the real figure
    assert out["second_draws"] > 0 and out["revisit_draws"] == 0
    assert out["attr_off"] == "off" and out["pref_saved"] == "off" and out["off_draws"] == 0


def _motion_css():
    from podengine.report import CSS
    viz = open(os.path.join(VX.VIZ_DIR, "viz.css")).read()
    return {"viz.css": viz, "report.py": CSS}


@pytest.mark.parametrize("name", ["viz.css", "report.py"])
def test_motion_css_animates_only_cheap_properties(name):
    import re
    css = _motion_css()[name]
    frames = re.findall(r"@keyframes\s+[\w-]+\s*\{((?:[^{}]*\{[^{}]*\})*)[^{}]*\}", css)
    assert frames, "no keyframes found"
    for body in frames:
        props = set(re.findall(r"([a-z-]+)\s*:", body))
        assert props <= {"transform", "opacity", "stroke-dashoffset"}, props
        assert "to{" not in body.replace(" ", "") or "stroke-dashoffset" in body   # only `from` (draw needs `to`)
    assert not re.search(r"transition\s*:\s*all", css)
    assert '[data-motion="off"]' in css and "prefers-reduced-motion" in css
    for tok in ("--ease-out:cubic-bezier(0.16,1,0.3,1)", "--dur-micro:120ms", "--dur-short:220ms",
                "--dur-long:420ms", "--dur-draw:900ms"):
        assert tok in css.replace(" ", ""), tok


def test_no_play_controls_left():
    js = open(os.path.join(VX.VIZ_DIR, "js", "app.js")).read() + open(os.path.join(VX.VIZ_DIR, "js", "views.js")).read()
    assert not os.path.exists(os.path.join(VX.VIZ_DIR, "js", "anim.js"))
    for word in ("Play", "Pause", "speed", "animator", "setInterval"):
        assert word not in js, word


def test_file_protocol_shows_instructions(exported):
    if not chrome_path():
        pytest.skip("Chrome/Chromium not installed")
    pytest.importorskip("websockets")
    d = exported[0]
    txt = run_page("file://" + str(d / "index.html"), "document.body.innerText",
                   done=lambda v: bool(v) and "served" in v, timeout=20)
    assert "python -m podengine serve" in txt
