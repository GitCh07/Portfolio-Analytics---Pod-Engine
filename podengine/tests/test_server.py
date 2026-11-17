"""Tests for the Optimize button's backend (podengine/server.py, podengine/optrun.py)."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from podengine.optrun import parse_holdings_text, parse_views_text


def test_parse_holdings():
    df = parse_holdings_text("SPY, 60\nqqq\t40\n# comment\nTLT 90\nCASH, $5,000\nSPY, 10", "shares")
    assert dict(zip(df["ticker"], df["shares"])) == {"SPY": 70, "QQQ": 40, "TLT": 90, "CASH": 5000}
    w = parse_holdings_text("ticker,weight\nSPY, 60%\nIEF, 0.3", "weight")
    assert dict(zip(w["ticker"], w["weight"])) == {"SPY": 0.6, "IEF": 0.3}
    for bad in ("SPY", "SPY, abc", "rm -rf /, 3", ""):
        with pytest.raises(ValueError):
            parse_holdings_text(bad, "shares")


def test_parse_views():
    v = parse_views_text("TLT, 4%, 60%\nQQQ - SPY, -3%\n")
    assert v[0]["assets"] == {"TLT": 1.0} and v[0]["return"] == 0.04 and v[0]["confidence"] == 0.6
    assert v[1]["assets"] == {"QQQ": 1.0, "SPY": -1.0} and v[1]["return"] == -0.03 and v[1]["confidence"] == 0.5
    for bad in ("TLT", "TLT, x", "TLT, 4%, 150%", "A-B-C, 1%"):
        with pytest.raises(ValueError):
            parse_views_text(bad)


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    import http.server
    from podengine.server import make_handler
    d = tmp_path_factory.mktemp("viz")
    (d / "data").mkdir()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(str(d), demo=True))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/", d
    srv.shutdown()


def _post(url, obj):
    req = urllib.request.Request(url + "api/optimize", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_status_and_optimize(api):
    url, d = api
    with urllib.request.urlopen(url + "api/status") as r:
        assert json.loads(r.read()) == {"ok": True, "demo": True}
    code, out = _post(url, {"holdings": "SPY, 300\nTLT, 400\nGLD, 100\nCASH, 5000", "kind": "shares",
                            "profile": "margin", "views": "TLT, 4%, 60%", "max_turnover": "40"})
    assert code == 200 and out["ok"] and out["synthetic"] is True
    assert [p["name"] for p in out["portfolios"]] == ["Current", "Minimum variance", "Risk parity", "Black-Litterman"]
    html = (d / "reports" / "optimizer.html").read_text()
    assert "Portfolio Optimizer" in html and "subtab" in html
    bl = out["portfolios"][-1]
    assert bl["turnover"] <= 0.40 + 1e-3


def test_bad_requests_are_rejected(api):
    url, _ = api
    for body in ({"holdings": "SPY", "kind": "shares"}, {"holdings": "SPY, 1\nTLT, 2", "kind": "bogus"},
                 {"holdings": "SPY, 1\nTLT, 2", "kind": "shares", "profile": "yolo"},
                 {"holdings": "SPY, 1\nTLT, 2", "kind": "shares", "vol_target": "500"}):
        code, out = _post(url, body)
        assert code == 400 and out["ok"] is False and out["error"]
