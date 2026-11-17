"""Local server for the viewer: serves viz/ and runs the optimizer on request.

  GET  /api/status    -> {"ok": true, "demo": bool}  (tells the page the Optimize button can work)
  POST /api/optimize  -> runs the optimizer on the posted holdings, writes reports/optimizer.html

Listens on 127.0.0.1 only, so nothing outside your computer can reach it. On a
static host (GitHub Pages) /api/* does not exist and the page says so.
"""
import functools
import http.server
import json
import os
import tempfile
import threading
import traceback
import webbrowser

from .vizexport import VIZ_DIR

MAX_BODY = 200_000
_lock = threading.Lock()


def _num_or_none(x, lo, hi, name):
    if x in (None, ""):
        return None
    v = float(x)
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return v


def run_request(req, viz_dir, demo=False, cache="data_cache"):
    """Validate a request from the Optimize form and run the optimizer."""
    from .optrun import optimize_and_report, parse_holdings_text, parse_views_text
    kind = req.get("kind", "shares")
    pos = parse_holdings_text(req.get("holdings", ""), kind)
    views = parse_views_text(req.get("views", ""))
    profile = req.get("profile") or "unconstrained"
    if profile not in ("unconstrained", "margin", "cash"):
        raise ValueError("unknown account profile")
    vt = _num_or_none(req.get("vol_target"), 0.5, 60, "target volatility (%)")
    mt = _num_or_none(req.get("max_turnover"), 1, 500, "maximum turnover (%)")
    cap = _num_or_none(req.get("capital"), 100, 1e10, "account size")
    if kind == "weight" and cap is None:
        cap = 100000.0
    out_dir = tempfile.mkdtemp(prefix="podengine_opt_")
    with _lock:
        res = optimize_and_report(pos, kind, views or None, profile=profile,
                                  vol_target=None if vt is None else vt / 100,
                                  max_turnover=None if mt is None else mt / 100, capital=cap,
                                  demo=demo, cache=cache, out_dir=out_dir, viz_dir=viz_dir,
                                  copy_to_viz=True, verbose=False)
    return {"ok": True, "report": "reports/optimizer.html", "synthetic": res["synthetic"],
            "equity": res["equity"], "notes": res["notes"], "portfolios": res["portfolios"]}


def make_handler(viz_dir, demo=False, cache="data_cache"):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/api/status":
                return self._json(200, {"ok": True, "demo": demo})
            return super().do_GET()

        def do_POST(self):
            if self.path.split("?")[0] != "/api/optimize":
                return self._json(404, {"ok": False, "error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._json(400, {"ok": False, "error": "request too large or empty"})
            try:
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                return self._json(200, run_request(req, viz_dir, demo, cache))
            except (ValueError, KeyError, TypeError) as e:
                return self._json(400, {"ok": False, "error": str(e)})
            except RuntimeError as e:
                return self._json(500, {"ok": False, "error": str(e)})
            except Exception as e:
                traceback.print_exc()
                return self._json(500, {"ok": False, "error": f"unexpected error: {e}"})

        def log_message(self, fmt, *args):
            if "/api/" in (args[0] if args else ""):
                super().log_message(fmt, *args)

    return functools.partial(Handler, directory=viz_dir)


def serve(viz_dir=None, port=8765, open_browser=True, demo=False, cache="data_cache"):
    root = viz_dir or VIZ_DIR
    if not os.path.exists(os.path.join(root, "data", "meta.json")):
        print("No viewer data yet. Run `python -m podengine viz` (or `run --viz`) first.")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(root, demo, cache))
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving {root} at {url}  (Ctrl+C to stop)" + ("  [demo prices for the Optimize button]" if demo else ""))
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0
