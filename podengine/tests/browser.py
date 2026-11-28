"""Minimal real-time headless-Chrome driver for the viewer tests (Chrome DevTools
Protocol over a websocket). The motion tests need real time: CSS animations and
requestAnimationFrame do not advance under headless 'virtual time'."""
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request


def chrome_path():
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    return mac if os.path.exists(mac) else None


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run_page(url, expr, done=lambda v: v not in (None, "", "pending"), timeout=45, media=None):
    """Open `url` in headless Chrome and poll JavaScript `expr` until `done(value)`.
    `media`: CSS media features to emulate, e.g. [{"name": "prefers-reduced-motion", "value": "reduce"}];
    the page is reloaded with them in effect."""
    from websockets.sync.client import connect
    exe = chrome_path()
    port = free_port()
    prof = tempfile.mkdtemp()
    proc = subprocess.Popen([exe, "--headless=new", "--no-sandbox", "--disable-gpu", f"--remote-debugging-port={port}",
                             "--remote-allow-origins=*", f"--user-data-dir={prof}", "--window-size=1200,900", url],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url, t0 = None, time.time()
        while not ws_url and time.time() - t0 < 20:
            try:
                pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2))
                ws_url = next((p["webSocketDebuggerUrl"] for p in pages if p.get("type") == "page"), None)
            except Exception:
                time.sleep(0.3)
        if not ws_url:
            raise RuntimeError("could not connect to headless Chrome")
        with connect(ws_url, max_size=None) as ws:
            n = 0

            def call(method, params):
                nonlocal n
                n += 1
                ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("id") == n:
                        return msg
            if media:
                call("Emulation.setEmulatedMedia", {"features": media})
                call("Page.reload", {"ignoreCache": True})
            t0 = time.time()
            while time.time() - t0 < timeout:
                msg = call("Runtime.evaluate", {"expression": expr, "returnByValue": True})
                val = msg.get("result", {}).get("result", {}).get("value")
                if done(val):
                    return val
                time.sleep(0.5)
        raise TimeoutError(f"page did not finish: last value {val!r}")
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)
