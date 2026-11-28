// Pod Engine viewer: tabs, the Motion toggle, per-view date slider and SVG drawing.
// ES module: defines no globals. Chart code (views.js) loads when a chart tab is first opened.
//
// Motion: each chart draws itself in once when its data arrives (first open, or a new date
// picked), then stays still. There is no render loop and nothing runs on a timer. The choice is
// stored under "motion" and applied as <html data-motion="on|off">; prefers-reduced-motion is honoured.

const STORE_KEY = "motion";
const DRAW_MS = 2600;                     // longest draw-in (stagger + 900ms draw), after which a chart is frozen

const TABS = [
  { id: "report", title: "Engine Report", kind: "report", src: "reports/report.html",
    hint: "python -m podengine run --viz" },
  { id: "optimizer", title: "Optimizer", kind: "report", src: "reports/optimizer.html",
    hint: "python -m podengine optimize --positions your.csv --viz" },
  { id: "fan", title: "Forecast Fan", kind: "view" },
  { id: "calibration", title: "Calibration", kind: "view" },
  { id: "regimes", title: "Regimes", kind: "view" },
  { id: "frontier", title: "Frontier", kind: "view" },
  { id: "exposure", title: "Exposure", kind: "view" },
];

function readPref() {
  try { return window.localStorage.getItem(STORE_KEY) !== "off"; } catch (e) { return true; }
}
function writePref(on) {
  try { window.localStorage.setItem(STORE_KEY, on ? "on" : "off"); } catch (e) { /* storage blocked */ }
}
export const motionOn = () => document.documentElement.dataset.motion !== "off" &&
  !matchMedia("(prefers-reduced-motion: reduce)").matches;

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const k in attrs || {}) {
    if (k === "class") e.className = attrs[k];
    else if (k === "text") e.textContent = attrs[k];
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), attrs[k]);
    else e.setAttribute(k, attrs[k]);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

export function init(root, opts = {}) {
  const base = opts.base || "";
  const state = {
    motion: readPref(), active: null, views: {}, viewsMod: null,
    meta: null, panels: {}, tabs: {},
  };
  root.classList.add("pv-root");
  root.textContent = "";

  if (location.protocol === "file:") {
    root.append(h("div", { class: "pv-msg" },
      h("p", { text: "This page has to be served, not opened as a file (browsers block loading its data from file://)." }),
      h("p", {}, "Run ", h("code", { text: "python -m podengine serve" }), " and open the address it prints.")));
    return { state, destroy() {} };
  }

  // ---------------------------------------------------------------- header, toggle, tabs
  const toggle = h("button", { class: "pv-motion", type: "button", "aria-pressed": "true",
    title: "Charts draw in once when their data arrives" }, h("span", { class: "pv-motion-dot" }), h("span", { text: "Motion" }));
  const meta = h("span", { class: "pv-meta", text: "" });
  const head = h("div", { class: "pv-head" }, h("span", { class: "pv-title", text: "Pod Engine" }), meta, toggle);
  const tabbar = h("div", { class: "pv-tabs", role: "tablist" });
  root.append(h("div", { class: "pv-top" }, head, tabbar));
  for (const t of TABS) {
    const btn = h("button", { class: "pv-tab", role: "tab", "aria-selected": "false", text: t.title,
      onclick: () => select(t.id) });
    const panel = h("section", { class: "pv-panel", role: "tabpanel", hidden: "" });
    state.tabs[t.id] = btn;
    state.panels[t.id] = panel;
    tabbar.append(btn);
    root.append(panel);
  }

  getJSON(base + "data/meta.json").then((m) => {
    state.meta = m;
    const cap = (x) => String(x).charAt(0).toUpperCase() + String(x).slice(1);
    meta.textContent = `${m.synthetic ? "Demo data · " : ""}${m.source} · ${cap(m.profile)} profile · as of ${m.asof}`;
  }).catch(() => { meta.textContent = "no exported data yet: run python -m podengine viz"; });

  // ---------------------------------------------------------------- motion toggle
  function setMotion(on) {
    state.motion = !!on;
    const v = state.motion ? "on" : "off";
    document.documentElement.dataset.motion = v;
    toggle.setAttribute("aria-pressed", state.motion ? "true" : "false");
    writePref(state.motion);
    root.querySelectorAll("iframe").forEach((f) => {           // the embedded reports follow the same switch
      try { f.contentDocument.documentElement.dataset.motion = v; } catch (e) { /* not loaded yet */ }
    });
  }
  toggle.addEventListener("click", () => setMotion(!state.motion));

  // ---------------------------------------------------------------- tab selection
  async function select(id) {
    if (state.active === id) return;
    for (const t of TABS) {
      state.tabs[t.id].setAttribute("aria-selected", t.id === id ? "true" : "false");
      state.panels[t.id].hidden = t.id !== id;
    }
    state.active = id;
    try { history.replaceState(null, "", "#" + id); } catch (e) { /* ignore */ }
    const tab = TABS.find((t) => t.id === id);
    if (tab.kind === "report") return openReport(tab);
    return openView(id);
  }

  async function openReport(tab) {
    const panel = state.panels[tab.id];
    if (panel.dataset.ready) return;
    panel.dataset.ready = "1";
    panel.classList.add("pv-full");
    const holder = h("div");
    if (tab.id === "optimizer") panel.append(optimizerForm(() => loadFrame(tab, holder, true)));
    panel.append(holder);
    await loadFrame(tab, holder, false);
  }

  async function loadFrame(tab, holder, fresh) {
    const r = await fetch(base + tab.src, { method: "HEAD", cache: "no-store" }).catch(() => null);
    holder.textContent = "";
    if (r && r.ok) {
      const f = h("iframe", { class: "pv-frame", src: base + tab.src + (fresh ? "?t=" + Date.now() : ""), title: tab.title });
      f.addEventListener("load", () => {
        try { f.contentDocument.documentElement.dataset.motion = state.motion ? "on" : "off"; } catch (e) { /* cross-origin */ }
      });
      holder.append(f);
    } else {
      holder.append(h("div", { class: "pv-card pv-msg", style: "margin:16px 20px" },
        h("p", { text: tab.id === "optimizer" ? "No optimizer report yet. Enter your holdings above and press Optimize."
          : `${tab.title} not found in this folder yet.` }),
        tab.id === "optimizer" ? null : h("p", {}, "Create it with ", h("code", { text: tab.hint }))));
    }
  }

  // ---------------------------------------------------------------- optimizer input form
  function optimizerForm(onDone) {
    const field = (label, el) => h("label", { class: "pv-field" }, h("span", { text: label }), el);
    const holdings = h("textarea", { spellcheck: "false", placeholder: "SPY, 60\nQQQ, 40\nTLT, 90\nGLD, 15\nCASH, 5000" });
    const kind = h("select", {}, h("option", { value: "shares", text: "Shares" }),
      h("option", { value: "value", text: "Dollar values" }), h("option", { value: "weight", text: "Weights (25% or 0.25)" }));
    const file = h("input", { type: "file", accept: ".csv,.txt" });
    const profile = h("select", {}, h("option", { value: "unconstrained", text: "Unconstrained: long/short, up to 6x (futures)" }),
      h("option", { value: "margin", text: "Margin: long/short, up to 2x" }), h("option", { value: "cash", text: "Cash: long only, no leverage" }));
    const vol = h("input", { type: "number", min: "0.5", max: "60", step: "0.5", placeholder: "Same as now" });
    const turn = h("input", { type: "number", min: "1", max: "500", step: "1", placeholder: "No limit" });
    const cap = h("input", { type: "number", min: "100", step: "100", placeholder: "From your holdings" });
    const views = h("textarea", { spellcheck: "false", placeholder: "TLT, 4%, 60%\nQQQ - SPY, -3%, 50%" });
    views.style.minHeight = "90px";
    const go = h("button", { class: "pv-btn pv-primary", type: "submit", text: "Optimize" });
    const status = h("span", { class: "pv-status" });
    const note = h("p", { class: "pv-note" });
    const form = h("form", {},
      h("div", { class: "pv-grid" },
        field("Holdings: one per line, ticker then amount (CASH line for cash)", holdings),
        h("div", { style: "display:flex;flex-direction:column;gap:12px" },
          field("Amounts are", kind), field("Or load a CSV (ticker plus shares, value or weight)", file),
          field("Account type", profile)),
        h("div", { style: "display:flex;flex-direction:column;gap:12px" },
          field("Target volatility, % a year (optional)", vol), field("Maximum turnover, % (optional)", turn),
          field("Account size, $ (optional; needed for weights)", cap)),
        field("Views (optional): ticker or LONG - SHORT, return a year, confidence", views)),
      h("div", { class: "pv-controls" }, go, status), note);
    const box = h("div", { class: "pv-form" }, h("details", { open: "" },
      h("summary", { text: "Optimize Your Portfolio" }), form));

    file.addEventListener("change", async () => {
      const f = file.files && file.files[0];
      if (!f) return;
      const rows = (await f.text()).split(/\r?\n/).map((l) => l.split(/[,;\t]/).map((x) => x.trim())).filter((r) => r[0]);
      const head = rows[0].map((x) => x.toLowerCase());
      const col = ["shares", "value", "weight"].find((k) => head.includes(k));
      const body = col || head[0] === "ticker" ? rows.slice(1) : rows;
      const j = col ? head.indexOf(col) : 1;
      if (col) kind.value = col;
      holdings.value = body.filter((r) => r[j] !== undefined).map((r) => `${r[0]}, ${r[j]}`).join("\n");
    });

    const controls = [holdings, kind, file, profile, vol, turn, cap, views, go];
    fetch(base + "api/status", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).catch(() => null).then((s) => {
      if (s && s.ok) {
        note.textContent = s.demo ? "Running in demo mode: prices are simulated, so results say nothing about real markets."
          : "Runs on your computer with free Yahoo Finance prices. Takes about 10 to 40 seconds.";
      } else {
        controls.forEach((c) => (c.disabled = true));
        note.append("This copy is read-only. To optimize your own holdings, run ",
          h("code", { text: "python -m podengine serve" }), " on your computer and open the address it prints. Your holdings never leave your computer.");
      }
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      go.disabled = true;
      status.className = "pv-status";
      status.textContent = "Optimizing...";
      try {
        const r = await fetch(base + "api/optimize", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ holdings: holdings.value, kind: kind.value, profile: profile.value, vol_target: vol.value,
            max_turnover: turn.value, capital: cap.value, views: views.value }) });
        const out = await r.json();
        if (!out.ok) throw new Error(out.error || "failed");
        status.textContent = `Done: account equity $${Math.round(out.equity).toLocaleString()}. Report updated below.`;
        await onDone();
      } catch (err) {
        status.className = "pv-status pv-err";
        status.textContent = "Could not optimize: " + err.message;
      } finally {
        go.disabled = false;
      }
    });
    return box;
  }

  async function openView(id) {
    const panel = state.panels[id];
    if (!state.viewsMod) state.viewsMod = await import("./views.js");
    if (!state.views[id]) {
      const def = state.viewsMod.VIEWS[id];
      let data;
      try { data = await getJSON(base + "data/" + def.file); } catch (e) {
        panel.append(h("div", { class: "pv-card pv-msg" }, h("p", { text: `No data for this view (${def.file}).` }),
          h("p", {}, "Run ", h("code", { text: "python -m podengine viz" }), " first.")));
        state.views[id] = { syncControls() {}, redraw() {} };
        return;
      }
      state.views[id] = makeView(id, def, data, panel);
    }
    state.views[id].show();
  }

  // ---------------------------------------------------------------- one view
  let uidSeq = 0;
  function makeView(id, def, data, panel) {
    const n = def.count(data);
    const v = { id, def, data, n, pos: Math.max(0, n - 1), drawnAt: -1, timer: 0, uid: `pv${id}${++uidSeq}` };
    const chart = h("div", { class: "pv-chart" });
    const tip = h("div", { class: "pv-tip", role: "tooltip" });
    const scrub = h("input", { type: "range", class: "pv-scrub", min: "0", max: String(Math.max(0, n - 1)), step: "1",
      value: String(v.pos), "aria-label": "Date" });
    const date = h("span", { class: "pv-date" });
    const extra = h("div");
    panel.append(h("div", { class: "pv-card" },
      h("div", { class: "pv-h", text: def.title }),
      ...def.labels(data).map((t) => h("span", { class: "pv-label", text: t })),
      h("p", { class: "pv-p", text: def.blurb }),
      def.legend ? def.legend(h) : null,
      h("div", { class: "pv-chartwrap" }, chart, tip),
      h("div", { class: "pv-controls" }, h("span", { text: "Date" }), scrub, date),
      extra));

    // Draw frame v.pos. `animate`: replay the draw-in (new data); otherwise the chart appears in its final state.
    let lastW = 0;
    v.render = (animate) => {
      const W = Math.round(chart.clientWidth);
      if (!W) return;
      lastW = W;                      // the resize observer then ignores this width (no redraw over the draw-in)
      clearTimeout(v.timer);
      chart.classList.toggle("pv-still", !animate);
      chart.innerHTML = def.svg(W, 440, data, v.pos, v.uid);
      date.textContent = def.dateOf(data, v.pos);
      scrub.value = String(v.pos);
      if (def.below) def.below(extra, data, v.pos, h);
      v.drawnAt = v.pos;
      // once the draw-in has finished, freeze it so re-showing the tab does not replay it
      if (animate) v.timer = setTimeout(() => chart.classList.add("pv-still"), DRAW_MS);
    };
    v.show = () => { if (v.drawnAt < 0) v.render(true); };

    scrub.addEventListener("input", () => { v.pos = Number(scrub.value); v.render(false); });   // dragging: no motion
    scrub.addEventListener("change", () => { v.pos = Number(scrub.value); v.render(true); });   // a date was picked: draw it in

    // hover: read values under the pointer (instant; no motion)
    chart.addEventListener("mousemove", (e) => {
      const hitRect = e.target.closest && e.target.closest(".pv-hit");
      if (!hitRect || !def.tip) { tip.hidden = true; return; }
      const r = hitRect.getBoundingClientRect(), box = chart.getBoundingClientRect();
      tip.innerHTML = def.tip(data, v.pos, Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)));
      tip.hidden = false;
      const x = e.clientX - box.left, y = e.clientY - box.top;
      tip.style.left = (x > box.width * 0.6 ? x - tip.offsetWidth - 14 : x + 14) + "px";
      tip.style.top = Math.max(0, y - 20) + "px";
    });
    chart.addEventListener("mouseleave", () => { tip.hidden = true; });
    tip.hidden = true;

    new ResizeObserver(() => {
      const W = Math.round(chart.clientWidth);
      if (!panel.hidden && W && W !== lastW && v.drawnAt >= 0) v.render(false);
      lastW = W;
    }).observe(chart);
    return v;
  }

  setMotion(state.motion);
  const first = (location.hash || "").slice(1);
  select(TABS.some((t) => t.id === first) ? first : "fan");

  return {
    state, select, setMotion,
    viewsLoaded: () => !!state.viewsMod,
    destroy() {},
  };
}
