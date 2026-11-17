// Pod Engine viewer: tabs, the animations toggle, per-view controls and static drawing.
// ES module: defines no globals. Drawing code (views.js) loads when a chart tab is first
// opened; the render loop (anim.js) loads only when animations are switched on.

const STORE_KEY = "podengine.viz.animations";

const TABS = [
  { id: "report", title: "Engine report", kind: "report", src: "reports/report.html",
    hint: "python -m podengine run --viz" },
  { id: "optimizer", title: "Optimizer", kind: "report", src: "reports/optimizer.html",
    hint: "python -m podengine optimize --positions your.csv --viz" },
  { id: "fan", title: "Forecast fan", kind: "view" },
  { id: "calibration", title: "Calibration", kind: "view" },
  { id: "regimes", title: "Regimes", kind: "view" },
  { id: "frontier", title: "Frontier", kind: "view" },
  { id: "exposure", title: "Exposure", kind: "view" },
];

function readPref() {
  try { return window.localStorage.getItem(STORE_KEY) === "on"; } catch (e) { return false; }
}
function writePref(on) {
  try { window.localStorage.setItem(STORE_KEY, on ? "on" : "off"); } catch (e) { /* storage blocked */ }
}

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
    animations: readPref(), active: null, views: {}, viewsMod: null, animMod: null, animator: null,
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
  const toggle = h("input", { type: "checkbox", role: "switch", "aria-label": "Animations" });
  toggle.checked = state.animations;
  const meta = h("span", { class: "pv-meta", text: "" });
  const head = h("div", { class: "pv-head" }, h("span", { class: "pv-title", text: "Pod Engine" }), meta,
    h("label", { class: "pv-toggle" }, toggle, h("span", { text: "Animations" })));
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
    meta.textContent = `${m.synthetic ? "DEMO DATA · " : ""}${m.source} · profile ${m.profile} · as of ${m.asof}`;
  }).catch(() => { meta.textContent = "no exported data yet: run python -m podengine viz"; });

  // ---------------------------------------------------------------- animations toggle
  async function setAnimations(on) {
    state.animations = !!on;
    toggle.checked = state.animations;
    writePref(state.animations);
    if (!state.animations) {
      if (state.animator) state.animator.stop();
    } else if (!state.animMod) {
      state.animMod = await import("./anim.js");
    }
    for (const id in state.views) state.views[id].syncControls();
  }
  toggle.addEventListener("change", () => setAnimations(toggle.checked));

  // ---------------------------------------------------------------- tab selection
  async function select(id) {
    if (state.active === id) return;
    if (state.animator) state.animator.stop();
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
    const r = await fetch(base + tab.src, { method: "HEAD", cache: "no-store" }).catch(() => null);
    if (r && r.ok) panel.append(h("iframe", { class: "pv-frame", src: base + tab.src, title: tab.title }));
    else panel.append(h("div", { class: "pv-card pv-msg" },
      h("p", { text: `${tab.title} not found in this folder yet.` }), h("p", {}, "Create it with ", h("code", { text: tab.hint }))));
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
    state.views[id].redraw();
  }

  // ---------------------------------------------------------------- one view
  function makeView(id, def, data, panel) {
    const n = def.count(data);
    const v = { id, def, data, n, pos: Math.max(0, n - 1), frac: def.staticFrac, playing: false, speed: 1 };
    const canvas = h("canvas", { class: "pv-canvas", role: "img", "aria-label": def.title });
    const play = h("button", { class: "pv-btn", text: "Play" });
    const speed = h("select", { class: "pv-select", "aria-label": "Speed" },
      ...[0.5, 1, 2, 4].map((s) => h("option", { value: String(s), text: s + "×" })));
    speed.value = "1";
    const scrub = h("input", { type: "range", class: "pv-scrub", min: "0", max: String(Math.max(0, n - 1)), step: "1",
      value: String(v.pos), "aria-label": "Date" });
    const date = h("span", { class: "pv-date" });
    const extra = h("div");
    panel.append(h("div", { class: "pv-card" },
      h("div", { class: "pv-h", text: def.title }),
      ...def.labels(data).map((t) => h("span", { class: "pv-label", text: t })),
      h("p", { class: "pv-p", text: def.blurb }),
      def.legend ? def.legend(h) : null,
      canvas,
      h("div", { class: "pv-controls" }, play, speed, scrub, date),
      extra));

    v.redraw = () => {
      const ctx = fit(canvas);
      if (!ctx) return;
      def.draw(ctx, canvas.clientWidth, canvas.clientHeight, data, v.pos, v.frac, theme(root));
      date.textContent = def.dateOf(data, v.pos);
      scrub.value = String(v.pos);
      if (def.below) def.below(extra, data, v.pos, h);
    };
    v.syncControls = () => {
      play.disabled = !state.animations;
      play.title = state.animations ? "" : "Turn animations on to play";
      if (!state.animations) v.playing = false;
      play.textContent = v.playing ? "Pause" : "Play";
    };
    v.tick = (dt) => {                       // called by the animator only
      const step = dt * def.rate * v.speed;
      v.frac += step;
      while (v.frac >= 1) { v.frac -= 1; v.pos += 1; }
      if (v.pos >= n - 1) { v.pos = n - 1; v.frac = def.staticFrac; v.playing = false; v.syncControls(); v.redraw(); return false; }
      v.redraw();
      return true;
    };
    play.addEventListener("click", () => {
      if (!state.animations) return;
      if (!v.playing) {
        if (v.pos >= n - 1) v.pos = 0;
        v.frac = 0;
        if (!state.animator) state.animator = state.animMod.createAnimator();
        state.animator.start(v, canvas);          // sets v.playing
      } else if (state.animator) state.animator.stop();   // clears v.playing
      v.syncControls();
    });
    speed.addEventListener("change", () => { v.speed = Number(speed.value); });
    scrub.addEventListener("input", () => {
      v.pos = Number(scrub.value);
      v.frac = def.staticFrac;
      v.redraw();
    });
    new ResizeObserver(() => { if (!panel.hidden) v.redraw(); }).observe(canvas);
    v.syncControls();
    return v;
  }

  setAnimations(state.animations);
  const first = (location.hash || "").slice(1);
  select(TABS.some((t) => t.id === first) ? first : "fan");

  return {
    state, select, setAnimations,
    animLoaded: () => !!state.animMod,
    viewsLoaded: () => !!state.viewsMod,
    destroy() { if (state.animator) state.animator.stop(); },
  };
}

function fit(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, hgt = canvas.clientHeight;
  if (!w || !hgt) return null;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hgt * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(hgt * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, hgt);
  return ctx;
}

function theme(root) {
  const cs = getComputedStyle(root);
  const g = (n) => cs.getPropertyValue("--pv-" + n).trim();
  return { text: g("text"), text2: g("text2"), muted: g("muted"), line: g("line"), surface: g("surface"),
    s1: g("s1"), s2: g("s2"), s3: g("s3"), s4: g("s4"), s5: g("s5"), band1: g("band1"), band2: g("band2"),
    mint: g("mint"), gold: g("gold"), neutral: g("neutral"), good: g("good"), warn: g("warn"), crit: g("crit") };
}
