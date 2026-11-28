// The five chart views, drawn as SVG. Every number was computed in Python
// (podengine/vizexport.py); this file only maps it to shapes.
//
// Motion follows one rule: a chart draws itself in once, from where its data starts,
// when its data arrives (first open, or a new date picked on the slider), then stays still.
//   bars grow from the zero line (an-gy), lines draw along their path (an-draw),
//   areas reveal left to right (an-clip), dots pop in last (an-pop), labels fade in after their mark.
// With motion off the markup is identical; it is simply already there.
//
// svg(width, height, data, pos, uid) returns the SVG markup for frame `pos`.
// tip(data, pos, frac) returns tooltip HTML for a pointer at fraction `frac` across the plot.

const pct = (v, d = 1) => (v == null || !isFinite(v) ? "–" : (v * 100).toFixed(d) + "%");
const NAMES = { calm: "Calm", stress: "Stress", crisis: "Crisis", single: "Single", unconditional: "Unconditional",
  equity: "Equity", rates: "Rates", credit: "Credit", real: "Real Assets", fx: "FX", custom: "Custom" };
const pretty = (n) => NAMES[n] || (String(n).charAt(0).toUpperCase() + String(n).slice(1).replace(/_/g, " "));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f2 = (x) => (Math.round(x * 100) / 100).toString();
const C = (name) => `var(--pv-${name})`;

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const e = span / n / step0;
  const step = step0 * (e > 5 ? 10 : e > 2 ? 5 : e > 1 ? 2 : 1);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}

// Scales, gridlines and tick labels. Axes do not animate: they are the frame the data arrives into.
function axes(W, H, xr, yr, fx, fy, pad = { l: 58, r: 18, t: 28, b: 30 }, opts = {}) {
  const b = { x0: pad.l, x1: W - pad.r, y0: pad.t, y1: H - pad.b };
  const X = (v) => b.x0 + ((v - xr[0]) / (xr[1] - xr[0] || 1)) * (b.x1 - b.x0);
  const Y = (v) => b.y1 - ((v - yr[0]) / (yr[1] - yr[0] || 1)) * (b.y1 - b.y0);
  let s = "";
  if (!opts.noY) for (const v of niceTicks(yr[0], yr[1], 5)) {
    s += `<line x1="${f2(b.x0)}" x2="${f2(b.x1)}" y1="${f2(Y(v))}" y2="${f2(Y(v))}" style="stroke:${C("line")}"/>`;
    s += `<text x="${f2(b.x0 - 8)}" y="${f2(Y(v) + 4)}" text-anchor="end" class="pv-tick">${esc(fy(v))}</text>`;
  }
  for (const v of (opts.xTicks || niceTicks(xr[0], xr[1], 6))) {
    s += `<text x="${f2(X(v))}" y="${H - 10}" text-anchor="middle" class="pv-tick">${esc(fx(v))}</text>`;
  }
  return { X, Y, b, s };
}

// Path through points; a null value breaks the line instead of drawing it as zero.
function pathD(pts) {
  let d = "", pen = false;
  for (const p of pts) {
    if (!p || p[1] == null || !isFinite(p[1])) { pen = false; continue; }
    d += (pen ? "L" : "M") + f2(p[0]) + " " + f2(p[1]);
    pen = true;
  }
  return d;
}
const areaD = (upper, lower) =>
  pathD(upper) + lower.slice().reverse().map((p) => "L" + f2(p[0]) + " " + f2(p[1])).join("") + "Z";

const line = (pts, color, width = 2, cls = "an-draw", delay = 0) =>
  `<path d="${pathD(pts)}" pathLength="1" class="${cls}" style="fill:none;stroke:${color};stroke-width:${width};` +
  `stroke-linejoin:round;animation-delay:${delay}ms"/>`;
const dashed = (pts, color, width = 1.5, dash = "6 5") =>
  `<path d="${pathD(pts)}" style="fill:none;stroke:${color};stroke-width:${width};stroke-dasharray:${dash}"/>`;
const dot = (x, y, r, color, delay, ring = true) =>
  `<circle cx="${f2(x)}" cy="${f2(y)}" r="${r}" class="an-pop" style="fill:${color};` +
  (ring ? `stroke:${C("surface")};stroke-width:2;` : "") + `animation-delay:${delay}ms"/>`;
const label = (x, y, text, anchor, delay, cls = "pv-cap") =>
  `<text x="${f2(x)}" y="${f2(y)}" text-anchor="${anchor}" class="${cls} an-fade" style="animation-delay:${delay}ms">${esc(text)}</text>`;

// Left-to-right reveal for areas and dashed lines (which cannot use the stroke-dashoffset draw).
const clipDef = (id, b, H) =>
  `<clipPath id="${id}"><rect class="an-clip" x="${f2(b.x0 - 2)}" y="0" width="${f2(b.x1 - b.x0 + 4)}" height="${H}" ` +
  `style="transform-origin:${f2(b.x0 - 2)}px 0"/></clipPath>`;
const hit = (b) => `<rect class="pv-hit" x="${f2(b.x0)}" y="${f2(b.y0)}" width="${f2(b.x1 - b.x0)}" height="${f2(b.y1 - b.y0)}"/>`;
const open = (W, H, title) => `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(title)}">`;

function legend(h, items) {
  return h("div", { class: "pv-legend" }, ...items.map(([name, color, dash]) =>
    h("span", {}, (() => { const i = h("i", dash ? { class: "pv-dash" } : {}); i.style.background = color;
      i.style.color = color; return i; })(), name)));
}

// ===================================================================== 1. forecast fan
const fan = {
  file: "fan.json", title: "Forecast Fan",
  blurb: "At each forecast date the Monte Carlo simulates the engine's target weights for the next month. The bands " +
    "are its 90% and 50% ranges and the blue line its median; the orange line is what those same weights actually did, " +
    "drawn in after the forecast. Pick a date to see that month's forecast and outcome.",
  labels: (d) => [d.label, "Gross of trading costs and financing", "Raw Monte Carlo bands (before the engine's self-correction)"],
  legend: (h) => legend(h, [["90% Range", C("band1")], ["50% Range", C("band2")], ["Median", C("s1")], ["Realised", C("s2")]]),
  count: (d) => d.frames.length,
  dateOf: (d, i) => (d.frames[i] ? d.frames[i].date : ""),
  svg(W, H, d, i, uid) {
    const f = d.frames[i];
    if (!f) return "";
    const n = f.realized.length, B = f.bands;
    const lo = Math.min(...B["5"], ...f.realized), hi = Math.max(...B["95"], ...f.realized);
    const pad = (hi - lo) * 0.08 || 0.01;
    const g = axes(W, H, [0, n - 1], [lo - pad - 1, hi + pad - 1], (v) => v + "d", (v) => pct(v, 1));
    const P = (arr) => arr.map((v, k) => [g.X(k), g.Y(v - 1)]);
    const last = P(f.realized)[n - 1];
    return open(W, H, this.title) + `<defs>${clipDef(uid + "c", g.b, H)}</defs>` + g.s +
      dashed([[g.X(0), g.Y(0)], [g.X(n - 1), g.Y(0)]], C("muted"), 1, "3 4") +
      `<g clip-path="url(#${uid}c)">` +
      `<path d="${areaD(P(B["95"]), P(B["5"]))}" style="fill:${C("band1")}"/>` +
      `<path d="${areaD(P(B["75"]), P(B["25"]))}" style="fill:${C("band2")}"/></g>` +
      line(P(B["50"]), C("s1"), 2, "an-draw", 0) +
      line(P(f.realized), C("s2"), 2.5, "an-draw", 450) +
      dot(last[0], last[1], 5, C("s2"), 1300) +
      label(g.b.x0, 17, `Forecast made ${f.date} · next ${n - 1} trading days`, "start", 0) +
      label(g.b.x1, 17, `Outcome landed at the ${Math.round(f.pit * 100)}th percentile of the forecast`, "end", 1400, "pv-cap pv-strong") +
      hit(g.b) + "</svg>";
  },
  tip(d, i, frac) {
    const f = d.frames[i];
    const n = f.realized.length, k = Math.round(frac * (n - 1));
    return `<b>Day ${k}</b><br>Realised ${pct(f.realized[k] - 1)}<br>Median ${pct(f.bands["50"][k] - 1)}<br>` +
      `50% range ${pct(f.bands["25"][k] - 1)} to ${pct(f.bands["75"][k] - 1)}<br>90% range ${pct(f.bands["5"][k] - 1)} to ${pct(f.bands["95"][k] - 1)}`;
  },
};

// ===================================================================== 2. calibration (PIT)
const calibration = {
  file: "calibration.json", title: "Calibration: Where Outcomes Landed in Their Forecasts",
  blurb: "For every past forecast, the percentile at which the real outcome landed inside the simulated distribution " +
    "(the probability integral transform, PIT). If the forecasts are honest, outcomes land evenly across percentiles " +
    "and the bars sit on the dashed uniform line. Bars piling up at both ends mean the forecast tails were too thin; " +
    "a middle-heavy shape means they were too fat. Pick a date to see the picture as it stood then.",
  labels: (d) => [d.label, `${d.horizon_days}-day forecasts, non-overlapping`, "Gross of costs"],
  legend: (h) => legend(h, [["Middle 80% of Percentiles", C("s1")], ["Outer 10% Each Side", C("s2")],
    ["Uniform (Perfect Calibration)", C("text2"), true]]),
  count: (d) => d.frames.length,
  dateOf: (d, i) => (d.frames[i] ? d.frames[i].date : ""),
  svg(W, H, d, i) {
    const a = d.frames[i];
    if (!a) return "";
    const bins = d.bins;
    const dens = a.counts.map((c) => c / (a.n / bins));
    const ymax = Math.max(2, Math.max(...dens) * 1.15);
    const g = axes(W, H, [0, 1], [0, ymax], (v) => Math.round(v * 100) + "th", (v) => v.toFixed(1) + "×");
    const bw = (g.b.x1 - g.b.x0) / bins;
    let bars = "";
    dens.forEach((v, k) => {
      const lo = k / bins, hi = (k + 1) / bins;
      const tail = lo < d.tail - 1e-9 || hi > 1 - d.tail + 1e-9;
      const y = g.Y(v), h = Math.max(g.Y(0) - y, 0);
      bars += `<rect x="${f2(g.X(lo) + 1.5)}" y="${f2(y)}" width="${f2(bw - 3)}" height="${f2(h)}" rx="3" class="an-gy" ` +
        `style="fill:${tail ? C("s2") : C("s1")};transform-origin:bottom;animation-delay:${k * 140}ms"/>`;
    });
    const end = bins * 140 + 200;
    const x = g.X(a.pit);
    return open(W, H, this.title) + g.s + bars +
      `<path d="M${f2(g.b.x0)} ${f2(g.Y(1))}H${f2(g.b.x1)}" class="an-fade" style="fill:none;stroke:${C("text2")};` +
      `stroke-width:1.5;stroke-dasharray:6 5;animation-delay:${end}ms"/>` +
      `<path d="M${f2(x)} ${f2(g.b.y1 + 2)}l-5 8h10z" class="an-pop" style="fill:${C("text")};animation-delay:${end}ms"/>` +
      label(g.b.x0, 17, `${a.n} forecasts up to ${a.date} · latest outcome at the ${Math.round(a.pit * 100)}th percentile (▲)`, "start", 0) +
      label(g.b.x1, 17, "Density relative to uniform", "end", 0) + hit(g.b) + "</svg>";
  },
  tip(d, i, frac) {
    const a = d.frames[i], k = Math.min(d.bins - 1, Math.floor(frac * d.bins));
    const lo = Math.round((k / d.bins) * 100), hi = Math.round(((k + 1) / d.bins) * 100);
    return `<b>${lo}th to ${hi}th percentile</b><br>${a.counts[k]} of ${a.n} outcomes (${pct(a.counts[k] / a.n, 0)}; uniform ${pct(1 / d.bins, 0)})`;
  },
  below(el, d, i, h) {
    const f = d.frames[i];
    if (!f) return;
    const exp = pct(f.tail_expected, 0), got = pct(f.tail_share, 0), p = f.tail_p == null ? "–" : f.tail_p.toFixed(3);
    const msg = [...{
      ok: ["Consistent with honest forecasts", `${got} of outcomes landed in the outer tails, against ${exp} expected (p = ${p}).`],
      thin: ["Tails too thin", `${got} of outcomes landed in the outer tails, against ${exp} expected (p = ${p}): the forecast ranges were too narrow.`],
      fat: ["Tails too fat", `Only ${got} of outcomes landed in the outer tails, against ${exp} expected (p = ${p}): the forecast ranges were wider than needed.`],
      insufficient: ["Too few forecasts to judge", `${f.n} so far; the flag starts at 30.`],
    }[f.flag]];
    let color = { ok: C("good"), thin: C("crit"), fat: C("warn"), insufficient: C("muted") }[f.flag];
    // the tail test can pass while the full distribution still fails uniformity: say so rather than "honest"
    const shapeOff = f.flag === "ok" && f.ks_p != null && f.ks_p < 0.05;
    if (shapeOff) {
      msg[0] = "Tails fine, shape off";
      msg[1] = `${got} of outcomes landed in the outer tails, against ${exp} expected (p = ${p}), but the outcomes are not ` +
        "spread evenly across the middle percentiles, so the forecast's centre or spread was off.";
      color = C("warn");
    }
    el.textContent = "";
    const box = h("div", { class: "pv-flag" }, h("b", { text: msg[0] + ". " }), msg[1],
      f.ks_p != null ? ` Kolmogorov-Smirnov test of uniformity: p = ${f.ks_p.toFixed(3)}.` : "");
    box.style.borderLeft = `4px solid ${color}`;
    el.append(box);
  },
};

// ===================================================================== 3. regime ridgeline
const regimes = {
  file: "regimes.json", title: "Regime Ridgeline",
  blurb: "The engine's portfolio, simulated forward one month starting in each market regime (then free to switch " +
    "regimes as the fitted model allows). The mint ridge is the regime the model thinks we are in on that date; the " +
    "dashed gold line is the unconditional distribution, starting from the long-run mix of regimes. Pick a date to see " +
    "how the regime picture looked then.",
  labels: (d) => [d.label, "Gross of trading costs and financing", `${(d.n_paths || 0).toLocaleString()} paths per ridge`],
  legend: (h) => legend(h, [["Current Regime", C("mint")], ["Other Regimes", C("neutral")], ["Unconditional", C("gold"), true]]),
  count: (d) => d.frames.length,
  dateOf: (d, i) => (d.frames[i] ? d.frames[i].date : ""),
  svg(W, H, d, i, uid) {
    const a = d.frames[i];
    if (!a) return "";
    const grid = d.grid, K = a.ridges.length, cur = a.current;
    const pad = { l: 120, r: 18, t: 30, b: 30 };
    const ticks = niceTicks(grid[0], grid[grid.length - 1], 6);
    const g = axes(W, H, [grid[0], grid[grid.length - 1]], [0, 1], (v) => pct(v, 0), () => "", pad, { noY: true, xTicks: ticks });
    const rowH = (g.b.y1 - g.b.y0) / (K + 0.6);
    const peak = Math.max(...a.unconditional.density, ...a.ridges.flatMap((r) => r.density));
    const amp = rowH * 1.55;
    let rows = "", labels = "";
    a.ridges.forEach((r, k) => {
      const base = g.b.y0 + rowH * (k + 1.4);
      const pts = r.density.map((v, j) => [g.X(grid[j]), base - (v / peak) * amp]);
      const flat = grid.map((x) => [g.X(x), base]);
      const un = a.unconditional.density.map((v, j) => [g.X(grid[j]), base - (v / peak) * amp]);
      const isCur = k === cur;
      rows += `<g class="an-rise" style="animation-delay:${k * 90}ms">` +
        `<path d="${areaD(pts, flat)}" style="fill:${isCur ? C("mint") : C("neutral")};fill-opacity:${isCur ? 0.6 : 0.35}"/>` +
        `<path d="${pathD(pts)}" style="fill:none;stroke:${isCur ? C("mint") : C("muted")};stroke-width:${isCur ? 2.5 : 1.2}"/>` +
        `<path d="${pathD(un)}" style="fill:none;stroke:${C("gold")};stroke-width:1.8;stroke-dasharray:6 4"/>` +
        `<path d="M${f2(g.b.x0)} ${f2(base)}H${f2(g.b.x1)}" style="stroke:${C("line")}"/></g>`;
      labels += `<text x="${pad.l - 12}" y="${f2(base - 6)}" text-anchor="end" class="${isCur ? "pv-rowlab pv-strong" : "pv-rowlab"} an-fade" ` +
        `style="animation-delay:${k * 90 + 200}ms">${esc(pretty(r.name))}</text>` +
        `<text x="${pad.l - 12}" y="${f2(base + 10)}" text-anchor="end" class="pv-tick an-fade" ` +
        `style="animation-delay:${k * 90 + 200}ms">p = ${pct(a.probs[k], 0)}</text>`;
    });
    const z = g.X(0);
    return open(W, H, this.title) + `<defs>${clipDef(uid + "c", g.b, H)}</defs>` + g.s +
      `<path d="M${f2(z)} ${f2(g.b.y0)}V${f2(g.b.y1)}" style="stroke:${C("line")}"/>` +
      `<g clip-path="url(#${uid}c)">${rows}</g>` + labels +
      label(g.b.x0, 17, `${a.date} · ${d.horizon_days}-day return over cash`, "start", 0) + "</svg>";
  },
  below(el, d, i, h) {
    const f = d.frames[i];
    if (!f) return;
    el.textContent = "";
    const parts = f.ridges.concat([f.unconditional]).map((r) =>
      `${pretty(r.name)}: median ${pct(r.q[1])}, 5th to 95th percentile ${pct(r.q[0])} to ${pct(r.q[2])}`);
    el.append(h("p", { class: "pv-p", text: parts.join(" · ") }));
  },
};

// ===================================================================== 4. rolling efficient frontier
const frontier = {
  file: "frontier.json", title: "Rolling Efficient Frontier",
  blurb: "Each date: the highest return achievable at each volatility from the assets then tradable, using the " +
    "trailing year of data, Ledoit-Wolf shrinkage covariance, Bayes-Stein shrunk average returns and the engine's own " +
    "leverage, position and shorting limits. The frontier is built from the same past data it is judged on, so it is " +
    "an in-sample best case no real portfolio can be expected to reach. The orange dot is where the engine's actual " +
    "holdings sat on the same scale, with the previous six dates fading behind it.",
  labels: (d) => [d.label, "In-sample, backward-looking", "Gross of costs",
    `Limits: ${d.constraints.allow_short ? "shorts allowed" : "long only"}, gross ≤ ${d.constraints.gross_cap}×, per asset ≤ ${pct(d.constraints.asset_cap, 0)}`],
  legend: (h) => legend(h, [["Frontier", C("s1")], ["Best Sharpe Point", C("s4")], ["Engine Portfolio", C("s2")]]),
  count: (d) => d.frames.length,
  dateOf: (d, i) => (d.frames[i] ? d.frames[i].date : ""),
  range(d) {
    if (d._rng) return d._rng;
    let xm = 0, ym = 0, yl = 0;
    for (const f of d.frames) {
      for (const [x, y] of f.frontier) { xm = Math.max(xm, x); ym = Math.max(ym, y); }
      xm = Math.max(xm, f.portfolio[0]); ym = Math.max(ym, f.portfolio[1]); yl = Math.min(yl, f.portfolio[1]);
    }
    return (d._rng = { x: [0, xm * 1.08], y: [yl - 0.02, ym * 1.05] });
  },
  svg(W, H, d, i) {
    const a = d.frames[i];
    if (!a) return "";
    const r = this.range(d);
    const g = axes(W, H, r.x, r.y, (v) => pct(v, 0), (v) => pct(v, 0));
    let trail = "";
    const first = Math.max(0, i - 6);
    for (let k = first; k < i; k++) {
      const p = d.frames[k].portfolio;
      trail += `<circle cx="${f2(g.X(p[0]))}" cy="${f2(g.Y(p[1]))}" r="3.5" class="an-pop" ` +
        `style="fill:${C("s2")};fill-opacity:${(0.15 + 0.1 * (k - first)).toFixed(2)};animation-delay:${900 + Math.min(k - first, 60) * 8}ms"/>`;
    }
    return open(W, H, this.title) + g.s +
      dashed([[g.X(0), g.Y(0)], [g.X(r.x[1]), g.Y(0)]], C("muted"), 1, "3 4") +
      line(a.frontier.map(([x, y]) => [g.X(x), g.Y(y)]), C("s1"), 2.5, "an-draw", 0) +
      // diamond as a plain path: a rotate() attribute would be re-centred by transform-box:fill-box and land off the frontier
      ((tx, ty) => `<path d="M${f2(tx)} ${f2(ty - 7)}L${f2(tx + 7)} ${f2(ty)}L${f2(tx)} ${f2(ty + 7)}L${f2(tx - 7)} ${f2(ty)}Z" ` +
        `class="an-pop" style="fill:${C("s4")};animation-delay:900ms"/>`)(g.X(a.tangency[0]), g.Y(a.tangency[1])) +
      trail + dot(g.X(a.portfolio[0]), g.Y(a.portfolio[1]), 7, C("s2"), 1000) +
      label(g.b.x0, 17, `${a.date} · trailing ${d.window_days} days · x: volatility, y: excess return per year`, "start", 0) +
      "</svg>";
  },
  below(el, d, i, h) {
    const f = d.frames[i];
    if (!f) return;
    el.textContent = "";
    el.append(h("p", { class: "pv-p", text:
      `${f.n_assets} assets · covariance shrinkage ${pct(f.shrinkage, 0)} · mean shrinkage ${pct(f.mean_shrinkage, 0)} · ` +
      `engine portfolio: ${pct(f.portfolio[0])} volatility, ${pct(f.portfolio[1])} trailing excess return · ` +
      `best-Sharpe point: ${pct(f.tangency[0])}, ${pct(f.tangency[1])}` }));
  },
};

// ===================================================================== 5. exposure
const CLASS_COLORS = ["s1", "s2", "s3", "s4", "s5"];
const exposure = {
  file: "exposure.json", title: "Exposure Over Time",
  blurb: "The engine's positions by asset class as a share of equity: longs stacked above zero, shorts below. The solid " +
    "line is gross exposure (long plus short, which is also the leverage multiple), the dashed line net exposure, and " +
    "the red dashed line the account's gross cap. Hover to read any date; the slider marks a date and shows its breakdown.",
  labels: (d) => [d.label, "Weights, not returns"],
  legend: null,
  count: (d) => d.dates.length,
  dateOf: (d, i) => d.dates[i] || "",
  svg(W, H, d, i, uid) {
    const n = d.dates.length, cls = d.classes;
    let ymax = d.gross_cap, ymin = 0;
    for (let t = 0; t < n; t++) {
      ymax = Math.max(ymax, d.gross[t]);
      ymin = Math.min(ymin, cls.reduce((s, c) => s + d.short[c][t], 0));
    }
    const years = [];
    for (let t = 0; t < n; t++) if (t === 0 || d.dates[t].slice(0, 4) !== d.dates[t - 1].slice(0, 4)) years.push(t);
    const step = Math.max(1, Math.ceil(years.length / 8));
    const g = axes(W, H, [0, n - 1], [ymin * 1.08, ymax * 1.08], (v) => (d.dates[Math.round(v)] || "").slice(0, 4),
      (v) => v.toFixed(1) + "×", undefined, { xTicks: years.filter((_, k) => k % step === 0) });
    // thin the series to about one point per pixel; areas look the same and the SVG stays small
    const every = Math.max(1, Math.floor(n / Math.max(200, g.b.x1 - g.b.x0)));
    const idx = [];
    for (let t = 0; t < n; t += every) idx.push(t);
    if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);
    let areas = "", up = idx.map(() => 0), dn = idx.map(() => 0);
    cls.forEach((c, k) => {
      const color = C(CLASS_COLORS[k % CLASS_COLORS.length]);
      const nu = idx.map((t, j) => up[j] + d.long[c][t]);
      areas += `<path d="${areaD(idx.map((t, j) => [g.X(t), g.Y(nu[j])]), idx.map((t, j) => [g.X(t), g.Y(up[j])]))}" style="fill:${color};fill-opacity:.78"/>`;
      up = nu;
      const nd = idx.map((t, j) => dn[j] + d.short[c][t]);
      areas += `<path d="${areaD(idx.map((t, j) => [g.X(t), g.Y(dn[j])]), idx.map((t, j) => [g.X(t), g.Y(nd[j])]))}" style="fill:${color};fill-opacity:.78"/>`;
      dn = nd;
    });
    const gross = idx.map((t) => [g.X(t), g.Y(d.gross[t])]);
    const net = idx.map((t) => [g.X(t), g.Y(d.net[t])]);
    return open(W, H, this.title) + `<defs>${clipDef(uid + "c", g.b, H)}</defs>` + g.s +
      `<g clip-path="url(#${uid}c)">${areas}` +
      `<path d="${pathD(gross)}" style="fill:none;stroke:${C("text")};stroke-width:1.6"/>` +
      `<path d="${pathD(net)}" style="fill:none;stroke:${C("text2")};stroke-width:1.4;stroke-dasharray:5 4"/></g>` +
      dashed([[g.X(0), g.Y(d.gross_cap)], [g.X(n - 1), g.Y(d.gross_cap)]], C("crit"), 1, "4 4") +
      `<path d="M${f2(g.b.x0)} ${f2(g.Y(0))}H${f2(g.b.x1)}" style="stroke:${C("muted")}"/>` +
      `<path d="M${f2(g.X(i))} ${f2(g.b.y0)}V${f2(g.b.y1)}" class="an-fade" style="stroke:${C("text")};animation-delay:900ms"/>` +
      label(g.b.x0, 17, `${d.dates[i]} · gross ${d.gross[i].toFixed(2)}× · net ${d.net[i] >= 0 ? "+" : ""}${d.net[i].toFixed(2)}×`, "start", 900) +
      hit(g.b) + "</svg>";
  },
  tip(d, i, frac) {
    const t = Math.round(frac * (d.dates.length - 1));
    const rows = d.classes.map((c) => `${pretty(c)}: ${pct(d.long[c][t], 0)} / ${pct(d.short[c][t], 0)}`).join("<br>");
    return `<b>${d.dates[t]}</b><br>Gross ${d.gross[t].toFixed(2)}× · net ${d.net[t] >= 0 ? "+" : ""}${d.net[t].toFixed(2)}×<br>${rows}`;
  },
  below(el, d, i, h) {
    el.textContent = "";
    const items = d.classes.map((c, k) => [`${pretty(c)}: long ${pct(d.long[c][i], 0)}, short ${pct(d.short[c][i], 0)}`,
      C(CLASS_COLORS[k % CLASS_COLORS.length])]);
    items.push(["Gross (Leverage)", C("text")], ["Net", C("text2"), true], ["Gross Cap", C("crit"), true]);
    el.append(legend(h, items));
  },
};

export const VIEWS = { fan, calibration, regimes, frontier, exposure };
