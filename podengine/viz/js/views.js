// The five chart views. Plain Canvas 2D, no external libraries. Every number drawn here
// was computed in Python (podengine/vizexport.py); this file only maps it to pixels.
// draw(ctx, width, height, data, pos, frac, theme): `pos` is the frame index, `frac` the
// progress toward the next frame (0..1) while animating.

const pct = (v, d = 1) => (v == null || !isFinite(v) ? "–" : (v * 100).toFixed(d) + "%");
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const NAMES = { calm: "Calm", stress: "Stress", crisis: "Crisis", single: "Single", unconditional: "Unconditional",
  equity: "Equity", rates: "Rates", credit: "Credit", real: "Real Assets", fx: "FX", custom: "Custom" };
const pretty = (n) => NAMES[n] || (String(n).charAt(0).toUpperCase() + String(n).slice(1).replace(/_/g, " "));

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const e = span / n / step0;
  const step = step0 * (e > 5 ? 10 : e > 2 ? 5 : e > 1 ? 2 : 1);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}

function frame(ctx, W, H, xr, yr, th, fx, fy, pad = { l: 58, r: 16, t: 26, b: 30 }) {
  const box = { x0: pad.l, x1: W - pad.r, y0: pad.t, y1: H - pad.b };
  const X = (v) => box.x0 + ((v - xr[0]) / (xr[1] - xr[0] || 1)) * (box.x1 - box.x0);
  const Y = (v) => box.y1 - ((v - yr[0]) / (yr[1] - yr[0] || 1)) * (box.y1 - box.y0);
  ctx.font = "12px -apple-system,Segoe UI,Roboto,sans-serif";
  ctx.lineWidth = 1;
  for (const v of niceTicks(yr[0], yr[1], 5)) {
    ctx.strokeStyle = th.line; ctx.beginPath(); ctx.moveTo(box.x0, Y(v)); ctx.lineTo(box.x1, Y(v)); ctx.stroke();
    ctx.fillStyle = th.muted; ctx.textAlign = "right"; ctx.fillText(fy(v), box.x0 - 8, Y(v) + 4);
  }
  ctx.textAlign = "center";
  for (const v of niceTicks(xr[0], xr[1], 6)) ctx.fillText(fx(v), X(v), H - 10);
  return { X, Y, box };
}

function line(ctx, pts, color, width = 2, dash = null, alpha = 1) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round";
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  ctx.restore();
}

function area(ctx, upper, lower, color, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha; ctx.fillStyle = color;
  ctx.beginPath();
  upper.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  for (let i = lower.length - 1; i >= 0; i--) ctx.lineTo(lower[i][0], lower[i][1]);
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

function caption(ctx, text, x, y, th, align = "left", color = null) {
  ctx.save();
  ctx.font = "12.5px -apple-system,Segoe UI,Roboto,sans-serif";
  ctx.fillStyle = color || th.text2; ctx.textAlign = align;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function legend(h, items) {
  return h("div", { class: "pv-legend" }, ...items.map(([name, color, dash]) =>
    h("span", {}, (() => { const i = h("i", dash ? { class: "pv-dash" } : {}); i.style.background = color;
      i.style.color = color; return i; })(), name)));
}

// ===================================================================== 1. forecast fan
const fan = {
  file: "fan.json", title: "Forecast Fan", staticFrac: 1, rate: 0.6,
  blurb: "At each forecast date the Monte Carlo simulates the engine's target weights for the next month. The bands " +
    "are its 90% and 50% ranges and its median; the orange line is what those same weights actually did. " +
    "Playing replays each month's outcome inside its forecast, then re-forecasts at the next date.",
  labels: (d) => [d.label, "Gross of trading costs and financing", "Raw Monte Carlo bands (before the engine's self-correction)"],
  legend: (h) => legend(h, [["90% range", "var(--pv-band1)"], ["50% range", "var(--pv-band2)"],
    ["Median", "var(--pv-s1)"], ["Realised", "var(--pv-s2)"]]),
  count: (d) => d.frames.length,
  dateOf: (d, i) => d.frames[i] ? d.frames[i].date : "",
  draw(ctx, W, H, d, i, frac, th) {
    const f = d.frames[i];
    if (!f) return;
    const n = f.realized.length;
    const b = f.bands;
    const show = Math.max(1, 1 + Math.round(clamp(frac, 0, 1) * (n - 1)));
    let lo = Math.min(...b["5"], ...f.realized), hi = Math.max(...b["95"], ...f.realized);
    const pad = (hi - lo) * 0.08 || 0.01;
    const g = frame(ctx, W, H, [0, n - 1], [lo - pad - 1, hi + pad - 1], th, (v) => v + "d", (v) => pct(v, 1));
    const P = (arr) => arr.map((v, k) => [g.X(k), g.Y(v - 1)]);
    area(ctx, P(b["95"]), P(b["5"]), th.band1, 0.9);
    area(ctx, P(b["75"]), P(b["25"]), th.band2, 0.9);
    line(ctx, P(b["50"]), th.s1, 2);
    line(ctx, [[g.X(0), g.Y(0)], [g.X(n - 1), g.Y(0)]], th.muted, 1, [3, 4]);
    const rp = P(f.realized.slice(0, show));
    line(ctx, rp, th.s2, 2.5);
    const last = rp[rp.length - 1];
    ctx.fillStyle = th.s2; ctx.beginPath(); ctx.arc(last[0], last[1], 4.5, 0, 7); ctx.fill();
    caption(ctx, `Forecast made ${f.date} · next ${n - 1} trading days`, g.box.x0, 16, th);
    if (show === n) caption(ctx, `Outcome landed at the ${Math.round(f.pit * 100)}th percentile of the forecast`,
      g.box.x1, 16, th, "right", th.text);
  },
};

// ===================================================================== 2. calibration (PIT)
const calibration = {
  file: "calibration.json", title: "Calibration: Where Outcomes Landed in Their Forecasts", staticFrac: 0, rate: 4, animated: false,
  blurb: "For every past forecast, the percentile at which the real outcome landed inside the simulated distribution " +
    "(the probability integral transform, PIT). If the forecasts are honest, outcomes land evenly across percentiles " +
    "and the bars sit on the dashed uniform line. Bars piling up at both ends mean the forecast tails were too thin; " +
    "an empty middle-heavy shape means they were too fat. Move the slider to see the picture as it stood at any past date.",
  labels: (d) => [d.label, `${d.horizon_days}-day forecasts, non-overlapping`, "Gross of costs"],
  legend: (h) => legend(h, [["Middle 80% of percentiles", "var(--pv-s1)"], ["Outer 10% each side", "var(--pv-s2)"],
    ["Uniform (perfect calibration)", "var(--pv-text2)", true]]),
  count: (d) => d.frames.length,
  dateOf: (d, i) => d.frames[i] ? d.frames[i].date : "",
  draw(ctx, W, H, d, i, frac, th) {
    const a = d.frames[i], b = d.frames[Math.min(i + 1, d.frames.length - 1)];
    if (!a) return;
    const bins = d.bins;
    const dens = a.counts.map((c, k) => {
      const da = c / (a.n / bins), db = b.counts[k] / (b.n / bins);
      return lerp(da, db, frac);
    });
    const ymax = Math.max(2, Math.max(...dens) * 1.15);
    const g = frame(ctx, W, H, [0, 1], [0, ymax], th, (v) => Math.round(v * 100) + "th", (v) => v.toFixed(1) + "×");
    const bw = (g.box.x1 - g.box.x0) / bins;
    dens.forEach((v, k) => {
      const lo = k / bins, hi = (k + 1) / bins;
      const tail = lo < d.tail - 1e-9 || hi > 1 - d.tail + 1e-9;
      ctx.fillStyle = tail ? th.s2 : th.s1;
      const x = g.X(lo) + 1, y = g.Y(v);
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, y, bw - 2, g.Y(0) - y, [4, 4, 0, 0]); else ctx.rect(x, y, bw - 2, g.Y(0) - y);
      ctx.fill();
    });
    line(ctx, [[g.X(0), g.Y(1)], [g.X(1), g.Y(1)]], th.text2, 1.5, [6, 5]);
    const x = g.X(a.pit);
    ctx.fillStyle = th.text;
    ctx.beginPath(); ctx.moveTo(x, g.box.y1 + 1); ctx.lineTo(x - 5, g.box.y1 + 9); ctx.lineTo(x + 5, g.box.y1 + 9); ctx.fill();
    caption(ctx, `${a.n} forecasts up to ${a.date} · latest outcome at the ${Math.round(a.pit * 100)}th percentile (▲)`,
      g.box.x0, 16, th);
    caption(ctx, "density relative to uniform", g.box.x1, 16, th, "right");
  },
  below(el, d, i, h) {
    const f = d.frames[i];
    if (!f) return;
    const exp = pct(f.tail_expected, 0), got = pct(f.tail_share, 0), p = f.tail_p == null ? "–" : f.tail_p.toFixed(3);
    const msg = {
      ok: ["Consistent with honest forecasts", `${got} of outcomes landed in the outer tails, against ${exp} expected (p = ${p}).`],
      thin: ["Tails too thin", `${got} of outcomes landed in the outer tails, against ${exp} expected (p = ${p}): the forecast ranges were too narrow.`],
      fat: ["Tails too fat", `Only ${got} of outcomes landed in the outer tails, against ${exp} expected (p = ${p}): the forecast ranges were wider than needed.`],
      insufficient: ["Too few forecasts to judge", `${f.n} so far; the flag starts at 30.`],
    }[f.flag];
    const color = { ok: "var(--pv-good)", thin: "var(--pv-crit)", fat: "var(--pv-warn)", insufficient: "var(--pv-muted)" }[f.flag];
    el.textContent = "";
    const box = h("div", { class: "pv-flag" }, h("b", { text: msg[0] + ". " }), msg[1],
      f.ks_p != null ? ` Kolmogorov-Smirnov test of uniformity: p = ${f.ks_p.toFixed(3)}.` : "");
    box.style.borderLeft = `4px solid ${color}`;
    el.append(box);
  },
};

// ===================================================================== 3. regime ridgeline
const regimes = {
  file: "regimes.json", title: "Regime Ridgeline", staticFrac: 0, rate: 1.5,
  blurb: "The engine's portfolio, simulated forward one month starting in each market regime (then free to switch " +
    "regimes as the fitted model allows). The mint ridge is the regime the model thinks we are in now; the dashed gold " +
    "line is the unconditional distribution, starting from the long-run mix of regimes. Playing moves through history " +
    "as regime probabilities shift.",
  labels: (d) => [d.label, "Gross of trading costs and financing", `${(d.n_paths || 0).toLocaleString()} paths per ridge`],
  legend: (h) => legend(h, [["Current regime", "var(--pv-mint)"], ["Other regimes", "var(--pv-neutral)"],
    ["Unconditional", "var(--pv-gold)", true]]),
  count: (d) => d.frames.length,
  dateOf: (d, i) => d.frames[i] ? d.frames[i].date : "",
  draw(ctx, W, H, d, i, frac, th) {
    const a = d.frames[i];
    if (!a) return;
    const b0 = d.frames[Math.min(i + 1, d.frames.length - 1)];
    const b = b0.ridges.length === a.ridges.length ? b0 : a;
    const mix = (x, y) => x.map((v, k) => lerp(v, y[k], frac));
    const ridges = a.ridges.map((r, k) => ({ name: pretty(r.name), dens: mix(r.density, b.ridges[k].density) }));
    const unc = mix(a.unconditional.density, b.unconditional.density);
    const probs = mix(a.probs, b.probs);
    const cur = probs.indexOf(Math.max(...probs));
    const grid = d.grid;
    const K = ridges.length;
    const pad = { l: 120, r: 16, t: 30, b: 30 };
    const g = frame(ctx, W, H, [grid[0], grid[grid.length - 1]], [0, 1], th, (v) => pct(v, 0), () => "", pad);
    ctx.clearRect(0, pad.t - 2, W, H - pad.t - pad.b + 4);      // ridgeline: no horizontal grid
    const rowH = (g.box.y1 - g.box.y0) / (K + 0.6);
    const peak = Math.max(...unc, ...ridges.flatMap((r) => r.dens));
    const amp = rowH * 1.55;
    const z = g.X(0);
    line(ctx, [[z, g.box.y0], [z, g.box.y1]], th.line, 1);
    ridges.forEach((r, k) => {
      const base = g.box.y0 + rowH * (k + 1.4);
      const pts = r.dens.map((v, j) => [g.X(grid[j]), base - (v / peak) * amp]);
      const flat = grid.map((x) => [g.X(x), base]);
      const isCur = k === cur;
      area(ctx, pts, flat, isCur ? th.mint : th.neutral, isCur ? 0.6 : 0.35);
      line(ctx, pts, isCur ? th.mint : th.muted, isCur ? 2.5 : 1.2);
      line(ctx, unc.map((v, j) => [g.X(grid[j]), base - (v / peak) * amp]), th.gold, 1.8, [6, 4]);
      line(ctx, [[g.box.x0, base], [g.box.x1, base]], th.line, 1);
      ctx.save();
      ctx.font = (isCur ? "600 " : "") + "13px -apple-system,Segoe UI,Roboto,sans-serif";
      ctx.fillStyle = isCur ? th.text : th.text2; ctx.textAlign = "right";
      ctx.fillText(r.name, pad.l - 12, base - 6);
      ctx.font = "12px -apple-system,Segoe UI,Roboto,sans-serif"; ctx.fillStyle = th.muted;
      ctx.fillText(`p = ${pct(probs[k], 0)}`, pad.l - 12, base + 10);
      ctx.restore();
    });
    caption(ctx, `${a.date} · ${d.horizon_days}-day return over cash`, g.box.x0, 16, th);
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
  file: "frontier.json", title: "Rolling Efficient Frontier", staticFrac: 0, rate: 1.5,
  blurb: "Each date: the highest return achievable at each volatility from the assets then tradable, using the " +
    "trailing year of data, Ledoit-Wolf shrinkage covariance, Bayes-Stein shrunk average returns and the engine's own " +
    "leverage, position and shorting limits. The frontier is built from the same past data it is judged on, so it is " +
    "an in-sample best case that no real portfolio can be expected to reach; the dot is where the engine's actual " +
    "holdings sat on the same scale.",
  labels: (d) => [d.label, "In-sample, backward-looking", "Gross of costs",
    `Limits: ${d.constraints.allow_short ? "shorts allowed" : "long only"}, gross ≤ ${d.constraints.gross_cap}×, per asset ≤ ${pct(d.constraints.asset_cap, 0)}`],
  legend: (h) => legend(h, [["Frontier", "var(--pv-s1)"], ["Best Sharpe point", "var(--pv-s4)"],
    ["Engine portfolio", "var(--pv-s2)"]]),
  count: (d) => d.frames.length,
  dateOf: (d, i) => d.frames[i] ? d.frames[i].date : "",
  _range(d) {
    if (d._rng) return d._rng;
    let xm = 0, ym = 0, yl = 0;
    for (const f of d.frames) {
      for (const [x, y] of f.frontier) { xm = Math.max(xm, x); ym = Math.max(ym, y); }
      xm = Math.max(xm, f.portfolio[0]); ym = Math.max(ym, f.portfolio[1]); yl = Math.min(yl, f.portfolio[1]);
    }
    return (d._rng = { x: [0, xm * 1.08], y: [yl - 0.02, ym * 1.05] });
  },
  draw(ctx, W, H, d, i, frac, th) {
    const a = d.frames[i];
    if (!a) return;
    const b = d.frames[Math.min(i + 1, d.frames.length - 1)];
    const r = this._range(d);
    const g = frame(ctx, W, H, r.x, r.y, th, (v) => pct(v, 0), (v) => pct(v, 0));
    line(ctx, [[g.X(0), g.Y(0)], [g.X(r.x[1]), g.Y(0)]], th.muted, 1, [3, 4]);
    line(ctx, a.frontier.map(([x, y]) => [g.X(x), g.Y(y)]), th.s1, 2.5, null, 1 - frac);
    if (frac > 0) line(ctx, b.frontier.map(([x, y]) => [g.X(x), g.Y(y)]), th.s1, 2.5, null, frac);
    const tx = lerp(a.tangency[0], b.tangency[0], frac), ty = lerp(a.tangency[1], b.tangency[1], frac);
    ctx.fillStyle = th.s4; ctx.save(); ctx.translate(g.X(tx), g.Y(ty)); ctx.rotate(Math.PI / 4);
    ctx.fillRect(-5, -5, 10, 10); ctx.restore();
    for (let k = Math.max(0, i - 6); k < i; k++) {              // fading trail of earlier positions
      const p = d.frames[k].portfolio;
      ctx.globalAlpha = 0.12 + 0.08 * (k - i + 7);
      ctx.fillStyle = th.s2; ctx.beginPath(); ctx.arc(g.X(p[0]), g.Y(p[1]), 3.5, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;
    const px = lerp(a.portfolio[0], b.portfolio[0], frac), py = lerp(a.portfolio[1], b.portfolio[1], frac);
    ctx.fillStyle = th.s2; ctx.strokeStyle = th.surface; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(g.X(px), g.Y(py), 7, 0, 7); ctx.fill(); ctx.stroke();
    caption(ctx, `${a.date} · trailing ${d.window_days} days · x: volatility, y: excess return per year`, g.box.x0, 16, th);
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
  file: "exposure.json", title: "Exposure Over Time", staticFrac: 0, rate: 30, animated: false,
  blurb: "The engine's positions by asset class as a share of equity: longs stacked above zero, shorts below. " +
    "The solid line is gross exposure (long plus short, which is also the leverage multiple), the dashed line net " +
    "exposure, and the red dashed line the account's gross cap. Move the slider to read the book on any date.",
  labels: (d) => [d.label, "Weights, not returns"],
  legend: null,
  count: (d) => d.dates.length,
  dateOf: (d, i) => d.dates[i] || "",
  draw(ctx, W, H, d, i, frac, th) {
    const n = d.dates.length;
    const cls = d.classes;
    const colors = cls.map((_, k) => th[CLASS_COLORS[k % CLASS_COLORS.length]]);
    let ymax = d.gross_cap, ymin = 0;
    for (let t = 0; t < n; t++) {
      ymax = Math.max(ymax, d.gross[t]);
      ymin = Math.min(ymin, cls.reduce((s, c) => s + d.short[c][t], 0));
    }
    const g = frame(ctx, W, H, [0, n - 1], [ymin * 1.08, ymax * 1.08], th,
      (v) => (d.dates[Math.round(v)] || "").slice(0, 4), (v) => v.toFixed(1) + "×");
    const draw = (upto, alpha) => {
      const idx = Array.from({ length: upto + 1 }, (_, t) => t);
      let up = idx.map(() => 0), dn = idx.map(() => 0);
      cls.forEach((c, k) => {
        const nu = idx.map((t) => up[t] + d.long[c][t]);
        area(ctx, idx.map((t) => [g.X(t), g.Y(nu[t])]), idx.map((t) => [g.X(t), g.Y(up[t])]), colors[k], 0.75 * alpha);
        up = nu;
        const nd = idx.map((t) => dn[t] + d.short[c][t]);
        area(ctx, idx.map((t) => [g.X(t), g.Y(dn[t])]), idx.map((t) => [g.X(t), g.Y(nd[t])]), colors[k], 0.75 * alpha);
        dn = nd;
      });
      line(ctx, idx.map((t) => [g.X(t), g.Y(d.gross[t])]), th.text, 1.6, null, alpha);
      line(ctx, idx.map((t) => [g.X(t), g.Y(d.net[t])]), th.text2, 1.4, [5, 4], alpha);
    };
    draw(n - 1, 0.22);                                          // whole history, faint
    draw(i, 1);                                                 // up to the cursor
    line(ctx, [[g.X(0), g.Y(d.gross_cap)], [g.X(n - 1), g.Y(d.gross_cap)]], th.crit, 1, [4, 4]);
    line(ctx, [[g.X(0), g.Y(0)], [g.X(n - 1), g.Y(0)]], th.muted, 1);
    line(ctx, [[g.X(i), g.box.y0], [g.X(i), g.box.y1]], th.text, 1);
    caption(ctx, `${d.dates[i]} · gross ${d.gross[i].toFixed(2)}× · net ${d.net[i] >= 0 ? "+" : ""}${d.net[i].toFixed(2)}×`, g.box.x0, 16, th);
  },
  below(el, d, i, h) {
    el.textContent = "";
    const items = d.classes.map((c, k) => [`${pretty(c)}: long ${pct(d.long[c][i], 0)}, short ${pct(d.short[c][i], 0)}`,
      `var(--pv-${CLASS_COLORS[k % CLASS_COLORS.length]})`]);
    items.push(["Gross (Leverage)", "var(--pv-text)"], ["Net", "var(--pv-text2)", true], ["Gross Cap", "var(--pv-crit)", true]);
    el.append(legend(h, items));
  },
};

export const VIEWS = { fan, calibration, regimes, frontier, exposure };
