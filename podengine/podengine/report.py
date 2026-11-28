"""Self-contained HTML report (no external files, works offline)."""
import html
import re
import json

import numpy as np
import pandas as pd

CSS = """
:root{color-scheme:light;--bg:#f6f6f4;--surface:#fcfcfb;--line:#e3e2de;--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--band1:#cde2fb;--band2:#9ec5f4;
--good:#0ca30c;--warn:#fab219;--crit:#d03b3b}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#111110;--surface:#1a1a19;--line:#32322f;--text:#f0efec;
--text2:#c3c2b7;--muted:#8a8984;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--band1:#184f95;--band2:#1c5cab}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 20px 60px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:38px 0 6px}h3{font-size:15px;margin:18px 0 6px}
p{margin:6px 0 10px;color:var(--text2);max-width:78ch}.sub{color:var(--text2);margin:0 0 14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:12px 0}
.banner{background:var(--surface);border:2px solid var(--crit);border-radius:10px;padding:12px 16px;margin:14px 0;color:var(--text)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin:12px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .k{font-size:12px;color:var(--text2)}.tile .v{font-size:22px;font-weight:600;margin-top:2px}.tile .n{font-size:12px;color:var(--muted);margin-top:2px}
.scroll{overflow:auto;max-height:440px;border-radius:6px}table{border-collapse:collapse;width:100%;font-size:13.5px}
thead th{position:sticky;top:0;background:var(--surface);z-index:1}
.subtabs{display:flex;gap:2px;overflow-x:auto;border-bottom:1px solid var(--line);margin:18px 0 4px;position:sticky;top:0;background:var(--bg);z-index:3}
.subtab{border:0;background:none;color:var(--text2);font:inherit;font-size:14px;padding:9px 12px;border-bottom:2px solid transparent;cursor:pointer;white-space:nowrap}
.subtab[aria-selected=true]{color:var(--text);border-bottom-color:var(--s1);font-weight:600}
.subpanel[hidden]{display:none}.subpanel>h2{margin-top:18px}
.motion-btn{position:fixed;top:12px;right:14px;z-index:6;display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);
background:var(--surface);color:var(--text2);border-radius:999px;padding:4px 11px;font:inherit;font-size:13px;cursor:pointer;
transition:border-color var(--dur-micro) var(--ease-out),color var(--dur-micro) var(--ease-out)}
.motion-btn::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--line)}
.motion-btn[aria-pressed=true]{color:var(--text)}.motion-btn[aria-pressed=true]::before{background:var(--good)}
.motion-btn[hidden]{display:none}
:root{--ease-out:cubic-bezier(0.16,1,0.3,1);--dur-micro:120ms;--dur-short:220ms;--dur-long:420ms;--dur-draw:900ms}
@keyframes grow-y{from{transform:scaleY(0)}}@keyframes grow-x{from{transform:scaleX(0)}}
@keyframes rise{from{opacity:0;transform:translateY(8px)}}@keyframes fade{from{opacity:0}}
@keyframes draw{from{stroke-dashoffset:1}to{stroke-dashoffset:0}}@keyframes pop{from{opacity:0;transform:scale(0.2)}}
.an-gy{transform-box:fill-box;animation:grow-y var(--dur-long) var(--ease-out) both}
.an-gx{animation:grow-x var(--dur-long) var(--ease-out) both}
.an-rise{animation:rise var(--dur-long) var(--ease-out) both}
.an-fade{animation:fade var(--dur-long) var(--ease-out) both}
.an-draw{stroke-dasharray:1;animation:draw var(--dur-draw) var(--ease-out) both}
.an-pop{transform-box:fill-box;transform-origin:center;animation:pop var(--dur-short) var(--ease-out) both}
.an-clip{transform-box:view-box;animation:grow-x var(--dur-draw) var(--ease-out) both}
.subpanel[data-drawn] *{animation:none!important}
[data-motion="off"] *,[data-motion="off"] *::before,[data-motion="off"] *::after{animation:none!important;transition:none!important}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:0.01ms!important;animation-iteration-count:1!important;animation-delay:0ms!important;transition-duration:0.01ms!important}}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--text2);font-weight:600}th:first-child,td:first-child{text-align:left}td.l,th.l{text-align:left;white-space:normal}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.chart{position:relative;width:100%}.chart svg{display:block;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--text2);margin:2px 0 6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px;
box-shadow:0 2px 10px rgba(0,0,0,.12);display:none;white-space:nowrap;z-index:5}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 9px;font-size:12px;color:var(--text2)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
small,.small{font-size:12.5px;color:var(--muted)}code{font-size:12.5px;background:var(--bg);padding:1px 5px;border-radius:4px}
"""

JS = r"""
const NS='http://www.w3.org/2000/svg';
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function fmt(v,f){if(v==null||isNaN(v))return '–';if(f==='pct')return (v*100).toFixed(1)+'%';if(f==='mult')return v.toFixed(2)+'x';return v.toFixed(2)}
function ticks(lo,hi,n){const st=Math.pow(10,Math.floor(Math.log10((hi-lo)/n)));const e=(hi-lo)/n/st;const s=st*(e>5?10:e>2?5:e>1?2:1);const o=[];for(let v=Math.ceil(lo/s)*s;v<=hi+1e-9;v+=s)o.push(v);return o}
function lineChart(id,d){const box=document.getElementById(id);if(!box)return;const W=1000,H=d.h||340,m={l:56,r:d.labels===false?16:120,t:10,b:26};
 const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img'},box);const n=d.x.length;const tr=v=>d.log?Math.log(v):v;
 let lo=Infinity,hi=-Infinity;const all=[...d.series];if(d.bands)d.bands.forEach(b=>{all.push({y:b.lo});all.push({y:b.hi})});
 all.forEach(s=>s.y.forEach(v=>{if(v!=null&&isFinite(v)){lo=Math.min(lo,tr(v));hi=Math.max(hi,tr(v))}}));const pad=(hi-lo)*0.05||1;lo-=pad;hi+=pad;
 const X=i=>m.l+(W-m.l-m.r)*i/(n-1),Y=v=>m.t+(H-m.t-m.b)*(1-(tr(v)-lo)/(hi-lo));
 let tk=d.log?[0.25,0.5,1,2,4,8,16,32,64,128].filter(v=>Math.log(v)>=lo&&Math.log(v)<=hi):ticks(lo,hi,5);
 tk.forEach(v=>{el('line',{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:'var(--line)','stroke-width':1},svg);
  const t=el('text',{x:m.l-8,y:Y(v)+4,'text-anchor':'end','font-size':12,fill:'var(--muted)'},svg);t.textContent=fmt(v,d.fmt)});
 const step=Math.max(1,Math.floor(n/6));for(let i=0;i<n;i+=step){const t=el('text',{x:X(i),y:H-6,'text-anchor':'middle','font-size':12,fill:'var(--muted)'},svg);t.textContent=d.xshort?d.xshort[i]:d.x[i]}
 if(d.bands){const cid=id+'-clip';const defs=el('defs',{},svg);const cp=el('clipPath',{id:cid},defs);
  el('rect',{x:m.l-2,y:0,width:W-m.l-m.r+4,height:H,class:'an-clip',style:`transform-origin:${m.l-2}px 0`},cp);
  const g=el('g',{'clip-path':`url(#${cid})`},svg);
  d.bands.forEach(b=>{let p='';b.hi.forEach((v,i)=>p+=(i?'L':'M')+X(i)+' '+Y(v));for(let i=n-1;i>=0;i--)p+='L'+X(i)+' '+Y(b.lo[i]);el('path',{d:p+'Z',fill:b.color,opacity:.75},g)})}
 const ends=[];d.series.forEach(s=>{let p='',pen=false;s.y.forEach((v,i)=>{if(v==null||!isFinite(v)){pen=false;return}p+=(pen?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1);pen=true});
  el('path',{d:p,pathLength:1,class:'an-draw',fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round'},svg);let j=n-1;while(j>0&&(s.y[j]==null))j--;ends.push({name:s.name,y:Y(s.y[j]),color:s.color})});
 if(d.labels!==false&&d.series.length<=5){ends.sort((a,b)=>a.y-b.y);for(let i=1;i<ends.length;i++)if(ends[i].y-ends[i-1].y<15)ends[i].y=ends[i-1].y+15;
  ends.forEach(e=>{el('circle',{cx:W-m.r+7,cy:e.y,r:4,fill:e.color,class:'an-pop',style:'animation-delay:900ms'},svg);const t=el('text',{x:W-m.r+15,y:e.y+4,'font-size':12,fill:'var(--text2)',class:'an-fade',style:'animation-delay:1000ms'},svg);t.textContent=e.name})}
 const cross=el('line',{y1:m.t,y2:H-m.b,stroke:'var(--muted)','stroke-width':1,visibility:'hidden'},svg);const tip=document.createElement('div');tip.className='tip';box.appendChild(tip);
 svg.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect();const px=(ev.clientX-r.left)/r.width*W;let i=Math.round((px-m.l)/(W-m.l-m.r)*(n-1));i=Math.max(0,Math.min(n-1,i));
  cross.setAttribute('x1',X(i));cross.setAttribute('x2',X(i));cross.setAttribute('visibility','visible');
  let h='<b>'+d.x[i]+'</b>';d.series.forEach(s=>{h+='<br><span class="dot" style="background:'+s.color+'"></span>'+s.name+': '+fmt(s.y[i],d.fmt)});
  if(d.bands)d.bands.forEach(b=>{h+='<br>'+b.name+': '+fmt(b.lo[i],d.fmt)+' to '+fmt(b.hi[i],d.fmt)});
  tip.innerHTML=h;tip.style.display='block';const left=(X(i)/W*r.width);tip.style.left=(left>r.width*0.6?left-tip.offsetWidth-12:left+12)+'px';tip.style.top='10px'});
 svg.addEventListener('mouseleave',()=>{cross.setAttribute('visibility','hidden');tip.style.display='none'})}
const motionOn=()=>document.documentElement.dataset.motion!=='off'&&!matchMedia('(prefers-reduced-motion: reduce)').matches;
function countUp(root){root.querySelectorAll('[data-count]').forEach(el=>{const m=el.dataset.count.match(/^([+−]?)(\d+(?:\.\d+)?)(.*)$/);
 if(!m)return;const end=+m[2],dec=(m[2].split('.')[1]||'').length,t0=performance.now();
 const step=t=>{const k=Math.min(1,(t-t0)/700),e=1-(1-k)**3;el.textContent=m[1]+(end*e).toFixed(dec)+m[3];if(k<1)requestAnimationFrame(step)};
 requestAnimationFrame(step)})}
function firstShow(p){if(!p||p.dataset.seen)return;p.dataset.seen='1';if(motionOn())countUp(p);
 setTimeout(()=>p.setAttribute('data-drawn',''),2600)}
document.addEventListener("click",e=>{const mb=e.target.closest(".motion-btn");if(mb){const on=document.documentElement.dataset.motion==='off';
  document.documentElement.dataset.motion=on?'on':'off';mb.setAttribute('aria-pressed',on?'true':'false');
  try{localStorage.setItem('motion',on?'on':'off')}catch(err){}return}
 const b=e.target.closest(".subtab");if(!b)return;
 document.querySelectorAll(".subtab").forEach(x=>x.setAttribute("aria-selected",x===b?"true":"false"));
 document.querySelectorAll(".subpanel").forEach(p=>p.hidden=p.id!==b.dataset.tab);window.scrollTo(0,0);
 firstShow(document.getElementById(b.dataset.tab))});
document.addEventListener("DOMContentLoaded",()=>{const mb=document.querySelector('.motion-btn');
 if(mb){if(window.top!==window)mb.hidden=true;mb.setAttribute('aria-pressed',document.documentElement.dataset.motion==='off'?'false':'true')}
 firstShow(document.querySelector('.subpanel:not([hidden])'))});
function barChart(id,d){const box=document.getElementById(id);if(!box)return;const W=500,rh=26,m={l:d.left||130,r:60,t:4,b:4},H=m.t+m.b+rh*d.labels.length;
 const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img',style:'max-width:560px'},box);const mx=Math.max(...d.values.map(Math.abs),1e-9),neg=Math.min(...d.values)<0;
 const x0=neg?m.l+(W-m.l-m.r)/2:m.l,sc=(neg?(W-m.l-m.r)/2:(W-m.l-m.r))/mx;el('line',{x1:x0,x2:x0,y1:m.t,y2:H-m.b,stroke:'var(--line)'},svg);
 const tip=document.createElement('div');tip.className='tip';box.appendChild(tip);
 d.labels.forEach((lb,i)=>{const v=d.values[i],y=m.t+i*rh,w=Math.abs(v)*sc;const t=el('text',{x:m.l-8,y:y+rh/2+4,'text-anchor':'end','font-size':12,fill:'var(--text2)'},svg);t.textContent=lb;
  const dl=Math.min(i,30)*18;
  const b=el('rect',{x:v<0?x0-w:x0,y:y+5,width:Math.max(w,1),height:rh-10,rx:3,fill:d.color||'var(--s1)',class:'an-gx',
   style:`transform-box:fill-box;transform-origin:${v<0?'right':'left'};animation-delay:${dl}ms`},svg);
  const tv=el('text',{x:v<0?x0-w-6:x0+w+6,y:y+rh/2+4,'text-anchor':v<0?'end':'start','font-size':12,fill:'var(--text)',class:'an-fade',style:`animation-delay:${dl+200}ms`},svg);tv.textContent=fmt(v,d.fmt);
  const hit=el('rect',{x:0,y:y,width:W,height:rh,fill:'transparent'},svg);hit.addEventListener('mousemove',()=>{tip.innerHTML='<b>'+lb+'</b>: '+fmt(v,d.fmt);tip.style.display='block';tip.style.left='40%';tip.style.top=(y/H*100)+'%'});
  hit.addEventListener('mouseleave',()=>tip.style.display='none')})}
"""


DISPLAY = {
    # pods
    "trend": "Trend", "value_momentum": "Value + Momentum", "defensive": "Defensive", "risk_parity": "Risk Parity",
    "reversal": "Reversal", "limits & band": "Limits & Band",
    # asset classes
    "equity": "Equity", "rates": "Rates", "credit": "Credit", "real": "Real Assets", "fx": "FX", "custom": "Custom",
    # regimes and states
    "calm": "Calm", "stress": "Stress", "crisis": "Crisis", "single": "Single", "unconditional": "Unconditional",
    "full": "Full", "halved": "Halved", "off": "Off", "benched": "Benched",
}
SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "per", "the", "to", "vs", "via", "is", "it", "x"}


def pretty(name):
    """Display name for an internal identifier (pod, asset class, regime, state)."""
    n = str(name)
    if n in DISPLAY:
        return DISPLAY[n]
    n = n.replace("_", " ")
    return n[:1].upper() + n[1:]


def title_case(text):
    """Capitalise the first letter of each word in a heading or label, keeping
    acronyms and units (VaR, ES, ETF, 1-day, 90%) as they are."""
    words = text.split(" ")
    out = []
    for i, w in enumerate(words):
        if not w or not w[0].isalpha() or (w[0].isupper()) or (i and w.lower() in SMALL_WORDS):
            out.append(w)
        else:
            out.append(w[0].upper() + w[1:])
    return " ".join(out)


def _title_tags(html_):
    """Title-case the text of headings, table headers and tile labels."""
    def fix(m):
        inner = re.sub(r"(^|>)([^<]+)", lambda t: t.group(1) + title_case(t.group(2)), m.group(2))
        return m.group(1) + inner + m.group(3)
    for pat in (r"(<h[123][^>]*>)(.*?)(</h[123]>)", r"(<th[^>]*>)(.*?)(</th>)", r"(<div class=k>)(.*?)(</div>)",
                r"(<span class=pv-tablabel>)(.*?)(</span>)"):
        html_ = re.sub(pat, fix, html_, flags=re.S)
    return html_


TAB_NAMES = [(r"^Today", "Today's Portfolio"), (r"^Pod scoreboard", "Pods"), (r"^Forecast ranges", "Forecast Ranges"),
             (r"^Walk-forward", "Backtest"), (r"^Is it real", "Validation"), (r"^Value at Risk", "Value at Risk"),
             (r"^Order sheet", "Order Sheet"), (r"^What this is", "About"), (r"^Your portfolio and", "Comparison"),
             (r"^Expected returns", "Views"), (r"^Trades", "Trades"), (r"^Limits", "Limits")]


def tabify(head_html, body_html):
    """Split the body at each <h2> into sub-tabs (one panel per section)."""
    parts = re.split(r"<h2>(.*?)</h2>", body_html, flags=re.S)
    pre, secs = parts[0], list(zip(parts[1::2], parts[2::2]))
    bar, panels = [], []
    for i, (h2, content) in enumerate(secs):
        clean = re.sub(r"^\d+\.\s*", "", re.sub("<[^>]+>", "", h2)).strip()
        label = next((lab for pat, lab in TAB_NAMES if re.search(pat, clean)), title_case(clean))
        sel = "true" if i == 0 else "false"
        bar.append(f"<button class=subtab role=tab aria-selected={sel} data-tab=t{i}><span class=pv-tablabel>{esc(label)}</span></button>")
        panels.append(f"<section class=subpanel id=t{i}{'' if i == 0 else ' hidden'}><h2>{clean}</h2>{content}</section>")
    return head_html + pre + "<nav class=subtabs role=tablist>" + "".join(bar) + "</nav>" + "".join(panels)


def pct(v, d=1):
    if v is None or not np.isfinite(v):
        return "–"
    v = 0.0 if abs(v) < 0.5 * 10 ** -(d + 2) else v        # no "-0.00%"
    return f"{v * 100:.{d}f}%"


def num(v, d=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def esc(s):
    return html.escape(str(s))


def table(headers, rows, left=()):
    h = "".join(f"<th{' class=l' if i in left else ''}>{esc(x)}</th>" for i, x in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td{' class=l' if i in left else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f"<div class=scroll><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>"


def tile(k, v, n=""):
    plain = re.sub(r"<[^>]+>", "", str(v))
    return (f"<div class=tile><div class=k>{esc(k)}</div><div class=v data-count=\"{html.escape(plain, quote=True)}\">{v}</div>"
            f"<div class=n>{n}</div></div>")


def legend(names, colors):
    return "<div class=legend>" + "".join(f"<span><i style='background:{c}'></i>{esc(n)}</span>"
                                           for n, c in zip(names, colors)) + "</div>"


COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"]
STATE_DOT = {"full": "var(--good)", "halved": "var(--warn)", "off": "var(--crit)", "benched": "var(--muted)"}


def _clean(a):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), 5) for v in a]


def build_report(b, path):
    info, cfg, f, val, mc = b["info"], b["cfg"], b["final"], b["validation"], b["mc"]
    st, ex, reg = b["stats"]["Engine"], b["exposure"], b["regime"]
    charts = []
    P = []
    P.append(f"<h1>Pod Engine Report</h1><p class=sub>As of {info['asof'].date()} · data: {esc(info['source'])} · "
             f"profile: <b>{esc(pretty(cfg['profile']))}</b> · {info['n_assets']} of {info['n_universe']} assets tradable · "
             f"backtest {info['first_trade'].date()} to {info['asof'].date()} ({info['years']:.1f} years)</p>")
    if info["synthetic"]:
        P.append("<div class=banner><b>Demo mode: this report runs on simulated data.</b> The prices were generated by "
                 "a random model so every part of the engine can be shown working offline. None of the performance "
                 "numbers below say anything about real markets. Run <code>python -m podengine run</code> (without "
                 "<code>--demo</code>) to use real prices.</div>")

    # ------------------------------------------------------------ today
    P.append("<h2>1. Today's portfolio</h2>")
    names = ["Calm", "Stress", "Crisis"][:len(reg["probs"])] if len(reg["probs"]) <= 3 else [f"State {i}" for i in range(len(reg["probs"]))]
    top = int(np.argmax(reg["probs"]))
    r5, r21 = mc["ranges"].get(5), mc["ranges"].get(21)
    rng_txt = lambda r: "–" if not r else f"{pct(r.get('adj90', r['raw90'])[0])} to {pct(r.get('adj90', r['raw90'])[1])}"
    binding = ", ".join(f["binding"]) if f["binding"] else "none"
    sc = f["scalars"]
    P.append("<div class=tiles>" + "".join([
        tile("Expected volatility", pct(f["ex_ante_vol"]), f"target {pct(cfg['vol_target'])} before adjustments"),
        tile("Gross leverage", f"{f['gross']:.2f}x", f"net {f['net']:+.2f}x · cap {cfg['gross_cap']:.1f}x"),
        tile("Market regime", f"{names[top]} {pct(reg['probs'][top], 0)}", f"read from {esc(reg['bench'])} returns"),
        tile("Turbulence", "–" if reg["turb"] is None else f"{reg['turb'] * 100:.0f}th pct", "cross-asset stress vs last 5 years"),
        tile("1-week range (90%)", rng_txt(r5), "self-corrected Monte Carlo"),
        tile("1-month range (90%)", rng_txt(r21), "self-corrected Monte Carlo"),
    ]) + "</div>")
    adj = [f"pods switched on: x{sc.get('pods_active', 1):.2f}"]
    if "kelly_cap" in sc:
        adj.append(f"Kelly cap binding at {pct(sc['kelly_cap'])}")
    adj += [f"turbulence: x{sc.get('turbulence', 1):.2f}", f"risk-model bias correction: ÷{sc.get('risk_model_bias', 1):.2f}"]
    P.append(f"<p>How the {pct(cfg['vol_target'])} target became {pct(f['ex_ante_vol'])}: " + "; ".join(adj) +
             f". Account limits binding: <b>{esc(binding)}</b>.</p>")

    cls = ex["by_cls"].sort_values("risk_share", ascending=False)
    P.append("<div class=grid2><div class=card><h3>Share of risk by asset class</h3><div class=chart id=c_cls></div>" +
             table(["Class", "Long", "Short", "Net", "Risk share"],
                   [[esc(pretty(r.cls)), pct(r.long), pct(r.short), pct(r.net), pct(r.risk_share)] for r in cls.itertuples()]) +
             "</div><div class=card><h3>Share of risk by pod</h3><div class=chart id=c_pod></div>"
             "<p class=small>A pod's share is how much of the portfolio's volatility comes from its positions. "
             "Negative means the pod is currently hedging the others. 'Limits & band' is the part of the portfolio that "
             "differs from the pods' combined wish because an account limit was binding.</p></div></div>")
    charts.append(("bar", "c_cls", {"labels": [pretty(c) for c in cls["cls"]], "values": _clean(cls["risk_share"]), "fmt": "pct"}))
    charts.append(("bar", "c_pod", {"labels": [pretty(x) for x in b["res"].pod_names] + ["Limits & Band"],
                                    "values": _clean(list(ex["pod_share"]) + [ex["resid_share"]]), "fmt": "pct"}))

    a = ex["assets"].reindex(ex["assets"]["weight"].abs().sort_values(ascending=False).index)
    a = a[a["weight"].abs() > 1e-4]
    P.append("<div class=card><h3>Target positions</h3>" + table(
        ["Ticker", "Name", "Class", "Weight", "Forecast vol", "Risk share"],
        [[esc(r.ticker), esc(r.name), esc(pretty(r.cls)), pct(r.weight), pct(r.vol), pct(r.risk_share)] for r in a.itertuples()],
        left=(1, 2)) + "<p class=small>Weights are fractions of account equity; negative = short. They can sum to more "
        "than 100% because low-volatility assets (bonds) need leverage to matter.</p></div>")

    # ------------------------------------------------------------ pods
    P.append("<h2>2. Pod scoreboard</h2><p>Each pod is an independent strategy run at the same stand-alone volatility "
             f"({pct(cfg['pod_vol'], 0)}). Its share of risk follows its shrunk Sharpe ratio: the out-of-sample record "
             "pulled toward a sceptical prior. A pod in drawdown is halved, then switched off, until it recovers. 'Benched' means the evidence for the pod's "
             "edge is at or below zero, so it gets no capital for now.</p>")
    rows = []
    for r in b["pod_table"].itertuples():
        dot = f"<span class=dot style='background:{STATE_DOT.get(r.state, 'var(--muted)')}'></span>{esc(pretty(r.state))}"
        rows.append([esc(pretty(r.pod)) + (" <span class=pill>Experimental</span>" if r.experimental else ""),
                     num(r.sharpe_raw), num(r.prior), f"{num(r.sharpe_shrunk)} ± {num(r.sharpe_sd)}",
                     pct(-r.drawdown), pct(r.max_dd), dot, pct(b["gov_avg"].get(r.pod, 1.0), 0),
                     pct(b["contrib_ann"].get(r.pod, 0.0))])
    P.append("<div class=card>" + table(["Pod", "Backtest Sharpe", "Prior", "Shrunk Sharpe", "Current drawdown",
                                         "Worst drawdown", "Status", "Avg. risk switch", "Added per year"], rows) +
             "<div class=chart id=c_pods></div>" + legend([pretty(c) for c in b["pods_cum"].columns], COLORS) +
             f"<p class=small>Chart: cumulative paper return of each pod at its own volatility target, after its own "
             f"trading and financing costs. 'Added per year' is the pod's share of the portfolio's actual P&L before "
             f"costs; the part caused by account limits, the trade band and price drift adds "
             f"{pct(b['contrib_ann'].get('limits & band', 0.0))} per year.</p>")
    for r in b["pod_table"].itertuples():
        P.append(f"<p class=small><b>{esc(pretty(r.pod))}</b>: {esc(r.idea)} <i>Source: {esc(r.source)}.</i></p>")
    P.append("</div>")
    pc = b["pods_cum"]
    charts.append(("line", "c_pods", {"x": [str(d.date()) for d in pc.index], "xshort": [str(d.year) for d in pc.index],
                                      "fmt": "pct", "h": 300,
                                      "series": [{"name": pretty(c), "color": COLORS[i % 5], "y": _clean(pc[c])}
                                                 for i, c in enumerate(pc.columns)]}))

    # ------------------------------------------------------------ ranges
    P.append("<h2>3. Forecast ranges and how honest they have been</h2><p>The Monte Carlo re-plays real historical "
             "days drawn from the regime it thinks we are in, scaled to today's volatility forecast. Then it is "
             "audited: at each past date the range was compared with what happened, and the quantile levels were "
             "nudged after every miss. Corrected coverage should sit close to the stated level.</p>")
    rows = []
    for key, label in (("port5", "Engine portfolio, 1 week"), ("port21", "Engine portfolio, 1 month"),
                       ("bench21", f"{reg['bench']} alone, 1 month")):
        c = mc["cal"][key]
        if c.get("n", 0):
            rows.append([label, c["n"], pct(c["raw90"]), pct(c["adj90"]), pct(c["raw50"]), pct(c["adj50"])])
    P.append("<div class=card><h3>Out-of-sample coverage</h3>" + table(
        ["Forecast", "Tests", "90% range: raw", "90%: self-corrected", "50% range: raw", "50%: self-corrected"], rows) +
        "<p class=small>Read: a '90% range' should contain the outcome 90% of the time. Raw = the simulation as is; "
        "self-corrected = after the adaptive adjustment.</p></div>")
    rows = []
    for h, label in ((1, "1 day"), (5, "1 week"), (21, "1 month")):
        r = mc["ranges"].get(h)
        if r:
            a9 = r.get("adj90", r["raw90"])
            a5 = r.get("adj50", r["raw50"])
            rows.append([label, f"{pct(a9[0])} to {pct(a9[1])}", f"{pct(a5[0])} to {pct(a5[1])}", pct(r["median"], 2),
                         pct(r["p_loss"], 0), pct(-r["es_97_5"])])
    P.append("<div class=card><h3>Today's portfolio, held as is</h3>" + table(
        ["Horizon", "90% range", "50% range", "Median", "Chance of loss", "Average of worst 2.5%"], rows) + "</div>")
    fan = mc["fan"]
    P.append("<div class=card><h3>The strategy over the next year</h3><div class=chart id=c_fan></div>"
             "<div class=legend><span><i style='background:var(--band1)'></i>90% of paths</span>"
             "<span><i style='background:var(--band2)'></i>middle 50%</span><span><i style='background:var(--s1)'></i>median</span></div>"
             f"<p>Value of 1.00 invested, {cfg['mc_paths']:,} simulated paths, centred on a Sharpe ratio of "
             f"{num(mc['sharpe_used'])} (the shrunk estimate, not the backtest average) at {pct(mc['vol_now'])} volatility. "
             f"Chance of being down after a year: <b>{pct(mc['p_loss_1y'], 0)}</b>. Chance of a drawdown worse than "
             f"10% / 20% / 30% along the way: <b>{pct(mc['p_dd'][0.10], 0)} / {pct(mc['p_dd'][0.20], 0)} / {pct(mc['p_dd'][0.30], 0)}</b>.</p>"
             f"<p><b>Kelly check.</b> Golden-section search on the simulated paths puts growth-optimal leverage at "
             f"<b>{num(mc['kelly_full'], 1)}x</b> today's size (the bell-curve formula would say {num(mc['kelly_gauss'], 1)}x; "
             f"fat tails pull it down). The engine never runs above {num(cfg['kelly_fraction'], 2)} of full Kelly, "
             f"because the edge estimate itself is uncertain.</p></div>")
    charts.append(("line", "c_fan", {"x": [f"day {d}" for d in fan["days"]], "xshort": [f"{d}d" for d in fan["days"]],
                                     "fmt": "num", "h": 300, "labels": False,
                                     "bands": [{"name": "90% range", "lo": fan["q"]["5"], "hi": fan["q"]["95"], "color": "var(--band1)"},
                                               {"name": "50% range", "lo": fan["q"]["25"], "hi": fan["q"]["75"], "color": "var(--band2)"}],
                                     "series": [{"name": "Median", "color": "var(--s1)", "y": fan["q"]["50"]}]}))

    # ------------------------------------------------------------ backtest
    P.append("<h2>4. Walk-forward backtest</h2><p>Every decision used only data available at that close; trades were "
             f"placed {cfg['exec_lag']} day later and paid trading, financing and short-borrow costs. Benchmarks are "
             "shown without costs.</p>")
    hdr = ["", "Return / yr", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Worst month", "Positive months", "Skew"]
    rows = [[esc(k), pct(s.get("cagr")), pct(s.get("vol")), num(s.get("sharpe")), num(s.get("sortino")), pct(s.get("max_dd")),
             pct(s.get("worst_month")), pct(s.get("pct_pos_months"), 0), num(s.get("skew"))] for k, s in b["stats"].items()]
    cv = b["curves"]
    keys = list(cv)
    P.append("<div class=card>" + table(hdr, rows) + "<h3>Growth of 1.00 (log scale)</h3><div class=chart id=c_eq></div>" +
             legend(keys, COLORS) + "<h3>Engine drawdown</h3><div class=chart id=c_dd></div></div>")
    x = [str(d.date()) for d in cv[keys[0]].index]
    xs = [str(d.year) for d in cv[keys[0]].index]
    charts.append(("line", "c_eq", {"x": x, "xshort": xs, "fmt": "num", "log": True,
                                    "series": [{"name": k, "color": COLORS[i], "y": _clean(cv[k])} for i, k in enumerate(keys)]}))
    charts.append(("line", "c_dd", {"x": x, "xshort": xs, "fmt": "pct", "h": 220, "labels": False,
                                    "series": [{"name": "Engine drawdown", "color": "var(--s1)", "y": _clean(b["dd"])}]}))
    c = b["costs"]
    P.append(f"<p>Average gross leverage {c['avg_gross']:.2f}x (net {c['avg_net']:+.2f}x). Turnover {c['turnover']:.1f}x "
             f"equity per year. Costs already deducted: trading {pct(c['trading'], 2)} and financing/borrow "
             f"{pct(c['financing'], 2)} per year.</p>")
    yr = b["yearly"]
    P.append("<div class=grid2><div class=card><h3>Calendar years</h3>" + table(
        ["Year"] + list(yr.columns), [[int(y)] + [pct(v) for v in row] for y, row in zip(yr.index, yr.to_numpy())]) + "</div>")
    P.append("<div class=card><h3>Worst drawdowns</h3>" + table(["Peak", "Trough", "Recovered", "Depth"], [
        [r["peak"].date(), r["trough"].date(), r["recovered"].date() if r["recovered"] is not None else "Not yet", pct(r["depth"])]
        for r in b["worst_dd"]]))
    if info["synthetic"]:
        P.append("<h3>Historical stress episodes</h3><p class=small>Not shown in demo mode: simulated prices have no real "
                 "2008 or 2020. With real data this table shows the engine and the benchmarks through each named crisis.</p>")
    elif b["stress"]:
        P.append("<h3>Historical stress episodes</h3>" + table(
            ["Episode", "From", "To"] + keys,
            [[esc(r["episode"]), r["from"], r["to"]] + [pct(r.get(k)) for k in keys] for r in b["stress"]]))
    P.append("</div></div>")

    # ------------------------------------------------------------ validation
    lo, hi = val["ci"]
    pb = val["pbo"]
    P.append("<h2>5. Is it real? The validation gate</h2>")
    P.append("<div class=tiles>" + "".join([
        tile("Sharpe ratio", num(val["sharpe"]), f"90% interval {num(lo)} to {num(hi)}"),
        tile("P(true Sharpe > 0)", pct(val["psr"], 0), "adjusts for skew and fat tails"),
        tile("Deflated Sharpe", pct(val["dsr"], 0), f"vs best-of-{val['n_trials']} luck hurdle {num(val['hurdle'])} (strict)"),
        tile("Overfitting probability", pct(pb.get("pbo"), 0) if pb.get("n_splits") else "–",
             f"{pb.get('n_splits', 0)} train/test splits" if pb.get("n_splits") else "run without --quick"),
        tile("Track record needed", "∞" if not np.isfinite(val["mtrl"]) else f"{val['mtrl']:.1f} yrs",
             "to be 95% sure Sharpe > 0"),
    ]) + "</div>")
    P.append(f"<p><b>How to read this.</b> The deflated Sharpe asks whether the result beats the best Sharpe you would "
             f"expect from {val['n_trials']} skill-less tries; every configuration ever run on this data is logged in "
             "<code>state/trials.jsonl</code>, so experimenting raises the hurdle automatically. The headline uses the "
             "stricter of two hurdles: one from the spread of the logged results ("
             f"{num(val['dsr_detail']['hurdle_trials'])}, deflated Sharpe {pct(val['dsr_detail']['dsr_trials'], 0)}) and one "
             f"assuming every trial was an independent zero-skill strategy ({num(val['dsr_detail']['hurdle_zero_skill'])}, "
             f"{pct(val['dsr_detail']['dsr_zero_skill'], 0)}). Ideas you tried by hand before running the engine are not in "
             "the log, so even the strict figure is a ceiling. Above about 95% is "
             "strong, 80-95% is suggestive, below that the result is not distinguishable from luck. Overfitting "
             "probability is how often the best-looking setting in one half of history ended up below the median in "
             "the other half; under 20% is healthy, near 50% means choosing settings by backtest is no better than chance. "
             "When all the variants below score within noise of each other, a high value is expected and the lesson is to "
             "keep the defaults rather than switch to whichever variant happened to win.</p>")
    if len(val["ablation"]) > 1:
        rows = [[esc(pretty(r["variant"])), num(r["sharpe"]), ("" if "d_sharpe" not in r else f"{r['d_sharpe']:+.2f}"), pct(r["vol"]),
                 pct(r["cagr"]), pct(r["max_dd"]), f"{r['turnover']:.1f}x"] for r in val["ablation"]]
        P.append("<div class=card><h3>Does each layer earn its place?</h3>" + table(
            ["Variant", "Sharpe", "vs baseline", "Volatility", "Return / yr", "Max drawdown", "Turnover"], rows) +
            "<p class=small>Each row changes one thing. If removing a layer <i>improves</i> the result on your data, that "
            "layer is not earning its place there; a difference under about ±0.1 is noise. The defaults were fixed in "
            "advance from the literature, not chosen from this table. 'Trade at same close' shows how much of the "
            "result depends on execution timing.</p></div>")
    vd = b["vol_diag"]
    P.append(f"<div class=card><h3>Volatility model scorecard</h3><p>Out-of-sample QLIKE loss (lower is better): cascade "
             f"model (HAR) <b>{num(vd['qlike_har'], 3)}</b> vs EWMA baseline <b>{num(vd['qlike_ewma'], 3)}</b>. Current "
             f"blend weight on HAR: {pct(vd['har_weight_now'], 0)}. Range data (open/high/low) in use: "
             f"{'yes' if vd['uses_range_data'] else 'no, closes only'}. Regime model: "
             + ", ".join(f"{n} ≈ {pct(v, 0)} vol" for n, v in zip(names, reg["vols"])) + f" on {esc(reg['bench'])}.</p></div>")

    # ------------------------------------------------------------ VaR
    if b.get("var"):
        P.append(var_section(b["var"], charts, 6))

    # ------------------------------------------------------------ orders
    od = b["orders"]
    P.append(f"<h2>7. Order sheet for a ${b['capital']:,.0f} account</h2>")
    if len(od):
        has_fut = od["futures"].astype(str).str.len().gt(0).any()
        hdr = ["Ticker", "Name", "Weight", "Dollars", "Price", "Shares"] + (["Or futures", "Futures notional", "Rounding gap", "Hedge ratio"] if has_fut else [])
        rows = []
        for r in od.itertuples():
            row = [esc(r.ticker), esc(r.name), pct(r.weight), f"{r.dollars:,.0f}", f"{r.price:,.2f}", f"{r.shares:,d}"]
            if has_fut:
                row += ([f"{int(r.contracts):+d} {esc(r.futures)}", f"{r.fut_notional:,.0f}", pct(r.rounding_gap, 2), esc(r.ratio_source)]
                        if r.futures else ["", "", "", ""])
            rows.append(row)
        P.append("<div class=card>" + table(hdr, rows, left=(1,)) +
                 "<p class=small>Shares are for the ETF. Where a futures column is filled, that contract is an alternative way "
                 "to hold the same exposure (not in addition). Futures counts are rounded to whole contracts; the rounding gap "
                 "is the resulting error as a share of the account. Bond-futures hedge ratios are approximate.</p></div>")
    P.append("""<h2>8. What this is, and what it is not</h2><div class=card>
<p><b>What is predictable.</b> Volatility (roughly half of its variation), correlations (some), and how wide the range of outcomes is. The engine puts its modelling effort there. Direction is barely predictable, so expected returns are shrunk hard and no position depends on a point forecast.</p>
<p><b>Known weak points.</b> (1) Free data only covers funds that exist today. (2) Positions are assumed to hold their target weight between rebalances. (3) Trading costs are flat assumptions per asset class; real costs rise in stress. (4) Value here means 'cheap versus five years ago', a price-only proxy. (5) Regime probabilities lag turning points. (6) Risk parity relies on stocks and bonds not falling together for long; 2022 is the cautionary case. (7) A backtest of a strategy designed after the fact always flatters it; that is what the deflated Sharpe and the trial log are for.</p>
<p><b>Not advice.</b> This is a research tool. It does not know your tax situation, margin terms or other holdings.</p></div>""")

    calls = "\n".join(f"{'lineChart' if kind == 'line' else 'barChart'}({json.dumps(cid)},{json.dumps(data)});"
                      for kind, cid, data in charts)
    body = "".join(P)
    cut = body.index("<h2>")
    main_html = tabify(body[:cut], body[cut:])
    doc = (f"<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Pod Engine report {info['asof'].date()}</title><script>try{{var o=localStorage.getItem('motion');if(o)document.documentElement.dataset.motion=o;}}catch(e){{}}</script><style>{CSS}</style></head><body><button class=motion-btn type=button aria-pressed=true>Motion</button><main>"
           + _title_tags(main_html) + f"<p class=small>Generated in {info['elapsed']:.0f}s.</p></main><script>{JS}\n{calls}</script></body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path


def save_tables(b, outdir):
    """CSV/JSON copies of the key outputs for use elsewhere."""
    res = b["res"]
    b["exposure"]["assets"].to_csv(f"{outdir}/target_weights.csv", index=False)
    b["orders"].to_csv(f"{outdir}/orders.csv", index=False)
    pd.DataFrame({"engine_return": res.returns, "engine_excess": res.excess_returns}).to_csv(f"{outdir}/daily_returns.csv")
    res.pod_frame().to_csv(f"{outdir}/pod_returns.csv")
    pd.DataFrame(b["validation"]["ablation"]).to_csv(f"{outdir}/ablation.csv", index=False)
    summary = {"asof": str(b["info"]["asof"].date()), "source": b["info"]["source"], "stats": b["stats"],
               "validation": {k: v for k, v in b["validation"].items() if k != "ablation"},
               "regime": b["regime"], "costs": b["costs"],
               "mc": {k: v for k, v in b["mc"].items() if k not in ("fan", "cal")},
               "coverage": {k: {kk: vv for kk, vv in c.items() if kk != "pit"} for k, c in b["mc"]["cal"].items()}}
    if b.get("var"):
        v = b["var"]
        bt = v["backtest"]
        summary["var"] = {"today": v["today"], "es_total": v["es_total"],
                          "backtest": {m: {str(lv): s for lv, s in by.items()} for m, by in bt.get("methods", {}).items()},
                          "es_test": bt.get("es_test")}
    with open(f"{outdir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


# ====================================================================== VaR

METHOD_NAMES = {"normal": "Normal (bell curve)", "fhs": "Filtered historical", "mc": "Regime Monte Carlo",
                "fhs_aci": "Filtered historical, self-corrected"}
LIGHT = {"green": "var(--good)", "yellow": "var(--warn)", "red": "var(--crit)"}


def pval(p):
    if p is None or not np.isfinite(p):
        return "–"
    flag = " <b>✗</b>" if p < 0.05 else ""
    return (f"{p:.3f}" if p >= 0.001 else "<0.001") + flag


def var_today_table(t):
    rows = []
    for m in ("normal", "fhs", "mc"):
        v = t[m]
        rows.append([METHOD_NAMES[m], pct(v["var95"], 2), pct(v["var99"], 2), pct(v["es975"], 2),
                     pct(v.get("var99_10d"), 2) if "var99_10d" in v else "–",
                     pct(v.get("es975_10d"), 2) if "es975_10d" in v else "–"])
    return table(["Method", "1-day VaR 95%", "1-day VaR 99%", "1-day ES 97.5%", "10-day VaR 99%", "10-day ES 97.5%"], rows)


def var_backtest_html(bt, what):
    if not bt or bt.get("n", 0) == 0:
        return "<p class=small>Not enough history for a VaR backtest.</p>"
    rows = []
    for m in ("normal", "fhs", "fhs_aci"):
        by = bt["methods"][m]
        n99, light = by["traffic"]
        for lv in (0.95, 0.99):
            s = by[lv]
            rows.append([METHOD_NAMES[m] if lv == 0.95 else "", f"{lv:.0%}", f"{s['breaches']} / {s['expected']:.0f}",
                         pct(s["rate"], 2), pval(s["kupiec_p"]), pval(s["christoffersen_p"]), pval(s["cc_p"]),
                         (f"<span class=dot style='background:{LIGHT[light]}'></span>{light} ({n99})" if lv == 0.99 else "")])
    e = bt["es_test"]
    html_ = (f"<p>{what} Forecasts for each day used only data up to the previous close: "
             f"{bt['n']:,} days, {bt['start'].date()} to {bt['end'].date()}.</p>" +
             table(["Method", "Level", "Breaches / expected", "Breach rate", "Kupiec p", "Christoffersen p",
                    "Combined p", "Basel light, last 250 days"], rows) +
             "<p class=small><b>Reading the tests.</b> A p-value under 0.05 (marked ✗) rejects the model. Kupiec asks "
             "whether breaches happen as often as stated. Christoffersen asks whether they bunch together, which means the "
             "model reacts too slowly. The Basel light counts 99% breaches in the last 250 days: 0-4 green, 5-9 yellow, "
             "10 or more red. The self-corrected line adjusts its quantile after every breach, so it should pass the "
             "frequency test by design; the clustering test is the real check on it.</p>"
             f"<p><b>Expected Shortfall test</b> (Acerbi-Székely Z2, filtered historical, 97.5%): Z2 = {num(e['z2'], 3)}, "
             f"p = {pval(e['p_value'])}. On the {e['n_tail']} days that broke the 97.5% VaR, the forecast ES averaged "
             f"{pct(e['avg_es'], 2)} and the actual loss averaged {pct(e['avg_loss_beyond_var975'], 2)}. Z2 near 0 means the "
             "ES was right on average; clearly negative means tail losses were larger than forecast. The p-value is "
             "computed by simulating from the model's own daily scenarios.</p>")
    return html_


def var_section(v, charts, n):
    t = v["today"]
    P = [f"<h2>{n}. Value at Risk</h2><p>How much today's target portfolio could lose. VaR 99% is the loss exceeded "
         "on 1 day in 100. Expected Shortfall (ES) 97.5% is the average loss on the worst 2.5% of days; it describes how "
         "bad the bad days are, which VaR does not. All figures are losses as a share of account equity, before "
         "trading costs.</p>"]
    P.append("<div class=card><h3>Today</h3>" + var_today_table(t) +
             "<p class=small>Normal uses the engine's covariance matrix and a bell curve. Filtered historical replays the "
             f"last {t['fhs']['n']:,} days of real joint moves rescaled to today's volatility. Regime Monte Carlo adds "
             "regime switching and volatility clustering, which matters most at 10 days, where the normal column just "
             "multiplies by √10.</p></div>")
    order = np.argsort(-np.abs(v["comp_es"]))[:12]
    P.append("<div class=grid2><div class=card><h3>Who contributes to ES (by asset)</h3><div class=chart id=c_ces></div>"
             f"<p class=small>Euler allocation: the average loss each position contributed on the worst 2.5% of "
             f"scenarios. The bars sum to the total ES of {pct(v['es_total'], 2)}; negative bars are positions "
             "that gained on those days.</p></div>")
    charts.append(("bar", "c_ces", {"labels": [v["tickers"][i] for i in order], "values": _clean(v["comp_es"][order]),
                                    "fmt": "pct", "left": 70}))
    rows = [[esc(pretty(nm)), pct(x, 2), pct(x / v["es_total"] if v["es_total"] else float("nan"), 0)]
            for nm, x in zip(v["pod_names"], v["pod_es"])]
    rows.append(["Limits & Band", pct(v["pod_resid"], 2), pct(v["pod_resid"] / v["es_total"] if v["es_total"] else float("nan"), 0)])
    P.append("<div class=card><h3>Who contributes to ES (by pod)</h3>" + table(["Pod", "ES contribution", "Share"], rows) +
             "</div></div>")
    bt = v["backtest"]
    P.append("<div class=card><h3>Backtest of the 1-day VaR</h3>" +
             var_backtest_html(bt, "Each day's VaR was compared with the market P&L of what the engine actually held.") +
             (("<div class=chart id=c_varbt></div>" + legend(["Daily P&L", "99% VaR (as a loss)"], [COLORS[0], COLORS[1]])
               + "<p class=small>Last three years. Breaches are the days the P&L line drops below the VaR line.</p>")
              if bt.get("n", 0) else "") + "</div>")
    if bt.get("n", 0):
        s = bt["series"]
        k = min(756, len(s["pnl"]))
        charts.append(("line", "c_varbt", {"x": [str(d.date()) for d in s["dates"][-k:]],
                                           "xshort": [str(d.date())[:7] for d in s["dates"][-k:]], "fmt": "pct", "h": 280,
                                           "labels": False,
                                           "series": [{"name": "Daily P&L", "color": COLORS[0], "y": _clean(s["pnl"][-k:])},
                                                      {"name": "99% VaR", "color": COLORS[1], "y": _clean(-s["var99"][-k:])}]}))
    return "".join(P)


# ====================================================================== optimiser report

def build_opt_report(r, cfg, path, info, orders=None):
    charts = []
    P = [f"<h1>Portfolio Optimizer</h1><p class=sub>As of {r['asof'].date()} · data: {esc(info['source'])} · profile: "
         f"<b>{esc(pretty(cfg['profile']))}</b> · account equity ${r['equity']:,.0f} · target volatility "
         f"{pct(r['target_vol'])}</p>"]
    if info.get("synthetic"):
        P.append("<div class=banner><b>Demo mode: simulated prices.</b> Every number below exercises the code; none "
                 "of it says anything about real markets.</div>")
    for n_ in r["notes"]:
        P.append(f"<p class=small>Note: {esc(n_)}</p>")

    # ---------------------------------------------------------- comparison
    ports = r["portfolios"]
    P.append("<h2>1. Your portfolio and three alternatives</h2><p>Same assets you hold. Expected returns are "
             "<b>model estimates</b>: they start from the returns that would make your current mix optimal at a Sharpe "
             f"ratio of {num(r['prior_sharpe'])}, then move toward any views you gave. They are deliberately modest; the "
             "risk columns are the more reliable ones.</p>")
    hdr = ["", "Expected excess return / yr", "Volatility", "Sharpe (model)", "1-day VaR 99%", "1-day ES 97.5%",
           "1-month 90% range", "Largest risk share", "Diversification ratio", "Gross", "Net", "Turnover", "Trading cost"]
    rows = []
    for q in ports:
        rows.append([f"<b>{esc(title_case(q['name']))}</b>" + (f"<br><span class=small>{esc(q['note'])}</span>" if q["note"] else ""),
                     pct(q["exp"], 2), pct(q["vol"]), num(q["sharpe"]), pct(q["var99"], 2), pct(q["es975"], 2),
                     f"{pct(q['range90'][0])} to {pct(q['range90'][1])}", pct(q["max_share"], 0), num(q["div_ratio"]),
                     f"{q['gross']:.2f}x", f"{q['net']:+.2f}x", pct(q["turnover"], 0), pct(q["cost"], 2)])
    P.append("<div class=card>" + table(hdr, rows, left=(0,)) +
             "<p class=small><b>Diversification ratio</b> = weighted average of the holdings' volatilities divided by the "
             "portfolio's volatility; 1.0 means no diversification benefit, higher is better. <b>Largest risk share</b> is "
             "the biggest single holding's share of portfolio volatility. <b>Turnover</b> is the total weight traded to get "
             "there; <b>trading cost</b> is the one-off cost of that trade. Black-Litterman can move a long way when a view is "
             "strong; <code>--max-turnover 0.3</code> caps the total weight it may trade.</p></div>")

    # ---------------------------------------------------------- weights
    names = r["tickers"]
    hdr = ["Ticker", "Name"] + [title_case(q["name"]) for q in ports]
    rows = [[esc(t), esc(nm)] + [pct(q["w"][i]) for q in ports] for i, (t, nm) in enumerate(zip(names, r["names"]))]
    rows.append(["", "<b>Cash (rest)</b>", *[pct(1 - q["net"]) for q in ports]])
    P.append("<div class=card><h3>Weights (share of equity)</h3>" + table(hdr, rows, left=(1,)) + "</div>")
    hdr = ["Ticker"] + [title_case(q["name"]) for q in ports]
    rows = [[esc(t)] + [pct(q["risk_share"][i], 0) for q in ports] for i, t in enumerate(names)]
    P.append("<div class=card><h3>Share of risk by holding</h3>" + table(hdr, rows) +
             "<p class=small>Each column sums to 100%. A negative share means the holding currently offsets the others' risk.</p></div>")

    # ---------------------------------------------------------- views
    P.append("<h2>2. Expected returns and your views</h2>")
    rows = [[esc(t), pct(pi, 2), pct(m, 2)] for t, pi, m in zip(names, r["Pi"], r["mu"])]
    P.append("<div class=grid2><div class=card><h3>Per holding</h3>" + table(["Ticker", "Starting point", "After views"], rows) +
             f"<p class=small>Starting point: {esc(r['prior_src'])}. Holdings that hedge the rest of the portfolio come out "
             "with low or negative starting returns; that is what makes holding them optimal, not a forecast that "
             "they will fall.</p></div>")
    if r["views"]:
        rows = [[esc(u["label"]), pct(u["view"], 2), pct(u["confidence"], 0), pct(u["prior"], 2), pct(u["posterior"], 2)]
                for u in r["views"]]
        P.append("<div class=card><h3>Your views</h3>" + table(["View", "You said", "Confidence", "Before", "Blended"], rows, left=(0,)) +
                 f"<p class=small>Blended = what the optimiser used. With confidence 50% a view gets the same weight as the "
                 f"starting point (τ = {r['tau']}). Views can also shift assets they don't name, through correlations.</p>")
    else:
        P.append("<div class=card><h3>Your views</h3><p>No views given, so Black-Litterman returns (close to) your current "
                 "portfolio. Add a views file with <code>--views</code> to express an opinion; see the README for the format.</p>")
    for lbl, why in r["skipped_views"]:
        P.append(f"<p class=small>Skipped view {esc(lbl)}: {esc(why)}</p>")
    P.append("</div></div>")

    # ---------------------------------------------------------- VaR
    if r["var_today"]:
        P.append("<h2>3. Value at Risk of your current portfolio</h2><div class=card>" + var_today_table(r["var_today"]))
        order = np.argsort(-np.abs(r["comp_es"]))
        P.append("<h3>Who contributes to ES</h3><div class=chart id=c_oces></div>"
                 f"<p class=small>The bars sum to the total ES of {pct(r['es_total'], 2)} (filtered historical, 97.5%).</p></div>")
        charts.append(("bar", "c_oces", {"labels": [names[i] for i in order], "values": _clean(r["comp_es"][order]),
                                         "fmt": "pct", "left": 70}))
        P.append("<div class=card><h3>Would this risk model have been right for your portfolio?</h3>" +
                 var_backtest_html(r["backtest"], "Your current weights were held constant over history (rebalanced daily) "
                                                   "and each day's 1-day VaR was compared with the result.") + "</div>")

    # ---------------------------------------------------------- trades
    if orders is not None and len(orders):
        P.append("<h2>4. Trades to reach the Black-Litterman portfolio</h2><div class=card>" + table(
            ["Ticker", "Current weight", "Target weight", "Price", "Current shares", "Target shares", "Trade"],
            [[esc(o.ticker), pct(o.w_now), pct(o.w_new), f"{o.price:,.2f}", f"{o.shares_now:,.0f}", f"{o.shares_new:,.0f}",
              f"{o.trade:+,.0f}"] for o in orders.itertuples()]) +
            "<p class=small>The other alternatives' trades are in <code>optimizer_trades.csv</code>.</p></div>")
    P.append("<h2>Limits</h2><div class=card><p>The optimiser only rearranges what you hold (plus anything added with "
             "<code>--add</code>). It uses a 1-month risk forecast, so it is not built for intraday trading. Options "
             "positions are not supported, because their payoffs are not linear in the underlying price. Expected "
             "returns are estimates with wide error bars: treat the Black-Litterman column as 'what my views imply', not "
             "as a forecast. This is a research tool, not investment advice.</p></div>")
    calls = "\n".join(f"{'lineChart' if kind == 'line' else 'barChart'}({json.dumps(cid)},{json.dumps(data)});"
                      for kind, cid, data in charts)
    body = "".join(P)
    cut = body.index("<h2>")
    doc = (f"<!DOCTYPE html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Portfolio optimiser {r['asof'].date()}</title><script>try{{var o=localStorage.getItem('motion');if(o)document.documentElement.dataset.motion=o;}}catch(e){{}}</script><style>{CSS}</style></head><body><button class=motion-btn type=button aria-pressed=true>Motion</button><main>"
           + _title_tags(tabify(body[:cut], body[cut:])) + f"</main><script>{JS}\n{calls}</script></body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
