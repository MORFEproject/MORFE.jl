#!/usr/bin/env python3
"""Generate branch_vs_Re.html, the Kármán limit-cycle branch in two switchable panels.

Reads `branch.v1.csv`, the committed copy of the order-9 `results/data/branch.csv` that
MORFEExamples' `karman_vortex_street/karman_vortex_street.ipynb` writes, and emits one self-contained asset in
the shell every other tutorial figure uses: a bar of panel buttons, a framed stage, a legend
of clickable order labels, a note, and a readout tinted by the curve it belongs to. The two
panels, peak lift and Strouhal number, share the same x-axis and the same four curves, so
they are one figure rather than two, exactly as the StructuralSVK backbone is.

Hovering always shows the nearest-point readout, but the view NEVER changes unless a toolbar
tool is armed first: click the magnifier then drag a rectangle to zoom, click the hand then
drag to pan, click home to reset. No wheel zoom, so scrolling the page never moves the plot.

    python3 website/tutorials/assets/karman/generate_charts.py

The notebook writes the rows in AMPLITUDE order, and they are plotted in file order. Orders
5 and 9 fold, so sorting by Re here would draw a zig-zag across the fold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "branch.v1.csv"
OUT = HERE / "branch_vs_Re.html"

BG, INK, INK2, INK3, HAIR = "#07070b", "#e8e8ee", "#a0a0ab", "#6e6e7e", "#26262f"
ORDERS = (3, 5, 7, 9)
ORDER_COLORS = {3: "#4063d8", 5: "#389826", 7: "#cb3c33", 9: "#9558b2"}

RE_C = 48.9844  # Hopf bifurcation, the ρ = 0 end of the order-9 branch
X_VIEW_MAX = 56.0  # default x-axis upper bound; the sweep runs to Re 70, pan/zoom reveals it

W, H = 900, 470
ML, MR, MT, MB = 78, 24, 22, 58  # margins

# One entry per panel. `baseline` is the y of the steady base flow, drawn solid then dashed
# through Re_c; None for an observable that has no steady value, such as a frequency.
# `trim_below` cuts an order once it has folded and run back past that Re: the order-9
# branch turns around near Re 55 and then dives, and that tail is the truncated series
# diverging rather than anything the model is saying.
PANELS = [
    dict(key="lift", name="max |lift|", col="max_abs_lift", dec=5,
         ylim=(-0.001, 0.015), baseline=0.0, trim_below={9: 52.0},
         note="Peak lift over one cycle. The grey line is the steady base flow, dashed "
              "where it has gone unstable; the branches grow out of it at Re_c."),
    dict(key="strouhal", name="Strouhal number", col="St", dec=4,
         ylim=(0.2675, 0.2775), baseline=None, trim_below={9: 52.0},
         note="St = ΩD/2πU. It leaves the Hopf point at the linear value 0.268 and rises "
              "as the cycle grows, a frequency shift no linear spectrum can give."),
]


def trim_fold(re_, v, below):
    """Cut a curve at the first point that is both descending and past `below`."""
    dropping = (np.arange(len(re_)) > 0) & (np.diff(re_, prepend=re_[0]) < 0)
    past = np.flatnonzero(dropping & (re_ < below))
    return (re_[:past[0]], v[:past[0]]) if len(past) else (re_, v)


def series(d, panel):
    """`{order: {c, p}}` for one panel, plus the x-range its curves span."""
    out, x_lo, x_hi = [], None, None
    for o in ORDERS:
        s = d[d["order"] == o]
        re_, v = s["Re"], s[panel["col"]]
        if o in panel["trim_below"]:
            re_, v = trim_fold(re_, v, panel["trim_below"][o])
        stride = max(1, len(re_) // 160)
        re_, v = re_[::stride], v[::stride]
        x_lo = re_.min() if x_lo is None else min(x_lo, re_.min())
        x_hi = re_.max() if x_hi is None else max(x_hi, re_.max())
        pts = ",".join(f"[{a:.4f},{b:.6g}]" for a, b in zip(re_, v))
        out.append(f"{o}:{{c:'{ORDER_COLORS[o]}',p:[{pts}]}}")
    return "{" + ",".join(out) + "}", x_lo, x_hi


def main():
    d = np.genfromtxt(CSV, delimiter=",", names=True)

    data_js, x_lo, x_hi = [], None, None
    for panel in PANELS:
        js, lo, hi = series(d, panel)
        data_js.append(f"{panel['key']}:{js}")
        x_lo = lo if x_lo is None else min(x_lo, lo)
        x_hi = hi if x_hi is None else max(x_hi, hi)
    # Both panels share the x home, so switching panel never moves the axis.
    x_lo = min(x_lo, RE_C - 0.6)   # room for the stable steady branch left of Re_c
    x_hi = min(x_hi, X_VIEW_MAX)   # data extends further; only the default view stops here

    panels_js = ",".join(
        "{}:{{name:{!r},ylabel:{!r},ylo:{},yhi:{},baseline:{},dec:{},note:{!r}}}".format(
            p["key"], p["name"], p["name"], p["ylim"][0], p["ylim"][1],
            "null" if p["baseline"] is None else p["baseline"], p["dec"], p["note"]
        ).replace("'", '"')
        for p in PANELS)
    keys_js = ",".join(f'"{p["key"]}"' for p in PANELS)
    buttons = "".join(f'<button id="panel-{p["key"]}">{p["name"]}</button>' for p in PANELS)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Limit-cycle branch against Re</title>
<style>
:root {{ --bg:#0a0a0f; --stage:{BG}; --ink:{INK}; --ink2:{INK2}; --ink3:{INK3};
  --hair:{HAIR}; --purple:#9558b2; --gray:#8a8d99; }}
* {{ box-sizing:border-box; }} html,body {{ margin:0; height:100%; }}
body {{ background:var(--bg); color:var(--ink); overflow:hidden;
  font:13px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif; }}
#wrap {{ display:flex; flex-direction:column; height:100%; padding:10px 12px; gap:8px; }}
#bar {{ display:flex; gap:6px; align-items:center; min-height:28px; }}
button {{ font:inherit; font-size:12px; padding:4px 10px; border-radius:5px; cursor:pointer;
  background:transparent; color:var(--ink2); border:1px solid var(--hair); }}
button:hover {{ border-color:var(--purple); color:var(--ink); }}
button.on {{ background:rgba(149,88,178,.16); border-color:var(--purple); color:var(--ink); }}
#meta {{ margin-left:auto; color:var(--ink3); font-size:11px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.04em; }}
#stage {{ position:relative; flex:1; min-height:0; border:1px solid var(--hair);
  border-radius:6px; background:var(--stage); overflow:hidden; }}
svg {{ display:block; width:100%; height:100%; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.axis {{ stroke:var(--ink3); stroke-width:1; }}
.grid {{ stroke:var(--hair); stroke-width:.8; }}
.tick, .label {{ fill:var(--ink3); font:11px ui-monospace,SFMono-Regular,Menlo,monospace; }}
.label {{ fill:var(--ink2); font-size:12px; }}
.curve {{ fill:none; stroke-width:2; }}
.tool {{ cursor:pointer; }}
.tool .btn {{ fill:rgba(255,255,255,.04); stroke:var(--hair); }}
.tool .icn {{ stroke:var(--ink3); }}
.tool:hover .btn {{ fill:rgba(255,255,255,.09); }}
.tool.active .btn {{ stroke:var(--purple); fill:rgba(149,88,178,.12); }}
.tool.active .icn {{ stroke:var(--purple); }}
svg.mode-zoom #hit {{ cursor:crosshair; }}
svg.mode-pan #hit {{ cursor:grab; }} svg.mode-pan.panning #hit {{ cursor:grabbing; }}
svg.mode-zoom, svg.mode-pan {{ touch-action:none; }}
#legend {{ display:flex; gap:14px; align-items:center; flex-wrap:wrap; }}
#legend button {{ border:0; padding:0; color:var(--ink3); font-size:11px; }}
#legend button.off {{ opacity:.3; }}
#legend i {{ display:inline-block; width:14px; height:0; vertical-align:3px;
  border-top:2.5px solid currentColor; margin-right:6px; }}
#note {{ color:var(--ink3); font-size:11.5px; min-height:1.3em; }}
.tip {{ position:absolute; pointer-events:none; display:none; background:#14141c;
  border:1px solid var(--hair); border-radius:5px; padding:6px 9px; font-size:11.5px;
  color:var(--ink); white-space:nowrap;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; z-index:5; }}
@media(max-width:560px) {{
  #wrap {{ padding:7px 9px; gap:6px; }} #meta, #note {{ display:none; }}
}}
</style></head>
<body><div id="wrap">
<div id="bar">{buttons}<span id="meta">one order-9 ROM · nested truncations</span></div>
<div id="stage">
<svg id="chart" viewBox="0 0 {W} {H}" role="img"
  aria-label="Peak lift and Strouhal number of the Kármán limit cycle against Reynolds number, at truncation orders 3, 5, 7 and 9">
  <defs><clipPath id="plot-clip"><rect x="{ML}" y="{MT}" width="{W-ML-MR}" height="{H-MT-MB}"/></clipPath></defs>
  <g id="grid"></g>
  <g id="axes"></g>
  <g clip-path="url(#plot-clip)">
    <line id="steady-solid" stroke="var(--gray)" stroke-width="1.6"/>
    <line id="steady-dashed" stroke="var(--gray)" stroke-width="1.6" stroke-dasharray="6 4"/>
    <g id="curves"></g>
    <g id="probe" visibility="hidden">
      <line id="cross-x" stroke="{INK3}" stroke-width=".7" stroke-dasharray="3 3"/>
      <line id="cross-y" stroke="{INK3}" stroke-width=".7" stroke-dasharray="3 3"/>
      <circle id="dot" r="4" stroke="{BG}" stroke-width="1.2"/>
    </g>
    <rect id="zoombox" visibility="hidden" fill="rgba(149,88,178,.10)" stroke="{INK3}"
      stroke-width=".8" stroke-dasharray="4 3" pointer-events="none"/>
  </g>
  <g id="hopf" visibility="hidden" pointer-events="none">
    <circle id="hopf-dot" r="4" fill="{INK}" stroke="{BG}" stroke-width="1.2"/>
    <text id="hopf-label" class="tick" fill="{INK2}"></text>
  </g>
  <text class="label" x="{(ML+W-MR)/2}" y="{H-16}" text-anchor="middle">Reynolds number</text>
  <text id="ylabel" class="label" transform="translate(18 {(MT+H-MB)/2}) rotate(-90)"
    text-anchor="middle"></text>
  <rect id="hit" x="{ML}" y="{MT}" width="{W-ML-MR}" height="{H-MT-MB}" fill="transparent"/>
  <g id="toolbar">
    <g class="tool" id="tool-zoom" transform="translate({W-MR-92} {MT-8})"><title>Zoom: drag a rectangle</title>
      <rect class="btn" width="26" height="26" rx="4"/>
      <g class="icn" transform="translate(4 4)">
        <circle cx="7.5" cy="7.5" r="5" fill="none" stroke-width="1.6"/>
        <line x1="11.2" y1="11.2" x2="16" y2="16" stroke-width="1.8"/>
        <line x1="5.2" y1="7.5" x2="9.8" y2="7.5" stroke-width="1.3"/>
        <line x1="7.5" y1="5.2" x2="7.5" y2="9.8" stroke-width="1.3"/>
      </g>
    </g>
    <g class="tool" id="tool-pan" transform="translate({W-MR-60} {MT-8})"><title>Pan: drag to move</title>
      <rect class="btn" width="26" height="26" rx="4"/>
      <g class="icn" transform="translate(4 3.5)">
        <path fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
          d="M7.2 10.5 V4.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M9.7 9 V3.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M12.2 9.3 V4.8 a1.25 1.25 0 0 1 2.5 0 V11.5 c0 3.4 -2 5.4 -4.9 5.4 c-2.3 0 -3.4 -.9 -4.5 -2.6 L3.4 11.1 a1.3 1.3 0 0 1 2.2 -1.3 l1.6 2.2"/>
      </g>
    </g>
    <g class="tool" id="tool-home" transform="translate({W-MR-28} {MT-8})"><title>Reset view</title>
      <rect class="btn" width="26" height="26" rx="4"/>
      <g class="icn" transform="translate(4 4)">
        <path fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
          d="M2.2 8.2 L9 2.6 L15.8 8.2 M4.2 7.5 V15 H13.8 V7.5"/>
      </g>
    </g>
  </g>
</svg>
<div id="tip" class="tip"></div>
</div>
<div id="legend"></div>
<div id="note"></div>
</div>
<script>
var DATA = {{{','.join(data_js)}}};
var PANELS = {{{panels_js}}};
var KEYS = [{keys_js}], ORDERS = [{','.join(map(str, ORDERS))}];
var HOME_X = {{xlo:{x_lo}, xhi:{x_hi}}};
var RE_C = {RE_C};
var NS = 'http://www.w3.org/2000/svg';
var ML={ML}, MR={MR}, MT={MT}, MB={MB}, W={W}, H={H};

var panel = KEYS[0];
var visible = {{}};
ORDERS.forEach(function(o) {{ visible[o] = true; }});
var V = {{}};
var svgEl = document.getElementById('chart');
var stage = document.getElementById('stage');
var hit = document.getElementById('hit');
var tip = document.getElementById('tip');
var zoombox = document.getElementById('zoombox');

function el(name, attrs) {{
  var n = document.createElementNS(NS, name);
  for (var k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}}
// Blend a curve colour toward the tooltip background, so a readout is tinted by the curve
// it belongs to: `amount` near 0 is almost the background, near 1 the colour itself.
function mix(hex, amount) {{
  var bg = [20, 20, 28];
  var rgb = [1, 3, 5].map(function(i) {{ return parseInt(hex.slice(i, i + 2), 16); }});
  return 'rgb(' + rgb.map(function(v, i) {{
    return Math.round(bg[i] + (v - bg[i]) * amount);
  }}).join(',') + ')';
}}

function px(x) {{ return ML + (x-V.xlo)/(V.xhi-V.xlo)*(W-ML-MR); }}
function py(y) {{ return H-MB - (y-V.ylo)/(V.yhi-V.ylo)*(H-MT-MB); }}
function dx(p) {{ return V.xlo + (p-ML)/(W-ML-MR)*(V.xhi-V.xlo); }}
function dy(p) {{ return V.ylo + (H-MB-p)/(H-MT-MB)*(V.yhi-V.ylo); }}

function niceTicks(lo, hi, n) {{
  var span = hi-lo; if (!(span > 0)) return [];
  var step = Math.pow(10, Math.floor(Math.log10(span/n)));
  [1,2,2.5,5,10].some(function(m) {{ if (span/(step*m) <= n) {{ step *= m; return true; }} }});
  var t = [], v = Math.ceil(lo/step)*step;
  for (; v <= hi+1e-12; v += step) t.push(Math.abs(v) < 1e-12 ? 0 : v);
  return t;
}}
function fmt(v) {{ return Math.abs(v) < 1e-12 ? '0' : parseFloat(v.toPrecision(6)).toString(); }}

function buildLegend() {{
  var legend = document.getElementById('legend');
  legend.replaceChildren();
  ORDERS.forEach(function(o) {{
    var b = document.createElement('button');
    b.className = visible[o] ? '' : 'off';
    b.innerHTML = '<i style="color:' + DATA[panel][o].c + '"></i>order ' + o;
    b.onclick = function() {{ visible[o] = !visible[o]; redraw(); }};
    legend.append(b);
  }});
}}

function hideProbe() {{
  document.getElementById('probe').setAttribute('visibility','hidden');
  tip.style.display = 'none';
}}

function redraw() {{
  var P = PANELS[panel];
  var grid = document.getElementById('grid'), axes = document.getElementById('axes');
  var curves = document.getElementById('curves');
  grid.replaceChildren(); axes.replaceChildren(); curves.replaceChildren();

  niceTicks(V.xlo, V.xhi, 7).forEach(function(tx) {{
    var X = px(tx); if (X < ML-0.5 || X > W-MR+0.5) return;
    grid.append(el('line', {{'class':'grid', x1:X, x2:X, y1:MT, y2:H-MB}}));
    var t = el('text', {{'class':'tick', x:X, y:H-MB+18, 'text-anchor':'middle'}});
    t.textContent = fmt(tx); axes.append(t);
  }});
  niceTicks(V.ylo, V.yhi, 6).forEach(function(ty) {{
    var Y = py(ty); if (Y < MT-0.5 || Y > H-MB+0.5) return;
    grid.append(el('line', {{'class':'grid', x1:ML, x2:W-MR, y1:Y, y2:Y}}));
    var t = el('text', {{'class':'tick', x:ML-9, y:Y+4, 'text-anchor':'end'}});
    t.textContent = fmt(ty); axes.append(t);
  }});
  axes.append(el('line', {{'class':'axis', x1:ML, x2:W-MR, y1:H-MB, y2:H-MB}}),
              el('line', {{'class':'axis', x1:ML, x2:ML, y1:MT, y2:H-MB}}));

  ORDERS.forEach(function(o) {{
    var s = DATA[panel][o];
    var path = el('polyline', {{'class':'curve', stroke:s.c,
      points: s.p.map(function(q) {{ return px(q[0]).toFixed(1)+','+py(q[1]).toFixed(1); }}).join(' ')}});
    if (!visible[o]) path.style.display = 'none';
    curves.append(path);
  }});

  document.getElementById('ylabel').textContent = P.ylabel;
  document.getElementById('note').textContent = P.note;
  KEYS.forEach(function(k) {{
    document.getElementById('panel-'+k).classList.toggle('on', k === panel);
  }});

  // steady (base-flow) branch: solid before the Hopf point, dashed after.
  var ss = document.getElementById('steady-solid');
  var sd = document.getElementById('steady-dashed');
  var hp = document.getElementById('hopf');
  if (P.baseline === null) {{
    ss.setAttribute('visibility','hidden'); sd.setAttribute('visibility','hidden');
    hp.setAttribute('visibility','hidden');
  }} else {{
    ss.setAttribute('visibility','visible'); sd.setAttribute('visibility','visible');
    var yb = py(P.baseline), xc = px(RE_C), xcl = Math.max(ML, Math.min(W-MR, xc));
    ss.setAttribute('x1', ML); ss.setAttribute('x2', xcl);
    ss.setAttribute('y1', yb); ss.setAttribute('y2', yb);
    sd.setAttribute('x1', xcl); sd.setAttribute('x2', W-MR);
    sd.setAttribute('y1', yb); sd.setAttribute('y2', yb);
    if (xc >= ML && xc <= W-MR && yb >= MT-2 && yb <= H-MB+6) {{
      hp.setAttribute('visibility','visible');
      document.getElementById('hopf-dot').setAttribute('cx', xc);
      document.getElementById('hopf-dot').setAttribute('cy', yb);
      var lbl = document.getElementById('hopf-label');
      lbl.setAttribute('x', Math.min(xc+9, W-MR-150));
      lbl.setAttribute('y', Math.max(yb-9, MT+12));
      lbl.textContent = 'Hopf · Re_c ≈ ' + RE_C.toFixed(2);
    }} else {{
      hp.setAttribute('visibility','hidden');
    }}
  }}
  hideProbe();
  buildLegend();
}}

function resetView() {{
  var P = PANELS[panel];
  V.xlo = HOME_X.xlo; V.xhi = HOME_X.xhi; V.ylo = P.ylo; V.yhi = P.yhi;
  redraw();
}}
function setPanel(k) {{ panel = k; resetView(); }}
KEYS.forEach(function(k) {{
  document.getElementById('panel-'+k).onclick = function() {{ setPanel(k); }};
}});

function svgPoint(ev) {{
  var pt = svgEl.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
  return pt.matrixTransform(svgEl.getScreenCTM().inverse());
}}

function updateProbe(p, ev) {{
  var best = null, dist = 400;
  ORDERS.forEach(function(o) {{
    if (!visible[o]) return;
    DATA[panel][o].p.forEach(function(q) {{
      var X = px(q[0]), Y = py(q[1]);
      if (X < ML || X > W-MR || Y < MT || Y > H-MB) return;
      var d = (X-p.x)*(X-p.x) + (Y-p.y)*(Y-p.y);
      if (d < dist) {{ dist = d; best = {{o:o, re:q[0], v:q[1], X:X, Y:Y}}; }}
    }});
  }});
  if (!best) {{ hideProbe(); return; }}
  document.getElementById('probe').setAttribute('visibility','visible');
  var color = DATA[panel][best.o].c;
  var vx = document.getElementById('cross-x'), vy = document.getElementById('cross-y');
  vx.setAttribute('x1', best.X); vx.setAttribute('x2', best.X);
  vx.setAttribute('y1', MT); vx.setAttribute('y2', H-MB);
  vy.setAttribute('x1', ML); vy.setAttribute('x2', W-MR);
  vy.setAttribute('y1', best.Y); vy.setAttribute('y2', best.Y);
  var dot = document.getElementById('dot');
  dot.setAttribute('cx', best.X); dot.setAttribute('cy', best.Y);
  dot.setAttribute('fill', color);

  tip.innerHTML = 'order ' + best.o + '<br>Re ' + best.re.toFixed(2) +
    '<br>' + PANELS[panel].ylabel + ' ' + best.v.toFixed(PANELS[panel].dec);
  tip.style.display = 'block';
  tip.style.background = mix(color, .18);
  tip.style.borderColor = mix(color, .62);
  var rect = stage.getBoundingClientRect();
  var left = ev.clientX - rect.left, top = ev.clientY - rect.top;
  tip.style.left = Math.min(Math.max(4, left + 12), rect.width - tip.offsetWidth - 4) + 'px';
  tip.style.top = Math.min(Math.max(4, top - tip.offsetHeight - 10),
                           rect.height - tip.offsetHeight - 4) + 'px';
}}

var mode = null, pan = null, zoomStart = null;
function setMode(m) {{
  mode = (mode === m) ? null : m;   // click again to disarm — no tool means no view changes
  svgEl.classList.toggle('mode-zoom', mode === 'zoom');
  svgEl.classList.toggle('mode-pan', mode === 'pan');
  document.getElementById('tool-zoom').classList.toggle('active', mode === 'zoom');
  document.getElementById('tool-pan').classList.toggle('active', mode === 'pan');
}}
document.getElementById('tool-zoom').addEventListener('click', function() {{ setMode('zoom'); }});
document.getElementById('tool-pan').addEventListener('click', function() {{ setMode('pan'); }});
document.getElementById('tool-home').addEventListener('click', resetView);

hit.addEventListener('mousedown', function(ev) {{
  if (mode === null) return;        // view changes only with an armed tool
  var p = svgPoint(ev);
  hideProbe(); ev.preventDefault();
  if (mode === 'pan') {{
    pan = {{x:p.x, y:p.y, xlo:V.xlo, xhi:V.xhi, ylo:V.ylo, yhi:V.yhi}};
    svgEl.classList.add('panning');
  }} else {{
    zoomStart = p;
  }}
}});
window.addEventListener('mouseup', function() {{
  if (zoomStart) {{
    var x1 = parseFloat(zoombox.getAttribute('x')), w = parseFloat(zoombox.getAttribute('width'));
    var y1 = parseFloat(zoombox.getAttribute('y')), h = parseFloat(zoombox.getAttribute('height'));
    zoombox.setAttribute('visibility','hidden');
    if (zoombox.getAttribute('data-live') === '1' && w > 8 && h > 8) {{
      V.xlo = dx(x1); V.xhi = dx(x1+w); V.yhi = dy(y1); V.ylo = dy(y1+h);
      redraw();
    }}
    zoombox.setAttribute('data-live','0');
    zoomStart = null;
  }}
  pan = null; svgEl.classList.remove('panning');
}});
hit.addEventListener('mousemove', function(ev) {{
  var p = svgPoint(ev);
  if (pan) {{
    var sx = (V.xhi-V.xlo)/(W-ML-MR), sy = (V.yhi-V.ylo)/(H-MT-MB);
    var mx = (pan.x-p.x)*sx, my = (p.y-pan.y)*sy;
    V.xlo = pan.xlo+mx; V.xhi = pan.xhi+mx; V.ylo = pan.ylo+my; V.yhi = pan.yhi+my;
    redraw(); return;
  }}
  if (zoomStart) {{
    var cx = Math.max(ML, Math.min(W-MR, p.x)), cy = Math.max(MT, Math.min(H-MB, p.y));
    zoombox.setAttribute('x', Math.min(zoomStart.x, cx));
    zoombox.setAttribute('y', Math.min(zoomStart.y, cy));
    zoombox.setAttribute('width', Math.abs(cx - zoomStart.x));
    zoombox.setAttribute('height', Math.abs(cy - zoomStart.y));
    zoombox.setAttribute('visibility','visible');
    zoombox.setAttribute('data-live','1');
    return;
  }}
  updateProbe(p, ev);
}});
hit.addEventListener('mouseleave', hideProbe);

resetView();
</script>
</body></html>
"""
    OUT.write_text(html)
    for stale in ("lift_vs_Re.html", "strouhal_vs_Re.html"):
        (HERE / stale).unlink(missing_ok=True)
    print(f"wrote {OUT.name}  ({OUT.stat().st_size // 1024} KB, "
          f"{len(PANELS)} panels × {len(ORDERS)} orders, Re {x_lo:.2f}..{x_hi:.2f})")


main()
