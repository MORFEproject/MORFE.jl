#!/usr/bin/env python3
"""Generate the interactive Kármán tutorial charts (lift_vs_Re.html, tke_vs_Re.html).

Reads examples/05_karman_vortex_street/results/comparison/comparison.csv and emits
self-contained HTML assets: SVG in the website colour scheme, drawn by vanilla JS with
hover readout, click-to-toggle orders, wheel zoom, drag pan, double-click reset.
Embedded in tutorial-karman.html via <iframe>. Rerun after regenerating the data:

    python3 website/assets/karman/generate_charts.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CSV = REPO / "examples/05_karman_vortex_street/results/comparison/comparison.csv"

BG, INK, INK2, INK3, HAIR = "#07070b", "#e8e8ee", "#a0a0ab", "#6e6e7e", "#26262f"
ORDER_COLORS = {3: "#4063d8", 5: "#389826", 7: "#cb3c33", 9: "#9558b2"}

W, H = 860, 470
ML, MR, MT, MB = 64, 20, 18, 46  # margins


def build(col: str, ylabel: str, ylim, out: Path):
    d = np.genfromtxt(CSV, delimiter=",", names=True)
    data_js = []
    x_lo = x_hi = y_max = None
    for o in (3, 5, 7, 9):
        s = d[d["order"] == o]
        re_, v = s["Re"], s[col]
        stride = max(1, len(re_) // 160)
        re_, v = re_[::stride], v[::stride]
        x_lo = re_.min() if x_lo is None else min(x_lo, re_.min())
        x_hi = re_.max() if x_hi is None else max(x_hi, re_.max())
        y_max = v.max() if y_max is None else max(y_max, v.max())
        arr = ",".join(f"[{a:.4f},{b:.6g}]" for a, b in zip(re_, v))
        data_js.append(f"{o}:{{c:'{ORDER_COLORS[o]}',p:[{arr}]}}")

    y_lo, y_hi = (0.0, 1.06 * y_max) if ylim is None else ylim

    legend = []
    for i, o in enumerate((3, 5, 7, 9)):
        lx = ML + 16 + i * 108
        legend.append(
            f'<g class="leg" id="leg-{o}" transform="translate({lx} {MT+14})" style="cursor:pointer">'
            f'<rect x="-6" y="-11" width="100" height="20" fill="transparent"/>'
            f'<line x1="0" y1="-3" x2="20" y2="-3" stroke="{ORDER_COLORS[o]}" stroke-width="2.5"/>'
            f'<text x="26" y="0">order {o}</text></g>')

    curves = "".join(
        f'<polyline id="curve-{o}" fill="none" stroke="{ORDER_COLORS[o]}" stroke-width="2"/>'
        for o in (3, 5, 7, 9))

    svg = f'''<svg id="chart" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <defs><clipPath id="clip"><rect x="{ML}" y="{MT}" width="{W-ML-MR}" height="{H-MT-MB}"/></clipPath></defs>
  <g id="grid" stroke="{HAIR}" stroke-width="0.6"></g>
  <g stroke="{INK3}" stroke-width="1" fill="none">
    <line x1="{ML}" y1="{H-MB}" x2="{W-MR}" y2="{H-MB}"/>
    <line x1="{ML}" y1="{MT}" x2="{ML}" y2="{H-MB}"/>
  </g>
  <g id="ticks" font-size="11" fill="{INK3}"></g>
  <text x="{(ML+W-MR)/2}" y="{H-8}" text-anchor="middle" font-size="12" fill="{INK2}">Re</text>
  <text x="16" y="{(MT+H-MB)/2}" text-anchor="middle" font-size="12" fill="{INK2}"
        transform="rotate(-90 16 {(MT+H-MB)/2})">{ylabel}</text>
  <g clip-path="url(#clip)">{curves}
    <g id="probe" visibility="hidden" pointer-events="none">
      <line id="probe-line" y1="{MT}" y2="{H-MB}" stroke="{INK3}" stroke-width="0.7" stroke-dasharray="3 3"/>
      <circle id="probe-dot" r="3.5" fill="{INK}"/>
    </g>
  </g>
  <g id="probe-tip" visibility="hidden" pointer-events="none">
    <rect x="0" y="0" rx="4" width="172" height="22" fill="rgba(20,20,28,0.95)" stroke="{HAIR}"/>
    <text id="probe-txt" x="8" y="15" font-size="11.5" fill="{INK}"></text>
  </g>
  <g font-size="12" fill="{INK2}">{''.join(legend)}</g>
  <text x="{W-MR}" y="{H-8}" text-anchor="end" font-size="10" fill="{INK3}">scroll = zoom · drag = pan · dbl-click = reset</text>
</svg>'''

    html = f'''<!doctype html>
<html><head><meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  html,body {{ margin:0; background:{BG}; font-family:'JetBrains Mono',monospace; }}
  svg {{ display:block; width:100%; height:auto; font-family:'JetBrains Mono',monospace; cursor:crosshair; }}
  svg.panning {{ cursor:grabbing; }}
  .leg.off line {{ opacity:0.25; }} .leg.off text {{ opacity:0.35; }}
</style></head>
<body>
{svg}
<script>
var DATA = {{{','.join(data_js)}}};
var HOME = {{xlo:{x_lo}, xhi:{x_hi}, ylo:{y_lo}, yhi:{y_hi}}};
var V = {{xlo:HOME.xlo, xhi:HOME.xhi, ylo:HOME.ylo, yhi:HOME.yhi}};
var ML={ML}, MR={MR}, MT={MT}, MB={MB}, W={W}, H={H};
var SVGNS = 'http://www.w3.org/2000/svg';
var svgEl = document.getElementById('chart');
var on = {{3:true,5:true,7:true,9:true}};

function px(x) {{ return ML + (x-V.xlo)/(V.xhi-V.xlo)*(W-ML-MR); }}
function py(y) {{ return H-MB - (y-V.ylo)/(V.yhi-V.ylo)*(H-MT-MB); }}
function dx(p) {{ return V.xlo + (p-ML)/(W-ML-MR)*(V.xhi-V.xlo); }}
function dy(p) {{ return V.ylo + (H-MB-p)/(H-MT-MB)*(V.yhi-V.ylo); }}

function niceTicks(lo, hi, n) {{
  var span = hi-lo, step = Math.pow(10, Math.floor(Math.log10(span/n)));
  [1,2,2.5,5,10].some(function(m) {{ if (span/(step*m) <= n) {{ step *= m; return true; }} }});
  var t = [], v = Math.ceil(lo/step)*step;
  for (; v <= hi+1e-12; v += step) t.push(v);
  return t;
}}
function fmt(v) {{ return Math.abs(v) < 1e-12 ? '0' : parseFloat(v.toPrecision(6)).toString(); }}

function redraw() {{
  var grid = document.getElementById('grid'), ticks = document.getElementById('ticks');
  while (grid.firstChild) grid.removeChild(grid.firstChild);
  while (ticks.firstChild) ticks.removeChild(ticks.firstChild);
  niceTicks(V.xlo, V.xhi, 7).forEach(function(tx) {{
    var X = px(tx); if (X < ML-0.5 || X > W-MR+0.5) return;
    var l = document.createElementNS(SVGNS,'line');
    l.setAttribute('x1',X); l.setAttribute('x2',X); l.setAttribute('y1',MT); l.setAttribute('y2',H-MB);
    grid.appendChild(l);
    var t = document.createElementNS(SVGNS,'text');
    t.setAttribute('x',X); t.setAttribute('y',H-MB+18); t.setAttribute('text-anchor','middle');
    t.textContent = fmt(tx); ticks.appendChild(t);
  }});
  niceTicks(V.ylo, V.yhi, 6).forEach(function(ty) {{
    var Y = py(ty); if (Y < MT-0.5 || Y > H-MB+0.5) return;
    var l = document.createElementNS(SVGNS,'line');
    l.setAttribute('y1',Y); l.setAttribute('y2',Y); l.setAttribute('x1',ML); l.setAttribute('x2',W-MR);
    grid.appendChild(l);
    var t = document.createElementNS(SVGNS,'text');
    t.setAttribute('x',ML-8); t.setAttribute('y',Y+3.5); t.setAttribute('text-anchor','end');
    t.textContent = fmt(ty); ticks.appendChild(t);
  }});
  [3,5,7,9].forEach(function(o) {{
    document.getElementById('curve-'+o).setAttribute('points',
      DATA[o].p.map(function(q) {{ return px(q[0]).toFixed(1)+','+py(q[1]).toFixed(1); }}).join(' '));
  }});
}}
redraw();

[3,5,7,9].forEach(function(o) {{
  document.getElementById('leg-'+o).addEventListener('click', function() {{
    on[o] = !on[o];
    document.getElementById('curve-'+o).style.display = on[o] ? '' : 'none';
    this.classList.toggle('off', !on[o]);
  }});
}});

function svgPoint(ev) {{
  var pt = svgEl.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
  return pt.matrixTransform(svgEl.getScreenCTM().inverse());
}}
function inPlot(p) {{ return p.x >= ML && p.x <= W-MR && p.y >= MT && p.y <= H-MB; }}

var probe = document.getElementById('probe'), tip = document.getElementById('probe-tip');
function hideProbe() {{ probe.setAttribute('visibility','hidden'); tip.setAttribute('visibility','hidden'); }}
function updateProbe(p) {{
  var best = null;
  [3,5,7,9].forEach(function(o) {{
    if (!on[o]) return;
    DATA[o].p.forEach(function(q) {{
      var ddx = px(q[0])-p.x, ddy = py(q[1])-p.y, d2 = ddx*ddx+ddy*ddy;
      if (!best || d2 < best.d2) best = {{d2:d2, o:o, re:q[0], v:q[1]}};
    }});
  }});
  if (!best) {{ hideProbe(); return; }}
  probe.setAttribute('visibility','visible'); tip.setAttribute('visibility','visible');
  var lx = px(best.re), ly = py(best.v);
  var pl = document.getElementById('probe-line'); pl.setAttribute('x1',lx); pl.setAttribute('x2',lx);
  var pd = document.getElementById('probe-dot'); pd.setAttribute('cx',lx); pd.setAttribute('cy',ly);
  pd.setAttribute('fill', DATA[best.o].c);
  var tx = Math.min(Math.max(lx+10, ML), W-MR-180), ty = Math.max(Math.min(ly-30, H-MB-26), MT+2);
  tip.setAttribute('transform','translate('+tx+' '+ty+')');
  document.getElementById('probe-txt').textContent =
    'ord '+best.o+' · Re '+best.re.toFixed(2)+' · '+best.v.toPrecision(3);
}}

var pan = null;
svgEl.addEventListener('mousedown', function(ev) {{
  var p = svgPoint(ev);
  if (!inPlot(p)) return;
  pan = {{x:p.x, y:p.y, xlo:V.xlo, xhi:V.xhi, ylo:V.ylo, yhi:V.yhi}};
  svgEl.classList.add('panning'); hideProbe(); ev.preventDefault();
}});
window.addEventListener('mouseup', function() {{ pan = null; svgEl.classList.remove('panning'); }});
svgEl.addEventListener('mousemove', function(ev) {{
  var p = svgPoint(ev);
  if (pan) {{
    var sx = (V.xhi-V.xlo)/(W-ML-MR), sy = (V.yhi-V.ylo)/(H-MT-MB);
    var mx = (pan.x-p.x)*sx, my = (p.y-pan.y)*sy;
    V.xlo = pan.xlo+mx; V.xhi = pan.xhi+mx; V.ylo = pan.ylo+my; V.yhi = pan.yhi+my;
    redraw(); return;
  }}
  if (!inPlot(p)) {{ hideProbe(); return; }}
  updateProbe(p);
}});
svgEl.addEventListener('mouseleave', function() {{ hideProbe(); pan = null; svgEl.classList.remove('panning'); }});

svgEl.addEventListener('wheel', function(ev) {{
  var p = svgPoint(ev);
  if (!inPlot(p)) return;
  ev.preventDefault();
  var f = Math.exp(ev.deltaY * 0.0016);
  var cx = dx(p.x), cy = dy(p.y);
  V.xlo = cx - (cx-V.xlo)*f; V.xhi = cx + (V.xhi-cx)*f;
  V.ylo = cy - (cy-V.ylo)*f; V.yhi = cy + (V.yhi-cy)*f;
  redraw(); updateProbe(p);
}}, {{passive:false}});

svgEl.addEventListener('dblclick', function() {{
  V.xlo = HOME.xlo; V.xhi = HOME.xhi; V.ylo = HOME.ylo; V.yhi = HOME.yhi;
  redraw();
}});
</script>
</body></html>'''
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size//1024} KB)")


build("max_abs_lift", "max |lift|", None, HERE / "lift_vs_Re.html")
build("avg_TKE", "period-averaged TKE", (0.0, 0.1), HERE / "tke_vs_Re.html")
