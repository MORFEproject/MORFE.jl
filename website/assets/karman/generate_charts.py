#!/usr/bin/env python3
"""Generate the interactive Kármán tutorial charts (lift_vs_Re.html, tke_vs_Re.html).

Reads examples/05_karman_vortex_street/results/comparison/comparison.csv and emits
self-contained HTML assets: SVG in the website colour scheme, drawn by vanilla JS.
Hovering always shows the nearest-point readout, but the view NEVER changes unless a
toolbar tool is armed first: click the magnifier then drag a rectangle to zoom, click
the hand then drag to pan, click home to reset. No wheel zoom (so scrolling the page
never moves the plot). Embedded in tutorial-karman.html via <iframe>. Rerun after
regenerating the data:

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

RE_C = 48.984  # Hopf bifurcation (η′_c → Re_c, from solve_rom.jl)

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
    x_lo = min(x_lo, RE_C - 0.6)   # room to show the solid (stable) steady branch left of Re_c

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
  <g clip-path="url(#clip)">
    <line id="steady-solid" stroke="{INK2}" stroke-width="1.6"/>
    <line id="steady-dashed" stroke="{INK2}" stroke-width="1.6" stroke-dasharray="6 4"/>
    {curves}
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
  <g id="hopf" visibility="hidden" pointer-events="none">
    <circle id="hopf-dot" r="4" fill="{INK}" stroke="{BG}" stroke-width="1.2"/>
    <text id="hopf-label" font-size="11.5" fill="{INK2}"></text>
  </g>
  <rect id="zoombox" visibility="hidden" fill="rgba(149,88,178,0.10)" stroke="{INK3}"
        stroke-width="0.8" stroke-dasharray="4 3" pointer-events="none"/>
  <g id="toolbar" font-family="inherit">
    <g class="tool" id="tool-zoom" transform="translate({W-MR-92} {MT-4})"><title>Zoom: drag a rectangle</title>
      <rect class="btn" x="0" y="0" width="26" height="26" rx="4"/>
      <g class="icn" transform="translate(4 4)">
        <circle cx="7.5" cy="7.5" r="5" fill="none" stroke-width="1.6"/>
        <line x1="11.2" y1="11.2" x2="16" y2="16" stroke-width="1.8"/>
        <line x1="5.2" y1="7.5" x2="9.8" y2="7.5" stroke-width="1.3"/>
        <line x1="7.5" y1="5.2" x2="7.5" y2="9.8" stroke-width="1.3"/>
      </g>
    </g>
    <g class="tool" id="tool-pan" transform="translate({W-MR-60} {MT-4})"><title>Pan: drag to move</title>
      <rect class="btn" x="0" y="0" width="26" height="26" rx="4"/>
      <g class="icn" transform="translate(4 3.5)">
        <path fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
          d="M7.2 10.5 V4.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M9.7 9 V3.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M12.2 9.3 V4.8 a1.25 1.25 0 0 1 2.5 0 V11.5 c0 3.4 -2 5.4 -4.9 5.4 c-2.3 0 -3.4 -0.9 -4.5 -2.6 L3.4 11.1 a1.3 1.3 0 0 1 2.2 -1.3 l1.6 2.2"/>
      </g>
    </g>
    <g class="tool" id="tool-home" transform="translate({W-MR-28} {MT-4})"><title>Reset view</title>
      <rect class="btn" x="0" y="0" width="26" height="26" rx="4"/>
      <g class="icn" transform="translate(4 4)">
        <path fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
          d="M2.2 8.2 L9 2.6 L15.8 8.2 M4.2 7.5 V15 H13.8 V7.5"/>
      </g>
    </g>
  </g>
</svg>'''

    html = f'''<!doctype html>
<html><head><meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  html,body {{ margin:0; background:{BG}; font-family:'JetBrains Mono',monospace; }}
  svg {{ display:block; width:100%; height:auto; font-family:'JetBrains Mono',monospace; }}
  svg.mode-zoom {{ cursor:crosshair; }}
  svg.mode-pan {{ cursor:grab; }} svg.mode-pan.panning {{ cursor:grabbing; }}
  .leg.off line {{ opacity:0.25; }} .leg.off text {{ opacity:0.35; }}
  .tool {{ cursor:pointer; }}
  .tool .btn {{ fill:rgba(255,255,255,0.04); stroke:{HAIR}; }}
  .tool .icn {{ stroke:{INK3}; }}
  .tool:hover .btn {{ fill:rgba(255,255,255,0.09); }}
  .tool.active .btn {{ stroke:#9558b2; fill:rgba(149,88,178,0.12); }}
  .tool.active .icn {{ stroke:#9558b2; }}
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
var RE_C = {RE_C};   // Hopf bifurcation; steady branch is y=0 (no fluctuation)

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
  // steady (base-flow) branch at y=0: solid before the Hopf point, dashed after.
  var yb = py(0), xc = px(RE_C), xcl = Math.max(ML, Math.min(W-MR, xc));
  var ss = document.getElementById('steady-solid');
  ss.setAttribute('x1', ML); ss.setAttribute('x2', xcl); ss.setAttribute('y1', yb); ss.setAttribute('y2', yb);
  var sd = document.getElementById('steady-dashed');
  sd.setAttribute('x1', xcl); sd.setAttribute('x2', W-MR); sd.setAttribute('y1', yb); sd.setAttribute('y2', yb);
  var hp = document.getElementById('hopf');
  if (xc >= ML && xc <= W-MR && yb >= MT-2 && yb <= H-MB+6) {{
    hp.setAttribute('visibility','visible');
    document.getElementById('hopf-dot').setAttribute('cx', xc);
    document.getElementById('hopf-dot').setAttribute('cy', yb);
    var lbl = document.getElementById('hopf-label');
    lbl.setAttribute('x', Math.min(xc+9, W-MR-150));
    lbl.setAttribute('y', Math.max(yb-9, MT+12));
    lbl.textContent = 'Hopf · Re_c ≈ 48.98';
  }} else {{
    hp.setAttribute('visibility','hidden');
  }}
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

var mode = null, pan = null, zoomStart = null;
var zoombox = document.getElementById('zoombox');
function setMode(m) {{
  mode = (mode === m) ? null : m;   // click again to disarm — no tool means no view changes
  svgEl.classList.toggle('mode-zoom', mode === 'zoom');
  svgEl.classList.toggle('mode-pan', mode === 'pan');
  document.getElementById('tool-zoom').classList.toggle('active', mode === 'zoom');
  document.getElementById('tool-pan').classList.toggle('active', mode === 'pan');
}}
document.getElementById('tool-zoom').addEventListener('click', function() {{ setMode('zoom'); }});
document.getElementById('tool-pan').addEventListener('click', function() {{ setMode('pan'); }});
document.getElementById('tool-home').addEventListener('click', function() {{
  V.xlo = HOME.xlo; V.xhi = HOME.xhi; V.ylo = HOME.ylo; V.yhi = HOME.yhi; redraw();
}});

svgEl.addEventListener('mousedown', function(ev) {{
  if (mode === null) return;        // view changes only with an armed tool
  var p = svgPoint(ev);
  if (!inPlot(p)) return;
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
      var nxlo = dx(x1), nxhi = dx(x1+w), nyhi = dy(y1), nylo = dy(y1+h);
      V.xlo = nxlo; V.xhi = nxhi; V.ylo = nylo; V.yhi = nyhi;
      redraw();
    }}
    zoombox.setAttribute('data-live','0');
    zoomStart = null;
  }}
  pan = null; svgEl.classList.remove('panning');
}});
svgEl.addEventListener('mousemove', function(ev) {{
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
  if (!inPlot(p)) {{ hideProbe(); return; }}
  updateProbe(p);
}});
svgEl.addEventListener('mouseleave', function() {{ hideProbe(); }});
</script>
</body></html>'''
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size//1024} KB)")


build("max_abs_lift", "max |lift|", (-0.001, 0.015), HERE / "lift_vs_Re.html")
build("avg_TKE", "period-averaged TKE", (-0.001, 0.02), HERE / "tke_vs_Re.html")
