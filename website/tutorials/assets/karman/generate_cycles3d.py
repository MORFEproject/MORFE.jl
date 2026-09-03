#!/usr/bin/env python3
"""Generate cycles_3d.html: the bifurcation diagram drawn as the cycles it is made of.

Every point of the branch is a whole periodic orbit, the circle z₁ = ρ·exp(iΩt) in the
reduced coordinates, so the diagram's real shape is a paraboloid of cycles growing out of
the base flow at Re_c. The flat amplitude-against-Re plots are this figure seen edge on.

Reads `branch.v1.csv` (order 9, the sheet born at the Hopf point) and emits a
self-contained SVG viewer in the same shell as the beam figure in the StructuralSVK
tutorial: a legend bar with a reset button over a framed stage, drag to orbit. The
projection is the yaw/pitch orthographic one from the Lorenz figure in the full-order-model
tutorial. Circles are drawn as circles: Re z₁ and Im z₁ share one scale, and only Re is
scaled apart from them.

    python3 website/tutorials/assets/karman/generate_cycles3d.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "branch.v1.csv"
OUT = HERE / "cycles_3d.html"

PURPLE, GRAY, INK, INK2, INK3, STAGE = ("#9558b2", "#8a8d99", "#e8e8ee", "#a0a0b0",
                                        "#6e6e7e", "#07070b")

ORDER = 9
N_CYCLES = 24            # circles drawn along the branch
N_PHASE = 96             # points per circle
RE_MAX = 52.0            # last cycle drawn; past it the sheet folds back over itself
RE_PAD = 1.7             # how far left of Re_c the stable steady branch is drawn


def main():
    d = np.genfromtxt(CSV, delimiter=",", names=True)
    s = d[d["order"] == ORDER]
    re_c = float(s["Re"][0])
    # Stop at RE_MAX: the sheet folds above it and comes back down through the same Re,
    # which would draw a second set of circles inside the first.
    over = np.flatnonzero(s["Re"] > RE_MAX)
    if len(over):
        s = s[:over[0]]

    # Space the cycles by amplitude, not by Re: the branch is nearly vertical at the Hopf
    # point, so Re-spacing would bunch every circle at the far end.
    idx = np.linspace(0, len(s) - 1, N_CYCLES).round().astype(int)
    th = np.linspace(0, 2 * math.pi, N_PHASE + 1)
    cycles = []
    for i in idx:
        rho, re_ = float(s["rho"][i]), float(s["Re"][i])
        pts = ",".join(f"[{re_:.4f},{rho * math.cos(t):.5f},{rho * math.sin(t):.5f}]"
                       for t in th)
        cycles.append(f"{{re:{re_:.4f},p:[{pts}]}}")

    rho_max = float(s["rho"][idx].max())
    # The steady branch spans the whole frame: solid where the base flow is stable, dashed
    # past Re_c where it is not. That needs room to the left of Re_c.
    re_lo, re_hi = re_c - RE_PAD, float(s["Re"][idx].max())

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Limit cycles against Re</title>
<style>
:root {{ --bg:#0a0a0f; --stage:#07070b; --ink:#e8e8ee; --ink2:#a0a0b0; --ink3:#6e6e7e;
  --hair:#26262f; --purple:#9558b2; --gray:#8a8d99; }}
* {{ box-sizing:border-box; }} html,body {{ margin:0; height:100%; }}
body {{ background:var(--bg); color:var(--ink); overflow:hidden;
  font:13px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif; }}
#wrap {{ display:flex; flex-direction:column; height:100%; padding:10px 12px; gap:8px; }}
#bar {{ display:flex; align-items:center; gap:14px; min-height:28px; flex-wrap:wrap; }}
.key {{ display:inline-flex; align-items:center; gap:6px; color:var(--ink3); font-size:11px; }}
.swatch {{ width:14px; height:0; border-top:2px solid currentColor; }}
.swatch.purple {{ color:var(--purple); }}
.swatch.gray {{ color:var(--gray); }}
.swatch.dash {{ color:var(--gray); border-top-style:dashed; }}
.swatch.dot {{ width:8px; height:8px; border:0; border-radius:50%; background:var(--ink); }}
button {{ margin-left:auto; font:inherit; font-size:12px; padding:4px 10px;
  border-radius:5px; cursor:pointer; background:transparent; color:var(--ink2);
  border:1px solid var(--hair); }}
button:hover {{ border-color:var(--purple); color:var(--ink); }}
#stage {{ position:relative; flex:1; min-height:0; border:1px solid var(--hair);
  border-radius:6px; background:var(--stage); overflow:hidden; }}
svg.scene {{ display:block; width:100%; height:100%; cursor:grab; touch-action:none; }}
svg.scene.drag {{ cursor:grabbing; }}
#hint {{ position:absolute; left:10px; bottom:8px; color:#4e4e5c; font-size:10.5px;
  pointer-events:none; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
#note {{ color:var(--ink3); font-size:11.5px; min-height:1.3em; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--ink2); }}
@media(max-width:560px) {{
  #wrap {{ padding:4px 6px; gap:3px; }}
  #bar {{ min-height:20px; gap:7px; flex-wrap:nowrap; }}
  .key {{ font-size:0; gap:0; }} .swatch {{ width:10px; }}
  button {{ padding:2px 6px; font-size:0; }} button::after {{ content:'reset'; font-size:9px; }}
  #note {{ display:none; }}
}}
</style></head>
<body><div id="wrap">
<div id="bar">
  <span class="key"><i class="swatch purple"></i>limit cycles</span>
  <span class="key"><i class="swatch gray"></i>steady flow, stable</span>
  <span class="key"><i class="swatch dash"></i>steady flow, unstable</span>
  <span class="key"><i class="swatch dot"></i>Hopf point</span>
  <button id="reset">reset view</button>
</div>
<div id="stage"><div id="hint">drag to orbit</div></div>
<div id="note">Each ring is one limit cycle, <code>z₁ = ρ·exp(iΩt)</code>, drawn at its own
  Reynolds number.</div>
</div>
<script>
var CYCLES = [{','.join(cycles)}];
var RE_C = {re_c:.4f}, RE_LO = {re_lo:.4f}, RE_HI = {re_hi:.4f}, RHO = {rho_max:.5f};
var NS = 'http://www.w3.org/2000/svg';
var MONO = 'ui-monospace,SFMono-Regular,Menlo,monospace';
var HOME = {{yaw:-0.62, pitch:0.30}};
var cam = {{yaw:HOME.yaw, pitch:HOME.pitch}};
var stage = document.getElementById('stage');

function el(name, attrs) {{
  var n = document.createElementNS(NS, name);
  for (var k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}}

// Orthographic yaw/pitch projection. Re is normalised by its own span, but Re z1 and
// Im z1 share one scale so that a circular orbit is drawn as a circle.
function draw() {{
  var Wp = stage.clientWidth, Hp = stage.clientHeight;
  if (!Wp || !Hp) return;
  var old = stage.querySelector('svg.scene'); if (old) old.remove();
  var svg = el('svg', {{viewBox:'0 0 '+Wp+' '+Hp, preserveAspectRatio:'none',
    'class':'scene'}});

  var reMid = (RE_LO + RE_HI) / 2, reExt = (RE_HI - RE_LO) || 1;
  var cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  var cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  function nrm(a,b,c) {{ return [(a-reMid)/reExt, b/(2*RHO), c/(2*RHO)]; }}
  function proj(a,b,c) {{
    var v = nrm(a,b,c);
    return [-v[0]*sy + v[1]*cy, -(v[0]*cy + v[1]*sy)*sp + v[2]*cp];
  }}
  function depth(a,b,c) {{
    var v = nrm(a,b,c);
    return (v[0]*cy + v[1]*sy)*cp + v[2]*sp;
  }}

  var LO = [RE_LO,-RHO,-RHO], HI = [RE_HI,RHO,RHO];
  var corners = [];
  for (var i=0;i<2;i++) for (var j=0;j<2;j++) for (var k=0;k<2;k++)
    corners.push([i?HI[0]:LO[0], j?HI[1]:LO[1], k?HI[2]:LO[2]]);
  var px = corners.map(function(c) {{ return proj(c[0],c[1],c[2]); }});
  var bx0=1e9,bx1=-1e9,by0=1e9,by1=-1e9;
  px.forEach(function(p) {{
    bx0=Math.min(bx0,p[0]); bx1=Math.max(bx1,p[0]);
    by0=Math.min(by0,p[1]); by1=Math.max(by1,p[1]);
  }});
  var m = 42;
  var sc = Math.min((Wp-2*m)/((bx1-bx0)||1), (Hp-2*m)/((by1-by0)||1));
  function P(a,b,c) {{
    var v = proj(a,b,c);
    return [Wp/2 + (v[0]-(bx0+bx1)/2)*sc, Hp/2 - (v[1]-(by0+by1)/2)*sc];
  }}
  function poly(pts, attrs) {{
    attrs.points = pts.map(function(q) {{
      var p = P(q[0],q[1],q[2]); return p[0].toFixed(1)+','+p[1].toFixed(1);
    }}).join(' ');
    return el('polyline', attrs);
  }}
  function label(a,b,c,txt,anchor,fill) {{
    var p = P(a,b,c);
    var t = el('text', {{x:p[0].toFixed(1), y:p[1].toFixed(1), 'font-size':10.5,
      fill:fill||'{INK3}', 'text-anchor':anchor||'middle', 'font-family':MONO}});
    t.textContent = txt; svg.appendChild(t);
  }}

  // Bounding box, so the depth of a ring is readable. The twelve edges, with the three
  // meeting the furthest corner drawn brighter: those are the ones the eye reads as the
  // back of the box rather than as clutter in front of the cycles.
  var far = 0, farD = 1e9;
  corners.forEach(function(c, i) {{
    var dd = depth(c[0],c[1],c[2]);
    if (dd < farD) {{ farD = dd; far = i; }}
  }});
  var box = el('g', {{fill:'none'}});
  for (var a=0;a<8;a++) for (var b=a+1;b<8;b++) {{
    var diff = 0;
    for (var t=0;t<3;t++) if (corners[a][t] !== corners[b][t]) diff++;
    if (diff !== 1) continue;
    var back = (a === far || b === far);
    box.appendChild(poly([corners[a], corners[b]],
      {{stroke: back ? '#34343f' : '#1e1e27', 'stroke-width': back ? 1 : 0.8}}));
  }}
  svg.appendChild(box);

  // Axes on the box edges nearest the viewer, each with ticks and a name.
  var yFront = -RHO, zFront = -RHO;   // which corner the Re axis is drawn along
  if (proj(RE_LO, RHO, -RHO)[1] > proj(RE_LO, -RHO, -RHO)[1]) yFront = RHO;
  var axes = el('g', {{fill:'none', stroke:'{INK3}', 'stroke-width':1}});
  axes.appendChild(poly([[RE_LO,yFront,zFront],[RE_HI,yFront,zFront]], {{}}));
  axes.appendChild(poly([[RE_HI,-RHO,zFront],[RE_HI,RHO,zFront]], {{}}));
  axes.appendChild(poly([[RE_HI,yFront,-RHO],[RE_HI,yFront,RHO]], {{}}));
  svg.appendChild(axes);

  // Labels sit just outside the box: a small fixed fraction of the half-width, not a
  // multiple of it, or they float away whenever RHO is larger than 1.
  var pad = RHO * 0.11, pad2 = RHO * 0.30;
  var yOut = yFront + Math.sign(yFront) * pad;      // just outside the Re z1 face
  var zOut = zFront - pad;                          // just below the floor
  var step = 1;
  while ((RE_HI - RE_LO) / step > 8) step += 1;
  for (var v = Math.ceil(RE_LO/step)*step; v <= RE_HI; v += step)
    label(v, yOut, zFront, String(v));
  label((RE_LO+RE_HI)/2, yFront + Math.sign(yFront)*pad2, zFront, 'Re', 'middle', '{INK2}');

  var rstep = RHO > 1.2 ? 1 : 0.5;
  var fmt = function(r) {{ return r.toFixed(rstep < 1 ? 1 : 0); }};
  for (var r = -Math.floor(RHO/rstep)*rstep; r <= RHO + 1e-9; r += rstep) {{
    if (Math.abs(r) < 1e-9) continue;
    label(RE_HI, r, zOut, fmt(r), 'start');         // Re z1 ticks along the floor edge
    label(RE_HI, yOut, r, fmt(r), 'start');         // Im z1 ticks up the vertical edge
  }}
  label(RE_HI, 0, zFront - pad2, 'Re z₁', 'start', '{INK2}');
  label(RE_HI, yFront + Math.sign(yFront)*pad2, 0, 'Im z₁', 'start', '{INK2}');

  // Painter's algorithm over the cycles and the two halves of the steady branch.
  var items = [];
  items.push({{z: depth((RE_LO+RE_C)/2,0,0), node: poly([[RE_LO,0,0],[RE_C,0,0]],
    {{stroke:'{GRAY}','stroke-width':1.8,fill:'none'}})}});
  items.push({{z: depth((RE_C+RE_HI)/2,0,0), node: poly([[RE_C,0,0],[RE_HI,0,0]],
    {{stroke:'{GRAY}','stroke-width':1.8,fill:'none','stroke-dasharray':'6 4'}})}});
  CYCLES.forEach(function(c, k) {{
    var t = k/(CYCLES.length-1);
    items.push({{z: depth(c.re,0,0), node: poly(c.p, {{
      stroke:'{PURPLE}', 'stroke-width':1.5, fill:'none',
      opacity:(0.5 + 0.5*t).toFixed(2)}})}});
  }});
  items.sort(function(a,b) {{ return a.z - b.z; }});
  items.forEach(function(it) {{ svg.appendChild(it.node); }});

  var hp = P(RE_C,0,0);
  svg.appendChild(el('circle', {{cx:hp[0].toFixed(1), cy:hp[1].toFixed(1), r:4,
    fill:'{INK}', stroke:'{STAGE}', 'stroke-width':1.2}}));
  label(RE_C, 0, 0, '', 'start');
  var t = el('text', {{x:(hp[0]+9).toFixed(1), y:(hp[1]-9).toFixed(1), 'font-size':11,
    fill:'{INK2}', 'font-family':MONO}});
  t.textContent = 'Re_c ≈ ' + RE_C.toFixed(2);
  svg.appendChild(t);

  stage.appendChild(svg);
}}

var drag = null;
function grab(on) {{
  var s = stage.querySelector('svg.scene');
  if (s) s.classList.toggle('drag', on);
}}
stage.addEventListener('pointerdown', function(ev) {{
  drag = {{x:ev.clientX, y:ev.clientY}}; grab(true);
}});
stage.addEventListener('pointermove', function(ev) {{
  if (!drag) return;
  cam.yaw += (ev.clientX - drag.x) * 0.008;
  cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + (ev.clientY - drag.y) * 0.006));
  drag = {{x:ev.clientX, y:ev.clientY}};
  draw(); grab(true);
}});
['pointerup','pointerleave','pointercancel'].forEach(function(e) {{
  stage.addEventListener(e, function() {{ drag = null; grab(false); }});
}});
document.getElementById('reset').addEventListener('click', function() {{
  cam.yaw = HOME.yaw; cam.pitch = HOME.pitch; draw();
}});
window.addEventListener('resize', draw);
draw();
</script>
</body></html>
"""
    OUT.write_text(html)
    print(f"wrote {OUT.name}  ({OUT.stat().st_size // 1024} KB, {len(cycles)} cycles, "
          f"Re {re_c:.2f}..{re_hi:.2f}, rho_max {rho_max:.3f})")


main()
