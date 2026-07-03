#!/usr/bin/env python3
"""Generate the animated Kármán hero figure (karman_flow.html).

Reads the shipped VTK output (base_flow.vtu + eigenmode_001.vtu) and emits a
self-contained WebGL asset animating the vorticity of

	w(x, t) = w_base(x) + eps * Re[ w_mode(x) e^{i θ(t)} ]

on the actual FE mesh (6 636 vertices, 12 943 triangles). The phase blend happens in
the vertex shader; the colormap is the site diverging palette as a 1-D LUT texture.
Embedded in tutorial-karman.html via <iframe>. Rerun after regenerating the VTUs:

	python3 website/assets/karman/generate_flow_anim.py
"""
from __future__ import annotations

import base64
from pathlib import Path

import meshio
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PV = REPO / "examples/05_karman_vortex_street/results/paraview"

BG, INK3, HAIR = "#07070b", "#6e6e7e", "#26262f"
OMEGA0 = 16.859  # rad/s — Hopf frequency (annotation + real-time scaling)
SLOW = 1 / 7    # playback speed relative to real time
EPS_REL = 0.35  # mode amplitude relative to base-flow vorticity scale

b = meshio.read(PV / "base_flow.vtu")
m = meshio.read(PV / "eigenmodes/eigenmode_001.vtu")

pts = b.points[:, :2].astype(np.float64)
tris = b.cells[0].data.astype(np.uint16)
wb = b.point_data["vorticity"][:, 0]
wre = m.point_data["vorticity_Re"][:, 0]
wim = m.point_data["vorticity_Im"][:, 0]

eps = EPS_REL * np.percentile(np.abs(wb), 99) / np.percentile(np.abs(wre), 99)
lim = np.percentile(np.abs(wb + eps * wre), 99)

# clip-space positions: x ∈ [0, 2.2] → [-1, 1], y ∈ [0, 0.41] → [-1, 1]
pos = np.empty_like(pts, dtype=np.float32)
pos[:, 0] = pts[:, 0] / 2.2 * 2 - 1
pos[:, 1] = pts[:, 1] / 0.41 * 2 - 1
scal = np.column_stack([wb, wre, wim]).astype(np.float32)

# cylinder disk + ring (drawn on top), clip space
NSEG = 48
th = np.linspace(0, 2 * np.pi, NSEG, endpoint=False)
cx, cy, r = 0.2, 0.2, 0.05
ring = np.column_stack([(cx + r * np.cos(th)) / 2.2 * 2 - 1,
						(cy + r * np.sin(th)) / 0.41 * 2 - 1]).astype(np.float32)
disk = np.vstack([[(cx / 2.2 * 2 - 1), (cy / 0.41 * 2 - 1)], ring, ring[:1]]).astype(np.float32)

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

html = f'''<!doctype html>
<html><head><meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  html,body {{ margin:0; background:{BG}; font-family:'JetBrains Mono',monospace; overflow:hidden; }}
  #wrap {{ position:relative; }}
  canvas {{ display:block; width:100%; height:auto; }}
  #play {{
    position:absolute; top:8px; right:8px; width:28px; height:28px;
    background:rgba(255,255,255,0.05); border:1px solid {HAIR}; border-radius:4px;
    color:{INK3}; cursor:pointer; font-size:12px; line-height:26px; text-align:center;
    user-select:none;
  }}
  #play:hover {{ background:rgba(255,255,255,0.1); color:#e8e8ee; }}
  #tag {{
    position:absolute; left:10px; bottom:8px; font-size:10px; letter-spacing:0.06em;
    color:{INK3}; pointer-events:none;
  }}
</style></head>
<body>
<div id="wrap">
  <canvas id="gl" width="1720" height="320"></canvas>
  <div id="play" title="play / pause">❚❚</div>
  <div id="tag">ω₀ = {OMEGA0:.2f} rad/s · playback {SLOW:.2g}× real time</div>
</div>
<script>
function decode(b64, T) {{
  var s = atob(b64), n = s.length, u = new Uint8Array(n);
  for (var i = 0; i < n; i++) u[i] = s.charCodeAt(i);
  return new T(u.buffer);
}}
var POS  = decode('{b64(pos)}', Float32Array);
var SCAL = decode('{b64(scal)}', Float32Array);
var IDX  = decode('{b64(tris)}', Uint16Array);
var DISK = decode('{b64(disk)}', Float32Array);
var RING = decode('{b64(ring)}', Float32Array);
var EPS = {eps:.6f}, LIM = {lim:.6f}, OMEGA = {OMEGA0} * {SLOW};

var cv = document.getElementById('gl');
var gl = cv.getContext('webgl', {{antialias: true}});

function sh(type, src) {{
  var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
  return s;
}}
function prog(vs, fs) {{
  var p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw gl.getProgramInfoLog(p);
  return p;
}}

var pField = prog(
  'attribute vec2 aPos; attribute vec3 aW; uniform float uCs, uSn;' +
  'varying float vT;' +
  'void main() {{' +
  '  float w = aW.x + ' + EPS + ' * (aW.y * uCs - aW.z * uSn);' +
  '  vT = clamp(w / ' + LIM + ' * 0.5 + 0.5, 0.004, 0.996);' +
  '  gl_Position = vec4(aPos, 0.0, 1.0);' +
  '}}',
  'precision mediump float; uniform sampler2D uLut; varying float vT;' +
  'void main() {{ gl_FragColor = texture2D(uLut, vec2(vT, 0.5)); }}');

var pFlat = prog(
  'attribute vec2 aPos; void main() {{ gl_Position = vec4(aPos, 0.0, 1.0); }}',
  'precision mediump float; uniform vec4 uCol; void main() {{ gl_FragColor = uCol; }}');

// site diverging LUT: julia-blue → page background → julia-red
var stops = [[0.00,'7ea2ff'],[0.20,'4063d8'],[0.44,'0b0b12'],[0.50,'07070b'],
             [0.56,'0b0b12'],[0.80,'cb3c33'],[1.00,'ff9d8a']];
function hex(h) {{ return [parseInt(h.substr(0,2),16), parseInt(h.substr(2,2),16), parseInt(h.substr(4,2),16)]; }}
var lut = new Uint8Array(256 * 4);
for (var i = 0; i < 256; i++) {{
  var t = i / 255, k = 0;
  while (k < stops.length - 2 && t > stops[k+1][0]) k++;
  var f = (t - stops[k][0]) / (stops[k+1][0] - stops[k][0]);
  var c0 = hex(stops[k][1]), c1 = hex(stops[k+1][1]);
  for (var c = 0; c < 3; c++) lut[i*4+c] = Math.round(c0[c] + (c1[c]-c0[c]) * Math.min(Math.max(f,0),1));
  lut[i*4+3] = 255;
}}
var tex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, tex);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);

function buf(data, target) {{
  var b = gl.createBuffer(); target = target || gl.ARRAY_BUFFER;
  gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW);
  return b;
}}
var bPos = buf(POS), bScal = buf(SCAL), bIdx = buf(IDX, gl.ELEMENT_ARRAY_BUFFER);
var bDisk = buf(DISK), bRing = buf(RING);

var aPosF = gl.getAttribLocation(pField, 'aPos'), aW = gl.getAttribLocation(pField, 'aW');
var uCs = gl.getUniformLocation(pField, 'uCs'), uSn = gl.getUniformLocation(pField, 'uSn');
var uLut = gl.getUniformLocation(pField, 'uLut');
var aPosD = gl.getAttribLocation(pFlat, 'aPos'), uCol = gl.getUniformLocation(pFlat, 'uCol');

function draw(theta) {{
  gl.viewport(0, 0, cv.width, cv.height);
  gl.useProgram(pField);
  gl.bindBuffer(gl.ARRAY_BUFFER, bPos);
  gl.enableVertexAttribArray(aPosF); gl.vertexAttribPointer(aPosF, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, bScal);
  gl.enableVertexAttribArray(aW); gl.vertexAttribPointer(aW, 3, gl.FLOAT, false, 0, 0);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(uLut, 0);
  gl.uniform1f(uCs, Math.cos(theta)); gl.uniform1f(uSn, Math.sin(theta));
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bIdx);
  gl.drawElements(gl.TRIANGLES, IDX.length, gl.UNSIGNED_SHORT, 0);
  gl.disableVertexAttribArray(aW);

  gl.useProgram(pFlat);
  gl.bindBuffer(gl.ARRAY_BUFFER, bDisk);
  gl.enableVertexAttribArray(aPosD); gl.vertexAttribPointer(aPosD, 2, gl.FLOAT, false, 0, 0);
  gl.uniform4f(uCol, 0.078, 0.078, 0.11, 1.0);
  gl.drawArrays(gl.TRIANGLE_FAN, 0, DISK.length / 2);
  gl.bindBuffer(gl.ARRAY_BUFFER, bRing);
  gl.vertexAttribPointer(aPosD, 2, gl.FLOAT, false, 0, 0);
  gl.uniform4f(uCol, 0.43, 0.43, 0.49, 1.0);
  gl.drawArrays(gl.LINE_LOOP, 0, RING.length / 2);
}}

var playing = true, t0 = performance.now(), phase0 = 0;
var btn = document.getElementById('play');
function toggle() {{
  if (playing) {{ phase0 = phase0 + (performance.now() - t0) / 1000 * OMEGA; btn.textContent = '▶'; }}
  else {{ t0 = performance.now(); btn.textContent = '❚❚'; }}
  playing = !playing;
}}
btn.addEventListener('click', toggle);
cv.addEventListener('click', toggle);

function frame() {{
  var theta = playing ? phase0 + (performance.now() - t0) / 1000 * OMEGA : phase0;
  draw(theta);
  requestAnimationFrame(frame);
}}
frame();
</script>
</body></html>'''

out = HERE / "karman_flow.html"
out.write_text(html)
print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
