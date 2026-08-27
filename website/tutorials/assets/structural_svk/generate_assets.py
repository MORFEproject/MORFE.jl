#!/usr/bin/env python3
"""Build the self-contained StructuralSVK tutorial assets.

The plotted data comes from one order-9 conservative ROM.  Orders 3, 5, 7,
and 9 are nested truncations of that same R polynomial and of the same W row
at Ferrite node 289, transverse direction y.

Run from the MORFE repository root:

    python3 website/tutorials/assets/structural_svk/generate_assets.py
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FERRITE = next(
    p
    for p in (
        REPO.parent / "MORFEFerrite" / "MORFEFerrite.jl",
        REPO.parent / "MORFEFerrite.jl",
    )
    if p.is_dir()
)
EXAMPLE = FERRITE / "examples" / "01_clamped_beam_ferrite"
REFERENCE = EXAMPLE / "reference_data"
MESH = EXAMPLE / "clamped_clamped_beam.msh"

PROBE_NODE = 289
PROBE_XYZ = (500.0, 10.0 / 3.0, 24.0)
PROBE_GLOBAL_DOF = 2468
PROBE_FREE_DOF = 2405
ORDERS = (3, 5, 7, 9)
COLORS = {3: "#4063d8", 5: "#389826", 7: "#cb3c33", 9: "#9558b2"}
N_RADIUS = 341
R_MAX = 85.0
N_PHASE = 4096
TRANSVERSE_THICKNESS = 10.0
MODE_MAX_Y = 10.0 * TRANSVERSE_THICKNESS
HOME_FREQUENCY_CHANGE = 13.0


def read_nodes_and_hexes(path: Path):
    lines = path.read_text().splitlines()

    i = lines.index("$Nodes") + 1
    nblocks, nnodes, *_ = map(int, lines[i].split())
    i += 1
    nodes = {}
    for _ in range(nblocks):
        entity_dim, _entity_tag, parametric, count = map(int, lines[i].split())
        i += 1
        tags = []
        while len(tags) < count:
            tags.extend(map(int, lines[i].split()))
            i += 1
        for tag in tags:
            values = list(map(float, lines[i].split()))
            i += 1
            nodes[tag] = tuple(values[:3])
            if parametric and len(values) != 3 + entity_dim:
                raise ValueError("unexpected parametric node record")
    assert len(nodes) == nnodes == 328
    assert lines[i] == "$EndNodes"

    i = lines.index("$Elements") + 1
    nblocks, _nelements, *_ = map(int, lines[i].split())
    i += 1
    hexes = []
    for _ in range(nblocks):
        _entity_dim, _entity_tag, element_type, count = map(int, lines[i].split())
        i += 1
        for _ in range(count):
            row = list(map(int, lines[i].split()))
            i += 1
            if element_type == 5:  # Gmsh 8-node hexahedron
                assert len(row) == 9
                hexes.append(tuple(row[1:]))
    assert len(hexes) == 120
    return nodes, hexes


def boundary_faces(hexes):
    local_faces = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    faces = {}
    counts = defaultdict(int)
    for cell in hexes:
        for loc in local_faces:
            face = tuple(cell[j] for j in loc)
            key = tuple(sorted(face))
            counts[key] += 1
            faces[key] = face
    return [faces[key] for key, count in counts.items() if count == 1]


def read_complex_csv(path: Path, prefix: str):
    rows = []
    with path.open(newline="") as io:
        for row in csv.DictReader(io):
            a, b = int(row["exp_1"]), int(row["exp_2"])
            rows.append(
                (a, b, complex(float(row[f"{prefix}_re"]), float(row[f"{prefix}_im"])))
            )
    return rows


def extract_first_mode():
    code = f'''using MORFE, MORFEFerrite
const SVK = MORFEFerrite.StructuralSVK
case = SVK.mechanical_model({json.dumps(str(MESH))};
    material = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
    damping = SVK.RayleighDamping(α = 0.0, β = 0.0),
    dirichlet = "Dirichlet")
n = {PROBE_NODE}
@assert all(isapprox.(Tuple(case.info.dh.grid.nodes[n].x), {PROBE_XYZ}; atol = 1e-8))
g = SVK.Common.node_dof(case.info.dh, n, 2)
@assert g == {PROBE_GLOBAL_DOF}
@assert case.info.free_to_local[g] == {PROBE_FREE_DOF}
(; spectral) = build_model(case; master = [1], expansion_order = 3)
phi = MORFE.right_modes(spectral)[:, 1]
phase = cis(-angle(phi[case.info.free_to_local[g]]))
u = real.(phase .* phi)
@assert u[case.info.free_to_local[g]] > 0
for node in eachindex(case.info.dh.grid.nodes)
    values = ntuple(3) do component
        dof = SVK.Common.node_dof(case.info.dh, node, component)
        haskey(case.info.free_to_local, dof) ? u[case.info.free_to_local[dof]] : 0.0
    end
    println("MODE_DATA ", node, " ", values[1], " ", values[2], " ", values[3])
end
'''
    result = subprocess.run(
        [
            "julia",
            f"--project={EXAMPLE}",
            "--startup-file=no",
            "-e",
            code,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mode = {}
    for line in result.stdout.splitlines():
        if line.startswith("MODE_DATA "):
            _, node, ux, uy, uz = line.split()
            mode[int(node)] = (float(ux), float(uy), float(uz))
    assert len(mode) == 328, result.stdout
    scale = MODE_MAX_Y / max(abs(u[1]) for u in mode.values())
    mode = {node: tuple(scale * value for value in u) for node, u in mode.items()}
    mode = {
        node: tuple(float(f"{value:.10g}") for value in u)
        for node, u in mode.items()
    }
    assert abs(max(abs(u[1]) for u in mode.values()) - MODE_MAX_Y) < 1e-10
    assert mode[PROBE_NODE][1] > 0.0
    return mode


def physical_amplitude(terms, order: int, radius: float, phase_basis):
    if radius == 0.0:
        return 0.0
    harmonics = defaultdict(complex)
    for a, b, coefficient in terms:
        degree = a + b
        if degree <= order:
            harmonics[a - b] += coefficient * radius**degree
    values = [0.0] * N_PHASE
    for harmonic, coefficient in harmonics.items():
        cosines, sines = phase_basis[harmonic]
        re, im = coefficient.real, coefficient.imag
        for i in range(N_PHASE):
            values[i] += re * cosines[i] - im * sines[i]
    return 0.5 * (max(values) - min(values))


def backbone_home_limits(resonant, w_terms, omega0, phase_basis):
    def signed_frequency_change(radius):
        omega = sum(
            coefficient.imag * radius ** (degree - 1)
            for degree, coefficient in resonant.items()
            if degree <= 9
        )
        return 100.0 * (omega / omega0 - 1.0) - HOME_FREQUENCY_CHANGE

    lo = 0.0
    flo = signed_frequency_change(lo)
    bracket = None
    for i in range(1, N_RADIUS):
        hi = R_MAX * i / (N_RADIUS - 1)
        fhi = signed_frequency_change(hi)
        if flo <= 0.0 <= fhi:
            bracket = (lo, hi)
            break
        lo, flo = hi, fhi
    assert bracket is not None

    lo, hi = bracket
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if signed_frequency_change(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    radius = 0.5 * (lo + hi)
    physical = (
        100.0
        * physical_amplitude(w_terms, 9, radius, phase_basis)
        / TRANSVERSE_THICKNESS
    )
    assert abs(signed_frequency_change(radius)) < 1e-10
    assert math.isclose(radius, 59.9418323, rel_tol=2e-9)
    assert math.isclose(physical, 73.2501101, rel_tol=2e-9)
    return {"modal": radius, "physical": physical}


def build_backbone_data():
    r_terms = read_complex_csv(REFERENCE / "R_coefficients_ref.csv", "R1")
    w_terms = read_complex_csv(
        REFERENCE / "W_node289_y_coefficients_ref.csv", "W_y"
    )
    resonant = {(a + b): c for a, b, c in r_terms if a == b + 1}
    assert tuple(sorted(resonant)) == (1, 3, 5, 7, 9)
    assert len(w_terms) == sum(degree + 1 for degree in range(1, 10)) == 54
    omega0 = resonant[1].imag
    assert omega0 > 0.0
    assert max(abs(c.real) for c in resonant.values()) < 1e-16

    phase_basis = {
        harmonic: (
            [math.cos(harmonic * 2.0 * math.pi * i / N_PHASE) for i in range(N_PHASE)],
            [math.sin(harmonic * 2.0 * math.pi * i / N_PHASE) for i in range(N_PHASE)],
        )
        for harmonic in range(-9, 10)
    }
    home = backbone_home_limits(resonant, w_terms, omega0, phase_basis)
    radii = [R_MAX * i / (N_RADIUS - 1) for i in range(N_RADIUS)]
    curves = {}
    for order in ORDERS:
        points = []
        for radius in radii:
            omega = sum(
                coefficient.imag * radius ** (degree - 1)
                for degree, coefficient in resonant.items()
                if degree <= order
            )
            amplitude = physical_amplitude(w_terms, order, radius, phase_basis)
            points.append(
                {
                    "r": round(radius, 12),
                    "amplitude": float(f"{amplitude:.14g}"),
                    "amplitude_ratio": float(
                        f"{amplitude / TRANSVERSE_THICKNESS:.14g}"
                    ),
                    "omega": float(f"{omega:.14g}"),
                    "omega_ratio": float(f"{omega / omega0:.14g}"),
                }
            )
        curves[str(order)] = {"color": COLORS[order], "points": points}

    data = {
        "schema": "morfe.structural_svk.backbone.v1",
        "source": "one conservative order-9 complex-normal-form ROM",
        "orders": list(ORDERS),
        "omega0": omega0,
        "radius_range": [0.0, R_MAX],
        "radius_samples": N_RADIUS,
        "phase_samples": N_PHASE,
        "probe": {
            "node": PROBE_NODE,
            "coordinate": list(PROBE_XYZ),
            "direction": "y",
            "global_dof": PROBE_GLOBAL_DOF,
            "free_dof": PROBE_FREE_DOF,
            "amplitude": "half peak-to-peak over one phase cycle",
            "units": "mesh length units",
            "transverse_thickness": TRANSVERSE_THICKNESS,
        },
        "curves": curves,
    }
    (HERE / "backbone.v1.json").write_text(json.dumps(data, indent=2) + "\n")
    return data, home


MESH_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clamped beam mesh and first mode</title>
<style>
:root{--bg:#0a0a0f;--stage:#07070b;--ink:#e8e8ee;--ink2:#a0a0b0;--ink3:#6e6e7e;--hair:#26262f;--purple:#9558b2;--blue:#4063d8;--green:#389826;--gray:#565963}
*{box-sizing:border-box}html,body{margin:0;height:100%}body{background:var(--bg);color:var(--ink);overflow:hidden;font:13px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif}
#wrap{display:flex;flex-direction:column;height:100%;padding:10px 12px;gap:8px}#bar{display:flex;align-items:center;gap:14px;min-height:28px;flex-wrap:wrap}
.key{display:inline-flex;align-items:center;gap:6px;color:var(--ink3);font-size:11px}.swatch{width:14px;height:8px;border:1px solid currentColor;background:currentColor}.swatch.gray{color:var(--gray)}.swatch.purple{color:var(--purple)}.swatch.blue{color:var(--blue)}.swatch.green{height:0;border:0;border-top:2px solid var(--green);color:var(--green)}
button{margin-left:auto;font:inherit;font-size:12px;padding:4px 10px;border-radius:5px;cursor:pointer;background:transparent;color:var(--ink2);border:1px solid var(--hair)}button:hover{border-color:var(--purple);color:var(--ink)}
#stage{position:relative;flex:1;min-height:0;border:1px solid var(--hair);border-radius:6px;background:var(--stage);overflow:hidden}canvas{display:block;width:100%;height:100%;cursor:grab;touch-action:none}canvas.drag{cursor:grabbing}
#note{color:var(--ink3);font-size:11.5px;min-height:1.3em}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink2)}
@media(max-width:560px){#wrap{padding:4px 6px;gap:3px}#bar{min-height:20px;gap:7px;flex-wrap:nowrap}.key{font-size:0;gap:0}.swatch{width:10px;height:6px}button{padding:2px 6px;font-size:0;white-space:nowrap}button::after{content:'reset';font-size:9px}#note{display:none}}
</style></head><body><div id="wrap">
<div id="bar"><span class="key" title="reference mesh" aria-label="reference mesh"><i class="swatch gray"></i>reference</span><span class="key" title="clamped ends" aria-label="clamped ends"><i class="swatch purple"></i>clamped</span><span class="key" title="first mode" aria-label="first mode"><i class="swatch blue"></i>mode 1</span><span class="key" title="node 289 y-displacement" aria-label="node 289 y-displacement"><i class="swatch green"></i>node 289 · y</span><button id="reset">reset view</button></div>
<div id="stage"><canvas id="mesh" aria-label="Rotatable gray reference mesh and opaque blue first bending mode of the clamped beam"></canvas></div>
<div id="note">Drag to rotate · mode displayed at <code>max |u<sub>y</sub>| / t<sub>y</sub> = 10</code></div>
</div><script>
const NODES=__NODES__,MODE=__MODE__,FACES=__FACES__,PROBE=289,NODE_IDS=Object.keys(NODES).map(Number);
const cv=document.getElementById('mesh'),ctx=cv.getContext('2d');let yaw=-.34,pitch=.42,drag=false,last=[0,0];
function isClamp(face){return face.every(id=>NODES[id][0]<1e-7||NODES[id][0]>999.999999)}
function coordinate(id,deformed){const q=NODES[id],u=MODE[id];return deformed?[q[0]+u[0],q[1]+u[1],q[2]+u[2]]:q}
function rotate(q){const x=q[0]-500,y=q[1]-5,z=q[2]-12,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z;return[x1,cp*y-sp*z1,sp*y+cp*z1]}
function rotateVector(q){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*q[0]+sy*q[2],z1=-sy*q[0]+cy*q[2];return[x1,cp*q[1]-sp*z1,sp*q[1]+cp*z1]}
function view(){const ref={},def={};let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;for(const id of NODE_IDS){ref[id]=rotate(coordinate(id,false));def[id]=rotate(coordinate(id,true));for(const p of[ref[id],def[id]]){xmin=Math.min(xmin,p[0]);xmax=Math.max(xmax,p[0]);ymin=Math.min(ymin,p[1]);ymax=Math.max(ymax,p[1])}}const w=cv.clientWidth,h=cv.clientHeight,padX=Math.min(92,w*.12),padY=Math.min(76,Math.max(16,h*.28)),s=Math.min((w-padX)/(xmax-xmin),(h-padY)/(ymax-ymin)),cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;return{ref,def,project:p=>[w/2+(p[0]-cx)*s,h/2-(p[1]-cy)*s,p[2]]}}
function polygon(points,fill,stroke,width){ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);for(let i=1;i<points.length;i++)ctx.lineTo(points[i][0],points[i][1]);ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}
function arrow(a,b,color,width,head=9){const dx=b[0]-a[0],dy=b[1]-a[1],angle=Math.atan2(dy,dx);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.beginPath();ctx.moveTo(b[0],b[1]);ctx.lineTo(b[0]-head*Math.cos(angle-.45),b[1]-head*Math.sin(angle-.45));ctx.lineTo(b[0]-head*Math.cos(angle+.45),b[1]-head*Math.sin(angle+.45));ctx.closePath();ctx.fill()}
function triad(){const size=Math.min(30,Math.max(16,cv.clientHeight*.24)),origin=[size+10,cv.clientHeight-size-10],axes=[[[1,0,0],'#cb3c33','x'],[[0,1,0],'#389826','y'],[[0,0,1],'#4063d8','z']];ctx.font='600 11px ui-monospace,SFMono-Regular,Menlo,monospace';ctx.textAlign='center';ctx.textBaseline='middle';for(const[v,color,label]of axes){const r=rotateVector(v),dx=size*r[0],dy=-size*r[1],end=[origin[0]+dx,origin[1]+dy],length=Math.hypot(dx,dy),lx=length>1e-6?end[0]+6*dx/length:end[0]+6,ly=length>1e-6?end[1]+6*dy/length:end[1]-6;arrow(origin,end,color,1.5,6);ctx.fillStyle=color;ctx.fillText(label,lx,ly)}ctx.fillStyle='#e8e8ee';ctx.beginPath();ctx.arc(origin[0],origin[1],2.2,0,Math.PI*2);ctx.fill();ctx.textAlign='start';ctx.textBaseline='alphabetic'}
function draw(){if(!cv.width)return;ctx.clearRect(0,0,cv.clientWidth,cv.clientHeight);const v=view(),items=[];for(const face of FACES){const clamp=isClamp(face),ref=face.map(id=>v.project(v.ref[id]));items.push({points:ref,z:ref.reduce((s,p)=>s+p[2],0)/ref.length,kind:clamp?'clamp':'reference'});if(!clamp){const mode=face.map(id=>v.project(v.def[id]));items.push({points:mode,z:mode.reduce((s,p)=>s+p[2],0)/mode.length,kind:'mode'})}}items.sort((a,b)=>a.z-b.z);for(const item of items){if(item.kind==='clamp')polygon(item.points,'#9558b2','#c096d4',1.15);else if(item.kind==='mode')polygon(item.points,'#4063d8','#89a6ff',.8);else polygon(item.points,'#3f424b','#797d87',.72)}const tail=v.project(v.ref[PROBE]),head=v.project(v.def[PROBE]);arrow(tail,head,'#389826',2.4,10);ctx.fillStyle='#389826';ctx.beginPath();ctx.arc(tail[0],tail[1],3.5,0,Math.PI*2);ctx.fill();ctx.font='600 11px ui-monospace,SFMono-Regular,Menlo,monospace';const label='y-displacement',mx=(tail[0]+head[0])/2+9,my=(tail[1]+head[1])/2-7,tw=ctx.measureText(label).width;ctx.fillStyle='#07070b';ctx.fillRect(mx-4,my-12,tw+8,17);ctx.fillStyle='#55b841';ctx.fillText(label,mx,my);triad()}
function fit(){const d=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=Math.round(w*d);cv.height=Math.round(h*d);ctx.setTransform(d,0,0,d,0,0);draw()}
cv.addEventListener('pointerdown',e=>{drag=true;last=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId);cv.classList.add('drag')});cv.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-last[0])*.008;pitch=Math.max(-1.25,Math.min(1.25,pitch+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY];draw()});function stop(){drag=false;cv.classList.remove('drag')}cv.addEventListener('pointerup',stop);cv.addEventListener('pointercancel',stop);document.getElementById('reset').onclick=()=>{yaw=-.34;pitch=.42;draw()};addEventListener('resize',fit);fit();
</script></body></html>'''


BACKBONE_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Structural SVK backbone</title>
<style>
:root{--bg:#0a0a0f;--stage:#07070b;--ink:#e8e8ee;--ink2:#a0a0b0;--ink3:#6e6e7e;--hair:#26262f;--purple:#9558b2}*{box-sizing:border-box}html,body{margin:0;height:100%}body{background:var(--bg);color:var(--ink);overflow:hidden;font:13px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif}
#wrap{display:flex;flex-direction:column;height:100%;padding:10px 12px;gap:8px}#bar{display:flex;gap:6px;align-items:center;min-height:28px}button{font:inherit;font-size:12px;padding:4px 10px;border-radius:5px;cursor:pointer;background:transparent;color:var(--ink2);border:1px solid var(--hair)}button:hover{border-color:var(--purple);color:var(--ink)}button.on{background:rgba(149,88,178,.16);border-color:var(--purple);color:var(--ink)}#meta{margin-left:auto;color:var(--ink3);font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.04em}
#stage{position:relative;flex:1;min-height:0;border:1px solid var(--hair);border-radius:6px;background:var(--stage);overflow:hidden}svg{display:block;width:100%;height:100%}.axis{stroke:var(--ink3);stroke-width:1}.grid{stroke:var(--hair);stroke-width:.8}.tick,.label{fill:var(--ink3);font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.label{fill:var(--ink2)}.curve{fill:none;stroke-width:2}.teaser .curve{stroke-width:6}.teaser .tick,.teaser .label{font-size:18px}.tool{cursor:pointer}.tool .btn{fill:rgba(255,255,255,.04);stroke:var(--hair)}.tool .icn{stroke:var(--ink3)}.tool:hover .btn{fill:rgba(255,255,255,.09)}.tool.active .btn{stroke:var(--purple);fill:rgba(149,88,178,.12)}.tool.active .icn{stroke:var(--purple)}svg.mode-zoom #hit{cursor:crosshair}svg.mode-pan #hit{cursor:grab}svg.mode-pan.panning #hit{cursor:grabbing}svg.mode-zoom,svg.mode-pan{touch-action:none}
#legend{display:flex;gap:14px;align-items:center;flex-wrap:wrap}#legend button{border:0;padding:0;color:var(--ink3);font-size:11px}#legend button.off{opacity:.3}#legend i{display:inline-block;width:14px;height:0;vertical-align:3px;border-top:2.5px solid currentColor;margin-right:6px}#note{color:var(--ink3);font-size:11.5px;min-height:1.3em}.tip{position:absolute;pointer-events:none;display:none;background:#14141c;border:1px solid var(--hair);border-radius:5px;padding:6px 9px;font-size:11.5px;color:var(--ink);white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;z-index:5}.compact #note{display:none}.compact #wrap{padding:7px 9px;gap:6px}.compact #meta{display:none}.teaser #wrap{padding:10px 12px;gap:8px}.teaser #bar,.teaser #toolbar,.teaser #probe,.teaser #zoombox,.teaser #note,.teaser #tip{display:none}.teaser #hit{pointer-events:none}.teaser #legend{gap:12px;padding:0 2px}.teaser #legend span{color:var(--ink3);font-size:11px}.teaser #legend i{border-top-width:6px}
</style></head><body><div id="wrap"><div id="bar"><button id="physical">transverse displacement</button><button id="modal">modal coordinate</button><span id="meta">one order-9 ROM · nested truncations</span></div>
<div id="stage"><svg id="chart" viewBox="0 0 900 470" role="img" aria-label="Order 3, 5, 7 and 9 backbone curves in physical and modal coordinates">
<defs><clipPath id="plot-clip"><rect x="78" y="22" width="798" height="388"/></clipPath></defs><g id="grid"></g><g id="axes"></g><g clip-path="url(#plot-clip)"><g id="curves"></g><g id="probe" visibility="hidden"><line id="cross-x" stroke="#6e6e7e" stroke-dasharray="3 3"/><line id="cross-y" stroke="#6e6e7e" stroke-dasharray="3 3"/><circle id="dot" r="4" stroke="#07070b"/></g><rect id="zoombox" visibility="hidden" fill="rgba(149,88,178,.10)" stroke="#6e6e7e" stroke-width=".8" stroke-dasharray="4 3" pointer-events="none"/></g>
<text class="label" x="474" y="459" text-anchor="middle">frequency change Δω / ω₀ [%]</text><text id="ylabel" class="label" transform="translate(18 235) rotate(-90)" text-anchor="middle"></text><rect id="hit" x="78" y="22" width="798" height="388" fill="transparent"/>
<g id="toolbar"><g class="tool" id="tool-zoom" transform="translate(784 14)"><title>Zoom: drag a rectangle</title><rect class="btn" width="26" height="26" rx="4"/><g class="icn" transform="translate(4 4)"><circle cx="7.5" cy="7.5" r="5" fill="none" stroke-width="1.6"/><line x1="11.2" y1="11.2" x2="16" y2="16" stroke-width="1.8"/><line x1="5.2" y1="7.5" x2="9.8" y2="7.5" stroke-width="1.3"/><line x1="7.5" y1="5.2" x2="7.5" y2="9.8" stroke-width="1.3"/></g></g><g class="tool" id="tool-pan" transform="translate(816 14)"><title>Pan: drag to move</title><rect class="btn" width="26" height="26" rx="4"/><g class="icn" transform="translate(4 3.5)"><path fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" d="M7.2 10.5 V4.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M9.7 9 V3.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M12.2 9.3 V4.8 a1.25 1.25 0 0 1 2.5 0 V11.5 c0 3.4 -2 5.4 -4.9 5.4 c-2.3 0 -3.4 -.9 -4.5 -2.6 L3.4 11.1 a1.3 1.3 0 0 1 2.2 -1.3 l1.6 2.2"/></g></g><g class="tool" id="tool-home" transform="translate(848 14)"><title>Reset view</title><rect class="btn" width="26" height="26" rx="4"/><g class="icn" transform="translate(4 4)"><path fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" d="M2.2 8.2 L9 2.6 L15.8 8.2 M4.2 7.5 V15 H13.8 V7.5"/></g></g></g></svg><div id="tip" class="tip"></div></div>
<div id="legend"></div><div id="note">Home view ends at order 9's 13% frequency change; arm a toolbar tool to zoom or pan.</div></div>
<script>
const DATA=__DATA__,ORDERS=[3,5,7,9],HOME={physical:{x0:-1,x1:13,y0:0,y1:__HOME_PHYSICAL__},modal:{x0:-1,x1:13,y0:0,y1:__HOME_MODAL__}},NS='http://www.w3.org/2000/svg',M={l:78,r:24,t:22,b:60},W=900,H=470,params=new URLSearchParams(location.search),teaser=params.has('teaser');let panel='physical',tool=null,pan=null,zoomStart=null;const V={...HOME.physical},visible={3:true,5:true,7:true,9:true},svg=document.getElementById('chart'),stage=document.getElementById('stage'),hit=document.getElementById('hit'),tip=document.getElementById('tip'),zoombox=document.getElementById('zoombox');if(params.has('compact'))document.body.classList.add('compact');if(teaser)document.body.classList.add('teaser');
function el(name,attrs={}){const node=document.createElementNS(NS,name);for(const[key,value]of Object.entries(attrs))node.setAttribute(key,value);return node}function xv(q){return 100*(q.omega_ratio-1)}function yv(q){return panel==='modal'?q.r:100*q.amplitude_ratio}function px(x){return M.l+(x-V.x0)/(V.x1-V.x0)*(W-M.l-M.r)}function py(y){return H-M.b-(y-V.y0)/(V.y1-V.y0)*(H-M.t-M.b)}function dx(x){return V.x0+(x-M.l)/(W-M.l-M.r)*(V.x1-V.x0)}function dy(y){return V.y0+(H-M.b-y)/(H-M.t-M.b)*(V.y1-V.y0)}function fmt(value,digits=2){return Number(value).toFixed(digits).replace(/0+$/,'').replace(/\.$/,'')}function niceTicks(lo,hi,count){const span=hi-lo;if(!(span>0))return[];let step=10**Math.floor(Math.log10(span/count));for(const multiple of[1,2,2.5,5,10])if(span/(step*multiple)<=count){step*=multiple;break}const ticks=[];for(let value=Math.ceil(lo/step)*step;value<=hi+1e-12;value+=step)ticks.push(Math.abs(value)<1e-12?0:value);return ticks}
function buildLegend(){const legend=document.getElementById('legend');legend.replaceChildren();for(const order of ORDERS){const item=document.createElement(teaser?'span':'button'),curve=DATA.curves[order];item.className=visible[order]?'':'off';item.innerHTML=`<i style="color:${curve.color}"></i>order ${order}`;if(!teaser)item.onclick=()=>{visible[order]=!visible[order];redraw()};legend.append(item)}}function hideProbe(){document.getElementById('probe').setAttribute('visibility','hidden');tip.style.display='none'}
function redraw(){const grid=document.getElementById('grid'),axes=document.getElementById('axes'),curves=document.getElementById('curves');grid.replaceChildren();axes.replaceChildren();curves.replaceChildren();for(const x of niceTicks(V.x0,V.x1,7)){const X=px(x);grid.append(el('line',{class:'grid',x1:X,x2:X,y1:M.t,y2:H-M.b}));const tick=el('text',{class:'tick',x:X,y:H-M.b+20,'text-anchor':'middle'});tick.textContent=fmt(x);axes.append(tick)}for(const y of niceTicks(V.y0,V.y1,8)){const Y=py(y);grid.append(el('line',{class:'grid',x1:M.l,x2:W-M.r,y1:Y,y2:Y}));const tick=el('text',{class:'tick',x:M.l-9,y:Y+4,'text-anchor':'end'});tick.textContent=fmt(y);axes.append(tick)}axes.append(el('line',{class:'axis',x1:M.l,x2:W-M.r,y1:H-M.b,y2:H-M.b}),el('line',{class:'axis',x1:M.l,x2:M.l,y1:M.t,y2:H-M.b}));for(const order of ORDERS){const curve=DATA.curves[order],path=el('polyline',{class:'curve',id:`curve-${order}`,stroke:curve.color,'vector-effect':teaser?'non-scaling-stroke':'none',points:curve.points.map(q=>`${px(xv(q)).toFixed(2)},${py(yv(q)).toFixed(2)}`).join(' ')});if(!visible[order])path.style.display='none';curves.append(path)}document.getElementById('ylabel').textContent=panel==='modal'?'modal coordinate |z₁|':'transverse displacement / thickness [%]';document.getElementById('physical').classList.toggle('on',panel==='physical');document.getElementById('modal').classList.toggle('on',panel==='modal');hideProbe();buildLegend()}
function resetView(){Object.assign(V,HOME[panel]);redraw()}function setPanel(next){panel=next;resetView()}document.getElementById('physical').onclick=()=>setPanel('physical');document.getElementById('modal').onclick=()=>setPanel('modal');
function setTool(next){tool=tool===next?null:next;svg.classList.toggle('mode-zoom',tool==='zoom');svg.classList.toggle('mode-pan',tool==='pan');document.getElementById('tool-zoom').classList.toggle('active',tool==='zoom');document.getElementById('tool-pan').classList.toggle('active',tool==='pan')}document.getElementById('tool-zoom').onclick=()=>setTool('zoom');document.getElementById('tool-pan').onclick=()=>setTool('pan');document.getElementById('tool-home').onclick=resetView;
function svgPoint(event){const point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;return point.matrixTransform(svg.getScreenCTM().inverse())}function inPlot(point){return point.x>=M.l&&point.x<=W-M.r&&point.y>=M.t&&point.y<=H-M.b}function mix(hex,amount){const rgb=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)),background=[20,20,28];return`rgb(${rgb.map((value,i)=>Math.round(background[i]+(value-background[i])*amount)).join(',')})`}
function updateProbe(point,event){let best=null,distance=400;for(const order of ORDERS)if(visible[order])for(const q of DATA.curves[order].points){const X=px(xv(q)),Y=py(yv(q));if(X<M.l||X>W-M.r||Y<M.t||Y>H-M.b)continue;const d=(X-point.x)**2+(Y-point.y)**2;if(d<distance){distance=d;best={order,q,X,Y}}}if(!best){hideProbe();return}document.getElementById('probe').setAttribute('visibility','visible');const vx=document.getElementById('cross-x'),vy=document.getElementById('cross-y'),dot=document.getElementById('dot'),color=DATA.curves[best.order].color;vx.setAttribute('x1',best.X);vx.setAttribute('x2',best.X);vx.setAttribute('y1',M.t);vx.setAttribute('y2',H-M.b);vy.setAttribute('x1',M.l);vy.setAttribute('x2',W-M.r);vy.setAttribute('y1',best.Y);vy.setAttribute('y2',best.Y);dot.setAttribute('cx',best.X);dot.setAttribute('cy',best.Y);dot.setAttribute('fill',color);const value=panel==='modal'?`modal coordinate |z₁| ${fmt(best.q.r,2)}`:`transverse displacement / thickness ${fmt(100*best.q.amplitude_ratio,1)}%`;tip.innerHTML=`order ${best.order}<br>frequency change ${fmt(xv(best.q))}%<br>${value}`;tip.style.display='block';tip.style.background=mix(color,.18);tip.style.borderColor=mix(color,.62);const rect=stage.getBoundingClientRect(),left=event.clientX-rect.left,top=event.clientY-rect.top;tip.style.left=Math.min(Math.max(4,left+12),rect.width-tip.offsetWidth-4)+'px';tip.style.top=Math.min(Math.max(4,top-tip.offsetHeight-10),rect.height-tip.offsetHeight-4)+'px'}
function updateZoomBox(point){const x=Math.max(M.l,Math.min(W-M.r,point.x)),y=Math.max(M.t,Math.min(H-M.b,point.y));zoombox.setAttribute('x',Math.min(zoomStart.x,x));zoombox.setAttribute('y',Math.min(zoomStart.y,y));zoombox.setAttribute('width',Math.abs(x-zoomStart.x));zoombox.setAttribute('height',Math.abs(y-zoomStart.y));zoombox.setAttribute('visibility','visible')}
hit.addEventListener('pointerdown',event=>{if(!tool)return;const point=svgPoint(event);if(!inPlot(point))return;event.preventDefault();hit.setPointerCapture(event.pointerId);hideProbe();if(tool==='pan'){pan={x:point.x,y:point.y,x0:V.x0,x1:V.x1,y0:V.y0,y1:V.y1};svg.classList.add('panning')}else zoomStart=point});
hit.addEventListener('pointermove',event=>{const point=svgPoint(event);if(pan){const sx=(pan.x1-pan.x0)/(W-M.l-M.r),sy=(pan.y1-pan.y0)/(H-M.t-M.b);const mx=(pan.x-point.x)*sx,my=(point.y-pan.y)*sy;V.x0=pan.x0+mx;V.x1=pan.x1+mx;V.y0=pan.y0+my;V.y1=pan.y1+my;redraw();return}if(zoomStart){updateZoomBox(point);return}if(inPlot(point))updateProbe(point,event);else hideProbe()});
function finishPointer(event,commit){if(zoomStart&&commit){const x=Number(zoombox.getAttribute('x')),y=Number(zoombox.getAttribute('y')),width=Number(zoombox.getAttribute('width')),height=Number(zoombox.getAttribute('height'));if(width>8&&height>8){const x0=dx(x),x1=dx(x+width),y1=dy(y),y0=dy(y+height);Object.assign(V,{x0,x1,y0,y1})}}zoomStart=null;pan=null;zoombox.setAttribute('visibility','hidden');svg.classList.remove('panning');if(hit.hasPointerCapture(event.pointerId))hit.releasePointerCapture(event.pointerId);redraw()}hit.addEventListener('pointerup',event=>finishPointer(event,true));hit.addEventListener('pointercancel',event=>finishPointer(event,false));hit.addEventListener('pointerleave',()=>{if(!pan&&!zoomStart)hideProbe()});redraw();
</script></body></html>'''


def thumbnail_svg(nodes, faces, mode):
    yaw, pitch = -0.34, 0.42

    def rotate(point):
        x, y, z = point[0] - 500.0, point[1] - 5.0, point[2] - 12.0
        x1 = math.cos(yaw) * x + math.sin(yaw) * z
        z1 = -math.sin(yaw) * x + math.cos(yaw) * z
        return (
            x1,
            math.cos(pitch) * y - math.sin(pitch) * z1,
            math.sin(pitch) * y + math.cos(pitch) * z1,
        )

    reference = {node: rotate(point) for node, point in nodes.items()}
    deformed = {
        node: rotate(tuple(point[i] + mode[node][i] for i in range(3)))
        for node, point in nodes.items()
    }
    all_points = [*reference.values(), *deformed.values()]
    xmin, xmax = min(p[0] for p in all_points), max(p[0] for p in all_points)
    ymin, ymax = min(p[1] for p in all_points), max(p[1] for p in all_points)
    scale = min(352.0 / (xmax - xmin), 177.0 / (ymax - ymin))
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    def project(point):
        return (200.0 + (point[0] - cx) * scale, 112.5 - (point[1] - cy) * scale)

    polygons = []
    for face in faces:
        clamp = all(nodes[node][0] < 1e-7 or nodes[node][0] > 999.999999 for node in face)
        polygons.append((sum(reference[node][2] for node in face) / 4.0, face, "clamp" if clamp else "reference"))
        if not clamp:
            polygons.append((sum(deformed[node][2] for node in face) / 4.0, face, "mode"))
    polygons.sort(key=lambda item: item[0])
    styles = {
        "reference": ("#3f424b", "#797d87", 0.55),
        "clamp": ("#9558b2", "#c096d4", 0.75),
        "mode": ("#4063d8", "#89a6ff", 0.6),
    }
    paths = []
    for _, face, kind in polygons:
        source = deformed if kind == "mode" else reference
        points = [project(source[node]) for node in face]
        command = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points) + " Z"
        fill, stroke, width = styles[kind]
        paths.append(
            f'<path d="{command}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 225" role="img" '
        'aria-labelledby="title desc"><title id="title">Clamped beam and first bending mode</title>'
        '<desc id="desc">A gray clamped beam mesh overlapped by its opaque blue first mode shape.</desc>'
        '<rect width="400" height="225" fill="#0a0a0f"/>' + "".join(paths) + "</svg>"
    )


def write_backbone_html(data, home):
    compact_data = json.dumps(data, separators=(",", ":"))
    backbone_html = (
        BACKBONE_HTML.replace("__DATA__", compact_data)
        .replace("__HOME_PHYSICAL__", f'{home["physical"]:.15g}')
        .replace("__HOME_MODAL__", f'{home["modal"]:.15g}')
    )
    (HERE / "backbone.html").write_text(backbone_html)


def write_assets(nodes, faces, mode, data, home):
    node_json = json.dumps({str(k): list(v) for k, v in sorted(nodes.items())}, separators=(",", ":"))
    mode_json = json.dumps({str(k): list(v) for k, v in sorted(mode.items())}, separators=(",", ":"))
    face_json = json.dumps(faces, separators=(",", ":"))
    (HERE / "beam_mesh.html").write_text(
        MESH_HTML.replace("__NODES__", node_json)
        .replace("__MODE__", mode_json)
        .replace("__FACES__", face_json)
    )
    write_backbone_html(data, home)
    (HERE / "thumb.svg").write_text(thumbnail_svg(nodes, faces, mode))


def main():
    if sys.argv[1:] == ["--backbone-only"]:
        data, home = build_backbone_data()
        write_backbone_html(data, home)
        print(f"wrote {(HERE / 'backbone.html').relative_to(REPO)}")
        return

    nodes, hexes = read_nodes_and_hexes(MESH)
    assert all(abs(a - b) < 1e-8 for a, b in zip(nodes[PROBE_NODE], PROBE_XYZ))
    mode = extract_first_mode()
    for node, point in nodes.items():
        if point[0] < 1e-7 or point[0] > 999.999999:
            assert max(abs(value) for value in mode[node]) < 1e-12
    faces = boundary_faces(hexes)
    assert len(faces) == 326
    data, home = build_backbone_data()
    assert max(
        point["amplitude_ratio"]
        for curve in data["curves"].values()
        for point in curve["points"]
    ) > 1.0
    for curve in data["curves"].values():
        for point in curve["points"]:
            assert math.isclose(
                point["amplitude_ratio"],
                point["amplitude"] / TRANSVERSE_THICKNESS,
                rel_tol=2e-13,
                abs_tol=1e-14,
            )
    write_assets(nodes, faces, mode, data, home)
    for name in ("beam_mesh.html", "backbone.html", "backbone.v1.json", "thumb.svg"):
        path = HERE / name
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
