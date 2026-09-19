# generate_assets.jl — the embedded figures for the "Dual-axis MEMS micromirror" tutorial.
#
#   clamp.html     every boundary surface of the mesh, the clamped ones purple; hovering a
#                  face shows its surface number
#   modes.html     the first modes, animated, the master marked
#   tilt.html      how the tilt of the mirror is computed, on the master mode
#   backbone.html  the backbone of every order: Δω in percent against the mirror tilt in
#                  degrees or the modal amplitude
#   thumb.svg      the mesh seen from the top, for the tutorial card
#
# The mesh, the clamp and the modes come from the example notebook itself: this script runs
# its first five code cells, the ones up to and including the mode table, so what is drawn is
# exactly what the notebook computes. Those cells also write the notebook's ParaView files, as
# running the notebook does. The master mode is read off the notebook's `p = …` line.
#
# The backbone is *not* computed here. It is read from `backbone.v1.csv`, a committed copy of
# the notebook's `results/data/backbone.csv`, the same arrangement the structural and Kármán
# tutorials use: this script renders, the example does the physics. Copy the file again when
# the notebook changes.
#
# Every output is standalone: no external script, stylesheet or font, so it opens from the
# file system and drops into an <iframe>.
#
# Run from the MORFE repository root, in the examples' environment:
#
#     julia --project=../MORFEExamples website/tutorials/assets/mems_micromirror/generate_assets.jl

using Printf

const HERE = @__DIR__
const REPO = normpath(joinpath(HERE, "..", "..", "..", ".."))
const EXAMPLES = first(filter(isdir, [joinpath(REPO, "..", "MORFEExamples"),
    joinpath(REPO, "..", "MORFEExamples", "MORFEExamples")]))
const NOTEBOOK = joinpath(EXAMPLES, "dual_axis_mems_micromirror",
    "dual_axis_mems_micromirror.ipynb")
const BACKBONE_CSV = joinpath(HERE, "backbone.v1.csv")

# ── run the notebook up to its mode table ────────────────────────────────────────────────

# The code cells of the notebook, in order. The JSON is read with a regular expression on
# purpose: the environment carries no JSON package, and nbformat stores each cell's source
# as a list of strings.
function code_cells(path)
    nb = read(path, String)
    cells = String[]
    string = raw"\"(?:[^\"\\]|\\.)*\""
    source = Regex("\"cell_type\": \"code\".*?\"source\": \\[\\s*((?:$string\\s*,?\\s*)*)\\]", "s")
    for m in eachmatch(source, nb)
        lines = [unescape_string(l.captures[1]) for l in eachmatch(r"\"((?:[^\"\\]|\\.)*)\"", m.captures[1])]
        push!(cells, join(lines))
    end
    return cells
end

const CELLS = code_cells(NOTEBOOK)
length(CELLS) == 10 || error("expected 10 code cells in $(basename(NOTEBOOK)), found $(length(CELLS))")
for cell in CELLS[1:5]
    include_string(Main, cell, NOTEBOOK) # `@__DIR__` in a cell is the notebook's folder
end
const MASTER = let m = match(r"^p = (\d+)"m, CELLS[6])
    m === nothing && error("no `p = …` line in the notebook's build_model cell")
    parse(Int, m.captures[1])
end

# ── data ─────────────────────────────────────────────────────────────────────────────────

# Every face of the facet sets `surface_<id>`, as node quadruples, with its surface id.
function boundary_faces(grid)
    faces, ids = NTuple{4, Int}[], Int[]
    for s in surface_ids(grid), facet in Ferrite.getfacetset(grid, "surface_$s")
        push!(faces, Tuple(Ferrite.facets(Ferrite.getcells(grid, facet[1]))[facet[2]]))
        push!(ids, s)
    end
    return faces, ids
end

# The nodes `faces` use, their coordinates as one flat vector, and the faces renumbered into
# that node list from 0, as JavaScript counts.
function compact(faces, points)
    used = sort!(unique(Iterators.flatten(faces)))
    index = Dict(n => i - 1 for (i, n) in enumerate(used))
    xyz = reduce(vcat, [collect(points[n]) for n in used])
    return used, xyz, [index[n] for f in faces for n in f]
end

# Displacement of mode pair `p` at every grid node; constrained degrees of freedom are 0.
function nodal_mode(case, sp, p)
    dh = case.info.dh
    ϕ = mode_shape(sp, p)
    u = zeros(Ferrite.ndofs(dh))
    for (dof, row) in case.info.free_to_local
        u[dof] = ϕ[row]
    end
    return Ferrite.evaluate_at_grid_nodes(dh, u, :u)
end

mode_label(m) = m.tilt_x ≥ 0.3 ? "mirror tilts about x" :
                m.tilt_y ≥ 0.3 ? "mirror tilts about y" : "no mirror tilt"

num(x) = iszero(x) ? "0" : string(Float32(x))
js(v::AbstractVector{<:Integer}) = "[" * join(v, ",") * "]"
js(v::AbstractVector{<:Real}; digits = nothing) =
    "[" * join((digits === nothing ? num(x) : num(round(x; digits)) for x in v), ",") * "]"

# The backbone rows of `backbone.v1.csv`, one curve per order.
function read_backbone(path)
    header, rows... = readlines(path)
    header == "order,r,omega,omega_ratio,tilt_deg" || error("unexpected header in $path: $header")
    table = [parse.(Float64, split(r, ",")) for r in rows]
    return map(sort(unique(Int(t[1]) for t in table))) do N
        t = [row for row in table if Int(row[1]) == N]
        (; N, r = getindex.(t, 2), ratio = getindex.(t, 4), tilt = getindex.(t, 5))
    end
end

# ── writers ──────────────────────────────────────────────────────────────────────────────

function write_assets()
    points = [Ferrite.get_node_coordinate(n) for n in Ferrite.getnodes(grid)]
    faces, ids = boundary_faces(grid)
    xyz_all = reduce(hcat, collect.(points))
    lo, hi = vec(minimum(xyz_all; dims = 2)), vec(maximum(xyz_all; dims = 2))
    center = (lo .+ hi) ./ 2
    radius = hypot(hi[1] - lo[1], hi[2] - lo[2]) / 2
    disp = 0.1 * (hi[1] - lo[1]) # displayed size of a mode's largest displacement
    frame = "\"clamped\":$(js(clamped)),\"center\":$(js(center)),\"radius\":$radius,\"disp\":$disp"

    # clamp.html: all boundary faces, no mode.
    used, xyz, flat = compact(faces, points)
    write(joinpath(HERE, "clamp.html"), mesh_html("Micromirror boundary surfaces",
        "{\"nodes\":$(js(xyz; digits = 1)),\"faces\":$(js(flat)),\"surf\":$(js(ids))," *
        "$frame,\"modes\":[],\"master\":0}"))

    # modes.html: the faces of the top plane only, which is enough for a layer this thin.
    top = maximum(x[3] for x in points)
    keep = [all(abs(points[n][3] - top) < 1e-6 for n in f) for f in faces]
    used, xyz, flat = compact(faces[keep], points)
    mode_json = map(modes) do m
        u = nodal_mode(case, sp, m.p)[used]
        scale = maximum(maximum(abs, v) for v in u)
        q = round.(Int, 1000 .* reduce(vcat, collect.(u)) ./ scale)
        "{\"p\":$(m.p),\"f\":$(num(m.frequency)),\"label\":\"$(mode_label(m))\",\"u\":$(js(q))}"
    end
    write(joinpath(HERE, "modes.html"), mesh_html("Micromirror vibration modes",
        "{\"nodes\":$(js(xyz; digits = 1)),\"faces\":$(js(flat)),\"surf\":$(js(ids[keep]))," *
        "$frame,\"modes\":[$(join(mode_json, ","))],\"master\":$MASTER}"))

    write(joinpath(HERE, "tilt.html"), mesh_html("Micromirror tilt", tilt_json(points, faces)))

    # backbone.html
    curves = read_backbone(BACKBONE_CSV)
    curve_json = map(curves) do c
        "\"$(c.N)\":{\"x\":$(js(100 .* (c.ratio .- 1))),\"tilt\":$(js(c.tilt)),\"r\":$(js(c.r))}"
    end
    write(joinpath(HERE, "backbone.html"),
        replace(BACKBONE_HTML, "__DATA__" => "{\"orders\":$(js([c.N for c in curves]))," *
                                             "\"curves\":{$(join(curve_json, ","))}}"))

    write(joinpath(HERE, "thumb.svg"), thumbnail(points, faces[keep], ids[keep], center, radius))
end

mesh_html(title, data) = replace(MESH_HTML, "__TITLE__" => title, "__DATA__" => data)

# tilt.html: the mirror face in the master mode, drawn exactly as the notebook fits it. Each
# node sits at its undeformed (x, y), lifted by its out-of-plane displacement u_z; the
# notebook's `fit` gives the plane u_z = a + b x + c y through those points, and the slope of
# that plane about the rotation axis (θy = -b or θx = c) is the sine of the tilt. The mode is
# scaled so that the drawn tilt is 15°.
function tilt_json(points, faces)
    u = nodal_mode(case, sp, MASTER)
    a, b, c = fit * [u[n][3] for n in mirror]
    about_y = abs(b) ≥ abs(c)
    slope = about_y ? -b : c
    R = maximum(hypot(x[1], x[2]) for x in points[mirror]) # the fitted disk
    s = sind(15) / abs(slope)
    lifted = [(x[1], x[2], s * u[n][3]) for (n, x) in enumerate(points)]
    plate = [f for f in faces
             if all(abs(points[n][3]) < 1e-6 && hypot(points[n][1], points[n][2]) < 500.001 for n in f)]
    used, xyz, flat = compact(plate, lifted)

    plane(x, y) = s * (a + b * x + c * y)
    circle(r, z) = reduce(vcat, [[r * cos(t), r * sin(t), z(r * cos(t), r * sin(t))]
                                 for t in range(0, 2π; length = 97)])
    # The tilt lives in the plane through the rotation axis' normal: x-z for a tilt about y.
    e = about_y ? (1.0, 0.0) : (0.0, 1.0)
    φ = atan(s * (about_y ? b : c))
    along(r, t) = [r * cos(t) * e[1], r * cos(t) * e[2], plane(0, 0) + r * sin(t)]
    arc = reduce(vcat, [along(1.15R, t) for t in range(0, φ; length = 25)])
    axis = about_y ? [0, -1.4R, plane(0, 0), 0, 1.4R, plane(0, 0)] :
           [-1.4R, 0, plane(0, 0), 1.4R, 0, plane(0, 0)]
    shape(kind, p; kw...) = "{\"kind\":\"$kind\",\"p\":$(js(Float64.(p); digits = 2))" *
        join(",\"$k\":" * (v isa AbstractString ? "\"$v\"" : v isa AbstractVector ? js(v) : string(v))
             for (k, v) in kw) * "}"
    extra = [
        shape("line", circle(R, (x, y) -> 0.0); color = "#8a8d99", dash = [5, 4], width = 1.2),
        shape("poly", circle(1.3R, plane); color = "#9558b2", alpha = 0.22, width = 1.5, lift = 1000),
        shape("dot", reduce(vcat, [[points[n][1], points[n][2], s * u[n][3]] for n in mirror]);
            color = "#e8e8ee", r = 2, lift = 2000),
        shape("line", axis; color = "#389826", dash = [6, 4], width = 1.6, lift = 3000),
        shape("label", axis[4:6]; color = "#389826", text = "rotation axis", lift = 3000),
        shape("line", [along(1.3R, 0.0); along(0.0, 0.0); along(1.3R, φ)]; color = "#cb3c33", width = 1.4, lift = 3000),
        shape("line", arc; color = "#cb3c33", width = 2.2, lift = 3000),
        shape("label", along(1.2R, φ / 2); color = "#cb3c33", text = "θ", lift = 3000),
    ]
    axis_name = about_y ? "y" : "x"
    note = "<b>mode $MASTER</b>, tilt exaggerated · dots: u_z of the mirror face at the nodes of " *
           "the fit · purple: the least-squares plane u_z = a + b x + c y · its slope about " *
           "$axis_name is sin θ"
    keys = "[[\"blue\",\"mirror face\"],[\"#e8e8ee\",\"fitted nodes\"],[\"purple\",\"fitted plane\"]," *
           "[\"#389826\",\"rotation axis\"],[\"#cb3c33\",\"tilt θ\"]]"
    return "{\"kind\":\"tilt\",\"nodes\":$(js(xyz; digits = 2)),\"faces\":$(js(flat))," *
           "\"surf\":$(js(zeros(Int, length(plate)))),\"clamped\":[],\"center\":[0,0,0]," *
           "\"radius\":$(1.45R),\"disp\":0,\"modes\":[],\"master\":$MASTER," *
           "\"home\":{\"yaw\":$(about_y ? -0.3 : -1.27),\"elev\":0.4},\"keys\":$keys," *
           "\"note\":\"$note\",\"extra\":[$(join(extra, ","))]}"
end

# Top view of the undeformed top plane with every element edge: clamped faces purple, free
# ones gray, as in clamp.html. Nothing overlaps in a top view, so one path per colour carries
# every face and the file stays small.
function thumbnail(points, faces, ids, center, radius)
    s = 0.92 * 225 / (2radius / sqrt(2)) # the die's side fills the height
    paths = Dict(true => IOBuffer(), false => IOBuffer())
    for (face, id) in zip(faces, ids)
        io = paths[id in clamped]
        for (k, n) in enumerate(face)
            @printf(io, "%s%.1f,%.1f", k == 1 ? "M" : "L",
                200 + s * (points[n][1] - center[1]), 112.5 - s * (points[n][2] - center[2]))
        end
        print(io, "Z")
    end
    path(io, fill, stroke) = "<path fill=\"$fill\" stroke=\"$stroke\" stroke-width=\"0.25\" " *
                             "stroke-linejoin=\"round\" d=\"$(String(take!(io)))\"/>"
    return "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 400 225\" role=\"img\" " *
           "aria-labelledby=\"title desc\"><title id=\"title\">Micromirror mesh</title>" *
           "<desc id=\"desc\">Top view of the hexahedral mesh of a dual-axis micromirror's " *
           "device layer: a circular mirror in three rings joined by torsion bars; the " *
           "clamped frame is purple.</desc>" *
           "<rect width=\"400\" height=\"225\" fill=\"#0a0a0f\"/>" *
           path(paths[false], "#565963", "#9a9daa") * path(paths[true], "#6f4388", "#b48bcc") *
           "</svg>\n"
end

# ── templates ────────────────────────────────────────────────────────────────────────────

# One rotatable mesh viewer for both mesh figures. Without modes it colours the surfaces
# and names the one under the cursor; with modes it animates the selected one. Zoom and
# pan go through a toolbar, as on the Kármán figures, never through the scroll wheel. Zoom and
# pan go through a toolbar, as on the Kármán figures, never through the scroll wheel. Zoom and
# pan go through a toolbar, as on the Kármán figures, never through the scroll wheel. Zoom and
# pan go through a toolbar, as on the Kármán figures, never through the scroll wheel.
const MESH_HTML = raw"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#0a0a0f;--stage:#07070b;--ink:#e8e8ee;--ink2:#a0a0b0;--ink3:#6e6e7e;--hair:#26262f;--purple:#9558b2;--blue:#4063d8;--gray:#565963}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--ink);overflow:hidden;font:13px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif}
#wrap{display:flex;flex-direction:column;height:100%;padding:10px 12px;gap:8px}
#bar{display:flex;align-items:center;gap:6px 14px;min-height:28px;flex-wrap:wrap}
.key{display:inline-flex;align-items:center;gap:6px;color:var(--ink3);font-size:11px}
.swatch{width:14px;height:8px;border:1px solid currentColor;background:currentColor}
button{font:inherit;font-size:12px;padding:4px 10px;border-radius:5px;cursor:pointer;background:transparent;color:var(--ink2);border:1px solid var(--hair)}
button:hover{border-color:var(--purple);color:var(--ink)}button.on{border-color:var(--purple);color:var(--ink);background:#9558b233}
#modes{display:flex;gap:4px;flex-wrap:wrap}#modes button{padding:3px 0;width:30px}#modes button.master{width:auto;padding:3px 9px;border-color:var(--blue);color:#b9c7ff}
.push{margin-left:auto;display:flex;gap:6px}
#stage{position:relative;flex:1;min-height:0;border:1px solid var(--hair);border-radius:6px;background:var(--stage);overflow:hidden}
canvas{display:block;width:100%;height:100%;cursor:grab;touch-action:none}canvas.drag{cursor:grabbing}
canvas.mode-zoom{cursor:crosshair}canvas.mode-pan.drag{cursor:grabbing}
.tools{position:absolute;top:8px;right:8px;display:flex;gap:6px;z-index:3}
.tool{width:26px;height:26px;padding:0;border-radius:4px;background:rgba(255,255,255,.04);display:grid;place-items:center}
.tool:hover{background:rgba(255,255,255,.09);border-color:var(--hair)}.tool.active{border-color:var(--purple);background:rgba(149,88,178,.12)}
.tool svg{width:18px;height:18px;fill:none;stroke:var(--ink3)}.tool.active svg{stroke:var(--purple)}
#zoombox{position:absolute;display:none;border:1px dashed var(--purple);background:rgba(149,88,178,.10);pointer-events:none;z-index:2}
#note{color:var(--ink3);font-size:11.5px;min-height:1.3em}#note b{color:var(--ink);font-weight:600}
</style></head><body><div id="wrap">
<div id="bar"></div>
<div id="stage"><canvas id="mesh" aria-label="__TITLE__, rotatable"></canvas><div id="zoombox"></div>
<div class="tools"><button class="tool" id="tool-zoom" title="Zoom: drag a rectangle"><svg viewBox="0 0 18 18"><circle cx="7.5" cy="7.5" r="5" stroke-width="1.6"/><line x1="11.2" y1="11.2" x2="16" y2="16" stroke-width="1.8"/><line x1="5.2" y1="7.5" x2="9.8" y2="7.5" stroke-width="1.3"/><line x1="7.5" y1="5.2" x2="7.5" y2="9.8" stroke-width="1.3"/></svg></button><button class="tool" id="tool-pan" title="Pan: drag to move"><svg viewBox="0 0 18 18"><path stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" d="M7.2 10.5 V4.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M9.7 9 V3.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M12.2 9.3 V4.8 a1.25 1.25 0 0 1 2.5 0 V11.5 c0 3.4 -2 5.4 -4.9 5.4 c-2.3 0 -3.4 -.9 -4.5 -2.6 L3.4 11.1 a1.3 1.3 0 0 1 2.2 -1.3 l1.6 2.2"/></svg></button><button class="tool" id="tool-home" title="Reset view"><svg viewBox="0 0 18 18"><path stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" d="M2.2 8.2 L9 2.6 L15.8 8.2 M4.2 7.5 V15 H13.8 V7.5"/></svg></button></div></div>
<div id="note"></div>
</div><script>
const D=__DATA__;
const cv=document.getElementById('mesh'),ctx=cv.getContext('2d'),bar=document.getElementById('bar'),note=document.getElementById('note'),zoombox=document.getElementById('zoombox');
const N=D.nodes.length/3,F=D.faces.length/4,CLAMP=new Set(D.clamped),MODES=D.modes.length>0,KIND=D.kind||(MODES?'modes':'clamp');
const RGB={gray:[86,89,99],purple:[149,88,178],blue:[64,99,216]},HOME=Object.assign({yaw:-0.55,elev:0.8,zoom:1,px:0,py:0},D.home||{});
let view={...HOME},mode=D.master,anim=MODES,t0=performance.now(),hover=0,items=[],pending=false,drag=null,tool=null;
const help='Drag to rotate; arm the zoom or pan tool to zoom or pan.'+(KIND==='clamp'?' Hover a face to read its surface number.':'');
function button(text,title,onclick){const b=document.createElement('button');b.textContent=text;b.title=title;b.onclick=onclick;return b}
function key(color,text){const k=document.createElement('span');k.className='key';k.innerHTML='<i class="swatch" style="color:'+(RGB[color]?'rgb('+RGB[color]+')':color)+'"></i>'+text;return k}
function setMode(p){mode=p;for(const b of document.querySelectorAll('#modes button'))b.classList.toggle('on',+b.dataset.p===p);describe();request()}
function describe(){if(KIND==='tilt'){note.innerHTML=D.note+' · '+help;return}
  if(!MODES){note.innerHTML=hover?'<b>surface '+hover+'</b> · '+(CLAMP.has(hover)?'clamped':'free')+' · '+help:help;return}
  const m=D.modes.find(q=>q.p===mode);note.innerHTML='<b>mode '+m.p+'</b> · frequency '+m.f.toFixed(3)+' · '+m.label+(m.p===D.master?' · <b>master mode</b>':'')+' · '+help}
function setup(){if(MODES){const box=document.createElement('span');box.id='modes';for(const m of D.modes){const b=button(m.p===D.master?m.p+' · master mode':String(m.p),'mode '+m.p+': '+m.label,()=>setMode(m.p));b.dataset.p=m.p;if(m.p===D.master)b.classList.add('master');box.append(b)}
  bar.append(box,key('blue','mode'),key('purple','clamped'));
  const push=document.createElement('span');push.className='push';
  const a=button('animate','oscillate the mode, or hold it at its largest displacement',()=>{anim=!anim;a.classList.toggle('on',anim);t0=performance.now();request()});a.classList.toggle('on',anim);
  push.append(a);bar.append(push);setMode(mode)}
  else if(KIND==='tilt'){for(const[c,t]of D.keys)bar.append(key(c,t));describe()}
  else{bar.append(key('purple','clamped: surface'+(D.clamped.length>1?'s ':' ')+D.clamped.join(', ')),key('gray','free'));describe()}}
function setTool(t){tool=tool===t?null:t;cv.classList.toggle('mode-zoom',tool==='zoom');cv.classList.toggle('mode-pan',tool==='pan');
  document.getElementById('tool-zoom').classList.toggle('active',tool==='zoom');document.getElementById('tool-pan').classList.toggle('active',tool==='pan')}
document.getElementById('tool-zoom').onclick=()=>setTool('zoom');document.getElementById('tool-pan').onclick=()=>setTool('pan');
document.getElementById('tool-home').onclick=()=>{view={...HOME};request()};
function draw(){pending=false;const w=cv.clientWidth,h=cv.clientHeight;if(!w)return;ctx.clearRect(0,0,w,h);
  const cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),ce=Math.cos(view.elev),se=Math.sin(view.elev),s=view.zoom*0.46*Math.min(w,h)/D.radius,ox=w/2+view.px,oy=h/2+view.py,C=D.center;
  const m=MODES?D.modes.find(q=>q.p===mode):null,amp=m?D.disp/1000*(anim?Math.cos((performance.now()-t0)/350):1):0;
  const ref=new Float64Array(3*N),def=m?new Float64Array(3*N):null;
  function put(out,i,x,y,z){x-=C[0];y-=C[1];z-=C[2];const x1=cy*x-sy*y,y1=sy*x+cy*y;out[3*i]=ox+s*x1;out[3*i+1]=oy-s*(se*y1+ce*z);out[3*i+2]=s*(se*z-ce*y1)}
  for(let i=0;i<N;i++){const x=D.nodes[3*i],y=D.nodes[3*i+1],z=D.nodes[3*i+2];put(ref,i,x,y,z);if(m)put(def,i,x+amp*m.u[3*i],y+amp*m.u[3*i+1],z+amp*m.u[3*i+2])}
  items=[];
  for(let f=0;f<F;f++){const id=D.surf[f],clamp=CLAMP.has(id);
    items.push(m&&!clamp?{f,P:def,color:'blue',id}:{f,P:ref,color:KIND==='tilt'?'blue':clamp?'purple':'gray',id})}
  // Overlays (tilt figure): polygons, lines, dots and labels, depth-sorted with the faces.
  for(const e of D.extra||[]){const n=e.p.length/3,P=new Float64Array(3*n);for(let i=0;i<n;i++)put(P,i,e.p[3*i],e.p[3*i+1],e.p[3*i+2]);items.push({e,P,n})}
  for(const it of items){let d=0;if(it.e){for(let k=0;k<it.n;k++)d+=it.P[3*k+2];it.depth=d/it.n+(it.e.lift||0)*s;continue}
    for(let k=0;k<4;k++)d+=it.P[3*D.faces[4*it.f+k]+2];it.depth=d/4}
  items.sort((a,b)=>a.depth-b.depth);
  for(const it of items){if(it.e){overlay(it);continue}
    const q=[0,1,2,3].map(k=>3*D.faces[4*it.f+k]),P=it.P;
    const ax=P[q[2]]-P[q[0]],ay=P[q[0]+1]-P[q[2]+1],az=P[q[2]+2]-P[q[0]+2],bx=P[q[3]]-P[q[1]],by=P[q[1]+1]-P[q[3]+1],bz=P[q[3]+2]-P[q[1]+2];
    const nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx,shade=0.35+0.65*Math.abs(nz)/(Math.hypot(nx,ny,nz)||1);
    const lit=KIND==='clamp'&&hover&&it.id===hover?0.35:0,c=RGB[it.color].map(v=>Math.round(Math.min(255,v*shade+255*lit)));
    ctx.beginPath();ctx.moveTo(P[q[0]],P[q[0]+1]);for(let k=1;k<4;k++)ctx.lineTo(P[q[k]],P[q[k]+1]);ctx.closePath();
    ctx.fillStyle='rgb('+c+')';ctx.fill();ctx.strokeStyle='rgba(255,255,255,'+(m?0.07:0.12)+')';ctx.lineWidth=0.5;ctx.stroke()}
  triad(cy,sy,ce,se);if(anim)request()}
function overlay(it){const e=it.e,P=it.P;ctx.save();ctx.globalAlpha=e.alpha??1;ctx.strokeStyle=ctx.fillStyle=e.color;ctx.lineWidth=e.width||1.5;ctx.setLineDash(e.dash||[]);
  if(e.kind==='dot'){for(let k=0;k<it.n;k++){ctx.beginPath();ctx.arc(P[3*k],P[3*k+1],e.r||2,0,7);ctx.fill()}}
  else if(e.kind==='label'){ctx.globalAlpha=1;ctx.font='600 13px ui-monospace,Menlo,monospace';ctx.fillText(e.text,P[0]+4,P[1]-4)}
  else{ctx.beginPath();ctx.moveTo(P[0],P[1]);for(let k=1;k<it.n;k++)ctx.lineTo(P[3*k],P[3*k+1]);if(e.kind==='poly'){ctx.closePath();ctx.fill();ctx.globalAlpha=Math.min(1,2*(e.alpha??1))}ctx.stroke()}
  ctx.restore()}
function triad(cy,sy,ce,se){const L=22,o=[L+14,cv.clientHeight-L-14];ctx.font='600 11px ui-monospace,Menlo,monospace';
  for(const[v,col,t]of[[[cy,se*sy],'#cb3c33','x'],[[-sy,se*cy],'#389826','y'],[[0,ce],'#4063d8','z']]){const e=[o[0]+L*v[0],o[1]-L*v[1]];
    ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(e[0],e[1]);ctx.stroke();ctx.fillText(t,e[0]+(v[0]>=0?3:-10),e[1]+(v[1]>=0?-3:10))}}
function request(){if(!pending){pending=true;requestAnimationFrame(draw)}}
function fit(){const d=devicePixelRatio||1;cv.width=Math.round(cv.clientWidth*d);cv.height=Math.round(cv.clientHeight*d);ctx.setTransform(d,0,0,d,0,0);draw()}
function inside(x,y,it){let c=false;for(let k=0,j=3;k<4;j=k++){const a=3*D.faces[4*it.f+k],b=3*D.faces[4*it.f+j],P=it.P;
  if((P[a+1]>y)!==(P[b+1]>y)&&x<(P[b]-P[a])*(y-P[a+1])/(P[b+1]-P[a+1])+P[a])c=!c}return c}
function local(e){const r=cv.getBoundingClientRect();return[e.clientX-r.left,e.clientY-r.top]}
function pick(e){const[x,y]=local(e);for(let i=items.length-1;i>=0;i--)if(inside(x,y,items[i]))return items[i].id;return 0}
// Dragging rotates; an armed tool turns it into a zoom rectangle or a pan. The wheel is left
// to the page, so scrolling past an embedded figure never zooms it.
cv.addEventListener('pointerdown',e=>{const[x,y]=local(e);drag={x0:x,y0:y,x,y};cv.setPointerCapture(e.pointerId);cv.classList.add('drag')});
cv.addEventListener('pointermove',e=>{if(!drag){if(KIND==='clamp'){const id=pick(e);if(id!==hover){hover=id;describe();request()}}return}
  const[x,y]=local(e),dx=x-drag.x,dy=y-drag.y;drag.x=x;drag.y=y;
  if(tool==='zoom'){Object.assign(zoombox.style,{display:'block',left:Math.min(x,drag.x0)+'px',top:Math.min(y,drag.y0)+'px',width:Math.abs(x-drag.x0)+'px',height:Math.abs(y-drag.y0)+'px'});return}
  if(tool==='pan'){view.px+=dx;view.py+=dy}else{view.yaw+=dx*0.008;view.elev=Math.max(-1.55,Math.min(1.55,view.elev+dy*0.008))}request()});
cv.addEventListener('pointerup',()=>{if(drag&&tool==='zoom'){const rw=Math.abs(drag.x-drag.x0),rh=Math.abs(drag.y-drag.y0);
    if(rw>8&&rh>8){const w=cv.clientWidth,h=cv.clientHeight,g=Math.min(w/rw,h/rh,60/view.zoom),cx=(drag.x+drag.x0)/2,cy=(drag.y+drag.y0)/2;
      view.px=g*(view.px-(cx-w/2));view.py=g*(view.py-(cy-h/2));view.zoom*=g;request()}}
  zoombox.style.display='none';drag=null;cv.classList.remove('drag')});
cv.addEventListener('pointerleave',()=>{if(KIND==='clamp'&&hover&&!drag){hover=0;describe();request()}});
addEventListener('resize',fit);setup();fit();
</script></body></html>
"""

const BACKBONE_HTML = raw"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Micromirror backbone</title>
<style>
:root{--bg:#0a0a0f;--stage:#07070b;--ink:#e8e8ee;--ink2:#a0a0b0;--ink3:#6e6e7e;--hair:#26262f;--purple:#9558b2}
*{box-sizing:border-box}html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--ink);overflow:hidden;font:13px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif}
#wrap{display:flex;flex-direction:column;height:100%;padding:10px 12px;gap:8px}
#bar{display:flex;gap:6px;align-items:center;min-height:28px;flex-wrap:wrap}
button{font:inherit;font-size:12px;padding:4px 10px;border-radius:5px;cursor:pointer;background:transparent;color:var(--ink2);border:1px solid var(--hair)}
button:hover{border-color:var(--purple);color:var(--ink)}button.on{border-color:var(--purple);color:var(--ink);background:rgba(149,88,178,.16)}
#meta{margin-left:auto;color:var(--ink3);font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.04em}
#stage{position:relative;flex:1;min-height:0;border:1px solid var(--hair);border-radius:6px;background:var(--stage);overflow:hidden}
svg{display:block;width:100%;height:100%;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.axis{stroke:var(--ink3);stroke-width:1}.grid{stroke:var(--hair);stroke-width:.8}
.tick,.label{fill:var(--ink3);font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.label{fill:var(--ink2);font-size:12px}
.curve{fill:none}.cross{stroke:var(--ink3);stroke-width:1;stroke-dasharray:3 3;opacity:.7}
.tool{cursor:pointer}.tool .btn{fill:rgba(255,255,255,.04);stroke:var(--hair)}.tool .icn{stroke:var(--ink3)}
.tool:hover .btn{fill:rgba(255,255,255,.09)}.tool.active .btn{stroke:var(--purple);fill:rgba(149,88,178,.12)}.tool.active .icn{stroke:var(--purple)}
svg.mode-zoom #hit{cursor:crosshair}svg.mode-pan #hit{cursor:grab}svg.mode-pan.panning #hit{cursor:grabbing}
#legend{display:flex;gap:14px;align-items:center;flex-wrap:wrap}#legend button{border:0;padding:0;color:var(--ink3);font-size:11px}
#legend button.off{opacity:.3}#legend i{display:inline-block;width:18px;height:0;vertical-align:3px;border-top:2.5px solid currentColor;margin-right:6px}
#note{color:var(--ink3);font-size:11.5px;min-height:1.3em}
.tip{position:absolute;pointer-events:none;display:none;background:#14141c;border:1px solid var(--hair);border-radius:5px;padding:6px 9px;font-size:11.5px;
  color:var(--ink);white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;z-index:5}
</style></head><body><div id="wrap">
<div id="bar"><button id="panel-tilt">mirror tilt [°]</button><button id="panel-modal">modal amplitude ρ</button><span id="meta">one ROM · lower orders are nested truncations</span></div>
<div id="stage"><svg id="chart" viewBox="0 0 900 470" role="img" aria-label="Backbone curves of the micromirror ROM, one per expansion order">
<defs><clipPath id="plot-clip"><rect x="78" y="22" width="798" height="390"/></clipPath></defs>
<g id="grid"></g><g id="axes"></g>
<g clip-path="url(#plot-clip)"><g id="curves"></g>
<g id="probe" visibility="hidden"><line id="cross-x" class="cross"/><line id="cross-y" class="cross"/><circle id="dot" r="4.5" stroke="#07070b" stroke-width="1.5"/></g>
<rect id="zoombox" fill="rgba(149,88,178,.10)" stroke="#9558b2" stroke-dasharray="4 3" visibility="hidden"/></g>
<text class="label" x="477" y="459" text-anchor="middle">Δω [%]</text>
<text id="ylabel" class="label" transform="translate(18 217) rotate(-90)" text-anchor="middle"></text>
<rect id="hit" x="78" y="22" width="798" height="390" fill="transparent"/>
<g id="toolbar"><g class="tool" id="tool-zoom" transform="translate(784 14)"><title>Zoom: drag a rectangle</title><rect class="btn" width="26" height="26" rx="4"/><g class="icn" fill="none" transform="translate(4 4)"><circle cx="7.5" cy="7.5" r="5" stroke-width="1.6"/><line x1="11.2" y1="11.2" x2="16" y2="16" stroke-width="1.8"/><line x1="5.2" y1="7.5" x2="9.8" y2="7.5" stroke-width="1.3"/><line x1="7.5" y1="5.2" x2="7.5" y2="9.8" stroke-width="1.3"/></g></g><g class="tool" id="tool-pan" transform="translate(816 14)"><title>Pan: drag to move</title><rect class="btn" width="26" height="26" rx="4"/><g class="icn" fill="none" transform="translate(4 4)"><path stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" d="M7.2 10.5 V4.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M9.7 9 V3.6 a1.25 1.25 0 0 1 2.5 0 V9.3 M12.2 9.3 V4.8 a1.25 1.25 0 0 1 2.5 0 V11.5 c0 3.4 -2 5.4 -4.9 5.4 c-2.3 0 -3.4 -.9 -4.5 -2.6 L3.4 11.1 a1.3 1.3 0 0 1 2.2 -1.3 l1.6 2.2"/></g></g><g class="tool" id="tool-home" transform="translate(848 14)"><title>Reset view</title><rect class="btn" width="26" height="26" rx="4"/><g class="icn" fill="none" transform="translate(4 4)"><path stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" d="M2.2 8.2 L9 2.6 L15.8 8.2 M4.2 7.5 V15 H13.8 V7.5"/></g></g></g>
</svg><div id="tip" class="tip"></div></div>
<div id="legend"></div><div id="note"></div></div>
<script>
const DATA=__DATA__,COLORS={3:'#4063d8',5:'#389826',7:'#cb3c33',9:'#9558b2'},NS='http://www.w3.org/2000/svg';
const ML=78,MR=24,MT=22,MB=58,W=900,H=470,ORDERS=DATA.orders,TOP=Math.max(...ORDERS),visible={},V={};
const PANELS={tilt:{key:'tilt',ylabel:'mirror tilt [°]'},modal:{key:'r',ylabel:'modal amplitude ρ'}};
const HELP='Hover a curve to read its values; arm the zoom or pan tool to change the view; click an order to hide it.';
let panel='tilt',tool=null,pan=null,zoomStart=null;for(const o of ORDERS)visible[o]=true;
const svg=document.getElementById('chart'),stage=document.getElementById('stage'),hit=document.getElementById('hit'),tip=document.getElementById('tip'),zoombox=document.getElementById('zoombox');
const color=o=>COLORS[o]||'#a0a0b0';
function el(name,attrs){const n=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n}
// Blend a curve colour toward the tooltip background, so a readout is tinted by the curve it
// belongs to: `amount` near 0 is almost the background, near 1 the colour itself.
function mix(hex,amount){const bg=[20,20,28],rgb=[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16));return'rgb('+rgb.map((v,i)=>Math.round(bg[i]+(v-bg[i])*amount)).join(',')+')'}
const px=x=>ML+(x-V.x0)/(V.x1-V.x0)*(W-ML-MR),py=y=>H-MB-(y-V.y0)/(V.y1-V.y0)*(H-MT-MB);
const dx=p=>V.x0+(p-ML)/(W-ML-MR)*(V.x1-V.x0),dy=p=>V.y0+(H-MB-p)/(H-MT-MB)*(V.y1-V.y0);
function niceTicks(lo,hi,n){const span=hi-lo;if(!(span>0))return[];let step=Math.pow(10,Math.floor(Math.log10(span/n)));
  [1,2,2.5,5,10].some(m=>{if(span/(step*m)<=n){step*=m;return true}});const t=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-12*Math.abs(hi||1);v+=step)t.push(Math.abs(v)<1e-15?0:v);return t}
const fmt=v=>Math.abs(v)<1e-15?'0':parseFloat(v.toPrecision(6)).toString();
function ys(o){return DATA.curves[o][PANELS[panel].key]}
function resetView(){let x=0,y=0;for(const o of ORDERS)if(visible[o]){x=Math.max(x,...DATA.curves[o].x);y=Math.max(y,...ys(o))}
  Object.assign(V,{x0:0,x1:(x||1)*1.04,y0:0,y1:(y||1)*1.12});redraw()}
function legend(){const box=document.getElementById('legend');box.replaceChildren();for(const o of ORDERS){const b=document.createElement('button');
  b.className=visible[o]?'':'off';b.innerHTML='<i style="color:'+color(o)+(o===TOP?'':';border-top-style:dashed')+'"></i>order '+o;
  b.onclick=()=>{visible[o]=!visible[o];redraw()};box.append(b)}}
function hideProbe(){document.getElementById('probe').setAttribute('visibility','hidden');tip.style.display='none'}
function redraw(){const grid=document.getElementById('grid'),axes=document.getElementById('axes'),curves=document.getElementById('curves');
  grid.replaceChildren();axes.replaceChildren();curves.replaceChildren();
  for(const t of niceTicks(V.x0,V.x1,7)){const X=px(t);if(X<ML-.5||X>W-MR+.5)continue;grid.append(el('line',{class:'grid',x1:X,x2:X,y1:MT,y2:H-MB}));
    const s=el('text',{class:'tick',x:X,y:H-MB+18,'text-anchor':'middle'});s.textContent=fmt(t);axes.append(s)}
  for(const t of niceTicks(V.y0,V.y1,6)){const Y=py(t);if(Y<MT-.5||Y>H-MB+.5)continue;grid.append(el('line',{class:'grid',x1:ML,x2:W-MR,y1:Y,y2:Y}));
    const s=el('text',{class:'tick',x:ML-9,y:Y+4,'text-anchor':'end'});s.textContent=fmt(t);axes.append(s)}
  axes.append(el('line',{class:'axis',x1:ML,x2:W-MR,y1:H-MB,y2:H-MB}),el('line',{class:'axis',x1:ML,x2:ML,y1:MT,y2:H-MB}));
  for(const o of ORDERS){if(!visible[o])continue;const c=DATA.curves[o],y=ys(o);
    curves.append(el('polyline',{class:'curve',stroke:color(o),'stroke-width':o===TOP?2.2:1.8,'stroke-dasharray':o===TOP?'none':'7 5',
      points:c.x.map((x,i)=>px(x).toFixed(1)+','+py(y[i]).toFixed(1)).join(' ')}))}
  document.getElementById('ylabel').textContent=PANELS[panel].ylabel;
  for(const k in PANELS)document.getElementById('panel-'+k).classList.toggle('on',k===panel);
  document.getElementById('note').textContent=HELP;hideProbe();legend()}
function setPanel(k){panel=k;resetView()}
for(const k in PANELS)document.getElementById('panel-'+k).onclick=()=>setPanel(k);
function svgPoint(e){const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;return p.matrixTransform(svg.getScreenCTM().inverse())}
function updateProbe(p,e){let best=null,dist=400;
  for(const o of ORDERS){if(!visible[o])continue;const c=DATA.curves[o],y=ys(o);
    for(let i=0;i<c.x.length;i++){const X=px(c.x[i]),Y=py(y[i]);if(X<ML||X>W-MR||Y<MT||Y>H-MB)continue;const d=(X-p.x)**2+(Y-p.y)**2;if(d<dist){dist=d;best={o,i,X,Y}}}}
  if(!best){hideProbe();return}
  const c=DATA.curves[best.o],col=color(best.o);
  document.getElementById('probe').setAttribute('visibility','visible');
  const cx=document.getElementById('cross-x'),cy=document.getElementById('cross-y'),dot=document.getElementById('dot');
  cx.setAttribute('x1',best.X);cx.setAttribute('x2',best.X);cx.setAttribute('y1',MT);cx.setAttribute('y2',H-MB);
  cy.setAttribute('x1',ML);cy.setAttribute('x2',W-MR);cy.setAttribute('y1',best.Y);cy.setAttribute('y2',best.Y);
  dot.setAttribute('cx',best.X);dot.setAttribute('cy',best.Y);dot.setAttribute('fill',col);
  tip.innerHTML='order = '+best.o+'<br>Δω = '+c.x[best.i].toPrecision(4)+' %<br>mirror tilt = '+c.tilt[best.i].toPrecision(4)+'°<br>ρ = '+c.r[best.i].toPrecision(4);
  tip.style.display='block';tip.style.background=mix(col,.18);tip.style.borderColor=mix(col,.62);
  const r=stage.getBoundingClientRect(),left=e.clientX-r.left,top=e.clientY-r.top;
  tip.style.left=Math.min(Math.max(4,left+12),r.width-tip.offsetWidth-4)+'px';
  tip.style.top=Math.min(Math.max(4,top-tip.offsetHeight-10),r.height-tip.offsetHeight-4)+'px'}
function setTool(t){tool=tool===t?null:t;svg.classList.toggle('mode-zoom',tool==='zoom');svg.classList.toggle('mode-pan',tool==='pan');
  document.getElementById('tool-zoom').classList.toggle('active',tool==='zoom');document.getElementById('tool-pan').classList.toggle('active',tool==='pan')}
document.getElementById('tool-zoom').addEventListener('click',()=>setTool('zoom'));
document.getElementById('tool-pan').addEventListener('click',()=>setTool('pan'));
document.getElementById('tool-home').addEventListener('click',resetView);
hit.addEventListener('mousedown',e=>{if(tool===null)return;const p=svgPoint(e);hideProbe();e.preventDefault();
  if(tool==='pan'){pan={x:p.x,y:p.y,...V};svg.classList.add('panning')}else zoomStart=p});
addEventListener('mouseup',()=>{if(zoomStart){const x=+zoombox.getAttribute('x'),w=+zoombox.getAttribute('width'),y=+zoombox.getAttribute('y'),h=+zoombox.getAttribute('height');
    zoombox.setAttribute('visibility','hidden');if(zoombox.getAttribute('data-live')==='1'&&w>8&&h>8){Object.assign(V,{x0:dx(x),x1:dx(x+w),y1:dy(y),y0:dy(y+h)});redraw()}
    zoombox.setAttribute('data-live','0');zoomStart=null}
  pan=null;svg.classList.remove('panning')});
hit.addEventListener('mousemove',e=>{const p=svgPoint(e);
  if(pan){const sx=(pan.x1-pan.x0)/(W-ML-MR),sy=(pan.y1-pan.y0)/(H-MT-MB),mx=(pan.x-p.x)*sx,my=(p.y-pan.y)*sy;
    Object.assign(V,{x0:pan.x0+mx,x1:pan.x1+mx,y0:pan.y0+my,y1:pan.y1+my});redraw();return}
  if(zoomStart){const cx=Math.max(ML,Math.min(W-MR,p.x)),cy=Math.max(MT,Math.min(H-MB,p.y));
    zoombox.setAttribute('x',Math.min(zoomStart.x,cx));zoombox.setAttribute('y',Math.min(zoomStart.y,cy));
    zoombox.setAttribute('width',Math.abs(cx-zoomStart.x));zoombox.setAttribute('height',Math.abs(cy-zoomStart.y));
    zoombox.setAttribute('visibility','visible');zoombox.setAttribute('data-live','1');return}
  updateProbe(p,e)});
hit.addEventListener('mouseleave',hideProbe);
resetView();
</script></body></html>
"""

write_assets()
println("wrote clamp.html, modes.html, tilt.html, backbone.html and thumb.svg to $HERE")
