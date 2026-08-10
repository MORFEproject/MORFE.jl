# viz.jl — self-contained lattice viewer for MultiindexSets.
#
# `write_lattice` serialises one or more multiindex sets into a single standalone HTML
# file: no external stylesheet, script, font or CDN, so the page works from the file
# system and can be embedded in the website with a plain <iframe>.
#
# Two renderers, chosen by the number of displayed coordinates:
#   2 coordinates → an SVG integer grid, one marker per exponent, hover tooltip
#   3 coordinates → a rotatable point lattice, orthographic projection, drag to orbit
# Sets in more variables expose selectors that pick which coordinates to display;
# exponents that collapse onto the same displayed point are aggregated.

using MORFE.Multiindices: MultiindexSet

"""
	LatticePanel(name, set; marks, values, value_label, note)

One switchable panel of a lattice figure.

- `name`: label on the panel's button.
- `set`: the `MultiindexSet` to draw.
- `marks`: optional `α → String` map tagging each exponent; recognised tags are
  `"kept"` (default), `"dropped"` and `"resonant"`.  Exponents tagged `"dropped"` are
  drawn hollow, so a panel can show a set together with what was removed from it.
- `values`: optional `α → Real` map; drives the colour ramp and appears in the tooltip.
- `value_label`: what `values` means, e.g. `"|s|"`.
- `note`: one line of prose under the panel.
"""
struct LatticePanel
    name::String
    set::MultiindexSet
    marks::Dict{Vector{Int}, String}
    values::Dict{Vector{Int}, Float64}
    value_label::String
    note::String
end

function LatticePanel(name, set::MultiindexSet;
        marks = Dict{Vector{Int}, String}(),
        values = Dict{Vector{Int}, Float64}(),
        value_label::String = "",
        note::String = "")
    LatticePanel(String(name), set,
        Dict{Vector{Int}, String}(Vector{Int}(k) => String(v) for (k, v) in marks),
        Dict{Vector{Int}, Float64}(Vector{Int}(k) => Float64(v) for (k, v) in values),
        value_label, note)
end

"""
	LatticeConditions(universe; filters, unions, note)

A figure whose buttons are *composable conditions* on one lattice, rather than a set of
independent panels.

- `universe`: the lattice every condition acts on.  Nothing outside it is ever drawn.
- `filters`: `name => predicate` pairs. Active filters **intersect**: a monomial is a
  member when it satisfies all of them. With none active the whole universe is a member.
- `unions`: `name => predicate` pairs. Active unions **add** their monomials back on top
  of whatever the filters left.
- `note`: one line of prose under the figure.

Predicates run here, in Julia; only the resulting per-monomial masks reach the page, so
the browser just intersects and unions bit vectors.
"""
struct LatticeConditions
    universe::MultiindexSet
    filters::Vector{Pair{String, Vector{Bool}}}
    unions::Vector{Pair{String, Vector{Bool}}}
    note::String
end

function LatticeConditions(universe::MultiindexSet;
        filters = [], unions = [], note::String = "")
    mask(pred) = Bool[pred(α) for α in universe.exponents]
    to_pairs(ps) = Pair{String, Vector{Bool}}[String(n) => mask(p) for (n, p) in ps]
    LatticeConditions(universe, to_pairs(filters), to_pairs(unions), note)
end

# ---------------------------------------------------------------------------------
# Serialisation to a JS literal.  Written by hand rather than through a JSON package
# so the demo keeps running on the bare root environment.

_js_num(x::Real) = isfinite(x) ? string(round(Float64(x), digits = 6)) : "null"
function _js_str(s::AbstractString)
    '"' * replace(String(s), '\\' => "\\\\", '"' => "\\\"",
        '\n' => "\\n") * '"'
end

function _panel_js(p::LatticePanel)
    exps = p.set.exponents
    nvar = isempty(exps) ? 0 : length(first(exps))
    pts = String[]
    for v in exps
        key = Vector{Int}(v)
        mark = get(p.marks, key, "kept")
        val = get(p.values, key, NaN)
        # No index is emitted: filtering renumbers the members, so a serialised
        # position would be wrong as soon as a condition is toggled.
        push!(pts,
            string("{a:[", join(key, ","),
                "],d:", sum(key), ",m:", _js_str(mark), ",v:", _js_num(val), "}"))
    end
    # `dropped` markers are shown but are not members of the set, so they must not
    # count towards the reported cardinality.
    n_members = count(v -> get(p.marks, Vector{Int}(v), "kept") != "dropped", exps)
    string("{name:", _js_str(p.name), ",nvar:", nvar, ",count:", n_members,
        ",valueLabel:", _js_str(p.value_label), ",note:", _js_str(p.note),
        ",pts:[", join(pts, ","), "]}")
end

"""
	write_lattice(path, panels; title, caption)

Write a standalone HTML lattice viewer for `panels` (a vector of [`LatticePanel`](@ref))
to `path`, creating parent directories as needed.  Returns `path`.
"""
function write_lattice(path::AbstractString, panels::AbstractVector{LatticePanel};
        title::AbstractString = "Multiindex lattice",
        caption::AbstractString = "")
    mkpath(dirname(path))
    data = "[" * join((_panel_js(p) for p in panels), ",") * "]"
    write(path, _lattice_html(title, caption, data))
    return path
end

function write_lattice(path, panel::LatticePanel; kwargs...)
    write_lattice(path, [panel]; kwargs...)
end

function _conditions_js(c::LatticeConditions)
    exps = c.universe.exponents
    nvar = isempty(exps) ? 0 : length(first(exps))
    pts = [string("{a:[", join(Vector{Int}(v), ","), "],d:", sum(v),
               ",m:\"kept\",v:null}") for v in exps]
    conds(ps) = "[" *
                join(
                    (string("{name:", _js_str(n), ",mask:[",
                         join((b ? "1" : "0" for b in m), ","), "]}")
                    for (n, m) in ps),
                    ",") * "]"
    string("{name:\"lattice\",nvar:", nvar, ",count:", length(exps),
        ",valueLabel:\"\",note:", _js_str(c.note),
        ",filters:", conds(c.filters), ",unions:", conds(c.unions),
        ",pts:[", join(pts, ","), "]}")
end

"""
	LatticeSpectrum(set, eigenvalues, radius; note)

A split figure: the lattice on the left, the superharmonics `s(α) = ⟨λ, α⟩` in the
complex plane on the right.

Monomials inside the band `|s(α)| < radius` are members; the rest stay visible as hollow
markers. Both panels colour by `|s(α)|`, and hovering a monomial on the left draws the
head-to-tail sum `α₁·λ₁ + α₂·λ₂ + …` that reaches its superharmonic.
"""
struct LatticeSpectrum
    set::MultiindexSet
    eigenvalues::Vector{ComplexF64}
    radius::Float64
    note::String
end

function LatticeSpectrum(set::MultiindexSet, eigenvalues, radius::Real; note::String = "")
    λ = ComplexF64[eigenvalues...]
    nvar = isempty(set.exponents) ? 0 : length(first(set.exponents))
    length(λ) == nvar || throw(ArgumentError(
        "got $(length(λ)) eigenvalues for a $nvar-variable set"))
    LatticeSpectrum(set, λ, Float64(radius), note)
end

# Unit exponent eᵢ shaped like `α`, so `α - eᵢ` stays an SVector.
function _unit_svector(α::SVector{N, Int}, i::Int) where {N}
    SVector{N, Int}(ntuple(j -> j == i ? 1 : 0, N))
end

function _spectrum_js(sp::LatticeSpectrum)
    exps = sp.set.exponents
    nvar = isempty(exps) ? 0 : length(first(exps))
    s(α) = sum(sp.eigenvalues .* α)
    inband(α) = abs(s(α)) < sp.radius
    # Outside the band, yet a factor of something inside it: exactly the monomials the
    # downward closure has to add back before the set is legal for `parametrise`.
    needed(β) = !inband(β) &&
                any(α -> inband(α) && all(β .<= α), exps)
    pts = String[]
    for v in exps
        z = s(v)
        mark = inband(v) ? "kept" : (needed(v) ? "needed" : "dropped")
        push!(pts,
            string("{a:[", join(Vector{Int}(v), ","), "],d:", sum(v),
                ",m:", _js_str(mark),
                ",v:", _js_num(abs(z)),
                ",re:", _js_num(real(z)), ",im:", _js_num(imag(z)), "}"))
    end
    lam = join(
        (string("[", _js_num(real(l)), ",", _js_num(imag(l)), "]")
        for l in sp.eigenvalues), ",")
    string("{name:\"spectrum\",nvar:", nvar,
        ",count:", count(v -> abs(s(v)) < sp.radius, exps),
        ",valueLabel:\"|s|\",note:", _js_str(sp.note),
        ",lam:[", lam, "],radius:", _js_num(sp.radius),
        ",pts:[", join(pts, ","), "]}")
end

"""
	write_lattice(path, spectrum::LatticeSpectrum; title, caption)

Write the split lattice / complex-plane viewer for `spectrum`.  Returns `path`.
"""
function write_lattice(path::AbstractString, sp::LatticeSpectrum;
        title::AbstractString = "Superharmonics",
        caption::AbstractString = "")
    mkpath(dirname(path))
    write(path,
        _lattice_html(title, caption, "[" * _spectrum_js(sp) * "]"; split = true))
    return path
end

"""
	write_lattice(path, conditions::LatticeConditions; title, caption)

Write a standalone viewer whose buttons toggle the conditions of `conditions` — filters
intersect, unions add — over a single lattice.  Returns `path`.
"""
function write_lattice(path::AbstractString, c::LatticeConditions;
        title::AbstractString = "Multiindex lattice",
        caption::AbstractString = "")
    mkpath(dirname(path))
    write(path, _lattice_html(title, caption, "[" * _conditions_js(c) * "]"))
    return path
end

# ---------------------------------------------------------------------------------
# The page.  Colours follow the site's dark palette (--bg #0a0a0f, --acc #9558B2).

function _lattice_html(title, caption, data; split::Bool = false)
    plane_div = split ? "<div id=\"plane\"></div>" : ""
    """
    		<!DOCTYPE html>
    		<html lang="en"><head><meta charset="utf-8">
    		<meta name="viewport" content="width=device-width, initial-scale=1">
    		<title>$(title)</title>
    		<style>
    		:root { --bg:#0a0a0f; --ink:#e8e8ee; --ink2:#a0a0b0; --ink3:#6e6e7e;
    				--hair:#26262f; --acc:#9558B2; --green:#389826; --red:#CB3C33; }
    		* { box-sizing:border-box; }
    		html, body { margin:0; height:100%; }
    		body { background:var(--bg); color:var(--ink); overflow:hidden;
    		  font:13px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; }
    		#wrap { display:flex; flex-direction:column; height:100%; padding:10px 12px; gap:8px; }
    		#bar { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
    		button { font:inherit; font-size:12px; padding:4px 10px; border-radius:5px; cursor:pointer;
    		  background:transparent; color:var(--ink2); border:1px solid var(--hair); }
    		button:hover { border-color:var(--acc); color:var(--ink); }
    		button.on { background:rgba(149,88,178,0.16); border-color:var(--acc); color:var(--ink); }
    		button.f { color:#d98a85; }
    		button.f.on { background:rgba(203,60,51,0.18); border-color:var(--red); color:#ffd9d6; }
    		button.u { color:#8fc487; }
    		button.u.on { background:rgba(56,152,38,0.18); border-color:var(--green); color:#d8f5d0; }
    		button.reset { color:var(--ink2); }
    		.grp { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
    		.grp.right { margin-left:auto; }
    		.grp .sep { color:var(--ink3); font-size:11px; letter-spacing:0.08em;
    		  text-transform:uppercase; margin-right:2px; }
    		select { font:inherit; font-size:12px; background:#111118; color:var(--ink);
    		  border:1px solid var(--hair); border-radius:5px; padding:3px 6px; }
    		#meta { margin-left:auto; color:var(--ink3); font-size:11px;
    		  font-family:ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing:0.04em; }
    		/* One row holding the lattice and, for spectrum figures, the complex plane beside it.
    		   With no #plane the lattice fills the row exactly as it does in every other figure. */
    		#panes { display:flex; flex:1; min-height:0; gap:10px; }
    		#stage { position:relative; flex:1.35; min-height:0; min-width:0;
    		  border:1px solid var(--hair); border-radius:6px; background:#07070b; overflow:hidden; }
    		/* Narrower than the lattice: the complex window is 1:3, so equal aspect leaves side
    		   margin in any wider pane — better to give that width to the lattice. */
    		#plane { position:relative; flex:0.72; min-height:0; min-width:0;
    		  border:1px solid var(--hair); border-radius:6px; background:#07070b; overflow:hidden; }
    		#stage svg, #stage canvas, #plane svg { display:block; width:100%; height:100%; }
    		#stage canvas { cursor:grab; touch-action:none; }
    		#stage canvas.drag { cursor:grabbing; }
    		#note { color:var(--ink3); font-size:11.5px; min-height:1.3em; }
    		#tip { position:absolute; pointer-events:none; opacity:0; transition:opacity .1s;
    		  background:#14141c; border:1px solid var(--hair); border-radius:5px;
    		  padding:6px 9px; font-size:11.5px; color:var(--ink); white-space:pre;
    		  font-family:ui-monospace, SFMono-Regular, Menlo, monospace; z-index:5; }
    		#legend { display:flex; gap:14px; color:var(--ink3); font-size:11px; align-items:center; }
    		#legend i { display:inline-block; width:9px; height:9px; border-radius:50%;
    		  margin-right:5px; vertical-align:-1px; }
    		#ramp { width:78px; height:8px; border-radius:2px; border:1px solid var(--hair);
    		  background:linear-gradient(90deg,#1a376e,#35a0c4,#cfe9f5); }
    		</style></head><body>
    		<div id="wrap">
    		  <div id="bar"></div>
    		  <div id="panes">
    			<div id="stage"><div id="tip"></div></div>
    			$(plane_div)
    		  </div>
    		  <div id="legend"></div>
    		  <div id="note">$(caption)</div>
    		</div>
    		<script>
    		const PANELS = $(data);
    		const COL = { kept:"#9558B2", resonant:"#389826", dropped:"#4a4a58" };
    		// Julia's brand colours, cycled over the layers of the third displayed coordinate.
    		const JULIA = ["#9558B2", "#389826", "#CB3C33", "#4063D8"];
    		const layerColour = k => JULIA[((k % JULIA.length) + JULIA.length) % JULIA.length];
    		const hex2rgb = h => [1,3,5].map(i => parseInt(h.slice(i, i+2), 16));
    		const mix = (h, t) => {           // blend a hex colour towards the panel background
    		  const [r,g,b] = hex2rgb(h), bg = [20,20,28];
    		  return "rgb(" + [r,g,b].map((v,i)=>Math.round(bg[i]+(v-bg[i])*t)).join(",") + ")";
    		};
    		const bar = document.getElementById("bar"), stage = document.getElementById("stage");
    		const tip = document.getElementById("tip"), note = document.getElementById("note");
    		const legend = document.getElementById("legend");
    		const BASE_NOTE = note.textContent;
    		let cur = 0, axes = null;          // axes: which coordinates are displayed

    		// Camera state lives outside the renderer so that orbiting survives a panel switch —
    		// the panels of a figure are successive filters of one lattice, and comparing them
    		// only works from a fixed viewpoint.  α₁ down-left, α₂ down-right, α₃ vertically up:
    		// cos(yaw) < 0 and sin(yaw) < 0 splay the first two axes downwards on either side,
    		// pitch < 0 lifts the third to the vertical.  Two departures from a true isometric:
    		//   · the larger |pitch| flattens the α₁–α₂ plane, i.e. looks from a lower angle;
    		//   · the +8° yaw offset turns the lattice off the 45° diagonal, where
    		//     (α₁+k, α₂+k, α₃+2k) would project onto (α₁, α₂, α₃) exactly.
    		const cam = { yaw: -3*Math.PI/4 + 0.14, pitch: -1.1 };

    		// ── Sequential ramp for |s|: deep blue → cyan → pale.  Deliberately outside the Julia
    		// purple/green/red used for the λ arrows, so a continuous scale can never be mistaken
    		// for one of the categorical eigenvalue colours.
    		function ramp(t) {
    		  t = Math.max(0, Math.min(1, t));
    		  const a = [26,55,110], b = [53,160,196], c = [207,233,245];
    		  const [p, q, u] = t < 0.5 ? [a, b, t*2] : [b, c, (t-0.5)*2];
    		  return "rgb(" + p.map((v,i)=>Math.round(v+(q[i]-v)*u)).join(",") + ")";
    		}
    		// Outside the band but required by downward closure.  Yellow, unused elsewhere.
    		const NEEDED = "#E8C33D";
    		const fmtC = (re, im) => {
    		  const r = Math.round(re*1000)/1000, i = Math.round(im*1000)/1000;
    		  return r + (i < 0 ? " − " + Math.abs(i) : " + " + i) + "i";
    		};
    		const finite = v => v !== null && v === v;
    		const sub = n => String(n).split("").map(d => "₀₁₂₃₄₅₆₇₈₉"[+d]).join("");
    		const sup = n => String(n).split("").map(d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[+d]).join("");
    		const axisName = slot => "α" + sub(axes[slot] + 1);

    		// The monomial as it would be typeset: exponent 0 drops the factor, exponent 1 drops
    		// the superscript, and the empty product is 1.
    		const monomialOf = a => {
    		  const factors = a.map((e, i) => e === 0 ? "" : "z" + sub(i + 1) + (e === 1 ? "" : sup(e)))
    							  .filter(s => s);
    		  return factors.length ? factors.join(" ") : "1";
    		};

    		function panelPoints(P) {
    		  // Aggregate onto the displayed coordinates; a point is a member if any exponent
    		  // mapping to it is a member.
    		  const byKey = new Map();
    		  for (const pt of P.pts) {
    			const c = axes.map(k => pt.a[k]), key = c.join(",");
    			let e = byKey.get(key);
    			if (!e) { e = { c, reps: [], mark: "dropped", v: NaN }; byKey.set(key, e); }
    			e.reps.push(pt);
    			if (pt.m !== "dropped") {
    			  if (e.mark === "dropped" || pt.m === "resonant") e.mark = pt.m;
    			  if (finite(pt.v) && !(e.v <= pt.v)) e.v = pt.v;
    			}
    		  }
    		  return [...byKey.values()];
    		}

    		function tooltipFor(P, e) {
    		  const lines = [];
    		  if (e.reps.length === 1) {
    			const p = e.reps[0];
    			lines.push("α = (" + p.a.join(", ") + ")",
    					   "z^α = " + monomialOf(p.a), "|α| = " + p.d);
    			if (p.re !== undefined) lines.push("s = " + fmtC(p.re, p.im));
    			else if (finite(p.v) && P.valueLabel) lines.push(P.valueLabel + " = " + p.v.toFixed(4));
    			if (p.m === "dropped") lines.push("outside the set");
    			if (p.m === "needed") lines.push("outside the set —", "the closure must add it back");
    			if (p.m === "resonant") lines.push("(resonant: s ≈ λᵣ)");
    		  } else {
    			lines.push("(" + axes.map((_, s) => axisName(s)).join(", ") + ") = (" +
    					   e.c.join(", ") + ")", e.reps.length + " exponents project here");
    			const members = e.reps.filter(p => p.m !== "dropped").length;
    			lines.push(members + " of them in the set");
    			if (finite(e.v) && P.valueLabel) lines.push("max " + P.valueLabel + " = " + e.v.toFixed(4));
    		  }
    		  return lines.join("\\n");
    		}

    		function showTip(html, x, y, colour) {
    		  tip.textContent = html; tip.style.opacity = 1;
    		  // Tint the box with the layer colour so the reading is tied to what is hovered.
    		  tip.style.background = colour ? mix(colour, 0.18) : "#14141c";
    		  tip.style.borderColor = colour ? mix(colour, 0.62) : "#26262f";
    		  const r = stage.getBoundingClientRect();
    		  const w = tip.offsetWidth, h = tip.offsetHeight;
    		  tip.style.left = Math.min(Math.max(4, x + 14), r.width - w - 4) + "px";
    		  tip.style.top  = Math.min(Math.max(4, y - h - 10), r.height - h - 4) + "px";
    		}
    		const hideTip = () => { tip.style.opacity = 0; };

    		// ── 2D renderer: an integer grid drawn as SVG
    		function render2D(P, pts) {
    		  const NS = "http://www.w3.org/2000/svg";
    		  const svg = document.createElementNS(NS, "svg");
    		  const W = stage.clientWidth, H = stage.clientHeight;
    		  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    		  const mx = Math.max(1, ...pts.map(p => p.c[0])), my = Math.max(1, ...pts.map(p => p.c[1]));
    		  const pad = { l: 48, r: 40, t: 32, b: 40 };   // room for the axis labels
    		  const sx = i => pad.l + i * (W - pad.l - pad.r) / mx;
    		  const sy = j => H - pad.b - j * (H - pad.t - pad.b) / my;
    		  const r = Math.max(3.2, Math.min(9, 0.42 * Math.min(
    			(W - pad.l - pad.r) / (mx + 1), (H - pad.t - pad.b) / (my + 1))));

    		  const add = (tag, attrs, parent) => {
    			const el = document.createElementNS(NS, tag);
    			for (const k in attrs) el.setAttribute(k, attrs[k]);
    			(parent || svg).appendChild(el); return el;
    		  };
    		  for (let i = 0; i <= mx; i++)
    			add("line", { x1:sx(i), y1:sy(0), x2:sx(i), y2:sy(my), stroke:"#1c1c26", "stroke-width":1 });
    		  for (let j = 0; j <= my; j++)
    			add("line", { x1:sx(0), y1:sy(j), x2:sx(mx), y2:sy(j), stroke:"#1c1c26", "stroke-width":1 });
    		  add("line", { x1:sx(0), y1:sy(0), x2:sx(mx)+14, y2:sy(0), stroke:"#3a3a48", "stroke-width":1.4 });
    		  add("line", { x1:sx(0), y1:sy(0), x2:sx(0), y2:sy(my)-14, stroke:"#3a3a48", "stroke-width":1.4 });

    		  const lbl = (x, y, t, anchor) => {
    			const e = add("text", { x, y, fill:"#6e6e7e", "font-size":10.5,
    			  "font-family":"ui-monospace, Menlo, monospace", "text-anchor":anchor || "middle" });
    			e.textContent = t; return e;
    		  };
    		  for (let i = 0; i <= mx; i++) lbl(sx(i), sy(0) + 16, i);
    		  for (let j = 0; j <= my; j++) lbl(sx(0) - 10, sy(j) + 4, j, "end");
    		  lbl(sx(mx) + 18, sy(0) + 4, axisName(0), "start");
    		  lbl(sx(0), sy(my) - 20, axisName(1));

    		  const vmax = Math.max(...pts.map(p => finite(p.v) ? p.v : 0), 0);
    		  for (const p of pts) {
    			const dropped = p.mark === "dropped";
    			const fill = dropped ? "none"
    			  : (finite(p.v) && vmax > 0 ? ramp(p.v / vmax) : COL[p.mark] || COL.kept);
    			const c = add("circle", { cx:sx(p.c[0]), cy:sy(p.c[1]), r: dropped ? r * 0.72 : r,
    			  fill, stroke: dropped ? "#4a4a58" : "rgba(0,0,0,0.45)",
    			  "stroke-width": dropped ? 1.3 : 0.8, "stroke-dasharray": dropped ? "2 2" : "none" });
    			add("circle", { cx:sx(p.c[0]), cy:sy(p.c[1]), r: Math.max(r, 9), fill:"transparent" })
    			  .addEventListener("mousemove", ev => {
    				const b = stage.getBoundingClientRect();
    				showTip(tooltipFor(P, p), ev.clientX - b.left, ev.clientY - b.top);
    			  });
    			c.style.pointerEvents = "none";
    		  }
    		  svg.addEventListener("mouseleave", hideTip);
    		  stage.appendChild(svg);
    		}

    		// ── 3D renderer: orthographic projection, drag to orbit
    		function render3D(P, pts) {
    		  const cv = document.createElement("canvas");
    		  stage.appendChild(cv);
    		  const ctx = cv.getContext("2d");
    		  let drag = null;   // camera itself is the shared `cam`, so it survives panel switches
    		  // Per-axis maxima; each axis is drawn one unit beyond its own maximum.
    		  const mxs = [0,1,2].map(k => Math.max(1, ...pts.map(p => p.c[k])));
    		  const ext = mxs.map(m => m + 1);
    		  const vmax = Math.max(...pts.map(p => finite(p.v) ? p.v : 0), 0);
    		  // The layer a point belongs to is its third displayed coordinate.  Spectrum figures
    		  // are the exception: they are plain white, so nothing on the lattice can be mistaken
    		  // for the categorical λ colours used by the arrows in the complex plane.
    		  const colourOf = p => p.mark === "resonant" ? COL.resonant
    			: (P.lam ? "#e8e8ee"
    					 : (finite(p.v) && vmax > 0 ? ramp(p.v/vmax) : layerColour(p.c[2])));

    		  // Unit-scale projection: centre the lattice on its own extent, rotate about the
    		  // vertical, then tilt.  Screen placement is applied afterwards by `fit`, so the
    		  // view fills the panel at any aspect ratio and any camera angle.
    		  function raw(c) {
    			const x = c[0] - mxs[0]/2, y = c[1] - mxs[1]/2, z = c[2] - mxs[2]/2;
    			const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    			const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    			const X = x*cy - y*sy, T = x*sy + y*cy;
    			return { X, Y: T*cp - z*sp, z: T*sp + z*cp };
    		  }

    		  // Scale and offset that fit every point and axis tip inside the stage.
    		  function fit(W, H) {
    			const probes = pts.map(p => p.c)
    			  .concat([[0,0,0]], [0,1,2].map(k => { const e = [0,0,0]; e[k] = ext[k]; return e; }));
    			let lo = { X: Infinity, Y: Infinity }, hi = { X: -Infinity, Y: -Infinity };
    			for (const c of probes) {
    			  const r = raw(c);
    			  lo.X = Math.min(lo.X, r.X); hi.X = Math.max(hi.X, r.X);
    			  lo.Y = Math.min(lo.Y, r.Y); hi.Y = Math.max(hi.Y, r.Y);
    			}
    			const pad = 34;
    			const dX = Math.max(hi.X - lo.X, 1e-6), dY = Math.max(hi.Y - lo.Y, 1e-6);
    			const s = Math.min((W - 2*pad)/dX, (H - 2*pad)/dY);
    			return { s, cx: W/2 - (lo.X + hi.X)/2 * s, cy: H/2 + (lo.Y + hi.Y)/2 * s };
    		  }

    		  let view = { s: 1, cx: 0, cy: 0 };
    		  function project(c) {
    			const r = raw(c);
    			return { x: view.cx + r.X*view.s, y: view.cy - r.Y*view.s, z: r.z };
    		  }

    		  function draw() {
    			const dpr = window.devicePixelRatio || 1;
    			const W = stage.clientWidth, H = stage.clientHeight;
    			cv.width = W*dpr; cv.height = H*dpr;
    			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    			ctx.clearRect(0, 0, W, H);

    			view = fit(W, H);
    			const q = pts.map(p => ({ p, s: project(p.c) }));
    			const zs = q.map(o => o.s.z), zlo = Math.min(...zs), zhi = Math.max(...zs);
    			const depth01 = z => (zhi - zlo) < 1e-9 ? 1 : (z - zlo) / (zhi - zlo);

    			// Lattice edges: exponents one unit apart, drawn behind the markers.  In-layer
    			// edges take the layer colour; edges crossing layers stay neutral.
    			const at = new Map(q.map(o => [o.p.c.join(","), o]));
    			ctx.lineWidth = 1;
    			for (const o of q) {
    			  if (o.p.mark === "dropped") continue;
    			  for (let k = 0; k < 3; k++) {
    				const nb = o.p.c.slice(); nb[k] += 1;
    				const t = at.get(nb.join(","));
    				if (!t || t.p.mark === "dropped") continue;
    				const a = 0.12 + 0.20*depth01((o.s.z + t.s.z)/2);
    				ctx.globalAlpha = a;
    				ctx.strokeStyle = k === 2 ? "#5a5a6a" : layerColour(o.p.c[2]);
    				ctx.beginPath(); ctx.moveTo(o.s.x, o.s.y); ctx.lineTo(t.s.x, t.s.y); ctx.stroke();
    			  }
    			}
    			ctx.globalAlpha = 1;

    			// Axis stubs, each running one unit past its own maximum.
    			const o0 = project([0,0,0]);
    			for (let k = 0; k < 3; k++) {
    			  const e = [0,0,0]; e[k] = ext[k];
    			  const p1 = project(e);
    			  ctx.strokeStyle = "#3a3a48"; ctx.lineWidth = 1.2;
    			  ctx.beginPath(); ctx.moveTo(o0.x, o0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
    			  // Arrow head, oriented along the projected axis direction.
    			  const dx = p1.x - o0.x, dy = p1.y - o0.y, L = Math.hypot(dx, dy) || 1;
    			  const ux = dx/L, uy = dy/L, hs = 6;
    			  ctx.beginPath();
    			  ctx.moveTo(p1.x, p1.y);
    			  ctx.lineTo(p1.x - hs*ux + hs*0.5*uy, p1.y - hs*uy - hs*0.5*ux);
    			  ctx.lineTo(p1.x - hs*ux - hs*0.5*uy, p1.y - hs*uy + hs*0.5*ux);
    			  ctx.closePath(); ctx.fillStyle = "#3a3a48"; ctx.fill();
    			  ctx.fillStyle = k === 2 ? "#8a8a9a" : "#6e6e7e";
    			  ctx.font = "12px ui-monospace, Menlo, monospace";
    			  ctx.fillText(axisName(k), p1.x + 7*ux + 4, p1.y + 7*uy + 4);
    			}

    			q.sort((a, b) => a.s.z - b.s.z);           // painter's algorithm
    			for (const o of q) {
    			  const d = depth01(o.s.z);
    			  const base = Math.max(3, Math.min(11, view.s * 0.13));   // relative to the fit
    			  const r = base * (0.62 + 0.38*d);
    			  ctx.globalAlpha = 0.42 + 0.58*d;
    			  if (o.p.mark === "needed") {
    				// Outside the band, but the downward closure has to put it back.
    				ctx.strokeStyle = NEEDED; ctx.lineWidth = 2.4;
    				ctx.beginPath(); ctx.arc(o.s.x, o.s.y, r*0.85, 0, 6.2832); ctx.stroke();
    			  } else if (o.p.mark === "dropped") {
    				ctx.globalAlpha *= 0.6;
    				ctx.strokeStyle = "#4a4a58"; ctx.lineWidth = 1.1;
    				ctx.beginPath(); ctx.arc(o.s.x, o.s.y, r*0.7, 0, 6.2832); ctx.stroke();
    			  } else {
    				ctx.fillStyle = colourOf(o.p);
    				ctx.beginPath(); ctx.arc(o.s.x, o.s.y, r, 0, 6.2832); ctx.fill();
    			  }
    			  o.r = r;
    			}
    			ctx.globalAlpha = 1;
    			cv._q = q;
    		  }

    		  cv.addEventListener("pointerdown", e => {
    			drag = { x: e.clientX, y: e.clientY }; cv.classList.add("drag");
    			cv.setPointerCapture(e.pointerId); hideTip();
    		  });
    		  cv.addEventListener("pointerup", e => {
    			drag = null; cv.classList.remove("drag");
    			try { cv.releasePointerCapture(e.pointerId); } catch (_) {}
    		  });
    		  cv.addEventListener("pointerleave", () => { hideTip(); clearPath(); });
    		  cv.addEventListener("pointermove", e => {
    			if (drag) {
    			  cam.yaw += (e.clientX - drag.x) * 0.009;
    			  cam.pitch += (e.clientY - drag.y) * 0.009;
    			  cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
    			  drag = { x: e.clientX, y: e.clientY };
    			  draw();
    			  return;
    			}
    			const b = stage.getBoundingClientRect();
    			const mxp = e.clientX - b.left, myp = e.clientY - b.top;
    			let best = null, bd = 1e9;
    			for (const o of (cv._q || [])) {
    			  const d = Math.hypot(o.s.x - mxp, o.s.y - myp);
    			  if (d < Math.max(o.r + 4, 9) && d < bd) { bd = d; best = o; }
    			}
    			if (best) {
    			  showTip(tooltipFor(P, best.p), mxp, myp,
    					  best.p.mark === "dropped" ? null : colourOf(best.p));
    			  // One rep only: a projected cluster has no single superharmonic to draw.
    			  if (P.lam && best.p.reps.length === 1) drawPath(P, best.p.reps[0]);
    			} else { hideTip(); clearPath(); }
    		  });

    		  draw();
    		  cv._redraw = draw;
    		}

    		// ── Complex plane: every superharmonic s(α) = ⟨λ, α⟩, with the band |s| < R drawn as a
    		// circle.  Hovering a monomial on the lattice draws the head-to-tail sum reaching it.
    		const plane = document.getElementById("plane");
    		let planeView = null;   // { sx, sy, NS, gPath, gHit } once rendered

    		function renderPlane(P) {
    		  plane.innerHTML = "";
    		  const NS = "http://www.w3.org/2000/svg";
    		  const W = plane.clientWidth, H = plane.clientHeight;
    		  const svg = document.createElementNS(NS, "svg");
    		  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    		  const add = (tag, attrs, parent) => {
    			const el = document.createElementNS(NS, tag);
    			for (const k in attrs) el.setAttribute(k, attrs[k]);
    			(parent || svg).appendChild(el); return el;
    		  };

    		  // Fixed window on the damped half-plane, where every superharmonic actually lives —
    		  // an auto-fit wastes the whole Re > 0 side.  EQUAL aspect throughout, so |s| reads as
    		  // a true distance and the band stays a circle rather than an ellipse.
    		  const R = P.radius;
    		  const view = P.window || [-5, 1, -9, 9];   // [reLo, reHi, imLo, imHi]
    		  const pad = 26;
    		  const s = Math.min((W - 2*pad) / (view[1] - view[0]),
    							 (H - 2*pad) / (view[3] - view[2]));
    		  const ox = W/2 - (view[0] + view[1])/2 * s;
    		  const oy = H/2 + (view[2] + view[3])/2 * s;
    		  const sx = re => ox + re*s, sy = im => oy - im*s;
    		  const lim = Math.max(Math.abs(view[0]), view[1], Math.abs(view[2]), view[3]);

    		  // Axes span the pane; only the ticks respect the window.
    		  add("line", { x1:6, y1:sy(0), x2:W-6, y2:sy(0), stroke:"#3a3a48", "stroke-width":1 });
    		  add("line", { x1:sx(0), y1:6, x2:sx(0), y2:H-6, stroke:"#3a3a48", "stroke-width":1 });
    		  const lbl = (x, y, t, anchor, fill) => {
    			const e = add("text", { x, y, fill: fill || "#6e6e7e", "font-size":10.5,
    			  "font-family":"ui-monospace, Menlo, monospace", "text-anchor":anchor || "middle" });
    			e.textContent = t; return e;
    		  };
    		  lbl(W - 8, sy(0) - 7, "Re", "end");
    		  lbl(sx(0) + 7, 16, "Im", "start");
    		  for (let t = Math.ceil(view[0]); t <= Math.floor(view[1]); t++) {
    			if (t === 0) continue;
    			add("line", { x1:sx(t), y1:sy(0)-3, x2:sx(t), y2:sy(0)+3, stroke:"#3a3a48" });
    		  }
    		  for (let t = Math.ceil(view[2]); t <= Math.floor(view[3]); t++) {
    			if (t === 0) continue;
    			add("line", { x1:sx(0)-3, y1:sy(t), x2:sx(0)+3, y2:sy(t), stroke:"#3a3a48" });
    		  }

    		  // the band
    		  add("circle", { cx:sx(0), cy:sy(0), r:R*s, fill:"none", stroke:"#9558B2",
    			"stroke-width":1.2, "stroke-dasharray":"4 3", opacity:0.75 });
    		  lbl(sx(0), sy(0) - R*s*1.05, "|s| = " + R, "start", "#9558b2");

    		  for (const p of P.pts) {
    			if (p.m === "needed") {
    			  add("circle", { cx:sx(p.re), cy:sy(p.im), r:4.6, fill:"none",
    				stroke:NEEDED, "stroke-width":2.4 });
    			} else if (p.m === "dropped") {
    			  add("circle", { cx:sx(p.re), cy:sy(p.im), r:3.4, fill:"none",
    				stroke:"#4a4a58", "stroke-width":1.2, "stroke-dasharray":"2 2" });
    			} else {
    			  add("circle", { cx:sx(p.re), cy:sy(p.im), r:4.8, fill:"#e8e8ee",
    				stroke:"rgba(0,0,0,0.45)", "stroke-width":0.8 });
    			}
    		  }

    		  const gPath = add("g", { class:"path" });
    		  plane.appendChild(svg);
    		  planeView = { sx, sy, NS, gPath, svg, add };
    		}

    		function clearPath() { if (planeView) planeView.gPath.innerHTML = ""; }

    		// Head-to-tail vector addition: s = α₁ λ₁ + α₂ λ₂ + α₃ λ₃.
    		// Arrows are coloured by which variable contributed them, matching the lattice layers.
    		function drawPath(P, pt) {
    		  if (!planeView) return;
    		  const { sx, sy, gPath } = planeView;
    		  gPath.innerHTML = "";
    		  const NS = planeView.NS;
    		  const mk = (tag, attrs) => {
    			const el = document.createElementNS(NS, tag);
    			for (const k in attrs) el.setAttribute(k, attrs[k]);
    			gPath.appendChild(el); return el;
    		  };
    		  let re = 0, im = 0;
    		  for (let k = 0; k < P.lam.length; k++) {
    			const [lr, li] = P.lam[k];
    			for (let n = 0; n < pt.a[k]; n++) {
    			  const re2 = re + lr, im2 = im + li;
    			  const x1 = sx(re), y1 = sy(im), x2 = sx(re2), y2 = sy(im2);
    			  const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
    			  const ux = dx/L, uy = dy/L, head = Math.min(7, L*0.34);
    			  mk("line", { x1, y1, x2: x2 - ux*head*0.6, y2: y2 - uy*head*0.6,
    				stroke: layerColour(k), "stroke-width":2, opacity:0.95 });
    			  mk("polygon", { points:
    				  [x2, y2,
    				   x2 - head*ux + head*0.42*uy, y2 - head*uy - head*0.42*ux,
    				   x2 - head*ux - head*0.42*uy, y2 - head*uy + head*0.42*ux].join(" "),
    				fill: layerColour(k) });
    			  // Repeated copies of one λ are collinear; a dot at each vertex makes them
    			  // countable instead of reading as a single long arrow.
    			  mk("circle", { cx:x2, cy:y2, r:2.4, fill:layerColour(k) });
    			  re = re2; im = im2;
    			}
    		  }
    		  mk("circle", { cx:sx(0), cy:sy(0), r:3, fill:"#e8e8ee" });
    		  mk("circle", { cx:sx(re), cy:sy(im), r:8, fill:"none",
    			stroke:"#e8e8ee", "stroke-width":1.6 });
    		}

    		function renderLegend(P, pts) {
    		  const seen = new Set(pts.map(p => p.mark));
    		  const is3D = axes.length === 3;
    		  const bits = [];
    		  if (P.lam) {
    			// No gradient here — the lattice is plain white so the λ colours stay unambiguous.
    			bits.push('<span><i style="background:#e8e8ee"></i>in the band</span>');
    		  } else if (P.valueLabel && pts.some(p => finite(p.v))) {
    			const vs = pts.filter(p => finite(p.v)).map(p => p.v);
    			bits.push('<span><span id="ramp" style="display:inline-block"></span>&nbsp; ' +
    			  P.valueLabel + " 0 … " + Math.max(...vs).toFixed(2) + "</span>");
    		  } else if (is3D) {
    			// One swatch per layer of the third displayed coordinate.
    			const layers = [...new Set(pts.filter(p => p.mark !== "dropped").map(p => p.c[2]))]
    			  .sort((a, b) => a - b);
    			for (const L of layers)
    			  bits.push('<span><i style="background:' + layerColour(L) + '"></i>' +
    				axisName(2) + " = " + L + "</span>");
    			if (seen.has("resonant"))
    			  bits.push('<span><i style="background:'+COL.resonant+'"></i>resonant</span>');
    		  } else {
    			if (seen.has("kept"))     bits.push('<span><i style="background:'+COL.kept+'"></i>in the set</span>');
    			if (seen.has("resonant")) bits.push('<span><i style="background:'+COL.resonant+'"></i>resonant</span>');
    		  }
    		  if (seen.has("dropped"))
    			bits.push('<span><i style="border:1px dashed #4a4a58"></i>' +
    			  (P.lam ? "outside the set" : "removed") + "</span>");
    		  if (seen.has("needed"))
    			bits.push('<span><i style="background:transparent;border:2px solid ' + NEEDED +
    			  '"></i>factor the closure must add</span>');
    		  if (P.lam) {
    			// Explain the arrow colours of the hover path.
    			const fmt = ([r, i]) => r + (i < 0 ? " − " + (-i) : " + " + i) + "i";
    			P.lam.forEach((l, k) => bits.push('<span><i style="background:' + layerColour(k) +
    			  '"></i>λ' + sub(k+1) + " = " + fmt(l) + "</span>"));
    		  }
    		  if (is3D) bits.push("<span>drag to orbit</span>");
    		  if (P.nvar > axes.length)
    			bits.push('<span>projected — hover for the exponents behind a point</span>');
    		  legend.innerHTML = bits.join("");
    		}

    		// ── Conditions mode: active filters intersect, active unions add on top.
    		// A monomial is a member when it passes every active filter (vacuously true when none
    		// is active) OR belongs to any active union.
    		const active = { f: new Set(), u: new Set() };

    		function applyConditions(P) {
    		  const fs = [...active.f].map(i => P.filters[i].mask);
    		  const us = [...active.u].map(i => P.unions[i].mask);
    		  let n = 0;
    		  P.pts.forEach((pt, i) => {
    			const passes = fs.length === 0 || fs.every(m => m[i]);
    			const added = us.some(m => m[i]);
    			pt.m = (passes || added) ? "kept" : "dropped";
    			if (pt.m === "kept") n++;
    		  });
    		  P.count = n;
    		  const names = [...active.f].map(i => P.filters[i].name);
    		  const adds = [...active.u].map(i => P.unions[i].name);
    		  let line = names.length ? names.join("  ∩  ") : "no filter — the whole lattice";
    		  if (adds.length) line += "   ∪  " + adds.join("  ∪  ");
    		  return line;
    		}

    		function render() {
    		  const P = PANELS[cur];
    		  if (!axes || axes.length !== Math.min(P.nvar, 3) || axes.some(k => k >= P.nvar))
    			axes = [...Array(Math.min(P.nvar, 3)).keys()];
    		  const conditional = !!P.filters;
    		  const line = conditional ? applyConditions(P) : null;
    		  stage.querySelectorAll("svg,canvas").forEach(e => e.remove());
    		  hideTip();
    		  const pts = panelPoints(P);
    		  (axes.length === 3 ? render3D : render2D)(P, pts);
    		  if (P.lam) renderPlane(P);
    		  renderLegend(P, pts);
    		  note.textContent = conditional ? line : (P.note || BASE_NOTE);
    		  document.getElementById("meta").textContent =
    			P.count + " monomials · " + P.nvar + " variables";
    		}

    		// Conditions bar: [reset] [red filters …]        [green unions …] [count]
    		function buildConditionBar(P) {
    		  bar.innerHTML = "";
    		  const left = document.createElement("span"); left.className = "grp";
    		  const right = document.createElement("span"); right.className = "grp right";

    		  const reset = document.createElement("button");
    		  reset.textContent = "reset"; reset.className = "reset";
    		  reset.onclick = () => { active.f.clear(); active.u.clear(); buildBar(); render(); };
    		  left.appendChild(reset);

    		  if (P.filters.length) {
    			const lab = document.createElement("span");
    			lab.className = "sep"; lab.textContent = "filter"; left.appendChild(lab);
    		  }
    		  P.filters.forEach((f, i) => {
    			const b = document.createElement("button");
    			b.textContent = f.name;
    			b.className = "f" + (active.f.has(i) ? " on" : "");
    			b.onclick = () => {
    			  active.f.has(i) ? active.f.delete(i) : active.f.add(i);
    			  buildBar(); render();
    			};
    			left.appendChild(b);
    		  });

    		  if (P.unions.length) {
    			const lab = document.createElement("span");
    			lab.className = "sep"; lab.textContent = "add"; right.appendChild(lab);
    		  }
    		  P.unions.forEach((u, i) => {
    			const b = document.createElement("button");
    			b.textContent = u.name;
    			b.className = "u" + (active.u.has(i) ? " on" : "");
    			b.onclick = () => {
    			  active.u.has(i) ? active.u.delete(i) : active.u.add(i);
    			  buildBar(); render();
    			};
    			right.appendChild(b);
    		  });

    		  const meta = document.createElement("span");
    		  meta.id = "meta"; meta.style.marginLeft = "14px"; right.appendChild(meta);
    		  bar.append(left, right);
    		}

    		function buildBar() {
    		  const cond = PANELS[cur].filters;
    		  if (cond) { buildConditionBar(PANELS[cur]); return; }
    		  bar.innerHTML = "";
    		  // A lone panel needs no switcher — the button would do nothing.
    		  if (PANELS.length > 1) PANELS.forEach((P, i) => {
    			const b = document.createElement("button");
    			b.textContent = P.name; b.className = i === cur ? "on" : "";
    			b.onclick = () => { cur = i; axes = null; buildBar(); render(); };
    			bar.appendChild(b);
    		  });
    		  const P = PANELS[cur];
    		  if (P.nvar > 3) {
    			// Which coordinates to display; duplicates are aggregated by panelPoints.
    			const want = Math.min(3, P.nvar);
    			if (!axes) axes = [...Array(want).keys()];
    			const wrap = document.createElement("span");
    			wrap.style.cssText = "display:flex;gap:5px;align-items:center;color:#6e6e7e;font-size:11px";
    			wrap.append("show ");
    			for (let s = 0; s < want; s++) {
    			  const sel = document.createElement("select");
    			  for (let k = 0; k < P.nvar; k++) {
    				const o = document.createElement("option");
    				o.value = k; o.textContent = "α" + (k+1); if (k === axes[s]) o.selected = true;
    				sel.appendChild(o);
    			  }
    			  sel.onchange = () => { axes[s] = +sel.value; render(); };
    			  wrap.appendChild(sel);
    			}
    			bar.appendChild(wrap);
    		  }
    		  const meta = document.createElement("span");
    		  meta.id = "meta"; bar.appendChild(meta);
    		}

    		// Surface render failures instead of leaving a silently blank stage.
    		addEventListener("error", e => {
    		  note.textContent = "render error: " + (e.message || e.error);
    		  note.style.color = "#CB3C33";
    		});

    		buildBar(); render();
    		let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(render, 120); });
    		</script></body></html>
    		"""
end
