# viz.jl — self-contained chart viewer for the full-order-model tutorial.
#
# Two writers, sharing one renderer core:
#
#   write_charts — switchable panels of xy-curves.  Line charts and phase portraits differ
#                  only by `equal_aspect`, which forces one data unit to occupy the same
#                  number of pixels on both axes; without it a circular orbit would be
#                  drawn as an ellipse.
#   write_split  — a split view: a rotatable 3-D phase-space plot on the left, the same
#                  trajectory's coordinates against time on the right.  Panels switch both
#                  panes together, so one figure can show the same orbit in two coordinate
#                  systems.
#
# Every output is a single standalone HTML file: no external stylesheet, script, font or
# CDN, so the page works from the file system and can be embedded in the website with a
# plain <iframe>.
#
# Deliberately imports nothing: the tutorial has to run under a plain `julia --project`,
# and the root environment carries no plotting package.

"""
	Curve(label, x, y; colour, dashed, width)

One curve. `colour` indexes the Julia brand palette (1-based, cycled); `dashed` draws the
curve as a reference rather than a result; `width` scales the stroke, for the occasion
where one curve of a panel is the answer and the rest are context.
"""
struct Curve
    label::String
    x::Vector{Float64}
    y::Vector{Float64}
    colour::Int
    dashed::Bool
    width::Float64
end

function Curve(label, x, y; colour::Integer = 1, dashed::Bool = false, width::Real = 1.0)
    xs = Float64.(collect(x))
    ys = Float64.(collect(y))
    length(xs) == length(ys) ||
        throw(ArgumentError("Curve \"$label\": x and y have lengths " *
                            "$(length(xs)) and $(length(ys))"))
    return Curve(String(label), xs, ys, Int(colour), dashed, Float64(width))
end

"""
	Orbit3D(label, x, y, z; colour, dashed)

One trajectory in phase space, for the left pane of a split figure.
"""
struct Orbit3D
    label::String
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    colour::Int
    dashed::Bool
end

function Orbit3D(label, x, y, z; colour::Integer = 1, dashed::Bool = false)
    xs, ys, zs = Float64.(collect(x)), Float64.(collect(y)), Float64.(collect(z))
    (length(xs) == length(ys) == length(zs)) ||
        throw(ArgumentError("Orbit3D \"$label\": x, y and z have lengths " *
                            "$(length(xs)), $(length(ys)) and $(length(zs))"))
    return Orbit3D(String(label), xs, ys, zs, Int(colour), dashed)
end

"""
	Arrow3D(from, to; label, colour)

A straight arrow in phase space, for annotating a 3-D pane with a direction the trajectory
itself does not show — a translation, an axis, an equilibrium offset.

Drawn after the polylines and never depth-sorted against them: an annotation that a fold of
the orbit could hide would be worse than one that always reads. `label`, if given, sits at
the tip.
"""
struct Arrow3D
    from::NTuple{3, Float64}
    to::NTuple{3, Float64}
    label::String
    colour::Int
end

function Arrow3D(from, to; label::AbstractString = "", colour::Integer = 3)
    f = NTuple{3, Float64}(Float64.(Tuple(from)))
    t = NTuple{3, Float64}(Float64.(Tuple(to)))
    return Arrow3D(f, t, String(label), Int(colour))
end

"""
	Surface3D(x, y, z)

A surface sampled on a grid: `z[i, j]` is the height above `(x[i], y[j])`. Drawn with a
painter's algorithm, so it occludes itself the way a wireframe cannot.
"""
struct Surface3D
    x::Vector{Float64}
    y::Vector{Float64}
    z::Matrix{Float64}
end

function Surface3D(x, y, z::AbstractMatrix)
    xs, ys = Float64.(collect(x)), Float64.(collect(y))
    size(z) == (length(xs), length(ys)) ||
        throw(ArgumentError("Surface3D: z is $(size(z)) but x, y have lengths " *
                            "$(length(xs)), $(length(ys))"))
    return Surface3D(xs, ys, Float64.(z))
end

"""
	Surface3D(x, y, f)

Sample `f(xᵢ, yⱼ)` over the grid.
"""
Surface3D(x, y, f::Function) = Surface3D(x, y, [f(a, b) for a in x, b in y])

"""
	SweptLine(x, amplitude; omega, phase, offset)

A line that sweeps across a surface as its parameter advances, given in closed form:

	F(xᵢ, t) = amplitude[i] · cos(omega · t + phase) + offset

No grid and no interpolation — `x` needs only as many points as the law has curvature in
it, which for a straight line is two. The line lies exactly on the surface; it is drawn
after it, so it reads as painted onto the sheet.

`offset` lifts it clear of the surface by a constant. At `offset = 0` the line and the
sheet are the same points, so which one a pixel shows is left to depth-buffer rounding and
the line stitches in and out of the surface; a lift of about 1% of the z-range is enough to
keep it consistently on top without reading as detached.
"""
struct SweptLine
    x::Vector{Float64}
    amplitude::Vector{Float64}
    omega::Float64
    phase::Float64
    offset::Float64
end

function SweptLine(x, amplitude; omega::Real, phase::Real = 0.0, offset::Real = 0.0)
    xs, as = Float64.(collect(x)), Float64.(collect(amplitude))
    length(xs) == length(as) ||
        throw(ArgumentError("SweptLine: x and amplitude have lengths " *
                            "$(length(xs)) and $(length(as))"))
    return SweptLine(xs, as, Float64(omega), Float64(phase), Float64(offset))
end

"""
	ChartPanel(name, series; xlabel, ylabel, note, equal_aspect)

One switchable panel of a `write_charts` figure.

- `name`: label on the panel's button.
- `series`: the curves to draw, in legend order.
- `xlabel`, `ylabel`: axis labels.
- `note`: one line of prose under the panel.
- `equal_aspect`: give both axes the same scale. Use it for phase portraits, where the
  shape of the orbit is the message; leave it off for time series.
- `curves3d`, `axes`: pass these instead of `series` to make the panel a rotatable 3-D
  wireframe. A panel is either 2-D or 3-D, never both.
"""
struct ChartPanel
    name::String
    series::Vector{Curve}
    curves3d::Vector{Orbit3D}
    surface::Union{Nothing, Surface3D}
    line::Union{Nothing, SweptLine}
    axes::NTuple{3, String}
    xlabel::String
    ylabel::String
    note::String
    equal_aspect::Bool
end

function ChartPanel(name, series::AbstractVector{Curve};
        xlabel::AbstractString = "",
        ylabel::AbstractString = "",
        note::AbstractString = "",
        equal_aspect::Bool = false)
    return ChartPanel(String(name), collect(series), Orbit3D[], nothing, nothing,
        ("x", "y", "z"), String(xlabel), String(ylabel), String(note), equal_aspect)
end

ChartPanel(name, s::Curve; kwargs...) = ChartPanel(name, [s]; kwargs...)

"""
	ChartPanel(name, curves3d; axes, note)

A 3-D panel: a wireframe of `curves3d`, drag to orbit.
"""
function ChartPanel(name, curves3d::AbstractVector{Orbit3D};
        axes = ("x", "y", "z"),
        note::AbstractString = "")
    return ChartPanel(String(name), Curve[], collect(curves3d), nothing, nothing,
        (String(axes[1]), String(axes[2]), String(axes[3])), "", "", String(note), false)
end

"""
	ChartPanel(name, surface; axes, note)

A 3-D panel showing a shaded, self-occluding surface. Drag to orbit. A line sweeps across
it at constant speed, showing the law at one value of the second coordinate. The line is
read from the same grid the surface is drawn from, so it lies exactly on the surface as
rendered rather than on the underlying law.
"""
function ChartPanel(name, surface::Surface3D;
        line::Union{Nothing, SweptLine} = nothing,
        axes = ("x", "y", "z"),
        note::AbstractString = "")
    return ChartPanel(String(name), Curve[], Orbit3D[], surface, line,
        (String(axes[1]), String(axes[2]), String(axes[3])), "", "", String(note), false)
end

"""
	SplitPanel(name, orbits, series; axes, xlabel, ylabel, note)

One switchable panel of a `write_split` figure: `orbits` fill the 3-D pane on the left and
`series` the time-plot pane on the right. `axes` names the three phase-space coordinates.
"""
struct SplitPanel
    name::String
    orbits::Vector{Orbit3D}
    series::Vector{Curve}
    arrows::Vector{Arrow3D}
    axes::NTuple{3, String}
    xlabel::String
    ylabel::String
    note::String
end

function SplitPanel(name, orbits::AbstractVector{Orbit3D}, series::AbstractVector{Curve};
        arrows::AbstractVector{Arrow3D} = Arrow3D[],
        axes = ("x", "y", "z"),
        xlabel::AbstractString = "t",
        ylabel::AbstractString = "",
        note::AbstractString = "")
    return SplitPanel(String(name), collect(orbits), collect(series), collect(arrows),
        (String(axes[1]), String(axes[2]), String(axes[3])),
        String(xlabel), String(ylabel), String(note))
end

"""
	write_charts(path, panels; title, caption) -> path

Write a standalone HTML figure holding one switchable panel per entry of `panels`.
"""
function write_charts(path::AbstractString, panels::AbstractVector{ChartPanel};
        title::AbstractString = "MORFE figure",
        caption::AbstractString = "")
    mkpath(dirname(path))
    data = "[" * join((_panel_js(p) for p in panels), ",") * "]"
    write(path, _charts_html(title, caption, data))
    return path
end

function write_charts(path::AbstractString, panel::ChartPanel; kwargs...)
    write_charts(path, [panel]; kwargs...)
end

"""
	PairPanel(name, time, phase; tylabel, pxlabel, pylabel, txlabel, equal_aspect, note)

One switchable panel of a `write_pairs` figure: `time` fills the left pane and `phase` the
right. `equal_aspect` applies to the phase pane only — a circular orbit has to read as a
circle, while a time trace should fill its pane.
"""
struct PairPanel
    name::String
    time::Vector{Curve}
    phase::Vector{Curve}
    txlabel::String
    tylabel::String
    pxlabel::String
    pylabel::String
    equal_aspect::Bool
    note::String
end

function PairPanel(name, time::AbstractVector{Curve}, phase::AbstractVector{Curve};
        txlabel::AbstractString = "t",
        tylabel::AbstractString = "",
        pxlabel::AbstractString = "",
        pylabel::AbstractString = "",
        equal_aspect::Bool = true,
        note::AbstractString = "")
    return PairPanel(String(name), collect(time), collect(phase), String(txlabel),
        String(tylabel), String(pxlabel), String(pylabel), equal_aspect, String(note))
end

"""
	write_pairs(path, panels; title, caption) -> path

Write a standalone HTML figure pairing a time plot on the left with a phase portrait on
the right, one switchable panel per entry of `panels`.
"""
function write_pairs(path::AbstractString, panels::AbstractVector{PairPanel};
        title::AbstractString = "MORFE figure",
        caption::AbstractString = "")
    mkpath(dirname(path))
    data = "[" * join((_pair_js(p) for p in panels), ",") * "]"
    write(path, _pairs_html(title, caption, data))
    return path
end

"""
	write_thumbnail(path, curves; width, height, pad) -> path

Write a bare standalone SVG of `curves` — no axes, no labels, no background — sized for a
website card. It is a glyph, not a figure: each axis is scaled to its own data so the
shape fills the frame.
"""
function write_thumbnail(path::AbstractString, curves::AbstractVector{Curve};
        width::Int = 400, height::Int = 225, pad::Int = 16)
    mkpath(dirname(path))
    x0 = minimum(minimum(c.x) for c in curves)
    x1 = maximum(maximum(c.x) for c in curves)
    y0 = minimum(minimum(c.y) for c in curves)
    y1 = maximum(maximum(c.y) for c in curves)
    sx(v) = pad + (v - x0) / ((x1 - x0) == 0 ? 1 : x1 - x0) * (width - 2pad)
    sy(v) = height - pad - (v - y0) / ((y1 - y0) == 0 ? 1 : y1 - y0) * (height - 2pad)
    body = map(curves) do c
        # One decimal is a tenth of a pixel at this size — finer than the card can show.
        d = join(
            (string(i == 1 ? "M" : "L", round(sx(c.x[i]), digits = 1), " ",
                 round(sy(c.y[i]), digits = 1)) for i in eachindex(c.x)),
            " ")
        string("<path d=\"", d, "\" fill=\"none\" stroke=\"", _hex(c.colour),
            "\" stroke-width=\"", round(1.7 * c.width, digits = 2),
            "\" stroke-linejoin=\"round\" stroke-linecap=\"round\" opacity=\"",
            c.dashed ? 0.6 : 1, "\"/>")
    end
    write(path,
        string("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 ", width, " ",
            height, "\" preserveAspectRatio=\"xMidYMid meet\" role=\"img\">",
            join(body), "</svg>"))
    return path
end

"""
	write_split(path, panels; title, caption) -> path

Write a standalone HTML figure with a rotatable 3-D phase-space pane beside a time-plot
pane. Drag the left pane to orbit the camera.
"""
function write_split(path::AbstractString, panels::AbstractVector{SplitPanel};
        title::AbstractString = "MORFE figure",
        caption::AbstractString = "")
    mkpath(dirname(path))
    data = "[" * join((_split_js(p) for p in panels), ",") * "]"
    write(path, _split_html(title, caption, data))
    return path
end

# ---------------------------------------------------------------------------------
# Serialisation.  Five significant digits keeps the pages small and is still far finer
# than any screen can resolve; the figures are drawings, not data files.

# Julia's brand colours, matching the JULIA array used inside the pages.
const _JULIA_HEX = ("#9558B2", "#389826", "#CB3C33", "#4063D8")
_hex(colour::Integer) = _JULIA_HEX[mod1(colour, length(_JULIA_HEX))]

_num(v::Real) = isfinite(v) ? string(round(Float64(v), sigdigits = 5)) : "null"
_arr(v) = "[" * join((_num(x) for x in v), ",") * "]"
_str(s) = "\"" * replace(String(s), "\\" => "\\\\", "\"" => "\\\"", "\n" => " ") * "\""

function _series_js(s::Curve)
    return "{l:$(_str(s.label)),c:$(s.colour - 1),d:$(s.dashed ? 1 : 0)," *
           "w:$(_num(s.width)),x:$(_arr(s.x)),y:$(_arr(s.y))}"
end

function _orbit_js(o::Orbit3D)
    return "{l:$(_str(o.label)),c:$(o.colour - 1),d:$(o.dashed ? 1 : 0)," *
           "x:$(_arr(o.x)),y:$(_arr(o.y)),z:$(_arr(o.z))}"
end

function _arrow_js(a::Arrow3D)
    return "{f:[$(_num(a.from[1])),$(_num(a.from[2])),$(_num(a.from[3]))]," *
           "t:[$(_num(a.to[1])),$(_num(a.to[2])),$(_num(a.to[3]))]," *
           "l:$(_str(a.label)),c:$(a.colour - 1)}"
end

function _panel_js(p::ChartPanel)
    return "{n:$(_str(p.name)),xl:$(_str(p.xlabel)),yl:$(_str(p.ylabel))," *
           "note:$(_str(p.note)),eq:$(p.equal_aspect ? 1 : 0)," *
           "ax:[$(_str(p.axes[1])),$(_str(p.axes[2])),$(_str(p.axes[3]))]," *
           "sf:$(_surface_js(p.surface)),ln:$(_line_js(p.line))," *
           "o:[" * join((_orbit_js(o) for o in p.curves3d), ",") * "]," *
           "s:[" * join((_series_js(s) for s in p.series), ",") * "]}"
end

_line_js(::Nothing) = "null"
function _line_js(l::SweptLine)
    return "{x:$(_arr(l.x)),a:$(_arr(l.amplitude)),w:$(_num(l.omega))," *
           "p:$(_num(l.phase)),o:$(_num(l.offset))}"
end

_surface_js(::Nothing) = "null"
function _surface_js(s::Surface3D)
    rows = join(
        ("[" * join((_num(s.z[i, j]) for j in axes(s.z, 2)), ",") * "]"
        for i in axes(s.z, 1)), ",")
    return "{x:$(_arr(s.x)),y:$(_arr(s.y)),z:[$rows]}"
end

function _pair_js(p::PairPanel)
    return "{n:$(_str(p.name)),note:$(_str(p.note))," *
           "t:{xl:$(_str(p.txlabel)),yl:$(_str(p.tylabel)),eq:0," *
           "s:[" * join((_series_js(s) for s in p.time), ",") * "]}," *
           "p:{xl:$(_str(p.pxlabel)),yl:$(_str(p.pylabel))," *
           "eq:$(p.equal_aspect ? 1 : 0)," *
           "s:[" * join((_series_js(s) for s in p.phase), ",") * "]}}"
end

function _split_js(p::SplitPanel)
    return "{n:$(_str(p.name)),note:$(_str(p.note))," *
           "ax:[$(_str(p.axes[1])),$(_str(p.axes[2])),$(_str(p.axes[3]))]," *
           "o:[" * join((_orbit_js(o) for o in p.orbits), ",") * "]," *
           "ar:[" * join((_arrow_js(a) for a in p.arrows), ",") * "]," *
           "t:{xl:$(_str(p.xlabel)),yl:$(_str(p.ylabel)),eq:0," *
           "s:[" * join((_series_js(s) for s in p.series), ",") * "]}}"
end

# ---------------------------------------------------------------------------------
# Shared page furniture.  Colours follow the site's dark palette (--bg #0a0a0f,
# --acc #9558B2), so a figure embedded in an <iframe> is indistinguishable from the page
# around it.

function _base_css()
    """
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
    #meta { margin-left:auto; color:var(--ink3); font-size:11px;
      font-family:ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing:0.04em; }
    .stage { position:relative; min-height:0;
      border:1px solid var(--hair); border-radius:6px; background:#07070b; overflow:hidden; }
    .stage svg { display:block; width:100%; height:100%; }
    #legend { display:flex; gap:14px; color:var(--ink3); font-size:11px;
      align-items:center; flex-wrap:wrap; }
    #legend i { display:inline-block; width:14px; height:0; vertical-align:3px;
      border-top:2px solid currentColor; margin-right:6px; }
    #note { color:var(--ink3); font-size:11.5px; min-height:1.3em; }
    .tip { position:absolute; pointer-events:none; opacity:0; transition:opacity .1s;
      background:#14141c; border:1px solid var(--hair); border-radius:5px;
      padding:6px 9px; font-size:11.5px; color:var(--ink); white-space:pre;
      font-family:ui-monospace, SFMono-Regular, Menlo, monospace; z-index:5; }
    """
end

# The renderer core, shared verbatim by both page templates.
function _base_js()
    """
    // Julia's brand colours, cycled over the series of a panel.
    const JULIA = ["#9558B2", "#389826", "#CB3C33", "#4063D8"];
    const colour = k => JULIA[((k % JULIA.length) + JULIA.length) % JULIA.length];
    const NS = "http://www.w3.org/2000/svg";

    // Blend a hex colour towards the tooltip background, so a readout can be tinted by
    // the curve it belongs to: `t` near 0 is almost the background, near 1 the colour.
    const hex2rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
    const mix = (h, t) => {
      const [r, g, b] = hex2rgb(h), bg = [20, 20, 28];
      return "rgb(" + [r, g, b].map((v, i) => Math.round(bg[i] + (v - bg[i]) * t)).join(",") + ")";
    };

    const fmt = v => {
      if (Math.abs(v) < 1e-12) return "0";
      const a = Math.abs(v);
      if (a >= 1e4 || a < 1e-3) return v.toExponential(1).replace("e+", "e");
      return String(Math.round(v * 1e6) / 1e6);
    };

    function el(tag, attrs, text) {
      const e = document.createElementNS(NS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      if (text !== undefined) e.textContent = text;
      return e;
    }

    // Nice round tick positions, at most `n` of them.
    function ticks(lo, hi, n) {
      const span = hi - lo;
      if (!(span > 0)) return [lo];
      const mag = Math.pow(10, Math.floor(Math.log10(span / n)));
      const norm = (span / n) / mag;
      const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
      const out = [];
      for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + step * 1e-9; t += step)
        out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
      return out;
    }

    function extent2(series) {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const s of series) for (let i = 0; i < s.x.length; i++) {
        const a = s.x[i], b = s.y[i];
        if (a === null || b === null) continue;
        if (a < x0) x0 = a; if (a > x1) x1 = a;
        if (b < y0) y0 = b; if (b > y1) y1 = b;
      }
      if (!isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
      const px = (x1 - x0) || 1, py = (y1 - y0) || 1;
      return [x0 - 0.04 * px, x1 + 0.04 * px, y0 - 0.07 * py, y1 + 0.07 * py];
    }

    // ── 2-D renderer.  Draws into `host` and records the mapping on it, so the hover
    // handler attached once at set-up can read back the current view.
    function draw2D(host, p) {
      // Stacked rows share one time axis, so only the bottom row spends height on tick
      // labels; the rest give it back to the plot.  `p.ml` and `p.xdom`, when supplied,
      // pin the left margin and the horizontal domain across a stack of rows, so the
      // plot rectangles line up exactly however different the vertical scales are.
      const W = host.clientWidth, H = host.clientHeight;
      const M = { l: p.ml || 58, r:14, t:10, b: p.xl ? 34 : 12 };
      host.querySelectorAll("svg").forEach(n => n.remove());
      // preserveAspectRatio="none": every coordinate below is already in pixels of this
      // element, so the viewBox must map to it one-to-one.  The default would letterbox
      // the drawing the moment the viewBox and the element disagreed on aspect ratio,
      // silently narrowing the plot.
      const svg = el("svg", { viewBox: `0 0 \${W} \${H}`, preserveAspectRatio:"none" });

      let [x0, x1, y0, y1] = extent2(p.s);
      if (p.xdom) { x0 = p.xdom[0]; x1 = p.xdom[1]; }
      const iw = W - M.l - M.r, ih = H - M.t - M.b;
      if (p.eq) {                    // one data unit = the same pixel count on both axes
        const want = iw / ih, have = (x1 - x0) / (y1 - y0);
        if (have < want) { const c = (x0 + x1) / 2, h = (y1 - y0) * want / 2; x0 = c - h; x1 = c + h; }
        else             { const c = (y0 + y1) / 2, h = (x1 - x0) / want / 2; y0 = c - h; y1 = c + h; }
      }
      const X = v => M.l + (v - x0) / (x1 - x0) * iw;
      const Y = v => M.t + (1 - (v - y0) / (y1 - y0)) * ih;
      host._view = { p, X, Y };
      host._surf = null;      // a 2-D panel has no surface to interrogate

      for (const t of ticks(x0, x1, 6)) {
        svg.appendChild(el("line", { x1:X(t), x2:X(t), y1:M.t, y2:M.t + ih,
          stroke:"#1c1c24", "stroke-width":1 }));
        if (p.xl) svg.appendChild(el("text", { x:X(t), y:M.t + ih + 15, fill:"#6e6e7e",
          "font-size":10.5, "text-anchor":"middle" }, fmt(t)));
      }
      for (const t of ticks(y0, y1, ih < 110 ? 3 : 6)) {
        svg.appendChild(el("line", { x1:M.l, x2:M.l + iw, y1:Y(t), y2:Y(t),
          stroke:"#1c1c24", "stroke-width":1 }));
        svg.appendChild(el("text", { x:M.l - 8, y:Y(t) + 3.5, fill:"#6e6e7e",
          "font-size":10.5, "text-anchor":"end" }, fmt(t)));
      }
      // Axes through the origin whenever it is in view — the sign structure of a
      // restoring force is the whole point of half these figures.
      if (y0 < 0 && y1 > 0) svg.appendChild(el("line", { x1:M.l, x2:M.l + iw, y1:Y(0),
        y2:Y(0), stroke:"#33333f", "stroke-width":1 }));
      if (x0 < 0 && x1 > 0) svg.appendChild(el("line", { x1:X(0), x2:X(0), y1:M.t,
        y2:M.t + ih, stroke:"#33333f", "stroke-width":1 }));

      svg.appendChild(el("text", { x:M.l + iw / 2, y:H - 6, fill:"#a0a0b0",
        "font-size":11, "text-anchor":"middle" }, p.xl));
      svg.appendChild(el("text", { x:13, y:M.t + ih / 2, fill:"#a0a0b0", "font-size":11,
        "text-anchor":"middle", transform:`rotate(-90 13 \${M.t + ih / 2})` }, p.yl));

      for (const s of p.s) {
        let d = "", pen = false;
        for (let i = 0; i < s.x.length; i++) {
          if (s.x[i] === null || s.y[i] === null) { pen = false; continue; }
          d += (pen ? "L" : "M") + X(s.x[i]).toFixed(2) + " " + Y(s.y[i]).toFixed(2) + " ";
          pen = true;
        }
        svg.appendChild(el("path", { d, fill:"none", stroke:colour(s.c),
          "stroke-width":(s.d ? 1.4 : 1.8) * (s.w || 1), "stroke-linejoin":"round",
          "stroke-linecap":"round", "stroke-dasharray":s.d ? "5 4" : "",
          opacity:s.d ? 0.75 : 1 }));
      }
      host.insertBefore(svg, host.firstChild);
    }

    // Hover: nearest sample across every series of the host's current view.
    function attachHover2D(host, tip) {
      host.addEventListener("pointermove", ev => {
        const v = host._view; if (!v) return;
        const r = host.getBoundingClientRect();
        const mx = ev.clientX - r.left, my = ev.clientY - r.top;
        let best = null, bd = 400;
        for (const s of v.p.s) for (let i = 0; i < s.x.length; i++) {
          if (s.x[i] === null || s.y[i] === null) continue;
          const dx = v.X(s.x[i]) - mx, dy = v.Y(s.y[i]) - my, d2 = dx * dx + dy * dy;
          if (d2 < bd) { bd = d2; best = { s, i }; }
        }
        if (!best) { tip.style.opacity = 0; return; }
        const s = best.s, i = best.i;
        // `xl` is blanked on every row of a stack but the last, so that only the bottom
        // row draws tick labels.  The readout still needs the axis's name, which is what
        // `xname` carries — otherwise a time plot would report "x = 12.3".
        const xn = v.p.xname || v.p.xl || "x";
        // A one-curve row already names itself on the y line, so the leading label would
        // just repeat it; keep it only where there is a curve to disambiguate.
        const head = (v.p.s.length > 1 || v.p.yl !== s.l) ? s.l + "\\n" : "";
        tip.textContent = `\${head}\${xn} = \${fmt(s.x[i])}\n\${v.p.yl || "y"} = \${fmt(s.y[i])}`;
        tip.style.opacity = 1;
        // Tint the box with the hovered curve's colour, so a readout over a panel of
        // several curves says which one it belongs to without being read.
        const col = colour(s.c);
        tip.style.background = mix(col, 0.18);
        tip.style.borderColor = mix(col, 0.62);
        const tw = tip.offsetWidth, th = tip.offsetHeight;
        tip.style.left = Math.min(Math.max(4, v.X(s.x[i]) + 12), host.clientWidth - tw - 4) + "px";
        tip.style.top = Math.min(Math.max(4, v.Y(s.y[i]) - th - 10), host.clientHeight - th - 4) + "px";
      });
      host.addEventListener("pointerleave", () => { tip.style.opacity = 0; });
    }

    // ── 3-D framing: orthographic projection shared by the wireframe and surface
    // renderers.  The frame is fitted to the projected corners of the data box rather than
    // to the content, so the view does not breathe as the camera turns.  Each axis is
    // normalised by its own extent — these are different physical quantities, and a common
    // scale would flatten whichever has the smaller range.
    function frame3D(host, lo, hi, cam, axes) {
      const W = host.clientWidth, H = host.clientHeight;
      host.querySelectorAll("svg").forEach(n => n.remove());
      const svg = el("svg", { viewBox: `0 0 \${W} \${H}`, preserveAspectRatio:"none" });
      const ctr = [0, 1, 2].map(k => (lo[k] + hi[k]) / 2);
      const ext = [0, 1, 2].map(k => (hi[k] - lo[k]) || 1);
      const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
      const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
      const nrm = (a, b, c) =>
        [(a - ctr[0]) / ext[0], (b - ctr[1]) / ext[1], (c - ctr[2]) / ext[2]];
      const proj = (a, b, c) => {
        const [X, Y, Z] = nrm(a, b, c);
        return [-X * sy + Y * cy, -(X * cy + Y * sy) * sp + Z * cp];
      };
      // Depth out of the screen, for painter's-algorithm sorting: the third axis of the
      // orthonormal frame whose first two are the screen right and up directions.
      const depth = (a, b, c) => {
        const [X, Y, Z] = nrm(a, b, c);
        return (X * cy + Y * sy) * cp + Z * sp;
      };

      const corners = [];
      for (const i of [0, 1]) for (const j of [0, 1]) for (const k of [0, 1])
        corners.push([i ? hi[0] : lo[0], j ? hi[1] : lo[1], k ? hi[2] : lo[2]]);
      const pc = corners.map(c => proj(c[0], c[1], c[2]));
      const bx0 = Math.min(...pc.map(p => p[0])), bx1 = Math.max(...pc.map(p => p[0]));
      const by0 = Math.min(...pc.map(p => p[1])), by1 = Math.max(...pc.map(p => p[1]));
      const m = 44;
      const sc = Math.min((W - 2 * m) / ((bx1 - bx0) || 1), (H - 2 * m) / ((by1 - by0) || 1));
      const SX = v => W / 2 + (v[0] - (bx0 + bx1) / 2) * sc;
      const SY = v => H / 2 - (v[1] - (by0 + by1) / 2) * sc;
      const P = (a, b, c) => { const v = proj(a, b, c); return [SX(v), SY(v)]; };

      for (let a = 0; a < 8; a++) for (let b = a + 1; b < 8; b++) {
        let diff = 0;
        for (let k = 0; k < 3; k++) if (corners[a][k] !== corners[b][k]) diff++;
        if (diff !== 1) continue;
        const p1 = P(...corners[a]), p2 = P(...corners[b]);
        svg.appendChild(el("line", { x1:p1[0], y1:p1[1], x2:p2[0], y2:p2[1],
          stroke:"#1c1c26", "stroke-width":1 }));
      }

      // Axis names at the midpoint of each edge leaving the low corner, pushed away from
      // the box centre so they never sit on the frame, plus the range at the two ends.
      const base = [lo[0], lo[1], lo[2]];
      const cen = P(ctr[0], ctr[1], ctr[2]);
      const drawLabels = () => {
        [0, 1, 2].forEach(k => {
          const tipPt = base.slice();
          tipPt[k] = hi[k];
          const p1 = P(...base), p2 = P(...tipPt);
          svg.appendChild(el("line", { x1:p1[0], y1:p1[1], x2:p2[0], y2:p2[1],
            stroke:"#43434f", "stroke-width":1.3 }));
          const mid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
          let ox = mid[0] - cen[0], oy = mid[1] - cen[1];
          const L = Math.hypot(ox, oy) || 1;
          ox = ox / L * 26; oy = oy / L * 26;
          svg.appendChild(el("text", { x:mid[0] + ox, y:mid[1] + oy + 4, fill:"#c8c8d4",
            "font-size":12, "text-anchor":"middle",
            "font-family":"ui-monospace, Menlo, monospace" }, axes[k]));
          [[p1, lo[k]], [p2, hi[k]]].forEach(([q, val]) => {
            let qx = q[0] - cen[0], qy = q[1] - cen[1];
            const M2 = Math.hypot(qx, qy) || 1;
            svg.appendChild(el("text", { x:q[0] + qx / M2 * 13, y:q[1] + qy / M2 * 13 + 3.5,
              fill:"#6e6e7e", "font-size":10, "text-anchor":"middle",
              "font-family":"ui-monospace, Menlo, monospace" }, fmt(val)));
          });
        });
      };
      return { svg, P, depth, drawLabels,
        commit: () => host.insertBefore(svg, host.firstChild) };
    }

    // ── 3-D polylines (trajectories), plus any annotation arrows.
    function draw3D(host, curves, axes, cam, markers, arrows) {
      arrows = arrows || [];
      let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      const grow = p => {
        for (let k = 0; k < 3; k++) {
          if (p[k] < lo[k]) lo[k] = p[k];
          if (p[k] > hi[k]) hi[k] = p[k];
        }
      };
      for (const s of curves) for (let i = 0; i < s.x.length; i++)
        grow([s.x[i], s.y[i], s.z[i]]);
      // Arrow endpoints stretch the box too: an annotation reaching past the trajectory
      // would otherwise be drawn outside the frame and silently clipped.
      for (const a of arrows) { grow(a.f); grow(a.t); }
      if (!isFinite(lo[0])) { lo = [0, 0, 0]; hi = [1, 1, 1]; }
      // Nothing here answers a 2-D readout, and a stale one from a previous panel would
      // otherwise keep firing over this drawing.
      host._view = null;
      host._surf = null;
      const F = frame3D(host, lo, hi, cam, axes);
      for (const s of curves) {
        let d = "";
        for (let i = 0; i < s.x.length; i++) {
          const q = F.P(s.x[i], s.y[i], s.z[i]);
          d += (i ? "L" : "M") + q[0].toFixed(2) + " " + q[1].toFixed(2) + " ";
        }
        F.svg.appendChild(el("path", { d, fill:"none", stroke:colour(s.c),
          "stroke-width":(s.d ? 1.2 : 1.5) * (s.w || 1), "stroke-linejoin":"round",
          "stroke-linecap":"round", "stroke-dasharray":s.d ? "5 4" : "",
          opacity:s.d ? 0.7 : 0.92 }));
        if (markers) {
          const q0 = F.P(s.x[0], s.y[0], s.z[0]);
          const q1 = F.P(s.x[s.x.length - 1], s.y[s.y.length - 1], s.z[s.z.length - 1]);
          F.svg.appendChild(el("circle", { cx:q0[0], cy:q0[1], r:3, fill:"none",
            stroke:colour(s.c), "stroke-width":1.4 }));
          F.svg.appendChild(el("circle", { cx:q1[0], cy:q1[1], r:3, fill:colour(s.c) }));
        }
      }
      // Arrows last, over the trajectory.  The head is built in screen space from the
      // projected direction: a 3-D cone would need its own depth sort against the polyline
      // for no gain, since the arrow is an annotation rather than part of the orbit.
      for (const a of arrows) {
        const p0 = F.P(a.f[0], a.f[1], a.f[2]);
        const p1 = F.P(a.t[0], a.t[1], a.t[2]);
        const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
        const L = Math.hypot(dx, dy);
        if (L < 1e-6) continue;                       // degenerate on this projection
        const ux = dx / L, uy = dy / L, head = Math.min(13, L * 0.3);
        const col = colour(a.c);
        F.svg.appendChild(el("circle", { cx:p0[0], cy:p0[1], r:2.6, fill:col }));
        F.svg.appendChild(el("line", { x1:p0[0], y1:p0[1],
          x2:p1[0] - ux * head * 0.85, y2:p1[1] - uy * head * 0.85,
          stroke:col, "stroke-width":2, "stroke-linecap":"round" }));
        F.svg.appendChild(el("polygon", { points:
            [p1[0], p1[1],
             p1[0] - head * ux + head * 0.42 * uy, p1[1] - head * uy - head * 0.42 * ux,
             p1[0] - head * ux - head * 0.42 * uy, p1[1] - head * uy + head * 0.42 * ux]
            .map(v => v.toFixed(2)).join(" "), fill:col }));
        if (a.l) {
          // Push the label past the tip along the arrow, so it never sits under the head.
          F.svg.appendChild(el("text", { x:p1[0] + ux * 14, y:p1[1] + uy * 14 + 4,
            fill:col, "font-size":12, "text-anchor":"middle",
            "font-family":"ui-monospace, Menlo, monospace" }, a.l));
        }
      }
      F.drawLabels();
      F.commit();
    }

    // ── 3-D surface, painter's algorithm.  Every cell of the grid becomes a filled quad;
    // sorting them far-to-near before drawing is what gives the occlusion a wireframe
    // cannot have.  Colour is diverging in the height, so the sign of the force is legible,
    // and a Lambert term off the facet normal keeps the corrugation readable.
    function drawSurface3D(host, sf, line, axes, cam) {
      host._view = null;                  // no 2-D readout over a surface
      const nx = sf.x.length, ny = sf.y.length;
      let zlo = Infinity, zhi = -Infinity;
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
        const v = sf.z[i][j];
        if (v < zlo) zlo = v;
        if (v > zhi) zhi = v;
      }
      const lo = [sf.x[0], sf.y[0], zlo], hi = [sf.x[nx - 1], sf.y[ny - 1], zhi];
      const F = frame3D(host, lo, hi, cam, axes);
      const zmax = Math.max(Math.abs(zlo), Math.abs(zhi)) || 1;

      const MID = [42, 42, 54], NEG = [64, 99, 216], POS = [203, 60, 51];
      const tint = (t, l) => {                     // t in [-1,1], l a lighting factor
        const c = t < 0 ? MID.map((m, k) => m + (NEG[k] - m) * Math.min(1, -t))
                        : MID.map((m, k) => m + (POS[k] - m) * Math.min(1, t));
        return "rgb(" + c.map(v => Math.max(0, Math.min(255, Math.round(v * l)))).join(",") + ")";
      };

      const quads = [];
      for (let i = 0; i < nx - 1; i++) for (let j = 0; j < ny - 1; j++) {
        const pts = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]]
          .map(([a, b]) => [sf.x[a], sf.y[b], sf.z[a][b]]);
        const zc = (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]) / 4;
        const dep = pts.reduce((s, p) => s + F.depth(p[0], p[1], p[2]), 0) / 4;
        // Facet normal in normalised units, for a cheap Lambert shade.
        const ux = [(pts[1][0] - pts[0][0]) / (hi[0] - lo[0] || 1), 0,
                    (pts[1][2] - pts[0][2]) / (2 * zmax)];
        const uy = [0, (pts[3][1] - pts[0][1]) / (hi[1] - lo[1] || 1),
                    (pts[3][2] - pts[0][2]) / (2 * zmax)];
        const n = [ux[1] * uy[2] - ux[2] * uy[1],
                   ux[2] * uy[0] - ux[0] * uy[2],
                   ux[0] * uy[1] - ux[1] * uy[0]];
        const nl = Math.hypot(n[0], n[1], n[2]) || 1;
        const lam = Math.abs(n[2] / nl);
        quads.push({ pts, zc, dep, l: 0.72 + 0.5 * lam });
      }
      quads.sort((a, b) => a.dep - b.dep);          // far first, near last

      for (const q of quads) {
        const d = q.pts.map((p, i) => {
          const s = F.P(p[0], p[1], p[2]);
          return (i ? "L" : "M") + s[0].toFixed(1) + " " + s[1].toFixed(1);
        }).join(" ") + " Z";
        const col = tint(q.zc / zmax, q.l);
        F.svg.appendChild(el("path", { d, fill:col, stroke:col,
          "stroke-width":0.6, "shape-rendering":"crispEdges" }));
      }
      // The swept line is drawn after the surface and never depth-sorted against it: it
      // lies on the surface by construction, offset by a hair, so anything that hid part
      // of it would only be hiding the sheet it is painted on.
      const hl = el("path", { fill:"none", stroke:"#f2f2f7", "stroke-width":3,
        "stroke-linejoin":"round", "stroke-linecap":"round" });
      const hlt = el("text", { fill:"#f2f2f7", "font-size":11.5, "text-anchor":"middle",
        "font-family":"ui-monospace, Menlo, monospace" });
      F.svg.appendChild(hl);
      F.svg.appendChild(hlt);
      host._surf = { line, P: F.P, hl, hlt, name: axes[1],
        y0: sf.y[0], y1: sf.y[ny - 1],
        t: (host._surf && host._surf.line === line) ? host._surf.t : sf.y[0] };
      if (line) drawSweptLine(host); else { hl.setAttribute("opacity", 0); hlt.setAttribute("opacity", 0); }
      F.drawLabels();
      F.commit();
    }

    // The line has a closed form, so it needs no grid and no interpolation:
    //     F(xᵢ, t) = amplitude[i]·cos(ω t + φ) + offset
    // For a straight line two endpoints are exact, whatever the surface's resolution.
    function drawSweptLine(host) {
      const S = host._surf; if (!S || !S.line) return;
      const L = S.line, n = L.x.length;
      const c = Math.cos(L.w * S.t + L.p);
      let d = "", last = null;
      for (let i = 0; i < n; i++) {
        const q = S.P(L.x[i], S.t, L.a[i] * c + L.o);
        d += (i ? "L" : "M") + q[0].toFixed(1) + " " + q[1].toFixed(1) + " ";
        last = q;
      }
      S.hl.setAttribute("d", d);
      S.hlt.textContent = S.name + " = " + fmt(S.t);
      S.hlt.setAttribute("x", last[0]);
      S.hlt.setAttribute("y", last[1] - 10);
    }

    // Drag the host to orbit the camera; `redraw` is called on every move.
    function attachOrbit(host, cam, redraw) {
      let drag = null;
      host.addEventListener("pointerdown", ev => {
        drag = { x: ev.clientX, y: ev.clientY };
        host.classList.add("drag");
        host.setPointerCapture(ev.pointerId);
      });
      host.addEventListener("pointermove", ev => {
        if (!drag) return;
        cam.yaw += (ev.clientX - drag.x) * 0.008;
        cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + (ev.clientY - drag.y) * 0.006));
        drag = { x: ev.clientX, y: ev.clientY };
        redraw();
      });
      const end = () => { drag = null; host.classList.remove("drag"); };
      host.addEventListener("pointerup", end);
      host.addEventListener("pointercancel", end);
    }
    """
end

function _charts_html(title, caption, data)
    """
    <!DOCTYPE html>
    <html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>$(title)</title>
    <style>
    $(_base_css())
    #stage { flex:1; }
    #stage.rot svg { cursor:grab; touch-action:none; }
    #stage.rot.drag svg { cursor:grabbing; }
    </style></head><body>
    <div id="wrap">
      <div id="bar"></div>
      <div id="stage" class="stage"><div id="tip" class="tip"></div></div>
      <div id="legend"></div>
      <div id="note">$(caption)</div>
    </div>
    <script>
    const PANELS = $(data);
    $(_base_js())

    const bar = document.getElementById("bar");
    const stage = document.getElementById("stage");
    const legend = document.getElementById("legend");
    const note = document.getElementById("note");
    const tip = document.getElementById("tip");
    const baseCaption = note.textContent;
    let cur = 0;
    const cam = { yaw: -0.62, pitch: 0.30 };
    const is3D = p => p.sf || (p.o && p.o.length > 0);
    const redraw3D = p => p.sf ? drawSurface3D(stage, p.sf, p.ln, p.ax, cam)
                               : draw3D(stage, p.o, p.ax, cam, false, p.ar);

    function render() {
      const p = PANELS[cur];
      if (is3D(p)) {
        redraw3D(p);
        stage.classList.add("rot");
        tip.style.opacity = 0;      // a readout left over from a 2-D panel
      } else {
        draw2D(stage, p);
        stage.classList.remove("rot");
      }
      // A surface names itself through its axes; a wireframe of many curves would
      // otherwise repeat one label per curve, so collapse duplicates.
      const shown = p.sf ? [] : (is3D(p) ? p.o : p.s);
      const seen = new Set();
      legend.innerHTML = shown.filter(s => !seen.has(s.l) && seen.add(s.l)).map(s =>
        `<span style="color:\${colour(s.c)}"><i></i><span style="color:#a0a0b0">\${s.l}</span></span>`)
        .join("");
      note.textContent = p.note || baseCaption;
    }

    attachHover2D(stage, tip);
    attachOrbit(stage, cam, () => {
      if (is3D(PANELS[cur])) redraw3D(PANELS[cur]);
    });

    // The surface's line sweeps on its own, at a constant speed — one pass over the whole
    // range every SWEEP seconds.  Only the line is rebuilt per frame; the quads are static
    // and merely get the line re-slotted among them, so this stays cheap.
    const SWEEP = 14;
    let last = null;
    function tick(now) {
      const S = stage._surf;
      if (S) {
        if (last !== null && S.line) {
          const step = (S.y1 - S.y0) * (now - last) / (SWEEP * 1000);
          S.t += step;
          if (S.t > S.y1) S.t = S.y0 + (S.t - S.y1);
          drawSweptLine(stage);
        }
        last = now;
      } else {
        last = null;
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    PANELS.forEach((p, k) => {
      const b = document.createElement("button");
      b.textContent = p.n;
      b.className = k === 0 ? "on" : "";
      b.onclick = () => {
        cur = k;
        [...bar.querySelectorAll("button")].forEach((o, j) => o.className = j === k ? "on" : "");
        render();
      };
      bar.appendChild(b);
    });
    if (PANELS.length > 1) {
      const m = document.createElement("span");
      m.id = "meta";
      m.textContent = PANELS.length + " panels";
      bar.appendChild(m);
    }
    addEventListener("resize", render);
    render();
    </script></body></html>
    """
end

function _pairs_html(title, caption, data)
    """
    <!DOCTYPE html>
    <html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>$(title)</title>
    <style>
    $(_base_css())
    #panes { display:flex; flex:1; min-height:0; gap:10px; }
    #panes .stage { flex:1; min-width:0; }
    </style></head><body>
    <div id="wrap">
      <div id="bar"></div>
      <div id="panes">
        <div id="stageT" class="stage"><div id="tipT" class="tip"></div></div>
        <div id="stageP" class="stage"><div id="tipP" class="tip"></div></div>
      </div>
      <div id="legend"></div>
      <div id="note">$(caption)</div>
    </div>
    <script>
    const PANELS = $(data);
    $(_base_js())

    const bar = document.getElementById("bar");
    const sT = document.getElementById("stageT");
    const sP = document.getElementById("stageP");
    const legend = document.getElementById("legend");
    const note = document.getElementById("note");
    const baseCaption = note.textContent;
    let cur = 0;

    function render() {
      const p = PANELS[cur];
      draw2D(sT, p.t);
      draw2D(sP, p.p);
      // The legend names the time pane's curves; the phase pane replots the same signal.
      legend.innerHTML = p.t.s.map(s =>
        `<span style="color:\${colour(s.c)}"><i></i><span style="color:#a0a0b0">\${s.l}</span></span>`)
        .join("");
      note.textContent = p.note || baseCaption;
    }

    attachHover2D(sT, tipT);
    attachHover2D(sP, tipP);
    PANELS.forEach((p, k) => {
      const b = document.createElement("button");
      b.textContent = p.n;
      b.className = k === 0 ? "on" : "";
      b.onclick = () => {
        cur = k;
        [...bar.querySelectorAll("button")].forEach((o, j) => o.className = j === k ? "on" : "");
        render();
      };
      bar.appendChild(b);
    });
    addEventListener("resize", render);
    render();
    </script></body></html>
    """
end

function _split_html(title, caption, data)
    """
    <!DOCTYPE html>
    <html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>$(title)</title>
    <style>
    $(_base_css())
    #panes { display:flex; flex:1; min-height:0; gap:10px; }
    #stage3d { flex:1.08; min-width:0; }
    /* one stacked plot per coordinate, sharing the time axis on the bottom row */
    #rows { flex:1; min-width:0; min-height:0; display:flex; flex-direction:column; gap:6px; }
    #rows .stage { flex:1; min-height:0; }
    #stage3d svg { cursor:grab; touch-action:none; }
    #stage3d.drag svg { cursor:grabbing; }
    #hint { position:absolute; left:10px; bottom:8px; color:#4e4e5c; font-size:10.5px;
      pointer-events:none; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; }
    </style></head><body>
    <div id="wrap">
      <div id="bar"></div>
      <div id="panes">
        <div id="stage3d" class="stage"><div id="hint">drag to orbit</div></div>
        <div id="rows"></div>
      </div>
      <div id="legend"></div>
      <div id="note">$(caption)</div>
    </div>
    <script>
    const PANELS = $(data);
    $(_base_js())

    const bar = document.getElementById("bar");
    const s3 = document.getElementById("stage3d");
    const rows = document.getElementById("rows");
    const legend = document.getElementById("legend");
    const note = document.getElementById("note");
    const baseCaption = note.textContent;
    let cur = 0;
    // The camera is shared across panels, so switching coordinate systems keeps the
    // viewing angle and the two pictures stay comparable.
    const cam = { yaw: -0.62, pitch: 0.30 };

    function render() {
      const p = PANELS[cur];
      draw3D(s3, p.o, p.ax, cam, true, p.ar);
      // One stacked plot per coordinate: each gets its own vertical scale, which is the
      // point — a shared axis would flatten whichever coordinate has the smaller range.
      rows.innerHTML = "";
      const last = p.t.s.length - 1;
      // One shared time domain and one shared left margin for the whole stack: the rows
      // must be pixel-aligned so a feature at time t sits above the same t in every plot.
      let gx0 = Infinity, gx1 = -Infinity;
      for (const s of p.t.s) for (const v of s.x) {
        if (v === null) continue;
        if (v < gx0) gx0 = v;
        if (v > gx1) gx1 = v;
      }
      if (!isFinite(gx0)) { gx0 = 0; gx1 = 1; }
      const pad = 0.02 * ((gx1 - gx0) || 1);
      const xdom = [gx0 - pad, gx1 + pad];
      // The vertical scales differ per row, so their tick labels differ in width. Size
      // one left margin to the widest label anywhere in the stack: every row then has
      // the same plot rectangle and no row's labels are clipped.
      let wide = 1;
      for (const s of p.t.s) {
        let y0 = Infinity, y1 = -Infinity;
        for (const v of s.y) {
          if (v === null) continue;
          if (v < y0) y0 = v;
          if (v > y1) y1 = v;
        }
        if (!isFinite(y0)) continue;
        const py = (y1 - y0) || 1;
        for (const t of ticks(y0 - 0.07 * py, y1 + 0.07 * py, 6))
          wide = Math.max(wide, fmt(t).length);
      }
      const ml = Math.max(46, Math.min(98, 22 + wide * 6.7));
      // Two passes, and the order matters. Drawing a row straight after appending it
      // would measure it while it is still the only child of a flex column — so it would
      // report the full height, and every later row would shrink it without its viewBox
      // knowing. Create every row first, then draw once they all have their final size.
      const hosts = p.t.s.map(() => {
        const host = document.createElement("div");
        host.className = "stage";
        const tp = document.createElement("div");
        tp.className = "tip";
        host.appendChild(tp);
        rows.appendChild(host);
        attachHover2D(host, tp);
        return host;
      });
      p.t.s.forEach((s, k) => {
        // `xl` only on the bottom row — every row shares the same time axis, so drawing
        // its ticks three times would be noise.  `xname` is set on all of them so the
        // hover readout still names the axis correctly.
        draw2D(hosts[k], { xl: k === last ? p.t.xl : "", xname: p.t.xl,
          yl: s.l, eq: 0, s: [s], xdom, ml });
      });
      legend.innerHTML = p.t.s.map(s =>
        `<span style="color:\${colour(s.c)}"><i></i><span style="color:#a0a0b0">\${s.l}</span></span>`)
        .join("");
      note.textContent = p.note || baseCaption;
    }

    attachOrbit(s3, cam,
      () => draw3D(s3, PANELS[cur].o, PANELS[cur].ax, cam, true, PANELS[cur].ar));

    // A lone panel needs no switcher — the button would name the only thing on screen and
    // do nothing when pressed.
    if (PANELS.length > 1) PANELS.forEach((p, k) => {
      const b = document.createElement("button");
      b.textContent = p.n;
      b.className = k === 0 ? "on" : "";
      b.onclick = () => {
        cur = k;
        [...bar.querySelectorAll("button")].forEach((o, j) => o.className = j === k ? "on" : "");
        render();
      };
      bar.appendChild(b);
    });
    if (PANELS.length > 1) {
      const m = document.createElement("span");
      m.id = "meta";
      m.textContent = PANELS.length + " coordinate systems";
      bar.appendChild(m);
    }
    addEventListener("resize", render);
    render();
    </script></body></html>
    """
end
