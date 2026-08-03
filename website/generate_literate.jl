# website/generate_literate.jl
# Run from the repo root:  julia website/generate_literate.jl
#
# Renders every example's main.jl as a STANDARDIZED Literate page —
# a straightforward single-column exposition (narrative from `#` comments,
# code in between), written to website/tutorials/literate/<id>.html.
# The custom tutorial pages link to these via a "Literate script" button;
# each Literate page links back to its custom page and the GitHub folder.
#
# Rendering is `execute = false` by design: generation is instant and needs no
# FEM dependencies; the numbers live in the examples' READMEs/validate.jl and
# on the custom pages. Rerun this script whenever an example's main.jl changes.

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Literate, Markdown

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FERRITE_ROOT = let
	cands = [joinpath(ROOT, "..", "MORFEFerrite", "MORFEFerrite.jl"),
		joinpath(ROOT, "..", "MORFEFerrite.jl")]
	i = findfirst(isdir, cands)
	i === nothing ? error("MORFEFerrite.jl checkout not found next to MORFE.jl") :
	normpath(cands[i])
end
const OUTDIR = joinpath(@__DIR__, "tutorials", "literate")
const GH_MORFE = "https://github.com/MORFEproject/MORFE.jl"
const GH_FERRITE = "https://github.com/MORFEproject/MORFEFerrite.jl"

# id, title, source main.jl, GitHub folder, custom tutorial page (site-relative
# to tutorials/) — `nothing` while the custom page does not exist yet.
const EXAMPLES = [
	(id = "full_order_model", title = "Building a full-order model",
		src = joinpath(ROOT, "examples", "internals", "full_order_model", "main.jl"),
		gh = "$GH_MORFE/tree/main/examples/internals/full_order_model",
		custom = "full_order_model.html"),
	(id = "multiindex_sets", title = "Constructing MultiindexSets",
		src = joinpath(ROOT, "examples", "internals", "multiindex_sets", "main.jl"),
		gh = "$GH_MORFE/tree/main/examples/internals/multiindex_sets",
		custom = "multiindex_sets.html"),
	(id = "01_clamped_beam_ferrite", title = "Clamped-clamped beam — StructuralSVK",
		src = joinpath(FERRITE_ROOT, "examples", "01_clamped_beam_ferrite", "main.jl"),
		gh = "$GH_FERRITE/tree/main/examples/01_clamped_beam_ferrite", custom = "structural_svk.html"),
	(id = "02_clamped_beam_gridap", title = "Clamped-clamped beam — Gridap backend",
		src = joinpath(ROOT, "examples", "02_clamped_beam_gridap", "main.jl"),
		gh = "$GH_MORFE/tree/main/examples/02_clamped_beam_gridap", custom = nothing),
	(id = "03_arch_comsol_wedge", title = "COMSOL arch wedge — StructuralSVK",
		src = joinpath(FERRITE_ROOT, "examples", "03_arch_comsol_wedge", "main.jl"),
		gh = "$GH_FERRITE/tree/main/examples/03_arch_comsol_wedge", custom = "structural_svk.html"),
	(id = "04_parametric_clamped_beam", title = "Two-parameter beam — ParametricStructural",
		src = joinpath(FERRITE_ROOT, "examples", "04_parametric_clamped_beam", "main.jl"),
		gh = "$GH_FERRITE/tree/main/examples/04_parametric_clamped_beam", custom = "parametric.html"),
	(id = "05_karman_vortex_street", title = "Kármán vortex street — FluidNavierStokes",
		src = joinpath(FERRITE_ROOT, "examples", "05_karman_vortex_street", "main.jl"),
		gh = "$GH_FERRITE/tree/main/examples/05_karman_vortex_street", custom = "karman.html"),
	(id = "06_dielectric_elastomer_actuator", title = "Dielectric elastomer actuator",
		src = joinpath(ROOT, "examples", "06_dielectric_elastomer_actuator", "main.jl"),
		gh = "$GH_MORFE/tree/main/examples/06_dielectric_elastomer_actuator", custom = nothing),
	(id = "07_parametric_arch", title = "Parametric sinusoidal arch — ParametricStructural",
		src = joinpath(FERRITE_ROOT, "examples", "07_parametric_arch", "main.jl"),
		gh = "$GH_FERRITE/tree/main/examples/07_parametric_arch", custom = "parametric.html"),
	(id = "08_mems_micromirror", title = "MEMS micromirror — StructuralSVK",
		src = joinpath(FERRITE_ROOT, "examples", "08_mems_micromirror", "main.jl"),
		gh = "$GH_FERRITE/tree/main/examples/08_mems_micromirror", custom = "micromirror.html"),
]

# ── Preprocessing: a LEADING triple-quoted docstring becomes `#` markdown, so
# legacy drivers render as prose + code rather than one giant code block.
function docstring_to_comments(code::String)
	m = match(r"\A\s*\"\"\"\n?(.*?)\"\"\"\s*\n"s, code)
	m === nothing && return code
	body = m.captures[1]
	prose = join(("# " * rstrip(l) for l in split(body, '\n')), "\n")
	return prose * "\n\n" * code[(m.offset+lastindex(m.match)):end]
end

# ── Standardized page shell (self-contained, matches the site's dark look).
function page_html(title, gh, custom, article)
	backlinks = String[]
	custom !== nothing && push!(backlinks,
		"<a class=\"btn\" href=\"../$custom\">Custom tutorial page</a>")
	push!(backlinks, "<a class=\"btn\" href=\"$gh\" target=\"_blank\" rel=\"noopener\">Example folder on GitHub ↗</a>")
	push!(backlinks, "<a class=\"btn\" href=\"../index.html\">All tutorials</a>")
	"""
	<!DOCTYPE html>
	<html lang="en"><head><meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>$title · MORFE Literate</title>
	<style>
	:root { --bg:#0a0a0f; --ink:#e8e8ee; --ink2:#a0a0b0; --hair:#26262f; --acc:#9558B2; }
	* { box-sizing:border-box; }
	body { margin:0; background:var(--bg); color:var(--ink);
	  font:16px/1.65 -apple-system, "Segoe UI", Roboto, sans-serif; }
	main { max-width: 840px; margin: 0 auto; padding: 40px 20px 80px; }
	h1 { font-size: 30px; margin: 8px 0 4px; }
	h2 { font-size: 21px; margin-top: 2.2em; border-bottom: 1px solid var(--hair); padding-bottom: 6px; }
	p, li { color: var(--ink2); }
	strong, code { color: var(--ink); }
	a { color: #b78ccf; }
	pre { background:#111118; border:1px solid var(--hair); border-radius:6px;
	  padding:14px 16px; overflow-x:auto; font-size:13.5px; line-height:1.55; }
	code { font-family: "JuliaMono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; }
	.kicker { font-size:11px; letter-spacing:0.14em; text-transform:uppercase; color:var(--acc); }
	.btnrow { display:flex; gap:10px; flex-wrap:wrap; margin:18px 0 30px; }
	.btn { display:inline-block; padding:8px 14px; border:1px solid var(--hair); border-radius:6px;
	  color:var(--ink); text-decoration:none; font-size:13.5px; }
	.btn:hover { border-color: var(--acc); }
	.note { border-left:3px solid var(--acc); background:rgba(149,88,178,0.07);
	  padding:10px 14px; border-radius:0 6px 6px 0; font-size:14px; color:var(--ink2); }
	</style></head><body><main>
	<span class="kicker">MORFE · Literate example</span>
	<h1>$title</h1>
	<div class="btnrow">$(join(backlinks, "\n"))</div>
	<div class="note">Standardized rendering of the example's <code>main.jl</code>
	(narrative comments + code, not executed here). Run it yourself with the
	commands in the script header; expected numbers are asserted by the example's
	<code>validate.jl</code> and quoted in its README.</div>
	$article
	</main></body></html>
	"""
end

mkpath(OUTDIR)
mktempdir() do tmp
	for ex in EXAMPLES
		isfile(ex.src) || (println("  ⚠ skipping $(ex.id): $(ex.src) not found"); continue)
		pre = docstring_to_comments(read(ex.src, String))
		srcfile = joinpath(tmp, ex.id * ".jl")
		write(srcfile, pre)
		Literate.markdown(srcfile, tmp; flavor = Literate.CommonMarkFlavor(),
			execute = false, credit = false)
		md = read(joinpath(tmp, ex.id * ".md"), String)
		article = string(Markdown.html(Markdown.parse(md)))
		out = joinpath(OUTDIR, ex.id * ".html")
		write(out, page_html(ex.title, ex.gh, ex.custom, article))
		println("  ✓ $(ex.id) → tutorials/literate/$(ex.id).html")
	end
end
println("Done.")
