# website/generate_documentation.jl
# Run from repo root:  julia --project website/generate_documentation.jl
# Loads MORFE, extracts every documented symbol via Base.Docs,
# writes website/documentation.html in the site's own style.

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MORFE, Markdown, Dates

const REPO_ROOT   = normpath(joinpath(@__DIR__, ".."))
const GITHUB_BASE = "https://github.com/MORFEproject/MORFE.jl/blob/main"

# ── Module list (mirrors docs/make.jl) ────────────────────────────────────────
const MODS = [
	(MORFE.Multiindices, "Multiindices"),
	(MORFE.Polynomials, "Polynomials"),
	(MORFE.MultilinearMaps, "MultilinearMaps"),
	(MORFE.ExternalSystems, "ExternalSystems"),
	(MORFE.FullOrderModel, "FullOrderModel"),
	(MORFE.Eigenproblems, "Eigenproblems"),
	(MORFE.Eigensolvers, "Eigensolvers"),
	(MORFE.Realification, "Realification"),
	(MORFE.Resonance, "Resonance"),
	(MORFE.InvarianceEquation, "InvarianceEquation"),
	(MORFE.MasterModeOrthogonality, "MasterModeOrthogonality"),
	(MORFE.ParametrisationObjects, "ParametrisationObjects"),
	(MORFE.MultilinearTerms, "MultilinearTerms"),
	(MORFE.LowerOrderCouplings, "LowerOrderCouplings"),
	(MORFE.CohomologicalEquations, "CohomologicalEquations"),
	(MORFE.ParametrisationMethod, "ParametrisationMethod"),
	(MORFE.FEMUtility, "FEMUtility"),
	(MORFE.BifurcationKitInterface, "BifurcationKitInterface"),
	(MORFE.InvarianceError, "InvarianceError"),
	# ParaviewExport moved to MORFEFerrite.jl (Ferrite/VTK export lives there now).
]

# ── Extraction helpers ────────────────────────────────────────────────────────
function has_own_doc(mod, sym)
	binding = Base.Docs.Binding(mod, sym)
	meta = try
		Base.Docs.meta(mod)
	catch
		;
		return false
	end
	haskey(meta, binding) && !isempty(meta[binding].docs)
end

# Docstrings attached to one binding, in the order they were written.
#
# `MultiDoc.docs` is an IdDict, so iterating it yields an arbitrary order; a symbol
# with several docstrings (a type plus its documented constructors) would get one of
# them picked at random.  `MultiDoc.order` records definition order and, for a type,
# always begins with the `Union{}` signature — the struct's own docstring, ahead of
# any constructor.  Every reader of a binding's documentation goes through here so
# the page can never link to, or lead with, a constructor in place of its type.
function ordered_docstrs(md)
	isempty(md.order) && return collect(values(md.docs))
	out = Any[]
	for sig in md.order
		d = get(md.docs, sig, nothing)
		d === nothing || push!(out, d)
	end
	return isempty(out) ? collect(values(md.docs)) : out
end

function item_kind(mod, sym)
	startswith(string(sym), "@") && return "macro"
	try
		obj = getfield(mod, sym)
		obj isa Module && return "skip"
		(obj isa DataType || obj isa UnionAll) && return "type"
		obj isa Function && return "function"
		return "constant"
	catch
		return "skip"
	end
end

function get_signatures(mod, sym, kind)
	kind in ("type", "macro", "constant", "skip") && return String[]
	try
		obj = getfield(mod, sym)
		sigs = String[]
		for m in methods(obj)
			s = replace(string(m), r"\s+(in \S+ at|@)\s+.*$" => "")
			push!(sigs, s)
		end
		return sigs
	catch
		return String[]
	end
end

function get_source_url(mod, sym, kind)
	kind == "skip" && return ""
	try
		# Prefer the docstring's own line number (opening """) so the GitHub
		# view lands at the top of the docstring rather than the function body.
		# `ordered_docstrs` puts the type's own docstring first, so a struct with
		# a documented constructor links to the struct, not the constructor.
		binding = Base.Docs.Binding(mod, sym)
		meta    = Base.Docs.meta(mod)
		if haskey(meta, binding)
			for docstr in ordered_docstrs(meta[binding])
				path = get(docstr.data, :path, nothing)
				line = get(docstr.data, :linenumber, nothing)
				(path === nothing || line === nothing) && continue
				file = expanduser(string(path))
				rel  = relpath(file, REPO_ROOT)
				startswith(rel, "..") && continue
				return "$(GITHUB_BASE)/$(rel)#L$(line)"
			end
		end
		# Fallback: use the first method definition line.
		obj = getfield(mod, sym)
		ms  = methods(obj)
		isempty(ms) && return ""
		# Handles both old "in Mod at path:line" and new "@ Mod path:line" formats.
		mm = match(r"(?:\bat\s+|@ \S+\s+)(\S+):(\d+)\s*$", string(first(ms)))
		mm === nothing && return ""
		file = expanduser(mm[1])
		rel  = relpath(file, REPO_ROOT)
		startswith(rel, "..") && return ""
		return "$(GITHUB_BASE)/$(rel)#L$(mm[2])"
	catch
		return ""
	end
end

function html_escape(s)
	s = replace(s, "&" => "&amp;")
	s = replace(s, "<" => "&lt;")
	s = replace(s, ">" => "&gt;")
	return s
end

# Render one DocStr to HTML via the stable Markdown public API.
# DocStr.text is a SimpleVector of String/Function elements; we join the strings.
# This avoids Base.Docs.parsedoc / Base.Docs.doc(::Binding) which are unstable
# internal APIs that have changed across Julia minor versions.
function _docstr_html(ds)::String
	text = join(t for t in ds.text if t isa AbstractString)
	sprint(show, MIME("text/html"), Markdown.parse(text))
end

function get_doc_html(mod, sym)
	binding = Base.Docs.Binding(mod, sym)
	meta = try
		Base.Docs.meta(mod)
	catch
		return ""
	end
	haskey(meta, binding) || return ""
	multidoc = meta[binding]
	isempty(multidoc.docs) && return ""
	html = mapreduce(_docstr_html, *, ordered_docstrs(multidoc); init = "")
	# Strip the "@ Module /path/to/file.jl:line" source-location paragraphs
	# that Julia's Docs renderer appends to each method docstring.
	html = replace(html, r"<p>@ \S+ [^\n<]*:\d+\s*</p>" => "")
	return html
end

struct Entry
	name::String
	kind::String
	signatures::Vector{String}
	doc_html::String
	source_url::String
end

struct ModData
	label::String
	desc_html::String
	entries::Vector{Entry}
end

function get_module_doc_html(mod)
	try
		binding = Base.Docs.Binding(mod, nameof(mod))
		meta    = Base.Docs.meta(mod)
		haskey(meta, binding) && !isempty(meta[binding].docs) || return ""
		html = mapreduce(_docstr_html, *, ordered_docstrs(meta[binding]); init = "")
		# Strip the tab-indented module name rendered as a leading code block
		html = replace(html, r"^(<div class=\"markdown\">)<pre><code[^>]*>[^<]+</code></pre>\n?" => s"\1")
		html = replace(html, r"<p>@ \S+ [^\n<]*:\d+\s*</p>" => "")
		return html
	catch
		return ""
	end
end

function extract_all()
	result = ModData[]
	for (mod, label) in MODS
		@info "Extracting $label..."
		desc_html = get_module_doc_html(mod)
		entries = Entry[]
		for sym in sort(names(mod, all = true))
			string(sym)[1] == '#' && continue
			has_own_doc(mod, sym) || continue
			kind = item_kind(mod, sym)
			kind == "skip" && continue
			sigs  = get_signatures(mod, sym, kind)
			dhtml = get_doc_html(mod, sym)
			push!(entries, Entry(string(sym), kind, sigs, dhtml, get_source_url(mod, sym, kind)))
		end
		isempty(entries) && continue
		push!(result, ModData(label, desc_html, entries))
		@info "  → $(length(entries)) entries"
	end
	return result
end

# ── Cross-reference resolution ────────────────────────────────────────────────
# Docstrings link with Documenter's `[`name`](@ref)`, which Documenter resolves
# when building docs/. This generator renders the same markdown with plain
# `Markdown.parse`, which knows nothing about `@ref` and emits a literal
# `href="@ref"` — a dead link on every cross-reference. The page already gives
# every entry the anchor `"$(modlabel)-$(name)"` and every module its label as an
# id, so the targets exist; they just need to be looked up after rendering.

# Recover a ref target from the rendered link body: drop the `<code>` wrapper and
# undo the entity escaping Julia's Markdown writer applies (it emits `&#33;` for
# `!`, so `solve_single_monomial&#33;` has to map back to the `!`-bearing anchor).
function ref_target_text(body::AbstractString)
	text = replace(String(body), r"<[^>]*>" => "")
	text = replace(text, r"&#(\d+);" => m -> string(Char(parse(Int, match(r"\d+", m).match))))
	text = replace(text, "&amp;" => "&", "&lt;" => "<", "&gt;" => ">", "&quot;" => "\"")
	return strip(text)
end

# name => [(module label, anchor)]; module labels map to their own heading id.
function build_ref_index(mods::Vector{ModData})
	index = Dict{String, Vector{Tuple{String, String}}}()
	for md in mods
		push!(get!(index, md.label, Tuple{String, String}[]), (md.label, md.label))
		for e in md.entries
			push!(get!(index, e.name, Tuple{String, String}[]), (md.label, "$(md.label)-$(e.name)"))
		end
	end
	return index
end

function lookup_ref(target::AbstractString, modlabel::String, index)
	# `Mod.name` names its module deliberately: honour it or fail, rather than
	# silently linking to a same-named symbol somewhere else.
	if occursin('.', target)
		parts   = rsplit(target, '.'; limit = 2)
		modpart = String(parts[1])
		cands   = get(index, String(parts[2]), nothing)
		cands === nothing && return nothing
		for (lbl, anchor) in cands
			lbl == modpart && return anchor
		end
		return nothing
	end
	cands = get(index, String(target), nothing)
	cands === nothing && return nothing
	# A target in the module being rendered wins over a same-named one elsewhere.
	for (lbl, anchor) in cands
		lbl == modlabel && return anchor
	end
	return first(cands)[2]
end

const REF_LINK_RE = r"<a href=\"@ref(?<target>[^\"]*)\">(?<body>.*?)</a>"s

# Unresolvable refs lose the `<a>` and keep their body: plain text reads better
# than a link that goes nowhere, and the caller warns so the ref gets fixed.
function resolve_refs(html::String, modlabel::String, index, unresolved::Vector{String})
	isempty(html) && return html
	return replace(html, REF_LINK_RE => function (m)
		mm     = match(REF_LINK_RE, m)
		body   = mm[:body]
		target = strip(String(mm[:target]))
		isempty(target) && (target = ref_target_text(body))
		anchor = lookup_ref(target, modlabel, index)
		anchor === nothing || return "<a href=\"#$(anchor)\">$(body)</a>"
		push!(unresolved, String(target))
		return String(body)
	end)
end

function resolve_all_refs(mods::Vector{ModData})
	index      = build_ref_index(mods)
	unresolved = String[]
	resolved   = ModData[]
	for md in mods
		entries = [Entry(
			e.name, e.kind, e.signatures,
			resolve_refs(e.doc_html, md.label, index, unresolved), e.source_url,
		) for e in md.entries]
		desc = resolve_refs(md.desc_html, md.label, index, unresolved)
		push!(resolved, ModData(md.label, desc, entries))
	end
	if isempty(unresolved)
		@info "All @ref cross-references resolved."
	else
		@warn "Unresolved @ref targets — rendered as plain text" targets = sort(unique(unresolved))
	end
	return resolved
end

# ── HTML generation ───────────────────────────────────────────────────────────
badge(kind) = kind == "type" ? "badge t" : kind == "macro" ? "badge m" : kind == "constant" ? "badge c" : "badge f"

function write_entry(io, modlabel, e::Entry)
	anchor = "$(modlabel)-$(e.name)"
	print(io, """<div class="doc doc-$(e.kind)" id="$(anchor)">\n""")
	print(io, """  <div class="doc-sig">\n""")
	print(io, """    <span class="$(badge(e.kind))">$(e.kind)</span>\n""")
	if isempty(e.signatures)
		name_html = isempty(e.source_url) ? html_escape(e.name) :
					"""<a href="$(e.source_url)" target="_blank" rel="noopener">$(html_escape(e.name))</a>"""
		print(io, """    <code>$(name_html)</code>\n""")
	else
		for sig in e.signatures
			if !isempty(e.source_url) && startswith(sig, e.name)
				rest = html_escape(sig[nextind(sig, lastindex(e.name)):end])
				linked = """<a href="$(e.source_url)" target="_blank" rel="noopener">$(html_escape(e.name))</a>$(rest)"""
				print(io, """    <code>$(linked)</code>\n""")
			else
				print(io, """    <code>$(html_escape(sig))</code>\n""")
			end
		end
	end
	print(io, """  </div>\n""")
	print(io, """  <div class="doc-body">\n$(e.doc_html)\n  </div>\n""")
	print(io, """</div>\n""")
end

function write_page(mods::Vector{ModData}, outpath::String)
	n_total = sum(length(md.entries) for md in mods)
	n_mods = length(mods)
	generated = Dates.format(Dates.now(), "d U yyyy")

	open(outpath, "w") do io
		# ── head ──────────────────────────────────────────────────────────────
		print(
			io,
			"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Code Documentation — MORFE.jl</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600&display=swap" rel="stylesheet" />
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" />
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<link rel="stylesheet" href="assets/site.css" />
<link rel="stylesheet" href="assets/docs.css" />
<style>
/* documentation-page overrides */
html { scroll-padding-top: var(--nav-h, 60px); }
.doc-module-h { font-size: 24px; font-weight: 500; letter-spacing: -0.02em; margin: 50px 0 6px; padding: 20px 0 12px; border-top: 1px solid var(--hair-2); border-bottom: 1px solid var(--hair); position: sticky; top: var(--nav-h, 60px); background: var(--bg); z-index: 10; }
.doc-module-desc { font-size: 14.5px; color: var(--ink-2); margin: 0 0 16px; line-height: 1.6; }
.doc-module-desc p, .doc-module-desc ul, .doc-module-desc ol { max-width: none; font-size: 14.5px; }
.doc-module-desc h1 { font-size: 13px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3); margin: 20px 0 6px; }
.doc-module-desc h2 { font-size: 13px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-3); margin: 16px 0 6px; }
.doc-module-desc h3 { font-size: 12px; font-weight: 600; color: var(--ink-2); margin: 12px 0 4px; }
.doc-module-desc table { font-size: 13px; border-collapse: collapse; margin: 8px 0; }
.doc-module-desc td, .doc-module-desc th { padding: 4px 12px 4px 0; border-bottom: 1px solid var(--hair); vertical-align: top; }
.doc-module-desc th { font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.doc-module-desc pre { background: #07070b; border: 1px solid var(--hair); border-radius: 4px; padding: 10px 14px; font-size: 12px; overflow-x: auto; margin: 8px 0; }
.doc-module-desc hr { border: 0; border-top: 1px solid var(--hair); margin: 16px 0; }
.doc { margin: 24px 0; position: relative; }
.doc .doc-sig { display: flex; flex-direction: column; gap: 3px; align-items: flex-start; }
/* Per-kind colour tints */
.doc.doc-function { background: rgba(64,99,216,0.05); border-color: rgba(64,99,216,0.18); }
.doc.doc-function .doc-sig { background: rgba(64,99,216,0.12); }
.doc.doc-type     { background: rgba(56,152,38,0.05);  border-color: rgba(56,152,38,0.18); }
.doc.doc-type .doc-sig     { background: rgba(56,152,38,0.12); }
.doc.doc-macro    { background: rgba(149,88,178,0.05); border-color: rgba(149,88,178,0.18); }
.doc.doc-macro .doc-sig    { background: rgba(149,88,178,0.12); }
.doc.doc-constant { background: rgba(203,60,51,0.05);  border-color: rgba(203,60,51,0.18); }
.doc.doc-constant .doc-sig { background: rgba(203,60,51,0.12); }
.doc .doc-sig .badge { position: absolute; top: 14px; right: 18px; }
.doc .doc-sig .badge.f { background: rgba(64,99,216,0.15); color: var(--jl-blue); border: 1px solid rgba(64,99,216,0.3); }
.doc .doc-sig .badge.t { background: rgba(56,152,38,0.15); color: var(--jl-green); border: 1px solid rgba(56,152,38,0.3); }
.doc .doc-sig .badge.m { background: rgba(149,88,178,0.15); color: var(--jl-purple); border: 1px solid rgba(149,88,178,0.3); }
.doc .doc-sig .badge.c { background: rgba(203,60,51,0.15); color: var(--jl-red); border: 1px solid rgba(203,60,51,0.3); }
.doc .doc-sig code { display: block; }
.doc .doc-sig code a { color: inherit; text-decoration: none; border-bottom: 1px dashed rgba(255,255,255,0.25); transition: border-color 0.15s, color 0.15s; }
.doc .doc-sig code a:hover { border-bottom-color: currentColor; }
.doc-body p:last-child { margin-bottom: 0; }
.doc-body pre { background: #07070b; border: 1px solid var(--hair); border-radius: 6px; padding: 14px 18px; overflow-x: auto; color: #d6d6df; font-size: 13px; margin: 10px 0; }
.doc-body pre code { font-size: inherit; background: transparent; border: 0; padding: 0; color: inherit; }
.doc-body p code, .doc-body li code { font-family: var(--mono); font-size: 0.9em; background: #07070b; border: 1px solid var(--hair); padding: 1px 5px; border-radius: 3px; color: #d6d6df; }
.doc-body h1, .doc-body h2, .doc-body h3, .doc-body h4 { font-size: 14px; font-weight: 600; margin: 14px 0 4px; color: var(--ink); border: 0; padding: 0; letter-spacing: 0; }
.doc-body ul, .doc-body ol { color: var(--ink-2); font-size: 14px; line-height: 1.65; padding-left: 20px; margin: 6px 0 10px; }
.doc-body blockquote { border-left: 2px solid var(--jl-purple); padding: 6px 12px; margin: 10px 0; background: rgba(149,88,178,0.05); color: var(--ink-2); }
</style>
<script>
document.addEventListener('DOMContentLoaded', function () {
  // Set --nav-h to the actual rendered nav height so sticky headings sit exactly below it.
  var nav = document.querySelector('nav.nav');
  if (nav) {
	var navH = nav.getBoundingClientRect().height;
	document.documentElement.style.setProperty('--nav-h', navH + 'px');
	// --nav-h stays the bare nav height because the sticky module heading's `top`
	// is keyed to it. Scroll padding also has to clear that heading, so any
	// scroll the JS handler below does not intercept still lands in view.
	var mh = document.querySelector('.doc-module-h');
	document.documentElement.style.scrollPaddingTop =
	  (navH + (mh ? mh.getBoundingClientRect().height : 0)) + 'px';
  }
  if (typeof renderMathInElement === 'function') {
	renderMathInElement(document.body, {
	  delimiters: [
		{left: '\\\\[', right: '\\\\]', display: true},
		{left: '\\\\(', right: '\\\\)', display: false}
	  ]
	});
  }
});
</script>
</head>
<body class="docs-page">

<div class="backdrop" aria-hidden="true"><canvas id="manifold-bg"></canvas></div>

<div class="site">

<nav id="site-nav" class="nav"></nav>
<script src="assets/nav.js"></script>

<div class="docs-shell">

<aside class="docs-side">
  <div class="ver"><span><a href="documentation.html">Code Documentation</a></span></div>
  <div class="search">
	<span class="ic mono">⌕</span>
	<input type="search" id="doc-search" placeholder="Filter…" autocomplete="off" />
  </div>

  <h4>Modules</h4>
  <ul>
""",
		)
		for md in mods
			print(io, "    <li><a href=\"#$(md.label)\">$(md.label)</a></li>\n")
		end
		print(
			io,
			"""  </ul>

  <h4>Resources</h4>
  <ul>
	<li><a href="index.html">Getting started</a></li>
	<li><a href="tutorials/index.html">Demo scripts</a></li>
	<li><a href="https://github.com/MORFEproject/MORFE.jl" target="_blank" rel="noopener">GitHub</a></li>
  </ul>
</aside>

<main class="docs-main">
  <div class="crumbs"><a href="index.html">MORFE.jl</a><span class="sep">/</span><span>Code Documentation</span></div>
  <h1>Code Documentation</h1>
  <p class="lede">$(n_mods) modules · $(n_total) documented entries · generated $(generated)</p>

""",
		)

		# ── content sections ───────────────────────────────────────────────
		for md in mods
			print(io, "  <div id=\"$(md.label)\" class=\"doc-module-h\">$(md.label)</div>\n")
			if !isempty(md.desc_html)
				print(io, "  <div class=\"doc-module-desc\" data-module=\"$(md.label)\">$(md.desc_html)</div>\n")
			end
			for e in md.entries
				write_entry(io, md.label, e)
			end
		end

		# ── close main + shell ─────────────────────────────────────────────
		print(
			io,
			"""
</main>
</div>

<footer class="foot">
  <div class="wrap">
	<div class="foot-grid">
	  <div>
		<div class="nav-logo" style="margin-bottom:14px;"><span class="dot"></span> MORFE<span style="color:var(--ink-3)">.jl</span></div>
		<p style="color:var(--ink-3);font-size:13px;max-width:36ch;">Model-Order Reduction for Finite Elements — direct parametrisation of invariant manifolds in Julia.</p>
	  </div>
	  <div><h4>Project</h4><ul><li><a href="features.html">Features</a></li><li><a href="tutorials/index.html">Tutorials</a></li><li><a href="documentation.html">Code Documentation</a></li></ul></div>
	  <div><h4>Code</h4><ul><li><a href="https://github.com/MORFEproject/MORFE.jl" target="_blank" rel="noopener">GitHub</a></li><li><a href="https://github.com/MORFEproject/MORFE.jl/issues" target="_blank" rel="noopener">Issues</a></li><li><a href="https://github.com/MORFEproject/MORFE.jl/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener">Contributing</a></li></ul></div>
	  <div><h4>Cite</h4><ul><li><a href="publications.html">Publications</a></li><li style="color:var(--ink-3);font-size:12px;">Citation details will appear with first publication.</li></ul></div>
	</div>
	<div style="display:flex;justify-content:space-between;align-items:center;margin-top:48px;padding-top:24px;border-top:1px solid var(--hair);color:var(--ink-3);font-size:12px;">
	  <div>© MORFE contributors · MIT license</div>
	  <div class="mono">generated · $(generated) · julia 1.11</div>
	</div>
  </div>
</footer>

</div>
<script src="assets/manifold-bg.js"></script>
<script>
// Live filter by text content
(function () {
  var input = document.getElementById('doc-search');
  if (!input) return;
  var entries  = document.querySelectorAll('.doc[id]');
  var headings = document.querySelectorAll('.doc-module-h[id]');
  var descMap  = {};
  document.querySelectorAll('.doc-module-desc[data-module]').forEach(function (d) {
	descMap[d.getAttribute('data-module')] = d;
  });
  input.addEventListener('input', function () {
	var q = this.value.trim().toLowerCase();
	entries.forEach(function (e) {
	  e.style.display = !q || e.textContent.toLowerCase().indexOf(q) >= 0 ? '' : 'none';
	});
	headings.forEach(function (h) {
	  var prefix = h.id + '-', visible = !q;
	  if (!visible) {
		entries.forEach(function (e) {
		  if (e.id.indexOf(prefix) === 0 && e.style.display !== 'none') visible = true;
		});
	  }
	  h.style.display = visible ? '' : 'none';
	  if (descMap[h.id]) descMap[h.id].style.display = q ? 'none' : '';
	});
  });
})();

// Sidebar scrollspy
(function () {
  var links    = document.querySelectorAll('.docs-side a[href^="#"]');
  var headings = document.querySelectorAll('.doc-module-h[id]');
  function update() {
	var scrollY = window.scrollY + 90, active = null;
	headings.forEach(function (h) { if (h.offsetTop <= scrollY) active = h.id; });
	links.forEach(function (a) { a.classList.toggle('active', a.getAttribute('href') === '#' + active); });
  }
  window.addEventListener('scroll', update, { passive: true });
  update();
})();

// In-page navigation for every "#" link: the sidebar module list and the @ref
// cross-references inside docstrings alike.
//
// Two things sit above the scroll position and the browser accounts for neither.
// The nav is fixed, and the module heading is sticky at top:var(--nav-h), so it
// parks over whatever an entry anchor scrolls to. A module heading is its own
// sticky element, so it needs the nav allowance only; an entry needs both.
//
// VERTICAL_MARGIN is separate from that: it is breathing room above a module
// heading, which starts a section and wants air above it however it was reached —
// sidebar or cross-reference. Entries take the sticky correction and nothing more,
// so a cross-reference lands its target directly under the chrome.
(function () {
  // Increase this value to get more space above a module heading
  var VERTICAL_MARGIN = 60;   // pixels

  function navHeight() {
	return parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--nav-h')) || 60;
  }

  function stickyHeadingHeight() {
	var h = document.querySelector('.doc-module-h');
	return h ? h.getBoundingClientRect().height : 0;
  }

  function targetOf(href) {
	if (!href || href.charAt(0) !== '#') return null;
	var id = href.slice(1);
	try { id = decodeURIComponent(id); } catch (err) { /* keep the raw id */ }
	return document.getElementById(id);
  }

  function scrollToTarget(target) {
	var isHeading = target.classList.contains('doc-module-h');

	// Temporarily switch to relative (same JS task, no repaint) so
	// getBoundingClientRect returns the natural flow top even when stuck.
	if (isHeading) target.style.setProperty('position', 'relative', 'important');
	var naturalTop = target.getBoundingClientRect().top + window.scrollY;
	if (isHeading) target.style.removeProperty('position');

	var offset = isHeading ? navHeight() + VERTICAL_MARGIN
						   : navHeight() + stickyHeadingHeight();
	window.scrollTo({ top: naturalTop - offset });
  }

  // Delegated so cross-references rendered inside docstrings are covered too.
  document.addEventListener('click', function (e) {
	if (!e.target.closest) return;
	var a = e.target.closest('a[href^="#"]');
	if (!a) return;
	var target = targetOf(a.getAttribute('href'));
	if (!target) return;
	e.preventDefault();
	history.replaceState(null, '', a.getAttribute('href'));
	scrollToTarget(target);
  });

  // Deep links land the same way: the browser's own jump ignores the sticky heading.
  window.addEventListener('hashchange', function () {
	var target = targetOf(location.hash);
	if (target) scrollToTarget(target);
  });
  if (location.hash) {
	var initial = targetOf(location.hash);
	if (initial) setTimeout(function () { scrollToTarget(initial); }, 0);
  }
})();
</script>
</body>
</html>
""",
		)
	end

	sz = round(filesize(outpath) / 1024; digits = 1)
	@info "Written $(outpath) ($(sz) KiB)"
end

# ── Main ──────────────────────────────────────────────────────────────────────
mods    = resolve_all_refs(extract_all())
outpath = joinpath(@__DIR__, "documentation.html")
write_page(mods, outpath)
