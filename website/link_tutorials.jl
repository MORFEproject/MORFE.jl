# website/link_tutorials.jl
# Run from repo root:  julia website/link_tutorials.jl
#
# Turns every `<code>Name</code>` in the hand-written tutorials into a link to that
# symbol's entry in the core or companion code-documentation page, so a name behaves the
# same way on a tutorial page as it already does inside a docstring (where
# `[`name`](@ref)` is resolved by generate_documentation.jl).
#
# The symbol map is scraped from both generated HTML pages rather than from the packages:
# those files are what actually ships, so every link written here points at an anchor that
# exists in the deployed artifact. It also means this script loads no packages.
#
# Re-runnable by construction — see `rewrite` below — so CI can safely refresh links on
# every deployment.

const HERE = @__DIR__
const DOC_PAGE = joinpath(HERE, "documentation.html")
const COMPANION_DOC_PAGE = joinpath(HERE, "morfeferrite-documentation.html")
const TUT_DIR = joinpath(HERE, "tutorials")
const DOC_HREF = "../documentation.html"          # tutorials sit one level down
const COMPANION_DOC_HREF = "../morfeferrite-documentation.html"

# ── Per-file overrides ────────────────────────────────────────────────────────────────
# The same word means different things on different pages, so an override is scoped to the
# file it is true for.  `Multiindices.multiindex` is a constructor, but on these pages
# `multiindex` means *the multiindex of a MultilinearMap* — the keyword argument and the
# calling convention it encodes.  `multiindex_sets.html` is deliberately absent: there the
# word will mean the `Multiindices` object, and the default resolution is the right one.
#
# Overrides also cover two cases the scraped map structurally cannot reach.  `MORFEFerrite`
# and `MORFESymbolicsExt` are package/module names with no entry of their own, so without an
# override they are unwrapped and left plain — that is how `MORFEFerrite` silently lost its
# link on structural_svk.html.  And a key may be a whole call signature rather than a bare
# name: the map is keyed on identifiers, so `model_from_symbolics(exprs, groups)` matches
# nothing, while an override keyed on that exact string does.
const OVERRIDES = Dict(
    "full_order_model.html" =>
        Dict("multiindex" => "$(DOC_HREF)#MultilinearMaps-MultilinearMap"),
    "karman.html" => Dict(
        "multiindex" => "$(DOC_HREF)#MultilinearMaps-MultilinearMap",
        "MORFEFerrite" => COMPANION_DOC_HREF,
        "MORFEFerrite.FluidNavierStokes" => "$(COMPANION_DOC_HREF)#FluidNavierStokes",
        # `build_model` is defined by four backends, so the scraped map drops it as a clash
        # and it would silently stay plain text.  This page calls the fluid method, whose
        # docstring is the one that documents `master`, `outer` and `expansion_order`.
        "build_model" => "$(COMPANION_DOC_HREF)#FluidNavierStokes-build_model"
    ),
    "structural_svk.html" => Dict(
        "build_model" => "$(COMPANION_DOC_HREF)#Common-build_model",
        "mechanical_model" => "$(COMPANION_DOC_HREF)#StructuralSVK-mechanical_model",
        "W" => "$(DOC_HREF)#ParametrisationObjects-Parametrisation",
        "R" => "$(DOC_HREF)#ParametrisationObjects-ReducedDynamics",
        "MORFEFerrite" => COMPANION_DOC_HREF
    ),
    "symbolics_ext.html" => Dict(
        "MORFESymbolicsExt" => "$(DOC_HREF)#SymbolicsExtension",
        "model_from_symbolics(exprs, groups)" =>
            "$(DOC_HREF)#SymbolicsExtension-model_from_symbolics",
        "model_from_symbolics(exprs, groups, ext_var, ext_exprs)" =>
            "$(DOC_HREF)#SymbolicsExtension-model_from_symbolics",
        "externalsystem_from_symbolics(exprs, var)" =>
            "$(DOC_HREF)#SymbolicsExtension-externalsystem_from_symbolics",
        "externalsystem_from_symbolics(ext_exprs, ext_var)" =>
            "$(DOC_HREF)#SymbolicsExtension-externalsystem_from_symbolics",
        "ExternalSystem((im*Ω, -im*Ω))" => "$(DOC_HREF)#ExternalSystems-ExternalSystem"
    )
)

# ── Symbol map ────────────────────────────────────────────────────────────────────────
const ENTRY_ID_RE = r"id=\"([A-Za-z][A-Za-z0-9]*)-([^\"]+)\""
const MODULE_ID_RE = r"id=\"([A-Za-z][A-Za-z0-9]*)\" class=\"doc-module-h\""

"""
	symbol_map(doc_html, page_href) -> (map, clashes)

`name => "page#anchor"` for every unambiguous documented entry and module heading.

Names documented by several modules are returned in `clashes` and omitted from the map.
Callers resolve only the meanings they know through a per-file override; this is how the
companion page's backend-specific `build_model` methods stay explicit.
"""
function symbol_map(doc_html::AbstractString, page_href::AbstractString)
    map = Dict{String, String}()
    owners = Dict{String, Vector{String}}()
    for m in eachmatch(ENTRY_ID_RE, doc_html)
        modlabel, name = m.captures[1], m.captures[2]
        push!(get!(owners, name, String[]), modlabel)
        map[name] = "$(page_href)#$(modlabel)-$(name)"
    end
    clashes = sort([n for (n, ms) in owners if length(unique(ms)) > 1])
    foreach(n -> delete!(map, n), clashes)
    for m in eachmatch(MODULE_ID_RE, doc_html)
        map[m.captures[1]] = "$(page_href)#$(m.captures[1])"
    end
    return map, clashes
end

# ── Rewriting ─────────────────────────────────────────────────────────────────────────
# `<a class="doc-link" href="…"><code>X</code></a>` — what step 1 removes and step 3 writes.
const WRAPPED_RE = r"<a class=\"doc-link\" href=\"[^\"]*\"><code>([^<]*)</code></a>"
# Any anchor and its contents, matched non-greedily: the span step 2 protects.
const ANCHOR_RE = r"<a\b[^>]*>.*?</a>"s
const CODE_RE = r"<code>([^<]*)</code>"
const IDENTIFIER_RE = r"^[A-Za-z_][A-Za-z0-9_!]*$"

"""
	rewrite(html, map, overrides) -> (html, n_linked, unresolved)

Three passes, and the order is what makes this idempotent:

 1. **strip** any wrapper a previous run wrote, so re-running refreshes the links instead
	of layering them or leaving stale ones behind;
 2. **mask** every `<a>…</a>` span, which is what keeps the GitHub example-folder links —
	whose body is itself a `<code>` — from being wrapped a second time;
 3. **wrap** the `<code>` tags that are left and that resolve.

Only `<code>` with no attributes is considered, so `<code class="nolink">` is the opt-out
for a name used in a sense the documentation does not cover.
"""
function rewrite(html::AbstractString, map::Dict{String, String}, overrides::Dict{
        String, String})
    html = replace(html, WRAPPED_RE => s"<code>\1</code>")

    # Mask, rewrite the gaps, then restore.  Splitting on the anchor spans means the code
    # inside them is never even looked at.
    spans = collect(eachmatch(ANCHOR_RE, html))
    pieces = String[]
    last = 1
    for m in spans
        push!(pieces, html[last:prevind(html, m.offset)])
        last = m.offset + ncodeunits(m.match)
    end
    push!(pieces, html[last:end])

    n_linked = 0
    unresolved = String[]
    linked = map_piece -> replace(map_piece,
        CODE_RE => function (whole)
            name = match(CODE_RE, whole).captures[1]
            target = get(overrides, name, get(map, name, nothing))
            if target === nothing
                occursin(IDENTIFIER_RE, name) && push!(unresolved, name)
                return whole
            end
            n_linked += 1
            return "<a class=\"doc-link\" href=\"$(target)\"><code>$(name)</code></a>"
        end)

    out = IOBuffer()
    for (k, piece) in enumerate(pieces)
        print(out, linked(piece))
        k <= length(spans) && print(out, spans[k].match)
    end
    return String(take!(out)), n_linked, unresolved
end

# ── Main ──────────────────────────────────────────────────────────────────────────────
function main()
    isfile(DOC_PAGE) ||
        error("$(DOC_PAGE) not found — run website/generate_documentation.jl first")
    isfile(COMPANION_DOC_PAGE) || error(
        "$(COMPANION_DOC_PAGE) not found — run website/generate_morfeferrite_documentation.jl first")

    core_symbols, core_clashes = symbol_map(read(DOC_PAGE, String), DOC_HREF)
    companion_symbols, companion_clashes = symbol_map(read(COMPANION_DOC_PAGE, String), COMPANION_DOC_HREF)
    symbols = copy(core_symbols)
    for (name, target) in companion_symbols
        haskey(symbols, name) || (symbols[name] = target)
    end
    files = sort(filter(f -> endswith(f, ".html"), readdir(TUT_DIR)))
    @info "symbol map: $(length(symbols)) names from core and companion documentation" core_clashes companion_clashes

    total = 0
    unresolved = String[]
    for f in files
        path = joinpath(TUT_DIR, f)
        html = read(path, String)
        new, n, miss = rewrite(html, symbols, get(OVERRIDES, f, Dict{String, String}()))
        new == html || write(path, new)
        total += n
        append!(unresolved, miss)
        println(rpad(f, 26), lpad(n, 3), " links")
    end
    println("\n$(total) links across $(length(files)) files")

    # Identifier-shaped names with no entry are usually a real documentation gap, or a
    # MORFEFerrite symbol this site does not document.  Worth seeing, not worth guessing at.
    isempty(unresolved) ||
        @info "not documented on this site — left as plain text" names=sort(unique(unresolved))
    return nothing
end

main()
