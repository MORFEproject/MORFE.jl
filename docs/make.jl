using Pkg

# When running locally (julia --project=docs docs/make.jl from the repo root),
# make sure the local MORFE source is used rather than any registered version.
# In CI this is handled by the workflow install step, but the call is idempotent.
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using MORFE
using Documenter

makedocs(
    sitename = "MORFE.jl",
    authors  = "MORFEproject contributors",
    modules  = [
        MORFE,
        MORFE.Multiindices,
        MORFE.Polynomials,
        MORFE.MultilinearMaps,
        MORFE.ExternalSystems,
        MORFE.FullOrderModel,
        MORFE.Eigensolvers,
        MORFE.EigenModesPropagation,
        MORFE.Realification,
        MORFE.Resonance,
        MORFE.InvarianceEquation,
        MORFE.MasterModeOrthogonality,
        MORFE.ParametrisationMethod,
        MORFE.MultilinearTerms,
        MORFE.LowerOrderCouplings,
    ],
    pages = [
        "Home"             => "index.md",
        "Project Overview" => "project-overview.md",
        "Theoretical Background" => [
            "Multiindices"      => "multiindices.md",
            "Polynomials"       => "polynomials.md",
            "Realification"     => "realification.md",
            "Multilinear Terms" => "multilinear_terms.md",
        ],
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(
        # Render math with MathJax 3 — handles complex LaTeX (sums, aligned
        # environments, macros) better than the default KaTeX engine.
        mathengine = Documenter.MathJax3(),

        # Show an "Edit on GitHub" pencil icon on every page.
        edit_link  = "main",

        # Canonical URL for SEO and cross-version links.
        canonical  = "https://morfeproject.github.io/MORFE.jl",

        # GitHub repo link shown in the top-right corner.
        repolink   = "https://github.com/MORFEproject/MORFE.jl",

        # Collapse the sidebar to depth 1 by default so the left panel isn't
        # overwhelming on first load; users can expand sections as needed.
        collapselevel = 1,

        # Use pretty /section/ URLs in CI; fall back to file.html locally so
        # the build can be opened directly from the filesystem without a server.
        prettyurls = get(ENV, "CI", nothing) == "true",

        # Increase the page-size warning threshold — the API page is large by
        # design (all module docstrings on one page).
        size_threshold_warn = 300 * 1024,   # 300 KiB
        size_threshold      = 600 * 1024,   # 600 KiB hard limit
    ),
    doctest  = false,
    warnonly = [:docs_block, :missing_docs],
)

deploydocs(
    repo         = "github.com/MORFEproject/MORFE.jl.git",
    devbranch    = "main",
    push_preview = true,
)
