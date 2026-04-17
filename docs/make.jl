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
        MORFE.JordanChain,
        MORFE.PropagateEigenmodes,
        MORFE.Realification,
        MORFE.Resonance,
        MORFE.InvarianceEquation,
        MORFE.MasterModeOrthogonality,
        MORFE.ParametrisationMethod,
        MORFE.MultilinearTerms,
        MORFE.LowerOrderCouplings,
        MORFE.CohomologicalEquations,
    ],
    pages = [
        "Home"             => "index.md",
        "Project Overview" => "project-overview.md",
        "Theoretical Background" => [
            "Multiindices"            => "multiindices.md",
            "Polynomials"             => "polynomials.md",
            "Realification"           => "realification.md",
            "Multilinear Terms"       => "multilinear_terms.md",
            "Cohomological Equations" => "cohomological_equations.md",
        ],
        "API Reference" => [
            "Multiindices"       => "api/multiindices.md",
            "Polynomials"        => "api/polynomials.md",
            "Realification"      => "api/realification.md",
            "Full Order Model"   => [
                "MultilinearMaps"  => "api/multilinear_maps.md",
                "ExternalSystems"  => "api/external_systems.md",
                "FullOrderModel"   => "api/full_order_model.md",
            ],
            "Spectral Decomposition" => [
                "Eigensolvers"        => "api/eigensolvers.md",
                "JordanChain"         => "api/jordan_chain.md",
                "PropagateEigenmodes" => "api/propagate_eigenmodes.md",
            ],
            "Parametrisation Method" => [
                "Resonance"               => "api/resonance.md",
                "InvarianceEquation"      => "api/invariance_equation.md",
                "MasterModeOrthogonality" => "api/master_mode_orthogonality.md",
                "ParametrisationMethod"   => "api/parametrisation_method.md",
                "MultilinearTerms"        => "api/multilinear_terms.md",
                "LowerOrderCouplings"     => "api/lower_order_couplings.md",
                "CohomologicalEquations"  => "api/cohomological_equations.md",
            ],
        ],
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

        # custom.css for improved typography and layout.
        # logo.svg in assets/ is auto-detected by Documenter and shown in the sidebar.
        assets = ["assets/custom.css"],

        # Collapse the sidebar to depth 2 so API sub-sections start expanded.
        collapselevel = 2,

        # Use pretty /section/ URLs in CI; fall back to file.html locally so
        # the build can be opened directly from the filesystem without a server.
        prettyurls = get(ENV, "CI", nothing) == "true",

        # Increase the page-size warning threshold — the API page is large by
        # design (all module docstrings on one page).
        size_threshold_warn = 300 * 1024,   # 300 KiB
        size_threshold      = 600 * 1024,   # 600 KiB hard limit
    ),
    doctest  = false,
    warnonly = [:docs_block, :missing_docs, :cross_references],
)
