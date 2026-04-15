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
    modules = [
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
        "Home" => "index.md",
        "Project Overview" => "project-overview.md",
        "Theoretical Background" => [
            "Multiindices"       => "multiindices.md",
            "Polynomials"        => "polynomials.md",
            "Realification"      => "realification.md",
            "Multilinear Terms"  => "multilinear_terms.md",
        ],
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(
        repolink = "https://github.com/MORFEproject/MORFE.jl",
        canonical = "https://morfeproject.github.io/MORFE.jl",
    ),
    doctest = false,
    warnonly = [:docs_block, :missing_docs],
)

deploydocs(
    repo      = "github.com/MORFEproject/MORFE.jl.git",
    devbranch = "main",
    push_preview = true,
)
