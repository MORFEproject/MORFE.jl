using Pkg
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
using MORFE

using Documenter

makedocs(
    sitename = "MORFE.jl",
    modules = Module[MORFE],
    pages = [
        "Home" => "index.md",
        "Project Overview" => "project-overview.md",
        "Theoretical Background" => [
            "Multiindices" => "multiindices.md",
            "Polynomials" => "polynomials.md",
            "Realification" => "realification.md",
            "Multilinear Terms" => "multilinear_terms.md",
        ],
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(
        repolink = "https://github.com/MORFEproject/MORFE.jl",
    ),
    doctest = false,
    warnonly = [:docs_block, :missing_docs],
)

deploydocs(
    repo = "github.com/MORFEproject/MORFE.jl.git",
    push_preview = true,
    devbranch = "main",
)
