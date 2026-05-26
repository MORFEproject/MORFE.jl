module BifurcationKitInterface

using ..ParametrisationMethod: ReducedDynamics

export make_bk_problem

"""
    make_bk_problem(R::ReducedDynamics; bifparam_index=1, z0=nothing, p0=nothing)

Wrap the reduced dynamics `R` as a `BifurcationKit.BifurcationProblem` for
continuation and bifurcation analysis.

The reduced dynamics ż = R(z, p) is split into master-mode coordinates `z`
(first `ROM` variables) and external parameters `p` (the remaining
`external_system_size` variables).  `bifparam_index` selects which component
of `p` is the continuation parameter.

Requires BifurcationKit.jl.  Load it with `using BifurcationKit` to activate
the MORFE extension.
"""
function make_bk_problem(R::ReducedDynamics; bifparam_index::Int = 1, z0 = nothing, p0 = nothing)
    error(
        "make_bk_problem requires BifurcationKit.jl.\n" *
        "Load it with `using BifurcationKit` to activate the MORFE extension.",
    )
end

end # module BifurcationKitInterface
