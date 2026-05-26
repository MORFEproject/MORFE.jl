module MORFEBifurcationKitExt

using MORFE
using MORFE.BifurcationKitInterface
using MORFE.Polynomials: evaluate
using MORFE.ParametrisationMethod: ReducedDynamics
using BifurcationKit
using LinearAlgebra

"""
    make_bk_problem(R; bifparam_index=1, z0=nothing, p0=nothing)

Wrap `ReducedDynamics R` as a `BifurcationKit.BifurcationProblem`.

The reduced ODE ż = R(z, p) is split into master-mode coordinates `z` (first
`ROM` entries) and external parameters `p` (the remaining `N_EXT` entries).
`bifparam_index` selects which component of `p` is the continuation parameter.

`z0` defaults to `zeros(ComplexF64, ROM)` and `p0` defaults to
`zeros(Float64, N_EXT)`.
"""
function MORFE.BifurcationKitInterface.make_bk_problem(
    R::ReducedDynamics{ROM, NVAR};
    bifparam_index::Int = 1,
    z0 = nothing,
    p0 = nothing,
) where {ROM, NVAR}
    N_EXT = R.external_system_size
    @assert 1 <= bifparam_index <= N_EXT "bifparam_index=$bifparam_index out of range 1:$N_EXT"

    z0_ = isnothing(z0) ? zeros(ComplexF64, ROM) : complex.(float.(z0))
    p0_ = isnothing(p0) ? zeros(Float64, N_EXT)  : float.(p0)

    function F(z, p)
        zp = vcat(z, p)
        rz = evaluate(R.poly, zp)
        return rz[1:ROM]
    end

    lens = BifurcationKit.@optic _[bifparam_index]

    return BifurcationKit.BifurcationProblem(F, z0_, p0_, lens)
end

end # module MORFEBifurcationKitExt
