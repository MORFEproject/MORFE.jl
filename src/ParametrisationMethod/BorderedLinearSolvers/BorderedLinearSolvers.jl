"""
	BorderedLinearSolvers

Numerical linear algebra for MORFE's constant-size bordered systems.

This internal module owns sparse backend selection, constant-pattern solver state,
symbolic-factorisation caching, numeric refactorisation, exact factor reuse, residual
verification, iterative refinement, and Pardiso extension hooks. It deliberately does not
assemble cohomological equations, schedule monomials, manage symmetry, or persist
checkpoints.

The supported sparse backends are KLU, UMFPACK, and extension-provided Pardiso. A cached
factorisation is retained only after a successful factorisation; failed cached KLU and
UMFPACK updates receive one fresh symbolic analysis before failure is reported.

# Source organisation

| File | Purpose |
|:-----|:--------|
| `SolverState.jl` | Backend markers, constant-pattern sparse storage, and cache lifetime |
| `FailureDiagnostics.jl` | Contextual failure categories and user-facing diagnostics |
| `Factorisations.jl` | Typed KLU and UMFPACK symbolic/numeric factorisation reuse |
| `AccuracyControl.jl` | Sparse backward errors and iterative refinement |
| `BorderedSolve.jl` | Backend dispatch, exact factor reuse, and in-place solves |
"""
module BorderedLinearSolvers

using LinearAlgebra
using SparseArrays
using SparseArrays.UMFPACK: UmfpackLU
using KLU: KLUFactorization, klu, klu_factor!

using ..InvarianceEquation: precompute_sparse_bordered_template

const _PARDISO_INACTIVE = "Pardiso solver object present but MORFEPardisoExt not active — internal error."

"""
	_try_build_pardiso_solver() -> solver or nothing

Extension hook implemented by `MORFEPardisoExt`. Return a configured Pardiso solver when
the extension is active, or `nothing` so automatic selection can fall back to KLU.
"""
_try_build_pardiso_solver(::Vararg{Any}) = nothing

"""Run Pardiso configuration and symbolic analysis for a bordered matrix."""
_pardiso_prepare!(args...) = error(_PARDISO_INACTIVE)

"""Run Pardiso numeric factorisation followed by a solve."""
_pardiso_factorise_solve!(args...) = error(_PARDISO_INACTIVE)

"""Solve with the current Pardiso numeric factorisation."""
_pardiso_solve!(args...) = error(_PARDISO_INACTIVE)

"""Release extension-owned Pardiso factorisation storage."""
_pardiso_release!(args...) = nothing

"""
	_configured_residual_tolerance(T, options) -> tolerance or nothing

Resolve the backward-error threshold in the real scalar type associated with `T`.
`options = nothing` selects the standard checked default used by the low-level constructor.
"""
function _configured_residual_tolerance(::Type{T}, options) where {T}
    RT = typeof(real(zero(T)))
    options === nothing && return sqrt(eps(RT)) / RT(100)
    options.residual_check == :off && return nothing
    return isnothing(options.residual_tolerance) ?
           sqrt(eps(RT)) / RT(100) : convert(RT, options.residual_tolerance)
end

include("SolverState.jl")
include("FailureDiagnostics.jl")
include("Factorisations.jl")
include("AccuracyControl.jl")
include("BorderedSolve.jl")

export SparseLinearSolverState
export _try_build_pardiso_solver, _pardiso_prepare!, _pardiso_factorise_solve!,
       _pardiso_solve!, _pardiso_release!

end # module BorderedLinearSolvers
