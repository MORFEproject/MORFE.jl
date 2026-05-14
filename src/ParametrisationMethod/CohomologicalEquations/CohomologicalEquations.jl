"""
	CohomologicalEquations

Solve the cohomological equations that arise in the parametrisation method for
computing Spectral Submanifolds (SSMs) of high-dimensional dynamical systems.

---

# Problem statement

Given a full-order model of order `ORD` in `FOM` degrees of freedom, the
parametrisation method seeks a change of coordinates

```
x(t) = W(z(t)),          z ∈ ℂᴺᵛᵃʳ
```

such that the reduced dynamics `ż = R(z)` is simpler.  Expanding both `W` and
`R` as dense polynomials in the `NVAR = ROM + N_EXT` reduced variables (ROM
master-mode amplitudes plus N_EXT external forcing amplitudes) and matching
coefficients monomial by monomial yields the **cohomological equations**: one
linear system per multi-index `α`.

---

# System structure for multi-index `α`

Let `s = ⟨λ, α⟩` be the superharmonic frequency and let `nR = |ℛ(α)|` be the
number of master modes resonant at `α`.  The cohomological linear system is

```
┌              ┐ ┌         ┐   ┌           ┐
│  L(s)  C(s)  │ │  W[α]   │ = │  RHS_inv  │   FOM rows  (invariance)
│  L̂(s)  Ĉ(s)  │ │  R_res  │   │  RHS_ort  │   nR  rows  (orthogonality)
└              ┘ └         ┘   └           ┘
```

where:
- `L(s)`  (`FOM × FOM`) is the parametrisation operator,
- `C(s)`  (`FOM × nR`) acts on the unknown resonant reduced-dynamics coefficients,
- `L̂(s)` (`nR × FOM`) is the orthogonality row operator,
- `Ĉ(s)` (`nR × nR`) is the orthogonality joint operator,
- `W[α] ∈ ℂᶠᵒᵐ` is the parametrisation coefficient,
- `R_res ∈ ℂⁿᴿ` are the reduced-dynamics coefficients for resonant modes.

Non-resonant master modes are trivially zero and excluded from the system.
External forcing modes are *known* and appear only on the right-hand side.

---

# Module contents

| Symbol | Description |
|:-------|:------------|
| [`InvarianceOperators`](@ref)       | Precomputed invariance-equation operator coefficients |
| [`OrthogonalityOperators`](@ref)    | Precomputed orthogonality-condition operator coefficients |
| [`LowerOrderResources`](@ref)       | Lower-order coupling data and buffers |
| [`CohomologicalBuffers`](@ref)      | Pre-allocated system-assembly scratch buffers |
| [`SparseLinearSolverState`](@ref)   | Sparse-path solver handles and pre-allocated RHS buffer |
| [`CohomologicalContext`](@ref)      | Composed struct bundling all precomputed operators and resources |
| [`solve_single_monomial!`](@ref)    | Solve the cohomological system for one multi-index |
| [`solve_cohomological_equations!`](@ref) | Solve for all multi-indices in causal order |
| [`solve_cohomological_problem`](@ref)    | High-level driver: precompute everything and solve |
"""
module CohomologicalEquations

using ..Multiindices: MultiindexSet, indices_in_box_with_bounded_degree,
                      build_exponent_index_map
using ..Polynomials: DensePolynomial
using ..ParametrisationMethod: Parametrisation, ReducedDynamics,
                               create_parametrisation_method_objects,
                               compute_higher_derivative_coefficients!,
                               multiindex_set
using ..LowerOrderCouplings: compute_lower_order_couplings
using ..InvarianceEquation: assemble_cohomological_matrix_and_rhs!,
                            precompute_master_column_polynomials,
                            precompute_external_column_polynomials,
                            build_sparse_L_and_rhs!,
                            precompute_sparse_L_template,
                            evaluate_column!,
                            evaluate_external_rhs!
using ..MasterModeOrthogonality: assemble_orthogonality_matrix_and_rhs!,
                                 precompute_orthogonality_operator_coefficients,
                                 precompute_orthogonality_column_polynomials
using ..FullOrderModel: NDOrderModel

using ..MultilinearTerms: compute_multilinear_terms, compute_multilinear_terms!,
                          build_multilinear_terms_cache, MultilinearTermsCache
using ..Resonance: ResonanceSet, is_resonant
using LinearAlgebra
using SparseArrays
using KLU: klu, klu!
using Pardiso: AbstractPardisoSolver, MKLPardisoSolver, solve as pardiso_solve
using StaticArrays: SVector

include("OperatorData.jl")
include("SolverResources.jl")
include("CohomologicalContext.jl")
include("ConjugateSymmetry.jl")
include("CohomologicalSolver.jl")
include("CohomologicalDriver.jl")

export CohomologicalContext,
       InvarianceOperators,
       OrthogonalityOperators,
       LowerOrderResources,
       CohomologicalBuffers,
       SparseLinearSolverState,
       NoConjugatePermutation,
       RealArithmeticBuffers,
       ConjugateSymmetryData,
       detect_conjugate_permutation,
       fill_conjugate_monomial!,
       solve_cohomological_equations!,
       solve_single_monomial!,
       solve_cohomological_problem

# ==============================================================================
# Utility helpers (public — used by solve_single_monomial! and the driver)
# ==============================================================================

# Copy the coefficients of an N_EXT-variable external polynomial into the last
# N_EXT rows of R's coefficient matrix.
function _embed_external_dynamics!(
        R::ReducedDynamics{ROM, NVAR, T},
        ext_poly::DensePolynomial{T, N_EXT, 2},
        mset::MultiindexSet{NVAR}
) where {ROM, NVAR, T, N_EXT}
    N_EXT > 0 || return nothing
    ext_coeffs = ext_poly.coefficients
    for (j, α_ext) in enumerate(ext_poly.multiindex_set.exponents)
        α_full = SVector{NVAR, Int}(ntuple(i -> i <= ROM ? 0 : α_ext[i - ROM], Val(NVAR)))
        idx_full = findfirst(==(α_full), mset.exponents)
        idx_full === nothing && continue
        for e in 1:N_EXT
            coeff = T(ext_coeffs[e, j])
            iszero(coeff) || (R.poly.coefficients[ROM + e, idx_full] = coeff)
        end
    end
    return nothing
end

# Return positions in `mset` of all unit-vector monomials eᵣ for r = 1 … NVAR.
function _linear_monomial_indices(mset::MultiindexSet{NVAR}) where {NVAR}
    indices = Int[]
    n_search = min(NVAR + 1, length(mset))
    for r in 1:NVAR
        e_r = SVector{NVAR, Int}(ntuple(i -> i == r ? 1 : 0, Val(NVAR)))
        idx = findfirst(==(e_r), view(mset.exponents, 1:n_search))
        idx !== nothing && push!(indices, idx)
    end
    return indices
end

# Return a compile-time-sized SVector{ROM, Bool} indicating which master modes
# are resonant with the monomial at position `monomial_idx`.
@inline function _resonance_vector(
        resonance_set::ResonanceSet,
        monomial_idx::Int,
        ::Val{ROM}
) where {ROM}
    return SVector{ROM, Bool}(ntuple(r -> is_resonant(resonance_set, monomial_idx, r), Val(ROM)))
end

# ==============================================================================
# Conjugate-symmetry dispatch helpers (private)
# ==============================================================================

# RB = Nothing: real arithmetic never available → always complex solve.
@inline function _sym_solve_monomial!(
        ctx, ::ConjugateSymmetryData{CP, Nothing}, ::Int,
        s, nR, resonance, lower_order_couplings, external_dynamics
) where {CP}
    _solve_monomial!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)
end

# RB = RealArithmeticBuffers: one runtime check per monomial.
# Real arithmetic is correct only for self-conjugate, NON-resonant monomials: the bordered
# system [L C; L̂ Ĉ] has complex C and L̂ columns whenever nR > 0, so taking real(·) of
# the bordered system gives the wrong solution for resonant monomials.
@inline function _sym_solve_monomial!(
        ctx, sym::ConjugateSymmetryData{CP, RealArithmeticBuffers}, idx,
        s, nR, resonance, lower_order_couplings, external_dynamics
) where {CP}
    if sym.monomial_map[idx] == idx && nR == 0
        _solve_monomial_real!(ctx, sym.real_buffers, s, nR, resonance,
                              lower_order_couplings, external_dynamics)
    else
        _solve_monomial!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)
    end
end

# ==============================================================================
# Solve a single monomial
# ==============================================================================

# Singleton used by the no-sym public overload; skip_bits is never indexed inside
# solve_single_monomial!, so a zero-length BitVector is correct here.
const _NO_SYM = ConjugateSymmetryData{NoConjugatePermutation, Nothing}(
    NoConjugatePermutation(), Int[], BitVector(), nothing
)

"""
	solve_single_monomial!(W, R, idx, ctx, model, ml_cache) -> nothing

Solve the cohomological equations for the monomial with multiindex-set position
`idx`, updating the coefficients of `W` and `R` in-place.

See the module docstring for a description of the algorithm.
"""
function solve_single_monomial!(
        W, R, idx::Int, ctx, model, ml_cache
)
    solve_single_monomial!(W, R, idx, ctx, _NO_SYM, model, ml_cache)
end

# Canonical implementation: dispatches the solve step at compile time via sym.
function solve_single_monomial!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        idx::Int,
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData,
        model::NDOrderModel,
        ml_cache::MultilinearTermsCache
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    multi = multiindex_set(W)[idx]

    s = sum(multi[i] * ctx.lambda_diag[i] for i in 1:NVAR)
    resonance = _resonance_vector(ctx.resonance_set, idx, Val(ROM))

    for v in ctx.lower_order.buffer
        fill!(v, zero(T))
    end
    lower_order_couplings = compute_lower_order_couplings(
        multi, W, R,
        ctx.lower_order.multiindex_dict,
        ctx.lower_order.buffer,
        ctx.lower_order.candidate_indices[idx],
        ctx.lower_order.unit_vectors
    )

    compute_multilinear_terms!(ctx.buffers.ml_result, model, idx, W, ml_cache)

    external_dynamics = view(R.poly.coefficients, (ROM + 1):NVAR, idx)
    nR = count(resonance)
    n_sys = FOM + nR

    # Compile-time dispatch on RB; one runtime BitVector check when RB ≠ Nothing.
    _sym_solve_monomial!(ctx, sym, idx, s, nR, resonance,
                         lower_order_couplings, external_dynamics)

    sol = view(ctx.buffers.rhs, 1:n_sys)
    W.poly.coefficients[:, 1, idx] .= view(sol, 1:FOM)

    rr = 1
    for r in 1:ROM
        if resonance[r]
            R.poly.coefficients[r, idx] = sol[FOM + rr]
            rr += 1
        else
            R.poly.coefficients[r, idx] = zero(T)
        end
    end

    compute_higher_derivative_coefficients!(
        W.poly.coefficients,
        view(R.poly.coefficients, 1:ROM, :),
        external_dynamics, s, idx,
        ctx.generalised_eigenmodes, lower_order_couplings
    )
    return nothing
end

# ==============================================================================
# Solve all monomials
# ==============================================================================

"""
	solve_cohomological_equations!(W, R, ctx, model, ml_cache) -> nothing

Solve the cohomological equations for **all** monomials in the multiindex set
of `W` and `R`, processing them in *causal order* (ascending total degree).
"""
function solve_cohomological_equations!(W, R, ctx, model, ml_cache)
    nterms = length(multiindex_set(W))
    sym = _build_conjugate_symmetry(NoConjugatePermutation(), ctx.linear_monomial_skip_set, nterms)
    solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache)
end

# Overload without active symmetry: skip_bits covers only linear monomials; uses sym-aware
# solve_single_monomial! to enable compile-time dispatch on RB.
function solve_cohomological_equations!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData{NoConjugatePermutation, RB},
        model::NDOrderModel,
        ml_cache::MultilinearTermsCache
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT, RB}
    nterms = length(multiindex_set(W))
    for idx in 1:nterms
        @inbounds sym.skip_bits[idx] && continue
        solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
    end
    return nothing
end

# Overload with active symmetry: secondaries are in skip_bits; primaries are solved
# then their conjugate is filled via fill_conjugate_monomial!.
function solve_cohomological_equations!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData{<:SVector, RB},
        model::NDOrderModel,
        ml_cache::MultilinearTermsCache
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT, RB}
    nterms = length(multiindex_set(W))
    cmap = sym.monomial_map

    for idx in 1:nterms
        @inbounds sym.skip_bits[idx] && continue   # skips linears AND secondary monomials

        solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)

        j = cmap[idx]
        j > idx && fill_conjugate_monomial!(W, R, j, idx, sym)
    end
    return nothing
end

end # module CohomologicalEquations
