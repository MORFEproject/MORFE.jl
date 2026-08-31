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

Let `s = ⟨λ, α⟩` be the superharmonic frequency and let `P = diag(ρ)` be the
diagonal `0/1` matrix of the master modes resonant at `α`, `Q = I − P`.  The
cohomological linear system is **bordered** and of **constant size** `FOM + ROM`:

```
┌                          ┐ ┌         ┐   ┌             ┐
│  L(s)     C(s) P         │ │  W[α]   │ = │  RHS_inv    │   FOM rows  (invariance)
│  P Ĵ(s)   P Ĉ(s) P + τQ  │ │  R[α]   │   │  P RHS_ort  │   ROM rows  (orthogonality)
└                          ┘ └         ┘   └             ┘
```

where:
- `L(s)`  (`FOM × FOM`) is the parametrisation operator,
- `C(s)`  (`FOM × ROM`) acts on the unknown reduced-dynamics coefficients,
- `Ĵ(s)` (`ROM × FOM`) is the orthogonality row operator, stacking the rows `Ĵ_r(s)`,
- `Ĉ(s)` (`ROM × ROM`) is the orthogonality joint operator,
- `W[α] ∈ ℂᶠᵒᵐ` is the parametrisation coefficient,
- `R[α] ∈ ℂᴿᴼᴹ` are the master reduced-dynamics coefficients,
- `τ = 1` on the non-resonant diagonal, so those rows read `R[r, α] = 0`.

Non-resonant master modes stay in the system as trivial equations rather than being
compacted away.  That is what keeps the size — and on the sparse path the sparsity
pattern — independent of `α`, so a single symbolic factorisation serves every
monomial.  It costs nothing in accuracy: permuting the unknowns as
`[W; R_res; R_non]` makes the matrix block triangular with an exactly decoupled `τI`
block, and each trivial row is a singleton in both its row and its column.  External
forcing modes are *known* and appear only on the right-hand side.

## Why the system is bordered rather than reduced

The border is not an optimisation but a conditioning requirement.  Inner resonance is
flagged by `|λ_r − s| < tol`, and `det L(λ_r) = 0` for every master eigenvalue, so
"monomial `α` is resonant" means precisely "`L(s_α)` is numerically singular" — and a
border is present only on exactly those monomials.  Eliminating `L(s)` first, by
forming `L(s)⁻¹·RHS` and `L(s)⁻¹C(s)`, would therefore apply an inverse that does not
usefully exist, with backward error growing like `κ(L(s_α)) → ∞`.

Factorising the bordered matrix as a whole keeps the backward error at `κ` of the
*bordered* matrix, which stays bounded because the border spans the near-null
directions of `L(s)` (Keller's bordering lemma).  This is why the linear algebra 
in `CohomologicalSolver.jl` never inverts `L(s)` alone, and why its numeric 
factorisation must re-pivot on every monomial.

---

# Module contents

| Symbol | Description |
|:-------|:------------|
| [`ParametrisationOptions`](@ref)      | Solver, grouping, verification, validation and output policy |
| [`CheckpointOptions`](@ref)           | Durable checkpoint and resume policy |
| [`InvarianceOperators`](@ref)       | Precomputed invariance-equation operator coefficients |
| [`OrthogonalityOperators`](@ref)    | Precomputed orthogonality-condition operator coefficients |
| [`LowerOrderResources`](@ref)       | Lower-order coupling data and buffers |
| [`CohomologicalBuffers`](@ref)      | Pre-allocated system-assembly scratch buffers |
| [`SparseLinearSolverState`](@ref)   | Sparse-path bordered template, index maps and solver handles |
| [`CohomologicalContext`](@ref)      | Composed struct bundling all precomputed operators and resources |
| [`solve_single_monomial!`](@ref)    | Solve the cohomological system for one multi-index |
| [`solve_cohomological_equations!`](@ref) | Solve for all multi-indices in causal order |
| [`solve_cohomological_equations_benchmarked!`](@ref) | Instrumented solve that writes timing CSV files |
| [`solve_cohomological_problem`](@ref)    | High-level driver: precompute everything and solve |

Conjugate symmetry is represented by [`ConjugateSymmetryData`](@ref). Sparse backends
share the extension interface described by [`_try_build_pardiso_solver`](@ref) and the
four `_pardiso_*` phase hooks.
"""
module CohomologicalEquations

using ..Multiindices: MultiindexSet, indices_in_box_with_bounded_degree,
                      build_exponent_index_map
using ..Polynomials: DensePolynomial
using ..ParametrisationObjects: validate_multiindex_set,
                                Parametrisation, ReducedDynamics,
                                create_parametrisation_method_objects,
                                compute_higher_derivative_coefficients!,
                                multiindex_set
using ..LowerOrderCouplings: compute_lower_order_couplings
using ..InvarianceEquation: assemble_cohomological_matrix_and_rhs!,
                            precompute_master_column_polynomials,
                            precompute_external_column_polynomials,
                            build_sparse_L_and_rhs!,
                            precompute_sparse_L_template,
                            precompute_sparse_bordered_template,
                            scatter_L_into_bordered!,
                            evaluate_column!,
                            evaluate_external_rhs!
using ..MasterModeOrthogonality: assemble_orthogonality_matrix_and_rhs!,
                                 precompute_orthogonality_operator_coefficients,
                                 precompute_orthogonality_column_polynomials
using ..FullOrderModel: NthOrderModel
using ..ExternalSystems: ExternalSystem, external_basis
# Conjugate-involution detection lives in the spectral layer (it is a statement about
# the spectrum and the external system, not about the solve). Imported here so the
# driver can use it, and re-exported below so the public names are unchanged.
using ..SpectralDecomposition: detect_conjugate_permutation,
                               external_conjugate_permutation,
                               full_conjugate_permutation
using ..SpectralDecomposition: SpectralData, right_modes, left_modes,
                               right_mode_derivatives, left_mode_blocks,
                               master_eigenvalues, master_conjugate_permutation

using ..MultilinearTerms: compute_multilinear_terms, compute_multilinear_terms!,
                          build_multilinear_terms_cache, MultilinearTermsCache
using ..Resonance: ResonanceSet, is_resonant
using LinearAlgebra
using SparseArrays
using SHA
using TOML
# `klu_factor!` is deliberately not the exported `klu!`. `klu!` maps to
# `klu_refactor`, which reuses the pivot sequence from the first factorisation;
# `klu_factor!` re-pivots while reusing the cached symbolic analysis, which is what a
# value-varying `s` requires. See `_refactorise!`.
using KLU: klu, klu_factor!
using StaticArrays: SVector, MVector

# Extension hooks: overridden by ext/MORFEPardisoExt.jl when Pardiso is loaded.
#
# The split mirrors Pardiso's own phases so the symbolic analysis is computed once
# and reused, which `Pardiso.solve` cannot do — it runs analysis + factorisation +
# solve and then releases everything on every call.
const _PARDISO_INACTIVE = "Pardiso solver object present but MORFEPardisoExt not active — internal error."

"""
	_try_build_pardiso_solver() -> solver or nothing

Extension hook implemented by `MORFEPardisoExt`. Return a configured Pardiso solver when
the extension is active, or `nothing` in the core package so `:auto` can fall back to KLU.
This is an extension interface, not an end-user solver-selection API.
"""
_try_build_pardiso_solver(::Vararg{Any}) = nothing

"""
	_pardiso_prepare!(solver, bordered) -> backend_matrix

Extension hook for Pardiso configuration and symbolic analysis (phase 11). It is called
once per sparse solver state and returns the matrix representation required by later phases.
"""
_pardiso_prepare!(args...) = error(_PARDISO_INACTIVE)

"""
	_pardiso_factorise_solve!(solver, matrix, solution, rhs) -> nothing

Extension hook that performs Pardiso numeric factorisation and solve (phases 22 and 33)
for the first monomial in an exact factor-reuse group.
"""
_pardiso_factorise_solve!(args...) = error(_PARDISO_INACTIVE)

"""
	_pardiso_solve!(solver, matrix, solution, rhs) -> nothing

Extension hook that reuses the current Pardiso numeric factorisation (phase 33) for a
subsequent monomial whose bordered matrix is exactly identical.
"""
_pardiso_solve!(args...) = error(_PARDISO_INACTIVE)

"""
	_pardiso_release!(solver, matrix) -> nothing

Extension hook that releases Pardiso's external factorisation storage (phase -1). The core
fallback is a no-op so finalisation remains safe when the extension is inactive.
"""
_pardiso_release!(args...) = nothing

export _try_build_pardiso_solver, _pardiso_prepare!, _pardiso_factorise_solve!,
       _pardiso_solve!, _pardiso_release!

include("SolverConfiguration.jl")
include("CheckpointIO.jl")
include("OperatorData.jl")
include("SolverResources.jl")
include("CohomologicalContext.jl")
include("ConjugateSymmetry.jl")
include("CohomologicalSolver.jl")
include("CohomologicalDriver.jl")

export CohomologicalContext,
       ParametrisationOptions,
       CheckpointOptions,
       checkpoint_fingerprint_data,
       InvarianceOperators,
       OrthogonalityOperators,
       LowerOrderResources,
       CohomologicalBuffers,
       SparseLinearSolverState,
       NoConjugatePermutation,
       ConjugateSymmetryData,
       fill_conjugate_monomial!,
       detect_conjugate_permutation,
       external_conjugate_permutation,
       full_conjugate_permutation,
       solve_cohomological_equations!,
       solve_cohomological_equations_benchmarked!,
       solve_single_monomial!,
       solve_cohomological_problem

# ==============================================================================
# Progress indicator (stderr, \r-based, no external dependencies)
# ==============================================================================

"""
	_SimpleProgress

Lightweight progress state for the `\r`-based terminal progress indicator.

# Fields

- `n_total::Int` — number of monomials that will actually be solved, which is fewer
  than the multiindex-set size whenever linear or conjugate-secondary monomials are
  skipped.  The reported fraction is against this, not against the set size.
- `enabled::Bool` — `false` when `stderr` is not a TTY, so CI logs stay clean
  without every call site having to test for it.
- `max_nl_degree::Int` — highest nonlinearity degree in the model.  Work per
  monomial grows with degree, so the fraction is raised to this power to make the
  displayed percentage track elapsed time rather than monomial count.
"""
struct _SimpleProgress
    n_total::Int
    enabled::Bool
    max_nl_degree::Int
end

"""
	_make_progress(n_total, show_progress, max_nl_degree) -> _SimpleProgress

Construct a `_SimpleProgress` tracker. `max_nl_degree` controls the work-weighted
percentage. Output is disabled automatically when `stderr` is not a TTY.
"""
function _make_progress(n_total::Int, show_progress::Bool, max_nl_degree::Int)
    return _SimpleProgress(n_total, show_progress && stderr isa Base.TTY, max_nl_degree)
end

"""
	_progress_tick!(p, n_done, degree)

Print an in-place `\r`-overwritten progress line to `stderr` showing the
current polynomial degree and the fraction of monomials solved.
No-op when `p.enabled == false`.
"""
function _progress_tick!(p::_SimpleProgress, n_done::Int, degree::Int)
    p.enabled || return
    percentage = round(100.0 * (n_done / p.n_total)^p.max_nl_degree; digits = 2)
    print(stderr,
        "\rSolving: order $degree \t Monomials: $n_done/$(p.n_total) \t Progress: $percentage%   "
    )
end

"""
	_progress_done!(p, n_done)

Print the final "Solved N monomials." completion line to `stderr` and clear
any trailing characters from the last `_progress_tick!` call.
No-op when `p.enabled == false`.
"""
function _progress_done!(p::_SimpleProgress, n_done::Int)
    p.enabled || return
    println(stderr, "\rSolved $n_done monomials." * " "^50)
end

# ==============================================================================
# Utility helpers (public — used by solve_single_monomial! and the driver)
# ==============================================================================

"""
	_embed_external_dynamics!(R, ext_poly, mset)

Copy coefficients from the `N_EXT`-variable external polynomial `ext_poly` into
the last `N_EXT` rows of `R`'s coefficient matrix, embedding them into the full
`NVAR = ROM + N_EXT` multiindex set `mset`.  No-op when `N_EXT == 0`.

Throws an `ArgumentError` when a *non-zero* external coefficient has no home in `mset`.
Silently skipping it — as this did previously — drops part of the external dynamics
without trace.  That is a live risk for a re-based external system: a change of external
coordinates turns `r₁²` into `r₁², r₁r₂, r₂²`, and a per-parameter box `mset` (as built by
`all_multiindices_in_box`) need not contain the cross terms even though it is downward
closed.  A monomial whose coefficient is exactly zero is skipped, since dropping it changes
nothing.
"""
function _embed_external_dynamics!(
        R::ReducedDynamics{ROM, NVAR, T},
        ext_poly::DensePolynomial{T, N_EXT, 2},
        mset::MultiindexSet{NVAR}
) where {ROM, NVAR, T, N_EXT}
    N_EXT > 0 || return nothing
    mdict = build_exponent_index_map(mset)
    ext_coeffs = ext_poly.coefficients
    for (j, α_ext) in enumerate(ext_poly.multiindex_set.exponents)
        α_full = SVector{NVAR, Int}(ntuple(i -> i <= ROM ? 0 : α_ext[i - ROM], Val(NVAR)))
        idx_full = get(mdict, α_full, nothing)
        if idx_full === nothing
            nz = [(e, ext_coeffs[e, j]) for e in 1:N_EXT if !iszero(ext_coeffs[e, j])]
            isempty(nz) && continue
            throw(ArgumentError("""
                External dynamics carry the monomial r^$(Tuple(α_ext)) with non-zero \
                coefficients $(nz) (as (row, value)), but the multiindex set has no entry \
                for the corresponding full exponent $(Tuple(α_full)).
                That term would be dropped without trace.  Enlarge `mset` to contain it — \
                a re-based external system can generate additional cross monomials.
                """))
        end
        for e in 1:N_EXT
            coeff = T(ext_coeffs[e, j])
            iszero(coeff) || (R.poly.coefficients[ROM + e, idx_full] = coeff)
        end
    end
    return nothing
end

"""
	_linear_monomial_indices(mset) -> Vector{Int}

Return the positions in `mset` of all unit-vector monomials `eᵣ` for `r = 1 … NVAR`
(i.e., the linear monomials).  Used to identify which entries of `W` and `R` are
initialised from eigenvectors rather than solved.
"""
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

"""
	_resonance_vector(resonance_set, monomial_idx, ::Val{ROM}) -> SVector{ROM, Bool}

Return a compile-time-sized boolean vector indicating which of the `ROM` master
modes are resonant with the monomial at position `monomial_idx` in the multiindex
set.  Using `Val{ROM}` enables the compiler to emit a fully unrolled ntuple loop.
"""
@inline function _resonance_vector(
        resonance_set::ResonanceSet,
        monomial_idx::Int,
        ::Val{ROM}
) where {ROM}
    return SVector{ROM, Bool}(ntuple(r -> is_resonant(resonance_set, monomial_idx, r), Val(ROM)))
end

# Per-monomial execution and all-monomial scheduling live in the dedicated includes.

"""
	solve_cohomological_equations!(W, R, ctx, model, ml_cache) -> nothing
	solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache;
		show_progress = true, grouping = :off) -> nothing

Solve the cohomological equations for **all** monomials in the multiindex set
of `W` and `R`, processing them in causal order (ascending total degree).

The overload without `sym` disables conjugate reconstruction. With
[`ConjugateSymmetryData`](@ref), secondary monomials are skipped and filled from their
solved primaries. `grouping` accepts `:off`, `:on`, or `:auto`: groups never cross a total
degree and reuse a numeric factorisation only when the superharmonic and resonance mask
make the bordered matrix exactly identical. `:auto` uses grouping only when it reduces the
number of factorisations. Checkpoint persistence is installed internally by
[`solve_cohomological_problem`](@ref), rather than exposed as a callback on this low-level
entry point.
"""
function solve_cohomological_equations!(
        W, R, ctx, model, ml_cache; show_progress::Bool = true,
        grouping::Symbol = :off)
    nterms = length(multiindex_set(W))
    sym = _build_conjugate_symmetry(NoConjugatePermutation(), ctx.linear_monomial_skip_set, nterms)
    return solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache;
        show_progress, grouping)
end

function solve_cohomological_equations!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData{NoConjugatePermutation},
        model::NthOrderModel,
        ml_cache::MultilinearTermsCache;
        show_progress::Bool = true,
        grouping::Symbol = :off
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    _solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache;
        show_progress, grouping)
    return nothing
end

function solve_cohomological_equations!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData{<:SVector},
        model::NthOrderModel,
        ml_cache::MultilinearTermsCache;
        show_progress::Bool = true,
        grouping::Symbol = :off
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    _solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache;
        show_progress, grouping)
    return nothing
end

include("CohomologicalSchedule.jl")
include("CohomologicalBenchmark.jl")

end # module CohomologicalEquations
