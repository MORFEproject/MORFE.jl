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
below never inverts `L(s)` alone, and why its numeric
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

# Source layout

The module is deliberately organised by responsibility:

| File | Responsibility |
|:-----|:---------------|
| `CohomologicalEquations.jl` | Mathematical assembly and solution of the bordered cohomological equations; module façade |
| `Configuration.jl` | User-facing solver and checkpoint options |
| `SolveState.jl` | Precomputed operators, reusable buffers, backend state, and the solve context |
| `Execution.jl` | Causal job planning, exact factor grouping, progress/checkpoint observers, and benchmark reporting |
| `ProblemSetup.jl` | End-to-end problem preparation and external-direction initialisation |
| `Checkpointing.jl` | Checkpoint format, validation, restoration, and atomic writes |
| `Symmetry.jl` | Conjugate-symmetry discovery bookkeeping and coefficient reconstruction |

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

include("Configuration.jl")
include("Checkpointing.jl")
include("SolveState.jl")
include("Symmetry.jl")
include("ProblemSetup.jl")

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
degree. Structurally equal groups use their first job's superharmonic for every member, so
the common resonance mask makes the bordered matrix exactly identical and its numeric
factorisation reusable. `:auto` uses grouping only when it reduces the number of
factorisations. Checkpoint persistence is installed internally by
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

# =============================================================================
# Solving the bordered cohomological system
#
# The system itself — its block structure, the resonance masking, and why the
# constant size is equivalent to the compacted one — is documented in the
# `CohomologicalEquations` module docstring.  What follows are the three properties
# the *solve* depends on; each constrains how the linear algebra may be done.
#
# 1. The bordered matrix is factorised whole; L(s) is never inverted on its own.
#    Inner resonance is flagged by |λ_r − s| < tol while det L(λ_r) = 0, so "α is
#    resonant" means precisely "L(s_α) is numerically singular" — and only resonant
#    monomials carry a border.  Forming L(s)⁻¹b and L(s)⁻¹C (bordering elimination)
#    therefore has backward error scaling with κ(L) → ∞ exactly where it would be
#    applied, and the two subsequent differences of large quantities cancel
#    catastrophically.  Factorising the bordered matrix keeps the backward error at
#    κ(M), which stays O(1) because the border spans L's near-null directions
#    (Keller's bordering lemma).
#
# 2. The numeric factorisation re-pivots on every monomial.  Property 1 holds only
#    if pivoting may move rows across the border, so a refactorisation that reuses a
#    frozen pivot sequence degrades precisely at the resonances.  `_refactorise!`
#    uses `klu_factor!`, which re-pivots while reusing the cached symbolic analysis;
#    KLU's exported `klu!` is `klu_refactor` and freezes the pivots.
#
# 3. No symmetry is declared to any backend.  The bordered pattern is structurally
#    symmetric whenever the L union pattern is — the border contributes a dense row
#    together with its matching dense column — which is the common FE case.  A
#    solver told to exploit that symmetry constrains its permutation to preserve it,
#    forfeiting the cross-border row interchange property 1 depends on.  Each
#    backend is left to analyse the matrix itself: no strategy is forced here, and
#    no symmetric matrix type is declared to Pardiso (see ext/MORFEPardisoExt.jl).
# =============================================================================

# =============================================================================
# Sparse factorise-and-solve
# =============================================================================

"""
	_refactorise_klu!(ss, A) -> KLU factorisation of `A`

Factorise the bordered matrix for the current monomial, reusing the symbolic analysis
cached in `ss.fact`.

The first call analyses and factorises; every later call redoes only the **numeric**
factorisation. Splitting them this way is what makes the constant-size formulation
pay off: the sparsity pattern is identical for every monomial, so the ordering and
symbolic phase — the expensive part — are computed once for the whole solve.

## Why `klu_factor!` and not `klu!`

The numeric phase must **re-pivot**. The bordered matrix changes value on every
monomial and its `(1,1)` block is near-singular at every resonance, where stability
depends on pivoting being free to exchange rows across the border. `klu_factor!`
re-pivots while reusing the cached symbolic analysis. KLU's exported `klu!` maps to
`klu_refactor`, which replays the pivot sequence chosen at the first monomial and so
loses accuracy exactly where it is needed.

## Preconditions and aliasing

`A`'s sparsity pattern must be identical on every call — [`SparseLinearSolverState`](@ref)
guarantees this by construction, since only `nzval` is ever written.

`klu` takes `A.nzval` by reference rather than copying it, so the factorisation and
the template share one value array: per-monomial assembly writes straight into what
KLU reads, with no copy-in step. `colptr`/`rowval` stay KLU's own (0-based) copies,
and are invariant in any case.

A factorisation is cached only if it succeeded, so a caller that catches the
singular-matrix error and retries gets a fresh analysis rather than a partially
initialised object. Singularity is reported through `issuccess` rather than an
exception (`check = false`, `allowsingular = true`) so that all monomials — including
the first — surface it through the same path.
"""
function _refactorise_klu!(ss::SparseLinearSolverState, A::SparseMatrixCSC)
    F = ss.fact
    if F === nothing
        # `check = false` so that a singular *first* monomial surfaces through the
        # caller's `issuccess` test like every later one, rather than throwing from a
        # different code path. `allowsingular` keeps KLU from halting inside the
        # factorisation so that `issuccess` is what decides.
        F = klu(A; check = false, allowsingular = true)
        # Only cache a factorisation that succeeded. A failed one is returned to the
        # caller (which raises) but not kept: if that error is ever caught and the
        # solve retried, the next call must redo the analysis rather than refactorise
        # from a half-initialised object.
        issuccess(F) && (ss.fact = F)
        return F
    end
    klu_factor!(F; check = false, allowsingular = true)
    return F
end

"""Backward-compatible internal alias for [`_refactorise_klu!`](@ref)."""
_refactorise!(ss::SparseLinearSolverState, A::SparseMatrixCSC) = _refactorise_klu!(ss, A)

"""
	_backward_error(state, x, norm_r, norm_b, norm_A) -> Real

Return the normwise backward error `norm_r / (norm_A * norm(x, Inf) + norm_b)`, guarded
against a zero denominator in the state's real scalar type.
"""
function _backward_error(ss::SparseLinearSolverState, x, norm_r, norm_b, norm_A)
    RT = typeof(ss.max_relative_residual)
    denominator = norm_A * norm(x, Inf) + norm_b
    return norm_r / max(denominator, floatmin(RT))
end

"""
	_sparse_inf_norm!(workspace, A) -> Real

Compute `norm(A, Inf)` for a CSC matrix using `workspace` for row sums and without
allocating a dense intermediate.
"""
function _sparse_inf_norm!(workspace, A::SparseMatrixCSC)
    fill!(workspace, zero(eltype(workspace)))
    @inbounds for column in axes(A, 2)
        for position in nzrange(A, column)
            workspace[A.rowval[position]] += abs(A.nzval[position])
        end
    end
    return maximum(real, workspace)
end

"""
	_sparse_residual!(residual, A, x) -> residual

Overwrite a vector containing `b` with `A*x - b` using SparseArrays' five-argument
`mul!` kernel. `residual` must not alias `A` or `x`.
"""
function _sparse_residual!(residual, A::SparseMatrixCSC, x)
    # Five-argument `mul!` computes A*x-b directly and uses SparseArrays' tuned
    # CSC kernel. `residual` contains b on entry and is safe as the output because
    # it does not alias either A or x.
    return mul!(residual, A, x, one(eltype(residual)), -one(eltype(residual)))
end

"""
	_klu_backward_error!(state, solution, norm_b) -> nothing

Check the normwise backward error of a KLU solution and, when necessary, perform up to
`state.max_refinement_steps` correction solves using the existing factorisation. Updates
the diagnostic maximum and refinement count, and throws if the tolerance is still missed.
"""
function _klu_backward_error!(ss::SparseLinearSolverState, x, norm_b)
    tolerance = ss.residual_tolerance
    tolerance === nothing && return nothing
    # `residual_work` contains the original right-hand side on entry. The five-argument
    # mul! forms A*x-b in the same vector, so the established KLU path pays for only one
    # persistent verification vector. A second vector is created only if refinement is
    # actually necessary.
    _sparse_residual!(ss.residual_work, ss.bordered, x)
    RT = typeof(ss.max_relative_residual)
    norm_r = norm(ss.residual_work, Inf)
    # Since ‖A‖∞‖x‖∞ + ‖b‖∞ ≥ ‖b‖∞, this inexpensive quantity is a rigorous
    # upper bound for the requested normwise backward error. Stable solves pass here,
    # avoiding a second sparse-matrix traversal and any additional workspace.
    relative = norm_r / max(norm_b, floatmin(RT))
    if relative > tolerance
        isempty(ss.refinement_work) && resize!(ss.refinement_work, length(x))
        copyto!(ss.refinement_work, ss.residual_work)
        norm_A = _sparse_inf_norm!(ss.residual_work, ss.bordered)
        relative = _backward_error(ss, x, norm_r, norm_b, norm_A)
        if relative > tolerance && ss.max_refinement_steps > 0
            # Reconstruct b = A*x-r in the persistent residual vector. The lazily
            # allocated refinement vector holds r and then each correction RHS.
            mul!(ss.residual_work, ss.bordered, x)
            ss.residual_work .-= ss.refinement_work
        end
        for _ in 1:ss.max_refinement_steps
            relative <= tolerance && break
            rmul!(ss.refinement_work, -one(eltype(x)))
            ldiv!(ss.fact, ss.refinement_work)
            x .+= ss.refinement_work
            ss.refinement_count += 1
            mul!(ss.refinement_work, ss.bordered, x)
            ss.refinement_work .-= ss.residual_work
            norm_r = norm(ss.refinement_work, Inf)
            relative = _backward_error(ss, x, norm_r, norm_b, norm_A)
        end
    end
    ss.max_relative_residual = max(ss.max_relative_residual, relative)
    relative <= tolerance || error(
        "bordered cohomological backward error $relative exceeds tolerance $tolerance")
    return nothing
end

"""
	_singular_bordered_system(s)

Throw an informative error for a singular bordered cohomological matrix.

The most important expected cause is an outer resonance: `s` lies on a non-master
eigenvalue, so the master border does not span the null direction of `L(s)`. Depending on
how the model and resonance data were assembled, rank-deficient operators, an insufficient
border, or numerical failure can produce the same factorisation result. Inspect
`ResonanceSet.outer_resonances` first, then verify the model and border ranks before
concluding that the master set must be enlarged.
"""
function _singular_bordered_system(s)
    error("""
      Singular bordered cohomological system at superharmonic s = $s.

      A likely cause is an outer resonance: s lies on a non-master eigenvalue and the
      master border does not span the null direction of L(s). Rank-deficient model
      operators, an insufficient border, or numerical failure can cause the same result.

      Check `ResonanceSet.outer_resonances` and the model/border ranks. If this is an outer
      resonance, enlarge the master set or lower the expansion order.""")
end

"""
	_bordered_solve!(ss, x, s) -> x

Solve `bordered * y = x` for `y`, overwriting `x` with the solution, and dispatching
to Pardiso when available and to the cached KLU factorisation otherwise.

Both branches reuse a symbolic analysis computed once: KLU through
[`_refactorise_klu!`](@ref), Pardiso through its phase split (`_pardiso_prepare!` once,
then numeric-factorise + solve per monomial).

KLU's `ldiv!` is genuinely in-place, so the KLU branch needs no intermediate buffer.
`ss.solve_scratch` exists for Pardiso, whose solve requires distinct input and output
arrays.
"""
function _bordered_solve!(
        ss::SparseLinearSolverState{T, <:KLUBackend{VERIFY}}, x::AbstractVector, s;
        reuse_factor::Val{REUSE} = Val(false)) where {T, REUSE, VERIFY}
    norm_b = zero(typeof(ss.max_relative_residual))
    if VERIFY
        copyto!(ss.residual_work, x)
        norm_b = norm(x, Inf)
    end
    if !REUSE || ss.fact === nothing
        F = _refactorise_klu!(ss, ss.bordered)
        issuccess(F) || _singular_bordered_system(s)
    end
    ldiv!(ss.fact, x)
    VERIFY && _klu_backward_error!(ss, x, norm_b)
    return x
end

function _bordered_solve!(
        ss::SparseLinearSolverState{T, <:PardisoBackend{P, VERIFY}}, x::AbstractVector, s;
        reuse_factor::Val{REUSE} = Val(false)) where {T, P, REUSE, VERIFY}
    # Pardiso's public phase interface always performs a numeric factorisation before
    # solving. Its one existing scratch vector preserves b and is reused for the
    # backward-error residual; no second persistent RHS copy is introduced.
    if ss.pardiso_matrix === nothing
        ss.pardiso_matrix = _pardiso_prepare!(ss.backend.solver, ss.bordered)
    end
    copyto!(ss.solve_scratch, x)
    norm_b = VERIFY ? norm(x, Inf) : zero(typeof(ss.max_relative_residual))
    if REUSE
        _pardiso_solve!(ss.backend.solver, ss.pardiso_matrix, x, ss.solve_scratch)
    else
        _pardiso_factorise_solve!(
            ss.backend.solver, ss.pardiso_matrix, x, ss.solve_scratch)
    end
    if VERIFY
        _sparse_residual!(ss.solve_scratch, ss.bordered, x)
        RT = typeof(ss.max_relative_residual)
        norm_r = norm(ss.solve_scratch, Inf)
        relative = norm_r / max(norm_b, floatmin(RT))
        if relative > ss.residual_tolerance
            isempty(ss.refinement_work) && resize!(ss.refinement_work, length(x))
            copyto!(ss.refinement_work, ss.solve_scratch)
            norm_A = _sparse_inf_norm!(ss.solve_scratch, ss.bordered)
            relative = _backward_error(ss, x, norm_r, norm_b, norm_A)
        end
        ss.max_relative_residual = max(ss.max_relative_residual, relative)
        relative <= ss.residual_tolerance || error(
            "bordered Pardiso backward error $relative exceeds tolerance $(ss.residual_tolerance)")
    end
    return x
end

# =============================================================================
# Shared bordered-system assembly (dense path)
# =============================================================================

"""
	_assemble_bordered_system!(ctx, s, resonance, lower_order_couplings, external_dynamics)

Assemble the `(FOM+ROM) × (FOM+ROM)` bordered cohomological system into
`ctx.buffers.system_matrix` and `ctx.buffers.rhs`.

- The first `FOM` rows come from the invariance equation (operator + nonlinear RHS).
- The last `ROM` rows come from the orthogonality conditions, with non-resonant modes
  contributing the trivial row `R[r, α] = 0`.

Called by the dense-path `_solve_monomial!`.
"""
function _assemble_bordered_system!(
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        s,
        resonance::SVector{ROM, Bool},
        lower_order_couplings,
        external_dynamics
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT, ROM}
    n_sys = FOM + ROM
    assemble_cohomological_matrix_and_rhs!(
        view(ctx.buffers.system_matrix, 1:FOM, 1:n_sys),
        view(ctx.buffers.rhs, 1:FOM),
        s, ctx.linear_terms,
        ctx.invariance.column_coeffs, ctx.invariance.E_coeffs,
        resonance, lower_order_couplings, external_dynamics,
        ctx.buffers.external_rhs
    )
    view(ctx.buffers.rhs, 1:FOM) .+= ctx.buffers.ml_result
    assemble_orthogonality_matrix_and_rhs!(
        view(ctx.buffers.system_matrix, (FOM + 1):n_sys, 1:n_sys),
        view(ctx.buffers.rhs, (FOM + 1):n_sys),
        s, ctx.orthogonality.J_coeffs,
        ctx.orthogonality.corner_coeffs, ctx.orthogonality.E_coeffs,
        resonance, lower_order_couplings, external_dynamics
    )
    return
end

# =============================================================================
# Dense-path monomial solve
# =============================================================================

"""
	_dense_backward_error(A, x, b) -> Real

Compute `norm(A*x-b, Inf) / (norm(A, Inf)*norm(x, Inf) + norm(b, Inf))` without
allocating a residual vector. The denominator is floored at the real scalar type's
smallest positive normal value.
"""
function _dense_backward_error(A, x, b)
    RT = typeof(real(zero(eltype(x))))
    norm_A = zero(RT)
    norm_x = norm(x, Inf)
    norm_b = norm(b, Inf)
    norm_r = zero(RT)
    @inbounds for row in axes(A, 1)
        row_sum = zero(RT)
        ax = zero(eltype(x))
        for column in axes(A, 2)
            value = A[row, column]
            row_sum += abs(value)
            ax += value * x[column]
        end
        norm_A = max(norm_A, row_sum)
        norm_r = max(norm_r, abs(ax - b[row]))
    end
    return norm_r / max(norm_A * norm_x + norm_b, floatmin(RT))
end

"""
	_solve_monomial!(ctx, s, resonance, lower_order_couplings, external_dynamics)

**Dense path.** Assemble the `(FOM+ROM)` bordered system via
`_assemble_bordered_system!`, then solve it in-place with `lu!` + `ldiv!`.
The solution is written into `ctx.buffers.rhs[1:FOM+ROM]`.
"""
function _solve_monomial!(
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        s,
        resonance::SVector{ROM, Bool},
        lower_order_couplings,
        external_dynamics,
        reuse_factor::Val{REUSE} = Val(false)
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT, ROM, REUSE}
    _assemble_bordered_system!(ctx, s, resonance, lower_order_couplings, external_dynamics)
    n_sys = FOM + ROM
    F = lu!(view(ctx.buffers.system_matrix, 1:n_sys, 1:n_sys), check = false)
    issuccess(F) || _singular_bordered_system(s)
    ldiv!(F, view(ctx.buffers.rhs, 1:n_sys))
    tolerance = ctx.buffers.residual_tolerance
    if tolerance !== nothing
        solution = ctx.buffers.dense_solution
        copyto!(solution, view(ctx.buffers.rhs, 1:n_sys))
        relative = typemax(typeof(tolerance))
        for refinement_step in 0:ctx.buffers.max_refinement_steps
            # Dense LU overwrites its matrix. Reassembly recovers the exact bordered
            # operator and right-hand side without retaining a second dense matrix.
            _assemble_bordered_system!(
                ctx, s, resonance, lower_order_couplings, external_dynamics)
            A = view(ctx.buffers.system_matrix, 1:n_sys, 1:n_sys)
            b = view(ctx.buffers.rhs, 1:n_sys)
            relative = _dense_backward_error(A, solution, b)
            relative <= tolerance && break
            refinement_step == ctx.buffers.max_refinement_steps && break
            isempty(ctx.buffers.dense_refinement) &&
                resize!(ctx.buffers.dense_refinement, n_sys)
            correction = ctx.buffers.dense_refinement
            mul!(correction, A, solution)
            @. correction = b - correction
            F = lu!(A, check = false)
            issuccess(F) || _singular_bordered_system(s)
            ldiv!(F, correction)
            solution .+= correction
        end
        relative <= tolerance || error(
            "bordered dense backward error $relative exceeds tolerance $tolerance")
        copyto!(view(ctx.buffers.rhs, 1:n_sys), solution)
    end
    return
end

# =============================================================================
# Canonical per-monomial pipeline
# =============================================================================

"""Internal marker selecting the allocation-free, uninstrumented solve path."""
struct _NoMonomialInstrumentation end
const _NO_MONOMIAL_INSTRUMENTATION = _NoMonomialInstrumentation()

"""
	_assemble_nonlinear_rhs!(instrumentation, ctx, model, idx, W, ml_cache)

Instrumentation hook for nonlinear right-hand-side assembly. The production method
returns `nothing`; benchmark instrumentation returns the corresponding `@timed` result.
"""
function _assemble_nonlinear_rhs!(::_NoMonomialInstrumentation,
        ctx, model, idx, W, ml_cache)
    compute_multilinear_terms!(ctx.buffers.ml_result, model, idx, W, ml_cache)
    return nothing
end

"""
	_solve_prepared_system!(instrumentation, ctx, s, resonance,
		lower_order_couplings, external_dynamics, reuse_factor)

Instrumentation hook for the bordered solve after all monomial-dependent inputs have been
prepared. `reuse_factor` may be true only for an exactly identical grouped matrix.
"""
function _solve_prepared_system!(::_NoMonomialInstrumentation,
        ctx, s, resonance, lower_order_couplings, external_dynamics,
        reuse_factor::Val)
    _solve_monomial!(
        ctx, s, resonance, lower_order_couplings, external_dynamics, reuse_factor)
    return nothing
end

"""
	_monomial_metrics(instrumentation, rhs_result, solve_result)

Convert instrumentation-specific phase results into the metrics delivered to solve
observers. The production path returns `nothing`.
"""
_monomial_metrics(::_NoMonomialInstrumentation, ::Nothing, ::Nothing) = nothing

"""
	_finalise_monomial!(W, R, idx, ctx, s, resonance, external_dynamics,
		lower_order_couplings) -> nothing

Unpack the bordered solution into the primary coefficients of `W` and the resonant rows of
`R`, write exact zeros to non-resonant master rows, and propagate the higher derivative
coefficients using the generalised right eigenmodes.
"""
function _finalise_monomial!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        idx::Int,
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM},
        s,
        resonance::SVector{ROM, Bool},
        external_dynamics,
        lower_order_couplings
) where {ORD, NVAR, T, ROM, FOM, ORDP1}
    n_sys = FOM + ROM
    sol = view(ctx.buffers.rhs, 1:n_sys)
    W.poly.coefficients[:, 1, idx] .= view(sol, 1:FOM)

    # Only resonant rows are read back. `R[r, alpha] = 0` on non-resonant modes is
    # the style choice defining the parametrisation, rather than a computed quantity.
    for r in 1:ROM
        R.poly.coefficients[r, idx] = resonance[r] ? sol[FOM + r] : zero(T)
    end

    compute_higher_derivative_coefficients!(
        W.poly.coefficients,
        view(R.poly.coefficients, 1:ROM, :),
        external_dynamics, s, idx,
        ctx.generalised_eigenmodes, lower_order_couplings)
    return nothing
end

"""
	_run_single_monomial!(instrumentation, W, R, idx, ctx, sym, model, ml_cache,
		reuse_factor, superharmonic = nothing) -> metrics

Canonical per-monomial pipeline shared by production and benchmark execution. Only
nonlinear-right-hand-side assembly and the bordered solve are instrumentation hooks;
preparation and coefficient finalisation stay outside benchmark timings.

Grouped execution supplies one canonical `superharmonic` for the entire structural
factor group. Direct and public single-monomial execution leave it as `nothing` and
compute the value from the monomial itself.
"""
function _run_single_monomial!(
        instrumentation,
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        idx::Int,
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        ::ConjugateSymmetryData,
        model::NthOrderModel,
        ml_cache::MultilinearTermsCache,
        reuse_factor::Val{REUSE} = Val(false),
        superharmonic = nothing
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT, REUSE}
    multi = multiindex_set(W)[idx]
    s = isnothing(superharmonic) ? _superharmonic(multi, ctx.lambda_diag) :
        superharmonic
    resonance = _resonance_vector(ctx.resonance_set, idx, Val(ROM))

    for buffer in ctx.lower_order.buffer
        fill!(buffer, zero(T))
    end
    lower_order_couplings = compute_lower_order_couplings(
        multi, W, R,
        ctx.lower_order.multiindex_dict,
        ctx.lower_order.buffer,
        ctx.lower_order.candidate_indices[idx],
        ctx.lower_order.unit_vectors)

    rhs_metrics = _assemble_nonlinear_rhs!(
        instrumentation, ctx, model, idx, W, ml_cache)
    external_dynamics = view(R.poly.coefficients, (ROM + 1):NVAR, idx)
    solve_metrics = _solve_prepared_system!(instrumentation,
        ctx, s, resonance, lower_order_couplings, external_dynamics, reuse_factor)

    _finalise_monomial!(W, R, idx, ctx, s, resonance,
        external_dynamics, lower_order_couplings)
    return _monomial_metrics(instrumentation, rhs_metrics, solve_metrics)
end

# Singleton used by the public no-symmetry overload. `skip_bits` is never indexed by
# the single-monomial pipeline, so a zero-length vector is sufficient.
const _NO_SYM = ConjugateSymmetryData{NoConjugatePermutation}(
    NoConjugatePermutation(), Int[], BitVector())

"""
	solve_single_monomial!(W, R, idx, ctx, model, ml_cache) -> nothing

Solve the cohomological equations for one multiindex-set position, updating `W` and `R`.
"""
function solve_single_monomial!(W, R, idx::Int, ctx, model, ml_cache)
    return solve_single_monomial!(W, R, idx, ctx, _NO_SYM, model, ml_cache)
end

function solve_single_monomial!(W, R, idx::Int, ctx, sym, model, ml_cache,
        reuse_factor::Bool)
    return reuse_factor ?
           solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache, Val(true)) :
           solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache, Val(false))
end

function solve_single_monomial!(W, R, idx::Int, ctx, sym, model, ml_cache,
        reuse_factor::Val = Val(false))
    _run_single_monomial!(_NO_MONOMIAL_INSTRUMENTATION,
        W, R, idx, ctx, sym, model, ml_cache, reuse_factor)
    return nothing
end

"""
	_solve_monomial!(ctx, s, resonance, lower_order_couplings, external_dynamics)

**Sparse path** (dispatched when `MT <: SparseMatrixCSC`). Writes the bordered matrix
into the constant-pattern template held by `ctx.sparse_solver` and solves it with at most
one new numeric factorisation. When `reuse_factor == Val(true)`, the existing factorisation
is reused because the caller has established exact matrix identity.

The `(1,1)` block is evaluated by the untouched `build_sparse_L_and_rhs!` Horner pass
on its own workspace — it needs the transient intermediates `L[j](s)` to accumulate
the lower-order RHS — and is then scattered into the template. The border blocks are
staged in `ctx.buffers.orthogonality_rows` by the same assembly routine the dense
path uses, then scattered into their strided positions in `nzval`.

Results are written into `ctx.buffers.rhs[1:FOM+ROM]`.
"""
function _solve_monomial!(
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        s,
        resonance::SVector{ROM, Bool},
        lower_order_couplings,
        external_dynamics,
        reuse_factor::Val{REUSE} = Val(false)
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT <: SparseMatrixCSC, ROM, REUSE}
    ss = ctx.sparse_solver::SparseLinearSolverState{T}
    M = ss.bordered
    Mnz = M.nzval
    n_sys = FOM + ROM

    # ── 1. L(s) block and the lower-order RHS accumulation ────────────────────
    rhs = view(ctx.buffers.rhs, 1:FOM)
    fill!(rhs, zero(T))
    build_sparse_L_and_rhs!(
        rhs, ss.L_template, ss.L_mappings,
        ctx.linear_terms, s, lower_order_couplings
    )
    scatter_L_into_bordered!(M, ss.L_template)

    # ── 2. External forcing and nonlinear contributions ───────────────────────
    evaluate_external_rhs!(rhs, s, external_dynamics, ctx.invariance.E_coeffs,
        ctx.buffers.external_rhs)
    rhs .+= ctx.buffers.ml_result

    # ── 3. Invariance border columns C(s)P — contiguous runs in nzval ─────────
    @inbounds for r in 1:ROM
        base = M.colptr[FOM + r]
        column = view(Mnz, base:(base + FOM - 1))
        if resonance[r]
            evaluate_column!(column, s, r, ctx.invariance.column_coeffs)
        else
            fill!(column, zero(T))
        end
    end

    # ── 4. Orthogonality rows, staged then scattered ──────────────────────────
    orth = view(ctx.buffers.orthogonality_rows, 1:ROM, 1:n_sys)
    assemble_orthogonality_matrix_and_rhs!(
        orth, view(ctx.buffers.rhs, (FOM + 1):n_sys), s,
        ctx.orthogonality.J_coeffs,
        ctx.orthogonality.corner_coeffs,
        ctx.orthogonality.E_coeffs,
        resonance, lower_order_couplings, external_dynamics
    )
    # P Ĵ(s): border rows of the first FOM columns (contiguous ROM-runs on both sides).
    @inbounds for c in 1:FOM
        base = ss.border_row_base[c]
        for r in 1:ROM
            Mnz[base + r - 1] = orth[r, c]
        end
    end
    # P Ĉ(s) P + τ Q: the ROM × ROM corner.
    @inbounds for q in 1:ROM
        base = M.colptr[FOM + q] + FOM
        for r in 1:ROM
            Mnz[base + r - 1] = orth[r, FOM + q]
        end
    end

    # ── 5. One factorisation, one solve ───────────────────────────────────────
    _bordered_solve!(
        ss, view(ctx.buffers.rhs, 1:n_sys), s; reuse_factor)
    return
end

include("Execution.jl")

end # module CohomologicalEquations
