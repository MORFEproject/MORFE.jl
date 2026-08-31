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
| [`InvarianceOperators`](@ref)       | Precomputed invariance-equation operator coefficients |
| [`OrthogonalityOperators`](@ref)    | Precomputed orthogonality-condition operator coefficients |
| [`LowerOrderResources`](@ref)       | Lower-order coupling data and buffers |
| [`CohomologicalBuffers`](@ref)      | Pre-allocated system-assembly scratch buffers |
| [`SparseLinearSolverState`](@ref)   | Sparse-path bordered template, index maps and solver handles |
| [`CohomologicalContext`](@ref)      | Composed struct bundling all precomputed operators and resources |
| [`solve_single_monomial!`](@ref)    | Solve the cohomological system for one multi-index |
| [`solve_cohomological_equations!`](@ref) | Solve for all multi-indices in causal order |
| [`solve_cohomological_problem`](@ref)    | High-level driver: precompute everything and solve |
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

_try_build_pardiso_solver(::Vararg{Any}) = nothing
_pardiso_prepare!(args...) = error(_PARDISO_INACTIVE)          # configure + phase 11, once
_pardiso_factorise_solve!(args...) = error(_PARDISO_INACTIVE)  # phases 22 + 33, per monomial
_pardiso_solve!(args...) = error(_PARDISO_INACTIVE)            # phase 33, reused factor group
_pardiso_release!(args...) = nothing                           # phase -1, on finalisation

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
       CohomologicalSolverConfig,
       CohomologicalCheckpoint,
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
	_make_progress(n_total, show_progress) -> _SimpleProgress

Construct a `_SimpleProgress` tracker.  Disables output automatically when
`stderr` is not a TTY (e.g. in CI or when piped).
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

# ==============================================================================
# Solve a single monomial
# ==============================================================================

# Singleton used by the no-sym public overload; skip_bits is never indexed inside
# solve_single_monomial!, so a zero-length BitVector is correct here.
const _NO_SYM = ConjugateSymmetryData{NoConjugatePermutation}(
    NoConjugatePermutation(), Int[], BitVector(), NTuple{2, Int}[]
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

function solve_single_monomial!(W, R, idx::Int, ctx, sym, model, ml_cache,
        reuse_factor::Bool)
    reuse_factor ?
    solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache, Val(true)) :
    solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache, Val(false))
end

# Canonical implementation: dispatches the solve step at compile time via sym.
function solve_single_monomial!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        idx::Int,
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        ::ConjugateSymmetryData,
        model::NthOrderModel,
        ml_cache::MultilinearTermsCache,
        reuse_factor::Val{REUSE} = Val(false)
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT, REUSE}
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
    n_sys = FOM + ROM

    _solve_monomial!(
        ctx, s, resonance, lower_order_couplings, external_dynamics, reuse_factor)

    sol = view(ctx.buffers.rhs, 1:n_sys)
    W.poly.coefficients[:, 1, idx] .= view(sol, 1:FOM)

    # Only resonant rows are read back.  `R[r, α] = 0` on non-resonant modes is the
    # style choice that *defines* the parametrisation, not a computed quantity, so it
    # is written directly rather than taken from the trivial rows of the solve — the
    # hard zeros must not be able to pick up round-off from a pivot ordering, a row
    # scaling, or a change of factoriser.
    for r in 1:ROM
        R.poly.coefficients[r, idx] = resonance[r] ? sol[FOM + r] : zero(T)
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
function solve_cohomological_equations!(
        W, R, ctx, model, ml_cache; show_progress::Bool = true,
        grouping::Symbol = :off, checkpoint_callback = nothing)
    nterms = length(multiindex_set(W))
    sym = _build_conjugate_symmetry(NoConjugatePermutation(), ctx.linear_monomial_skip_set, nterms)
    solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache;
        show_progress, grouping, checkpoint_callback)
end

struct StructuralFactorKey{NVAR, ROM}
    exponents::SVector{NVAR, Int}
    resonance::SVector{ROM, Bool}
end

function _eigenvalue_representatives(lambda_diag)
    representatives = collect(eachindex(lambda_diag))
    for i in eachindex(lambda_diag)
        if iszero(lambda_diag[i])
            representatives[i] = 0
            continue
        end
        for j in firstindex(lambda_diag):(i - 1)
            if isequal(lambda_diag[i], lambda_diag[j])
                representatives[i] = representatives[j]
                break
            end
        end
    end
    return representatives
end

function _has_structural_factor_reuse(lambda_diag)
    for i in eachindex(lambda_diag)
        iszero(lambda_diag[i]) && return true
        for j in firstindex(lambda_diag):(i - 1)
            isequal(lambda_diag[i], lambda_diag[j]) && return true
        end
    end
    return false
end

function _structural_factor_key(multi::SVector{NVAR, Int},
        resonance::SVector{ROM, Bool}, representatives) where {NVAR, ROM}
    exponents = zeros(MVector{NVAR, Int})
    for i in 1:NVAR
        representative = representatives[i]
        representative == 0 || (exponents[representative] += multi[i])
    end
    return StructuralFactorKey(SVector(exponents), resonance)
end

function _cohomological_groups(ctx, sym, mset, grouping::Symbol, ::Val{ROM}) where {ROM}
    grouping == :off && return nothing
    grouping == :auto && !_has_structural_factor_reuse(ctx.lambda_diag) && return nothing
    representatives = _eigenvalue_representatives(ctx.lambda_diag)

    NVAR = length(ctx.lambda_diag)
    Key = StructuralFactorKey{NVAR, ROM}
    ordered_groups = Vector{Vector{Int}}()
    degrees = sort!(unique(sum(mset[idx])
    for idx in eachindex(mset.exponents)
    if !sym.skip_bits[idx]))
    for degree in degrees
        keys = Key[]
        groups = Dict{Key, Vector{Int}}()
        for idx in eachindex(mset.exponents)
            sym.skip_bits[idx] && continue
            sum(mset[idx]) == degree || continue
            resonance = _resonance_vector(ctx.resonance_set, idx, Val(ROM))
            key = _structural_factor_key(mset[idx], resonance, representatives)
            haskey(groups, key) || (groups[key] = Int[]; push!(keys, key))
            push!(groups[key], idx)
        end
        append!(ordered_groups, (groups[key] for key in keys))
    end
    grouping == :auto && length(ordered_groups) == count(!, sym.skip_bits) && return nothing
    return ordered_groups
end

function _checkpoint_event!(callback, event, degree, indices = Int[])
    isnothing(callback) || callback(event, degree, indices)
    return nothing
end

# Overload without active symmetry: skip_bits covers only linear monomials; uses sym-aware
# solve_single_monomial! to enable compile-time dispatch on RB.
function solve_cohomological_equations!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData{NoConjugatePermutation},
        model::NthOrderModel,
        ml_cache::MultilinearTermsCache;
        show_progress::Bool = true,
        grouping::Symbol = :off,
        checkpoint_callback = nothing
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    nterms = length(multiindex_set(W))
    groups = _cohomological_groups(ctx, sym, multiindex_set(W), grouping, Val(ROM))
    n_to_solve = count(!, sym.skip_bits)
    prog = _make_progress(n_to_solve, show_progress, model.max_nl_degree)
    n_done = 0
    current_degree = 0
    if groups === nothing
        # The established allocation-free GrLex path remains untouched when no exact
        # structural factor reuse is available.
        if isnothing(checkpoint_callback)
            for idx in 1:nterms
                @inbounds sym.skip_bits[idx] && continue
                degree = sum(multiindex_set(W)[idx])
                solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
                n_done += 1
                _progress_tick!(prog, n_done, degree)
            end
        else
            for idx in 1:nterms
                @inbounds sym.skip_bits[idx] && continue
                degree = sum(multiindex_set(W)[idx])
                current_degree != 0 && degree != current_degree &&
                    _checkpoint_event!(checkpoint_callback, :degree, current_degree)
                current_degree = degree
                solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
                _checkpoint_event!(checkpoint_callback, :group, degree, [idx])
                n_done += 1
                _progress_tick!(prog, n_done, degree)
            end
        end
    else
        for group in groups
            for (position, idx) in enumerate(group)
                degree = sum(multiindex_set(W)[idx])
                current_degree != 0 && degree != current_degree &&
                    _checkpoint_event!(checkpoint_callback, :degree, current_degree)
                current_degree = degree
                if position == 1
                    solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
                else
                    solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache, Val(true))
                end
                n_done += 1
                _progress_tick!(prog, n_done, degree)
            end
            _checkpoint_event!(checkpoint_callback, :group, current_degree, group)
        end
    end
    isnothing(checkpoint_callback) || current_degree == 0 ||
        _checkpoint_event!(checkpoint_callback, :degree, current_degree)
    _progress_done!(prog, n_done)
    return nothing
end

# Overload with active symmetry: secondaries are in skip_bits; primaries are solved
# then their conjugate is filled via fill_conjugate_monomial!.
function solve_cohomological_equations!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData{<:SVector},
        model::NthOrderModel,
        ml_cache::MultilinearTermsCache;
        show_progress::Bool = true,
        grouping::Symbol = :off,
        checkpoint_callback = nothing
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    pairs = sym.primary_pairs
    groups = _cohomological_groups(ctx, sym, multiindex_set(W), grouping, Val(ROM))
    n_to_solve = count(!, sym.skip_bits)
    prog = _make_progress(n_to_solve, show_progress, model.max_nl_degree)
    n_done = 0
    current_degree = 0
    if groups === nothing
        ptr = 1
        if isnothing(checkpoint_callback)
            for idx in eachindex(multiindex_set(W).exponents)
                @inbounds sym.skip_bits[idx] && continue
                degree = sum(multiindex_set(W)[idx])
                solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
                n_done += 1
                _progress_tick!(prog, n_done, degree)
                while ptr <= length(pairs) && @inbounds sym.skip_bits[pairs[ptr][1]]
                    ptr += 1
                end
                if ptr <= length(pairs) && @inbounds pairs[ptr][1] == idx
                    src, dst = @inbounds pairs[ptr]
                    fill_conjugate_monomial!(W, R, dst, src, sym)
                    ptr += 1
                end
            end
        else
            for idx in eachindex(multiindex_set(W).exponents)
                @inbounds sym.skip_bits[idx] && continue
                degree = sum(multiindex_set(W)[idx])
                current_degree != 0 && degree != current_degree &&
                    _checkpoint_event!(checkpoint_callback, :degree, current_degree)
                current_degree = degree
                solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
                n_done += 1
                _progress_tick!(prog, n_done, degree)
                while ptr <= length(pairs) && @inbounds sym.skip_bits[pairs[ptr][1]]
                    ptr += 1
                end
                if ptr <= length(pairs) && @inbounds pairs[ptr][1] == idx
                    src, dst = @inbounds pairs[ptr]
                    fill_conjugate_monomial!(W, R, dst, src, sym)
                    _checkpoint_event!(checkpoint_callback, :group, degree, [src, dst])
                    ptr += 1
                else
                    _checkpoint_event!(checkpoint_callback, :group, degree, [idx])
                end
            end
        end
    else
        secondary_for_primary = Dict(src => dst for (src, dst) in pairs)
        for group in groups
            for (position, idx) in enumerate(group)
                degree = sum(multiindex_set(W)[idx])
                current_degree != 0 && degree != current_degree &&
                    _checkpoint_event!(checkpoint_callback, :degree, current_degree)
                current_degree = degree
                if position == 1
                    solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
                else
                    solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache, Val(true))
                end
                n_done += 1
                _progress_tick!(prog, n_done, degree)
                if haskey(secondary_for_primary, idx)
                    fill_conjugate_monomial!(W, R, secondary_for_primary[idx], idx, sym)
                end
            end
            chunk_indices = copy(group)
            append!(chunk_indices,
                (secondary_for_primary[idx]
                for idx in group
                if haskey(secondary_for_primary, idx)))
            sort!(unique!(chunk_indices))
            _checkpoint_event!(checkpoint_callback, :group, current_degree, chunk_indices)
        end
    end
    isnothing(checkpoint_callback) || current_degree == 0 ||
        _checkpoint_event!(checkpoint_callback, :degree, current_degree)
    _progress_done!(prog, n_done)
    return nothing
end

include("CohomologicalBenchmark.jl")

end # module CohomologicalEquations
