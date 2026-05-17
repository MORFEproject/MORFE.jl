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
	ThreadedMonoSolveResources,
	NoConjugatePermutation,
	ConjugateSymmetryData,
	fill_conjugate_monomial!,
	detect_conjugate_permutation,
	solve_cohomological_equations!,
	solve_cohomological_equations_threaded!,
	solve_single_monomial!,
	solve_cohomological_problem

# ==============================================================================
# Progress indicator (stderr, \r-based, no external dependencies)
# ==============================================================================

"""
	_SimpleProgress

Lightweight progress state for the `\r`-based terminal progress indicator.
`enabled` is set to `false` when `stderr` is not a TTY so that CI logs stay clean.
"""
struct _SimpleProgress
	n_total::Int
	enabled::Bool
end

"""
	_make_progress(n_total, show_progress) -> _SimpleProgress

Construct a `_SimpleProgress` tracker.  Disables output automatically when
`stderr` is not a TTY (e.g. in CI or when piped).
"""
function _make_progress(n_total::Int, show_progress::Bool)
	return _SimpleProgress(n_total, show_progress && stderr isa Base.TTY)
end

"""
	_progress_tick!(p, n_done, degree)

Print an in-place `\r`-overwritten progress line to `stderr` showing the
current polynomial degree and the fraction of monomials solved.
No-op when `p.enabled == false`.
"""
function _progress_tick!(p::_SimpleProgress, n_done::Int, degree::Int)
	p.enabled || return
	percentage = 100.0 * n_done / p.n_total
	print(stderr,
		"\rSolving: order $degree \t Progress: $n_done/$(p.n_total) ",
		"($(round(percentage; digits = 2))%)   ")
end

"""
	_progress_done!(p, n_done)

Print the final "Solved N monomials." completion line to `stderr` and clear
any trailing characters from the last `_progress_tick!` call.
No-op when `p.enabled == false`.
"""
function _progress_done!(p::_SimpleProgress, n_done::Int)
	p.enabled || return
	println(stderr, "\rSolved $n_done monomials." * " "^30)
end

# ==============================================================================
# Utility helpers (public — used by solve_single_monomial! and the driver)
# ==============================================================================

"""
	_embed_external_dynamics!(R, ext_poly, mset)

Copy coefficients from the `N_EXT`-variable external polynomial `ext_poly` into
the last `N_EXT` rows of `R`'s coefficient matrix, embedding them into the full
`NVAR = ROM + N_EXT` multiindex set `mset`.  No-op when `N_EXT == 0`.
"""
function _embed_external_dynamics!(
	R::ReducedDynamics{ROM, NVAR, T},
	ext_poly::DensePolynomial{T, N_EXT, 2},
	mset::MultiindexSet{NVAR},
) where {ROM, NVAR, T, N_EXT}
	N_EXT > 0 || return nothing
	mdict = build_exponent_index_map(mset)
	ext_coeffs = ext_poly.coefficients
	for (j, α_ext) in enumerate(ext_poly.multiindex_set.exponents)
		α_full = SVector{NVAR, Int}(ntuple(i -> i <= ROM ? 0 : α_ext[i-ROM], Val(NVAR)))
		idx_full = get(mdict, α_full, nothing)
		idx_full === nothing && continue
		for e in 1:N_EXT
			coeff = T(ext_coeffs[e, j])
			iszero(coeff) || (R.poly.coefficients[ROM+e, idx_full] = coeff)
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
	::Val{ROM},
) where {ROM}
	return SVector{ROM, Bool}(ntuple(r -> is_resonant(resonance_set, monomial_idx, r), Val(ROM)))
end

# ==============================================================================
# Solve a single monomial
# ==============================================================================

# Singleton used by the no-sym public overload; skip_bits is never indexed inside
# solve_single_monomial!, so empty fields are correct here.
const _NO_SYM = ConjugateSymmetryData{NoConjugatePermutation}(
	NoConjugatePermutation(), BitVector(), NTuple{2, Int}[], Int[],
)

"""
	solve_single_monomial!(W, R, idx, ctx, model, ml_cache) -> nothing

Solve the cohomological equations for the monomial with multiindex-set position
`idx`, updating the coefficients of `W` and `R` in-place.

See the module docstring for a description of the algorithm.
"""
function solve_single_monomial!(
	W, R, idx::Int, ctx, model, ml_cache,
)
	solve_single_monomial!(W, R, idx, ctx, _NO_SYM, model, ml_cache)
end

# Canonical implementation: dispatches the solve step at compile time via sym.
function solve_single_monomial!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	idx::Int,
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
	::ConjugateSymmetryData,
	model::NDOrderModel,
	ml_cache::MultilinearTermsCache,
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
		ctx.lower_order.unit_vectors,
	)

	compute_multilinear_terms!(ctx.buffers.ml_result, model, idx, W, ml_cache)

	external_dynamics = view(R.poly.coefficients, (ROM+1):NVAR, idx)
	nR = count(resonance)
	n_sys = FOM + nR

	_solve_monomial!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)

	sol = view(ctx.buffers.rhs, 1:n_sys)
	W.poly.coefficients[:, 1, idx] .= view(sol, 1:FOM)

	rr = 1
	for r in 1:ROM
		if resonance[r]
			R.poly.coefficients[r, idx] = sol[FOM+rr]
			rr += 1
		else
			R.poly.coefficients[r, idx] = zero(T)
		end
	end

	compute_higher_derivative_coefficients!(
		W.poly.coefficients,
		view(R.poly.coefficients, 1:ROM, :),
		external_dynamics, s, idx,
		ctx.generalised_eigenmodes, lower_order_couplings,
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
function solve_cohomological_equations!(W, R, ctx, model, ml_cache; show_progress::Bool = true)
	sym = _build_conjugate_symmetry(NoConjugatePermutation(), ctx.linear_monomial_skip_set,
		multiindex_set(W))
	solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache; show_progress)
end

# Overload without active symmetry.  Uses sym-aware solve_single_monomial! to enable
# compile-time dispatch on NoConjugatePermutation.
function solve_cohomological_equations!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
	sym::ConjugateSymmetryData{NoConjugatePermutation},
	model::NDOrderModel,
	ml_cache::MultilinearTermsCache;
	show_progress::Bool = true,
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
	jobs = sym.solve_jobs
	prog = _make_progress(length(jobs), show_progress)
	n_done = 0
	for (idx, _) in jobs
		solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
		n_done += 1
		_progress_tick!(prog, n_done, sum(multiindex_set(W)[idx]))
	end
	_progress_done!(prog, n_done)
	return nothing
end

# Overload with active symmetry: secondaries are in skip_bits; solve_jobs contains only
# primaries.  Each primary is solved then its conjugate secondary is filled.
function solve_cohomological_equations!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
	sym::ConjugateSymmetryData{<:SVector},
	model::NDOrderModel,
	ml_cache::MultilinearTermsCache;
	show_progress::Bool = true,
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
	jobs = sym.solve_jobs
	prog = _make_progress(length(jobs), show_progress)
	n_done = 0
	for (idx, dst) in jobs
		solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
		n_done += 1
		_progress_tick!(prog, n_done, sum(multiindex_set(W)[idx]))
		iszero(dst) || fill_conjugate_monomial!(W, R, dst, idx, sym)
	end
	_progress_done!(prog, n_done)
	return nothing
end

# ==============================================================================
# Threaded solve infrastructure (dense path only)
# ==============================================================================

"""
    _solve_single_threaded!(W, R, idx, ctx, local_res, sym, model, ml_cache)

Solve the cohomological system for monomial `idx` using thread-local `local_res`
for all mutable scratch (system buffers + lower-order coupling buffer).

`local_res.buffers.ml_result` must already contain the multilinear-terms
contribution for this monomial (computed before entering the parallel region).

This function is identical in logic to `solve_single_monomial!` but reads from
`local_res.buffers` and `local_res.lo_buffer` instead of `ctx.buffers` and
`ctx.lower_order.buffer`, eliminating all cross-thread data races.
"""
function _solve_single_threaded!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        idx::Int,
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        local_res::ThreadedMonoSolveResources{T},
) where {T, ORD, ORDP1, NVAR, FOM, ROM, LT, MT}
    lb = local_res.buffers

    multi     = multiindex_set(W)[idx]
    s         = sum(multi[i] * ctx.lambda_diag[i] for i in 1:NVAR)
    resonance = _resonance_vector(ctx.resonance_set, idx, Val(ROM))
    nR        = count(resonance)
    n_sys     = FOM + nR

    # ── Lower-order couplings (thread-local buffer) ───────────────────────────
    for v in local_res.lo_buffer
        fill!(v, zero(T))
    end
    lower_order_couplings = compute_lower_order_couplings(
        multi, W, R,
        ctx.lower_order.multiindex_dict,
        local_res.lo_buffer,
        ctx.lower_order.candidate_indices[idx],
        ctx.lower_order.unit_vectors,
    )

    external_dynamics = view(R.poly.coefficients, (ROM + 1):NVAR, idx)

    # ── System assembly (into thread-local buffers) ───────────────────────────
    assemble_cohomological_matrix_and_rhs!(
        view(lb.system_matrix, 1:FOM, 1:n_sys),
        view(lb.rhs, 1:FOM),
        s, ctx.linear_terms,
        ctx.invariance.C_coeffs, ctx.invariance.E_coeffs,
        resonance, lower_order_couplings, external_dynamics,
        lb.external_rhs
    )
    view(lb.rhs, 1:FOM) .+= lb.ml_result
    assemble_orthogonality_matrix_and_rhs!(
        view(lb.system_matrix, (FOM + 1):n_sys, 1:n_sys),
        view(lb.rhs, (FOM + 1):n_sys),
        s, ctx.orthogonality.J_coeffs,
        ctx.orthogonality.C_coeffs, ctx.orthogonality.E_coeffs,
        resonance, lower_order_couplings, external_dynamics
    )

    # ── Dense solve (Schur complement for resonant) ───────────────────────────
    _dense_solve_bordered!(lb, FOM, nR)

    # ── Extract solution and write to W, R ────────────────────────────────────
    sol = view(lb.rhs, 1:n_sys)
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
        ctx.generalised_eigenmodes, lower_order_couplings,
    )
    return nothing
end

"""
    solve_cohomological_equations_threaded!(W, R, ctx, sym, model, ml_cache, thread_res)

Thread-parallel variant of `solve_cohomological_equations!` for the **dense path**.

Primary monomials at the same polynomial degree are mutually independent and are
solved in parallel with `Threads.@threads`.  After all primaries at a degree are
done, their conjugate secondaries are filled sequentially (degree barrier).

`compute_multilinear_terms!` is serialized via a `ReentrantLock`; its cost is
`O(FOM)` per monomial versus `O(FOM³)` for the LU, so lock contention is negligible.

`thread_res[tid]` must be pre-allocated via `alloc_thread_resources` before this call.
"""
function solve_cohomological_equations_threaded!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData,
        model::NDOrderModel,
        ml_cache::MultilinearTermsCache,
        thread_res::Vector{<:ThreadedMonoSolveResources{T}};
        show_progress::Bool = true,
) where {T, ORD, ORDP1, NVAR, FOM, ROM, LT, MT}
    jobs   = sym.solve_jobs
    bounds = sym.degree_boundaries
    n_deg  = length(bounds) - 1
    ml_lock = ReentrantLock()

    prog   = _make_progress(length(jobs), show_progress)
    n_done = Threads.Atomic{Int}(0)

    for d in 1:n_deg
        lo = bounds[d]
        hi = bounds[d + 1] - 1

        # All degree-d primary monomials are independent — solve in parallel.
        Threads.@threads for k in lo:hi
            tid       = Threads.threadid()
            idx       = jobs[k][1]
            local_res = thread_res[tid]

            # Multilinear terms: serialized (O(FOM) vs O(FOM³) for LU — negligible).
            @lock ml_lock begin
                compute_multilinear_terms!(local_res.buffers.ml_result, model, idx, W, ml_cache)
            end

            _solve_single_threaded!(W, R, idx, ctx, local_res)

            nd = Threads.atomic_add!(n_done, 1) + 1
            _progress_tick!(prog, nd, sum(multiindex_set(W)[idx]))
        end

        # Degree barrier: fill conjugate secondaries sequentially.
        for k in lo:hi
            (idx, dst) = jobs[k]
            iszero(dst) || fill_conjugate_monomial!(W, R, dst, idx, sym)
        end
    end

    _progress_done!(prog, n_done[])
    return nothing
end

end # module CohomologicalEquations
