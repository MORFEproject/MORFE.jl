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

# Backward-compatible internal name retained for the established KLU regression tests.
_refactorise!(ss::SparseLinearSolverState, A::SparseMatrixCSC) = _refactorise_klu!(ss, A)

function _backward_error(ss::SparseLinearSolverState, x, norm_r, norm_b, norm_A)
    RT = typeof(ss.max_relative_residual)
    denominator = norm_A * norm(x, Inf) + norm_b
    return norm_r / max(denominator, floatmin(RT))
end

function _sparse_inf_norm!(workspace, A::SparseMatrixCSC)
    fill!(workspace, zero(eltype(workspace)))
    @inbounds for column in axes(A, 2)
        for position in nzrange(A, column)
            workspace[A.rowval[position]] += abs(A.nzval[position])
        end
    end
    return maximum(real, workspace)
end

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

function _assemble_nonlinear_rhs!(::_NoMonomialInstrumentation,
        ctx, model, idx, W, ml_cache)
    compute_multilinear_terms!(ctx.buffers.ml_result, model, idx, W, ml_cache)
    return nothing
end

function _solve_prepared_system!(::_NoMonomialInstrumentation,
        ctx, s, resonance, lower_order_couplings, external_dynamics,
        reuse_factor::Val)
    _solve_monomial!(
        ctx, s, resonance, lower_order_couplings, external_dynamics, reuse_factor)
    return nothing
end

_monomial_metrics(::_NoMonomialInstrumentation, ::Nothing, ::Nothing) = nothing

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
		reuse_factor) -> metrics

Canonical per-monomial pipeline shared by production and benchmark execution. Only
nonlinear-right-hand-side assembly and the bordered solve are instrumentation hooks;
preparation and coefficient finalisation stay outside benchmark timings.
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
        reuse_factor::Val{REUSE} = Val(false)
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT, REUSE}
    multi = multiindex_set(W)[idx]
    s = sum(multi[i] * ctx.lambda_diag[i] for i in 1:NVAR)
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
