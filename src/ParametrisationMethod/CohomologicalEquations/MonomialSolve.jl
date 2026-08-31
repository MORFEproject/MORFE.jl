# Canonical single-monomial cohomological pipeline and its public API.

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
	_solve_prepared_system!(instrumentation, ctx, index, multiindex, s,
		resonance, lower_order_couplings, external_dynamics, reuse_factor)

Instrumentation hook for the bordered solve after all monomial-dependent inputs have been
prepared. `reuse_factor` may be true only for an exactly identical grouped matrix.
"""
function _solve_prepared_system!(::_NoMonomialInstrumentation,
        ctx, index, multiindex, s, resonance, lower_order_couplings, external_dynamics,
        reuse_factor::Val)
    _solve_monomial!(
        ctx, index, multiindex, s, resonance,
        lower_order_couplings, external_dynamics, reuse_factor)
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
	_run_single_monomial!(instrumentation, W, R, idx, ctx, model, ml_cache,
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
        ctx, idx, multi, s, resonance,
        lower_order_couplings, external_dynamics, reuse_factor)

    _finalise_monomial!(W, R, idx, ctx, s, resonance,
        external_dynamics, lower_order_couplings)
    return _monomial_metrics(instrumentation, rhs_metrics, solve_metrics)
end

"""
	solve_single_monomial!(W, R, idx, ctx, model, ml_cache) -> nothing

Solve the cohomological equations for one multiindex-set position, updating `W` and `R`.
"""
function solve_single_monomial!(W, R, idx::Int, ctx, model, ml_cache)
    _run_single_monomial!(_NO_MONOMIAL_INSTRUMENTATION,
        W, R, idx, ctx, model, ml_cache, Val(false))
    return nothing
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
        W, R, idx, ctx, model, ml_cache, reuse_factor)
    return nothing
end

"""
	_solve_monomial!(ctx, index, multiindex, s, resonance,
		lower_order_couplings, external_dynamics)

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
        index::Int,
        multiindex,
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
        ss, view(ctx.buffers.rhs, 1:n_sys), s,
        index, multiindex, resonance, ctx.resonance_set; reuse_factor)
    return
end
