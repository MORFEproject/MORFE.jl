# =============================================================================
# Sparse-linear-system dispatch
# =============================================================================

"""
    _sparse_solve(pardiso, klu_cache, A, B) -> X

Solve `A * X = B` using the best available sparse factoriser.

Dispatch priority:
1. **Pardiso** — used when an `AbstractPardisoSolver` is available (MKL or open-source).
2. **KLU** — uses SuiteSparse KLU with lazy symbolic caching: symbolic factorisation
   is computed once on the first call and reused via `klu!(F, A)` on subsequent calls.
3. **UMFPACK** (fallback) — plain `lu(A) \\ B` when `klu_cache` is `nothing`.
"""
_sparse_solve(ps::AbstractPardisoSolver, ::Any, A, B) = pardiso_solve(ps, A, B)

function _sparse_solve(::Nothing, klu_cache::Ref{Any}, A::SparseMatrixCSC, B)
    if klu_cache[] === nothing
        F = klu(A)
        klu_cache[] = F
    else
        klu!(klu_cache[], A)
    end
    return klu_cache[] \ B
end

_sparse_solve(::Nothing, ::Nothing, A, B) = lu(A) \ B   # dense-path fallback

# =============================================================================
# Shared bordered-system assembly (dense paths)
# =============================================================================

"""
    _assemble_bordered_system!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)

Assemble the `(FOM+nR) × (FOM+nR)` bordered cohomological system into
`ctx.buffers.system_matrix` and `ctx.buffers.rhs`.

- The first `FOM` rows come from the invariance equation (operator + nonlinear RHS).
- The next `nR` rows come from the orthogonality conditions.

Called by both the dense-path `_solve_monomial!` overloads.
"""
function _assemble_bordered_system!(
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        s, nR,
        resonance,
        lower_order_couplings,
        external_dynamics
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT}
    n_sys = FOM + nR
    assemble_cohomological_matrix_and_rhs!(
        view(ctx.buffers.system_matrix, 1:FOM, 1:n_sys),
        view(ctx.buffers.rhs, 1:FOM),
        s, ctx.linear_terms,
        ctx.invariance.C_coeffs, ctx.invariance.E_coeffs,
        resonance, lower_order_couplings, external_dynamics,
        ctx.buffers.external_rhs
    )
    view(ctx.buffers.rhs, 1:FOM) .+= ctx.buffers.ml_result
    assemble_orthogonality_matrix_and_rhs!(
        view(ctx.buffers.system_matrix, (FOM + 1):n_sys, 1:n_sys),
        view(ctx.buffers.rhs, (FOM + 1):n_sys),
        s, ctx.orthogonality.J_coeffs,
        ctx.orthogonality.C_coeffs, ctx.orthogonality.E_coeffs,
        resonance, lower_order_couplings, external_dynamics
    )
    return
end

# =============================================================================
# Dense-path monomial solve
# =============================================================================

"""
    _solve_monomial!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)

**Dense path.** Assemble the `(FOM+nR)` bordered system via
`_assemble_bordered_system!`, then solve it in-place with `lu!` + `ldiv!`.
The solution is written into `ctx.buffers.rhs[1:FOM+nR]`.
"""
function _solve_monomial!(
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        s, nR,
        resonance,
        lower_order_couplings,
        external_dynamics
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT}
    _assemble_bordered_system!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)
    n_sys = FOM + nR
    F = lu!(view(ctx.buffers.system_matrix, 1:n_sys, 1:n_sys), check = false)
    ldiv!(F, view(ctx.buffers.rhs, 1:n_sys))
    return
end

"""
    _solve_monomial!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)

**Sparse path** (dispatched when `MT <: SparseMatrixCSC`). Builds `L(s)` as a
`SparseMatrixCSC` from the union-pattern template, then:

- **Non-resonant** (`nR == 0`): solves `L(s) W_α = rhs` directly via `_sparse_solve`.
- **Resonant** (`nR > 0`): performs one sparse factorisation with `(1 + nR)` right-hand
  sides (the RHS vector plus the `C_r` columns), then eliminates the resonant
  reduced-dynamics block via a dense Schur complement.

Results are written into `ctx.buffers.rhs[1:FOM+nR]`.
"""
function _solve_monomial!(
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        s, nR,
        resonance,
        lower_order_couplings,
        external_dynamics
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT <: SparseMatrixCSC}
    ss = ctx.sparse_solver::SparseLinearSolverState{T}

    # ── 1. Build L(s) and accumulate lower-order RHS ──────────────────────────
    rhs = view(ctx.buffers.rhs, 1:FOM)
    fill!(rhs, zero(T))
    L_s = build_sparse_L_and_rhs!(
        rhs, ss.L_template, ss.L_mappings,
        ctx.linear_terms, s, lower_order_couplings
    )

    # ── 2. External forcing and nonlinear contributions ───────────────────────
    evaluate_external_rhs!(rhs, s, external_dynamics, ctx.invariance.E_coeffs,
        ctx.buffers.external_rhs)
    rhs .+= ctx.buffers.ml_result

    if nR == 0
        # ── 3a. Non-resonant: L * W_α = rhs ──────────────────────────────────
        ss.rhs_extended[:, 1] .= rhs
        view(ctx.buffers.rhs, 1:FOM) .= _sparse_solve(
            ss.pardiso, ss.klu_cache, L_s, view(ss.rhs_extended, :, 1))
        return
    end

    # ── 3b. Resonant: bordered system via sparse solve + Schur complement ──────
    # Assemble C_r columns (FOM × nR) into system_matrix rows 1:FOM.
    C_r = view(ctx.buffers.system_matrix, 1:FOM, 1:nR)
    col = 1
    for r in eachindex(resonance)
        resonance[r] || continue
        evaluate_column!(view(C_r, :, col), s, r, ctx.invariance.C_coeffs)
        col += 1
    end

    # One factorisation, (1+nR) solves using pre-allocated rhs_extended buffer
    # (avoids the hcat(Vector(rhs), C_r) allocation on every resonant monomial).
    ss.rhs_extended[:, 1] .= rhs
    ss.rhs_extended[:, 2:(nR + 1)] .= C_r
    X = _sparse_solve(ss.pardiso, ss.klu_cache, L_s, view(ss.rhs_extended, :, 1:(nR + 1)))
    W_prime = view(X, :, 1)
    C_prime = view(X, :, 2:(nR + 1))

    # Orthogonality rows: nR × (FOM+nR) — reuse system_matrix rows FOM+1:FOM+nR.
    M_orth = view(ctx.buffers.system_matrix, (FOM + 1):(FOM + nR), 1:(FOM + nR))
    g = view(ctx.buffers.rhs, (FOM + 1):(FOM + nR))
    assemble_orthogonality_matrix_and_rhs!(
        M_orth, g, s,
        ctx.orthogonality.J_coeffs,
        ctx.orthogonality.C_coeffs,
        ctx.orthogonality.E_coeffs,
        resonance, lower_order_couplings, external_dynamics
    )
    J_r = view(M_orth, :, 1:FOM)
    Cbar_r = view(M_orth, :, (FOM + 1):(FOM + nR))

    # Schur complement: S = J_r * C_prime - Ĉ_r
    S = J_r * C_prime - Matrix{T}(Cbar_r)
    r_α = S \ (J_r * W_prime .- g)

    view(ctx.buffers.rhs, 1:FOM) .= W_prime .- C_prime * r_α
    view(ctx.buffers.rhs, (FOM + 1):(FOM + nR)) .= r_α
    return
end
