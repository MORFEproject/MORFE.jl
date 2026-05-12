# =============================================================================
# Sparse-linear-system dispatch
# =============================================================================

# Solve A*X = B.  Priority: Pardiso → KLU (with lazy symbolic caching) → UMFPACK.
# KLU caches the symbolic factorization on the first call via klu_cache::Ref{Any};
# subsequent calls reuse it via klu!(), skipping the symbolic step entirely.
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
# Dense-path monomial solve
# =============================================================================

# Assembles the full (FOM+nR)×(FOM+nR) bordered system into ctx.buffers.system_matrix,
# factors in-place with lu!, and writes the solution into ctx.buffers.rhs[1:FOM+nR].
function _solve_monomial!(
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
        s,
        ctx.linear_terms,
        ctx.invariance.C_coeffs,
        ctx.invariance.E_coeffs,
        resonance,
        lower_order_couplings,
        external_dynamics,
        ctx.buffers.external_rhs
    )
    view(ctx.buffers.rhs, 1:FOM) .+= ctx.buffers.ml_result

    assemble_orthogonality_matrix_and_rhs!(
        view(ctx.buffers.system_matrix, (FOM + 1):n_sys, 1:n_sys),
        view(ctx.buffers.rhs, (FOM + 1):n_sys),
        s,
        ctx.orthogonality.J_coeffs,
        ctx.orthogonality.C_coeffs,
        ctx.orthogonality.E_coeffs,
        resonance,
        lower_order_couplings,
        external_dynamics
    )

    F = lu!(view(ctx.buffers.system_matrix, 1:n_sys, 1:n_sys), check = false)
    ldiv!(F, view(ctx.buffers.rhs, 1:n_sys))
    return
end

# =============================================================================
# Sparse-path monomial solve (dispatch on MT <: SparseMatrixCSC)
# =============================================================================

# Builds L(s) as SparseMatrixCSC, then solves the bordered system via
# Pardiso/KLU + dense Schur complement for the resonant reduced-dynamics block.
function _solve_monomial!(
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        s, nR,
        resonance,
        lower_order_couplings,
        external_dynamics
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT <: SparseMatrixCSC}
    ss = ctx.sparse_solver   # SparseLinearSolverState{T}

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
        view(ctx.buffers.rhs, 1:FOM) .= _sparse_solve(ss.pardiso, ss.klu_cache, L_s, Vector(rhs))
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
