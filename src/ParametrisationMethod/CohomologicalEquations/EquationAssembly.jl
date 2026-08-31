# Assembly and dense solution of the bordered cohomological equation.

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
	_solve_monomial!(ctx, index, multiindex, s, resonance,
		lower_order_couplings, external_dynamics)

**Dense path.** Assemble the `(FOM+ROM)` bordered system via
`_assemble_bordered_system!`, then solve it in-place with `lu!` + `ldiv!`.
The solution is written into `ctx.buffers.rhs[1:FOM+ROM]`.
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
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT, ROM, REUSE}
    _assemble_bordered_system!(ctx, s, resonance, lower_order_couplings, external_dynamics)
    n_sys = FOM + ROM
    F = try
        lu!(view(ctx.buffers.system_matrix, 1:n_sys, 1:n_sys), check = false)
    catch error
        _is_unrecoverable_failure(error) && rethrow()
        _throw_bordered_failure(:factorisation, :dense,
            index, multiindex, s, resonance, ctx.resonance_set;
            detail = "dense LU factorisation raised an exception", cause = error)
    end
    issuccess(F) || _throw_bordered_failure(:factorisation, :dense,
        index, multiindex, s, resonance, ctx.resonance_set;
        detail = "dense LU reported a singular factorisation")
    try
        ldiv!(F, view(ctx.buffers.rhs, 1:n_sys))
    catch error
        _is_unrecoverable_failure(error) && rethrow()
        _throw_bordered_failure(:solve, :dense,
            index, multiindex, s, resonance, ctx.resonance_set;
            detail = "dense triangular solve raised an exception", cause = error)
    end
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
            F = try
                lu!(A, check = false)
            catch error
                _is_unrecoverable_failure(error) && rethrow()
                _throw_bordered_failure(:factorisation, :dense,
                    index, multiindex, s, resonance, ctx.resonance_set;
                    detail = "dense refinement factorisation raised an exception",
                    cause = error)
            end
            issuccess(F) || _throw_bordered_failure(:factorisation, :dense,
                index, multiindex, s, resonance, ctx.resonance_set;
                detail = "dense refinement factorisation reported singularity")
            try
                ldiv!(F, correction)
            catch error
                _is_unrecoverable_failure(error) && rethrow()
                _throw_bordered_failure(:solve, :dense,
                    index, multiindex, s, resonance, ctx.resonance_set;
                    detail = "dense refinement solve raised an exception", cause = error)
            end
            solution .+= correction
        end
        relative <= tolerance || _throw_bordered_failure(:accuracy, :dense,
            index, multiindex, s, resonance, ctx.resonance_set;
            detail = "backward error $relative exceeds tolerance $tolerance")
        copyto!(view(ctx.buffers.rhs, 1:n_sys), solution)
    end
    return
end
