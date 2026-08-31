# Allocation-conscious backward-error checks and iterative refinement.

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
	_sparse_inf_norm!(workspace, matrix) -> Real

Compute `norm(matrix, Inf)` for a CSC matrix using `workspace` for row sums and without
allocating a dense intermediate.
"""
function _sparse_inf_norm!(workspace, matrix::SparseMatrixCSC)
    fill!(workspace, zero(eltype(workspace)))
    @inbounds for column in axes(matrix, 2)
        for position in nzrange(matrix, column)
            workspace[matrix.rowval[position]] += abs(matrix.nzval[position])
        end
    end
    return maximum(real, workspace)
end

"""
	_sparse_residual!(residual, matrix, solution) -> residual

Overwrite a vector containing `b` with `matrix * solution - b` using SparseArrays'
five-argument `mul!` kernel.
"""
function _sparse_residual!(residual, matrix::SparseMatrixCSC, solution)
    return mul!(residual, matrix, solution,
        one(eltype(residual)), -one(eltype(residual)))
end

"""
	_sparse_backward_error!(state, factorisation, solution, norm_b) -> Real

Check the normwise backward error of a KLU or UMFPACK solution and, when necessary,
perform at most `state.max_refinement_steps` correction solves with the supplied typed
factorisation. Update the diagnostic maximum and refinement count and return the final
backward error.
"""
function _sparse_backward_error!(
        ss::SparseLinearSolverState, factorisation, solution, norm_b)
    tolerance = ss.residual_tolerance
    tolerance === nothing && return zero(typeof(ss.max_relative_residual))
    _sparse_residual!(ss.residual_work, ss.bordered, solution)
    RT = typeof(ss.max_relative_residual)
    norm_r = norm(ss.residual_work, Inf)
    relative = norm_r / max(norm_b, floatmin(RT))
    if relative > tolerance
        isempty(ss.refinement_work) && resize!(ss.refinement_work, length(solution))
        copyto!(ss.refinement_work, ss.residual_work)
        norm_A = _sparse_inf_norm!(ss.residual_work, ss.bordered)
        relative = _backward_error(ss, solution, norm_r, norm_b, norm_A)
        if relative > tolerance && ss.max_refinement_steps > 0
            mul!(ss.residual_work, ss.bordered, solution)
            ss.residual_work .-= ss.refinement_work
        end
        for _ in 1:ss.max_refinement_steps
            relative <= tolerance && break
            rmul!(ss.refinement_work, -one(eltype(solution)))
            ldiv!(factorisation, ss.refinement_work)
            solution .+= ss.refinement_work
            ss.refinement_count += 1
            mul!(ss.refinement_work, ss.bordered, solution)
            ss.refinement_work .-= ss.residual_work
            norm_r = norm(ss.refinement_work, Inf)
            relative = _backward_error(ss, solution, norm_r, norm_b, norm_A)
        end
    end
    ss.max_relative_residual = max(ss.max_relative_residual, relative)
    return relative
end
