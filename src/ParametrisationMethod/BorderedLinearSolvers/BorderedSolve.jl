# Backend dispatch for factorising, reusing, solving, and verifying bordered systems.

"""Dispatch numeric refactorisation to the selected built-in sparse backend."""
function _refactorise_sparse!(
        ss::SparseLinearSolverState{T, <:KLUBackend}, matrix) where {T}
    return _refactorise_klu!(ss, matrix)
end
function _refactorise_sparse!(
        ss::SparseLinearSolverState{T, <:UMFPACKBackend}, matrix) where {T}
    return _refactorise_umfpack!(ss, matrix)
end

"""Return the selected backend's cached factor with its concrete type restored."""
function _cached_sparse_factor(
        ss::SparseLinearSolverState{T, <:KLUBackend}, matrix) where {T}
    return _cached_klu_factor(ss, matrix)
end
function _cached_sparse_factor(
        ss::SparseLinearSolverState{T, <:UMFPACKBackend}, matrix) where {T}
    return _cached_umfpack_factor(ss, matrix)
end

"""Shared allocation-conscious solve for the built-in KLU and UMFPACK backends."""
function _suite_sparse_bordered_solve!(
        ss::SparseLinearSolverState{T}, solution::AbstractVector,
        superharmonic, index::Int, multiindex, resonance, resonance_set,
        reuse_factor::Val{REUSE}, ::Val{VERIFY}) where {T, REUSE, VERIFY}
    norm_b = zero(typeof(ss.max_relative_residual))
    if VERIFY
        copyto!(ss.residual_work, solution)
        norm_b = norm(solution, Inf)
    end
    had_cached_factorisation = ss.fact !== nothing
    cached_factorisation = ss.fact
    factorisation = if !REUSE || ss.fact === nothing
        fresh = try
            _refactorise_sparse!(ss, ss.bordered)
        catch error
            _is_unrecoverable_failure(error) && rethrow()
            ss.fact = nothing
            _throw_bordered_failure(:factorisation, _backend_name(ss.backend),
                index, multiindex, superharmonic, resonance, resonance_set;
                recovery_attempted = had_cached_factorisation,
                detail = had_cached_factorisation ?
                         "cached numeric factorisation failed and the fresh retry raised an exception" :
                         "fresh factorisation raised an exception",
                cause = error)
        end
        issuccess(fresh) || _throw_bordered_failure(
            :factorisation, _backend_name(ss.backend),
            index, multiindex, superharmonic, resonance, resonance_set;
            recovery_attempted = had_cached_factorisation,
            detail = had_cached_factorisation ?
                     "cached numeric factorisation and one fresh retry both failed" :
                     "fresh factorisation failed")
        fresh
    else
        _cached_sparse_factor(ss, ss.bordered)
    end
    recovery_attempted = had_cached_factorisation &&
                         factorisation !== cached_factorisation
    try
        ldiv!(factorisation, solution)
    catch error
        _is_unrecoverable_failure(error) && rethrow()
        _throw_bordered_failure(:solve, _backend_name(ss.backend),
            index, multiindex, superharmonic, resonance, resonance_set;
            recovery_attempted,
            detail = "backend triangular solve raised an exception", cause = error)
    end
    if VERIFY
        relative = try
            _sparse_backward_error!(ss, factorisation, solution, norm_b)
        catch error
            _is_unrecoverable_failure(error) && rethrow()
            _throw_bordered_failure(:solve, _backend_name(ss.backend),
                index, multiindex, superharmonic, resonance, resonance_set;
                recovery_attempted,
                detail = "iterative-refinement solve raised an exception", cause = error)
        end
        relative <= ss.residual_tolerance || _throw_bordered_failure(
            :accuracy, _backend_name(ss.backend),
            index, multiindex, superharmonic, resonance, resonance_set;
            recovery_attempted,
            detail = "backward error $relative exceeds tolerance $(ss.residual_tolerance)")
    end
    return solution
end

"""
	_bordered_solve!(state, solution, superharmonic, index, multiindex,
		resonance, resonance_set; reuse_factor = Val(false)) -> solution

Solve the bordered system in place through KLU, UMFPACK, or Pardiso. Exact grouped reuse
skips numeric factorisation only when the caller proves that the assembled matrix is
identical. Monomial data is carried solely to produce contextual failure diagnostics.
"""
function _bordered_solve!(
        ss::SparseLinearSolverState{T, <:KLUBackend{VERIFY}},
        solution::AbstractVector, superharmonic, index::Int, multiindex,
        resonance, resonance_set;
        reuse_factor::Val{REUSE} = Val(false)) where {T, REUSE, VERIFY}
    return _suite_sparse_bordered_solve!(ss, solution, superharmonic, index,
        multiindex, resonance, resonance_set, reuse_factor, Val(VERIFY))
end

function _bordered_solve!(
        ss::SparseLinearSolverState{T, <:UMFPACKBackend{VERIFY}},
        solution::AbstractVector, superharmonic, index::Int, multiindex,
        resonance, resonance_set;
        reuse_factor::Val{REUSE} = Val(false)) where {T, REUSE, VERIFY}
    return _suite_sparse_bordered_solve!(ss, solution, superharmonic, index,
        multiindex, resonance, resonance_set, reuse_factor, Val(VERIFY))
end

function _bordered_solve!(
        ss::SparseLinearSolverState{T, <:PardisoBackend{P, VERIFY}},
        solution::AbstractVector, superharmonic, index::Int, multiindex,
        resonance, resonance_set;
        reuse_factor::Val{REUSE} = Val(false)) where {T, P, REUSE, VERIFY}
    norm_b = zero(typeof(ss.max_relative_residual))
    try
        if ss.pardiso_matrix === nothing
            ss.pardiso_matrix = _pardiso_prepare!(ss.backend.solver, ss.bordered)
        end
        copyto!(ss.solve_scratch, solution)
        VERIFY && (norm_b = norm(ss.solve_scratch, Inf))
        if REUSE
            _pardiso_solve!(ss.backend.solver, ss.pardiso_matrix,
                solution, ss.solve_scratch)
        else
            _pardiso_factorise_solve!(ss.backend.solver, ss.pardiso_matrix,
                solution, ss.solve_scratch)
        end
    catch error
        _is_unrecoverable_failure(error) && rethrow()
        _throw_bordered_failure(REUSE ? :solve : :factorisation, :pardiso,
            index, multiindex, superharmonic, resonance, resonance_set;
            recovery_attempted = false,
            detail = REUSE ? "Pardiso solve phase failed" :
                     "Pardiso analysis or numeric factorisation phase failed",
            cause = error)
    end
    if VERIFY
        relative = try
            _sparse_residual!(ss.solve_scratch, ss.bordered, solution)
            RT = typeof(ss.max_relative_residual)
            norm_r = norm(ss.solve_scratch, Inf)
            value = norm_r / max(norm_b, floatmin(RT))
            if value > ss.residual_tolerance
                isempty(ss.refinement_work) &&
                    resize!(ss.refinement_work, length(solution))
                copyto!(ss.refinement_work, ss.solve_scratch)
                norm_A = _sparse_inf_norm!(ss.solve_scratch, ss.bordered)
                value = _backward_error(ss, solution, norm_r, norm_b, norm_A)
            end
            value
        catch error
            _is_unrecoverable_failure(error) && rethrow()
            _throw_bordered_failure(:solve, :pardiso,
                index, multiindex, superharmonic, resonance, resonance_set;
                recovery_attempted = false,
                detail = "Pardiso residual verification raised an exception",
                cause = error)
        end
        ss.max_relative_residual = max(ss.max_relative_residual, relative)
        relative <= ss.residual_tolerance || _throw_bordered_failure(
            :accuracy, :pardiso, index, multiindex, superharmonic,
            resonance, resonance_set;
            recovery_attempted = false,
            detail = "backward error $relative exceeds tolerance $(ss.residual_tolerance)")
    end
    return solution
end
