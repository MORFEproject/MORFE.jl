# =============================================================================
# Solve planning and execution
# =============================================================================

"""One primary monomial solve and its optional conjugate reconstruction target."""
struct _SolveJob
    index::Int
    conjugate_target::Int
end

abstract type _AbstractSolvePlan end

"""Causal jobs executed as independent singleton factorisation groups."""
struct _DirectSolvePlan <: _AbstractSolvePlan
    jobs::Vector{_SolveJob}
end

"""Degree-local groups whose jobs share an exactly identical bordered matrix."""
struct _GroupedSolvePlan <: _AbstractSolvePlan
    groups::Vector{Vector{_SolveJob}}
    n_jobs::Int
end

"""Exact identity key for reusable bordered factorisations."""
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

_job_target(::ConjugateSymmetryData{NoConjugatePermutation}, ::Int) = 0
function _job_target(sym::ConjugateSymmetryData{<:SVector}, index::Int)
    target = @inbounds sym.monomial_map[index]
    return target > index ? target : 0
end

"""
	_build_solve_jobs(sym) -> Vector{_SolveJob}

Build jobs from the *current* skip mask. Checkpoint restoration and external-direction
initialisation mutate that mask after symmetry discovery, so jobs must not be cached in
`ConjugateSymmetryData`.
"""
function _build_solve_jobs(sym::ConjugateSymmetryData)
    jobs = _SolveJob[]
    sizehint!(jobs, count(!, sym.skip_bits))
    for index in eachindex(sym.skip_bits)
        @inbounds sym.skip_bits[index] && continue
        push!(jobs, _SolveJob(index, _job_target(sym, index)))
    end
    return jobs
end

function _group_solve_jobs(ctx, mset, jobs::Vector{_SolveJob}, ::Val{ROM}) where {ROM}
    representatives = _eigenvalue_representatives(ctx.lambda_diag)
    NVAR = length(ctx.lambda_diag)
    Key = StructuralFactorKey{NVAR, ROM}
    ordered_groups = Vector{Vector{_SolveJob}}()

    first_job = firstindex(jobs)
    while first_job <= lastindex(jobs)
        degree = sum(mset[jobs[first_job].index])
        keys = Key[]
        groups = Dict{Key, Vector{_SolveJob}}()
        next_job = first_job
        while next_job <= lastindex(jobs) &&
              sum(mset[jobs[next_job].index]) == degree
            job = jobs[next_job]
            resonance = _resonance_vector(ctx.resonance_set, job.index, Val(ROM))
            key = _structural_factor_key(
                mset[job.index], resonance, representatives)
            if !haskey(groups, key)
                groups[key] = _SolveJob[]
                push!(keys, key)
            end
            push!(groups[key], job)
            next_job += 1
        end
        append!(ordered_groups, (groups[key] for key in keys))
        first_job = next_job
    end
    return ordered_groups
end

"""
	_build_solve_plan(ctx, sym, mset, grouping, Val(ROM)) -> _AbstractSolvePlan

Create a causal direct or grouped plan. Grouping is exact and never crosses a total
degree. `:auto` returns the direct plan unless structural reuse is possible and actually
reduces the number of numeric factorisations.
"""
function _build_solve_plan(ctx, sym, mset, grouping::Symbol, ::Val{ROM}) where {ROM}
    grouping in (:off, :on, :auto) || throw(ArgumentError(
        "grouping must be :auto, :off, or :on"))
    jobs = _build_solve_jobs(sym)
    grouping == :off && return _DirectSolvePlan(jobs)
    grouping == :auto && !_has_structural_factor_reuse(ctx.lambda_diag) &&
        return _DirectSolvePlan(jobs)

    groups = _group_solve_jobs(ctx, mset, jobs, Val(ROM))
    grouping == :auto && length(groups) == length(jobs) &&
        return _DirectSolvePlan(jobs)
    return _GroupedSolvePlan(groups, length(jobs))
end

# =============================================================================
# Typed lifecycle observers
# =============================================================================

abstract type _AbstractSolveObserver end
struct _NoSolveObserver <: _AbstractSolveObserver end
const _NO_SOLVE_OBSERVER = _NoSolveObserver()

struct _CompositeSolveObserver{A <: _AbstractSolveObserver, B <: _AbstractSolveObserver} <:
       _AbstractSolveObserver
    first::A
    second::B
end

mutable struct _ProgressSolveObserver <: _AbstractSolveObserver
    progress::_SimpleProgress
    n_done::Int
end

struct _CheckpointSolveObserver{S, W, R, M, SS} <: _AbstractSolveObserver
    session::S
    W::W
    R::R
    mset::M
    sparse_solver::SS
end

_compose_observers(a::_NoSolveObserver, b::_NoSolveObserver) = _NO_SOLVE_OBSERVER
_compose_observers(a::_NoSolveObserver, b::_AbstractSolveObserver) = b
_compose_observers(a::_AbstractSolveObserver, b::_NoSolveObserver) = a
function _compose_observers(a::_AbstractSolveObserver, b::_AbstractSolveObserver)
    _CompositeSolveObserver(a, b)
end

_on_job_complete!(::_NoSolveObserver, args...) = nothing
_on_group_complete!(::_NoSolveObserver, args...) = nothing
_on_degree_complete!(::_NoSolveObserver, args...) = nothing
_finish_observer!(::_NoSolveObserver) = nothing

function _on_job_complete!(observer::_CompositeSolveObserver, args...)
    _on_job_complete!(observer.first, args...)
    _on_job_complete!(observer.second, args...)
    return nothing
end
function _on_group_complete!(observer::_CompositeSolveObserver, args...)
    _on_group_complete!(observer.first, args...)
    _on_group_complete!(observer.second, args...)
    return nothing
end
function _on_degree_complete!(observer::_CompositeSolveObserver, args...)
    _on_degree_complete!(observer.first, args...)
    _on_degree_complete!(observer.second, args...)
    return nothing
end
function _finish_observer!(observer::_CompositeSolveObserver)
    _finish_observer!(observer.first)
    _finish_observer!(observer.second)
    return nothing
end

function _on_job_complete!(observer::_ProgressSolveObserver, job, degree, metrics)
    observer.n_done += 1
    _progress_tick!(observer.progress, observer.n_done, degree)
    return nothing
end
_on_group_complete!(::_ProgressSolveObserver, args...) = nothing
_on_degree_complete!(::_ProgressSolveObserver, args...) = nothing
function _finish_observer!(observer::_ProgressSolveObserver)
    _progress_done!(observer.progress, observer.n_done)
    return nothing
end

function _completed_indices(job::_SolveJob)
    job.conjugate_target == 0 ?
    [job.index] :
    sort!([job.index, job.conjugate_target])
end
function _completed_indices(group::Vector{_SolveJob})
    indices = Int[]
    sizehint!(indices, 2 * length(group))
    for job in group
        push!(indices, job.index)
        job.conjugate_target == 0 || push!(indices, job.conjugate_target)
    end
    return sort!(unique!(indices))
end

_on_job_complete!(::_CheckpointSolveObserver, args...) = nothing
function _on_group_complete!(observer::_CheckpointSolveObserver, degree, group)
    observer.session.options.granularity == :factor_group || return nothing
    _write_chunk!(observer.session, observer.W, observer.R, degree,
        _completed_indices(group), observer.sparse_solver; degree_complete = false)
    return nothing
end
function _on_degree_complete!(observer::_CheckpointSolveObserver, degree)
    if observer.session.options.granularity == :degree
        indices = [index
                   for index in eachindex(observer.mset.exponents)
                   if sum(observer.mset[index]) == degree]
        _write_chunk!(observer.session, observer.W, observer.R, degree,
            indices, observer.sparse_solver; degree_complete = true)
    else
        _mark_degree_complete!(observer.session, degree)
    end
    return nothing
end
_finish_observer!(::_CheckpointSolveObserver) = nothing

# =============================================================================
# Shared executor
# =============================================================================

_fill_job_conjugate!(W, R, job, ::ConjugateSymmetryData{NoConjugatePermutation}) = nothing
function _fill_job_conjugate!(W, R, job, sym::ConjugateSymmetryData{<:SVector})
    job.conjugate_target == 0 ||
        fill_conjugate_monomial!(W, R, job.conjugate_target, job.index, sym)
    return nothing
end

function _execute_job!(instrumentation, observer, W, R, job, degree,
        ctx, sym, model, ml_cache, reuse_factor::Val)
    metrics = _run_single_monomial!(instrumentation,
        W, R, job.index, ctx, sym, model, ml_cache, reuse_factor)
    _fill_job_conjugate!(W, R, job, sym)
    _on_job_complete!(observer, job, degree, metrics)
    return nothing
end

function _execute_solve_plan!(plan::_DirectSolvePlan, instrumentation, observer,
        W, R, ctx, sym, model, ml_cache)
    mset = multiindex_set(W)
    current_degree = 0
    for job in plan.jobs
        degree = sum(mset[job.index])
        if current_degree != 0 && degree != current_degree
            _on_degree_complete!(observer, current_degree)
        end
        current_degree = degree
        _execute_job!(instrumentation, observer, W, R, job, degree,
            ctx, sym, model, ml_cache, Val(false))
        _on_group_complete!(observer, degree, job)
    end
    current_degree == 0 || _on_degree_complete!(observer, current_degree)
    _finish_observer!(observer)
    return nothing
end

function _execute_solve_plan!(plan::_GroupedSolvePlan, instrumentation, observer,
        W, R, ctx, sym, model, ml_cache)
    mset = multiindex_set(W)
    current_degree = 0
    for group in plan.groups
        degree = sum(mset[first(group).index])
        if current_degree != 0 && degree != current_degree
            _on_degree_complete!(observer, current_degree)
        end
        current_degree = degree
        for (position, job) in enumerate(group)
            if position == 1
                _execute_job!(instrumentation, observer, W, R, job, degree,
                    ctx, sym, model, ml_cache, Val(false))
            else
                _execute_job!(instrumentation, observer, W, R, job, degree,
                    ctx, sym, model, ml_cache, Val(true))
            end
        end
        _on_group_complete!(observer, degree, group)
    end
    current_degree == 0 || _on_degree_complete!(observer, current_degree)
    _finish_observer!(observer)
    return nothing
end

function _solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache;
        show_progress::Bool, grouping::Symbol,
        observer::_AbstractSolveObserver = _NO_SOLVE_OBSERVER,
        instrumentation = _NO_MONOMIAL_INSTRUMENTATION)
    return _solve_cohomological_equations_typed!(W, R, ctx, sym, model, ml_cache,
        observer, instrumentation, show_progress, grouping)
end

function _solve_cohomological_equations_typed!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx, sym, model, ml_cache,
        observer, instrumentation, show_progress, grouping
) where {ORD, NVAR, T, ROM}
    plan = _build_solve_plan(ctx, sym, multiindex_set(W), grouping, Val(ROM))
    n_jobs = plan isa _DirectSolvePlan ? length(plan.jobs) : plan.n_jobs
    progress = _ProgressSolveObserver(
        _make_progress(n_jobs, show_progress, model.max_nl_degree), 0)
    observers = _compose_observers(progress, observer)
    _execute_solve_plan!(plan, instrumentation, observers,
        W, R, ctx, sym, model, ml_cache)
    return nothing
end
