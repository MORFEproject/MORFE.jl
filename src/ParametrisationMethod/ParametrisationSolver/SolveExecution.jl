# Typed observers and the common direct/grouped execution engine.

# =============================================================================
# Typed lifecycle observers
# =============================================================================

"""
	_AbstractSolveObserver

Internal lifecycle interface notified after jobs, factor groups, completed degrees, and
the complete plan. Concrete observers keep progress, checkpointing, and benchmarking out
of the numerical executor.
"""
abstract type _AbstractSolveObserver end

"""No-op solve observer used when no optional lifecycle action is requested."""
struct _NoSolveObserver <: _AbstractSolveObserver end
const _NO_SOLVE_OBSERVER = _NoSolveObserver()

"""
	_CompositeSolveObserver{A, B}

Forward every lifecycle event to two observers in order.

# Fields

- `first::A` — observer notified first.
- `second::B` — observer notified after `first` completes.
"""
struct _CompositeSolveObserver{A <: _AbstractSolveObserver, B <: _AbstractSolveObserver} <:
       _AbstractSolveObserver
    first::A
    second::B
end

"""
	_ProgressSolveObserver

Track completed primary jobs and update the terminal progress indicator.

# Fields

- `progress::_SimpleProgress` — configured terminal progress state.
- `n_done::Int` — number of primary jobs reported so far.
"""
mutable struct _ProgressSolveObserver <: _AbstractSolveObserver
    progress::_SimpleProgress
    n_done::Int
end

"""
	_CheckpointSolveObserver{S, W, R, M, SS}

Commit coefficient chunks and degree markers at lifecycle boundaries.

# Fields

- `session::S` — active [`CheckpointSession`](@ref).
- `W::W` — parametrisation whose completed coefficient slices are persisted.
- `R::R` — reduced dynamics persisted alongside `W`.
- `mset::M` — multiindex set used to collect all indices in degree-granularity mode.
- `sparse_solver::SS` — sparse solver diagnostics, or `nothing` on the dense path.
"""
struct _CheckpointSolveObserver{S, W, R, M, SS} <: _AbstractSolveObserver
    session::S
    W::W
    R::R
    mset::M
    sparse_solver::SS
end

"""
	_compose_observers(a, b) -> _AbstractSolveObserver

Elide no-op observers and otherwise return a composite that preserves notification order.
"""
_compose_observers(a::_NoSolveObserver, b::_NoSolveObserver) = _NO_SOLVE_OBSERVER
_compose_observers(a::_NoSolveObserver, b::_AbstractSolveObserver) = b
_compose_observers(a::_AbstractSolveObserver, b::_NoSolveObserver) = a
function _compose_observers(a::_AbstractSolveObserver, b::_AbstractSolveObserver)
    _CompositeSolveObserver(a, b)
end

"""
	_on_job_complete!(observer, job, degree, metrics)
	_on_group_complete!(observer, degree, group)
	_on_degree_complete!(observer, degree)
	_finish_observer!(observer)

Lifecycle hooks implemented by solve observers. Every execution path emits job events in
causal order, one group event after its coefficients (including conjugate secondaries) are
final, degree events only after all groups in that degree, and one final event.
"""
_on_job_complete!(::_NoSolveObserver, args...) = nothing

"""Notify a solve observer after one factor group has been fully committed in memory."""
_on_group_complete!(::_NoSolveObserver, args...) = nothing

"""Notify a solve observer after every job in one polynomial degree is complete."""
_on_degree_complete!(::_NoSolveObserver, args...) = nothing

"""Finish a solve observer after the complete plan has executed successfully."""
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

"""
	_completed_indices(job_or_group) -> Vector{Int}

Return the sorted, unique coefficient positions made final by a job or factor group,
including reconstructed conjugate secondaries.
"""
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

"""
	_fill_job_conjugate!(W, R, job, sym) -> nothing

Reconstruct a job's conjugate secondary after its primary coefficients are final. This is
a compile-time no-op when conjugate symmetry is disabled.
"""
_fill_job_conjugate!(W, R, job, ::ConjugateSymmetryData{NoConjugatePermutation}) = nothing
function _fill_job_conjugate!(W, R, job, sym::ConjugateSymmetryData{<:SVector})
    job.conjugate_target == 0 ||
        fill_conjugate_monomial!(W, R, job.conjugate_target, job.index, sym)
    return nothing
end

"""
	_execute_job!(instrumentation, observer, W, R, job, degree, ctx, sym, model,
		ml_cache, reuse_factor, superharmonic = nothing) -> nothing

Run one canonical monomial pipeline, reconstruct its optional conjugate, and notify the
observer only after both coefficient positions are final.
"""
function _execute_job!(instrumentation, observer, W, R, job, degree,
        ctx, sym, model, ml_cache, reuse_factor::Val, superharmonic = nothing)
    metrics = _run_single_monomial!(instrumentation,
        W, R, job.index, ctx, model, ml_cache, reuse_factor, superharmonic)
    _fill_job_conjugate!(W, R, job, sym)
    _on_job_complete!(observer, job, degree, metrics)
    return nothing
end

"""
	_execute_solve_plan!(plan, instrumentation, observer, W, R, ctx, sym, model,
		ml_cache) -> nothing

Execute a direct or grouped plan in causal degree order. The grouped overload factorises
the first job in each group and passes `Val(true)` only to later jobs with a provably
identical bordered matrix.
"""
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
        degree = sum(mset[first(group.jobs).index])
        if current_degree != 0 && degree != current_degree
            _on_degree_complete!(observer, current_degree)
        end
        current_degree = degree
        for (position, job) in enumerate(group.jobs)
            if position == 1
                _execute_job!(instrumentation, observer, W, R, job, degree,
                    ctx, sym, model, ml_cache, Val(false), group.superharmonic)
            else
                _execute_job!(instrumentation, observer, W, R, job, degree,
                    ctx, sym, model, ml_cache, Val(true), group.superharmonic)
            end
        end
        _on_group_complete!(observer, degree, group.jobs)
    end
    current_degree == 0 || _on_degree_complete!(observer, current_degree)
    _finish_observer!(observer)
    return nothing
end

"""
	_solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache; ...) -> nothing

Internal execution boundary that specialises the optional observer and instrumentation
before entering the typed solve loop.
"""
function _solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache;
        show_progress::Bool, grouping::Symbol,
        observer::_AbstractSolveObserver = _NO_SOLVE_OBSERVER,
        instrumentation = _NO_MONOMIAL_INSTRUMENTATION)
    return _solve_cohomological_equations_typed!(W, R, ctx, sym, model, ml_cache,
        observer, instrumentation, show_progress, grouping)
end

"""
	_solve_cohomological_equations_typed!(W, R, ctx, sym, model, ml_cache,
		observer, instrumentation, show_progress, grouping) -> nothing

Build the solve plan, compose progress with the requested observer, and execute the plan.
This method exists as the type-stable target of the public scheduling overloads.
"""
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

"""
	solve_cohomological_equations!(W, R, ctx, model, ml_cache) -> nothing
	solve_cohomological_equations!(W, R, ctx, symmetry, model, ml_cache;
		show_progress = true, grouping = :off) -> nothing

Solve every scheduled cohomological equation in causal degree order. Conjugate secondary
coefficients are reconstructed immediately after their primary, and exact grouped solves
reuse the first job's factorisation without grouping across degrees.
"""
function solve_cohomological_equations!(
        W, R, ctx, model, ml_cache; show_progress::Bool = true,
        grouping::Symbol = :off)
    nterms = length(multiindex_set(W))
    symmetry = _build_conjugate_symmetry(
        NoConjugatePermutation(), ctx.linear_monomial_skip_set, nterms)
    return solve_cohomological_equations!(W, R, ctx, symmetry, model, ml_cache;
        show_progress, grouping)
end

function solve_cohomological_equations!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        symmetry::ConjugateSymmetryData,
        model::NthOrderModel,
        ml_cache::MultilinearTermsCache;
        show_progress::Bool = true,
        grouping::Symbol = :off
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    _solve_cohomological_equations!(W, R, ctx, symmetry, model, ml_cache;
        show_progress, grouping)
    return nothing
end

# =============================================================================
# Benchmarked cohomological solve variants
#
# Writes two CSV files to `benchmark_dir`:
#   benchmark_per_monomial.csv — one row per solved monomial
#   benchmark_per_order.csv   — one aggregate row per polynomial degree
#
# Both files are buffered in `IOBuffer`s and written at the end of the solve loop to
# avoid per-row file-system overhead. Existing files are overwritten; the writes are
# not transactional.
