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
# =============================================================================

struct _BenchmarkInstrumentation end
const _BENCHMARK_INSTRUMENTATION = _BenchmarkInstrumentation()

function _assemble_nonlinear_rhs!(::_BenchmarkInstrumentation,
        ctx, model, idx, W, ml_cache)
    return @timed compute_multilinear_terms!(
        ctx.buffers.ml_result, model, idx, W, ml_cache)
end

function _solve_prepared_system!(::_BenchmarkInstrumentation,
        ctx, s, resonance, lower_order_couplings, external_dynamics,
        reuse_factor::Val)
    return @timed _solve_monomial!(
        ctx, s, resonance, lower_order_couplings, external_dynamics, reuse_factor)
end

function _monomial_metrics(::_BenchmarkInstrumentation, rhs_result, solve_result)
    return (
        rhs_time = rhs_result.time,
        rhs_bytes = rhs_result.bytes,
        solve_time = solve_result.time,
        solve_bytes = solve_result.bytes)
end

function _timed_solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
    return _run_single_monomial!(_BENCHMARK_INSTRUMENTATION,
        W, R, idx, ctx, sym, model, ml_cache, Val(false))
end

# =============================================================================
# Per-order accumulator (mutable for in-place reset)
# =============================================================================
# Totals for the monomials of one polynomial order, reset between orders by
# `_reset!` rather than reallocated. Fields:
#   n_solved     — monomials solved at this order
#   rhs_time     — seconds spent assembling right-hand sides
#   rhs_bytes    — bytes allocated while assembling them
#   solve_time   — seconds spent in the linear solve
#   solve_bytes  — bytes allocated by the linear solve
# Deliberately left without a docstring: it is a benchmarking internal, and a
# docstring would publish it to the generated API page.

mutable struct _OrderAccum
    n_solved::Int
    rhs_time::Float64
    rhs_bytes::Int
    solve_time::Float64
    solve_bytes::Int
end

_OrderAccum() = _OrderAccum(0, 0.0, 0, 0.0, 0)

function _reset!(a::_OrderAccum)
    a.n_solved = 0
    a.rhs_time = 0.0
    a.rhs_bytes = 0
    a.solve_time = 0.0
    a.solve_bytes = 0
    return nothing
end

# =============================================================================
# CSV helpers
# =============================================================================

function _write_benchmark_headers(mono_io::IO, order_io::IO)
    println(mono_io,
        "order,monomial_idx,exponents,rhs_time_s,rhs_alloc_bytes," *
        "solve_time_s,solve_alloc_bytes,monomial_total_time_s,cumul_time_s")
    println(order_io,
        "order,n_solved,rhs_time_s,rhs_alloc_bytes," *
        "solve_time_s,solve_alloc_bytes,order_total_time_s,cumul_time_s,mem_live_bytes")
end

function _flush_order_row!(order_io::IO, order::Int, a::_OrderAccum, cumul_time::Float64)
    order_total = a.rhs_time + a.solve_time
    mem = Base.gc_live_bytes()
    println(order_io,
        "$order,$(a.n_solved),$(a.rhs_time),$(a.rhs_bytes)," *
        "$(a.solve_time),$(a.solve_bytes),$order_total,$cumul_time,$mem")
end

function _write_benchmark_csvs(mono_io::IOBuffer, order_io::IOBuffer, benchmark_dir::AbstractString)
    mkpath(benchmark_dir)
    open(joinpath(benchmark_dir, "benchmark_per_monomial.csv"), "w") do f
        write(f, take!(mono_io))
    end
    open(joinpath(benchmark_dir, "benchmark_per_order.csv"), "w") do f
        write(f, take!(order_io))
    end
end

mutable struct _BenchmarkSolveObserver{M} <: _AbstractSolveObserver
    mset::M
    monomial_io::IOBuffer
    order_io::IOBuffer
    benchmark_dir::String
    cumulative_time::Float64
    current_order::Int
    order_accumulator::_OrderAccum
end

function _BenchmarkSolveObserver(mset, benchmark_dir)
    monomial_io = IOBuffer()
    order_io = IOBuffer()
    _write_benchmark_headers(monomial_io, order_io)
    return _BenchmarkSolveObserver(
        mset, monomial_io, order_io, String(benchmark_dir), 0.0, -1, _OrderAccum())
end

function _on_job_complete!(observer::_BenchmarkSolveObserver, job, degree, metrics)
    observer.current_order = degree
    observer.cumulative_time += metrics.rhs_time + metrics.solve_time
    accumulator = observer.order_accumulator
    accumulator.n_solved += 1
    accumulator.rhs_time += metrics.rhs_time
    accumulator.rhs_bytes += metrics.rhs_bytes
    accumulator.solve_time += metrics.solve_time
    accumulator.solve_bytes += metrics.solve_bytes

    exponents = join(observer.mset[job.index], "_")
    monomial_total = metrics.rhs_time + metrics.solve_time
    println(observer.monomial_io,
        "$degree,$(job.index),$exponents,$(metrics.rhs_time),$(metrics.rhs_bytes)," *
        "$(metrics.solve_time),$(metrics.solve_bytes),$monomial_total," *
        "$(observer.cumulative_time)")
    return nothing
end

_on_group_complete!(::_BenchmarkSolveObserver, args...) = nothing
function _on_degree_complete!(observer::_BenchmarkSolveObserver, degree)
    _flush_order_row!(observer.order_io, degree,
        observer.order_accumulator, observer.cumulative_time)
    _reset!(observer.order_accumulator)
    return nothing
end
function _finish_observer!(observer::_BenchmarkSolveObserver)
    _write_benchmark_csvs(observer.monomial_io, observer.order_io,
        observer.benchmark_dir)
    return nothing
end

# =============================================================================
# Benchmarked solve loops
# =============================================================================

"""
	solve_cohomological_equations_benchmarked!(W, R, ctx, sym, model, ml_cache;
		benchmark_dir, show_progress = true) -> nothing

Run the causal cohomological solve while timing nonlinear right-hand-side assembly and the
bordered linear solve separately. On successful completion it overwrites two files:

- `benchmark_per_monomial.csv` contains order, monomial index and exponents, time and
  allocations for each measured phase, monomial total, and cumulative measured time.
- `benchmark_per_order.csv` aggregates those quantities by polynomial degree and records
  live GC bytes when each order is flushed.

Rows are buffered in memory and written only after the solve completes. The measured time
does not include lower-order coupling assembly, coefficient unpacking, higher-derivative
updates, progress output, or CSV writing. This diagnostic loop supports conjugate filling
but deliberately does not use structural factor grouping or checkpoint callbacks.

`benchmark_dir` is required and is created when necessary. `show_progress` has the same TTY
behavior as in [`solve_cohomological_equations!`](@ref).
"""
function solve_cohomological_equations_benchmarked!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData,
        model::NthOrderModel,
        ml_cache::MultilinearTermsCache;
        benchmark_dir::AbstractString,
        show_progress::Bool = true
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    plan = _build_solve_plan(ctx, sym, multiindex_set(W), :off, Val(ROM))
    benchmark = _BenchmarkSolveObserver(multiindex_set(W), benchmark_dir)
    progress = _ProgressSolveObserver(
        _make_progress(length(plan.jobs), show_progress, model.max_nl_degree), 0)
    observer = _compose_observers(progress, benchmark)
    _execute_solve_plan!(plan, _BENCHMARK_INSTRUMENTATION, observer,
        W, R, ctx, sym, model, ml_cache)
    return nothing
end
