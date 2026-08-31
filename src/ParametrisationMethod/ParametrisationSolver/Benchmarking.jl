# Benchmark instrumentation and CSV reporting for the direct solve schedule.

# =============================================================================

"""
	_BenchmarkInstrumentation

Instrumentation marker that times nonlinear right-hand-side assembly and the bordered
solve while leaving preparation and coefficient finalisation outside the measurements.
"""
struct _BenchmarkInstrumentation end
const _BENCHMARK_INSTRUMENTATION = _BenchmarkInstrumentation()

function _assemble_nonlinear_rhs!(::_BenchmarkInstrumentation,
        ctx, model, idx, W, ml_cache)
    return @timed compute_multilinear_terms!(
        ctx.buffers.ml_result, model, idx, W, ml_cache)
end

function _solve_prepared_system!(::_BenchmarkInstrumentation,
        ctx, index, multiindex, s, resonance, lower_order_couplings, external_dynamics,
        reuse_factor::Val)
    return @timed _solve_monomial!(
        ctx, index, multiindex, s, resonance,
        lower_order_couplings, external_dynamics, reuse_factor)
end

function _monomial_metrics(::_BenchmarkInstrumentation, rhs_result, solve_result)
    return (
        rhs_time = rhs_result.time,
        rhs_bytes = rhs_result.bytes,
        solve_time = solve_result.time,
        solve_bytes = solve_result.bytes)
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
"""
	_OrderAccum

Mutable per-degree benchmark totals, reset and reused between polynomial orders.

# Fields

- `n_solved::Int` — primary monomials measured in the current degree.
- `rhs_time::Float64` — cumulative nonlinear right-hand-side assembly time.
- `rhs_bytes::Int` — cumulative allocations during right-hand-side assembly.
- `solve_time::Float64` — cumulative bordered-solve time.
- `solve_bytes::Int` — cumulative allocations during bordered solves.
"""
mutable struct _OrderAccum
    n_solved::Int
    rhs_time::Float64
    rhs_bytes::Int
    solve_time::Float64
    solve_bytes::Int
end

_OrderAccum() = _OrderAccum(0, 0.0, 0, 0.0, 0)

"""Reset every per-degree benchmark accumulator field in place."""
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

"""Write the fixed per-monomial and per-degree benchmark CSV headers."""
function _write_benchmark_headers(mono_io::IO, order_io::IO)
    println(mono_io,
        "order,monomial_idx,exponents,rhs_time_s,rhs_alloc_bytes," *
        "solve_time_s,solve_alloc_bytes,monomial_total_time_s,cumul_time_s")
    println(order_io,
        "order,n_solved,rhs_time_s,rhs_alloc_bytes," *
        "solve_time_s,solve_alloc_bytes,order_total_time_s,cumul_time_s,mem_live_bytes")
end

"""Append one completed degree's aggregate benchmark row to `order_io`."""
function _flush_order_row!(order_io::IO, order::Int, a::_OrderAccum, cumul_time::Float64)
    order_total = a.rhs_time + a.solve_time
    mem = Base.gc_live_bytes()
    println(order_io,
        "$order,$(a.n_solved),$(a.rhs_time),$(a.rhs_bytes)," *
        "$(a.solve_time),$(a.solve_bytes),$order_total,$cumul_time,$mem")
end

"""
	_write_benchmark_csvs(monomial_io, order_io, benchmark_dir) -> nothing

Create `benchmark_dir` and write both completed in-memory CSV buffers to their stable
filenames.
"""
function _write_benchmark_csvs(mono_io::IOBuffer, order_io::IOBuffer, benchmark_dir::AbstractString)
    mkpath(benchmark_dir)
    open(joinpath(benchmark_dir, "benchmark_per_monomial.csv"), "w") do f
        write(f, take!(mono_io))
    end
    open(joinpath(benchmark_dir, "benchmark_per_order.csv"), "w") do f
        write(f, take!(order_io))
    end
end

"""
	_BenchmarkSolveObserver{M}

Accumulate per-monomial and per-degree benchmark rows in memory and write them after a
successful solve.

# Fields

- `mset::M` — multiindex set used to render monomial exponents.
- `monomial_io::IOBuffer` — buffered per-monomial CSV rows.
- `order_io::IOBuffer` — buffered per-degree CSV rows.
- `benchmark_dir::String` — destination directory for both CSV files.
- `cumulative_time::Float64` — cumulative measured RHS and solve time.
- `current_order::Int` — degree of the most recently measured job.
- `order_accumulator::_OrderAccum` — reusable totals for the current degree.
"""
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
behaviour as in [`solve_cohomological_equations!`](@ref).
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
