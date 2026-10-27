# Benchmark instrumentation and CSV reporting for every solve schedule.

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
        reuse_factor::Val{REUSE}) where {REUSE}
    timed = @timed _solve_monomial!(
        ctx, index, multiindex, s, resonance,
        lower_order_couplings, external_dynamics, reuse_factor)
    return (time = timed.time, bytes = timed.bytes, factor_reused = REUSE)
end

function _monomial_metrics(::_BenchmarkInstrumentation, rhs_result, solve_result)
    return (
        rhs_time = rhs_result.time,
        rhs_bytes = rhs_result.bytes,
        solve_time = solve_result.time,
        solve_bytes = solve_result.bytes,
        factor_reused = solve_result.factor_reused)
end

# =============================================================================
# Per-order accumulator
# =============================================================================
"""
	_OrderAccum

Mutable benchmark totals for the monomials of one polynomial degree. A cross-degree plan
interleaves degrees, so the observer keeps one accumulator per degree until that degree
completes.

# Fields

- `n_solved::Int` — primary monomials measured in the degree.
- `n_factorised::Int` — those of them that computed a numeric factorisation.
- `rhs_time::Float64` — cumulative nonlinear right-hand-side assembly time.
- `rhs_bytes::Int` — cumulative allocations during right-hand-side assembly.
- `solve_time::Float64` — cumulative bordered-solve time.
- `solve_bytes::Int` — cumulative allocations during bordered solves.
"""
mutable struct _OrderAccum
    n_solved::Int
    n_factorised::Int
    rhs_time::Float64
    rhs_bytes::Int
    solve_time::Float64
    solve_bytes::Int
end

_OrderAccum() = _OrderAccum(0, 0, 0.0, 0, 0.0, 0)

# =============================================================================
# CSV helpers
# =============================================================================

"""Write the fixed per-monomial and per-degree benchmark CSV headers."""
function _write_benchmark_headers(mono_io::IO, order_io::IO)
    println(mono_io,
        "order,monomial_idx,exponents,rhs_time_s,rhs_alloc_bytes," *
        "solve_time_s,solve_alloc_bytes,factor_reused,monomial_total_time_s,cumul_time_s")
    println(order_io,
        "order,n_solved,n_factorised,rhs_time_s,rhs_alloc_bytes," *
        "solve_time_s,solve_alloc_bytes,order_total_time_s,cumul_time_s,mem_live_bytes")
end

"""Append one completed degree's aggregate benchmark row to `order_io`."""
function _flush_order_row!(order_io::IO, order::Int, a::_OrderAccum, cumul_time::Float64)
    order_total = a.rhs_time + a.solve_time
    mem = Base.gc_live_bytes()
    println(order_io,
        "$order,$(a.n_solved),$(a.n_factorised),$(a.rhs_time),$(a.rhs_bytes)," *
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
- `order_accumulators::Dict{Int, _OrderAccum}` — totals of every degree not yet complete.
"""
mutable struct _BenchmarkSolveObserver{M} <: _AbstractSolveObserver
    mset::M
    monomial_io::IOBuffer
    order_io::IOBuffer
    benchmark_dir::String
    cumulative_time::Float64
    order_accumulators::Dict{Int, _OrderAccum}
end

function _BenchmarkSolveObserver(mset, benchmark_dir)
    monomial_io = IOBuffer()
    order_io = IOBuffer()
    _write_benchmark_headers(monomial_io, order_io)
    return _BenchmarkSolveObserver(
        mset, monomial_io, order_io, String(benchmark_dir), 0.0,
        Dict{Int, _OrderAccum}())
end

function _on_job_complete!(observer::_BenchmarkSolveObserver, job, degree, metrics)
    observer.cumulative_time += metrics.rhs_time + metrics.solve_time
    accumulator = get!(_OrderAccum, observer.order_accumulators, degree)
    accumulator.n_solved += 1
    metrics.factor_reused || (accumulator.n_factorised += 1)
    accumulator.rhs_time += metrics.rhs_time
    accumulator.rhs_bytes += metrics.rhs_bytes
    accumulator.solve_time += metrics.solve_time
    accumulator.solve_bytes += metrics.solve_bytes

    exponents = join(observer.mset[job.index], "_")
    monomial_total = metrics.rhs_time + metrics.solve_time
    println(observer.monomial_io,
        "$degree,$(job.index),$exponents,$(metrics.rhs_time),$(metrics.rhs_bytes)," *
        "$(metrics.solve_time),$(metrics.solve_bytes),$(metrics.factor_reused)," *
        "$monomial_total,$(observer.cumulative_time)")
    return nothing
end

_on_group_complete!(::_BenchmarkSolveObserver, args...) = nothing
function _on_degree_complete!(observer::_BenchmarkSolveObserver, degree)
    _flush_order_row!(observer.order_io, degree,
        pop!(observer.order_accumulators, degree), observer.cumulative_time)
    return nothing
end
function _finish_observer!(observer::_BenchmarkSolveObserver)
    _write_benchmark_csvs(observer.monomial_io, observer.order_io,
        observer.benchmark_dir)
    return nothing
end

# =============================================================================
# Benchmarked schedule execution
# =============================================================================

"""
	_execute_benchmarked_schedule!(W, R, context, symmetry, model, cache;
		benchmark_dir, grouping = :off, show_progress = true) -> nothing

Run the causal cohomological solve under `grouping` while timing nonlinear right-hand-side
assembly and the bordered linear solve separately. On successful completion it overwrites
two files:

- `benchmark_per_monomial.csv` contains order, monomial index and exponents, time and
  allocations for each measured phase, whether the solve reused the previous job's numeric
  factorisation (`factor_reused`), monomial total, and cumulative measured time.
- `benchmark_per_order.csv` aggregates those quantities by polynomial degree, counts the
  numeric factorisations (`n_factorised`), and records live GC bytes when each order is
  flushed. Rows appear in the order degrees complete, which a cross-degree plan need not
  make increasing.

Rows are buffered in memory and written only after the solve completes. The measured time
does not include lower-order coupling assembly, coefficient unpacking, higher-derivative
updates, progress output, or CSV writing. A reused factorisation still pays for assembling
the bordered matrix, because that pass also accumulates the right-hand side; only the
numeric factorisation is skipped. Benchmarking does not combine with checkpoint callbacks.

`benchmark_dir` is required and is created when necessary. `grouping` takes the values of
[`ParametrisationOptions`](@ref); `show_progress` follows the ordinary schedule's TTY
behaviour.
"""
function _execute_benchmarked_schedule!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData,
        model::NthOrderModel,
        ml_cache::MultilinearTermsCache;
        benchmark_dir::AbstractString,
        grouping::Symbol = :off,
        show_progress::Bool = true
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    _execute_solve_schedule!(W, R, ctx, sym, model, ml_cache;
        show_progress, grouping,
        observer = _BenchmarkSolveObserver(multiindex_set(W), benchmark_dir),
        instrumentation = _BENCHMARK_INSTRUMENTATION)
    return nothing
end
