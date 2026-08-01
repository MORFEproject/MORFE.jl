# =============================================================================
# Benchmarked cohomological solve variants
#
# Writes two CSV files to `benchmark_dir`:
#   benchmark_per_monomial.csv — one row per solved monomial
#   benchmark_per_order.csv   — one aggregate row per polynomial degree
#
# Both files are written atomically at the end of the solve loop (all rows
# buffered in IOBuffers to avoid per-row file-system overhead).
# =============================================================================

"""
	_timed_solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)

Identical to `solve_single_monomial!` but wraps `compute_multilinear_terms!`
(nonlinear RHS assembly) and `_solve_monomial!` (linear system solve) with
`@timed`, and returns a NamedTuple `(rhs_time, rhs_bytes, solve_time, solve_bytes)`.
"""
function _timed_solve_single_monomial!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	idx::Int,
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
	::ConjugateSymmetryData,
	model::NDOrderModel,
	ml_cache::MultilinearTermsCache,
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
	multi = multiindex_set(W)[idx]

	s         = sum(multi[i] * ctx.lambda_diag[i] for i in 1:NVAR)
	resonance = _resonance_vector(ctx.resonance_set, idx, Val(ROM))

	for v in ctx.lower_order.buffer
		fill!(v, zero(T))
	end
	lower_order_couplings = compute_lower_order_couplings(
		multi, W, R,
		ctx.lower_order.multiindex_dict,
		ctx.lower_order.buffer,
		ctx.lower_order.candidate_indices[idx],
		ctx.lower_order.unit_vectors,
	)

	rhs_r = @timed compute_multilinear_terms!(ctx.buffers.ml_result, model, idx, W, ml_cache)

	external_dynamics = view(R.poly.coefficients, (ROM+1):NVAR, idx)
	n_sys = FOM + ROM

	solve_r = @timed _solve_monomial!(ctx, s, resonance, lower_order_couplings, external_dynamics)

	sol = view(ctx.buffers.rhs, 1:n_sys)
	W.poly.coefficients[:, 1, idx] .= view(sol, 1:FOM)

	# Mirrors solve_single_monomial!: only resonant rows are read back.
	for r in 1:ROM
		R.poly.coefficients[r, idx] = resonance[r] ? sol[FOM+r] : zero(T)
	end

	compute_higher_derivative_coefficients!(
		W.poly.coefficients,
		view(R.poly.coefficients, 1:ROM, :),
		external_dynamics, s, idx,
		ctx.generalised_eigenmodes, lower_order_couplings,
	)

	return (
		rhs_time    = rhs_r.time,
		rhs_bytes   = rhs_r.bytes,
		solve_time  = solve_r.time,
		solve_bytes = solve_r.bytes,
	)
end

# =============================================================================
# Per-order accumulator (mutable for in-place reset)
# =============================================================================

mutable struct _OrderAccum
	n_solved::Int
	rhs_time::Float64
	rhs_bytes::Int
	solve_time::Float64
	solve_bytes::Int
end

_OrderAccum() = _OrderAccum(0, 0.0, 0, 0.0, 0)

function _reset!(a::_OrderAccum)
	a.n_solved    = 0
	a.rhs_time    = 0.0
	a.rhs_bytes   = 0
	a.solve_time  = 0.0
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

# =============================================================================
# Benchmarked solve loops
# =============================================================================

"""
	solve_cohomological_equations_benchmarked!(W, R, ctx, sym, model, ml_cache;
											   benchmark_dir, show_progress) -> nothing

Like `solve_cohomological_equations!` but records per-monomial and per-order
timing and writes two CSV files to `benchmark_dir` on completion.
"""
function solve_cohomological_equations_benchmarked!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
	sym::ConjugateSymmetryData{NoConjugatePermutation},
	model::NDOrderModel,
	ml_cache::MultilinearTermsCache;
	benchmark_dir::AbstractString,
	show_progress::Bool = true,
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
	nterms     = length(multiindex_set(W))
	n_to_solve = count(!b for b in sym.skip_bits)
	prog       = _make_progress(n_to_solve, show_progress, model.max_nl_degree)
	n_done     = 0

	mono_io  = IOBuffer()
	order_io = IOBuffer()
	_write_benchmark_headers(mono_io, order_io)

	cumul_time  = 0.0
	current_ord = -1
	oa          = _OrderAccum()

	for idx in 1:nterms
		@inbounds sym.skip_bits[idx] && continue

		multi = multiindex_set(W)[idx]
		ord   = sum(multi)

		if ord != current_ord && current_ord != -1
			_flush_order_row!(order_io, current_ord, oa, cumul_time)
			_reset!(oa)
		end
		current_ord = ord

		t = _timed_solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)

		cumul_time     += t.rhs_time + t.solve_time
		oa.n_solved    += 1
		oa.rhs_time    += t.rhs_time
		oa.rhs_bytes   += t.rhs_bytes
		oa.solve_time  += t.solve_time
		oa.solve_bytes += t.solve_bytes

		exps       = join(multi, "_")
		mono_total = t.rhs_time + t.solve_time
		println(mono_io,
			"$ord,$idx,$exps,$(t.rhs_time),$(t.rhs_bytes)," *
			"$(t.solve_time),$(t.solve_bytes),$mono_total,$cumul_time")

		n_done += 1
		_progress_tick!(prog, n_done, ord)
	end

	current_ord != -1 && _flush_order_row!(order_io, current_ord, oa, cumul_time)
	_progress_done!(prog, n_done)
	_write_benchmark_csvs(mono_io, order_io, benchmark_dir)
	return nothing
end

function solve_cohomological_equations_benchmarked!(
	W::Parametrisation{ORD, NVAR, T},
	R::ReducedDynamics{ROM, NVAR, T},
	ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
	sym::ConjugateSymmetryData{<:SVector},
	model::NDOrderModel,
	ml_cache::MultilinearTermsCache;
	benchmark_dir::AbstractString,
	show_progress::Bool = true,
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
	nterms     = length(multiindex_set(W))
	pairs      = sym.primary_pairs
	ptr        = 1
	n_to_solve = count(!b for b in sym.skip_bits)
	prog       = _make_progress(n_to_solve, show_progress, model.max_nl_degree)
	n_done     = 0

	mono_io  = IOBuffer()
	order_io = IOBuffer()
	_write_benchmark_headers(mono_io, order_io)

	cumul_time  = 0.0
	current_ord = -1
	oa          = _OrderAccum()

	for idx in 1:nterms
		@inbounds sym.skip_bits[idx] && continue

		multi = multiindex_set(W)[idx]
		ord   = sum(multi)

		if ord != current_ord && current_ord != -1
			_flush_order_row!(order_io, current_ord, oa, cumul_time)
			_reset!(oa)
		end
		current_ord = ord

		t = _timed_solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)

		cumul_time     += t.rhs_time + t.solve_time
		oa.n_solved    += 1
		oa.rhs_time    += t.rhs_time
		oa.rhs_bytes   += t.rhs_bytes
		oa.solve_time  += t.solve_time
		oa.solve_bytes += t.solve_bytes

		exps       = join(multi, "_")
		mono_total = t.rhs_time + t.solve_time
		println(mono_io,
			"$ord,$idx,$exps,$(t.rhs_time),$(t.rhs_bytes)," *
			"$(t.solve_time),$(t.solve_bytes),$mono_total,$cumul_time")

		n_done += 1
		_progress_tick!(prog, n_done, ord)

		while ptr <= length(pairs) && @inbounds sym.skip_bits[pairs[ptr][1]]
			ptr += 1
		end
		if ptr <= length(pairs) && @inbounds pairs[ptr][1] == idx
			(src, dst) = @inbounds pairs[ptr]
			fill_conjugate_monomial!(W, R, dst, src, sym)
			ptr += 1
		end
	end

	current_ord != -1 && _flush_order_row!(order_io, current_ord, oa, cumul_time)
	_progress_done!(prog, n_done)
	_write_benchmark_csvs(mono_io, order_io, benchmark_dir)
	return nothing
end
