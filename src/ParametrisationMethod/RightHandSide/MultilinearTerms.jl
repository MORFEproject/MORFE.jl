module MultilinearTerms

using LinearAlgebra: axpy!
using StaticArrays: SVector

using ..Multiindices: indices_in_box_with_bounded_degree,
	factorisations_asymmetric, factorisations_fully_symmetric, factorisations_groupwise_symmetric,
	bounded_index_tuples
using ..ParametrisationMethod: Parametrisation
using ..FullOrderModel: NDOrderModel, MultilinearMap

export compute_multilinear_terms, build_multilinear_terms_cache, MultilinearTermsCache

# =======================================================================
# 1. Symmetry classification
# =======================================================================
#
# `MultilinearMap.multiindex` records how many factor slots belong to each
# derivative order. Three structural cases arise:
#
#   FullyAsymmetric    — all entries ≤ 1: every slot has a distinct order.
#                        `t.f!` can write directly into the accumulator (no scratch).
#   FullySymmetric     — exactly one positive entry > 1: all slots share one order,
#                        and each factorisation carries a symmetry multiplier.
#   GroupwiseSymmetric — multiple positive entries > 1: slots span several orders,
#                        and each factorisation carries a combined multiplier.
#
# Using abstract-type dispatch on these tags lets Julia specialise the hot
# accumulation code at compile time with no runtime branches.

abstract type SymmetryType end
struct FullyAsymmetric    <: SymmetryType end
struct FullySymmetric     <: SymmetryType end
struct GroupwiseSymmetric <: SymmetryType end

"""
	symmetry_type(t) → SymmetryType

Classify a `MultilinearMap` by the symmetry pattern of its `multiindex`.
`FullyAsymmetric` is tested first so that a single entry equal to 1
(no repeated slot) takes the cheaper no-scratch path.
"""
function symmetry_type(t::MultilinearMap)
	all(x -> x <= 1, t.multiindex) && return FullyAsymmetric()
	count(>(0), t.multiindex) == 1 && return FullySymmetric()
	return GroupwiseSymmetric()
end

# =======================================================================
# 2. Derivative order helper
# =======================================================================

"""
	_derivative_orders(t) → NTuple

Map each factor slot to its 1-based derivative index according to `t.multiindex`.

Example: `multiindex = (2, 1)` → `(1, 1, 2)`  (two slots for order 1, one for order 2).
"""
function _derivative_orders(t::MultilinearMap)
	deg = sum(t.multiindex)
	return ntuple(deg) do slot
		cumulative = 0
		for (k, cnt) in enumerate(t.multiindex)
			cumulative += cnt
			slot <= cumulative && return k
		end
	end
end

# =======================================================================
# 3. Per-symmetry factorisation accumulation
# =======================================================================
#
# Three functions, one per symmetry class.  All increment `accum` in place
# (`t.f!` must also use += semantics).
#
# FullyAsymmetric: no scratch needed — `t.f!` writes directly into `accum`.
# FullySymmetric / GroupwiseSymmetric: zero `scratch`, call `t.f!`, then
#   `axpy!(count, scratch, accum)` to apply the symmetry multiplier.

function _accum_asymmetric!(accum,
	t, W, set, rem, deg, orders, candidate_indices, args_ext)
	for idx_tuple in factorisations_asymmetric(set, rem, deg, candidate_indices)
		@inbounds args = ntuple(i -> @view(W[:, orders[i], idx_tuple[i]]), Val(deg))
		t.f!(accum, args..., args_ext...)
	end
end

function _accum_symmetric!(accum, scratch,
	t, W, set, rem, deg, deriv_idx, candidate_indices, args_ext)
	for (idx_tuple, count) in factorisations_fully_symmetric(set, rem, deg, candidate_indices)
		fill!(scratch, 0)
		@inbounds args = ntuple(i -> @view(W[:, deriv_idx, idx_tuple[i]]), Val(deg))
		t.f!(scratch, args..., args_ext...)
		axpy!(count, scratch, accum)
	end
end

function _accum_groupwise!(accum, scratch,
	t, W, set, rem, deg, orders, candidate_indices, args_ext)
	for (idx_tuple, count) in factorisations_groupwise_symmetric(set, rem, t.multiindex, candidate_indices)
		fill!(scratch, 0)
		@inbounds args = ntuple(i -> @view(W[:, orders[i], idx_tuple[i]]), Val(deg))
		t.f!(scratch, args..., args_ext...)
		axpy!(count, scratch, accum)
	end
end

# Dispatch wrapper: routes to the right accumulation function at compile time.
function _accum_for_split!(accum, scratch, t, ::FullyAsymmetric, W, set, rem, deg, candidate_indices, args_ext)
	_accum_asymmetric!(accum, t, W, set, rem, deg, _derivative_orders(t), candidate_indices, args_ext)
end
function _accum_for_split!(accum, scratch, t, ::FullySymmetric, W, set, rem, deg, candidate_indices, args_ext)
	deriv_idx = findfirst(>(0), t.multiindex)::Int
	_accum_symmetric!(accum, scratch, t, W, set, rem, deg, deriv_idx, candidate_indices, args_ext)
end
function _accum_for_split!(accum, scratch, t, ::GroupwiseSymmetric, W, set, rem, deg, candidate_indices, args_ext)
	_accum_groupwise!(accum, scratch, t, W, set, rem, deg, _derivative_orders(t), candidate_indices, args_ext)
end

# =======================================================================
# 4. Term-level accumulation
# =======================================================================

"""
	accumulate_multilinear_term!(result, scratch, temp, t, parametrisation,
	                              exp, candidate_indices, external_exp, unit_vectors)

Add the contribution of one nonlinear term `t` for exponent `exp` to `result`.

**me = 0 fast path** (no external slots): there is exactly one split with
`ext_count = 1` and empty `args_ext`.  We skip `bounded_index_tuples` and
accumulate directly into `result`, saving one `fill!` and one `axpy!` O(FOM)
pass per term compared with the general path.

**me > 0 general path**: iterate over external splits, accumulate each split
into `temp`, then `axpy!(ext_count, temp, result)`.
"""
function accumulate_multilinear_term!(result, scratch, temp,
	t::MultilinearMap{ORD}, parametrisation::Parametrisation{ORD, NVAR},
	exp::SVector{NVAR}, candidate_indices, external_exp, unit_vectors) where {ORD, NVAR}

	W   = parametrisation.poly.coefficients
	set = parametrisation.poly.multiindex_set
	me  = t.multiplicity_external
	deg = t.deg - me
	ROM = NVAR - parametrisation.external_system_size
	sym = symmetry_type(t)

	if me == 0
		_accum_for_split!(result, scratch, t, sym, W, set, exp, deg, candidate_indices, ())
	else
		for (ext_idx, ext_multiindex_external, ext_count) in bounded_index_tuples(me, external_exp)
			ext_multiindex = SVector(ntuple(i -> i <= ROM ? 0 : ext_multiindex_external[i - ROM], Val(NVAR)))
			rem      = exp - ext_multiindex
			args_ext = ntuple(i -> unit_vectors[ext_idx[i]], me)
			fill!(temp, 0)
			_accum_for_split!(temp, scratch, t, sym, W, set, rem, deg, candidate_indices, args_ext)
			axpy!(ext_count, temp, result)
		end
	end
end

# =======================================================================
# 5. Public API — non-cached
# =======================================================================

"""
	compute_multilinear_terms(model, exp, parametrisation) → Vector

Return the sum of all nonlinear-term contributions for exponent `exp`.

Scratch buffers and shared precomputations (`unit_vectors`, `candidate_indices`,
`external_exp`) are allocated once and reused across all terms.
"""
function compute_multilinear_terms(model::NDOrderModel{ORD}, exp::SVector{NVAR},
	parametrisation::Parametrisation{ORD, NVAR}) where {ORD, NVAR}

	set     = parametrisation.poly.multiindex_set
	FOM     = size(parametrisation)
	T       = eltype(parametrisation.poly)
	deg_max = sum(exp)

	result  = zeros(T, FOM)
	scratch = similar(result)  # per-factorisation scratch (symmetric branches)
	temp    = similar(result)  # per-external-split accumulator (me > 0 path)

	external_system_size = parametrisation.external_system_size
	ROM = NVAR - external_system_size
	unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size))
	                for j in 1:external_system_size]
	candidate_indices = indices_in_box_with_bounded_degree(set, exp, 1, deg_max)
	external_exp = SVector(ntuple(i -> exp[ROM + i], external_system_size))

	for t in model.nonlinear_terms
		t.deg > deg_max && continue
		accumulate_multilinear_term!(result, scratch, temp, t, parametrisation,
			exp, candidate_indices, external_exp, unit_vectors)
	end
	return result
end

# =======================================================================
# 6. Factorisation cache — structs and construction
# =======================================================================
#
# Building a cache moves all factorisation enumeration out of the solve loop.
# The O(FOM) work per call (fill!, t.f!, axpy!) is unchanged; only the
# bookkeeping calls (which allocate small Vectors and tuples) are pre-computed.

"""One precomputed (index-tuple, multiplier) pair from any factorisation iterator."""
struct FactorisationEntry
	factor_indices::Vector{Int}  # indices into W's monomial axis
	multiplier::Int              # symmetry count; always 1 for FullyAsymmetric
end

"""
	CachedSplit

Precomputed data for one `(monomial, term, external-split)` triple.

- `ext_count`        — multiplicity from `bounded_index_tuples`; always 1 for me = 0.
- `args_ext_indices` — indices into `unit_vectors` to reconstruct `args_ext`; empty for me = 0.
- `is_asymmetric`    — if true, no scratch buffer is needed during replay.
- `orders`           — derivative index per factor slot (length = deg_internal).
- `entries`          — precomputed list of `FactorisationEntry` values.
"""
struct CachedSplit
	ext_count::Int
	args_ext_indices::Vector{Int}
	is_asymmetric::Bool
	orders::Vector{Int}
	entries::Vector{FactorisationEntry}
end

"""
	MultilinearTermsCache

Container of precomputed factorisation data for every (monomial, term) pair.
`splits[l][t_idx]` is a `Vector{CachedSplit}`; an empty vector means the term
degree exceeds the monomial degree and contributes nothing.

Build once with `build_multilinear_terms_cache`; reuse across the entire solve.
"""
struct MultilinearTermsCache
	splits::Vector{Vector{Vector{CachedSplit}}}
end

# --- Cache-building helpers (called once at setup time) ----------------

# Return per-slot derivative orders as a plain Vector for storage in CachedSplit.
_orders_for_cache(::FullySymmetric, t, deg)  = fill(findfirst(>(0), t.multiindex)::Int, deg)
_orders_for_cache(::SymmetryType,   t, deg)  = collect(Int, _derivative_orders(t))

# Collect all factorisation entries for one (term, rem) pair.
function _collect_entries(::FullyAsymmetric, t, mset, rem, deg, candidate_indices)
	[FactorisationEntry(collect(Int, idx), 1)
	 for idx in factorisations_asymmetric(mset, rem, deg, candidate_indices)]
end
function _collect_entries(::FullySymmetric, t, mset, rem, deg, candidate_indices)
	[FactorisationEntry(collect(Int, idx), count)
	 for (idx, count) in factorisations_fully_symmetric(mset, rem, deg, candidate_indices)]
end
function _collect_entries(::GroupwiseSymmetric, t, mset, rem, deg, candidate_indices)
	[FactorisationEntry(copy(idx_vec), count)
	 for (idx_vec, count) in factorisations_groupwise_symmetric(mset, rem, t.multiindex, candidate_indices)]
end

"""
	build_multilinear_terms_cache(model, parametrisation) → MultilinearTermsCache

Precompute all factorisation data for every monomial and term.
Call once before the solve loop; valid as long as the multiindex set is unchanged.
"""
function build_multilinear_terms_cache(
	model::NDOrderModel{ORD}, parametrisation::Parametrisation{ORD, NVAR}) where {ORD, NVAR}

	mset    = parametrisation.poly.multiindex_set
	L       = length(mset)
	n_terms = length(model.nonlinear_terms)
	external_system_size = parametrisation.external_system_size
	ROM     = NVAR - external_system_size

	all_splits = Vector{Vector{Vector{CachedSplit}}}(undef, L)

	for l in 1:L
		exp     = mset.exponents[l]
		deg_max = sum(exp)
		candidate_indices = indices_in_box_with_bounded_degree(mset, exp, 1, deg_max)
		external_exp = SVector(ntuple(i -> exp[ROM + i], external_system_size))

		term_splits = Vector{Vector{CachedSplit}}(undef, n_terms)

		for (t_idx, t) in enumerate(model.nonlinear_terms)
			if t.deg > deg_max
				term_splits[t_idx] = CachedSplit[]
				continue
			end

			me      = t.multiplicity_external
			deg     = t.deg - me
			sym     = symmetry_type(t)
			is_asym = sym isa FullyAsymmetric
			orders  = _orders_for_cache(sym, t, deg)
			splits  = CachedSplit[]

			if me == 0
				entries = _collect_entries(sym, t, mset, exp, deg, candidate_indices)
				push!(splits, CachedSplit(1, Int[], is_asym, orders, entries))
			else
				for (ext_idx, ext_multiindex_external, ext_count) in bounded_index_tuples(me, external_exp)
					ext_multiindex = SVector(ntuple(i -> i <= ROM ? 0 : ext_multiindex_external[i - ROM], Val(NVAR)))
					rem     = exp - ext_multiindex
					entries = _collect_entries(sym, t, mset, rem, deg, candidate_indices)
					push!(splits, CachedSplit(ext_count, collect(Int, ext_idx), is_asym, orders, entries))
				end
			end

			term_splits[t_idx] = splits
		end

		all_splits[l] = term_splits
	end

	return MultilinearTermsCache(all_splits)
end

# =======================================================================
# 7. Public API — cached
# =======================================================================

"""
	_replay_split!(result, scratch, temp, t, W, split, deg, unit_vectors)

Replay one `CachedSplit`: apply all precomputed factorisation entries and
accumulate into `result`.

me = 0 (empty `args_ext_indices`): accumulate directly into `result`.
me > 0: accumulate into `temp`, then `axpy!(ext_count, temp, result)`.
"""
function _replay_split!(result, scratch, temp, t, W, split, deg, unit_vectors)
	if isempty(split.args_ext_indices)
		# me = 0 fast path: accumulate directly into result.
		args_ext = ()
		accum    = result
	else
		args_ext = ntuple(i -> unit_vectors[split.args_ext_indices[i]], length(split.args_ext_indices))
		fill!(temp, 0)
		accum = temp
	end

	if split.is_asymmetric
		for entry in split.entries
			@inbounds args = ntuple(k -> @view(W[:, split.orders[k], entry.factor_indices[k]]), Val(deg))
			t.f!(accum, args..., args_ext...)
		end
	else
		for entry in split.entries
			fill!(scratch, 0)
			@inbounds args = ntuple(k -> @view(W[:, split.orders[k], entry.factor_indices[k]]), Val(deg))
			t.f!(scratch, args..., args_ext...)
			axpy!(entry.multiplier, scratch, accum)
		end
	end

	isempty(split.args_ext_indices) || axpy!(split.ext_count, temp, result)
end

"""
	compute_multilinear_terms(model, exp_index, parametrisation, cache) → Vector

Cached variant: replays precomputed factorisation data instead of calling
factorisation routines.  The O(FOM) work per call is identical to the non-cached
version; bookkeeping allocations are eliminated.

`exp_index` is the 1-based index into `parametrisation`'s multiindex set.
"""
function compute_multilinear_terms(model::NDOrderModel{ORD}, exp_index::Int,
	parametrisation::Parametrisation{ORD, NVAR},
	cache::MultilinearTermsCache) where {ORD, NVAR}

	W       = parametrisation.poly.coefficients
	FOM     = size(parametrisation)
	T       = eltype(parametrisation.poly)
	deg_max = sum(parametrisation.poly.multiindex_set.exponents[exp_index])

	result  = zeros(T, FOM)
	scratch = similar(result)
	temp    = similar(result)

	external_system_size = parametrisation.external_system_size
	unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size))
	                for j in 1:external_system_size]

	for (t_idx, t) in enumerate(model.nonlinear_terms)
		t.deg > deg_max && continue
		deg = t.deg - t.multiplicity_external

		for split in cache.splits[exp_index][t_idx]
			_replay_split!(result, scratch, temp, t, W, split, deg, unit_vectors)
		end
	end
	return result
end

end # module
