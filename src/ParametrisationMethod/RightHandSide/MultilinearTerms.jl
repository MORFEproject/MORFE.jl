module MultilinearTerms

using LinearAlgebra: axpy!
using StaticArrays: SVector

using ..Multiindices: indices_in_box_with_bounded_degree,
	factorisations_asymmetric, factorisations_fully_symmetric, factorisations_groupwise_symmetric,
	bounded_index_tuples, FactorisationEntry
using ..ParametrisationMethod: Parametrisation
using ..FullOrderModel: NDOrderModel, MultilinearMap

export compute_multilinear_terms, build_multilinear_terms_cache, MultilinearTermsCache

# -----------------------------------------------------------------------
# Symmetry classification
# -----------------------------------------------------------------------
#
# MultilinearMap.multiindex[k] is the number of factor slots that use the
# k-th derivative.  Three cases arise, each with a different accumulation strategy:
#
#   FullyAsymmetric    — all entries ≤ 1: distinct orders, t.f! goes directly
#                        into the accumulator (no scratch buffer needed).
#   FullySymmetric     — one positive entry > 1: all slots share one derivative,
#                        each factorisation carries a symmetry count.
#   GroupwiseSymmetric — multiple positive entries: slots span several derivatives,
#                        each factorisation carries a combined symmetry count.
#
# Dispatching on these tags lets Julia specialise the hot inner loop at
# compile time.

abstract type SymmetryType end
struct FullyAsymmetric    <: SymmetryType end
struct FullySymmetric     <: SymmetryType end
struct GroupwiseSymmetric <: SymmetryType end

function symmetry_type(t::MultilinearMap)
	all(x -> x <= 1, t.multiindex) && return FullyAsymmetric()
	count(>(0), t.multiindex) == 1  && return FullySymmetric()
	return GroupwiseSymmetric()
end

# -----------------------------------------------------------------------
# Derivative order helper
# -----------------------------------------------------------------------

"""
	_derivative_orders(t) → NTuple

Map each factor slot to its 1-based derivative index.
Example: `multiindex = (2, 1)` → `(1, 1, 2)`.
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

# -----------------------------------------------------------------------
# Per-split accumulation — one dispatch method per symmetry class
# -----------------------------------------------------------------------
#
# Each method accumulates the contributions for one (external-system) split
# of the exponent `rem` into `accum`.  `t.f!` must use += semantics.
#
# The caller is responsible for zeroing `accum` before calling when needed
# (see accumulate_multilinear_term! below).

# No scratch needed: t.f! writes directly into accum.
function _accumulate_split!(accum, _scratch,
		t, ::FullyAsymmetric, W, set, rem, deg, candidate_indices, args_ext)
	orders = _derivative_orders(t)
	for entry in factorisations_asymmetric(set, rem, deg, candidate_indices)
		@inbounds args = ntuple(i -> @view(W[:, orders[i], entry.factor_indices[i]]), Val(deg))
		t.f!(accum, args..., args_ext...)
	end
end

# All factor slots share one derivative; each factorisation has a symmetry count.
function _accumulate_split!(accum, scratch,
		t, ::FullySymmetric, W, set, rem, deg, candidate_indices, args_ext)
	deriv_idx = findfirst(>(0), t.multiindex)::Int
	for entry in factorisations_fully_symmetric(set, rem, deg, candidate_indices)
		fill!(scratch, 0)
		@inbounds args = ntuple(i -> @view(W[:, deriv_idx, entry.factor_indices[i]]), Val(deg))
		t.f!(scratch, args..., args_ext...)
		axpy!(entry.multiplier, scratch, accum)
	end
end

# Factor slots span several derivatives; each factorisation has a combined count.
function _accumulate_split!(accum, scratch,
		t, ::GroupwiseSymmetric, W, set, rem, deg, candidate_indices, args_ext)
	orders = _derivative_orders(t)
	for entry in factorisations_groupwise_symmetric(set, rem, t.multiindex, candidate_indices)
		fill!(scratch, 0)
		@inbounds args = ntuple(i -> @view(W[:, orders[i], entry.factor_indices[i]]), Val(deg))
		t.f!(scratch, args..., args_ext...)
		axpy!(entry.multiplier, scratch, accum)
	end
end

# -----------------------------------------------------------------------
# Term-level accumulation
# -----------------------------------------------------------------------

"""
	accumulate_multilinear_term!(result, scratch, temp, t, parametrisation,
	                              exp, candidate_indices, external_exp, unit_vectors)

Add the contribution of nonlinear term `t` for exponent `exp` to `result`.

When `t` has no external (forcing) slots (`me = 0`) there is exactly one split,
so we skip `bounded_index_tuples` and write directly into `result`, saving one
`fill!` and one `axpy!` per term.  For `me > 0` each split is accumulated into
`temp` first, then scaled into `result` via `axpy!`.
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
		_accumulate_split!(result, scratch, t, sym, W, set, exp, deg, candidate_indices, ())
	else
		for (ext_idx, ext_multiindex_external, ext_count) in bounded_index_tuples(me, external_exp)
			ext_multiindex = SVector(ntuple(i -> i <= ROM ? 0 : ext_multiindex_external[i - ROM], Val(NVAR)))
			rem      = exp - ext_multiindex
			args_ext = ntuple(i -> unit_vectors[ext_idx[i]], me)
			fill!(temp, 0)
			_accumulate_split!(temp, scratch, t, sym, W, set, rem, deg, candidate_indices, args_ext)
			axpy!(ext_count, temp, result)
		end
	end
end

# -----------------------------------------------------------------------
# Public API — non-cached
# -----------------------------------------------------------------------

"""
	compute_multilinear_terms(model, exp, parametrisation) → Vector

Return the sum of all nonlinear-term contributions for exponent `exp`.

Scratch buffers and shared data (`unit_vectors`, `candidate_indices`,
`external_exp`) are allocated once and reused across all terms.
"""
function compute_multilinear_terms(model::NDOrderModel{ORD}, exp::SVector{NVAR},
		parametrisation::Parametrisation{ORD, NVAR}) where {ORD, NVAR}

	set     = parametrisation.poly.multiindex_set
	FOM     = size(parametrisation)
	T       = eltype(parametrisation.poly)
	deg_max = sum(exp)

	result  = zeros(T, FOM)
	scratch = similar(result)   # per-factorisation scratch (symmetric branches)
	temp    = similar(result)   # per-external-split accumulator (me > 0 only)

	external_system_size = parametrisation.external_system_size
	ROM = NVAR - external_system_size
	unit_vectors     = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size)) for j in 1:external_system_size]
	candidate_indices = indices_in_box_with_bounded_degree(set, exp, 1, deg_max)
	external_exp     = SVector(ntuple(i -> exp[ROM + i], external_system_size))

	for t in model.nonlinear_terms
		t.deg > deg_max && continue
		accumulate_multilinear_term!(result, scratch, temp, t, parametrisation,
			exp, candidate_indices, external_exp, unit_vectors)
	end
	return result
end

# -----------------------------------------------------------------------
# Factorisation cache — structs
# -----------------------------------------------------------------------
#
# Use case: the parametrisation solve loop calls compute_multilinear_terms
# once per monomial (O(L) calls, where L = |mset|).  Each call invokes the
# factorisation routines, which enumerate index tuples and allocate Vectors.
# For large L this GC pressure is measurable.
#
# Since factorisation results depend only on the multiindex set and the model
# structure — not on the parametrisation coefficients W — they can be
# precomputed once and replayed on every call.
#
# What IS cached: for each (monomial, term, external-split) triple,
#   - which coefficient indices to load from W,
#   - their symmetry multipliers,
#   - which external unit vectors to pass as forcing arguments.
#
# What is NOT cached: the O(FOM) arithmetic (fill!, t.f!, axpy!).
# Those operations read from W, which changes at every solve step.

"""
	CachedSplit

Precomputed bookkeeping for one `(monomial l, term t, external-split)` triple.

- `ext_count`        — multiplicity of this external-variable split
                       (from `bounded_index_tuples`); always 1 when `me = 0`.
- `args_ext_indices` — indices into the `unit_vectors` array that reconstruct
                       the external forcing arguments; empty when `me = 0`.
- `is_asymmetric`    — true iff `t` is `FullyAsymmetric` (no scratch buffer
                       needed when replaying).
- `orders`           — derivative index for each factor slot (length = deg_internal).
- `entries`          — list of `FactorisationEntry` values, one per factorisation
                       of the remainder exponent.
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

All precomputed factorisation bookkeeping for a given `(model, parametrisation)`
pair.  `splits[l][t_idx]` is the list of `CachedSplit` values for monomial `l`
and term `t_idx`; an empty list means the term degree exceeds the monomial degree
and it contributes nothing.

**Build** once before the solve loop with `build_multilinear_terms_cache`.
**Use** by passing to the `(model, exp_index, parametrisation, cache)` overload of
`compute_multilinear_terms` inside the loop.  The cache is valid as long as the
multiindex set and the model structure are unchanged (i.e. across all solve steps).
"""
struct MultilinearTermsCache
	splits::Vector{Vector{Vector{CachedSplit}}}
end

# -----------------------------------------------------------------------
# Cache construction helpers
# -----------------------------------------------------------------------

# Per-slot orders as a Vector (for storage in CachedSplit).
_orders_for_cache(::FullySymmetric, t, deg) = fill(findfirst(>(0), t.multiindex)::Int, deg)
_orders_for_cache(::SymmetryType,   t, deg) = collect(Int, _derivative_orders(t))

# Route to the right factorisation function for cache construction.
# GroupwiseSymmetric needs t.multiindex; the others do not.
_collect_entries(::FullyAsymmetric,    t, mset, rem, deg, cands) = factorisations_asymmetric(mset, rem, deg, cands)
_collect_entries(::FullySymmetric,     t, mset, rem, deg, cands) = factorisations_fully_symmetric(mset, rem, deg, cands)
_collect_entries(::GroupwiseSymmetric, t, mset, rem, deg, cands) = factorisations_groupwise_symmetric(mset, rem, t.multiindex, cands)

"""
	build_multilinear_terms_cache(model, parametrisation) → MultilinearTermsCache

Precompute all factorisation data for every monomial and term.
Valid as long as the multiindex set is unchanged.
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
		external_exp      = SVector(ntuple(i -> exp[ROM + i], external_system_size))

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

# -----------------------------------------------------------------------
# Public API — cached
# -----------------------------------------------------------------------

# Replay one CachedSplit into result.
# me = 0 (empty args_ext_indices): accumulate directly into result.
# me > 0: accumulate into temp, then axpy!(ext_count, temp, result).
function _replay_split!(result, scratch, temp, t, W, split, deg, unit_vectors)
	if isempty(split.args_ext_indices)
		accum    = result
		args_ext = ()
	else
		fill!(temp, 0)
		accum    = temp
		args_ext = ntuple(i -> unit_vectors[split.args_ext_indices[i]], length(split.args_ext_indices))
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

Cached variant of `compute_multilinear_terms`: replays precomputed factorisation
data instead of calling the factorisation routines.

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
	unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size)) for j in 1:external_system_size]

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
