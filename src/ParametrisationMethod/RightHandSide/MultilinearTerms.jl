module MultilinearTerms

using LinearAlgebra: axpy!
using StaticArrays: SVector

using ..Multiindices: indices_in_box_with_bounded_degree,
	factorisations_asymmetric, factorisations_fully_symmetric, factorisations_groupwise_symmetric,
	bounded_index_tuples, FactorisationEntry
using ..ParametrisationMethod: Parametrisation
using ..FullOrderModel: NDOrderModel, MultilinearMap
using ..MultilinearMaps: AbstractMultilinearMap, FEMMultilinearMap,
	fem_elements, fem_n_qp, fem_ndofs_per_cell,
	scatter_qp!, accumulate_qp!, assemble_element!, fem_getdetJdV

export compute_multilinear_terms, compute_multilinear_terms!, build_multilinear_terms_cache, MultilinearTermsCache

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

function symmetry_type(t::AbstractMultilinearMap)
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
function _derivative_orders(t::AbstractMultilinearMap)
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
struct MultilinearTermsCache{T}
	splits::Vector{Vector{Vector{CachedSplit}}}
	# FEM-batched splits: fem_splits[l][t_idx] is Vector{FEMCachedSplit{DEG}} for FEM terms,
	# or an empty Vector for closure terms. Element type is Any because DEG varies per term.
	fem_splits::Vector{Vector{Vector{Any}}}
	result_buffer::Vector{T}
	scratch_buffer::Vector{T}
	temp_buffer::Vector{T}
	unit_vectors::Vector   # Vector of SVector{N_EXT, Int}; empty when N_EXT == 0
	# Pre-allocated element-local residual buffer, shared across all FEM terms.
	# The qp-level field buffer (∇W_qp) is type-specific and owned by each FEMMultilinearMap
	# via fem_qp_buffer(t) — see MultilinearMaps.jl interface.
	fem_Fe::Vector{T}     # size: max_ndofs_per_cell
end

# -----------------------------------------------------------------------
# FEM-specific cache structs
# -----------------------------------------------------------------------

"""
	FEMFactorisationEntry{DEG}

One factorisation entry for the FEM-batched path.

- `multiplier`           — symmetry count (same as FactorisationEntry.multiplier).
- `local_factor_indices` — NTuple of length DEG: for each factor slot, the index into
                           `FEMCachedSplit.unique_cols` for the enclosing split.
                           NTuple (not Vector) so that `ntuple(k->..., Val(DEG))` in the
                           hot loop is unrolled at compile time.
"""
struct FEMFactorisationEntry{DEG}
	multiplier::Int
	local_factor_indices::NTuple{DEG, Int}
end

"""
	FEMCachedSplit{DEG}

Precomputed bookkeeping for one (monomial, FEM-term, external-split) triple.

- `ext_count`        — external multiplicity (1 when me = 0).
- `args_ext_indices` — unit vector indices for external args (empty when me = 0).
- `unique_cols`      — deduplicated list of (derivative_order, W_col_idx) pairs across all
                       entries in this split.  Scattered to qp-level gradients once per element.
- `fem_entries`      — one FEMFactorisationEntry{DEG} per factorisation, with local indices
                       into unique_cols.
"""
struct FEMCachedSplit{DEG}
	ext_count::Int
	args_ext_indices::Vector{Int}
	unique_cols::Vector{Tuple{Int,Int}}
	fem_entries::Vector{FEMFactorisationEntry{DEG}}
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

# Build a FEMCachedSplit{DEG} from an already-constructed CachedSplit.
# Called once per split at cache-build time; zero allocation in the hot path.
function _build_fem_cached_split(::Val{DEG}, cs::CachedSplit) where {DEG}
	# 1. Enumerate unique (derivative_order, W_col_idx) pairs across all entries.
	unique_cols  = Tuple{Int,Int}[]
	col_to_local = Dict{Tuple{Int,Int}, Int}()
	for entry in cs.entries
		for k in 1:DEG
			oc = (cs.orders[k], entry.factor_indices[k])
			if !haskey(col_to_local, oc)
				push!(unique_cols, oc)
				col_to_local[oc] = length(unique_cols)
			end
		end
	end
	# 2. Map each entry's factor slots to local indices in unique_cols.
	fem_entries = [
		FEMFactorisationEntry{DEG}(
			entry.multiplier,
			ntuple(k -> col_to_local[(cs.orders[k], entry.factor_indices[k])], Val(DEG))
		)
		for entry in cs.entries
	]
	return FEMCachedSplit{DEG}(cs.ext_count, cs.args_ext_indices, unique_cols, fem_entries)
end

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

	all_splits     = Vector{Vector{Vector{CachedSplit}}}(undef, L)
	all_fem_splits = Vector{Vector{Vector{Any}}}(undef, L)

	for l in 1:L
		exp     = mset.exponents[l]
		deg_max = sum(exp)
		candidate_indices = indices_in_box_with_bounded_degree(mset, exp, 1, deg_max)
		external_exp      = SVector(ntuple(i -> exp[ROM + i], external_system_size))

		term_splits     = Vector{Vector{CachedSplit}}(undef, n_terms)
		fem_term_splits = Vector{Vector{Any}}(undef, n_terms)

		for (t_idx, t) in enumerate(model.nonlinear_terms)
			if t.deg > deg_max
				term_splits[t_idx]     = CachedSplit[]
				fem_term_splits[t_idx] = []
				continue
			end

			me      = t.multiplicity_external
			deg     = t.deg - me

			if t isa FEMMultilinearMap
				# Build CachedSplit first (to reuse factorisation logic), then convert to FEMCachedSplit.
				sym     = symmetry_type(t)
				is_asym = sym isa FullyAsymmetric
				orders  = _orders_for_cache(sym, t, deg)
				raw_splits = CachedSplit[]
				if me == 0
					entries = _collect_entries(sym, t, mset, exp, deg, candidate_indices)
					push!(raw_splits, CachedSplit(1, Int[], is_asym, orders, entries))
				else
					for (ext_idx, ext_multiindex_external, ext_count) in bounded_index_tuples(me, external_exp)
						ext_multiindex = SVector(ntuple(i -> i <= ROM ? 0 : ext_multiindex_external[i - ROM], Val(NVAR)))
						rem     = exp - ext_multiindex
						entries = _collect_entries(sym, t, mset, rem, deg, candidate_indices)
						push!(raw_splits, CachedSplit(ext_count, collect(Int, ext_idx), is_asym, orders, entries))
					end
				end
				term_splits[t_idx] = raw_splits  # kept for reference / fallback
				fem_term_splits[t_idx] = [_build_fem_cached_split(Val(deg), cs) for cs in raw_splits]
			else
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
				term_splits[t_idx]     = splits
				fem_term_splits[t_idx] = []
			end
		end

		all_splits[l]     = term_splits
		all_fem_splits[l] = fem_term_splits
	end

	T    = eltype(parametrisation.poly)
	FOM  = size(parametrisation)
	unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size))
	                for j in 1:external_system_size]

	# Compute the element-residual buffer size from the FEM terms in the model.
	# The qp gradient buffer (∇W_qp) is term-specific and lives inside each FEMMultilinearMap.
	max_ndofs = 0
	for t in model.nonlinear_terms
		t isa FEMMultilinearMap || continue
		max_ndofs = max(max_ndofs, fem_ndofs_per_cell(t))
	end
	fem_Fe = zeros(T, max_ndofs)

	return MultilinearTermsCache{T}(all_splits, all_fem_splits,
	                                zeros(T, FOM), zeros(T, FOM), zeros(T, FOM),
	                                unit_vectors, fem_Fe)
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

# Replay one FEMCachedSplit using a single element-loop (RHS-C batched path).
# Fe is the pre-allocated element-residual buffer from MultilinearTermsCache.
# The qp-level gradient buffer ∇W_qp is obtained from the term via fem_qp_buffer(t),
# which allows each FEMMultilinearMap subtype to use its own concrete element type.
function _replay_fem_split!(result, t::FEMMultilinearMap, W, fem_split::FEMCachedSplit{DEG},
		Fe) where {DEG}

	∇W_qp = fem_qp_buffer(t)   # Matrix{QP_TYPE}(max_unique, n_qp) — owned by the term

	if isempty(fem_split.args_ext_indices)
		accum = result
	else
		# External-forcing case: accumulate into a temporary slice of Fe, then axpy!.
		# (Handling is analogous to _replay_split! for the me>0 branch.)
		fill!(Fe, zero(eltype(Fe)))
		accum = Fe   # will be axpy!-ed into result after the loop
	end

	n_qp   = fem_n_qp(t)
	n_dofs = fem_ndofs_per_cell(t)
	n_uniq = length(fem_split.unique_cols)

	for element in fem_elements(t)

		# 1. Scatter each unique (order, col) W column to qp-level field quantities.
		for i in 1:n_uniq
			(order, col) = fem_split.unique_cols[i]
			scatter_qp!(@view(∇W_qp[i, 1:n_qp]), @view(W[:, order, col]), element, t)
		end

		# 2. Accumulate contributions from ALL fem_entries at each quadrature point.
		fill!(@view(Fe[1:n_dofs]), zero(eltype(Fe)))
		for q in 1:n_qp
			dΩ = fem_getdetJdV(element, q, t)
			for fem_entry in fem_split.fem_entries
				@inbounds ∇W_args = ntuple(k -> ∇W_qp[fem_entry.local_factor_indices[k], q], Val(DEG))
				accumulate_qp!(@view(Fe[1:n_dofs]), ∇W_args, fem_entry.multiplier, element, q, dΩ, t)
			end
		end

		# 3. Scatter element residual to global accumulator.
		assemble_element!(accum, @view(Fe[1:n_dofs]), element, t)
	end

	isempty(fem_split.args_ext_indices) || axpy!(fem_split.ext_count, Fe, result)
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

"""
	compute_multilinear_terms!(result, model, exp_index, parametrisation, cache) → nothing

In-place variant: zeros `result` then accumulates all nonlinear contributions into it.
Uses `cache.scratch_buffer`, `cache.temp_buffer`, and `cache.unit_vectors` so that no
heap allocation occurs during the inner solve loop.
"""
function compute_multilinear_terms!(
		result::AbstractVector,
		model::NDOrderModel{ORD}, exp_index::Int,
		parametrisation::Parametrisation{ORD, NVAR},
		cache::MultilinearTermsCache) where {ORD, NVAR}

	fill!(result, zero(eltype(result)))
	W       = parametrisation.poly.coefficients
	deg_max = sum(parametrisation.poly.multiindex_set.exponents[exp_index])
	scratch = cache.scratch_buffer
	temp    = cache.temp_buffer
	unit_vectors = cache.unit_vectors

	for (t_idx, t) in enumerate(model.nonlinear_terms)
		t.deg > deg_max && continue
		deg = t.deg - t.multiplicity_external
		if t isa FEMMultilinearMap
			# RHS-C batched path: one element-loop per split instead of n_entries loops.
			for fem_split in cache.fem_splits[exp_index][t_idx]
				_replay_fem_split!(result, t, W, fem_split, cache.fem_Fe)
			end
		else
			for split in cache.splits[exp_index][t_idx]
				_replay_split!(result, scratch, temp, t, W, split, deg, unit_vectors)
			end
		end
	end
	return nothing
end

end # module
