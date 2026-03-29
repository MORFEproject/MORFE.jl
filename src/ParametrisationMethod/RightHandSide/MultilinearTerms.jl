module MultilinearTerms

using ..Multiindices: indices_in_box_with_bounded_degree,
	factorisations_asymmetric, factorisations_fully_symmetric, factorisations_groupwise_symmetric,
	bounded_index_tuples

using ..ParametrisationMethod: Parametrisation
using ..FullOrderModel: NDOrderModel, FirstOrderModel, MultilinearMap

# -----------------------------------------------------------------------
# Symmetry classification  (pure function, no allocation)
# -----------------------------------------------------------------------

abstract type SymmetryType end
struct FullyAsymmetric <: SymmetryType end  # all multiindex entries ≤ 1
struct FullySymmetric <: SymmetryType end  # exactly one positive entry with value > 1
struct GroupwiseSymmetric <: SymmetryType end  # multiple positive entries

"""
	symmetry_type(t) → SymmetryType

Classify the symmetry of `t` from its `multiindex`. The asymmetric check
precedes the fully-symmetric one so that the degenerate case of a single
entry of value 1 — which needs no sym_count scaling — is routed to the
cheaper asymmetric path.
"""
function symmetry_type(t::MultilinearMap)
	all(x -> x <= 1, t.multiindex) && return FullyAsymmetric()
	count(>(0), t.multiindex) == 1 && return FullySymmetric()
	return GroupwiseSymmetric()
end

# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

"""
	_derivative_orders(t) → NTuple{ORD, Int}

Return a stack-allocated tuple of length `t.deg` where entry `i` is the
1-based derivative order for factor slot `i`, according to the multiplicities
in `t.multiindex`.

Example: `t.multiindex = (2, 1, 1)` → `(1, 1, 2, 3)`.
"""
function _derivative_orders(t::MultilinearMap{ORD}) where {ORD}
	return ntuple(Val(ORD)) do slot
		cumulative = 0
		for (deriv_idx, cnt) in enumerate(t.multiindex)
			cumulative += cnt
			slot <= cumulative && return deriv_idx
		end
	end
end

# -----------------------------------------------------------------------
# Accumulation functions
# -----------------------------------------------------------------------

"""
	accumulate_asymmetric!(result, t, orders, deg_internal, rem, set, W, candidate_indices, args_ext)

All factor slots belong to distinct derivative orders (every entry of
`t.multiindex` is ≤ 1). `t.f!` increments its first argument in place, so
no temporary buffer or multiplicity scaling is needed — each call
accumulates directly into `result`.
"""
function accumulate_asymmetric!(result,
	t::MultilinearMap{ORD}, orders, deg_internal::Int,
	rem, set, W, candidate_indices, args_ext) where {ORD}

	for idx_tuple in factorisations_asymmetric(set, rem, deg_internal, candidate_indices)
		@inbounds args = ntuple(i -> W[idx_tuple[i]][orders[i]], Val(deg_internal))
		t.f!(result, args..., args_ext...)
	end
end

"""
	accumulate_symmetric!(result, temp, t, deriv_idx, deg_internal, rem, set, W, candidate_indices, args_ext)

All factor slots share derivative order `deriv_idx`. Each factorisation
carries `sym_count` to account for internal permutation symmetry; `t.f!`
evaluates into the scratch buffer `temp`, which is then fused-broadcast
scaled and accumulated into `result` with no intermediate allocation.
"""
function accumulate_symmetric!(result, temp,
	t::MultilinearMap{ORD}, deriv_idx::Int, deg_internal::Int,
	rem, set, W, candidate_indices, args_ext) where {ORD}

	for (idx_tuple, sym_count) in factorisations_fully_symmetric(set, rem, deg_internal, candidate_indices)
		fill!(temp, 0)
		@inbounds args = ntuple(i -> W[idx_tuple[i]][deriv_idx], Val(deg_internal))
		t.f!(temp, args..., args_ext...)
		@. result += sym_count * temp
	end
end

"""
	accumulate_partial!(result, temp, t, orders, deg_internal, rem, set, W, candidate_indices, args_ext)

Factor slots belong to multiple derivative orders (groupwise symmetry). Each
factorisation carries `total_count` accounting for combined within- and
between-group permutation symmetry.
"""
function accumulate_partial!(result, temp,
	t::MultilinearMap{ORD}, orders, deg_internal::Int,
	rem, set, W, candidate_indices, args_ext) where {ORD}

	for (idx_tuple, total_count) in factorisations_groupwise_symmetric(set, rem, t.multiindex, candidate_indices)
		fill!(temp, 0)
		@inbounds args = ntuple(i -> W[idx_tuple[i]][orders[i]], Val(deg_internal))
		t.f!(temp, args..., args_ext...)
		@. result += total_count * temp
	end
end

# -----------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------

"""
	accumulate_multilinear_term!(result, temp, temp2, t, exp, parametrisation)

Classify the symmetry type of `t`, then iterate over all external-forcing
index splits and delegate to the appropriate accumulation function.

# Scratch buffers
- `temp`  — accumulation target for one (f_idx, f_multiindex) split; zeroed
			 at the start of each outer iteration.
- `temp2` — inner scratch for `accumulate_symmetric!` / `accumulate_partial!`,
			 zeroed before each individual `t.f!` evaluation.

Both buffers are allocated once by the caller and reused across all terms,
so this function performs no heap allocation of its own.

# Contract on `t.f!`
`t.f!` must *increment* (not overwrite) its first argument. The asymmetric
path relies on this to accumulate multiple factorisation contributions
directly into `temp` without a temporary.
"""
function accumulate_multilinear_term!(result, temp, temp2,
	t::MultilinearMap{ORD}, exp, parametrisation::Parametrisation{ORD}) where {ORD}

	W            = parametrisation.poly.coefficients
	set          = parametrisation.multiindex_set
	me           = t.multiplicity_external
	deg_internal = t.deg - me
	sym          = symmetry_type(t)

	# orders: shared by the asymmetric and partial branches; skipped for symmetric.
	orders = sym isa FullySymmetric ? nothing : _derivative_orders(t)
	# deriv_idx: used only by the symmetric branch.
	deriv_idx = sym isa FullySymmetric ? findfirst(>(0), t.multiindex) : 0

	# unit_vectors: built only when external forcing is present.
	forcing_size = parametrisation.forcing_size
	unit_vectors = me > 0 ?
				   [SVector(ntuple(k -> k == j ? 1 : 0, forcing_size)) for j in 1:forcing_size] :
				   SVector{0, Int}[]

	for (f_idx, f_multiindex, f_count) in bounded_index_tuples(me, exp)
		rem = exp - f_multiindex
		# candidate_indices must reflect rem (the exponent after
		# stripping the external forcing contribution).
		candidate_indices = indices_in_box_with_bounded_degree(set, rem, 1, sum(rem))
		args_ext          = me > 0 ? ntuple(i -> unit_vectors[f_idx[i]], me) : ()

		fill!(temp, 0)

		if sym isa FullyAsymmetric
			accumulate_asymmetric!(temp, t, orders, deg_internal,
				rem, set, W, candidate_indices, args_ext)
		elseif sym isa FullySymmetric
			accumulate_symmetric!(temp, temp2, t, deriv_idx, deg_internal,
				rem, set, W, candidate_indices, args_ext)
		else
			accumulate_partial!(temp, temp2, t, orders, deg_internal,
				rem, set, W, candidate_indices, args_ext)
		end

		@. result += f_count * temp
	end
end

# -----------------------------------------------------------------------
# Top-level routines
# -----------------------------------------------------------------------

"""
	compute_multilinear_terms(model, exp, parametrisation)

Sum contributions of all nonlinear terms in `model` for exponent vector `exp`
and return the accumulated result array.

`temp` and `temp2` are allocated once here and threaded through the entire
call stack, so no allocation occurs inside the term loop.
"""
function compute_multilinear_terms(model::NDOrderModel{ORD}, exp,
	parametrisation::Parametrisation{ORD}) where {ORD}

	deg_max     = sum(exp)
	first_coeff = parametrisation.poly.coefficients[1][1]
	result      = zeros(eltype(first_coeff), size(first_coeff))
	temp        = similar(result)
	temp2       = similar(result)

	for t in model.nonlinear_terms
		t.deg > deg_max && continue
		accumulate_multilinear_term!(result, temp, temp2, t, exp, parametrisation)
	end
	return result
end

end # module
