module MultilinearTerms

using StaticArrays: SVector

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
struct FullySymmetric <: SymmetryType end   # exactly one positive entry with value > 1
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
function _derivative_orders(t::MultilinearMap)
	deg = sum(t.multiindex)
	return ntuple(deg) do slot
		cumulative = 0
		for (deriv_idx, cnt) in enumerate(t.multiindex)
			cumulative += cnt
			slot <= cumulative && return deriv_idx
		end
	end
end

"""
	_sym_param(sym, t)

Return the symmetry-branch parameter that `_accumulate_inner!` needs:
- `FullySymmetric`    → `Int`  (the single non-zero derivative index).
- All other branches  → `NTuple` of per-slot derivative orders.

Dispatching here eliminates both sentinel values (`orders = nothing`,
`deriv_idx = 0`) and the `isa` check that previously computed them.
"""
_sym_param(::FullySymmetric, t) = findfirst(>(0), t.multiindex)::Int
_sym_param(::SymmetryType, t) = _derivative_orders(t)

# -----------------------------------------------------------------------
# AccumContext — per-iteration context bundle
# -----------------------------------------------------------------------

"""
	AccumContext{W, S, R, CI, AE}

Immutable, stack-allocated bundle of the data that varies across the
outer forcing-split loop in `accumulate_multilinear_term!`.

# Fields
- `W`                  — coefficient array of the parametrisation polynomial.
- `set`                — multiindex set of the parametrisation.
- `rem`                — exponent remainder after stripping the forcing contribution.
- `candidate_indices`  — pre-filtered indices compatible with `rem`.
- `args_ext`           — tuple of unit vectors for the external forcing slots.

Bundling these four arguments reduces every `_accumulate_inner!` specialisation
from nine positional arguments to five, and makes the per-call allocation
pattern explicit and easy to audit.
"""
struct AccumContext{W, S, R, CI, AE}
	W::W
	set::S
	rem::R
	candidate_indices::CI
	args_ext::AE
end

# -----------------------------------------------------------------------
# _accumulate_inner! — dispatched on SymmetryType
# -----------------------------------------------------------------------

"""
	_accumulate_inner!(accum, scratch, t, ::FullyAsymmetric, orders, deg_internal, ctx)

All factor slots belong to distinct derivative orders. `t.f!` increments
`accum` directly; no scratch buffer or multiplicity scaling is needed.
"""
function _accumulate_inner!(accum, _scratch,
	t::MultilinearMap{ORD}, ::FullyAsymmetric, orders, deg_internal::Int,
	ctx::AccumContext) where {ORD}

	for idx_tuple in factorisations_asymmetric(ctx.set, ctx.rem, deg_internal, ctx.candidate_indices)
		@inbounds args = ntuple(i -> ctx.W[idx_tuple[i]][orders[i]], Val(deg_internal))
		t.f!(accum, args..., ctx.args_ext...)
	end
end

"""
	_accumulate_inner!(accum, scratch, t, ::FullySymmetric, deriv_idx, deg_internal, ctx)

All factor slots share a single derivative order `deriv_idx`. Each
factorisation carries `sym_count`; `scratch` is zeroed before each `t.f!`
evaluation, then fused-broadcast scaled and accumulated into `accum`.
"""
function _accumulate_inner!(accum, scratch,
	t::MultilinearMap{ORD}, ::FullySymmetric, deriv_idx::Int, deg_internal::Int,
	ctx::AccumContext) where {ORD}

	for (idx_tuple, sym_count) in factorisations_fully_symmetric(ctx.set, ctx.rem, deg_internal, ctx.candidate_indices)
		fill!(scratch, 0)
		@inbounds args = ntuple(i -> ctx.W[idx_tuple[i]][deriv_idx], Val(deg_internal))
		t.f!(scratch, args..., ctx.args_ext...)
		@. accum += sym_count * scratch
	end
end

"""
	_accumulate_inner!(accum, scratch, t, ::GroupwiseSymmetric, orders, deg_internal, ctx)

Factor slots belong to multiple derivative orders. Each factorisation carries
`total_count` accounting for combined within- and between-group permutation
symmetry. `scratch` is zeroed before each `t.f!` evaluation.
"""
function _accumulate_inner!(accum, scratch,
	t::MultilinearMap{ORD}, ::GroupwiseSymmetric, orders, deg_internal::Int,
	ctx::AccumContext) where {ORD}

	for (idx_tuple, total_count) in factorisations_groupwise_symmetric(ctx.set, ctx.rem, t.multiindex, ctx.candidate_indices)
		fill!(scratch, 0)
		@inbounds args = ntuple(i -> ctx.W[idx_tuple[i]][orders[i]], Val(deg_internal))
		t.f!(scratch, args..., ctx.args_ext...)
		@. accum += total_count * scratch
	end
end

# -----------------------------------------------------------------------
# accumulate_multilinear_term!
# -----------------------------------------------------------------------

"""
	accumulate_multilinear_term!(result, temp, temp2, t, exp, parametrisation, unit_vectors)

Classify the symmetry type of `t`, iterate over all external-forcing index
splits, and delegate to the appropriate `_accumulate_inner!` specialisation.

# Argument roles
- `result`       — running total across all terms; incremented in place.
- `temp`         — per-forcing-split accumulation buffer; zeroed each iteration.
- `temp2`        — per-factorisation scratch for the symmetric/partial branches.
- `unit_vectors` — precomputed forcing unit vectors, passed in from
				   `compute_multilinear_terms` to avoid repeated allocation.

# Dispatch contract
`symmetry_type(t)` returns a `FullyAsymmetric`, `FullySymmetric`, or
`GroupwiseSymmetric` tag. `_sym_param` extracts the branch-specific constant
(an `Int` for symmetric, an `NTuple` for the others) without `isa` checks or
sentinel values. `_accumulate_inner!` is then specialised at compile time by
the Julia dispatcher on both the symmetry tag and the param type.

# Contract on `t.f!`
`t.f!` must *increment* (not overwrite) its first argument. The asymmetric
branch relies on this to accumulate multiple factorisation contributions
directly into `temp` without a per-factorisation temporary.
"""
function accumulate_multilinear_term!(result, temp, temp2,
	t::MultilinearMap{ORD}, exp, parametrisation::Parametrisation{ORD},
	unit_vectors) where {ORD}

	W            = parametrisation.poly.coefficients
	set          = parametrisation.poly.multiindex_set
	me           = t.multiplicity_external
	deg_internal = t.deg - me

	sym   = symmetry_type(t)
	param = _sym_param(sym, t)

	for (f_idx, f_multiindex, f_count) in bounded_index_tuples(me, exp)
		rem               = exp - f_multiindex
		candidate_indices = indices_in_box_with_bounded_degree(set, rem, 1, sum(rem))
		args_ext          = me > 0 ? ntuple(i -> unit_vectors[f_idx[i]], me) : ()
		ctx               = AccumContext(W, set, rem, candidate_indices, args_ext)

		fill!(temp, 0)
		_accumulate_inner!(temp, temp2, t, sym, param, deg_internal, ctx)
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

`temp` and `temp2` are allocated once and threaded through the entire call
stack; `unit_vectors` is likewise built once here (rather than inside each
`accumulate_multilinear_term!` call) and shared across all terms.
No allocation occurs inside the term loop.
"""
function compute_multilinear_terms(model::NDOrderModel{ORD}, exp,
	parametrisation::Parametrisation{ORD}) where {ORD}

	deg_max     = sum(exp)
	first_coeff = parametrisation.poly.coefficients[1][1]
	result      = zeros(eltype(first_coeff), size(first_coeff))
	temp        = similar(result)
	temp2       = similar(result)

	forcing_size = parametrisation.forcing_size
	unit_vectors = forcing_size > 0 ?
				   [SVector(ntuple(k -> k == j ? 1 : 0, forcing_size)) for j in 1:forcing_size] :
				   SVector{0, Int}[]

	for t in model.nonlinear_terms
		t.deg > deg_max && continue
		accumulate_multilinear_term!(result, temp, temp2, t, exp, parametrisation, unit_vectors)
	end
	return result
end

end # module
