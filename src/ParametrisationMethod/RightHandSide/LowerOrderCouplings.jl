"""
Module `LowerOrderCouplings` — computation of lower-order coupling vectors `ξ[α]`.

For each monomial `α` the cohomological right-hand side includes a coupling term

	ξ[α] = Σ_{β+γ=α, |β|≥1} R_β · (|γ₁| W_{γ+e₁} + … + |γₙ| W_{γ+eₙ})

that links the current monomial to all lower-degree coefficients of `W` and `R`
already computed.  `compute_lower_order_couplings` evaluates this sum efficiently
using a precomputed dictionary and candidate index list.
"""
module LowerOrderCouplings

using LinearAlgebra
using StaticArrays
using ..Multiindices: MultiindexSet, build_exponent_index_map,
	indices_in_box_with_bounded_degree
using ..Polynomials: nvars
using ..ParametrisationMethod: Parametrisation, ReducedDynamics, coefficients,
	multiindex_set
using StaticArrays: SVector

export compute_lower_order_couplings

"""
	_is_zero_coeff(coeff) -> Bool

Return `true` when the FOM × ORD coefficient slice `coeff` is entirely zero.
Used to skip trivial contributions without entering the accumulation loop.
"""
@inline _is_zero_coeff(coeff::AbstractMatrix) = iszero(coeff)

"""
	_sum_degree_one_terms!(accumulator, upper_bound, multiindex_dict,
						   red_coefficients, param_coefficients, unit_vectors)

Accumulate the `|β| = 1` part of the lower-order coupling sum into `accumulator`.

For each unit-vector `eⱼ` with `eⱼ ≤ upper_bound`, and for each `i < j`, adds
`R[i, eⱼ] * (upper_bound - eⱼ + eᵢ)[i] * W[:, :, upper_bound - eⱼ + eᵢ]`
to `accumulator`.  Only reads precomputed `multiindex_dict` — no allocations.
"""
@inline function _sum_degree_one_terms!(accumulator::Vector{Vector{T}},
	upper_bound::SVector{NVAR, Int},
	multiindex_dict::Dict{SVector{NVAR, Int}, Int},
	red_coefficients::AbstractMatrix{T},
	param_coefficients::AbstractArray{T, 3},
	unit_vectors::AbstractVector{SVector{NVAR, Int}}) where {NVAR, T}
	ORD = size(param_coefficients, 2)

	for j in 1:NVAR
		upper_bound[j] < 1 && continue

		unit_vec = unit_vectors[j]
		idx_unit = get(multiindex_dict, unit_vec, nothing)
		idx_unit === nothing && continue

		red_coeff = @view red_coefficients[:, idx_unit]   # length-NVAR view

		# difference = upper_bound - eⱼ
		diff = upper_bound - unit_vec   # SVector{NVAR,Int}

		for i in 1:(j-1)
			@inbounds red_val = red_coeff[i]
			iszero(red_val) && continue

			# exponent for parametrisation: diff + eᵢ
			param_exp = diff + unit_vectors[i]   # SVector{NVAR,Int}
			param_idx = get(multiindex_dict, param_exp, nothing)
			param_idx === nothing && continue

			param_coeff = @view param_coefficients[:, :, param_idx]  # FOM × ORD view
			_is_zero_coeff(param_coeff) && continue

			factor = param_exp[i] * red_val
			for k in 1:ORD
				axpy!(factor, @view(param_coeff[:, k]), accumulator[k])
			end
		end
	end
end

"""
	_sum_higher_degree_terms!(accumulator, upper_bound, mset, multiindex_dict,
							  red_coefficients, param_coefficients,
							  candidate_idxs, unit_vectors)

Accumulate the `|β| ≥ 2` part of the lower-order coupling sum into `accumulator`.

Iterates over `candidate_idxs` (precomputed as exponents componentwise ≤ `upper_bound`
with total degree in `[2, sum(upper_bound)-1]`) and adds the same weighted
`R[i, β] * W[:, :, upper_bound - β + eᵢ]` contributions as `_sum_degree_one_terms!`.
"""
@inline function _sum_higher_degree_terms!(accumulator::Vector{Vector{T}},
	upper_bound::SVector{NVAR, Int},
	mset::MultiindexSet,
	multiindex_dict::Dict{SVector{NVAR, Int}, Int},
	red_coefficients::AbstractMatrix{T},
	param_coefficients::AbstractArray{T, 3},
	candidate_idxs::AbstractVector{Int},
	unit_vectors::AbstractVector{SVector{NVAR, Int}}) where {NVAR, T}
	ORD = size(param_coefficients, 2)
	isempty(candidate_idxs) && return

	exps = mset.exponents
	@inbounds for idx in candidate_idxs
		multiindex = exps[idx]
		red_coeff = @view red_coefficients[:, idx]   # length-ROM view

		diff = upper_bound - multiindex

		for i in 1:NVAR
			@inbounds red_val = red_coeff[i]
			iszero(red_val) && continue

			param_exp = diff + unit_vectors[i]
			param_idx = get(multiindex_dict, param_exp, nothing)
			param_idx === nothing && continue

			param_coeff = @view param_coefficients[:, :, param_idx]  # FOM × ORD view
			_is_zero_coeff(param_coeff) && continue

			factor = param_exp[i] * red_val
			for k in 1:ORD
				axpy!(factor, @view(param_coeff[:, k]), accumulator[k])
			end
		end
	end
end

"""
	compute_lower_order_couplings(upper_bound, parametrisation, reduced_dynamics,
								  multiindex_dict, accumulator, candidate_idxs,
								  unit_vectors) -> accumulator

**Performance path.** Evaluate the lower-order coupling vector `ξ[α]` for monomial
`α = upper_bound` and accumulate the result into the pre-zeroed `accumulator`.

All auxiliary data (`multiindex_dict`, `accumulator`, `candidate_idxs`,
`unit_vectors`) must be pre-allocated by the caller (typically via `LowerOrderResources`)
to avoid heap allocation in the inner solve loop.  Returns `accumulator` for convenience.
"""
function compute_lower_order_couplings(upper_bound::SVector{NVAR, Int},
	parametrisation::Parametrisation{ORD, NVAR, T},
	reduced_dynamics::ReducedDynamics{ROM, NVAR, T},
	multiindex_dict::Dict{SVector{NVAR, Int}, Int},
	accumulator::Vector{Vector{T}},
	candidate_idxs::AbstractVector{Int},
	unit_vectors::AbstractVector{SVector{NVAR, Int}},
) where {ORD, NVAR, ROM, T}
	total_deg_upper = sum(upper_bound)
	total_deg_upper < 2 && return accumulator

	mset = multiindex_set(parametrisation)

	red_coefficients = coefficients(reduced_dynamics)    # Matrix{T}   (ROM × L)
	param_coefficients = coefficients(parametrisation)     # Array{T,3}  (FOM × ORD × L)

	_sum_degree_one_terms!(accumulator, upper_bound,
		multiindex_dict, red_coefficients, param_coefficients, unit_vectors)

	_sum_higher_degree_terms!(accumulator, upper_bound,
		mset, multiindex_dict, red_coefficients, param_coefficients, candidate_idxs,
		unit_vectors)

	return accumulator
end

"""
	compute_lower_order_couplings(upper_bound, parametrisation, reduced_dynamics) -> Vector{Vector{T}}

**Allocating convenience wrapper.** Builds the multiindex dictionary, accumulator,
and candidate index list internally, then delegates to the performance path.

Use the 7-argument form inside performance-critical loops; this overload is intended
for interactive use or testing.
"""
function compute_lower_order_couplings(
	upper_bound::AbstractVector{Int},
	parametrisation::Parametrisation{ORD, NVAR, T},
	reduced_dynamics::ReducedDynamics{ROM, NVAR, T},
) where {ORD, NVAR, ROM, T}
	mset = multiindex_set(parametrisation)
	FOM = size(coefficients(parametrisation), 1)
	multiindex_dict = build_exponent_index_map(mset)
	accumulator = [zeros(T, FOM) for _ in 1:ORD]
	total_deg = sum(upper_bound)
	candidate_idxs = total_deg < 2 ? Int[] :
					 indices_in_box_with_bounded_degree(mset, upper_bound, 2, total_deg)
	ub_svec = SVector{NVAR, Int}(upper_bound)
	unit_vectors = [SVector{NVAR, Int}(ntuple(k -> k == j ? 1 : 0, Val(NVAR)))
					for j in 1:NVAR]
	return compute_lower_order_couplings(ub_svec, parametrisation, reduced_dynamics,
		multiindex_dict, accumulator, candidate_idxs, unit_vectors)
end

end # module
