module LowerOrderCouplings

using LinearAlgebra
using StaticArrays
using ..Multiindices: MultiindexSet, indices_in_box_with_bounded_degree,
	build_exponent_index_map
using ..Polynomials: nvars
using ..ParametrisationMethod: Parametrisation, ReducedDynamics, coefficients, multiindex_set

export compute_lower_order_couplings

# Helper: check whether a coefficient slice (FOM × ORD matrix view) is all zero
@inline _is_zero_coeff(coeff::AbstractMatrix) = iszero(coeff)

# Helper: sum over unit‑vector multi‑indices (total degree 1) from reduced dynamics
@inline function _sum_degree_one_terms!(accumulator::Vector{Vector{T}},
	upper_bound::SVector{NVAR, Int},
	multiindex_dict::Dict{SVector{NVAR, Int}, Int},
	red_coefficients::AbstractMatrix{T},
	param_coefficients::AbstractArray{T, 3}) where {NVAR, T}
	ORD = size(param_coefficients, 2)
	# Pre‑compute all unit vectors as SVector
	unit_vectors = [SVector{NVAR, Int}(ntuple(k -> k == j ? 1 : 0, Val(NVAR)))
					for j in 1:NVAR]

	for j in 1:NVAR
		upper_bound[j] < 1 && continue

		unit_vec = unit_vectors[j]
		idx_unit = get(multiindex_dict, unit_vec, nothing)
		idx_unit === nothing && continue

		red_coeff = @view red_coefficients[:, idx_unit]   # length-ROM view

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
				acc_vec = accumulator[k]
				@inbounds for l in eachindex(acc_vec)
					acc_vec[l] += factor * param_coeff[l, k]
				end
			end
		end
	end
end

# Helper: sum over multi‑indices of total degree ≥ 2 from reduced dynamics
@inline function _sum_higher_degree_terms!(accumulator::Vector{Vector{T}},
	upper_bound::SVector{NVAR, Int},
	mset::MultiindexSet,
	multiindex_dict::Dict{SVector{NVAR, Int}, Int},
	red_coefficients::AbstractMatrix{T},
	param_coefficients::AbstractArray{T, 3},
	total_deg_upper::Int) where {NVAR, T}
	ORD = size(param_coefficients, 2)
	candidate_idxs = indices_in_box_with_bounded_degree(
		mset, collect(upper_bound), 2, total_deg_upper)
	isempty(candidate_idxs) && return

	# Pre‑compute unit vectors
	unit_vectors = [SVector{NVAR, Int}(ntuple(k -> k == i ? 1 : 0, Val(NVAR)))
					for i in 1:NVAR]

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
				acc_vec = accumulator[k]
				@inbounds for l in eachindex(acc_vec)
					acc_vec[l] += factor * param_coeff[l, k]
				end
			end
		end
	end
end

# -------------------------------------------------------------------
# Main entry point
function compute_lower_order_couplings(upper_bound::SVector{NVAR, Int},
	parametrisation::Parametrisation{ORD, NVAR, T},
	reduced_dynamics::ReducedDynamics{ROM, NVAR, T}) where {ORD, NVAR, ROM, T}
	FOM = size(parametrisation)
	total_deg_upper = sum(upper_bound)
	total_deg_upper < 2 && return SVector{ORD, Vector{T}}(ntuple(_ -> zeros(T, FOM), ORD))

	mset = multiindex_set(parametrisation)
	@assert mset.exponents == multiindex_set(reduced_dynamics).exponents

	multiindex_dict = build_exponent_index_map(mset)  # Dict{SVector{NVAR,Int},Int}

	red_coefficients   = coefficients(reduced_dynamics)    # Matrix{T}   (ROM × L)
	param_coefficients = coefficients(parametrisation)     # Array{T,3}  (FOM × ORD × L)

	# Accumulator: one FOM-vector per order component
	accumulator = [zeros(T, FOM) for _ in 1:ORD]

	_sum_degree_one_terms!(accumulator, upper_bound,
		multiindex_dict, red_coefficients, param_coefficients)

	_sum_higher_degree_terms!(accumulator, upper_bound,
		mset, multiindex_dict, red_coefficients, param_coefficients, total_deg_upper)

	return SVector{ORD, Vector{T}}(tuple(accumulator...))
end

end # module
