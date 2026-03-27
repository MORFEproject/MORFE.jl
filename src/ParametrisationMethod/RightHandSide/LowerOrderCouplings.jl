module LowerOrderCouplings

using LinearAlgebra
using StaticArrays
using ..Multiindices: MultiindexSet, indices_in_box_with_bounded_degree,
                      build_exponent_index_map
using ..Polynomials: coeffs, multiindex_set, nvars
using ..ParametrisationMethod: Parametrisation, ReducedDynamics

export compute_lower_order_couplings

# -------------------------------------------------------------------
# Helper: sum over unit‑vector multi‑indices (total degree 1) from reduced dynamics
@inline function _sum_degree_one_terms(upper_bound::SVector{NVAR, Int},
        multiindex_dict::Dict{SVector{NVAR, Int}, Int},
        red_coeffs::Vector{SVector{ROM, T}},
        param_coeffs::Vector{SVector{ORD, SVector{FOM, T}}},
        init_zero::SVector{ORD, SVector{FOM, T}}) where {NVAR, ORD, FOM, ROM, T}
    summation = init_zero

    # Pre‑compute all unit vectors as SVector
    unit_vectors = [SVector{NVAR, Int}(ntuple(k -> k == j ? 1 : 0, Val(NVAR)))
                    for j in 1:NVAR]

    for j in 1:NVAR
        upper_bound[j] < 1 && continue

        unit_vec = unit_vectors[j]
        idx_unit = get(multiindex_dict, unit_vec, nothing)
        idx_unit === nothing && continue

        reduced_dynamics_coeff = red_coeffs[idx_unit]   # SVector{ROM,T}

        # difference = upper_bound - eⱼ
        diff = upper_bound - unit_vec   # SVector{NVAR,Int}

        for i in 1:(j - 1)
            @inbounds reduced_dynamics_val = reduced_dynamics_coeff[i]
            iszero(reduced_dynamics_val) && continue

            # exponent for parametrisation: diff + eᵢ
            param_exp = diff + unit_vectors[i]   # SVector{NVAR,Int}
            param_idx = get(multiindex_dict, param_exp, nothing)
            param_idx === nothing && continue

            @inbounds parametrisation_coeff = param_coeffs[param_idx]
            iszero(parametrisation_coeff) && continue

            # factor = upper_bound[i] - 0 + 1 = diff[i] + 1 = param_exp[i]
            summation += muladd(
                parametrisation_coeff, param_exp[i] * reduced_dynamics_val, init_zero)
        end
    end
    return summation
end

# -------------------------------------------------------------------
# Helper: sum over multi‑indices of total degree ≥ 2 from reduced dynamics
@inline function _sum_higher_degree_terms(upper_bound::SVector{NVAR, Int},
        mset::MultiindexSet,
        multiindex_dict::Dict{SVector{NVAR, Int}, Int},
        red_coeffs::Vector{SVector{ROM, T}},
        param_coeffs::Vector{SVector{ORD, SVector{FOM, T}}},
        total_deg_upper::Int,
        init_zero::SVector{ORD, SVector{FOM, T}}) where {NVAR, ORD, FOM, ROM, T}
    summation = init_zero

    candidate_idxs = indices_in_box_with_bounded_degree(
        mset, collect(upper_bound), 2, total_deg_upper)
    isempty(candidate_idxs) && return summation

    # Pre‑compute unit vectors
    unit_vectors = [SVector{NVAR, Int}(ntuple(k -> k == i ? 1 : 0, Val(NVAR)))
                    for i in 1:NVAR]

    exps = mset.exponents
    @inbounds for idx in candidate_idxs
        multiindex = exps[idx]
        reduced_dynamics_coeff = red_coeffs[idx]

        diff = upper_bound - multiindex

        for i in 1:NVAR
            @inbounds reduced_dynamics_val = reduced_dynamics_coeff[i]
            iszero(reduced_dynamics_val) && continue

            param_exp = diff + unit_vectors[i]
            param_idx = get(multiindex_dict, param_exp, nothing)
            param_idx === nothing && continue

            @inbounds parametrisation_coeff = param_coeffs[param_idx]
            iszero(parametrisation_coeff) && continue

            summation += muladd(
                parametrisation_coeff, param_exp[i] * reduced_dynamics_val, init_zero)
        end
    end
    return summation
end

# -------------------------------------------------------------------
# Main entry point
function compute_lower_order_couplings(upper_bound::SVector{NVAR, Int},
        parametrisation::Parametrisation{ORD, NVAR, T},
        reduced_dynamics::ReducedDynamics{ROM, NVAR, T}) where {ORD, NVAR, ROM, T}
    FOM = length(parametrisation.coefficients[1][1])
    total_deg_upper = sum(upper_bound)
    init_zero = zero(SVector{ORD, SVector{FOM, T}})   # safe even if polynomial is empty
    total_deg_upper < 2 && return init_zero

    mset = multiindex_set(parametrisation)
    @assert mset.exponents == multiindex_set(reduced_dynamics).exponents

    multiindex_dict = build_exponent_index_map(mset)  # Dict{SVector{NVAR,Int},Int}

    red_coeffs = coeffs(reduced_dynamics)   # Vector{SVector{ROM,T}}
    param_coeffs = coeffs(parametrisation)    # Vector{SVector{ORD,SVector{FOM,T}}}

    sum_deg1 = _sum_degree_one_terms(upper_bound,
        multiindex_dict,
        red_coeffs,
        param_coeffs,
        init_zero)

    sum_higher = _sum_higher_degree_terms(upper_bound,
        mset,
        multiindex_dict,
        red_coeffs,
        param_coeffs,
        total_deg_upper,
        init_zero)

    return sum_deg1 + sum_higher
end

end # module