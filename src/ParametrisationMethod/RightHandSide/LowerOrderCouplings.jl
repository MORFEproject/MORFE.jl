module LowerOrderCouplings

using LinearAlgebra
include(joinpath(@__DIR__, "../../Polynomials.jl"))
using .Polynomials: MultiindexSet, indices_in_box_with_bounded_degree, build_exponent_index_map
using .Polynomials: DensePolynomial, coeffs, multiindex_set, nvars

export compute_lower_order_couplings

# -------------------------------------------------------------------
# Helper: sum over unit‑vector multi‑indices (total degree 1) from reduced dynamics
"""
    _sum_degree_one_terms(upper_bound, multiindex_dict, red_coeffs, param_coeffs, init_zero)

Compute the contribution to the low‑order coupling sum 
where the reduced dynamics is associated to unit vectors (i.e., `multiindex = eⱼ`).
"""
@inline function _sum_degree_one_terms(upper_bound::NTuple{N,Int},
                                       multiindex_dict::Dict{NTuple{N,Int},Int},
                                       red_coeffs::Vector{Vector{TF}},
                                       param_coeffs::Vector{TW},
                                       init_zero) where {N, TW, TF}
    summation = init_zero
    n = length(upper_bound)

    # Pre‑compute all unit vectors (as tuples) to avoid repeated allocation
    unit_vectors = [ntuple(k -> k == j ? 1 : 0, Val(N)) for j in 1:n]

    for j in 1:n
        upper_bound[j] < 1 && continue

        unit_vec = unit_vectors[j]
        idx_unit = get(multiindex_dict, unit_vec, nothing)
        idx_unit === nothing && continue

        reduced_dynamics_coeff = red_coeffs[idx_unit]   # vector of length n
        # iszero(reduced_dynamics_coeff) && continue

        # difference = upper_bound - eⱼ
        diff = ntuple(k -> upper_bound[k] - (k == j ? 1 : 0), Val(N))

        for i in 1:(j-1)
            @inbounds reduced_dynamics_val = reduced_dynamics_coeff[i]
            iszero(reduced_dynamics_val) && continue

            # exponent for parametrisation: diff + eᵢ
            param_exp = ntuple(k -> diff[k] + (k == i ? 1 : 0), Val(N))
            param_idx = get(multiindex_dict, param_exp, nothing)
            param_idx === nothing && continue

            @inbounds parametrisation_coeff = param_coeffs[param_idx]
            iszero(parametrisation_coeff) && continue

            # factor = upper_bound[i] - 0 + 1 = diff[i] + 1 = param_exp[i]
            summation += muladd(parametrisation_coeff, param_exp[i] * reduced_dynamics_val, init_zero)
        end
    end
    return summation
end

# -------------------------------------------------------------------
# Helper: sum over multi‑indices of total degree ≥ 2 from reduced dynamics
"""
    _sum_higher_degree_terms(upper_bound, multiindex_set, multiindex_dict,
                              red_coeffs, param_coeffs, total_deg_upper, init_zero)

Compute the contribution from multi‑indices `multiindex` with total degree between 2 (inclusive) and
`total_deg_upper` (exclusive) that are componentwise ≤ `upper_bound`.
"""
@inline function _sum_higher_degree_terms(upper_bound::NTuple{N,Int},
                                          multiindex_set::MultiindexSet,
                                          multiindex_dict::Dict{NTuple{N,Int},Int},
                                          red_coeffs::Vector{Vector{TF}},
                                          param_coeffs::Vector{TW},
                                          total_deg_upper::Int,
                                          init_zero) where {N, TW, TF}
    summation = init_zero
    n = length(upper_bound)

    candidate_idxs = indices_in_box_with_bounded_degree(multiindex_set, collect(upper_bound), 2, total_deg_upper)
    isempty(candidate_idxs) && return summation

    exps = multiindex_set.exponents
    @inbounds for idx in candidate_idxs
        multiindex = view(exps, :, idx)
        reduced_dynamics_coeff = red_coeffs[idx]
        # iszero(reduced_dynamics_coeff) && continue

        # difference = upper_bound - multiindex
        diff = ntuple(k -> upper_bound[k] - multiindex[k], Val(N))

        for i in 1:n
            @inbounds reduced_dynamics_val = reduced_dynamics_coeff[i]
            iszero(reduced_dynamics_val) && continue

            # exponent for parametrisation: diff + eᵢ
            param_exp = ntuple(k -> diff[k] + (k == i ? 1 : 0), Val(N))
            param_idx = get(multiindex_dict, param_exp, nothing)
            param_idx === nothing && continue

            @inbounds parametrisation_coeff = param_coeffs[param_idx]
            iszero(parametrisation_coeff) && continue

            # factor = upper_bound[i] - multiindex[i] + 1 = diff[i] + 1 = param_exp[i]
            summation += muladd(parametrisation_coeff, param_exp[i] * reduced_dynamics_val, init_zero)
        end
    end
    return summation
end

# -------------------------------------------------------------------
# Main entry point
"""
    compute_lower_order_couplings(upper_bound, parametrisation, reduced_dynamics)

Compute the combined sum that appears in the low‑order coupling conditions for normal form computations. 
The two polynomials must share the same multi‑index set (same exponents).

# Arguments
- `upper_bound::NTuple{N,Int}`: an N-tuple of non‑negative integers defining the box `[0, upper_bound]`.
- `parametrisation::DensePolynomial{TW}`: a polynomial whose coefficients are scalars of type `TW`.
- `reduced_dynamics::DensePolynomial{TF}`: a polynomial whose coefficients are vectors of length
  `nvars(parametrisation)` of type `TF`.

# Returns
A value of type `promote_op(*, TW, Int, TF)` representing the sum.
"""
function compute_lower_order_couplings(upper_bound::NTuple{N,Int},
                                     parametrisation::DensePolynomial{TW},
                                     reduced_dynamics::DensePolynomial{TF}) where {N, TW, TF}
    n = nvars(parametrisation)
    @assert nvars(reduced_dynamics) == n "Number of variables must match"
    @assert length(upper_bound) == n "Length of upper_bound must equal number of variables"
    @assert all(≥(0), upper_bound) "upper_bound must be non‑negative"

    total_deg_upper = sum(upper_bound)
    # Early return: if total degree < 2, the sum is empty → zero
    total_deg_upper < 2 && return zero(Base.promote_op(*, TW, Int, eltype(TF)))

    # Both polynomials must be defined on the same multi‑index set
    mset = multiindex_set(parametrisation)
    @assert mset.exponents == multiindex_set(reduced_dynamics).exponents

    # Build a dictionary for fast lookup of exponent → index
    multiindex_dict = build_exponent_index_map(mset)

    red_coeffs   = coeffs(reduced_dynamics)   # Vector{TF} where TF <: AbstractVector
    param_coeffs = coeffs(parametrisation)    # Vector{TW} where TW <: Number

    # Determine a zero value of the correct type for accumulation.
    # If the parametrisation has at least one coefficient, use its zero.
    # Otherwise, fall back to a zero of the element type of the reduced dynamics (scalar).
    # (If both are empty, the result is zero, but we cannot know the shape; we assume scalar.)
    init_zero = if !isempty(param_coeffs)
        zero(param_coeffs[1])
    else
        zero(TW)
    end

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