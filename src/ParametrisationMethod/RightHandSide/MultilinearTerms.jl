module MultilinearTerms

using ..Multiindices: indices_in_box_with_bounded_degree,
    factorisations_asymmetric, factorisations_fully_symmetric, factorisations_groupwise_symmetric

using ..ParametrisationMethod: Parametrisation

using ..FullOrderModel: NDOrderModel, FirstOrderModel, MultilinearMap

# ----------------------------------------------------------------------
# Internal helper
# ----------------------------------------------------------------------

"""
    _derivative_orders(multilinear_term)

Return a vector of length `multilinear_term.deg` where each entry is the derivative order
(1‑based) that the corresponding factor position belongs to, according to the
multiplicities in `multilinear_term.multiindex`.

# Example
If `multilinear_term.multiindex = (2,1,1)`, total degree = 4, the function returns `[1,1,2,3]`.
The first two factors come from the 0th derivative (x), the third from the
1st derivative (x') and the last from the 2nd derivative (x'').
"""
function _derivative_orders(multilinear_term::MultilinearMap)
    orders = Int[]
    for (deriv_idx, cnt) in enumerate(multilinear_term.multiindex)
        for _ in 1:cnt
            push!(orders, deriv_idx)
        end
    end
    return orders
end

# ----------------------------------------------------------------------
# Accumulation functions (specialised by symmetry type)
# ----------------------------------------------------------------------

"""
    accumulate_asymmetric!(result, multilinear_term, orders, exp, parametrisation)

Accumulate contributions for a fully asymmetric multilinear_term 
(no multiplicities in `multilinear_term.multiindex` are larger than 1, e.g. `(0,1,1,0)`).  
For each ordered factorisation `idx_tuple` obtained from
`factorisations_asymmetric`, evaluate `multilinear_term.f!` with arguments taken from the
appropriate coefficients in `parametrisation` and add the result directly to `result`.

# Arguments
- `result`: array to accumulate into (modified in place).
- `multilinear_term`: a nonlinear multilinear_term with fields `deg`, `multiindex`, `f!`.
- `orders`: mapping from factor position → derivative order (from `_derivative_orders`).
- `exp`: exponent vector.
- `parametrisation`: tuple of matrices; `parametrisation[d]` is the matrix for derivative order `d-1` (d=1 for x, d=2 for x', …).
"""
function accumulate_asymmetric!(result, multilinear_term::MultilinearMap{ORD}, orders::AbstractVector{Int}, 
        exp::AbstractVector{Int}, parametrisation::Parametrisation{ORD}, candidate_indices::AbstractVector{Int}) where ORD

    W = parametrisation.coeffs
    k = multilinear_term.deg
    factorizations = factorisations_asymmetric(parametrisation.multiindex_set, exp, k, candidate_indices)
    for idx_tuple in factorizations
        @inbounds args = ntuple(i -> W[idx_tuple[i]][orders[i]], k)
        multilinear_term.f!(result, args...)   # direct accumulation (no symmetry factor needed)
    end
end

"""
    accumulate_symmetric!(result, multilinear_term, exp, parametrisation)

Accumulate contributions for a fully symmetric multilinear_term (only one positive entry in
`multilinear_term.multiindex`, e.g. `(0,3,0)`).  For each factorisation `(idx_tuple, count)` from
`factorisations_fully_symmetric`, evaluate `multilinear_term.f!` into a temporary array, scale by
`count`, and add to `result`.

The factor `count` accounts for the number of permutations that yield the same
ordered tuple due to symmetry inside the single group.
"""
function accumulate_symmetric!(result, multilinear_term::MultilinearMap{ORD}, 
        exp::AbstractVector{Int}, parametrisation::Parametrisation{ORD}, candidate_indices::AbstractVector{Int}) where ORD

    W = parametrisation.coeffs
    k = multilinear_term.deg
    # derivative order = position of the only positive entry in multiindex
    deriv_idx = findfirst(>(0), multilinear_term.multiindex)
    factorizations = factorisations_fully_symmetric(parametrisation.multiindex_set, exp, k, candidate_indices)
    temp = similar(result)          # reused for all factorisations
    for (idx_tuple, count) in factorizations
        fill!(temp, 0)
        @inbounds args = ntuple(i -> W[idx_tuple[i]][deriv_idx], k)
        multilinear_term.f!(temp, args...)
        @inbounds result .+= count .* temp
    end
end

# for first order systems
function accumulate_symmetric!(result, multilinear_term::MultilinearMap{1}, exp::AbstractVector{Int}, 
        parametrisation::Parametrisation{1}, candidate_indices::AbstractVector{Int})

    W = parametrisation.coeffs
    k = multilinear_term.deg
    temp = similar(result)          # reused for all factorisations
    for (idx_tuple, count) in factorisations_fully_symmetric(parametrisation.multiindex_set, exp, k, candidate_indices)
        fill!(temp, 0)
        @inbounds args = ntuple(i -> W[idx_tuple[i]], k)
        multilinear_term.f!(temp, args...)
        @inbounds result .+= count .* temp
    end
end

"""
    accumulate_partial!(result, multilinear_term, orders, exp, parametrisation)

Accumulate contributions for a partially symmetric multilinear_term (multiple positive entries
in `multilinear_term.multiindex`, e.g. `(2,1)`).  For each factorisation `(flat_indices, total_count)`
from `factorisations_groupwise_symmetric`, evaluate `multilinear_term.f!` into a temporary array,
scale by `total_count`, and add to `result`.

The group sizes are extracted from the positive entries of `multilinear_term.multiindex`.
"""
function accumulate_partial!(result, multilinear_term::MultilinearMap{ORD}, orders::AbstractVector{Int}, 
        exp::AbstractVector{Int}, parametrisation::Parametrisation{ORD}, candidate_indices::AbstractVector{Int}) where ORD

    W = parametrisation.coeffs
    k = multilinear_term.deg
    factorizations = factorisations_groupwise_symmetric(parametrisation.multiindex_set, exp, multilinear_term.multiindex, candidate_indices)
    temp = similar(result)                      # reused for all factorisations
    for (flat_indices, total_count) in factorizations
        fill!(temp, 0)
        @inbounds args = ntuple(i -> W[flat_indices[i]][orders[i]], k)
        multilinear_term.f!(temp, args...)
        @inbounds result .+= total_count .* temp
    end
end

# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

"""
    accumulate_multilinear_term!(result, multilinear_term, exp, parametrisation, multiindex_set, candidate_indices)

Demultilinear_termine the symmetry type of `multilinear_term` based on its `multiindex` and call the
appropriate accumulation function.

The symmetry type is decided by counting how many distinct derivative orders appear:
- All entries are less than 2 → fully asymmetric.
- If exactly one positive entry → fully symmetric.
- Otherwise → partially symmetric.
"""
function accumulate_multilinear_term!(result, multilinear_term::MultilinearMap{ORD}, 
        exp::AbstractVector{Int}, parametrisation::Parametrisation{ORD}) where {ORD}

    candidate_indices = indices_in_box_with_bounded_degree(parametrisation.multiindex_set, exp, 1, sum(exp))

    nz = count(>(0), multilinear_term.multiindex)
    if nz == multilinear_term.deg               # fully asymmetric: all non‑zero entries are 1
        orders = _derivative_orders(multilinear_term)
        accumulate_asymmetric!(result, multilinear_term, orders, exp, parametrisation, candidate_indices)
    elseif nz == 1                   # fully symmetric: only one derivative order appears
        accumulate_symmetric!(result, multilinear_term, exp, parametrisation, candidate_indices)
    else                              # partially symmetric
        orders = _derivative_orders(multilinear_term)
        accumulate_partial!(result, multilinear_term, orders, exp, parametrisation, candidate_indices)
    end
end

# ----------------------------------------------------------------------
# Top‑level routine
# ----------------------------------------------------------------------

"""
    compute_multilinear_terms(model::NDOrderModel{N}, exp::AbstractVector{Int}, parametrisation::NTuple{N, Parametrisation{ORD}}) where ORD

Sum the contributions of all nonlinear multilinear_terms in `model` and return the accumulated
array.

# Arguments
- `model`: an object with a field `nonlinear_terms` (a list of multilinear_terms).
- `exp`: exponent vector.
- `parametrisation`: a tuple of dense polynomials; `parametrisation[d]` corresponds to derivative order `d-1` (d=1 for x, d=2 for x', …).

# Returns
An array of the same size and element type as `parametrisation[1]` containing the total sum.
"""
function compute_multilinear_terms(model::NDOrderModel{ORD}, exp::AbstractVector{Int}, parametrisation::Parametrisation{ORD}) where ORD

    # Use the first coefficient to termine size and element type
    first_coeff = parametrisation.coeffs[1][1]
    result = zeros(eltype(first_coeff), size(first_coeff))
    for multilinear_term in model.nonlinear_terms
        accumulate_multilinear_term!(result, multilinear_term, exp, parametrisation)
    end
    return result
end

# Compute multilinear terms for first order systems
function compute_multilinear_terms(model::FirstOrderModel, exp::AbstractVector{Int}, parametrisation::Parametrisation{1})

    candidate_indices = indices_in_box_with_bounded_degree(parametrisation.multiindex_set, exp, 1, sum(exp))

    # Use the first coefficient to termine size and element type
    first_coeff = parametrisation.coeffs[1]
    result = zeros(eltype(first_coeff), size(first_coeff))
    for multilinear_term in model.nonlinear_terms
        accumulate_symmetric!(result, multilinear_term, exp, parametrisation, candidate_indices)
    end
    return result
end

end # module