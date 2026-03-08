module Polynomials

include("Multiindices.jl")
using .Multiindices

include("ArrayAlgebra.jl")
using .ArrayAlgebra

export MultiindexSet, Grlex, Lex, Grevlex, MonomialOrder, find_in_set

export AbstractPolynomial, DensePolynomial,
       polynomial_from_dict, polynomial_from_pairs,
       coeffs, multiindex_set, nvars, all_multiindices_up_to,
       coefficient, has_term, find_term, find_in_multiindex_set,
       zero, evaluate, extract_component, each_term, similar_poly

# ---------- Abstract type ---------- 
# Can be expanded to include other polyonomials types
abstract type AbstractPolynomial{T} end

Base.eltype(::Type{<:AbstractPolynomial{T}}) where T = T
Base.eltype(p::AbstractPolynomial{T}) where T = T

# ---------- DensePolynomial ----------
"""
    DensePolynomial{T,O<:MonomialOrder} <: AbstractPolynomial{T}

Dense representation: coefficients aligned with the full multiindex_set.
- `coeffs::Vector{T}`: length = number of multiindex_set monomials.
- `multiindex_set::MultiindexSet{O}`: reference to the monomial set.
"""
mutable struct DensePolynomial{T,O<:MonomialOrder} <: AbstractPolynomial{T}
    coeffs::Vector{T}
    multiindex_set::MultiindexSet{O}
    function DensePolynomial{T,O}(coeffs::Vector{T}, multiindex_set::MultiindexSet{O}) where {T,O}
        @assert length(coeffs) == size(multiindex_set.exponents,2) "coeffs length must match multiindex_set size"
        new{T,O}(coeffs, multiindex_set)
    end
end

function DensePolynomial(coeffs::Vector{T}, multiindex_set::MultiindexSet{O}) where {T,O}
    DensePolynomial{T,O}(coeffs, multiindex_set)
end

# Construct from dict and order type (creates new multiindex_set)
function DensePolynomial(dict::Dict{Vector{Int}, T}, ::Type{O}) where {T,O<:MonomialOrder}
    isempty(dict) && return DensePolynomial(T[], MultiindexSet(Matrix{Int}(undef, 0, 0), O()))
    exps = collect(keys(dict))
    multiindex_set = MultiindexSet(exps, O())
    coeffs = zeros(T, size(multiindex_set.exponents,2))
    for (j, exp) in enumerate(eachcol(multiindex_set.exponents))
        coeffs[j] = get(dict, exp, zero(T))
    end
    return DensePolynomial(coeffs, multiindex_set)
end

function DensePolynomial(dict::Dict{Vector{Int}, T}, multiindex_set::MultiindexSet{O}) where {T,O}
    n_coeffs = size(multiindex_set.exponents, 2)
    # Verify that all dict keys are present in the set (optional, can be omitted for speed)
    for exp in keys(dict)
        isnothing(find_in_set(multiindex_set, exp)) && 
            error("dict contains exponent $exp not in the given multiindex_set")
    end
    # Build coefficient vector directly from the dictionary, using zero(T) for missing entries
    coeffs = [get(dict, exp, zero(T)) for exp in eachcol(multiindex_set.exponents)]
    return DensePolynomial(coeffs, multiindex_set)
end

# ---------- polynomial_from_dict convenience ----------
polynomial_from_dict(::Type{DensePolynomial}, args...; kwargs...) = DensePolynomial(args...; kwargs...)

# ---------- Accessors ----------
coeffs(p::DensePolynomial) = p.coeffs
multiindex_set(p::AbstractPolynomial) = p.multiindex_set
nvars(p::AbstractPolynomial) = size(multiindex_set(p).exponents, 1)
Base.length(p::DensePolynomial) = length(p.coeffs)

# ---------- Term lookup ----------
"""
    find_in_multiindex_set(p::AbstractPolynomial, exp::Vector{Int}) -> Union{Int,Nothing}

Return the column index of `exp` in the polynomial's multiindex_set, or `nothing` if not present.
"""
find_in_multiindex_set(p::AbstractPolynomial, exp::Vector{Int}) = find_in_set(multiindex_set(p), exp)

"""
    has_term(p::AbstractPolynomial, exp::Vector{Int}) -> Bool

Check whether the polynomial contains a term with exponent `exp`.
"""
has_term(p::DensePolynomial, exp::Vector{Int}) = !isnothing(find_in_multiindex_set(p, exp))

"""
    coefficient(p::AbstractPolynomial, exp::Vector{Int}) -> eltype(p)

Return the coefficient of the term with exponent `exp`, or zero if not present.
"""
function coefficient(p::DensePolynomial, exp::Vector{Int})
    idx = find_in_multiindex_set(p, exp)
    isnothing(idx) && return zero(eltype(p))
    return p.coeffs[idx]
end

"""
    find_term(p::AbstractPolynomial, exp::Vector{Int}) -> Union{Int,Nothing}

Return the position in the polynomial's storage (index in `coeffs`) of the term with exponent `exp`,
or `nothing` if not present.
"""
find_term(p::DensePolynomial, exp::Vector{Int}) = find_in_multiindex_set(p, exp)

# ---------- polynomial_from_pairs (alternative input) ----------
"""
    polynomial_from_pairs(::Type{DensePolynomial{T,O}}, pairs::Vector{Pair{Vector{Int},T}}) where {T,O}

Construct a polynomial from a vector of (exponent => coefficient) pairs.
Useful for building polynomials programmatically.
"""
function polynomial_from_pairs(::Type{DensePolynomial{T,O}}, pairs::Vector{Pair{Vector{Int},T}}) where {T,O<:MonomialOrder}
    dict = Dict(pairs)
    return DensePolynomial(dict, O)
end

import Base: zero

# ---------- Zero polynomial constructors ----------

"""
    zero(::Type{DensePolynomial{T}}, set::MultiindexSet{O}) where {T,O}

Construct a zero polynomial in the given monomial basis `set`.
For dense polynomials, this returns a vector of zeros of length `size(set.exponents,2)`.
"""

function Base.zero(::Type{DensePolynomial{T}}, set::MultiindexSet{O}) where {T,O}
    coeffs = fill(zero(T), size(set.exponents, 2))
    return DensePolynomial(coeffs, set)
end

function Base.zero(::Type{DensePolynomial{T,O}}, set::MultiindexSet{O}) where {T,O}
    coeffs = fill(zero(T), size(set.exponents, 2))
    return DensePolynomial(coeffs, set)
end

Base.iszero(p::DensePolynomial) = all(iszero, p.coeffs)

# ---------- Evaluate polynomial ----------

"""
    evaluate(poly::AbstractPolynomial, vals::Vector{<:Number})

Evaluate a polynomial at the given variable values.
"""
function evaluate(poly::AbstractPolynomial, vals::Vector{<:Number})
    @assert nvars(poly) == length(vals)
    T = eltype(poly)
    result = T == Any ? zero(ComplexF64) : zero(T)
    for (exp, coeff) in each_term(poly)
        term = coeff
        for (j, e) in enumerate(exp)
            term *= vals[j]^e
        end
        result += term
    end
    return result
end

"""
    evaluate(poly::DensePolynomial{T,O}, vals::Vector{<:Number}, idx::Int) where {T,O}

Evaluate the polynomial at `vals` and return the `idx`-th component of each coefficient.
Assumes the polynomial's coefficients are vectors and that `idx` selects one component.
"""
function evaluate(poly::DensePolynomial{NTuple{N,T},O}, vals::Vector{<:Number}, idx::Int) where {N,T,O}
    @assert nvars(poly) == length(vals)
    result = zero(T)
    for (exp, coeff_tuple) in each_term(poly)   # each_term already skips zero coefficients
        coeff = coeff_tuple[idx]
        term = coeff
        for (j, e) in enumerate(exp)
            term *= vals[j]^e
        end
        result += term
    end
    return result
end

"""
    extract_component(poly::DensePolynomial{NTuple{L,T},O}, idx::Int) where {L,T,O}

Return a new polynomial whose coefficients are the `idx`-th component of the original
tuple coefficients.
"""
function extract_component(poly::DensePolynomial{NTuple{L,T},O}, idx::Int) where {L,T,O}
    new_coeffs = [c[idx] for c in poly.coeffs]
    return DensePolynomial(new_coeffs, poly.multiindex_set)
end

# ------------------------------------------------------------
#  Helpers for iterating over terms and constructing similar polynomials
# ------------------------------------------------------------

"""
    each_term(poly::AbstractPolynomial)

Return a generator that yields `(exponent_vector, coefficient)` for every
non‑zero term of `poly`. For dense polynomials, zero coefficients are skipped.
"""
function each_term(poly::DensePolynomial)
    exps = multiindex_set(poly).exponents
    coeffs_vec = coeffs(poly)
    return ((view(exps, :, j), coeffs_vec[j]) for j in 1:size(exps,2) if !iszero(coeffs_vec[j]))
end

"""
    similar_poly(dict::Dict{Vector{Int}, C}, poly::AbstractPolynomial, nvars::Int)

Construct a new polynomial of the same concrete type and monomial order as `poly`
from the dictionary `dict` (exponents → coefficients).  The new polynomial will
have `nvars` variables (the length of exponent vectors in `dict` must match
`nvars`; if `dict` is empty an empty set with the correct number of rows is created).
"""
function similar_poly(dict::Dict{Vector{Int}, C}, poly::DensePolynomial{T,O}, nvars::Int) where {T,C,O}
    # Early return for empty dictionary: create an empty multiindex set and zero coefficients.
    if isempty(dict)
        exponents = Matrix{Int}(undef, nvars, 0)
        mset = MultiindexSet(exponents, O())
        return DensePolynomial{T,O}(T[], mset)
    end

    # Non‑empty dictionary: build the multiindex set from its keys.
    mset = MultiindexSet(collect(keys(dict)), O())

    # Determine a zero value of the correct type for missing coefficients.
    if T <: AbstractArray
        # For array coefficients we need the size. Use the dictionary values to get an example.
        example = first(values(dict))
        zero_val = zero(example)          # zero(::Array) returns an all‑zero array of same size
    else
        zero_val = zero(T)
    end

    # Build the coefficient vector, filling missing entries with zero_val.
    coeffs = [haskey(dict, exp) ? convert(T, dict[exp]) : zero_val for exp in eachcol(mset.exponents)]
    return DensePolynomial{T,O}(coeffs, mset)
end

end # module