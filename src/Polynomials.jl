module Polynomials

using LinearAlgebra

using ..Multiindices

export AbstractPolynomial, DensePolynomial,
       polynomial_from_dict, polynomial_from_pairs,
       coeffs, multiindex_set, nvars, all_multiindices_up_to,
       coefficient, has_term, find_term, find_in_multiindex_set,
       zero, evaluate, extract_component, each_term, similar_poly

# ---------- Abstract type ----------
abstract type AbstractPolynomial{T} end

Base.eltype(::Type{<:AbstractPolynomial{T}}) where T = T
Base.eltype(p::AbstractPolynomial{T}) where T = T

# ---------- DensePolynomial ----------
"""
    DensePolynomial{T} <: AbstractPolynomial{T}

Dense representation: coefficients aligned with the full multiindex_set.
- `coeffs::Vector{T}`: length = number of multiindex_set monomials.
- `multiindex_set::MultiindexSet`: reference to the monomial set (always Grlex‑ordered).

For scalar polynomials, `T <: Number`. For vector‑valued polynomials,
`T <: AbstractVector` (e.g., `Vector{Float64}`).
"""
mutable struct DensePolynomial{T} <: AbstractPolynomial{T}
    coeffs::Vector{T}
    multiindex_set::MultiindexSet
    function DensePolynomial{T}(coeffs::Vector{T}, multiindex_set::MultiindexSet) where T
        @assert length(coeffs) == size(multiindex_set.exponents, 2) "coeffs length must match multiindex_set size"
        new{T}(coeffs, multiindex_set)
    end
end

# Convenience constructors
DensePolynomial(coeffs::Vector{T}, mset::MultiindexSet) where T = DensePolynomial{T}(coeffs, mset)

"""
    DensePolynomial(dict::Dict{Vector{Int}, T}) where T

Construct a polynomial from a dictionary mapping exponent vectors to coefficients.
The multiindex set is built from the keys and sorted in Grlex order.
"""
function DensePolynomial(dict::Dict{Vector{Int}, T}) where T
    isempty(dict) && return DensePolynomial(T[], MultiindexSet(Matrix{Int}(undef, 0, 0)))
    exps = collect(keys(dict))
    mset = MultiindexSet(exps)          # automatically sorted in Grlex
    coeffs = [get(dict, exp, zero(T)) for exp in eachcol(mset.exponents)]
    return DensePolynomial(coeffs, mset)
end

# ---------- Accessors ----------
coeffs(p::DensePolynomial) = p.coeffs
multiindex_set(p::AbstractPolynomial) = p.multiindex_set
nvars(p::AbstractPolynomial) = size(multiindex_set(p).exponents, 1)
Base.length(p::DensePolynomial) = length(p.coeffs)

# ---------- Term lookup (using AbstractVector{Int}) ----------
"""
    find_in_multiindex_set(p::AbstractPolynomial, exp::AbstractVector{Int}) -> Union{Int,Nothing}

Return the column index of `exp` in the polynomial's multiindex_set, or `nothing` if not present.
"""
find_in_multiindex_set(p::AbstractPolynomial, exp::AbstractVector{Int}) = find_in_set(multiindex_set(p), exp)

"""
    has_term(p::DensePolynomial, exp::AbstractVector{Int}) -> Bool

Check whether the polynomial contains a term with exponent `exp`.
"""
has_term(p::DensePolynomial, exp::AbstractVector{Int}) = !isnothing(find_in_multiindex_set(p, exp))

"""
    coefficient(p::DensePolynomial, exp::AbstractVector{Int}) -> eltype(p)

Return the coefficient of the term with exponent `exp`, or zero if not present.
"""
function coefficient(p::DensePolynomial, exp::AbstractVector{Int})
    idx = find_in_multiindex_set(p, exp)
    isnothing(idx) && return zero(eltype(p))
    return p.coeffs[idx]
end

"""
    find_term(p::DensePolynomial, exp::AbstractVector{Int}) -> Union{Int,Nothing}

Return the position in the polynomial's storage (index in `coeffs`) of the term with exponent `exp`,
or `nothing` if not present.
"""
find_term(p::DensePolynomial, exp::AbstractVector{Int}) = find_in_multiindex_set(p, exp)

# ---------- polynomial_from_pairs (alternative input) ----------
"""
    polynomial_from_pairs(::Type{DensePolynomial{T}}, pairs::Vector{Pair{Vector{Int},T}}) where T

Construct a polynomial from a vector of (exponent => coefficient) pairs.
Useful for building polynomials programmatically.
"""
function polynomial_from_pairs(::Type{DensePolynomial{T}}, pairs::Vector{Pair{Vector{Int},T}}) where T
    dict = Dict(pairs)
    return DensePolynomial(dict)
end

# Convenience method using default type
polynomial_from_pairs(pairs::Vector{Pair{Vector{Int},T}}) where T = DensePolynomial(Dict(pairs))

import Base: zero

# ---------- Zero polynomial constructors ----------
"""
    zero(::Type{DensePolynomial{T}}, set::MultiindexSet) where T<:Number

Construct a zero scalar polynomial in the given monomial basis `set`.
"""
function Base.zero(::Type{DensePolynomial{T}}, set::MultiindexSet) where T<:Number
    coeffs = zeros(T, size(set.exponents, 2))
    return DensePolynomial(coeffs, set)
end

"""
    zero(::Type{DensePolynomial{Vector{T}}}, set::MultiindexSet, n_components::Int) where T

Construct a zero vector‑valued polynomial with `n_components` components.
Each coefficient is a zero vector of length `n_components`.
"""
function Base.zero(::Type{DensePolynomial{Vector{T}}}, set::MultiindexSet, n_components::Int) where T
    coeffs = [zeros(T, n_components) for _ in 1:size(set.exponents, 2)]
    return DensePolynomial{Vector{T}}(coeffs, set)
end

"""
    zero(::Type{DensePolynomial{NTuple{N,T}}}, set::MultiindexSet) where {N,T}

Construct a zero tuple‑valued polynomial with `N` components.
Each coefficient is a zero tuple of length `N`.
"""
function Base.zero(::Type{DensePolynomial{NTuple{N,T}}}, set::MultiindexSet) where {N,T}
    coeffs = [ntuple(__ -> zero(T), N) for _ in 1:size(set.exponents, 2)]
    println("coeffs =\n", typeof(coeffs))
    return DensePolynomial{NTuple{N,T}}(coeffs, set)
end

Base.iszero(p::DensePolynomial) = all(iszero, p.coeffs)

# ---------- Evaluate polynomial ----------
"""
    evaluate(poly::DensePolynomial{<:Number}, vals::Vector{<:Number})

Evaluate a scalar polynomial at the given variable values.
"""
function evaluate(poly::DensePolynomial{<:Number}, vals::Vector{<:Number})
    @assert nvars(poly) == length(vals) "Number of variables mismatch"
    T = eltype(poly)
    result = zero(T)
    for (exp, coeff) in each_term(poly)
        monom = one(eltype(vals))
        for (j, e) in enumerate(exp)
            monom *= vals[j]^e
        end
        result += coeff * monom
    end
    return result
end

"""
    evaluate(poly::DensePolynomial{Vector{T}}, vals::Vector{<:Number}) where T

Evaluate a vector‑valued polynomial at the given variable values.
Returns a vector of length equal to the number of components.
"""
function evaluate(poly::DensePolynomial{Vector{T}}, vals::Vector{<:Number}) where T
    @assert nvars(poly) == length(vals) "Number of variables mismatch"
    # Determine number of components from first non‑zero coefficient (if any)
    n_comp = 0
    for coeff in poly.coeffs
        if !iszero(coeff)
            n_comp = length(coeff)
            break
        end
    end
    if n_comp == 0
        # All coefficients are zero → result is zero vector of unknown length.
        # Return empty vector; user can check with iszero(poly) first.
        return T[]
    end
    result = zeros(T, n_comp)
    for (exp, coeff_vec) in each_term(poly)
        monom = one(eltype(vals))
        for (j, e) in enumerate(exp)
            monom *= vals[j]^e
        end
        # Use broadcasting for efficiency; this is a BLAS level‑1 operation.
        result .+= coeff_vec .* monom
    end
    return result
end

"""
    extract_component(poly::DensePolynomial{Vector{T}}, idx::Int) where T

Return a new scalar polynomial whose coefficients are the `idx`-th component of the original
vector coefficients.
"""
function extract_component(poly::DensePolynomial{Vector{T}}, idx::Int) where T
    new_coeffs = [c[idx] for c in poly.coeffs]
    return DensePolynomial(new_coeffs, poly.multiindex_set)
end

# ---------- Iteration over terms ----------
"""
    each_term(poly::AbstractPolynomial)

Return a generator that yields `(exponent_vector, coefficient)` for every
non‑zero term of `poly`. For dense polynomials, zero coefficients are skipped.
"""
function each_term(poly::DensePolynomial)
    exps = multiindex_set(poly).exponents
    coeffs_vec = coeffs(poly)
    return ((view(exps, :, j), coeffs_vec[j]) for j in 1:size(exps, 2) if !iszero(coeffs_vec[j]))
end

# ---------- similar_poly (construct polynomial of same type from dictionary) ----------
"""
    similar_poly(dict::Dict{Vector{Int}, C}, poly::DensePolynomial{T}, nvars::Int) where {T,C}

Construct a new polynomial of the same concrete type as `poly` from the dictionary
`dict` (exponents → coefficients). The new polynomial will have `nvars` variables.
For vector‑valued polynomials, the dictionary values must have the same length as the
original polynomial's coefficients. If `dict` is empty, a zero polynomial with zero
coefficients is returned; for vector polynomials the component dimension must be
inferable from `poly` (i.e., `poly` must be non‑empty or you must use the specialized
`zero` constructor).
"""
function similar_poly(dict::Dict{Vector{Int}, C}, poly::DensePolynomial{T}, nvars::Int) where {T<:Number,C<:Number}
    if isempty(dict)
        exponents = Matrix{Int}(undef, nvars, 0)
        mset = MultiindexSet(exponents)
        return DensePolynomial{T}(T[], mset)
    end
    mset = MultiindexSet(collect(keys(dict)))
    coeffs = [haskey(dict, exp) ? convert(T, dict[exp]) : zero(T) for exp in eachcol(mset.exponents)]
    return DensePolynomial{T}(coeffs, mset)
end

function similar_poly(dict::Dict{Vector{Int}, C}, poly::DensePolynomial{Vector{T}}, nvars::Int) where {T,C<:AbstractVector}
    if isempty(dict)
        # Cannot infer component dimension; require poly to be non‑empty or use zero constructor.
        if isempty(poly.coeffs)
            error("Cannot create empty vector polynomial without component dimension. " *
                  "Use zero(DensePolynomial{Vector{T}}, set, n_components).")
        end
        # Infer component dimension from poly's coefficients.
        n_comp = length(poly.coeffs[1])
        exponents = Matrix{Int}(undef, nvars, 0)
        mset = MultiindexSet(exponents)
        return DensePolynomial{Vector{T}}([zeros(T, n_comp) for _ in 1:0], mset)
    end
    mset = MultiindexSet(collect(keys(dict)))
    n_comp = length(first(values(dict)))
    zero_vec = zeros(T, n_comp)
    coeffs = [haskey(dict, exp) ? convert(Vector{T}, dict[exp]) : copy(zero_vec) for exp in eachcol(mset.exponents)]
    return DensePolynomial{Vector{T}}(coeffs, mset)
end

# ---------- Basic arithmetic (optional, but useful) ----------
# These are not strictly required by the original API, but they improve usability.

import Base: +, -, *, ==

# Scalar multiplication (both orders)
*(s::Number, p::DensePolynomial{<:Number}) = DensePolynomial(s .* p.coeffs, p.multiindex_set)
*(p::DensePolynomial{<:Number}, s::Number) = s * p

# For vector polynomials, scalar multiplication is elementwise
*(s::Number, p::DensePolynomial{Vector{T}}) where T = DensePolynomial([s * c for c in p.coeffs], p.multiindex_set)
*(p::DensePolynomial{Vector{T}}, s::Number) where T = s * p

# Addition (same multiindex set)
function +(p1::DensePolynomial{T}, p2::DensePolynomial{T}) where T
    @assert p1.multiindex_set == p2.multiindex_set "Cannot add polynomials with different multiindex sets"
    DensePolynomial(p1.coeffs .+ p2.coeffs, p1.multiindex_set)
end

# Subtraction
-(p1::DensePolynomial{T}, p2::DensePolynomial{T}) where T = p1 + (-1)*p2

# Equality (requires same multiindex set)
function ==(p1::DensePolynomial, p2::DensePolynomial)
    p1.multiindex_set == p2.multiindex_set && p1.coeffs == p2.coeffs
end

end # module