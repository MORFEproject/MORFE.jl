module Polynomials

include("Multiindices.jl")
using .Multiindices

export MultiindexSet, Grlex, Lex, Grevlex, MonomialOrder, find_in_set

export AbstractPolynomial, SparsePolynomial, DensePolynomial,
       polynomial_from_dict, polynomial_from_pairs,
       coeffs, indices, multiindex_set, nvars,
       coefficient, has_term, find_term, find_in_multiindex_set,
       convert_dense_to_sparse, convert_sparse_to_dense

# ---------- Abstract type ----------
abstract type AbstractPolynomial{T} end

Base.eltype(::Type{<:AbstractPolynomial{T}}) where T = T
Base.eltype(p::AbstractPolynomial{T}) where T = T

# ---------- SparsePolynomial ----------
"""
    SparsePolynomial{T,O<:MonomialOrder} <: AbstractPolynomial{T}

Sparse representation using a shared monomial multiindex_set.
- `coeffs::Vector{T}`: non‑zero coefficients.
- `indices::Vector{Int}`: corresponding column indices in `multiindex_set`.
- `multiindex_set::MultiindexSet{O}`: reference to the monomial set.
"""
struct SparsePolynomial{T,O<:MonomialOrder} <: AbstractPolynomial{T}
    coeffs::Vector{T}
    indices::Vector{Int}
    multiindex_set::MultiindexSet{O}
    function SparsePolynomial{T,O}(coeffs::Vector{T}, indices::Vector{Int}, multiindex_set::MultiindexSet{O}) where {T,O}
        @assert length(coeffs) == length(indices) "coeffs and indices must have same length"
        @assert all(1 ≤ i ≤ size(multiindex_set.exponents,2) for i in indices) "indices out of multiindex_set range"
        new{T,O}(coeffs, indices, multiindex_set)
    end
end

# Type‑stable constructor (infers O from multiindex_set)
function SparsePolynomial(coeffs::Vector{T}, indices::Vector{Int}, multiindex_set::MultiindexSet{O}) where {T,O}
    SparsePolynomial{T,O}(coeffs, indices, multiindex_set)
end

# Construct from dict and order type (creates new multiindex_set)
function SparsePolynomial(dict::Dict{Vector{Int}, T}, ::Type{O}) where {T,O<:MonomialOrder}
    isempty(dict) && return SparsePolynomial(T[], Int[], MultiindexSet(Matrix{Int}(undef, 0, 0), O()))
    exps = collect(keys(dict))
    multiindex_set = MultiindexSet(exps, O())
    idxs = [find_in_set(multiindex_set, exp) for exp in exps]  # guaranteed to exist
    coeffs = [dict[exp] for exp in exps]
    # Sort by multiindex_set order (already sorted, but we need to align coeffs with indices)
    perm = sortperm(idxs)
    return SparsePolynomial(coeffs[perm], idxs[perm], multiindex_set)
end

# Construct from dict and existing multiindex_set (multiindex_set must contain all exponents)
function SparsePolynomial(dict::Dict{Vector{Int}, T}, multiindex_set::MultiindexSet{O}) where {T,O}
    exps = collect(keys(dict))
    idxs = [find_in_set(multiindex_set, exp) for exp in exps]
    any(isnothing, idxs) && error("dict contains exponents not in the given multiindex_set")
    idxs = Int.(idxs)  # convert from Union{Int,Nothing} to Int
    coeffs = [dict[exp] for exp in exps]
    perm = sortperm(idxs)
    return SparsePolynomial(coeffs[perm], idxs[perm], multiindex_set)
end

# ---------- DensePolynomial ----------
"""
    DensePolynomial{T,O<:MonomialOrder} <: AbstractPolynomial{T}

Dense representation: coefficients aligned with the full multiindex_set.
- `coeffs::Vector{T}`: length = number of multiindex_set monomials.
- `multiindex_set::MultiindexSet{O}`: reference to the monomial set.
"""
struct DensePolynomial{T,O<:MonomialOrder} <: AbstractPolynomial{T}
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

# Construct from dict and existing multiindex_set
function DensePolynomial(dict::Dict{Vector{Int}, T}, multiindex_set::MultiindexSet{O}) where {T,O}
    n_coeffs = size(multiindex_set.exponents, 2)
    # Verify that all dict keys are present in the set
    for exp in keys(dict)
        isnothing(find_in_set(multiindex_set, exp)) && 
            error("dict contains exponent $exp not in the given multiindex_set")
    end
    if isempty(dict)
        # If the dictionary is empty, we cannot obtain a sample value.
        # For scalar T, zeros(T, n_coeffs) works; for vector T it will error.
        # This case is rarely used with non‑scalar T; if needed, a zero argument should be added.
        coeffs = zeros(T, n_coeffs)
    else
        sample_val = first(values(dict))
        zero_elem = zero(sample_val)          # e.g. [0.0, 0.0] for a vector
        coeffs = [zero_elem for _ in 1:n_coeffs]   # independent copies
        for (j, exp) in enumerate(eachcol(multiindex_set.exponents))
            coeffs[j] = get(dict, exp, zero_elem)
        end
    end
    return DensePolynomial(coeffs, multiindex_set)
end

# ---------- polynomial_from_dict convenience ----------
polynomial_from_dict(::Type{DensePolynomial}, args...; kwargs...) = DensePolynomial(args...; kwargs...)
polynomial_from_dict(::Type{SparsePolynomial}, args...; kwargs...) = SparsePolynomial(args...; kwargs...)

# ---------- Accessors ----------
coeffs(p::SparsePolynomial) = p.coeffs
coeffs(p::DensePolynomial) = p.coeffs
indices(p::SparsePolynomial) = p.indices
multiindex_set(p::AbstractPolynomial) = p.multiindex_set
nvars(p::AbstractPolynomial) = size(multiindex_set(p).exponents, 1)
Base.length(p::DensePolynomial) = length(p.coeffs)
Base.length(p::SparsePolynomial) = length(p.coeffs)

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
function has_term(p::SparsePolynomial, exp::Vector{Int})
    idx = find_in_multiindex_set(p, exp)
    isnothing(idx) && return false
    return idx in p.indices
end

"""
    coefficient(p::AbstractPolynomial, exp::Vector{Int}) -> eltype(p)

Return the coefficient of the term with exponent `exp`, or zero if not present.
"""
function coefficient(p::DensePolynomial, exp::Vector{Int})
    idx = find_in_multiindex_set(p, exp)
    isnothing(idx) && return zero(eltype(p))
    return p.coeffs[idx]
end
function coefficient(p::SparsePolynomial, exp::Vector{Int})
    idx = find_in_multiindex_set(p, exp)
    isnothing(idx) && return zero(eltype(p))
    pos = findfirst(==(idx), p.indices)
    isnothing(pos) && return zero(eltype(p))
    return p.coeffs[pos]
end

"""
    find_term(p::AbstractPolynomial, exp::Vector{Int}) -> Union{Int,Nothing}

Return the position in the polynomial's storage (index in `coeffs`) of the term with exponent `exp`,
or `nothing` if not present.
"""
find_term(p::DensePolynomial, exp::Vector{Int}) = find_in_multiindex_set(p, exp)
function find_term(p::SparsePolynomial, exp::Vector{Int})
    idx = find_in_multiindex_set(p, exp)
    isnothing(idx) && return nothing
    return findfirst(==(idx), p.indices)
end

# ---------- Conversion between sparse and dense ----------
"""
    convert_sparse_to_dense(p::SparsePolynomial{T,O}) -> DensePolynomial{T,O}

Convert a sparse polynomial to a dense polynomial sharing the same multiindex_set.
"""
function convert_sparse_to_dense(p::SparsePolynomial{T,O}) where {T,O}
    dense_coeffs = zeros(T, size(multiindex_set(p).exponents,2))
    for (c, i) in zip(p.coeffs, p.indices)
        dense_coeffs[i] = c
    end
    return DensePolynomial(dense_coeffs, p.multiindex_set)
end

"""
    convert_dense_to_sparse(p::DensePolynomial{T,O}; tol=0) -> SparsePolynomial{T,O}

Convert a dense polynomial to sparse, dropping coefficients with absolute value ≤ `tol`.
"""
function convert_dense_to_sparse(p::DensePolynomial{T,O}; tol=0) where {T,O}
    sparse_coeffs = T[]
    sparse_idxs = Int[]
    for (i, c) in enumerate(p.coeffs)
        if abs(c) > tol
            push!(sparse_coeffs, c)
            push!(sparse_idxs, i)
        end
    end
    return SparsePolynomial(sparse_coeffs, sparse_idxs, p.multiindex_set)
end

# ---------- polynomial_from_pairs (alternative input) ----------
"""
    polynomial_from_pairs(::Type{DensePolynomial{T,O}}, pairs::Vector{Pair{Vector{Int},T}}) where {T,O}
    polynomial_from_pairs(::Type{SparsePolynomial{T,O}}, pairs::Vector{Pair{Vector{Int},T}}) where {T,O}

Construct a polynomial from a vector of (exponent => coefficient) pairs.
Useful for building polynomials programmatically.
"""
function polynomial_from_pairs(::Type{DensePolynomial{T,O}}, pairs::Vector{Pair{Vector{Int},T}}) where {T,O<:MonomialOrder}
    dict = Dict(pairs)
    return DensePolynomial(dict, O)
end

function polynomial_from_pairs(::Type{SparsePolynomial{T,O}}, pairs::Vector{Pair{Vector{Int},T}}) where {T,O<:MonomialOrder}
    dict = Dict(pairs)
    return SparsePolynomial(dict, O)
end

import Base: zero

# ---------- Zero polynomial constructors ----------

"""
    zero(p::DensePolynomial)

Return a zero polynomial of the same concrete type, element type, order,
and number of variables as `p`, but with an empty multiindex set
(i.e., no terms). For a zero polynomial in the same monomial basis as `p`,
use `zero(typeof(p), multiindex_set(p))`.
"""
function zero(p::DensePolynomial{T,O}) where {T,O}
    n = nvars(p)
    empty_set = MultiindexSet(zeros(Int, n, 0), O())
    return DensePolynomial(T[], empty_set)
end

"""
    zero(p::SparsePolynomial)

Return a zero polynomial of the same concrete type, element type, order,
and number of variables as `p`, but with an empty multiindex set.
"""
function zero(p::SparsePolynomial{T,O}) where {T,O}
    n = nvars(p)
    empty_set = MultiindexSet(zeros(Int, n, 0), O())
    return SparsePolynomial(T[], Int[], empty_set)
end

"""
    zero(::Type{DensePolynomial{T,O}}, nvars::Int) where {T,O}
    zero(::Type{SparsePolynomial{T,O}}, nvars::Int) where {T,O}

Construct a zero polynomial of the given type with `nvars` variables
and an empty multiindex set.
"""
function zero(::Type{DensePolynomial{T,O}}, nvars::Int) where {T,O}
    empty_set = MultiindexSet(zeros(Int, nvars, 0), O())
    return DensePolynomial(T[], empty_set)
end

function zero(::Type{SparsePolynomial{T,O}}, nvars::Int) where {T,O}
    empty_set = MultiindexSet(zeros(Int, nvars, 0), O())
    return SparsePolynomial(T[], Int[], empty_set)
end

"""
    zero(::Type{DensePolynomial{T,O}}, set::MultiindexSet{O}) where {T,O}
    zero(::Type{SparsePolynomial{T,O}}, set::MultiindexSet{O}) where {T,O}

Construct a zero polynomial in the given monomial basis `set`.
For dense polynomials, this returns a vector of zeros of length `size(set.exponents,2)`.
For sparse polynomials, it returns an empty polynomial (no terms) because the
sparse representation omits zero coefficients.
"""
function zero(::Type{DensePolynomial{T,O}}, set::MultiindexSet{O}) where {T,O}
    return DensePolynomial(zeros(T, size(set.exponents,2)), set)
end

function zero(::Type{SparsePolynomial{T,O}}, set::MultiindexSet{O}) where {T,O}
    return SparsePolynomial(T[], Int[], set)
end

end # module