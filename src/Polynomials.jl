module Polynomials

export AbstractPolynomial, SparsePolynomial, DensePolynomial, MonomialBasis,
       polynomial_from_dict, coeffs, exponents, nvars

"""
    AbstractPolynomial{T}

Abstract type for multivariate polynomials with coefficient type `T`.
All concrete polynomial types must implement:
- `coeffs(p)`: vector of coefficients
- `exponents(p)`: matrix where each column is an exponent vector
- `nvars(p)`: number of variables
- `Base.length(p)`: number of terms
- `Base.getindex(p, i)`: return `(coeff, exponent_vector)` for term i
"""
abstract type AbstractPolynomial{T} end

Base.eltype(::Type{<:AbstractPolynomial{T}}) where T = T
Base.eltype(p::AbstractPolynomial{T}) where T = T

"""
    SparsePolynomial{C<:AbstractVector} <: AbstractPolynomial{eltype(C)}

Sparse representation of a multivariate polynomial.

# Fields
- `coeffs::C`: vector of coefficients (numeric or array‑valued, e.g. for matrix polynomials)
- `indices::Matrix{Int}`: each column is the exponent vector of a term;
  `indices[:, i]` corresponds to `coeffs[i]`.

The number of rows of `indices` is the number of variables.
"""
struct SparsePolynomial{C<:AbstractVector} <: AbstractPolynomial{eltype(C)}
    coeffs::C
    indices::Matrix{Int}
end

SparsePolynomial(coeffs::C, indices::Matrix{Int}) where C<:AbstractVector = SparsePolynomial{C}(coeffs, indices)

coeffs(p::SparsePolynomial) = p.coeffs
exponents(p::SparsePolynomial) = p.indices
nvars(p::SparsePolynomial) = size(p.indices, 1)
Base.length(p::SparsePolynomial) = length(p.coeffs)
Base.getindex(p::SparsePolynomial, i::Int) = (p.coeffs[i], p.indices[:, i])
Base.setindex!(p::SparsePolynomial, coeff, idx::Int) = (p.coeffs[idx] = coeff)

"""
    MonomialBasis

A fixed set of monomials (exponent vectors) shared among dense polynomials.

# Fields
- `exponents::Matrix{Int}`: matrix where each column is an exponent vector.
  The number of rows is the number of variables; the number of columns is the number of monomials.
"""
struct MonomialBasis
    exponents::Matrix{Int}
    MonomialBasis(exponents::Matrix{Int}) = new(exponents)
end

"""
    DensePolynomial{T} <: AbstractPolynomial{T}

Dense representation of a multivariate polynomial.

# Fields
- `coeffs::Vector{T}`: vector of coefficients aligned with the basis.
- `basis::MonomialBasis`: the common monomial basis defining the monomials.

The `i`-th coefficient corresponds to the `i`-th monomial in `basis`.
"""
struct DensePolynomial{T} <: AbstractPolynomial{T}
    coeffs::Vector{T}
    basis::MonomialBasis
    function DensePolynomial(coeffs::Vector{T}, basis::MonomialBasis) where T
        @assert length(coeffs) == size(basis.exponents, 2) "Number of coefficients must match number of basis monomials"
        new{T}(coeffs, basis)
    end
end

coeffs(p::DensePolynomial) = p.coeffs
exponents(p::DensePolynomial) = p.basis.exponents
nvars(p::DensePolynomial) = size(p.basis.exponents, 1)
Base.length(p::DensePolynomial) = length(p.coeffs)
Base.getindex(p::DensePolynomial, i::Int) = (p.coeffs[i], p.basis.exponents[:, i])

function polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, ::Type{SparsePolynomial}) where T
    if isempty(dict)
        indices = Matrix{Int}(undef, nvars, 0)
        coeffs = Vector{T}()
        return SparsePolynomial(coeffs, indices)
    end
    exps = collect(keys(dict))
    sort!(exps, by=exp -> (length(exp), exp))
    @assert all(length(exp) == nvars for exp in exps) "All exponent vectors must have length $nvars"
    nterms = length(exps)
    indices = Matrix{Int}(undef, nvars, nterms)
    coeffs = Vector{T}(undef, nterms)
    for (j, exp) in enumerate(exps)
        indices[:, j] = exp
        coeffs[j] = dict[exp]
    end
    return SparsePolynomial(coeffs, indices)
end

function polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, ::Type{DensePolynomial}) where T
    if isempty(dict)
        basis = MonomialBasis(Matrix{Int}(undef, nvars, 0))
        return DensePolynomial(Vector{T}(), basis)
    end
    exps = collect(keys(dict))
    sort!(exps, by=exp -> (length(exp), exp))
    @assert all(length(exp) == nvars for exp in exps) "All exponent vectors must have length $nvars"
    nterms = length(exps)
    indices = Matrix{Int}(undef, nvars, nterms)
    for (j, exp) in enumerate(exps)
        indices[:, j] = exp
    end
    basis = MonomialBasis(indices)
    coeffs = [dict[exp] for exp in exps]
    return DensePolynomial(coeffs, basis)
end

"""
    polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, poly::SparsePolynomial) -> SparsePolynomial

Construct a sparse polynomial of the same concrete type (same coefficient container type)
as the given `poly`. Useful when the coefficient container is, e.g., a vector of arrays.
"""
function polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, poly::SparsePolynomial) where T
    C = typeof(poly.coeffs)
    if isempty(dict)
        indices = Matrix{Int}(undef, nvars, 0)
        coeffs = C()
        return SparsePolynomial(coeffs, indices)
    end
    exps = collect(keys(dict))
    sort!(exps, by=exp -> (length(exp), exp))
    @assert all(length(exp) == nvars for exp in exps)
    nterms = length(exps)
    indices = Matrix{Int}(undef, nvars, nterms)
    coeffs = similar(poly.coeffs, T, nterms)
    for (j, exp) in enumerate(exps)
        indices[:, j] = exp
        coeffs[j] = dict[exp]
    end
    return SparsePolynomial(coeffs, indices)
end

"""
    polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, poly::DensePolynomial) -> DensePolynomial

Construct a dense polynomial with the same basis ordering as the one produced from `dict`.
The basis is built from the sorted exponent vectors.
"""
function polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, poly::DensePolynomial) where T
    if isempty(dict)
        basis = MonomialBasis(Matrix{Int}(undef, nvars, 0))
        coeffs = T[]
        return DensePolynomial(coeffs, basis)
    end
    exps = collect(keys(dict))
    sort!(exps, by=exp -> (length(exp), exp))
    @assert all(length(exp) == nvars for exp in exps)
    nterms = length(exps)
    indices = Matrix{Int}(undef, nvars, nterms)
    for (j, exp) in enumerate(exps)
        indices[:, j] = exp
    end
    basis = MonomialBasis(indices)
    coeffs = [dict[exp] for exp in exps]
    return DensePolynomial(coeffs, basis)
end

end
