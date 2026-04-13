module Polynomials

# ============================================================
#  Design overview
# ============================================================
#
#  NEW layout   coefficients::Array{T, N}   (last axis = monomials)
#               ┌─────────────────────────┐
#               │ c[1,1] c[1,2] … c[1,L] │  row 1 (output dim 1)
#               │ c[2,1] c[2,2] … c[2,L] │  row 2 (output dim 2)
#               │   …                    │
#               └─────────────────────────┘
#                ↑ one single contiguous block in RAM
#
#  Scalar polynomial:  N=1, shape (L,)           → dot product
#  Vector polynomial:  N=2, shape (K, L)         → BLAS gemv  (K×L) * (L,)
#  Tensor polynomial:  N=k+1, shape (d1,…dk, L)  → reshape + gemv
#
#  All coefficient shapes share this convention, so `evaluate` always
#  reduces to LinearAlgebra.mul! / dot, getting BLAS-level throughput.
#
#  The coefficient array field is typed as `AbstractArray{T,N}` so that
#  a memory-mapped array (Mmap.jl) can be used transparently for
#  datasets that exceed available RAM (see `mmap_polynomial`).
# ============================================================

using LinearAlgebra
using Mmap
using StaticArrays: SVector
using Base.Threads: @threads
using ..Multiindices: MultiindexSet, find_in_set, zero_multiindex

export DensePolynomial,
	polynomial_from_pairs,
	coefficients, multiindex_set, nvars, nmonomials, coeff_shape,
	coefficient, has_term, find_term, find_in_multiindex_set,
	evaluate, extract_component, each_term, similar_poly,
	linear_matrix_of_polynomial,
	mmap_polynomial

# ─────────────────────────────────────────────────────────────────────────────
# Struct
# ─────────────────────────────────────────────────────────────────────────────

"""
	DensePolynomial{T, NVAR, N, A}

Cache-friendly dense polynomial with a single contiguous coefficient array.

## Type parameters
| param | meaning |
|-------|---------|
| `T`   | Scalar element type (`Float64`, `ComplexF64`, …). Must be a concrete bits type for best performance. |
| `NVAR`| Number of input variables. |
| `N`   | `ndims(coefficients)`. Number of axes: `N = 1` for scalar, `N = 2` for vector-valued, etc. |
| `A`   | Concrete array type (`Array{T,N}` normally; can be an `Mmap` array). |

## Fields
- `coefficients::A` — shape `(d1, …, d_{N-1}, L)`.  The **last** axis
  indexes the `L` monomials in `multiindex_set`.  Leading axes describe
  the coefficient shape (none for scalar, `(K,)` for a K-vector, etc.).
- `multiindex_set::MultiindexSet{NVAR}` — Grlex-ordered monomial basis.
- `max_exponents::SVector{NVAR,Int}` — per-variable max exponent (used to
  pre-allocate the power table in `evaluate`).
"""
struct DensePolynomial{T <: Number, NVAR, N, A <: AbstractArray{T, N}}
	coefficients::A
	multiindex_set::MultiindexSet{NVAR}
	max_exponents::SVector{NVAR, Int}

	# ── primary constructor ──────────────────────────────────────────────
	function DensePolynomial(
		coefficients::A,
		mset::MultiindexSet{NVAR},
	) where {T, NVAR, N, A <: AbstractArray{T, N}}

		L = length(mset)
		@assert size(coefficients, N) == L string(
			"Last axis of coefficients ($(size(coefficients, N))) ",
			"must equal the number of monomials ($L).")

		# Optionally assert T is a bits type for best performance:
		# @assert isbitstype(T) "Coefficient element type $T should be a bits type."

		max_arr = zeros(Int, NVAR)
		for exp in mset.exponents, j in 1:NVAR
			exp[j] > max_arr[j] && (max_arr[j] = exp[j])
		end
		new{T, NVAR, N, A}(coefficients, mset, SVector{NVAR, Int}(max_arr))
	end
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructors
# ─────────────────────────────────────────────────────────────────────────────

# NOTE: Vector{T<:Number} and AbstractMatrix{T<:Number} are already subtypes of
# AbstractArray{T,1} and AbstractArray{T,2} respectively, so they dispatch
# directly to the primary constructor above — no wrapper methods needed.

"""
	DensePolynomial(coeff_vec::Vector{SVector{K,T}}, mset)

Vector-valued polynomial from the old `Vector{SVector}` layout.
Converts to a contiguous `(K × L)` matrix, filling columns in parallel.
"""
function DensePolynomial(
	coeff_vec::Vector{SVector{K, T}},
	mset::MultiindexSet{NVAR},
) where {K, T <: Number, NVAR}
	L = length(mset)
	@assert length(coeff_vec) == L
	mat = Matrix{T}(undef, K, L)
	@threads for i in 1:L          # independent columns → parallel fill
		@inbounds mat[:, i] = coeff_vec[i]
	end
	DensePolynomial(mat, mset)
end

"""
	DensePolynomial(dict::Dict{Vector{Int}, T}) where T<:Number

Scalar polynomial from an `exponent → coefficient` dictionary.
"""
function DensePolynomial(dict::Dict{Vector{Int}, T}) where {T <: Number}
	isempty(dict) && return DensePolynomial(T[], MultiindexSet())
	N = length(first(keys(dict)))
	svdict = Dict{SVector{N, Int}, T}(SVector{N, Int}(k) => v for (k, v) in dict)
	mset = MultiindexSet(collect(keys(svdict)))
	L = length(mset)
	coeffs = Vector{T}(undef, L)
	@threads for i in 1:L
		@inbounds coeffs[i] = get(svdict, mset.exponents[i], zero(T))
	end
	DensePolynomial(coeffs, mset)
end

"""
	DensePolynomial(dict::Dict{Vector{Int}, SVector{K,T}}) where {K,T}

Vector-valued polynomial from an `exponent → SVector` dictionary.
Coefficients are laid out column-by-column (parallel fill).
"""
function DensePolynomial(dict::Dict{Vector{Int}, SVector{K, T}}) where {K, T <: Number}
	isempty(dict) && return DensePolynomial(Matrix{T}(undef, K, 0), MultiindexSet())
	N = length(first(keys(dict)))
	svdict = Dict{SVector{N, Int}, SVector{K, T}}(SVector{N, Int}(k) => v for (k, v) in dict)
	mset = MultiindexSet(collect(keys(svdict)))
	L = length(mset)
	mat = Matrix{T}(undef, K, L)
	@threads for i in 1:L
		@inbounds mat[:, i] = get(svdict, mset.exponents[i], zero(SVector{K, T}))
	end
	DensePolynomial(mat, mset)
end

# ─────────────────────────────────────────────────────────────────────────────
# Memory-mapped constructor  (huge datasets)
# ─────────────────────────────────────────────────────────────────────────────

"""
	mmap_polynomial(path, coeff_size::NTuple, T::Type, mset; write=false)

Return a `DensePolynomial` whose coefficient array is backed by a
memory-mapped file at `path`.  `coeff_size` is the leading axes of the
coefficient array (empty `()` for scalar, `(K,)` for K-vector, …).

The OS page-cache handles RAM pressure automatically: only accessed pages
are loaded, so polynomials larger than available RAM work transparently.

## Example
```julia
# 5-vector polynomial with 1 000 000 monomials, stored on disk
p = mmap_polynomial("coeffs.bin", (5,), Float64, mset; write=true)
```
"""
function mmap_polynomial(
	path::AbstractString,
	coeff_size::NTuple{M, Int},
	::Type{T},
	mset::MultiindexSet;
	write::Bool = false,
) where {M, T <: Number}
	L = length(mset)
	full_size = (coeff_size..., L)
	mode = write ? "w+" : "r"
	io = open(path, mode)
	arr = Mmap.mmap(io, Array{T, M+1}, full_size)
	close(io)
	return DensePolynomial(arr, mset)
end

# ─────────────────────────────────────────────────────────────────────────────
# Accessors
# ─────────────────────────────────────────────────────────────────────────────

coefficients(p::DensePolynomial) = p.coefficients
multiindex_set(p::DensePolynomial) = p.multiindex_set
nvars(::DensePolynomial{T, NVAR}) where {T, NVAR} = NVAR
nmonomials(p::DensePolynomial{T, NVAR, N}) where {T, NVAR, N} = size(p.coefficients, N)

"""
	coeff_shape(p) -> NTuple

Leading axes of the coefficient array.  `()` for scalar, `(K,)` for
K-vector, `(m,n)` for a matrix-valued polynomial, etc.
"""
coeff_shape(p::DensePolynomial{T, NVAR, N}) where {T, NVAR, N} =
	size(p.coefficients)[1:(N-1)]

Base.length(p::DensePolynomial) = nmonomials(p)
Base.eltype(::DensePolynomial{T}) where {T} = T
Base.iszero(p::DensePolynomial) = all(iszero, p.coefficients)

# ─────────────────────────────────────────────────────────────────────────────
# Term lookup
# ─────────────────────────────────────────────────────────────────────────────

find_in_multiindex_set(p::DensePolynomial{T, NVAR}, exp::SVector{NVAR, Int}) where {T, NVAR} =
	find_in_set(multiindex_set(p), exp)

function find_in_multiindex_set(p::DensePolynomial{T, NVAR}, exp::AbstractVector{Int}) where {T, NVAR}
	find_in_multiindex_set(p, SVector{NVAR, Int}(exp))
end

has_term(p::DensePolynomial, exp::AbstractVector{Int}) =
	!isnothing(find_in_multiindex_set(p, exp))

find_term(p::DensePolynomial, exp::AbstractVector{Int}) =
	find_in_multiindex_set(p, exp)

"""
	coefficient(p, exp) -> scalar or view

Return the coefficient of the monomial with exponent `exp`.
For scalar polynomials this is a `T`; for vector-valued it is a `Vector{T}`
view into the backing array — **no allocation**.
"""
function coefficient(p::DensePolynomial{T, NVAR, 1}, exp::AbstractVector{Int}) where {T, NVAR}
	idx = find_in_multiindex_set(p, exp)
	isnothing(idx) && return zero(T)
	return p.coefficients[idx]
end

function coefficient(p::DensePolynomial{T, NVAR, 2}, exp::AbstractVector{Int}) where {T, NVAR}
	idx = find_in_multiindex_set(p, exp)
	K = size(p.coefficients, 1)
	isnothing(idx) && return zeros(T, K)
	return @view p.coefficients[:, idx]   # zero-copy slice
end

# ─────────────────────────────────────────────────────────────────────────────
# polynomial_from_pairs
# ─────────────────────────────────────────────────────────────────────────────

function polynomial_from_pairs(pairs::Vector{Pair{Vector{Int}, T}}) where {T <: Number}
	DensePolynomial(Dict(pairs))
end

function polynomial_from_pairs(pairs::Vector{Pair{Vector{Int}, SVector{K, T}}}) where {K, T <: Number}
	DensePolynomial(Dict(pairs))
end

# Typed dispatch: polynomial_from_pairs(DensePolynomial{T}, pairs) — backward compat
function polynomial_from_pairs(
	::Type{DensePolynomial{T}}, pairs::Vector{Pair{Vector{Int}, T}}) where {T <: Number}
	DensePolynomial(Dict(pairs))
end

# ─────────────────────────────────────────────────────────────────────────────
# Zero polynomial
# ─────────────────────────────────────────────────────────────────────────────

"""
	zero(DensePolynomial{T}, mset)              # scalar
	zero(DensePolynomial{T}, coeff_shape, mset) # tensor-valued
"""
function Base.zero(::Type{DensePolynomial{T}}, mset::MultiindexSet) where {T <: Number}
	DensePolynomial(zeros(T, length(mset)), mset)
end

function Base.zero(
	::Type{DensePolynomial{T}},
	cshape::NTuple{M, Int},
	mset::MultiindexSet,
) where {T <: Number, M}
	DensePolynomial(zeros(T, cshape..., length(mset)), mset)
end

# ─────────────────────────────────────────────────────────────────────────────
# Monomial evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
	_precompute_powers(T, vals, max_exps) -> NTuple of Vectors

For each variable j, store `vals[j]^e` for e = 0 … max_exps[j].
Avoids repeated `^` calls inside the inner loop.
"""
@inline function _precompute_powers(
	::Type{T},
	vals::AbstractVector{<:Number},
	max_exps::SVector{NVAR, Int},
) where {T, NVAR}
	Tv = promote_type(T, eltype(vals))
	ntuple(NVAR) do j
		v = Tv(vals[j])
		pw = Vector{Tv}(undef, max_exps[j] + 1)
		pw[1] = one(Tv)
		@inbounds for e in 1:max_exps[j]
			pw[e+1] = pw[e] * v
		end
		pw
	end
end

@inline function _monomial(exp::SVector{NVAR, Int}, powers::NTuple{NVAR, <:AbstractVector}) where {NVAR}
	m = one(eltype(first(powers)))
	@inbounds for j in 1:NVAR
		m *= powers[j][exp[j]+1]
	end
	m
end

"""
	_monomial_vector!(m, poly, vals)

Fill the pre-allocated vector `m` (length L) with the value of each
monomial in `poly.multiindex_set` evaluated at `vals`.
This is the central loop; called once per `evaluate`.
"""
function _monomial_vector!(
	m::Vector{Tv},
	poly::DensePolynomial{T, NVAR},
	vals::AbstractVector{<:Number},
) where {Tv, T, NVAR}
	powers = _precompute_powers(Tv, vals, poly.max_exponents)
	exps   = poly.multiindex_set.exponents
	@inbounds for i in eachindex(m)
		m[i] = _monomial(exps[i], powers)
	end
	return m
end

# ─────────────────────────────────────────────────────────────────────────────
# evaluate
# ─────────────────────────────────────────────────────────────────────────────

"""
	evaluate(poly::DensePolynomial{T,NVAR,1}, vals) -> T

Scalar polynomial evaluation: `sum(c_i * m_i)` without conjugation.
"""
function evaluate(
	poly::DensePolynomial{T, NVAR, 1},
	vals::AbstractVector{<:Number},
) where {T, NVAR}
	@assert length(vals) == NVAR "Expected $NVAR variables, got $(length(vals))"
	iszero(poly) && return zero(T)
	L  = nmonomials(poly)
	Tv = promote_type(T, eltype(vals))
	m  = Vector{Tv}(undef, L)
	_monomial_vector!(m, poly, vals)
	return mapreduce(*, +, poly.coefficients, m)
end

"""
	evaluate(poly::DensePolynomial{T,NVAR,2}, vals) -> Vector{T}

Vector-valued polynomial evaluation via BLAS `gemv`:

	result = coefficients  (K × L)  *  m  (L,)  →  (K,)

A single BLAS call, no per-term allocation.
"""
function evaluate(
	poly::DensePolynomial{T, NVAR, 2},
	vals::AbstractVector{<:Number},
) where {T, NVAR}
	@assert length(vals) == NVAR "Expected $NVAR variables, got $(length(vals))"
	K = size(poly.coefficients, 1)
	L = nmonomials(poly)
	iszero(poly) && return zeros(T, K)
	Tv = promote_type(T, eltype(vals))
	m  = Vector{Tv}(undef, L)
	_monomial_vector!(m, poly, vals)
	return poly.coefficients * m        # BLAS gemv  (K×L)·(L,)
end

"""
	evaluate(poly::DensePolynomial{T,NVAR,N}, vals) -> Array

General tensor-valued case: reshape to (prod(leading_dims), L), gemv, reshape back.
"""
function evaluate(
	poly::DensePolynomial{T, NVAR, N},
	vals::AbstractVector{<:Number},
) where {T, NVAR, N}
	@assert length(vals) == NVAR "Expected $NVAR variables, got $(length(vals))"
	leading = coeff_shape(poly)
	K = prod(leading)
	L = nmonomials(poly)
	iszero(poly) && return zeros(T, leading...)
	Tv = promote_type(T, eltype(vals))
	m  = Vector{Tv}(undef, L)
	_monomial_vector!(m, poly, vals)
	flat = reshape(poly.coefficients, K, L)  # view, no copy
	res  = flat * m                           # BLAS gemv
	return reshape(res, leading...)
end

"""
	evaluate(poly::DensePolynomial{T,NVAR,2}, vals, component::Int) -> T

Evaluate a single component of a vector-valued polynomial without
allocating a full output vector.
"""
function evaluate(
	poly::DensePolynomial{T, NVAR, 2},
	vals::AbstractVector{<:Number},
	component::Int,
) where {T, NVAR}
	@assert 1 <= component <= size(poly.coefficients, 1) "Component out of range"
	L  = nmonomials(poly)
	Tv = promote_type(T, eltype(vals))
	m  = Vector{Tv}(undef, L)
	_monomial_vector!(m, poly, vals)
	return mapreduce(*, +, @view(poly.coefficients[component, :]), m)
end

# ─────────────────────────────────────────────────────────────────────────────
# extract_component  (vector → scalar)
# ─────────────────────────────────────────────────────────────────────────────

"""
	extract_component(poly::DensePolynomial{T,NVAR,2}, idx) -> DensePolynomial{T,NVAR,1}

Return the `idx`-th component as a scalar polynomial.
The coefficient vector is a **copy** of row `idx` of the matrix.
"""
function extract_component(poly::DensePolynomial{T, NVAR, 2}, idx::Int) where {T, NVAR}
	row = Vector{T}(poly.coefficients[idx, :])  # copy for contiguous 1-D layout
	return DensePolynomial(row, poly.multiindex_set)
end

# ─────────────────────────────────────────────────────────────────────────────
# each_term  iterator
# ─────────────────────────────────────────────────────────────────────────────

"""
	each_term(poly) -> generator of (exponent, coefficient)

Yields `(SVector{NVAR,Int}, coeff)` for every nonzero monomial.

- Scalar (N=1): `coeff` is a `T`.
- Vector (N=2): `coeff` is a `SubArray{T,1}` view (no allocation).
"""
function each_term(poly::DensePolynomial{T, NVAR, 1}) where {T, NVAR}
	exps = poly.multiindex_set.exponents
	c    = poly.coefficients
	return ((exps[i], c[i]) for i in 1:nmonomials(poly) if !iszero(c[i]))
end

function each_term(poly::DensePolynomial{T, NVAR, 2}) where {T, NVAR}
	exps = poly.multiindex_set.exponents
	c    = poly.coefficients
	return ((exps[i], @view(c[:, i])) for i in 1:nmonomials(poly) if !iszero(@view(c[:, i])))
end

# ─────────────────────────────────────────────────────────────────────────────
# similar_poly
# ─────────────────────────────────────────────────────────────────────────────

function similar_poly(dict::Dict{SVector{NVAR, Int}, T}) where {NVAR, T <: Number}
	isempty(dict) && return DensePolynomial(T[], MultiindexSet(Vector{SVector{NVAR, Int}}()))
	mset = MultiindexSet(collect(keys(dict)))
	L = length(mset)
	c = Vector{T}(undef, L)
	@threads for i in 1:L
		@inbounds c[i] = get(dict, mset.exponents[i], zero(T))
	end
	DensePolynomial(c, mset)
end

function similar_poly(dict::Dict{SVector{NVAR, Int}, SVector{K, T}}) where {NVAR, K, T <: Number}
	isempty(dict) && return DensePolynomial(Matrix{T}(undef, K, 0), MultiindexSet(Vector{SVector{NVAR, Int}}()))
	mset = MultiindexSet(collect(keys(dict)))
	L = length(mset)
	mat = Matrix{T}(undef, K, L)
	@threads for i in 1:L
		@inbounds mat[:, i] = get(dict, mset.exponents[i], zero(SVector{K, T}))
	end
	DensePolynomial(mat, mset)
end

"""
	similar_poly(dict::Dict{SVector{NVAR,Int}, Vector{T}}) where {NVAR, T<:Number}

Vector-valued polynomial from a dictionary mapping exponents to `Vector{T}` coefficients.
All vectors must have the same length K.
"""
function similar_poly(dict::Dict{SVector{NVAR, Int}, Vector{T}}) where {NVAR, T <: Number}
	isempty(dict) && return DensePolynomial(Matrix{T}(undef, 0, 0), MultiindexSet(Vector{SVector{NVAR, Int}}()))
	K = length(first(values(dict)))
	mset = MultiindexSet(collect(keys(dict)))
	L = length(mset)
	mat = Matrix{T}(undef, K, L)
	@threads for i in 1:L
		@inbounds mat[:, i] = get(dict, mset.exponents[i], zeros(T, K))
	end
	DensePolynomial(mat, mset)
end

# ─────────────────────────────────────────────────────────────────────────────
# linear_matrix_of_polynomial
# ─────────────────────────────────────────────────────────────────────────────

"""
	linear_matrix_of_polynomial(poly::DensePolynomial{T,NVAR,2}) -> Matrix{T}

Return the `(K × NVAR)` matrix `A` such that the linear part equals `A * x`.
Reads directly from the coefficient matrix — no intermediate arrays.
"""
function linear_matrix_of_polynomial(poly::DensePolynomial{T, NVAR, 2}) where {T, NVAR}
	K = size(poly.coefficients, 1)
	A = zeros(T, K, NVAR)
	for (i, exp) in enumerate(poly.multiindex_set.exponents)
		s = sum(exp)
		s > 1 && break                              # Grlex order: rest are higher-degree
		if s == 1
			j = findfirst(==(1), exp)::Int
			@inbounds A[:, j] = @view poly.coefficients[:, i]
		end
	end
	return A
end

# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic
# ─────────────────────────────────────────────────────────────────────────────

import Base: +, -, *, ==

*(s::Number, p::DensePolynomial) = DensePolynomial(s .* p.coefficients, p.multiindex_set)
*(p::DensePolynomial, s::Number) = s * p

function +(p1::DensePolynomial{T, NVAR, N}, p2::DensePolynomial{T, NVAR, N}) where {T, NVAR, N}
	@assert p1.multiindex_set == p2.multiindex_set "Polynomials must share the same multiindex set"
	DensePolynomial(p1.coefficients .+ p2.coefficients, p1.multiindex_set)
end

-(p1::DensePolynomial, p2::DensePolynomial) = p1 + (-1) * p2

==(p1::DensePolynomial, p2::DensePolynomial) =
	p1.multiindex_set == p2.multiindex_set && p1.coefficients == p2.coefficients

end # module Polynomials
