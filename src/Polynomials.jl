module Polynomials

using LinearAlgebra
using StaticArrays: SVector, SArray, StaticArray
using ..Multiindices: MultiindexSet, find_in_set, zero_multiindex #, nvars

export DensePolynomial,
	polynomial_from_pairs,
	coefficients, multiindex_set, nvars,
	coefficient, has_term, find_term, find_in_multiindex_set,
	zero, evaluate, extract_component, each_term, similar_poly

# ---------- DensePolynomial ----------
"""
	DensePolynomial{C,NVAR}

Dense representation: coefficients aligned with the full multiindex_set.
- `coefficients::Vector{C}`: length = number of monomials in `multiindex_set`.
- `multiindex_set::MultiindexSet{NVAR}`: reference to the monomial set 
	of `NVAR` variables (always Grlex‑ordered).
- `max_exponents::SVector{NVAR,Int}`: maximum exponent for each variable 
	across all monomials in the polynomial

For scalar polynomials, `C <: Number`. For vector‑valued polynomials,
`C` should be a static array type (e.g., `SVector{3, Float64}`).
"""
struct DensePolynomial{C, NVAR}
	coefficients::Vector{C}
	multiindex_set::MultiindexSet{NVAR}
	max_exponents::SVector{NVAR, Int}

	function DensePolynomial{C}(
		coefficients::Vector{C}, mset::MultiindexSet{NVAR}) where {C, NVAR}
		@assert length(coefficients) == length(mset)
		# compute max_exponents from the multiindex set
		max_arr = zeros(Int, NVAR)
		for exp in mset.exponents
			for j in 1:NVAR
				if exp[j] > max_arr[j]
					max_arr[j] = exp[j]
				end
			end
		end
		new{C, NVAR}(coefficients, mset, SVector{NVAR, Int}(max_arr))
	end
end

# Convenience constructors
function DensePolynomial(coefficients::Vector{C}, mset::MultiindexSet) where {C}
	DensePolynomial{C}(coefficients, mset)
end

"""
	DensePolynomial(dict::Dict{Vector{Int}, C}) where C

Construct a polynomial from a dictionary mapping exponent vectors to coefficients.
The multiindex set is built from the keys and sorted in Grlex order.
Coefficients can be numbers or static arrays.
"""
function DensePolynomial(dict::Dict{Vector{Int}, C}) where {C}
	isempty(dict) && return DensePolynomial(C[], MultiindexSet())
	N = length(first(keys(dict)))
	# Convert dictionary keys to SVector for efficient lookup
	svdict = Dict{SVector{N, Int}, C}(SVector{N, Int}(k) => v for (k, v) in dict)
	mset = MultiindexSet(collect(keys(svdict)))  # sorts in Grlex
	coefficients = [get(svdict, exp, zero(C)) for exp in mset.exponents]
	return DensePolynomial(coefficients, mset)
end

# ---------- Accessors ----------
coefficients(p::DensePolynomial) = p.coefficients
multiindex_set(p::DensePolynomial) = p.multiindex_set
nvars(::DensePolynomial{<:Any, NVAR}) where {NVAR} = NVAR
Base.length(p::DensePolynomial) = length(p.coefficients)
Base.eltype(::DensePolynomial{C}) where {C} = C

# ---------- Term lookup (using AbstractVector{Int}) ----------
"""
	find_in_multiindex_set(p::DensePolynomial, exp::SVector{NVAR,Int}) -> Union{Int,Nothing}

Return the column index of `exp` in the polynomial's multiindex_set, or `nothing` if not present.
"""
find_in_multiindex_set(p::DensePolynomial{<:Any, NVAR}, exp::SVector{NVAR, Int}) where {NVAR} = find_in_set(
	multiindex_set(p), exp)

function find_in_multiindex_set(
	p::DensePolynomial{<:Any, NVAR}, exp::AbstractVector{Int}) where {NVAR}
	find_in_multiindex_set(p, SVector{NVAR}(exp))
end

"""
	has_term(p::DensePolynomial, exp::AbstractVector{Int}) -> Bool

Check whether the polynomial contains a term with exponent `exp`.
"""
has_term(p::DensePolynomial, exp::AbstractVector{Int}) = !isnothing(find_in_multiindex_set(
	p, exp))

"""
	coefficient(p::DensePolynomial, exp::AbstractVector{Int}) -> eltype(p)

Return the coefficient of the term with exponent `exp`, or zero if not present.
"""
function coefficient(p::DensePolynomial, exp::AbstractVector{Int})
	idx = find_in_multiindex_set(p, exp)
	isnothing(idx) && return zero(eltype(p))
	return p.coefficients[idx]
end

"""
	find_term(p::DensePolynomial, exp::AbstractVector{Int}) -> Union{Int,Nothing}

Return the position in the polynomial's storage (index in `coefficients`) of the term with exponent `exp`,
or `nothing` if not present.
"""
find_term(p::DensePolynomial, exp::AbstractVector{Int}) = find_in_multiindex_set(p, exp)

# ---------- polynomial_from_pairs (alternative input) ----------
"""
	polynomial_from_pairs(::Type{DensePolynomial{C}}, pairs::Vector{Pair{Vector{Int},C}}) where C

Construct a polynomial from a vector of (exponent => coefficient) pairs.
Useful for building polynomials programmatically.
"""
function polynomial_from_pairs(
	::Type{DensePolynomial{C}}, pairs::Vector{Pair{Vector{Int}, C}}) where {C}
	dict = Dict(pairs)
	return DensePolynomial(dict)
end

# Convenience method using default type
function polynomial_from_pairs(pairs::Vector{Pair{Vector{Int}, C}}) where {C}
	DensePolynomial(Dict(pairs))
end

import Base: zero

# ---------- Zero polynomial constructors ----------
"""
	zero(::Type{DensePolynomial{C}}, set::MultiindexSet) where C

Construct a zero polynomial (all coefficients zero) in the given monomial basis `set`.
Works for both scalar (`C<:Number`) and static‑array (`C<:StaticArray`) coefficient types.
"""
function Base.zero(::Type{DensePolynomial{C}}, set::MultiindexSet) where {C}
	coefficients = [zero(C) for _ in 1:length(set)]
	return DensePolynomial{C}(coefficients, set)
end

Base.iszero(p::DensePolynomial) = all(iszero, p.coefficients)

# ---------- Evaluate polynomial ----------

"""
	_precompute_powers(vals::AbstractVector{<:Number}, max_exponents::Vector{Int})

Precompute `vals[j]^e` for e = 0..max_exponents[j] for each variable j.
Returns a vector of vectors (or a vector of arrays) for efficient monomial evaluation.
"""
@inline function _precompute_powers(::Type{C}, vals::AbstractVector{<:Number},
	max_exps::SVector{NVAR, Int}) where {C, NVAR}
	T = promote_type(eltype(C), eltype(vals))   # monomial type
	powers = ntuple(j -> Vector{T}(undef, max_exps[j] + 1), NVAR)
	for j in 1:NVAR
		v = vals[j]
		pow = powers[j]
		pow[1] = one(T)
		@inbounds for e in 1:max_exps[j]
			pow[e+1] = pow[e] * v
		end
	end
	return powers
end

"""
	_monomial(exp::SVector{N,Int}, powers::NTuple{NVAR,Vector{T}}) where N

Compute the monomial value from precomputed powers. (x1, x2, ..., xN)^(a1, a2, ..., aN)
"""
@inline function _monomial(
	exp::SVector{NVAR, Int}, powers::NTuple{NVAR, Vector{T}}) where {NVAR, T}
	m = one(T)
	@inbounds for j in 1:NVAR
		m *= powers[j][exp[j]+1]
	end
	return m
end

"""
	evaluate(poly::DensePolynomial{C}, vals::AbstractVector{<:Number}) where C

Evaluate the polynomial at the given variable values.
Returns a value of type `C` (number or static array).
"""
function evaluate(poly::DensePolynomial{C}, vals::AbstractVector{<:Number}) where {C}
	@assert nvars(poly) == length(vals)
	iszero(poly) && return zero(C)

	powers = _precompute_powers(C, vals, poly.max_exponents)
	result = zero(C)
	exps = poly.multiindex_set.exponents
	coefficients = poly.coefficients
	@inbounds for i in 1:length(coefficients)
		c = coefficients[i]
		iszero(c) && continue
		m = _monomial(exps[i], powers)
		result += c * m
	end
	return result
end

"""
	evaluate(poly::DensePolynomial{SVector{L,T}}, vals::AbstractVector{<:Number}, component::Int) where {L,T}

Evaluate the `component`-th component of a vector-valued polynomial at the given variable values.
"""
function evaluate(poly::DensePolynomial{SVector{L, T}},
	vals::AbstractVector{<:Number}, component::Int) where {L, T}
	@assert nvars(poly)==length(vals) "Number of variables mismatch"
	@assert 1<=component<=L "Component index out of range"
	if iszero(poly)
		return zero(T)
	end
	# Precompute powers
	powers = _precompute_powers(SVector{L, T}, vals, poly.max_exponents)
	result = zero(T)
	exps = poly.multiindex_set.exponents
	coefficients = poly.coefficients
	@inbounds for i in 1:length(coefficients)
		c = coefficients[i][component]
		iszero(c) && continue
		m = _monomial(exps[i], powers)
		result += c * m
	end
	return result
end

# ---------- Extract component from vector polynomial ----------
"""
	extract_component(poly::DensePolynomial{SVector{L,T}}, idx::Int) where {L,T}

For a vector‑valued polynomial (coefficients are `SVector{L,T}`), extract the `idx`-th component
as a scalar polynomial.
"""
function extract_component(poly::DensePolynomial{SVector{L, T}}, idx::Int) where {L, T}
	new_coefficients = [c[idx] for c in poly.coefficients]
	return DensePolynomial{T}(new_coefficients, poly.multiindex_set)
end

# ---------- Iteration over terms ----------
"""
	each_term(poly::DensePolynomial)

Return a generator that yields `(exponent_vector, coefficient)` for every
non‑zero term of `poly`. For dense polynomials, zero coefficients are skipped.
"""
function each_term(poly::DensePolynomial)
	exps = poly.multiindex_set.exponents
	coefficients_vec = poly.coefficients
	return ((exps[j], coefficients_vec[j]) for j in 1:length(exps) if !iszero(coefficients_vec[j]))
end

# ---------- similar_poly (construct polynomial of same type from dictionary) ----------
"""
	similar_poly(dict::Dict{SVector{NVAR,Int}, C}) where {NVAR,C}

Construct a new polynomial from the dictionary `dict` (exponents → coefficients).
The polynomial will have `NVAR` variables and coefficient type `C`.
If `dict` is empty, an empty polynomial with zero coefficients is returned.
"""
function similar_poly(dict::Dict{SVector{NVAR, Int}, C}) where {NVAR, C}
	if isempty(dict)
		# Create an empty MultiindexSet with NVAR variables (no exponents)
		mset = MultiindexSet(Vector{SVector{NVAR, Int}}())
		return DensePolynomial(C[], mset)
	end

	mset = MultiindexSet(collect(keys(dict)))

	# Build coefficient vector aligned with the sorted exponent set
	coefficients = Vector{C}(undef, length(mset.exponents))
	@inbounds for (i, exp) in enumerate(mset.exponents)
		coefficients[i] = get(dict, exp, zero(C))
	end

	return DensePolynomial(coefficients, mset)
end

# ---------- Basic arithmetic (optional) ----------
import Base: +, -, *, ==

function *(s::Number, p::DensePolynomial{C}) where {C}
	DensePolynomial(s .* p.coefficients, p.multiindex_set)
end
*(p::DensePolynomial{C}, s::Number) where {C} = s * p

function +(p1::DensePolynomial{C}, p2::DensePolynomial{C}) where {C}
	@assert p1.multiindex_set==p2.multiindex_set "Cannot add polynomials with different multiindex sets"
	DensePolynomial(p1.coefficients .+ p2.coefficients, p1.multiindex_set)
end

-(p1::DensePolynomial{C}, p2::DensePolynomial{C}) where {C} = p1 + (-1) * p2

function ==(p1::DensePolynomial, p2::DensePolynomial)
	p1.multiindex_set == p2.multiindex_set && p1.coefficients == p2.coefficients
end

end # module
