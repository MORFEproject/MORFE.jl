module ParametrisationMethod

using StaticArrays: SVector
using ..Multiindices: MultiindexSet
using ..Polynomials: DensePolynomial

export Parametrisation, ReducedDynamics, create_parametrisation_method_objects

"""
	Parametrisation{ORD, FOM, NVAR, T} = DensePolynomial{SVector{ORD, SVector{FOM, T}}, NVAR}

A dense polynomial whose coefficients are `SVector{ORD, SVector{FOM, T}}`.
Represents a parametrisation mapping from reduced coordinates to full‑state coordinates.
- `ORD` is the native order of the full order ODE
- `FOM` is the size of the full order model, in native order
- `NVAR` is the number of variables in the reduced order model
- `T` is the numeric type (e.g. Complex)

It represents a mapping from reduced coordinates 
`z` to the full state vector `x` and its derivative `ẋ` (or just `x` for first‑order systems). 
The outer dimension `ORD` corresponds to the order of the original ODE:
  * `ORD = 1` : first‑order system, mapping `ξ` → `x`.
  * `ORD = 2` : second‑order system, mapping `ξ` → `(x, ẋ)`.
"""
const Parametrisation{ORD, NVAR, T} = DensePolynomial{
	SVector{ORD, Vector{T}}, NVAR}

"""
	ReducedDynamics{ROM, NVAR, T} = DensePolynomial{SVector{ROM, T}, NVAR}

A dense polynomial whose coefficients are `SVector{ROM, T}`.
Represents the reduced dynamics on a manifold of dimension `ROM`.
- `ROM` is the size of the reduced order model, a first order system
- `NVAR` is the number of variables in the reduced order model 
	(in reality, `NVAR` should be the same as `ROM`, we only keep them separate for future flexibility)
- `T` is the numeric type (e.g. Complex)
"""
const ReducedDynamics{ROM, NVAR, T} = DensePolynomial{SVector{ROM, T}, NVAR}

"""
	create_parametrisation_method_objects(mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int, ROM::Int, ::Type{T}=Complex) where {T<:Number, NVAR}

Create a consistent pair of polynomials:
- `W`: a `Parametrisation{ORD, FOM, NVAR, T}` with zero coefficients,
- `R`: a `ReducedDynamics{ROM, NVAR, T}` with zero coefficients.

Both polynomials share the same multiindex set `mset` and element type `T`.
"""
function create_parametrisation_method_objects(
	mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int, ROM::Int,
	::Type{T} = Complex) where {T <: Number, NVAR}
	# Parametrisation coefficients: SVector{ORD, SVector{FOM, T}} zeros
	inner_zero = Vector{T}(zeros(T, FOM))
	outer_zero = SVector{ORD, Vector{T}}(ntuple(_ -> inner_zero, ORD))
	W_coeffs = [outer_zero for _ in 1:length(mset)]
	W = DensePolynomial(W_coeffs, mset)   # Parametrisation{ORD,FOM,NVAR,T}

	# Reduced dynamics coefficients: SVector{ROM, T} zeros
	R_coeffs = [SVector{ROM, T}(zeros(T, ROM)) for _ in 1:length(mset)]
	R = DensePolynomial(R_coeffs, mset)   # ReducedDynamics{ROM,NVAR,T}

	return (W, R)
end

# In practice, ROM = NVAR
function create_parametrisation_method_objects(
	mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int,
	::Type{T} = Complex) where {T <: Number, NVAR}
	# Parametrisation coefficients: SVector{ORD, SVector{FOM, T}} zeros
	inner_zero = Vector{T}(zeros(T, FOM))
	outer_zero = SVector{ORD, Vector{T}}(ntuple(_ -> inner_zero, ORD))
	W_coeffs = [outer_zero for _ in 1:length(mset)]
	W = DensePolynomial(W_coeffs, mset)   # Parametrisation{ORD,FOM,NVAR,T}

	# Reduced dynamics coefficients: SVector{ROM, T} zeros with ROM=NVAR
	R_coeffs = [SVector{NVAR, T}(zeros(T, NVAR)) for _ in 1:length(mset)]
	R = DensePolynomial(R_coeffs, mset)   # ReducedDynamics{ROM,NVAR,T}

	return (W, R)
end

end # module
