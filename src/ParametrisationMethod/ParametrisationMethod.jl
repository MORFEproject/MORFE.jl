module ParametrisationMethod

using StaticArrays: SVector
using ..Multiindices: MultiindexSet
using ..Polynomials: DensePolynomial

export Parametrisation, ReducedDynamics, create_parametrisation_method_objects

"""
	Parametrisation{ORD, NVAR, T}

A dense polynomial with a contiguous `(FOM, ORD, L)` coefficient array.
Represents a parametrisation mapping from reduced coordinates and forcing variables to the full state.

- `ORD`: native order of the full ODE (1 for first‑order, 2 for second‑order).
- `NVAR`: total number of variables = reduced coordinates + forcing variables.
- `T`: numeric element type (e.g., `ComplexF64`).

Layout: `coefficients[:, ord, l]` is the full‑state vector (length FOM) for the
`ord`-th time derivative of the `l`-th monomial coefficient.
"""
struct Parametrisation{ORD, NVAR, T}
	poly::DensePolynomial{T, NVAR, 3, Array{T, 3}}
	external_system_size::Int

	function Parametrisation(poly::DensePolynomial{T, NVAR, 3, Array{T, 3}}, external_system_size::Int) where {T, NVAR}
		ORD = size(poly.coefficients, 2)
		@assert external_system_size >= 0 "external_system_size must be non‑negative"
		new{ORD, NVAR, T}(poly, external_system_size)
	end
end

Base.size(W::Parametrisation) = size(W.poly.coefficients, 1) # FOM: full‑order state dimension
multiindex_set(W::Parametrisation) = W.poly.multiindex_set
coefficients(W::Parametrisation) = W.poly.coefficients

"""
	ReducedDynamics{ROM, NVAR, T}

A dense polynomial whose coefficients are `SVector{ROM, T}`.
Represents the reduced dynamics on a manifold of dimension `ROM`.

- `ROM`: dimension of the reduced state (first‑order system).
- `NVAR`: total number of variables = ROM + external_system_size.
- `T`: numeric type.

The polynomial is stored in `poly`, and `external_system_size` gives the number of forcing variables.
The dynamics are: ż = R(z, r), where r are the forcing variables.
"""
struct ReducedDynamics{ROM, NVAR, T}
	poly::DensePolynomial{T, NVAR, 2, Matrix{T}}
	external_system_size::Int

	function ReducedDynamics(poly::DensePolynomial{T, NVAR, 2, Matrix{T}}, external_system_size::Int) where {T, NVAR}
		ROM = size(poly.coefficients, 1)
		@assert external_system_size >= 0 "external_system_size must be non‑negative"
		@assert ROM + external_system_size == NVAR "ROM + external_system_size must equal NVAR; got $(ROM + external_system_size) vs $NVAR"
		new{ROM, NVAR, T}(poly, external_system_size)
	end
end

Base.size(::ReducedDynamics{ROM}) where {ROM} = ROM
multiindex_set(R::ReducedDynamics) = R.poly.multiindex_set
coefficients(R::ReducedDynamics) = R.poly.coefficients

"""
	create_parametrisation_method_objects(mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int, ROM::Int, external_system_size::Int, ::Type{T}=Complex)

Create a consistent pair of polynomials:
- `W`: a `Parametrisation{ORD, NVAR, T}` with zero coefficients,
- `R`: a `ReducedDynamics{ROM, NVAR, T}` with zero coefficients.

Both polynomials share the same multiindex set `mset` and element type `T`.
The total number of variables `NVAR` must satisfy `NVAR == ROM + external_system_size`.
`FOM` is the full‑order dimension (size of the state vector). It is not stored but used
to initialise the coefficient vectors correctly.

# Arguments
- `mset`: multiindex set for `NVAR` variables.
- `ORD`: native order of the full ODE (1 or 2).
- `FOM`: dimension of the full‑order state in its native order.
- `ROM`: dimension of the reduced state.
- `external_system_size`: number of forcing variables (default 0).
- `T`: element type.
"""
function create_parametrisation_method_objects(
	mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int, ROM::Int, external_system_size::Int,
	::Type{T} = Complex) where {T <: Number, NVAR}
	# Validate variable count
	@assert NVAR == ROM + external_system_size "Multiindex set has $NVAR variables, but ROM + external_system_size = $(ROM + external_system_size)"

	# Parametrisation coefficients: (FOM, ORD, L) 3-D array
	W_poly = DensePolynomial(zeros(T, FOM, ORD, length(mset)), mset)
	W = Parametrisation(W_poly, external_system_size)

	# Reduced dynamics coefficients: (ROM, L) matrix
	R_poly = DensePolynomial(zeros(T, ROM, length(mset)), mset)
	R = ReducedDynamics(R_poly, external_system_size)

	return (W, R)
end

# For the special case where ROM = NVAR (no forcing, and reduced dimension equals variable count)
function create_parametrisation_method_objects(
	mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int,
	::Type{T} = Complex) where {T <: Number, NVAR}
	return create_parametrisation_method_objects(mset, ORD, FOM, NVAR, 0, T)
end

# after solving for the parametrisation coefficients for the zero-th time derivative in the full‑order ODE, 
# we can compute the coefficients for the higher time derivatives using the superharmonic structure of the invariance equation
# after having the x components of the parametrisation W, we compute the x' and x'' components 
function compute_higher_derivative_coefficients!(
	param_coeff::AbstractMatrix{T}, red_coeff::AbstractVector{T}, superharmonic::T, global_index::Int,
	generalised_eigenmodes::AbstractVector{<:AbstractMatrix{T}}, low_order_couplings::AbstractMatrix{T}) where {T}

	ORD = size(param_coeff, 2)
	for j in 1:(ORD-1)
		param_coeff[:, j+1, global_index] .= (param_coeff[:, j, global_index] * superharmonic)
		+ (generalised_eigenmodes[j] * red_coeff[:, global_index])
		+ low_order_couplings[:, j]
	end
end

end # module
