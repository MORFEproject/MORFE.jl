module ParametrisationMethod

using LinearAlgebra: mul!
using StaticArrays: SVector
using ..Multiindices: MultiindexSet
using ..Polynomials: DensePolynomial

export Parametrisation, ReducedDynamics, create_parametrisation_method_objects,
	compute_higher_derivative_coefficients!

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
		@assert external_system_size >= 0 "external_system_size must be non‑negative"
		ROM = NVAR - external_system_size
		@assert ROM > 0 "ROM = NVAR - external_system_size must be positive; got $(ROM)"
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
	R_poly = DensePolynomial(zeros(T, NVAR, length(mset)), mset)
	# THE REDUCED DYNAMICS POLYNOMIAL HAS NVAR VARIABLES, NOT ROM, BECAUSE IT DEPENDS ON ALL REDUCED + EXTERNAL VARS
	# ADDITIONALLY, THE COEFFICIENTS ARE NVAR-VECTORS, NOT SCALARS, SO THE COEFFICIENT ARRAY HAS SHAPE (NVAR, L)
	# THE LAST ROWS OF THE COEFFICIENTS CORRESPOND TO THE EXTERNAL SYSTEM TERMS, WHICH ARE DIRECTLY COPIED FROM THE FULL ORDER MODEL
	R = ReducedDynamics(R_poly, external_system_size)

	return (W, R)
end

# For the special case where ROM = NVAR (no forcing, and reduced dimension equals variable count)
function create_parametrisation_method_objects(
	mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int,
	::Type{T} = Complex) where {T <: Number, NVAR}
	return create_parametrisation_method_objects(mset, ORD, FOM, NVAR, 0, T)
end

"""
	compute_higher_derivative_coefficients!(
		param_coeff, red_coeff, external_dynamics, superharmonic, global_index,
		generalised_eigenmodes, lower_order_couplings
	) -> nothing

Compute the higher time‑derivative coefficients `W^(j+1)[α]` for `j = 1 … ORD-1`
using the superharmonic recurrence

```
W^(j+1)[α] = s · W^(j)[α]  +  Φ_master · R[α]  +  Φ_ext · e_dyn  +  ξ[j]
```

where:
- `s = superharmonic` is the frequency `⟨λ, α⟩`,
- `Φ = generalised_eigenmodes` (`FOM × NVAR`) collects the right eigenmodes,
- `R[α] = red_coeff[:, global_index]` (`ROM`‑vector) contains the master‑mode
  reduced‑dynamics coefficients at the current monomial (already solved),
- `e_dyn = external_dynamics` (`N_EXT`‑vector) contains the *known* external
  dynamics at the current monomial,
- `ξ[j] = lower_order_couplings[j]` (`FOM`‑vector) contains the coupling
  from lower‑order monomials at derivative order `j`.

Modifies `param_coeff` in‑place.  Does nothing when `ORD = 1` (no higher
derivatives exist for a first‑order ODE).

## Arguments

- `param_coeff :: AbstractArray{T, 3}` — shape `FOM × ORD × L`; the coefficient
  tensor of the parametrisation polynomial.
- `red_coeff :: AbstractMatrix{T}` — shape `ROM × L`; master‑mode reduced‑dynamics
  coefficients.
- `external_dynamics :: AbstractVector{T}` — length `N_EXT`; known external
  dynamics at the current monomial.
- `superharmonic :: T` — scalar `s = ⟨λ, α⟩`.
- `global_index :: Int` — monomial index into the last axis of `param_coeff` and
  the last axis of `red_coeff`.
- `generalised_eigenmodes :: AbstractMatrix{T}` — shape `FOM × NVAR`; right
  generalised eigenvectors (master modes in columns `1:ROM`, external modes in
  `ROM+1:NVAR`).
- `lower_order_couplings :: AbstractVector{<:AbstractVector{T}}` — length `ORD`;
  element `j` is a length‑`FOM` vector `ξ[j]` produced by
  [`LowerOrderCouplings.compute_lower_order_couplings`](@ref).
"""
function compute_higher_derivative_coefficients!(
	param_coeff::AbstractArray{T, 3},
	red_coeff::AbstractMatrix{T},
	external_dynamics::AbstractVector{T},
	superharmonic::T,
	global_index::Int,
	generalised_eigenmodes::AbstractMatrix{T},
	lower_order_couplings::AbstractVector{<:AbstractVector{T}},
) where {T}
	ORD = size(param_coeff, 2)
	ROM = size(red_coeff, 1)
	NVAR = size(generalised_eigenmodes, 2)
	N_EXT = NVAR - ROM

	Rα = view(red_coeff, :, global_index)

	for j in 1:(ORD-1)
		Wj  = view(param_coeff, :, j, global_index)
		Wj1 = view(param_coeff, :, j + 1, global_index)

		# W^(j+1)[α] = s·W^(j)[α] + ξ[j]
		@. Wj1 = superharmonic * Wj + lower_order_couplings[j]

		# + Φ_master · R[α]
		mul!(Wj1, view(generalised_eigenmodes, :, 1:ROM), Rα, one(T), one(T))

		# + Φ_ext · e_dyn  (only if external modes are present)
		if N_EXT > 0
			mul!(Wj1, view(generalised_eigenmodes, :, (ROM+1):NVAR), external_dynamics, one(T), one(T))
		end
	end

	return nothing
end

end # module
