module FullOrderModel

using LinearAlgebra
using SparseArrays
using StaticArrays: SVector

using ..Polynomials: DensePolynomial
using ..MultilinearMap: MultilinearMap, evaluate_term!
using ..ExternalSystem: ExternalSystem

export NDOrderModel, FirstOrderModel, linear_first_order_matrices, evaluate_nonlinear_terms!

abstract type AbstractFullOrderModel end

"""

	NDOrderModel{ORD, ORDP1, N_NL, MT} <: AbstractFullOrderModel

Representation of an ORD-th (ORDP1=ORD+1) order dynamical system of the form

	B_ORD x^(ORD) + ... + B_1 x^(1) + B_0 x = F(x^(ORD-1), …, x^(1), x, r, …, r)

where:
- x^(n) is the n-th derivative of x (x^(n) = d_t^n x)
- x^(0) = x is the state vector
- B_i are the coefficient matrices
- F is a multilinear polynomial function of the derivatives and the external state vector r
- The external state r satisfies its own first‑order dynamics r' = g(r)

# Generic type parameters

- `ORD` defines the order of the ODE.
- `ORDP1` is the number of linear terms (from 0 through ORD). It must satisfy ORDP1 == ORD+1.
- `N_NL` is the number of nonlinear terms in the tuple nonlinear_terms.
- `N_EXT` is the size of the external system.
- `T` is the numeric type.
- `MT` is the matrix type that forms the ORDP1-tuple of linear_terms.

# Fields

- `n_fom`: dimension of the full‑order state vector x.
- `linear_terms`: tuple of linear coefficient matrices (B_0, …, B_ORD).
- `nonlinear_terms`: tuple of `MultilinearMap` representing nonlinear contributions.
- `external_system`: object of tyle `ExternalSystem` defining the external dynamics.

# Representation

- Linear terms are stored as a tuple `(B_0, …, B_ORD)`
- Nonlinear terms are represented as a collection of `MultilinearMap`s

Each `MultilinearMap` defines:
- which derivatives are involved (via `multiindex`)
- how many times the external state appears as an argument (via `multiplicity_external`)
- the combined degree `deg = sum(multiindex) + multiplicity_external`

# Notes
- All matrices must have identical size.
- The nonlinear structure is stored in sparse form (only active terms).
- TODO For large `K`, a `Vector` may be more appropriate than an `NTuple`.
"""
struct NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT <: AbstractMatrix{T}} <: AbstractFullOrderModel
	n_fom::Int
	linear_terms::NTuple{ORDP1, MT}
	nonlinear_terms::NTuple{N_NL, MultilinearMap{ORD}}
	external_system::Union{Nothing, ExternalSystem{N_EXT, T}}

	"""
	NDOrderModel(linear_terms, nonlinear_terms, external_dynamics = DensePolynomial{...}())

	Construct an `NDOrderModel` with external system.

	# Checks performed
	- Correct relationship between `ORD` and `ORDP1`.
	- All matrices in `linear_terms` must be adequately sized.
	"""
	function NDOrderModel(
		linear_terms::NTuple{ORDP1, MT},
		nonlinear_terms::NTuple{N_NL, MultilinearMap{ORD}},
		external_dynamics::DensePolynomial{SVector{N_EXT, T}, N_EXT},
	) where {ORD, ORDP1, N_NL, N_EXT, T, MT <: AbstractMatrix{T}}
		@assert ORDP1 == ORD + 1
		n_fom = size(linear_terms[1], 1)
		@assert all(size(B) == (n_fom, n_fom) for B in linear_terms)

		new{ORD, ORDP1, N_NL, N_EXT, T, MT}(n_fom, linear_terms, nonlinear_terms, ExternalSystem(external_dynamics))
	end

	# Constructor without external system
	function NDOrderModel(
		linear_terms::NTuple{ORDP1, MT},
		nonlinear_terms::NTuple{N_NL, MultilinearMap{ORD}},
	) where {ORD, ORDP1, N_NL, T, MT <: AbstractMatrix{T}}
		@assert ORDP1 == ORD + 1
		n_fom = size(linear_terms[1], 1)
		@assert all(size(B) == (n_fom, n_fom) for B in linear_terms)

		new{ORD, ORDP1, N_NL, 0, T, MT}(n_fom, linear_terms, nonlinear_terms, nothing)
	end
end

"""
	evaluate_nonlinear_terms!(res, model, order, state_vectors, r = nothing)

Evaluate all nonlinear terms of a given polynomial degree for an `NDOrderModel`.

# Arguments
- `res`: output vector (modified in-place)
- `model`: the `NDOrderModel`
- `order`: degree of the nonlinear terms to evaluate
- `state_vectors`: tuple `(x, x^(1), …, x^(ORD-1))` of state derivatives
- `r`: external state vector (default `nothing`). Must be provided if any term uses external variables.
"""
function evaluate_nonlinear_terms!(res, model::NDOrderModel{ORD, ORDP1, N_NL, MT},
	order, state_vectors, r = nothing) where {ORD, ORDP1, N_NL, MT}
	order <= 0 && return res
	@assert length(res) == model.n_fom "Result vector length does not match full‑order state dimension"

	for term in model.nonlinear_terms
		if term.deg == order
			evaluate_term!(res, term, state_vectors, r)
		end
	end
end

"""
	linear_first_order_matrices(model::NDOrderModel)

Construct the matrices A and B of the equivalent linear first-order system:

	B Ẋ = A X

obtained from the ORD-th order model

	B_ORD x^(ORD) + ... + B_1 x^(1) + B_0 x = F(...)

by introducing the augmented state vector

	X = [x, x^(1), ..., x^(ORD-1)].

and the (ORD*n_fom x ORD*n_fom)-block matrices

	B = [ I   0   0   ⋯   0
		  0   I   0   ⋯   0
		  ⋮       ⋱
		  0   0   0   ⋯  B_ORD ]

and

	A = [ 0   I   0   ⋯   0
		  0   0   I   ⋯   0
		  ⋮       ⋱
		 -B₀ -B₁ -B₂ ⋯ -B_{ORD-1} ]

where `I` is the `n_fom × n_fom` identity matrix.
"""
function linear_first_order_matrices(model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT}
) where {ORD, ORDP1, N_NL, N_EXT, T, MT <: SparseMatrixCSC{T}}
	n = model.n_fom
	#T = eltype(model.linear_terms[1])
	total = ORD * n

	B = spzeros(T, total, total)
	A = spzeros(T, total, total)
	Id = sparse(one(T) * I, n, n)

	# --- B matrix ---
	for i in 1:(ORD-1)
		rows = ((i-1)*n+1):(i*n)
		B[rows, rows] .= Id
	end

	# last block
	rows = ((ORD-1)*n+1):(ORD*n)
	B[rows, rows] .= model.linear_terms[end]   # B_ORD

	# --- A matrix ---

	# shift identities
	for i in 1:(N-1)
		rows = ((i-1)*n+1):(i*n)
		cols = (i*n+1):((i+1)*n)
		A[rows, cols] .= Id
	end

	# last row: -B0 ... -B_{ORD-1}
	lastrow = ((ORD-1)*n+1):(ORD*n)

	for i in 1:N
		cols = ((i-1)*n+1):(i*n)
		A[lastrow, cols] .= -model.linear_terms[i]
	end

	return A, B
end

function linear_first_order_matrices(model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, T, MT}
) where {ORD, ORDP1, N_NL, N_EXT, T, MT <: AbstractMatrix{T}}
	n = model.n_fom
	# T = eltype(model.linear_terms[1])
	total = ORD * n

	B = zeros(T, total, total)
	A = zeros(T, total, total)
	Id = Matrix{T}(I, n, n)

	# --- B matrix ---
	for i in 1:(ORD-1)
		rows = ((i-1)*n+1):(i*n)
		B[rows, rows] .= Id
	end

	# last block
	rows = ((ORD-1)*n+1):(ORD*n)
	B[rows, rows] .= model.linear_terms[end]   # B_ORD

	# --- A matrix ---

	# shift identities
	for i in 1:(ORD-1)
		rows = ((i-1)*n+1):(i*n)
		cols = (i*n+1):((i+1)*n)
		A[rows, cols] .= Id
	end

	# last row: -B0 ... -B_{ORD-1}
	lastrow = ((ORD-1)*n+1):(ORD*n)

	for i in 1:ORD
		cols = ((i-1)*n+1):(i*n)
		A[lastrow, cols] .= -model.linear_terms[i]
	end

	return A, B
end

"""
	FirstOrderModel{MT, N_NL} <: AbstractFullOrderModel

Optimised representation of a first‑order dynamical system

	B₁ ẋ + B₀ x = F(x)

where `F(x)` is a polynomial/multilinear function of `x`.

# Fields
- `n_fom`: dimension of the full‑order state vector x.
- `B0`, `B1`: the linear coefficient matrices.
- `nonlinear_terms`: tuple of `MultilinearMap{1}` representing nonlinear contributions.

# Construction
	FirstOrderModel((B0, B1), nonlinear_terms)

`nonlinear_terms` can be any iterable of `MultilinearMap{1}`.
"""
struct FirstOrderModel{MT, N_NL} <: AbstractFullOrderModel
	n_fom::Int
	B0::MT
	B1::MT
	nonlinear_terms::NTuple{N_NL, MultilinearMap{1}}

	function FirstOrderModel(
		linear_terms::NTuple{2, MT},
		nonlinear_terms::NTuple{N_NL, MultilinearMap{1}}) where {
		MT, N_NL}
		B0, B1 = linear_terms
		@assert size(B0)==size(B1) "Linear matrices must have identical size"
		n_fom = size(B0, 1)
		new{MT, N_NL}(n_fom, B0, B1, nonlinear_terms)
	end
end

"""
	evaluate_nonlinear_terms!(res, model::FirstOrderModel, order, state_vectors)

Evaluate all nonlinear terms of given `order` and accumulate into `res`.
`state_vectors` must be a 1‑tuple `(x,)`.
"""
function evaluate_nonlinear_terms!(res, model::FirstOrderModel,
	order::Int, state_vector)
	order <= 1 && return res
	@assert length(res) == model.n_fom "Result vector length does not match full‑order state dimension"

	@inbounds for term in model.nonlinear_terms
		deg = term.deg
		if deg == order
			term.f!(res, ntuple(_ -> state_vector, deg)...)
		end
	end
	return res
end

"""
	linear_first_order_matrices(model::FirstOrderModel)

Return the matrices `(A, B)` of the equivalent linear first‑order system
`B Ẋ = A X`.  Because the model is already first order, `X = x` and

	A = -B₀,    B = B₁.
"""
function linear_first_order_matrices(model::FirstOrderModel)
	return -model.B0, model.B1
end

end # module
