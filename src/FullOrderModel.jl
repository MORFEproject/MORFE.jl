module FullOrderModel

using LinearAlgebra
using SparseArrays
using StaticArrays: SVector

using ..Polynomials: DensePolynomial, evaluate

export NDOrderModel, FirstOrderModel,
	MultilinearMap,
	linear_first_order_matrices, evaluate_nonlinear_terms!

abstract type AbstractFullOrderModel end

"""
	MultilinearMap{N, F}

Represents a single monomial term of order deg in the nonlinear function of an `NDOrderModel`.

A term is represented using a multiindex stored in the NTuple 
	`multiindex` = (i_0, ..., i_{N-1})  
where i_k is the multiplicity of the derivative x^(k). So the i_k specifies how many times the derivative x^(k) appears as an argument. 
In addition the function accepts `multiplicity_external` external variables r_1, r_2, ...
which satisfy the first order dynamic system r' = dynamics_external(r),
where dynamics_external is a DensePolynomial defined in NDOrderModel. The influence in f! is described by `multiplicity_external`

During evaluation the multilinear map is called as

	f!(res,
	   x^(0), ... repeated i_0 times,
	   x^(1), ... repeated i_1 times,
	   ...
	   x^(N-1), ...repeated i_{N-1} times,
	   r, ... repeated `multiplicity_external` times)

# Important Notes
- Each `MultilinearMap` **must implement a multilinear map**, i.e., it should be linear in each of its arguments independently.
- The function `f!` accumulates (adds) into `res` and must be callable with the appropriate number of arguments.
- If one i_k is larger than 1 we assume the input arguments are symmetric by permutation. For example:
	multiindex = (0, 2,...)
	f!(res, x^(1)_1, x^(1)_2, ...) = f!(res, x^(1)_2, x^(1)_1, ...)

"""
struct MultilinearMap{N, F}
	f!::F
	multiindex::NTuple{N, UInt8}
	multiplicity_external::UInt8
	deg::UInt8
end

"""
	MultilinearMap{N}(f, multiindex)

Create a multilinear term for a system of order N without external dynamics.

# Arguments
- `f!`: in-place evaluation function
- `multiindex`: tuple specifying which derivatives are used (values are `UInt8`)
"""
function MultilinearMap(f!, multiindex::NTuple{N, <:Integer}) where {N}
	mi = convert.(UInt8, multiindex)
	deg = sum(mi)
	# Check if input arguments of f matches deg
	ms = methods(f!)
	@assert length(ms)==1 "Function $(f!) must have exactly one method to determine number of inputs"
	@assert ms[1].nargs==deg + 2 "Function $(f!) must accept $(deg+1) arguments (`res` and $deg inputs) instead of $(ms[1].nargs - 1)"
	@assert deg>=2 "Function $(f!) must have degree at least 2, but has degree $deg"

	return MultilinearMap{N, typeof(f!)}(f!, mi, UInt8(0), deg)
end

# Create a multilinear term for a first order system.
function MultilinearMap(f!)
	ms = methods(f!)
	@assert length(ms)==1 "Function $(f!) must have exactly one method to determine number of inputs"
	deg = ms[1].nargs - 2 # subtract the function itself and `res`
	@assert deg>=2 "Function $(f!) must have degree at least 2, but has degree $deg"
	@assert deg <= 255 "Degree $deg is too large for UInt8"
	multiindex = (UInt8(deg),)
	return MultilinearMap{1, typeof(f!)}(f!, multiindex, UInt8(0), UInt8(deg))
end

function MultilinearMap(
	f!, multiindex::NTuple{N, <:Integer}, multiplicity_external::Integer) where {N}
	mi = convert.(UInt8, multiindex)
	me = convert(UInt8, multiplicity_external)
	deg = sum(mi) + me
	# Check if input arguments of f matches deg
	ms = methods(f!)
	@assert length(ms)==1 "Function $(f!) must have exactly one method to determine number of inputs"
	@assert ms[1].nargs==deg + 2 "Function $(f!) must accept $(deg+1) arguments (`res` and $deg inputs) instead of $(ms[1].nargs - 1)"
	@assert (deg>=2) || (me>=1) "Function $(f!) does not depend the external state, hence it must have degree at least 2, but it has degree $deg"

	return MultilinearMap{N, typeof(f!)}(f!, mi, me, deg)
end

"""
	evaluate_term!(res, term, xs, r)

Evaluate a single `MultilinearMap` and accumulate (adds) the result into `res`.

# Arguments
- `res`: output vector (modified in-place)
- `term`: multilinear term
- `xs`: tuple `(x, x^(1), …, x^(N-1))` of state derivatives
- `r`: external state vector (or `nothing` if not used). If `r` is `nothing` but the term expects external arguments, an error is thrown.
"""
@inline function evaluate_term!(res, term::MultilinearMap{N}, xs, r) where {N}
	inds = term.multiindex
	me = term.multiplicity_external
	total_args = term.deg

	# Build the argument list
	args = ntuple(total_args) do k
		if k <= sum(inds)
			# Pick from xs based on multiindex
			s = 0
			for j in 1:N
				s += inds[j]
				if k ≤ s
					return @inbounds xs[j]
				end
			end
		else
			# Pick from external state
			if r === nothing
				error("Term expects external arguments but no external state provided")
			end
			return r
		end
	end
	term.f!(res, args...)
end

"""

	NDOrderModel{N, NP1, N_NL, MT} <: AbstractFullOrderModel

Representation of an N-th (NP1=N+1) order dynamical system of the form

	B_N x^(N) + ... + B_1 x^(1) + B_0 x = F(x^(N-1), …, x^(1), x, r, …, r)

where:
- x^(n) is the n-th derivative of x (x^(n) = d_t^n x)
- x^(0) = x is the state vector
- B_i are the coefficient matrices
- F is a multilinear polynomial function of the derivatives and the external state vector r
- The external state r satisfies its own first‑order dynamics r' = g(r)

# Generic type parameters

- `N` defines the order of the ODE.

- `NP1` is the number of linear terms (from 0 through N). It must satisfy NP1 == N+1.

- `N_NL` is the number of nonlinear terms in the tuple nonlinear_terms.

- `MT` is the type of the matrices in the NP1-tuple linear_terms.

# Fields

- `n_fom`: dimension of the full‑order state vector x.
- `n_ext`: dimension of the external forcing state r (0 if no forcing).
- `linear_terms`: tuple of linear coefficient matrices (B_0, …, B_N).
- `nonlinear_terms`: tuple of `MultilinearMap` representing nonlinear contributions.
- `external_dynamics`: polynomial defining the dynamics of external forcing variables r.

# Representation

- Linear terms are stored as a tuple `(B_0, …, B_N)`
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
struct NDOrderModel{N, NP1, N_NL, MT <: AbstractMatrix} <: AbstractFullOrderModel
	n_fom::Int
	n_ext::Int
	linear_terms::NTuple{NP1, MT}
	nonlinear_terms::NTuple{N_NL, MultilinearMap{N}}
	external_dynamics::DensePolynomial{<:SVector, 1}

	"""
	NDOrderModel(linear_terms, nonlinear_terms, external_dynamics = DensePolynomial{...}())

	Construct an `NDOrderModel` and validate consistency.

	# Checks performed
	- All matrices in `linear_terms` must have identical size.
	- Correct relationship between `N` and `NP1`.
	- The external dynamics polynomial must be of type DensePolynomial{<:SVector, 1}.
	- The external state dimension is extracted from the polynomial coefficients.
	- For each nonlinear term, `multiplicity_external ≤ n_ext`.
	"""
	function NDOrderModel(
		linear_terms::NTuple{NP1, MT},
		nonlinear_terms::NTuple{N_NL, MultilinearMap{N}},
		external_dynamics::DensePolynomial{<:SVector, 1},
	) where {N, NP1, N_NL, MT <: AbstractMatrix}
		@assert NP1 == N + 1
		n_fom = size(linear_terms[1], 1)
		@assert all(size(B) == (n_fom, n_fom) for B in linear_terms)

		# Determine external state dimension from external_dynamics coefficients
		if isempty(external_dynamics.coefficients)
			n_ext = 0
		else
			# All coefficients are SVector; take the length of the first coefficient
			n_ext = length(external_dynamics.coefficients[1])
		end

		# Consistency: any nonlinear term that uses external variables must have multiplicity_external ≤ n_ext
		for term in nonlinear_terms
			@assert term.multiplicity_external <= n_ext "Term uses $(term.multiplicity_external) external variables, but model has only $n_ext"
		end

		new{N, NP1, N_NL, MT}(n_fom, n_ext, linear_terms, nonlinear_terms, external_dynamics)
	end
end

"""
	evaluate_nonlinear_terms!(res, model, order, state_vectors, r = nothing)

Evaluate all nonlinear terms of a given polynomial degree for an `NDOrderModel`.

# Arguments
- `res`: output vector (modified in-place)
- `model`: the `NDOrderModel`
- `order`: degree of the nonlinear terms to evaluate
- `state_vectors`: tuple `(x, x^(1), …, x^(N-1))` of state derivatives
- `r`: external state vector (default `nothing`). Must be provided if any term uses external variables.
"""
function evaluate_nonlinear_terms!(res, model::NDOrderModel{N, NP1, N_NL, MT},
	order, state_vectors, r = nothing) where {N, NP1, N_NL, MT}
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

obtained from the N-th order model

	B_N x^(N) + ... + B_1 x^(1) + B_0 x = F(...)

by introducing the augmented state vector

	X = [x, x^(1), ..., x^(N-1)].

and the (N*n x N*n)-block matrices

	B = [ I   0   0   ⋯   0
		  0   I   0   ⋯   0
		  ⋮       ⋱
		  0   0   0   ⋯  B_N ]

and

	A = [ 0   I   0   ⋯   0
		  0   0   I   ⋯   0
		  ⋮       ⋱
		 -B₀ -B₁ -B₂ ⋯ -B_{N-1} ]

where `I` is the `n × n` identity matrix.
"""
function linear_first_order_matrices(model::NDOrderModel{
	N, NP1, N_NL, MT}) where {N, NP1, N_NL, MT <: SparseMatrixCSC}
	n = model.n_fom
	T = eltype(model.linear_terms[1])
	total = N * n

	B = spzeros(T, total, total)
	A = spzeros(T, total, total)
	Id = sparse(one(T) * I, n, n)

	# --- B matrix ---
	for i in 1:(N-1)
		rows = ((i-1)*n+1):(i*n)
		B[rows, rows] .= Id
	end

	# last block
	rows = ((N-1)*n+1):(N*n)
	B[rows, rows] .= model.linear_terms[end]   # B_N

	# --- A matrix ---

	# shift identities
	for i in 1:(N-1)
		rows = ((i-1)*n+1):(i*n)
		cols = (i*n+1):((i+1)*n)
		A[rows, cols] .= Id
	end

	# last row: -B0 ... -B_{N-1}
	lastrow = ((N-1)*n+1):(N*n)

	for i in 1:N
		cols = ((i-1)*n+1):(i*n)
		A[lastrow, cols] .= -model.linear_terms[i]
	end

	return A, B
end

function linear_first_order_matrices(model::NDOrderModel{
	N, NP1, N_NL, MT}) where {N, NP1, N_NL, MT <: AbstractMatrix}
	n = model.n_fom
	T = eltype(model.linear_terms[1])
	total = N * n

	B = zeros(T, total, total)
	A = zeros(T, total, total)
	Id = Matrix{T}(I, n, n)

	# --- B matrix ---
	for i in 1:(N-1)
		rows = ((i-1)*n+1):(i*n)
		B[rows, rows] .= Id
	end

	# last block
	rows = ((N-1)*n+1):(N*n)
	B[rows, rows] .= model.linear_terms[end]   # B_N

	# --- A matrix ---

	# shift identities
	for i in 1:(N-1)
		rows = ((i-1)*n+1):(i*n)
		cols = (i*n+1):((i+1)*n)
		A[rows, cols] .= Id
	end

	# last row: -B0 ... -B_{N-1}
	lastrow = ((N-1)*n+1):(N*n)

	for i in 1:N
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
