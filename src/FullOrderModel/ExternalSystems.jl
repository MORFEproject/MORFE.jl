module ExternalSystems

using StaticArrays, LinearAlgebra

export ExternalSystem

using ..Multiindices: all_multiindices_up_to

using ..Polynomials: DensePolynomial, linear_matrix_of_polynomial

"""
	ExternalSystem{N_EXT, T, EigenvalueType}

Represents a dynamical system of the form:

	dr/dt = f(r) = A r + higher-order terms

where `r ∈ ℝ^{N_EXT}` (or ℂ^{N_EXT}), `A` is a constant matrix, and the dynamics are
given by a polynomial expansion. The structure stores the full polynomial,
the linear matrix, and its eigenvalues.

# Fields
- `first_order_dynamics::DensePolynomial{T, N_EXT}` – the full polynomial dynamics (vector-valued: maps ℂ^{N_EXT} → ℂ^{N_EXT}).
- `linear_matrix::SMatrix{N_EXT, N_EXT, T}` – the linear part (Jacobian at the origin).
- `eigenvalues::SVector{N_EXT, EigenvalueType}` – eigenvalues of the linear matrix.
  If `T<:Real`, `EigenvalueType = Complex{T}`; otherwise `EigenvalueType = T`.

# Constructors
1. `ExternalSystem(first_order_dynamics)`
   Build from a polynomial. Computes linear matrix and associated automatically.

2. `ExternalSystem(first_order_dynamics, eigenvalues)`
   Same as above, but with precomputed eigenvalues.

3. `ExternalSystem(eigenvalues)`
   Construct a purely linear system `dx/dt = diag(eigenvalues) * x`, i.e., decoupled linear dynamics.
"""
struct ExternalSystem{N_EXT, T, EigenvalueType}
	first_order_dynamics::DensePolynomial{T, N_EXT}
	linear_matrix::SMatrix{N_EXT, N_EXT, T}
	eigenvalues::SVector{N_EXT, EigenvalueType}
end

# Helper type alias for eigenvalue storage
_evtype(::Type{T}) where {T <: Real} = Complex{T}
_evtype(::Type{T}) where {T <: Complex} = T

# Constructor from polynomial only; eigenvalues computed automatically
function ExternalSystem(first_order_dynamics::DensePolynomial{SVector{N_EXT, T}, N_EXT}) where {N_EXT, T}
	linear_matrix = SMatrix{N_EXT, N_EXT, T}(linear_matrix_of_polynomial(first_order_dynamics))

	# Compute eigenvalues of the linear matrix
	evals = eigvals(Matrix(linear_matrix))                # returns Vector{ComplexF64} or similar
	EigenvalueType = _evtype(T)                          # Complex{T} if T<:Real else T
	eigenvalues = SVector{N_EXT, EigenvalueType}(convert.(EigenvalueType, evals))

	ExternalSystem{N_EXT, T, EigenvalueType}(first_order_dynamics, linear_matrix, eigenvalues)
end

# Constructor from polynomial and eigenvalues; compute linear matrix 
function ExternalSystem(
	first_order_dynamics::DensePolynomial{SVector{N_EXT, T}, N_EXT},
	eigenvalues::SVector{N_EXT, EigenvalueType};
	check::Bool = true,
	rtol::Real = 1e-10,
	atol::Real = 1e-12,
) where {N_EXT, T, EigenvalueType}
	linear_matrix = SMatrix{N_EXT, N_EXT, T}(linear_matrix_of_polynomial(first_order_dynamics))

	if check
		actual = eigvals(Matrix(linear_matrix))
		actual_ev = SVector{N_EXT, EigenvalueType}(actual)
		if !all(isapprox.(sort(actual_ev, by = x -> (real(x), imag(x))),
			sort(eigenvalues, by = x -> (real(x), imag(x))),
			rtol = rtol, atol = atol))
			error("Provided eigenvalues do not match the eigenvalues of the linear matrix.")
		end
	end

	ExternalSystem{N_EXT, T, EigenvalueType}(first_order_dynamics, linear_matrix, eigenvalues)
end

# Constructor for purely linear, decoupled system: dx/dt = diag(eigenvalues) * x
function ExternalSystem(eigenvalues::NTuple{N_EXT, E}) where {N_EXT, E}
	# Build the polynomial: each coordinate gets its own linear term
	coeffs = [SVector{N_EXT, E}(ntuple(k -> k == j ? eigenvalues[j] : zero(E), Val(N_EXT))) for j in 1:N_EXT]
	# Multi-index set containing all monomials of degree ≤ 1
	multiindex_set = all_multiindices_up_to(N_EXT, 1)
	deleteat!(multiindex_set.exponents, 1)  # remove zero exponent (constant term)
	polynomial = DensePolynomial(coeffs, multiindex_set)

	linear_matrix = SMatrix{N_EXT, N_EXT, E}(Diagonal(collect(eigenvalues)))
	ev_svec = SVector{N_EXT, E}(eigenvalues)

	# EigenvalueType = E (since E is already the eigenvalue element type)
	ExternalSystem{N_EXT, E, E}(polynomial, linear_matrix, ev_svec)
end

# Convenience constructor for real eigenvalues (promotes to Complex)
function ExternalSystem(eigenvalues::NTuple{N_EXT, T}) where {N_EXT, T <: Real}
	ExternalSystem(ntuple(i -> Complex{T}(eigenvalues[i]), Val(N_EXT)))
end

end # module
