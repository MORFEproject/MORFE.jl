"""
Module `ExternalSystems` — representation of autonomous external dynamical systems
that drive a full-order model.

An `ExternalSystem` encodes a finite-dimensional autonomous ODE

	ṙ = f(r) = A r + higher-order terms

whose state `r ∈ ℂᴺᴱˣᵀ` appears as a forcing argument in the nonlinear terms
of an `NDOrderModel`.  The module stores the polynomial dynamics together with
the linear matrix `A` and its eigenvalues, which enter the cohomological equations
as external superharmonic frequencies.

# Upper-triangularity requirement

`A` **must be upper triangular**.  The cohomological equations are solved monomial
by monomial in GrLex order, which makes the solve causal: every coefficient a
monomial needs is already available when it is reached.  The `|β| = 1` branch of the
lower-order coupling needs `W[α − eⱼ + eᵢ]`, a coefficient of the *same* total degree
as `α`, and that coefficient precedes `α` in GrLex only when `i < j`.  Accordingly
`LowerOrderCouplings._sum_degree_one_terms!` reads only the strictly upper triangle
of the reduced linear dynamics `Λ`.

`Λ` is block structured: its master block is diagonal, its lower-left block vanishes
because the external system is autonomous, and its lower-right block is `A`.  So `Λ`
is upper triangular exactly when `A` is, and a strictly-lower entry of `A` would be
discarded without trace.  All three constructors reject such an `A`.
"""
module ExternalSystems

using StaticArrays, LinearAlgebra

export ExternalSystem

using ..Multiindices: all_multiindices_up_to

using ..Polynomials: DensePolynomial, linear_matrix_of_polynomial

"""
	ExternalSystem{N_EXT, T, EigenvalueType}

Represents a dynamical system of the form:

	dr/dt = f(r) = A r + higher-order terms

where `r ∈ ℝ^{N_EXT}` (or ℂ^{N_EXT}), `A` is a constant **upper-triangular** matrix, and
the dynamics are given by a polynomial expansion. The structure stores the full
polynomial, the linear matrix, and its eigenvalues.

`A` must be upper triangular — see the module docstring for the GrLex-causality reason.
Because it is triangular, its eigenvalues are its diagonal entries, and they are stored
**by variable index**: `eigenvalues[e] == linear_matrix[e, e]` is the eigenvalue that
belongs to external variable `e`.  This is not a sorted spectrum; the ordering carries
meaning, since resonance detection contracts the eigenvalues against multiindex
components position by position.

# Fields

- `first_order_dynamics::DensePolynomial{T, N_EXT, 2}` — the full polynomial
  dynamics, vector-valued: maps ℂ^{N_EXT} → ℂ^{N_EXT}.
- `linear_matrix::SMatrix{N_EXT, N_EXT, T}` — the linear part, i.e. the Jacobian at
  the origin.  Static, since `N_EXT` is small and known at compile time.
- `eigenvalues::SVector{N_EXT, EigenvalueType}` — eigenvalues of `linear_matrix`, i.e.
  its diagonal in variable order, cached because they set the external part of the
  superharmonic `s`.  When `T <: Real` these are still complex, so
  `EigenvalueType = Complex{T}`; otherwise `EigenvalueType = T`.

# Constructors
1. `ExternalSystem(first_order_dynamics)`
   Build from a polynomial. Computes linear matrix and associated eigenvalues automatically.

2. `ExternalSystem(first_order_dynamics, eigenvalues)`
   Same as above, but with precomputed eigenvalues.

3. `ExternalSystem(eigenvalues)`
   Construct a purely linear system `dx/dt = diag(eigenvalues) * x`, i.e., decoupled linear
   dynamics.  Diagonal by construction, so it always satisfies the triangularity requirement.
"""
struct ExternalSystem{N_EXT, T, EigenvalueType}
    first_order_dynamics::DensePolynomial{T, N_EXT, 2}
    linear_matrix::SMatrix{N_EXT, N_EXT, T}
    eigenvalues::SVector{N_EXT, EigenvalueType}
end

"""
	_evtype(::Type{T}) -> Type

Return the eigenvalue storage type for scalar type `T`: `Complex{T}` when `T <: Real`,
or `T` itself when `T <: Complex`.
"""
_evtype(::Type{T}) where {T <: Real} = Complex{T}
_evtype(::Type{T}) where {T <: Complex} = T

"""
	_check_upper_triangular(linear_matrix)

Throw an `ArgumentError` when `linear_matrix` has a non-zero entry below the diagonal.

Monomials are solved in GrLex order, so the `|β| = 1` lower-order coupling can only read
`Λ[i, j]` for `i < j` (`LowerOrderCouplings._sum_degree_one_terms!`); a strictly-lower entry
of the external linear matrix would be silently discarded by the solver rather than producing
a wrong-but-visible answer.  See the `ExternalSystems` module docstring.
"""
function _check_upper_triangular(linear_matrix::AbstractMatrix)
    istriu(linear_matrix) && return nothing
    offenders = [(i, j)
                 for i in axes(linear_matrix, 1), j in axes(linear_matrix, 2)
                 if i > j && !iszero(linear_matrix[i, j])]
    throw(ArgumentError("""
       The external system's linear matrix must be upper triangular; \
       non-zero entries below the diagonal at $(offenders).
       The cohomological equations are solved monomial by monomial in GrLex order, so the \
       degree-one lower-order coupling can only read Λ[i, j] for i < j.  A strictly-lower \
       entry of the external linear matrix is therefore dropped without trace.
       Reorder the external variables so the coupling runs upwards, or diagonalise the \
       external dynamics before building the ExternalSystem.
       """))
end

# Constructor from polynomial only; eigenvalues computed automatically
function ExternalSystem(first_order_dynamics::DensePolynomial{T, N_EXT, 2}) where {N_EXT, T}
    linear_matrix = SMatrix{N_EXT, N_EXT, T}(linear_matrix_of_polynomial(first_order_dynamics))
    _check_upper_triangular(linear_matrix)

    # Eigenvalues of a triangular matrix are its diagonal, in variable order — exact, and
    # correctly paired with the external variables (`eigvals` returns LAPACK's ordering,
    # which permutes even for a diagonal matrix).
    EigenvalueType = _evtype(T)                          # Complex{T} if T<:Real else T
    eigenvalues = SVector{N_EXT, EigenvalueType}(convert.(EigenvalueType, diag(linear_matrix)))

    ExternalSystem{N_EXT, T, EigenvalueType}(first_order_dynamics, linear_matrix, eigenvalues)
end

# Constructor from polynomial and eigenvalues; compute linear matrix
#
# `check` governs only the eigenvalue-consistency comparison.  Upper-triangularity is a
# structural contract of the solver, so it is enforced unconditionally.
#
# The comparison is elementwise, not a sorted multiset compare: `eigenvalues[e]` must be the
# eigenvalue of external variable `e`, because resonance detection contracts the eigenvalues
# against multiindex components position by position.  A permuted vector is a genuine error.
function ExternalSystem(
        first_order_dynamics::DensePolynomial{T, N_EXT, 2},
        eigenvalues::SVector{N_EXT, EigenvalueType};
        check::Bool = true,
        rtol::Real = 1e-10,
        atol::Real = 1e-12
) where {N_EXT, T, EigenvalueType}
    linear_matrix = SMatrix{N_EXT, N_EXT, T}(linear_matrix_of_polynomial(first_order_dynamics))
    _check_upper_triangular(linear_matrix)

    if check
        actual_ev = SVector{N_EXT, EigenvalueType}(convert.(EigenvalueType, diag(linear_matrix)))
        if !all(isapprox.(actual_ev, eigenvalues, rtol = rtol, atol = atol))
            error("""
                 Provided eigenvalues do not match the eigenvalues of the linear matrix.
                 Expected $(actual_ev) (the diagonal, in variable order), got $(eigenvalues).
                 The ordering is significant: eigenvalues[e] must belong to external variable e.
                 """)
        end
    end

    ExternalSystem{N_EXT, T, EigenvalueType}(first_order_dynamics, linear_matrix, eigenvalues)
end

# Constructor for purely linear, decoupled system: dx/dt = diag(eigenvalues) * x
function ExternalSystem(eigenvalues::NTuple{N_EXT, E}) where {N_EXT, E}
    # Build the polynomial: diagonal coefficient matrix (N_EXT × N_EXT)
    multiindex_set = all_multiindices_up_to(N_EXT, 1)
    deleteat!(multiindex_set.exponents, 1)  # remove zero exponent (constant term)
    coeffs = Matrix{E}(Diagonal(collect(eigenvalues)))
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
