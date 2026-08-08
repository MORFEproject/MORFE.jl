"""
Module `ExternalSystems` — representation of autonomous external dynamical systems
that drive a full-order model.

An `ExternalSystem` encodes a finite-dimensional autonomous ODE

	ṙ = f(r) = A r + higher-order terms

whose state `r ∈ ℂᴺᴱˣᵀ` appears as a forcing argument in the nonlinear terms
of an `NDOrderModel`.  The module stores the polynomial dynamics together with
the linear matrix `A` and its eigenvalues, which enter the cohomological equations
as external superharmonic frequencies.

# Upper-triangularity, and how it is obtained

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
discarded without trace.

A non-triangular `A` is therefore not rejected but **re-based**: the constructor finds
a basis `Q` in which the linear part is upper triangular, re-expresses the *whole*
polynomial in the new coordinates `r′` where `r = Q r′`,

	ṙ′ = U r′ + Q⁻¹ g(Q r′),        U = Q⁻¹ A Q  upper triangular

and stores `Q` in the `basis` field.  Everything downstream — `W`'s external columns,
the reduced external coordinates, `R`'s external rows — is then expressed in `r′`;
`external_basis` recovers `Q` so results can be mapped back to the physical `r`.
The solver feeds nonlinear terms the *physical* external argument automatically (see
[`external_argument_vectors`](@ref)), so term definitions never need to change.

See [`_triangularising_basis`](@ref) for how `Q` is chosen, and why a real system's
conjugate structure is preserved exactly.
"""
module ExternalSystems

using StaticArrays, LinearAlgebra

export ExternalSystem, external_basis, external_argument_vectors, to_physical_external

using ..Multiindices: all_multiindices_up_to

using ..Polynomials: DensePolynomial, linear_matrix_of_polynomial, compose_linear,
                     coefficients, multiindex_set, find_in_multiindex_set

"""
	ExternalSystem{N_EXT, T, EigenvalueType}

Represents a dynamical system of the form:

	dr/dt = f(r) = A r + higher-order terms

where `r ∈ ℝ^{N_EXT}` (or ℂ^{N_EXT}), `A` is a constant **upper-triangular** matrix, and
the dynamics are given by a polynomial expansion. The structure stores the full
polynomial, the linear matrix, and its eigenvalues.

`A` must be upper triangular — see the module docstring for the GrLex-causality reason.
A caller-supplied polynomial whose linear part is not triangular is re-based rather than
rejected, and `basis` then records the change of coordinates.

Because it is triangular, `A`'s eigenvalues are its diagonal entries, and they are stored
**by variable index**: `eigenvalues[e] == linear_matrix[e, e]` is the eigenvalue that
belongs to external variable `e`.  This is not a sorted spectrum; the ordering carries
meaning, since resonance detection contracts the eigenvalues against multiindex
components position by position.

# Fields

- `first_order_dynamics::DensePolynomial{T, N_EXT, 2}` — the full polynomial
  dynamics, vector-valued: maps ℂ^{N_EXT} → ℂ^{N_EXT}.  In the re-based coordinates
  `r′` when `basis !== nothing`.
- `linear_matrix::SMatrix{N_EXT, N_EXT, T}` — the linear part, i.e. the Jacobian at
  the origin.  Static, since `N_EXT` is small and known at compile time.  Always
  *exactly* upper triangular, and always re-derived from `first_order_dynamics`, so
  the two can never disagree.
- `eigenvalues::SVector{N_EXT, EigenvalueType}` — eigenvalues of `linear_matrix`, i.e.
  its diagonal in variable order, cached because they set the external part of the
  superharmonic `s`.  When `T <: Real` these are still complex, so
  `EigenvalueType = Complex{T}`; otherwise `EigenvalueType = T`.
- `basis::Union{Nothing, SMatrix{N_EXT, N_EXT, T}}` — the change of external
  coordinates `Q` with `r = Q r′`, or `nothing` when the supplied dynamics were
  already triangular and nothing was touched.

# Constructors
1. `ExternalSystem(first_order_dynamics)`
   Build from a polynomial. Computes the linear matrix and associated eigenvalues
   automatically, re-basing first if the linear part is not upper triangular.

2. `ExternalSystem(eigenvalues)`
   Construct a purely linear system `dx/dt = diag(eigenvalues) * x`, i.e., decoupled linear
   dynamics.  Diagonal by construction, so it always satisfies the triangularity requirement.
"""
struct ExternalSystem{N_EXT, T, EigenvalueType}
    first_order_dynamics::DensePolynomial{T, N_EXT, 2}
    linear_matrix::SMatrix{N_EXT, N_EXT, T}
    eigenvalues::SVector{N_EXT, EigenvalueType}
    basis::Union{Nothing, SMatrix{N_EXT, N_EXT, T}}
end

"""
	external_basis(sys) -> Union{Nothing, SMatrix}

The change of external coordinates `Q` with `r = Q r′`, or `nothing` when the system was
left in the coordinates it was given in.  `nothing` is the common case and is what every
consumer branches on; it means "reduced external coordinates *are* the physical ones".
"""
external_basis(sys::ExternalSystem) = sys.basis
external_basis(::Nothing) = nothing

"""
	_evtype(::Type{T}) -> Type

Return the eigenvalue storage type for scalar type `T`: `Complex{T}` when `T <: Real`,
or `T` itself when `T <: Complex`.
"""
_evtype(::Type{T}) where {T <: Real} = Complex{T}
_evtype(::Type{T}) where {T <: Complex} = T

"""
	_unit(::Val{N}, j) -> SVector{N, Int}

The `j`-th unit multiindex / basis vector in `N` variables.
"""
@inline _unit(::Val{N}, j::Int) where {N} = SVector{N, Int}(ntuple(k -> k == j ? 1 : 0, Val(N)))

"""
	_subdiagonal_offenders(linear_matrix) -> Vector{Tuple{Int, Int}}

The `(i, j)` positions of the non-zero entries strictly below the diagonal.

Monomials are solved in GrLex order, so the `|β| = 1` lower-order coupling can only read
`Λ[i, j]` for `i < j` (`LowerOrderCouplings._sum_degree_one_terms!`); a strictly-lower entry
of the external linear matrix would be silently discarded by the solver rather than producing
a wrong-but-visible answer.  Such entries are what triggers a re-basing, and this list is
what the diagnostic reports.  See the `ExternalSystems` module docstring.
"""
function _subdiagonal_offenders(linear_matrix::AbstractMatrix)
    return [(i, j)
            for i in axes(linear_matrix, 1), j in axes(linear_matrix, 2)
            if i > j && !iszero(linear_matrix[i, j])]
end

"""
	_triangularising_basis(A) -> Union{Nothing, NamedTuple}

Choose a basis `Q` in which `Q⁻¹ A Q` is upper triangular, or `nothing` when `A` already is.

Returns `(; Q, U, Qinv, route)` with `route ∈ (:eigen, :schur)`.

# Two requirements, and why the route matters

**Triangularity** is the GrLex-causality constraint (module docstring); both routes give it.

**Conjugate structure** is the second, and only one route gives it.  `Realification.realify`
applies a single `conj_map` across all `NVAR` variables, external ones included, and
`fill_conjugate_monomial!` implements `W_{P·γ} = conj(W_γ)` for a real FOM.  Preserving that
needs an involution `σ` with `Q[:, σ(k)] = conj(Q[:, k])` *and* `λ_{σ(k)} = conj(λ_k)`; a
basis that breaks it silently invalidates any `conjugate_permutation` touching external
indices — a wrong answer, not an error.

A **Schur** basis breaks it in general: in `A = Q U Qᴴ` the first Schur vector `Q[:, 1]` is
an eigenvector for `λ₁`, but `Q[:, 2]` is a Schur vector, not an eigenvector for `λ₂`, so
`Q[:, 2] ≠ conj(Q[:, 1])` unless `A` is normal.

The **eigenvector** basis of a *real* `A` gives both.  LAPACK's `dgeev` returns complex
eigenvalues in adjacent conjugate pairs, each pair's eigenvectors stored as the real and
imaginary parts of one column pair, so `V[:, k+1] == conj(V[:, k])` holds **bit-exactly**;
real eigenvalues get real, self-conjugate eigenvectors.  And `U = Diagonal(λ)` is diagonal,
which is strictly better than triangular: it lets the solver take the uncoupled fast path
(`coupled_external = !isdiag(...)` in `CohomologicalDriver`).

So: real and diagonalisable → `eigen`; anything else (complex `A`, or real but
near-defective, where the eigenvector basis would be catastrophically ill-conditioned) →
complex `schur`.  Note `schur` on a *real* matrix returns the real Schur form, which is only
*quasi*-triangular — it must be given a complex matrix.
"""
function _triangularising_basis(A::SMatrix{N, N, T}) where {N, T}
    istriu(A) && return nothing
    Am = Matrix(A)

    if all(isreal, Am)
        F = eigen(real(Am))
        V, λ = F.vectors, F.values
        # Reject a defective or near-defective basis: `cond(V)` blows up to ~1/eps and the
        # residual stops being at round-off.  Falling through to Schur is the right answer
        # there — a badly conditioned Q would corrupt every transformed coefficient.
        resid = opnorm(real(Am) * V - V * Diagonal(λ))
        scale = max(opnorm(real(Am)), one(real(T)))
        if cond(V) < 1 / sqrt(eps(Float64)) && resid <= sqrt(eps(Float64)) * scale
            TC = promote_type(T, eltype(V))
            return (; Q = SMatrix{N, N, TC}(V),
                U = SMatrix{N, N, TC}(Diagonal(λ)),
                Qinv = inv(V),
                route = :eigen)
        end
    end

    F = schur(complex(Am))
    TC = promote_type(T, eltype(F.Z))
    return (; Q = SMatrix{N, N, TC}(F.Z),
        U = SMatrix{N, N, TC}(F.T),
        Qinv = F.Z',                       # unitary: the adjoint is the exact inverse
        route = :schur)
end

"""
	_zero_subdiagonal_linear_block!(poly; tol)

Zero the strictly-lower triangle of `poly`'s linear block, after checking it is only
round-off.

`compose_linear` produces sub-diagonal entries of size ~`1e-17·‖A‖` where exact arithmetic
would give zero, and `istriu` tests **exact** zeros — so without this the "triangularised"
system is still not triangular by the package's own predicate, and `linear_matrix` would
disagree with the polynomial it is supposed to be the Jacobian of.

`linear_matrix_of_polynomial` sets `A[:, j] = coefficients[:, i_j]` where `i_j` is the mset
position of the unit multiindex `eⱼ`, so `A[k, j]` *is* `coefficients[k, i_j]` and the strict
lower triangle is `k > j` within each unit-multiindex column.

A residual above `tol` means the basis is wrong; that must error rather than be truncated
away, because truncating it would discard genuine coupling.
"""
function _zero_subdiagonal_linear_block!(
        poly::DensePolynomial{T, N_EXT, 2};
        tol::Real = sqrt(eps(real(T)))
) where {T, N_EXT}
    A = linear_matrix_of_polynomial(poly)
    residual = norm(tril(A, -1))
    scale = max(norm(A), eps(real(T)))
    residual <= tol * scale || error("""
       Change of external coordinates did not triangularise the linear part: the strictly \
       lower triangle has norm $(residual), which is not round-off relative to $(scale).
       This means the basis returned by `_triangularising_basis` does not satisfy \
       Q⁻¹ A Q upper triangular.  Refusing to truncate, because discarding these entries \
       would silently drop genuine coupling.
       """)

    for j in 1:N_EXT
        i = find_in_multiindex_set(poly, _unit(Val(N_EXT), j))
        # `compose_linear` builds its multiindex set from the keys it generated, so a linear
        # column that cancels to zero leaves no unit multiindex behind — nothing to zero.
        i === nothing && continue
        @inbounds poly.coefficients[(j + 1):N_EXT, i] .= zero(T)
    end
    return nothing
end

"""
	_rebase(poly, Q, Qinv) -> DensePolynomial

Re-express `ṙ = f(r)` in the coordinates `r′` with `r = Q r′`, giving `ṙ′ = Q⁻¹ f(Q r′)`.

The substitution itself is `compose_linear` (in `Polynomials`); the left multiplication by
`Q⁻¹` is a single matrix product on the coefficient array, since the polynomial is
vector-valued with one row per component.

The promotion on the first line is load-bearing: `compose_linear` types its accumulators
from the *input* polynomial, so a real polynomial composed with a complex `Q` cannot store
the products back.  `promote_type` rather than a blanket `complex` keeps a real `A` with an
all-real spectrum in real arithmetic.
"""
function _rebase(poly::DensePolynomial{T0, N_EXT, 2}, Q, Qinv) where {T0, N_EXT}
    TC = promote_type(T0, eltype(Q))
    poly_c = DensePolynomial(TC.(coefficients(poly)), multiindex_set(poly))
    composed = compose_linear(poly_c, Matrix(Q))
    rebased = DensePolynomial(Qinv * coefficients(composed), multiindex_set(composed))
    _zero_subdiagonal_linear_block!(rebased)
    return rebased
end

function _info_rebased(offenders, route, eigenvalues, promoted::Bool)
    routeline = route === :eigen ?
                "  · route: eigenvector basis (A is real and diagonalisable).  U is diagonal,\n" *
                "    and the conjugate pairing of the external variables is preserved exactly." :
                "  · route: complex Schur (A is complex, or real but near-defective).  Q is unitary;\n" *
                "    the external variables are NOT conjugate-paired in this basis."
    @info "ExternalSystem: the linear matrix was not upper triangular — non-zero entries below\n" *
          "the diagonal at $(offenders).  It has been re-based rather than rejected:\n" *
          routeline * "\n" *
          "  · the external coordinates are now r′, related to the physical r by r = Q r′;\n" *
          "    `external_basis(sys)` returns Q.\n" *
          "  · eigenvalues (diag U, in variable order) = $(eigenvalues)\n" *
          (promoted ?
           "  · the polynomial was promoted to complex arithmetic by the change of basis.\n" :
           "") *
          "  · nonlinear terms are fed the physical external argument automatically, so their\n" *
          "    definitions need no change.\n" *
          "  · W's external columns, R's external rows and the reduced external coordinates\n" *
          "    are all expressed in r′."
end

# Constructor from polynomial only; eigenvalues computed automatically.
#
# `linear_matrix` is always re-derived from the (possibly re-based) polynomial rather than
# taken from the decomposition, so the field and the polynomial it describes cannot drift
# apart by whatever `compose_linear` accumulated in round-off.
function ExternalSystem(first_order_dynamics::DensePolynomial{
        T0, N_EXT, 2}) where {N_EXT, T0}
    A0 = SMatrix{N_EXT, N_EXT, T0}(linear_matrix_of_polynomial(first_order_dynamics))

    basis_choice = _triangularising_basis(A0)
    poly, basis = if basis_choice === nothing
        # Untouched path: same object, same element type, no diagnostic.
        first_order_dynamics, nothing
    else
        rebased = _rebase(first_order_dynamics, basis_choice.Q, basis_choice.Qinv)
        rebased, basis_choice.Q
    end

    T = eltype(coefficients(poly))
    linear_matrix = SMatrix{N_EXT, N_EXT, T}(linear_matrix_of_polynomial(poly))
    @assert istriu(linear_matrix) "external linear matrix must be exactly upper triangular"

    # Eigenvalues of a triangular matrix are its diagonal, in variable order — exact, and
    # correctly paired with the external variables (`eigvals` returns LAPACK's ordering,
    # which permutes even for a diagonal matrix).
    EigenvalueType = _evtype(T)                          # Complex{T} if T<:Real else T
    eigenvalues = SVector{N_EXT, EigenvalueType}(convert.(EigenvalueType, diag(linear_matrix)))

    if basis_choice !== nothing
        _info_rebased(_subdiagonal_offenders(A0), basis_choice.route, eigenvalues, T !== T0)
    end

    ExternalSystem{N_EXT, T, EigenvalueType}(
        poly, linear_matrix, eigenvalues,
        basis === nothing ? nothing :
        SMatrix{N_EXT, N_EXT, T}(basis))
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

    # EigenvalueType = E (since E is already the eigenvalue element type).
    # Diagonal by construction, so there is nothing to re-base.
    ExternalSystem{N_EXT, E, E}(polynomial, linear_matrix, ev_svec, nothing)
end

# Convenience constructor for real eigenvalues (promotes to Complex)
function ExternalSystem(eigenvalues::NTuple{N_EXT, T}) where {N_EXT, T <: Real}
    ExternalSystem(ntuple(i -> Complex{T}(eigenvalues[i]), Val(N_EXT)))
end

# =============================================================================
# Physical external arguments
# =============================================================================
#
# The solver works in *reduced* external coordinates r′; a model's nonlinear terms are
# always evaluated at *physical* external coordinates r = Q r′.  The conversion happens
# exactly where the external argument is materialised — never inside a term, so terms are
# never wrapped or rebuilt and `evaluate_nonlinear_terms!` keeps its existing meaning
# ("r is the physical external state").
#
# There are two shapes of external argument, and one helper each.

"""
	external_argument_vectors(sys, N_EXT) -> Vector{<:SVector{N_EXT}}

The external argument the solver passes for each external variable `j`.

During the cohomological solve the external factor of a monomial is a *basis* vector, not a
state: the multilinear term is evaluated once per external direction and the coefficient is
read off.  In the model's own coordinates that vector is the unit vector `eⱼ`; after a
re-basing it is the physical direction `Q eⱼ = Q[:, j]`.  Multilinearity makes the
substitution exact — `f(…, Qeⱼ, Qe_k) = Σ_{a,b} Q[a,j] Q[b,k] f(…, e_a, e_b)`.

With `basis === nothing` this returns the integer unit vectors, i.e. exactly what the solver
used before re-basing existed.
"""
function external_argument_vectors(sys::Union{Nothing, ExternalSystem}, N_EXT::Int)
    Q = external_basis(sys)
    Q === nothing && return [_unit(Val(N_EXT), j) for j in 1:N_EXT]
    return [Q[:, j] for j in 1:N_EXT]        # SVector columns of an SMatrix
end

"""
	to_physical_external(sys, r) -> r_physical

Map a *reduced* external coordinate vector `r′` to the physical `r = Q r′`.

The state-valued counterpart of [`external_argument_vectors`](@ref), used where an external
argument is a genuine state rather than a basis direction — notably the invariance-error
residual, which samples reduced coordinates and must feed the model physical ones.
Identity when the system was never re-based.
"""
function to_physical_external(sys::Union{Nothing, ExternalSystem}, r)
    ((Q = external_basis(sys)) === nothing ? r : Q * r)
end
to_physical_external(::Union{Nothing, ExternalSystem}, ::Nothing) = nothing

end # module
