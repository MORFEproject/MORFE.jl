using Test
using MORFE
using MORFE.Polynomials: find_in_multiindex_set, linear_matrix_of_polynomial
using LinearAlgebra
using StaticArrays: SVector
using SparseArrays

"""
Build a purely linear vector-valued DensePolynomial representing x -> A*x,
where A is a (N×N) matrix.
"""
function _linear_polynomial(A::AbstractMatrix{T}) where {T}
    N = size(A, 1)
    @assert size(A, 2) == N
    ms = all_multiindices_up_to(N, 1)
    L = length(ms)
    mat = zeros(T, N, L)
    p0 = DensePolynomial(mat, ms)
    for j in 1:N
        e = zeros(Int, N)
        e[j] = 1
        idx = find_in_multiindex_set(p0, e)
        mat[:, idx] = A[:, j]
    end
    return DensePolynomial(mat, ms)
end

"""
Build a nonlinear DensePolynomial representing x -> A*x + quadratic terms.
The quadratic part is just a fixed small perturbation to make it nonlinear.
"""
function _nonlinear_polynomial(A::AbstractMatrix{T}, quad_scale::T = T(0.1)) where {T}
    N = size(A, 1)
    ms = all_multiindices_up_to(N, 2)
    L = length(ms)
    mat = zeros(T, N, L)
    p0 = DensePolynomial(mat, ms)
    for j in 1:N
        e = zeros(Int, N)
        e[j] = 1
        idx = find_in_multiindex_set(p0, e)
        mat[:, idx] = A[:, j]
    end
    # Add a small quadratic term: coefficient for x1² in the first component
    e2 = zeros(Int, N)
    e2[1] = 2
    idx2 = find_in_multiindex_set(p0, e2)
    if !isnothing(idx2)
        mat[1, idx2] = quad_scale
    end
    return DensePolynomial(mat, ms)
end

@testset "ExternalSystem" begin
    @testset "ExternalSystem Constructor 1: from polynomial only" begin
        @testset "1D linear system: ṙ = λr" begin
            λ = -2.0 + 1.0im
            A = reshape([λ], 1, 1)
            poly = _linear_polynomial(A)
            sys = ExternalSystem(poly)

            @test sys.linear_matrix ≈ A
            @test length(sys.eigenvalues) == 1
            @test sys.eigenvalues[1] ≈ λ
        end

        @testset "2D diagonal linear system: eigenvalues recovered correctly" begin
            λ1, λ2 = -1.0 + 2.0im, -1.0 - 2.0im
            A = ComplexF64[λ1 0; 0 λ2]
            poly = _linear_polynomial(A)
            sys = ExternalSystem(poly)

            evs = sort(sys.eigenvalues, by = x -> (real(x), imag(x)))
            expected = sort([λ1, λ2], by = x -> (real(x), imag(x)))
            @test evs ≈ expected
        end

        @testset "2D upper-triangular system: linear matrix stored correctly" begin
            λ1, λ2 = -1.0 + 2.0im, -1.0 - 2.0im
            A = ComplexF64[λ1 5.0; 0 λ2]
            poly = _linear_polynomial(A)
            sys = ExternalSystem(poly)

            @test sys.linear_matrix ≈ A
            @test sys.eigenvalues[1] ≈ λ1
            @test sys.eigenvalues[2] ≈ λ2
        end

        @testset "eigenvalues follow variable order, not LAPACK order" begin
            # `eigvals` reorders even for triangular (and diagonal) matrices, but
            # Resonance._superharmonics contracts eigenvalues against multiindex
            # components position by position, so eigenvalues[e] must be A[e, e].
            λ1, λ2 = -1.0 + 2.0im, -1.0 - 2.0im
            A = ComplexF64[λ1 5.0; 0 λ2]
            @test !isapprox(eigvals(A), diag(A))   # guard: the orders really do differ

            sys = ExternalSystem(_linear_polynomial(A))
            @test collect(sys.eigenvalues) == diag(A)   # exact, not merely ≈
        end

        # ── Re-basing ────────────────────────────────────────────────────────────
        # A non-triangular linear part is repaired by a change of external coordinates
        # rather than rejected.  The property that must hold afterwards is not "close to
        # triangular" but *exactly* triangular, since `istriu` tests exact zeros and the
        # solver silently drops anything below the diagonal.

        @testset "non-upper-triangular linear matrix is re-based, not rejected" begin
            A = [0.0 -1.0; 1.0 0.0]   # rotation generator; not triangular
            sys = ExternalSystem(_linear_polynomial(A))
            Q = external_basis(sys)

            @test Q !== nothing
            @test istriu(sys.linear_matrix)
            # Exactly zero, not merely small: round-off left below the diagonal would make
            # the system non-triangular by the package's own predicate.
            @test all(iszero, tril(Matrix(sys.linear_matrix), -1))
            # Q U Q⁻¹ recovers the matrix we started from.
            @test Matrix(Q) * Matrix(sys.linear_matrix) * inv(Matrix(Q)) ≈ A
            # Same spectrum, as a multiset — the ordering is the new basis's, not the old.
            @test sort(collect(sys.eigenvalues), by = imag) ≈ sort(eigvals(A), by = imag)
            @test collect(sys.eigenvalues) == diag(sys.linear_matrix)
        end

        @testset "a real matrix keeps its conjugate structure exactly" begin
            # `realify` applies one conj_map across all variables, so a re-basing that broke
            # the external conjugate pairing would silently invalidate it.  A real matrix
            # takes the eigenvector route, where LAPACK returns bit-exactly conjugate
            # adjacent pairs — hence `==`, not `≈`.
            A = [0.0 -1.0; 1.0 0.0]
            sys = ExternalSystem(_linear_polynomial(A))
            Q = external_basis(sys)
            σ = external_conjugate_permutation(sys)

            @test σ !== nothing
            for k in eachindex(σ)
                @test Q[:, σ[k]] == conj(Q[:, k])
                @test sys.eigenvalues[σ[k]] == conj(sys.eigenvalues[k])
            end
            # The eigenvector route also diagonalises, which lets the solver take its
            # uncoupled external fast path.
            @test isdiag(Matrix(sys.linear_matrix))
        end

        @testset "lower-triangular linear matrix is re-based" begin
            A = ComplexF64[-1.0 0.0; 3.0 -2.0]
            sys = ExternalSystem(_linear_polynomial(A))

            @test external_basis(sys) !== nothing
            @test istriu(sys.linear_matrix)
            @test sort(real(collect(sys.eigenvalues))) ≈ sort(real(eigvals(A)))
            # Real spectrum ⇒ every variable is its own conjugate.
            @test external_conjugate_permutation(sys) == [1, 2]
        end

        @testset "complex matrix takes the Schur route" begin
            A = ComplexF64[0.0 -1.0; 1.0+0.3im 0.0]
            sys = ExternalSystem(_linear_polynomial(A))
            Q = external_basis(sys)

            @test istriu(sys.linear_matrix)
            @test Matrix(Q)' * Matrix(Q) ≈ I           # Schur basis is unitary
            @test Matrix(Q) * Matrix(sys.linear_matrix) * Matrix(Q)' ≈ A
            # Schur vectors are not eigenvectors, so there is no conjugate pairing to offer.
            @test external_conjugate_permutation(sys) === nothing
        end

        @testset "near-defective real matrix falls back to Schur" begin
            # An eigenvector basis here would have cond(V) ~ 1e10 and corrupt every
            # transformed coefficient, so the conditioning guard must reject it.
            A = ComplexF64[1.0 1.0; 1e-20 1.0]
            sys = ExternalSystem(_linear_polynomial(A))
            Q = external_basis(sys)

            @test istriu(sys.linear_matrix)
            @test Matrix(Q)' * Matrix(Q) ≈ I           # unitary ⇒ Schur was chosen
        end

        @testset "an already-triangular system is left completely untouched" begin
            # The regression guard for every existing caller: nothing about this path may
            # change, down to object identity and element type.
            A = ComplexF64[-1.0 5.0; 0.0 -2.0]
            poly = _linear_polynomial(A)
            sys = ExternalSystem(poly)

            @test external_basis(sys) === nothing
            @test sys.first_order_dynamics === poly        # same object, not a copy
            @test eltype(sys.linear_matrix) === eltype(A)  # element type unpromoted
            @test sys.linear_matrix == A                   # exact, not ≈
            @test collect(sys.eigenvalues) == diag(A)
        end

        @testset "linear_matrix stays consistent with the polynomial" begin
            # `linear_matrix` is documented as the Jacobian at the origin of
            # `first_order_dynamics`.  It is re-derived from the (possibly re-based)
            # polynomial rather than taken from the decomposition, so the two cannot drift.
            for A in (ComplexF64[-1.0 5.0; 0.0 -2.0],       # untouched path
                ComplexF64[0.0 -1.0; 1.0 0.0])              # re-based path
                sys = ExternalSystem(_linear_polynomial(A))
                rederived = linear_matrix_of_polynomial(sys.first_order_dynamics)
                @test Matrix(sys.linear_matrix) == rederived   # exact
            end
        end

        @testset "re-basing transforms the whole polynomial, not just the linear part" begin
            # ṙ′ = U r′ + Q⁻¹ g(Q r′): the higher-order terms must move too, or the stored
            # system would describe different dynamics from the one supplied.
            A = ComplexF64[0.0 -1.0; 2.0 0.0]
            poly = _nonlinear_polynomial(A)
            sys = ExternalSystem(poly)
            Q = Matrix(external_basis(sys))
            Qi = inv(Q)

            err = 0.0
            for _ in 1:100
                r = SVector{2, ComplexF64}(randn(ComplexF64, 2))
                lhs = evaluate(sys.first_order_dynamics, SVector{2, ComplexF64}(Qi * r))
                rhs = Qi * evaluate(poly, r)
                err = max(err, norm(lhs - rhs))
            end
            @test err < 1e-12
        end

        @testset "polynomial field stored correctly" begin
            A = ComplexF64[-1 0; 0 -2]
            poly = _linear_polynomial(A)
            sys = ExternalSystem(poly)

            @test sys.first_order_dynamics == poly
        end

        @testset "nonlinear polynomial: linear matrix = Jacobian at origin" begin
            A = ComplexF64[-1.0 0.0; 0.0 -3.0]
            poly = _nonlinear_polynomial(A)
            sys = ExternalSystem(poly)

            @test sys.linear_matrix ≈ A
        end
    end
    # The `ExternalSystem(first_order_dynamics, eigenvalues)` constructor was removed: it
    # re-derived `linear_matrix` from the polynomial anyway and only *checked* the supplied
    # eigenvalues, so it added a failure mode without adding information — and after a
    # re-basing the supplied ordering describes coordinates that no longer exist.
    @testset "the polynomial + eigenvalues constructor no longer exists" begin
        A = ComplexF64[-1.0 0.0; 0.0 -2.0]
        poly = _linear_polynomial(A)
        evs = SVector{2, ComplexF64}(-1.0 + 0im, -2.0 + 0im)

        @test_throws MethodError ExternalSystem(poly, evs)
    end

    @testset "ExternalSystem Constructor 3: from eigenvalues only (purely linear diagonal)" begin
        @testset "1D: single complex eigenvalue" begin
            λ = -3.0 + 0.5im
            sys = ExternalSystem((λ,))

            @test sys.eigenvalues[1] ≈ λ
            @test sys.linear_matrix ≈ reshape([λ], 1, 1)
        end

        @testset "2D complex conjugate pair" begin
            λ1, λ2 = -1.0 + 2.0im, -1.0 - 2.0im
            sys = ExternalSystem((λ1, λ2))

            @test sys.linear_matrix ≈ Diagonal([λ1, λ2])
            evs = sort(sys.eigenvalues, by = x -> (real(x), imag(x)))
            expected = sort([λ1, λ2], by = x -> (real(x), imag(x)))
            @test evs ≈ expected
        end

        @testset "real eigenvalues are promoted to Complex" begin
            λ1, λ2 = -1.0, -2.0
            sys = ExternalSystem((λ1, λ2))

            @test eltype(sys.eigenvalues) == ComplexF64
            @test real(sys.eigenvalues[1]) ≈ λ1
            @test real(sys.eigenvalues[2]) ≈ λ2
            @test all(iszero ∘ imag, sys.eigenvalues)
        end

        @testset "4D system: diagonal structure preserved" begin
            evals = (-1.0 + 1.0im, -1.0 - 1.0im, -2.0 + 0.5im, -2.0 - 0.5im)
            sys = ExternalSystem(evals)

            A = Matrix(sys.linear_matrix)
            @test diag(A) ≈ collect(evals)
            @test norm(A - Diagonal(diag(A)))≈0.0 atol=1e-14
        end

        @testset "polynomial dynamics evaluate correctly: f(r) = diag(λ)*r" begin
            λ1, λ2 = -2.0 + 0.0im, -3.0 + 0.0im
            sys = ExternalSystem((λ1, λ2))
            r = ComplexF64[1.5, -0.5]

            result = evaluate(sys.first_order_dynamics, r)
            expected = [λ1 * r[1], λ2 * r[2]]
            @test result ≈ expected
        end
    end
    @testset "EigenvalueType inference (_evtype)" begin
        @testset "Real polynomial → Complex eigenvalue type" begin
            A = [-1.0 2.0; 0.0 -3.0]
            poly = _linear_polynomial(A)
            sys = ExternalSystem(poly)

            @test eltype(sys.eigenvalues) == Complex{Float64}
        end

        @testset "Complex polynomial → Complex eigenvalue type (same)" begin
            λ1, λ2 = -1.0 + 2.0im, -1.0 - 2.0im
            A = ComplexF64[λ1 0; 0 λ2]
            poly = _linear_polynomial(A)
            sys = ExternalSystem(poly)

            @test eltype(sys.eigenvalues) == ComplexF64
        end

        @testset "eigenvalue-only constructor with real input → Complex" begin
            sys = ExternalSystem((-1.0, -2.0))
            @test eltype(sys.eigenvalues) == Complex{Float64}
        end

        @testset "eigenvalue-only constructor with complex input → same type" begin
            sys = ExternalSystem((-1.0 + 0.0im, -2.0 + 0.0im))
            @test eltype(sys.eigenvalues) == ComplexF64
        end
    end
    @testset "Consistency across constructors" begin
        @testset "both constructors agree for a diagonal system" begin
            λ1, λ2 = -1.0 + 2.0im, -1.0 - 2.0im
            A = ComplexF64[λ1 0; 0 λ2]
            poly = _linear_polynomial(A)

            sys1 = ExternalSystem(poly)
            sys3 = ExternalSystem((λ1, λ2))

            @test sys1.linear_matrix ≈ sys3.linear_matrix

            # No sorting: both constructors must agree position by position, since
            # eigenvalues[e] is the eigenvalue of external variable e.
            @test sys1.eigenvalues ≈ sys3.eigenvalues

            # Diagonal input ⇒ nothing to re-base, on either path.
            @test external_basis(sys1) === nothing
            @test external_basis(sys3) === nothing
        end

        @testset "linear_matrix matches linear_matrix_of_polynomial" begin
            λ1, λ2 = -2.0 + 1.0im, -2.0 - 1.0im
            A = ComplexF64[λ1 0; 0 λ2]
            poly = _linear_polynomial(A)
            sys = ExternalSystem(poly)

            A_extracted = linear_matrix_of_polynomial(sys.first_order_dynamics)
            @test Matrix(sys.linear_matrix) ≈ A_extracted
        end
    end

    # ── Physical external arguments ──────────────────────────────────────────────
    # The solver works in reduced external coordinates r′ but evaluates terms at the
    # physical r = Q r′.  The conversion lives at the point the argument is materialised,
    # so these two helpers are the whole mechanism — no term is ever wrapped.
    @testset "external arguments are materialised in physical coordinates" begin
        @testset "untouched system yields the plain integer unit vectors" begin
            sys = ExternalSystem((im * 2.0, -im * 2.0))
            v = external_argument_vectors(sys, 2)

            @test v == [SVector{2, Int}(1, 0), SVector{2, Int}(0, 1)]
            @test eltype(eltype(v)) === Int      # exactly what the solver used before
            # And the state form is the identity, not a copy through a matrix.
            r = ComplexF64[3.0, -1.0]
            @test to_physical_external(sys, r) === r
        end

        @testset "a model without an external system still works" begin
            @test external_argument_vectors(nothing, 0) == SVector{0, Int}[]
            @test to_physical_external(nothing, nothing) === nothing
        end

        @testset "re-based system yields the columns of Q" begin
            A = [0.0 -1.0; 1.0 0.0]
            sys = ExternalSystem(_linear_polynomial(A))
            Q = external_basis(sys)
            v = external_argument_vectors(sys, 2)

            # Q eⱼ = Q[:, j], exactly — this is what makes the substitution in the
            # multilinear terms correct without touching the terms.
            for j in 1:2
                @test v[j] == Q[:, j]
            end
            r = SVector{2, ComplexF64}(0.4, -1.3)
            @test to_physical_external(sys, r) ≈ Q * r
        end

        @testset "arguments are isbits statics, so indexing them cannot allocate" begin
            # The vectors are built once per solve, but indexing them happens in the
            # innermost loop, so the elements must stay stack-allocated statics.  That is
            # asserted on the element type rather than with `@allocated(...) == 0`:
            # indexing a `Vector{T}` with isbits `T` cannot touch the heap, whereas the
            # allocation count of a toy call depends on the escape analysis of whichever
            # architecture and Julia version is running — the old assertion held on
            # aarch64 and reported 48 bytes on x86_64.
            sys = ExternalSystem((im * 2.0, -im * 2.0))
            v = external_argument_vectors(sys, 2)
            @test eltype(v) <: SVector{2}
            @test isbitstype(eltype(v))

            # Same for the re-based path, where the elements are columns of an SMatrix.
            rebased = ExternalSystem(_linear_polynomial([0.0 -1.0; 1.0 0.0]))
            w = external_argument_vectors(rebased, 2)
            @test eltype(w) <: SVector{2}
            @test isbitstype(eltype(w))
        end
    end

    @testset "conjugate permutation is derived, not hand-written" begin
        # These are the literals callers write today; the helper must reproduce them, and
        # additionally handle an odd N_EXT, which `ROM + 2k, ROM + 2k - 1` cannot express.
        karman = ExternalSystem((0.0 + 0.0im,))                  # η′ real ⇒ self-conjugate
        dielectric = ExternalSystem((im * 3.0, -im * 3.0))
        odd = ExternalSystem((im * 2.0, -im * 2.0, 0.0 + 0.0im))

        @test full_conjugate_permutation([2, 1], karman) == [2, 1, 3]
        @test full_conjugate_permutation([2, 1], dielectric) == [2, 1, 4, 3]
        @test full_conjugate_permutation([2, 1], odd) == [2, 1, 4, 3, 5]

        # A Schur-re-based system has no conjugate structure, and must say so rather than
        # hand back a permutation that would silently corrupt the conjugate fill.
        schur_sys = ExternalSystem(_linear_polynomial(ComplexF64[0.0 -1.0; 1.0+0.3im 0.0]))
        @test external_conjugate_permutation(schur_sys) === nothing
        @test_throws ArgumentError full_conjugate_permutation([2, 1], schur_sys)
    end
end #@testset "ExternalSystem"
