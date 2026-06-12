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

        @testset "2D non-diagonal real system: linear matrix stored correctly" begin
            A = [0.0 -1.0; 1.0 0.0]   # rotation generator; eigenvalues ±i
            poly = _linear_polynomial(A)
            sys = ExternalSystem(poly)

            @test sys.linear_matrix ≈ A
            evs = sort(sys.eigenvalues, by = imag)
            @test evs[1]≈-1.0im atol=1e-12
            @test evs[2]≈1.0im atol=1e-12
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
    @testset "ExternalSystem Constructor 2: from polynomial + precomputed eigenvalues" begin
        @testset "correct eigenvalues accepted without error" begin
            λ1, λ2 = -2.0 + 1.0im, -2.0 - 1.0im
            A = ComplexF64[λ1 0; 0 λ2]
            poly = _linear_polynomial(A)
            evs = SVector{2, ComplexF64}(λ1, λ2)

            # Should not throw
            sys = @test_nowarn ExternalSystem(poly, evs)
            @test sys.eigenvalues ≈ evs
        end
        @testset "wrong eigenvalues raise an error (check=true)" begin
            A = ComplexF64[-1.0 0; 0 -2.0]
            poly = _linear_polynomial(A)
            bad_evs = SVector{2, ComplexF64}(99.0 + 0im, 88.0 + 0im)

            @test_throws ErrorException ExternalSystem(poly, bad_evs; check = true)
        end
        @testset "check=false bypasses eigenvalue validation" begin
            A = ComplexF64[-1.0 0; 0 -2.0]
            poly = _linear_polynomial(A)
            bad_evs = SVector{2, ComplexF64}(99.0 + 0im, 88.0 + 0im)

            # Skipping the check should not throw
            sys = @test_nowarn ExternalSystem(poly, bad_evs; check = false)
            @test sys.eigenvalues == bad_evs
        end

        @testset "linear matrix computed same as constructor 1" begin
            λ1, λ2 = -1.0 + 2.0im, -1.0 - 2.0im
            A = ComplexF64[λ1 0; 0 λ2]
            poly = _linear_polynomial(A)
            evs = SVector{2, ComplexF64}(λ1, λ2)

            sys1 = ExternalSystem(poly)
            sys2 = ExternalSystem(poly, evs)
            @test sys1.linear_matrix ≈ sys2.linear_matrix
        end
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
            A = [0.0 -1.0; 1.0 0.0]
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
        @testset "all three constructors agree for a diagonal system" begin
            λ1, λ2 = -1.0 + 2.0im, -1.0 - 2.0im
            A = ComplexF64[λ1 0; 0 λ2]
            poly = _linear_polynomial(A)
            evs = SVector{2, ComplexF64}(λ1, λ2)

            sys1 = ExternalSystem(poly)
            sys2 = ExternalSystem(poly, evs)
            sys3 = ExternalSystem((λ1, λ2))

            @test sys1.linear_matrix ≈ sys2.linear_matrix
            @test sys1.linear_matrix ≈ sys3.linear_matrix

            sort_key = x -> (real(x), imag(x))
            @test sort(collect(sys1.eigenvalues), by = sort_key) ≈
                  sort(collect(sys2.eigenvalues), by = sort_key)
            @test sort(collect(sys1.eigenvalues), by = sort_key) ≈
                  sort(collect(sys3.eigenvalues), by = sort_key)
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
end #@testset "ExternalSystem"