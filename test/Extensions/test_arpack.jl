using Arpack
using LinearMaps

@testset "generalised eigenpairs" begin
    A = Diagonal(ComplexF64[1, 2, 3, 4, 5, 6])
    B = Diagonal(ComplexF64[1, 1, 2, 2, 1, 1])

    unshifted = generalised_eigenpairs(A, B; nev = 2, which = :LM,
        sort_largest_real = true)
    @test unshifted.nconv == 2
    @test sort(real.(unshifted.values)) ≈ [5.0, 6.0]
    for i in eachindex(unshifted.values)
        @test norm((A - unshifted.values[i] * B) * unshifted.vectors[:, i]) < 1e-8
    end

    shifted = generalised_eigenpairs(A, B; nev = 2, shift = 1.6 + 0im)
    @test shifted.nconv == 2
    @test sort(real.(shifted.values)) ≈ [1.5, 2.0] atol=1e-8
    @test_throws AssertionError generalised_eigenpairs(A, B[1:5, 1:5]; nev = 2)
end

@testset "Arpack right and left modes" begin
    n = 18
    K = spdiagm(0 => collect(1.0:n))
    C = spdiagm(0 => fill(0.08, n))
    M = spdiagm(0 => ones(n))
    model = NthOrderModel((K, C, M))
    solver = ArpackEigensolver(2)
    λ, Y = eigensolve(model, solver)
    λl, X = eigensolve_left(model, solver)
    A, B = linear_first_order_matrices(model)
    @test size(Y) == size(X) == (n, 2, 2)
    @test sort(λl; by = abs) ≈ sort(λ; by = abs) rtol=1e-7
    for i in eachindex(λ)
        @test norm((A - λ[i] * B) * vec(Y[:, :, i])) < 1e-7
        @test norm(vec(X[:, :, i])' * (A - λl[i] * B)) < 1e-7
    end
end

@testset "structural modal damping" begin
    M = spdiagm(0 => ones(5))
    K = spdiagm(0 => [1.0, 4.0, 9.0, 16.0, 25.0])
    under = StructureModalDampingEigensolver(2, 0.02, 0.01)
    λu, Yu = eigensolve(M, K, under)
    @test size(Yu) == (5, 2, 4)
    @test all(imag.(λu[1:2:end]) .> 0)
    @test λu[2:2:end] ≈ conj.(λu[1:2:end])

    over = StructureModalDampingEigensolver(1, 4.0, 0.0)
    λo, _ = eigensolve(M, K, over)
    @test all(iszero, imag.(λo))

    C = under.α * M + under.β * K
    ep = spectrum(NthOrderModel((K, C, M)), under; sorter! = (args...) -> nothing)
    @test size(ep.left_eigenmodes_orders) == size(ep.eigenmodes)
end
