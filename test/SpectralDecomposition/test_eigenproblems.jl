using Test
using MORFE
using LinearAlgebra
using SparseArrays
using Random

# Generate spd sparse matrices
function random_spd_matrix(n::Int, density::Float64 = 0.05)
    B = sprandn(n, n, density)
    A = B' * B + I
    return sparse(A)
end

"""
    right_residual(A, B, λ, y)

Compute ||(A - λB)y||
"""
function right_residual(A, B, λ, y)
    return norm((A - λ * B) * y)
end

"""
    left_residual(A, B, λ, x)

Compute ||xᴴ(A - λB)||
"""
function left_residual(A, B, λ, x)
    return norm(x' * (A - λ * B))
    # return norm((A' - conj(λ) * B') * x)
end

@testset "Eigenproblems" begin

    #Create a minimal second order NDOrderModel for testing.
    FOM = 4
    ORD = 2
    K = random_spd_matrix(FOM)
    C = random_spd_matrix(FOM)
    M = random_spd_matrix(FOM)
    model = NDOrderModel((K, C, M))
    A, B = linear_first_order_matrices(model)

    model_dense = NDOrderModel((Matrix(K), Matrix(C), Matrix(M)))
    A_dense, B_dense = linear_first_order_matrices(model_dense)

    # @testset "Solvers" begin
    #     @testset "DefaultEigensolver" begin
    #         ep = solve_eigenproblem(
    #             model_dense;
    #             solver = DefaultEigensolver(),
    #             sorter! = (args...) -> nothing,
    #             normaliser! = (args...) -> nothing)

    #         λs = ep.eigenvalues
    #         Ys = ep.eigenmodes
    #         Xs = ep.left_eigenmodes

    #         neig = length(λs)
    #         @test size(Ys) == (FOM, ORD, neig)
    #         @test size(Xs) == (FOM, ORD, neig)

    #         for k in 1:neig
    #             # reshape 3D mode -> vector
    #             y = vec(Ys[:, :, k])
    #             x = vec(Xs[:, :, k])

    #             r_right = right_residual(A_dense, B_dense, λs[k], y)
    #             r_left = left_residual(A_dense, B_dense, λs[k], x)

    #             @test r_right < 1e-8
    #             @test r_left < 1e-8
    #         end
    #     end

    #     @testset "ArpackEigensolver" begin
    #         nev = 2
    #         ep = solve_eigenproblem(
    #             model;
    #             solver = ArpackEigensolver(nev),
    #             sorter! = (args...) -> nothing,
    #             normaliser! = (args...) -> nothing)

    #         λs = ep.eigenvalues
    #         Ys = ep.eigenmodes
    #         Xs = ep.left_eigenmodes

    #         neig = length(λs)
    #         @test neig == nev
    #         @test size(Ys) == (FOM, ORD, neig)
    #         @test size(Xs) == (FOM, ORD, neig)

    #         for k in 1:neig
    #             # reshape 3D mode -> vector
    #             y = vec(Ys[:, :, k])
    #             x = vec(Xs[:, :, k])

    #             r_right = right_residual(A, B, λs[k], y)
    #             r_left = left_residual(A, B, λs[k], x)

    #             @test r_right < 1e-8
    #             @test r_left < 1e-8
    #         end
    #     end

    #     @testset "MorfeEigensolver" begin
    #         nev = 2
    #         ep = solve_eigenproblem(
    #             model;
    #             solver = MorfeEigensolver(nev, 0.0 + 0.0 * im),
    #             sorter! = (args...) -> nothing,
    #             normaliser! = (args...) -> nothing)

    #         λs = ep.eigenvalues
    #         Ys = ep.eigenmodes
    #         Xs = ep.left_eigenmodes

    #         neig = length(λs)
    #         @test neig == nev
    #         @test size(Ys) == (FOM, ORD, neig)
    #         @test size(Xs) == (FOM, ORD, neig)

    #         for k in 1:neig
    #             # reshape 3D mode -> vector
    #             y = vec(Ys[:, :, k])
    #             x = vec(Xs[:, :, k])

    #             r_right = right_residual(A, B, λs[k], y)
    #             r_left = left_residual(A, B, λs[k], x)

    #             @test r_right < 1e-8
    #             @test r_left < 1e-8
    #         end
    #     end

    #     @testset "StructureModalDampingEigensolver" begin
    #         α = 0.25
    #         β = 0.25
    #         C = α * M + β * K
    #         model_modal_damping = NDOrderModel((K, C, M))
    #         A, B = linear_first_order_matrices(model_modal_damping)

    #         nev = 2
    #         ep = solve_eigenproblem(
    #             model;
    #             solver = StructureModalDampingEigensolver(nev, α, β),
    #             sorter! = (args...) -> nothing,
    #             normaliser! = (args...) -> nothing)

    #         λs = ep.eigenvalues
    #         Ys = ep.eigenmodes
    #         Xs = ep.left_eigenmodes

    #         neig = length(λs)
    #         @test neig == nev * ORD
    #         @test size(Ys) == (FOM, ORD, neig)
    #         @test size(Xs) == (FOM, ORD, neig)

    #         for k in 1:neig
    #             # reshape 3D mode -> vector
    #             y = vec(Ys[:, :, k])
    #             x = vec(Xs[:, :, k])

    #             r_right = right_residual(A, B, λs[k], y)
    #             r_left = left_residual(A, B, λs[k], x)

    #             @test r_right < 1e-8
    #             @test r_left < 1e-8
    #         end
    #     end
    # end

    @testset "Utilities" begin
        @testset "sort_by_magnitude!" begin
            λ = ComplexF64[
                5 + 0im,
                1 + 0im,
                3 + 0im
            ]

            # 2 x 2 x 3
            Φ = zeros(ComplexF64, 2, 2, 3)

            Φ[:, :, 1] .= 10
            Φ[:, :, 2] .= 20
            Φ[:, :, 3] .= 30

            MORFE.Eigenproblems.sort_by_magnitude!(λ, Φ)

            @test λ == ComplexF64[
                1 + 0im,
                3 + 0im,
                5 + 0im
            ]

            # verify modes were permuted consistently
            @test all(Φ[:, :, 1] .== 20)
            @test all(Φ[:, :, 2] .== 30)
            @test all(Φ[:, :, 3] .== 10)
        end

        @testset "sort_left_eigenmodes" begin
            λr = ComplexF64[
                1 + 0im,
                2 + 0im,
                3 + 0im
            ]

            λl = ComplexF64[
                3 + 0im,
                1 + 0im,
                2 + 0im
            ]

            Ψ = zeros(ComplexF64, 2, 2, 3)

            Ψ[:, :, 1] .= 30
            Ψ[:, :, 2] .= 10
            Ψ[:, :, 3] .= 20

            λl, Ψ = MORFE.Eigenproblems.sort_left_eigenmodes(λr, λl, Ψ)

            @test λl == λr

            @test all(Ψ[:, :, 1] .== 10)
            @test all(Ψ[:, :, 2] .== 20)
            @test all(Ψ[:, :, 3] .== 30)
        end

        @testset "normalise_biorthogonal!" begin
            FOM = 4

            K = Matrix(random_spd_matrix(FOM))
            C = Matrix(random_spd_matrix(FOM))
            M = Matrix(random_spd_matrix(FOM))

            model = NDOrderModel((K, C, M))

            _, B = linear_first_order_matrices(model)

            # Test normalise_biorthogonal! directly on raw 3-D arrays so the
            # check is independent of what Eigenproblem chooses to store.
            # Must replicate the sort + match steps from solve_eigenproblem so
            # that left and right eigenvectors correspond index-by-index.
            solver = DefaultEigensolver()
            λ, Y = MORFE.Eigenproblems.solve(model, solver)
            sort_by_magnitude!(λ, Y)
            λ_left, X = MORFE.Eigenproblems.solve_left(model, solver)
            _, X = MORFE.Eigenproblems.sort_left_eigenmodes(λ, λ_left, X)

            normalise_biorthogonal!(model, Y, X)

            neig = length(λ)
            for i in 1:neig
                for j in 1:neig
                    yi = vec(Y[:, :, i])
                    xj = vec(X[:, :, j])

                    val = xj' * B * yi
                    if i == j
                        @test isapprox(val, 1.0 + 0.0im; atol = 1e-8)
                    else
                        @test isapprox(val, 0.0 + 0.0im; atol = 1e-8)
                    end
                end
            end
        end

        # The structural left-block builder is a specialisation of the general
        # recurrence: same formula, with the right position mode standing in for the
        # left slice and `apply = identity` instead of `adjoint`. Both substitutions
        # are licensed by self-adjointness of the pencil (M, K real symmetric,
        # C = αM + βK). These tests pin that equivalence down.
        @testset "left_eigenmode_orders_from_slice ≡ structural specialisation" begin
            n = 6
            K = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
            M = Matrix(1.0I, n, n)
            C = 0.01 * M + 0.002 * K
            λ = ComplexF64[0.3 + 1.1im, 0.3 - 1.1im]
            Y = zeros(ComplexF64, n, 2, 2)
            for k in 1:2
                Y[:, 1, k] .= ComplexF64.(1:n) ./ n
                Y[:, 2, k] .= λ[k] .* Y[:, 1, k]
            end

            structural = MORFE.Eigenproblems._structural_left_eigenmode_orders(λ, Y, M, C)

            # The formula the structural builder is documented to implement:
            #   φ_2 = ϕ,  φ_1 = (conj(λ) M + C) ϕ
            reference = Array{ComplexF64}(undef, n, 2, 2)
            for k in 1:2
                ϕ = view(Y, :, 1, k)
                reference[:, 2, k] .= ϕ
                reference[:, 1, k] .= conj(λ[k]) .* (M * ϕ) .+ C * ϕ
            end
            # Bit-identical, not just ≈: routing through the shared recurrence must
            # not perturb the structural path at all.
            @test structural == reference

            # With exactly symmetric M and C the adjoint route agrees too. It is
            # `apply = identity` that makes this hold *unconditionally* — one ulp of
            # assembly asymmetry is enough to separate them, which is why the
            # structural path pins the operator rather than relying on symmetry.
            @test M == transpose(M) && C == transpose(C)
            adjoint_route = left_eigenmode_orders_from_slice(
                (M, C, M), view(Y, :, 1, :), λ)
            @test adjoint_route == reference
        end
    end
end
