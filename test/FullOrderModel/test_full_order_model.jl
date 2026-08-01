using Test
using MORFE
using LinearAlgebra
using SparseArrays

@testset "FullOrderModel" begin
    @testset "NDOrderModel constructors" begin
        n = 3
        B0 = rand(n, n)
        B1 = rand(n, n)

        # simplest case: no nonlinear terms, no external system
        model = MORFE.FullOrderModel.NDOrderModel((B0, B1))

        @test model.n_fom == n
        @test length(model.linear_terms) == 2
        @test model.nonlinear_terms == ()
        @test model.external_system === nothing
    end

    @testset "linear_first_order_matrices (dense)" begin
        n = 2
        B0 = [1.0 0; 0 1]
        B1 = [2.0 0; 0 2]

        model = MORFE.FullOrderModel.NDOrderModel((B0, B1))

        A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)

        @test size(A) == (2, 2)
        @test size(B) == (2, 2)

        @test A ≈ -B0
        @test B ≈ B1
    end

    @testset "linear_first_order_matrices (higher order)" begin
        n = 2

        B0 = 1.0 * Matrix(I, n, n)
        B1 = 2.0 * Matrix(I, n, n)
        B2 = 3.0 * Matrix(I, n, n)

        model = MORFE.FullOrderModel.NDOrderModel((B0, B1, B2))  # ORD = 2

        A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)

        @test size(A) == (2n, 2n)
        @test size(B) == (2n, 2n)

        # Check last block of B
        @test B[(n + 1):end, (n + 1):end] ≈ B2

        # Check last row of A
        @test A[(n + 1):end, 1:n] ≈ -B0
        @test A[(n + 1):end, (n + 1):end] ≈ -B1
    end

    @testset "linear_first_order_matrices (sparse)" begin
        n = 2

        B0 = 1.0 * sparse(I, n, n)
        B1 = 2.0 * sparse(I, n, n)

        model = MORFE.FullOrderModel.NDOrderModel((B0, B1))

        A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)

        @test issparse(A)
        @test issparse(B)

        @test A ≈ -Matrix(B0)
        @test B ≈ Matrix(B1)
    end

    @testset "FirstOrderModel basics" begin
        n = 3
        B0 = rand(n, n)
        B1 = rand(n, n)

        model = MORFE.FullOrderModel.FirstOrderModel((B0, B1), ())

        @test model.n_fom == n
        @test model.B0 == B0
        @test model.B1 == B1
    end

    @testset "FirstOrderModel linear matrices" begin
        n = 2
        B0 = [1.0 0; 0 1]
        B1 = [2.0 0; 0 2]

        model = MORFE.FullOrderModel.FirstOrderModel((B0, B1), ())

        A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)

        @test A ≈ -B0
        @test B ≈ B1
    end

    @testset "evaluate_nonlinear_terms! no-op cases" begin
        n = 3
        B0 = rand(n, n)
        B1 = rand(n, n)

        model = MORFE.FullOrderModel.NDOrderModel((B0, B1))

        res = zeros(n)

        # order <= 0 → should do nothing
        MORFE.FullOrderModel.evaluate_nonlinear_terms!(res, model, 0, (zeros(n),))

        @test res == zeros(n)
    end
end # @testset "FullOrderModel"
