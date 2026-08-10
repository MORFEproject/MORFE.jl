using Test
using MORFE
using LinearAlgebra
using SparseArrays

@testset "FullOrderModel" begin
    @testset "NthOrderModel constructors" begin
        n = 3
        B0 = rand(n, n)
        B1 = rand(n, n)

        # simplest case: no nonlinear terms, no external system
        model = MORFE.FullOrderModel.NthOrderModel((B0, B1))

        @test model.n_fom == n
        @test length(model.linear_terms) == 2
        @test model.nonlinear_terms == ()
        @test model.external_system === nothing
    end

    @testset "linear_first_order_matrices (dense)" begin
        n = 2
        B0 = [1.0 0; 0 1]
        B1 = [2.0 0; 0 2]

        model = MORFE.FullOrderModel.NthOrderModel((B0, B1))

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

        model = MORFE.FullOrderModel.NthOrderModel((B0, B1, B2))  # ORD = 2

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

        model = MORFE.FullOrderModel.NthOrderModel((B0, B1))

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

        model = MORFE.FullOrderModel.NthOrderModel((B0, B1))

        res = zeros(n)

        # order <= 0 → should do nothing
        MORFE.FullOrderModel.evaluate_nonlinear_terms!(res, model, 0, (zeros(n),))

        @test res == zeros(n)
    end

    @testset "external terms require an external system" begin
        n = 3
        K = Matrix{Float64}(I, n, n)
        forcing = MultilinearMap((res, r) -> (res .+= sum(r)), (0, 0), 1)
        quad = MultilinearMap((res, x, y) -> (res .+= x .* y), (2, 0);
            fully_asymmetric = false)

        @testset "rejected without one" begin
            # Previously this constructed fine and only failed mid-solve, inside
            # `evaluate_term!`.
            @test_throws "without an external system" NthOrderModel((K, K, K), (forcing,))
            @test_throws "term 1" NthOrderModel((K, K, K), (forcing,))
            @test_throws "external factor(s)" NthOrderModel((K, K, K), (quad, forcing))
        end

        @testset "accepted with one" begin
            ext = MORFE.ExternalSystems.ExternalSystem((0.0 + 1.0im, 0.0 - 1.0im))
            @test NthOrderModel((K, K, K), (forcing,), ext) isa NthOrderModel
        end

        @testset "purely internal terms are unaffected" begin
            @test NthOrderModel((K, K, K), (quad,)) isa NthOrderModel
        end
    end

    @testset "implicit-symmetry @info" begin
        n = 3
        K = Matrix{Float64}(I, n, n)
        f!(res, x, y) = (res .+= x .* y)

        @testset "fires when fully_asymmetric is unset" begin
            # multiindex = (2,) implies f! is symmetric in its two slots — an assumption
            # the caller may not have realised it was making.
            term = MultilinearMap(f!, (2,))
            @test_logs (:info, r"did not set `fully_asymmetric`") NthOrderModel(
                (K, K), (term,))
        end

        @testset "silent when fully_asymmetric is acknowledged" begin
            # `@test_logs` with no pattern asserts nothing at Info or above is emitted.
            for fa in (false, true)
                term = MultilinearMap(f!, (2,); fully_asymmetric = fa)
                @test_logs NthOrderModel((K, K), (term,))
            end
        end

        @testset "silent when the multiindex implies no symmetry" begin
            g!(res, x, xd) = (res .+= x .* xd)
            term = MultilinearMap(g!, (1, 1))
            @test_logs NthOrderModel((K, K, K), (term,))
        end
    end

    @testset "show" begin
        n = 4
        B0 = Matrix{Float64}(I, n, n)
        cubic = MultilinearMap((res, x, y, z) -> (res .+= x .* y .* z), (3, 0))
        model = NthOrderModel((B0, 0.01 * B0, B0), (cubic,))

        verbose = sprint(show, MIME"text/plain"(), model)
        @test occursin("NthOrderModel{ORD = 2, N_EXT = 0}", verbose)
        @test occursin("FOM = 4", verbose)
        @test occursin("dense", verbose)
        @test occursin("(deg 3)", verbose)
        @test occursin("external  : none", verbose)
        # The point of the method: a term's own fields must not be dumped (for a FEM
        # backend that would drag the whole DofHandler into the REPL).
        @test !occursin("f!", verbose)

        sparse_model = NthOrderModel(
            (sparse(B0), sparse(0.01 * B0), sparse(B0)), (cubic,))
        @test occursin("sparse", sprint(show, MIME"text/plain"(), sparse_model))

        compact = sprint(show, model)
        @test occursin("NthOrderModel{ORD=2, N_EXT=0}", compact)
        @test occursin("FOM=4", compact)
        @test occursin("1 nonlinear term", compact)
        @test !occursin('\n', compact)
    end
end # @testset "FullOrderModel"
