@testset "ParametrisationMethod" begin
    @testset "ParametrisationMethod structs" begin
        @testset "create_parametrisation_method_objects (ORD=1, no forcing)" begin
            # 2-variable multiindex set up to degree 3
            mset = all_multiindices_up_to(2, 3)
            FOM = 4   # full-order dimension
            ROM = 2   # reduced dimension
            W, R = create_parametrisation_method_objects(mset, 1, FOM, ComplexF64)

            @testset "coefficients()" begin
                @test MORFE.ParametrisationMethod.coefficients(W) == W.poly.coefficients
                @test MORFE.ParametrisationMethod.coefficients(R) == R.poly.coefficients
            end

            # Types
            @test W isa Parametrisation{1, 2, ComplexF64}
            @test R isa ReducedDynamics{2, 2, ComplexF64}

            # Sizes
            @test size(W) == FOM
            @test size(R) == ROM

            # Coefficient tensor shapes
            @test size(MORFE.ParametrisationMethod.coefficients(W)) ==
                  (FOM, 1, length(mset))
            @test size(MORFE.ParametrisationMethod.coefficients(R)) == (ROM, length(mset))

            # All coefficients initialised to zero
            @test iszero(MORFE.ParametrisationMethod.coefficients(W))
            @test iszero(MORFE.ParametrisationMethod.coefficients(R))

            # Shared multiindex set
            @test MORFE.ParametrisationMethod.multiindex_set(W) ===
                  MORFE.ParametrisationMethod.multiindex_set(R)
            @test length(MORFE.ParametrisationMethod.multiindex_set(W)) == length(mset)
        end

        @testset "create_parametrisation_method_objects (ORD=2, with forcing)" begin
            mset = all_multiindices_up_to(3, 3)   # 2 master modes + 1 forcing variable
            FOM = 6
            ROM = 2
            N_EXT = 1
            W, R = create_parametrisation_method_objects(
                mset, 2, FOM, ROM, N_EXT, ComplexF64)

            @test W isa Parametrisation{2, 3, ComplexF64}
            @test R isa ReducedDynamics{2, 3, ComplexF64}

            # The ORD=2 axis means W stores W and Ẇ
            @test size(MORFE.ParametrisationMethod.coefficients(W)) ==
                  (FOM, 2, length(mset))
            # ROM (= NVAR - N_EXT = 2) is the first type parameter
            @test size(R) == ROM
            # external_system_size stored correctly
            @test W.external_system_size == N_EXT
            @test R.external_system_size == N_EXT
        end

        @testset "create_parametrisation_method_objects argument check" begin
            mset = all_multiindices_up_to(2, 2)
            # ROM + N_EXT = 3 ≠ NVAR = 2 → should throw
            @test_throws AssertionError create_parametrisation_method_objects(
                mset, 1, 4, 2, 1, ComplexF64)
        end

        @testset "ReducedDynamics ROM > 0 assertion" begin
            # If NVAR = N_EXT the ROM would be zero → should throw
            mset = all_multiindices_up_to(2, 2)
            @test_throws AssertionError create_parametrisation_method_objects(
                mset, 1, 4, 0, 2, ComplexF64)
        end

        @testset "compute_higher_derivative_coefficients!" begin
            # Set up a minimal second-order problem.
            # FOM = 2, ROM = 1 (one master mode), no forcing.
            # We test the recurrence W^(2)[α] = s·W^(1)[α] + Φ·R[α] + ξ[1]
            FOM = 2
            ROM = 1
            NVAR = 1
            L = 3   # three monomials

            param_coeff = zeros(ComplexF64, FOM, 2, L)
            red_coeff = zeros(ComplexF64, ROM, L)
            generalised_eigenmodes = ones(ComplexF64, FOM, NVAR)   # Φ = [[1],[1]]
            external_dynamics = ComplexF64[]

            # Fill the first derivative slice for monomial 2
            k = 2
            W1 = [1.0 + 0.0im, 2.0 + 0.0im]
            param_coeff[:, 1, k] .= W1
            R_k = [0.5 + 0.0im]
            red_coeff[:, k] .= R_k

            # Lower-order coupling vector (ξ[1])
            xi1 = [0.1 + 0.0im, 0.2 + 0.0im]
            lower_order_couplings = [xi1]

            s = 3.0 + 0.0im

            MORFE.ParametrisationMethod.compute_higher_derivative_coefficients!(
                param_coeff, red_coeff, external_dynamics,
                s, k, generalised_eigenmodes, lower_order_couplings)

            # Expected: W^(2)[k] = s·W^(1)[k] + Φ·R[k] + ξ[1]
            #         = 3·[1,2] + [[1],[1]]·[0.5] + [0.1,0.2]
            #         = [3,6] + [0.5,0.5] + [0.1,0.2]
            #         = [3.6, 6.7]
            expected_W2 = s .* W1 .+ generalised_eigenmodes * R_k .+ xi1
            @test param_coeff[:, 2, k]≈expected_W2 rtol=1e-12

            # Other monomials must remain untouched
            @test iszero(param_coeff[:, 2, 1])
            @test iszero(param_coeff[:, 2, 3])
        end

        @testset "compute_higher_derivative_coefficients! noop for ORD=1" begin
            FOM = 3
            ROM = 1
            NVAR = 1
            L = 2
            param_coeff = zeros(ComplexF64, FOM, 1, L)
            red_coeff = ones(ComplexF64, ROM, L)
            eigenmodes = ones(ComplexF64, FOM, NVAR)
            ext_dyn = ComplexF64[]
            xi = [zeros(ComplexF64, FOM)]   # length-1 but ORD-1 = 0 iterations

            param_before = copy(param_coeff)
            MORFE.ParametrisationMethod.compute_higher_derivative_coefficients!(
                param_coeff, red_coeff, ext_dyn, 2.0 + 0im, 1, eigenmodes, xi)
            @test param_coeff == param_before   # must be unchanged
        end
    end
end #@testset "ParametrisationMethod"