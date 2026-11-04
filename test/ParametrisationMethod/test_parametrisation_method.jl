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
            # ORD = 2, FOM = 2, one master mode, no forcing. Position 1 is the unit
            # monomial e₁, whose level-1 coefficient Φ is read in place; k = 2 is a
            # higher monomial: W^(2)[k] = s·W^(1)[k] + Φ·R[k] + ξ[1].
            PM = MORFE.ParametrisationMethod
            FOM, L, k = 2, 3, 2
            param_coeff = zeros(ComplexF64, FOM, 2, L)
            Φ = ComplexF64[1.0, 1.0]
            param_coeff[:, 1, 1] .= Φ
            W1 = ComplexF64[1.0, 2.0]
            param_coeff[:, 1, k] .= W1
            red_coeff = zeros(ComplexF64, 1, L)
            red_coeff[1, k] = 0.5
            xi1 = ComplexF64[0.1, 0.2]
            s = 3.0 + 0.0im

            PM.compute_higher_derivative_coefficients!(
                param_coeff, red_coeff, ComplexF64[], s, k, [xi1])

            # = 3·[1, 2] + [1, 1]·0.5 + [0.1, 0.2] = [3.6, 6.7]
            @test param_coeff[:, 2, k]≈s .* W1 .+ Φ .* 0.5 .+ xi1 rtol=1e-12
            @test iszero(param_coeff[:, 2, 1])   # other monomials untouched
            @test iszero(param_coeff[:, 2, 3])
        end

        @testset "compute_higher_derivative_coefficients! noop for ORD=1" begin
            param_coeff = zeros(ComplexF64, 3, 1, 2)
            param_before = copy(param_coeff)
            MORFE.ParametrisationMethod.compute_higher_derivative_coefficients!(
                param_coeff, ones(ComplexF64, 1, 2), ComplexF64[], 2.0 + 0im, 2,
                [zeros(ComplexF64, 3)])
            @test param_coeff == param_before
        end

        @testset "compute_higher_derivative_coefficients! reads level j (ORD = 3)" begin
            # W^(j+1) uses Φ^(j), the unit monomial's coefficient at the SAME level j: a
            # master mode's velocity column is λφ, not φ.
            PM = MORFE.ParametrisationMethod
            FOM, L, k = 2, 2, 2
            φ, λ = ComplexF64[1.0, 2.0], 0.5im
            param_coeff = zeros(ComplexF64, FOM, 3, L)
            param_coeff[:, 1, 1] .= φ
            param_coeff[:, 2, 1] .= λ .* φ
            W1 = ComplexF64[1.0, 2.0]
            param_coeff[:, 1, k] .= W1
            red_coeff = zeros(ComplexF64, 1, L)
            red_coeff[1, k] = 0.3
            ξ = [ComplexF64[0.1, 0.2], ComplexF64[-0.4, 0.5]]
            s = 3.0 + 0.0im

            PM.compute_higher_derivative_coefficients!(
                param_coeff, red_coeff, ComplexF64[], s, k, ξ)
            W2 = s .* W1 .+ φ .* 0.3 .+ ξ[1]
            @test param_coeff[:, 2, k] ≈ W2
            @test param_coeff[:, 3, k] ≈ s .* W2 .+ (λ .* φ) .* 0.3 .+ ξ[2]
            # Level 1 at every level (the old bug) would give a different answer.
            @test !(param_coeff[:, 3, k] ≈ s .* W2 .+ φ .* 0.3 .+ ξ[2])
        end

        @testset "compute_higher_derivative_coefficients! self_variable" begin
            # Unit monomial e₂ (an external direction): its own column is already the
            # s-term, so it is excluded exactly. Dyadic values make the arithmetic exact,
            # hence `==`.
            PM = MORFE.ParametrisationMethod
            φ, W1 = ComplexF64[1.0, 2.0], ComplexF64[1.0, -1.0]
            param_coeff = zeros(ComplexF64, 2, 2, 2)
            param_coeff[:, 1, 1] .= φ          # e₁, master
            param_coeff[:, 1, 2] .= W1         # e₂, the direction just solved
            red_coeff = ComplexF64[0.0 0.25]   # R_master at e₂
            e_dyn = ComplexF64[0.5]            # its own external eigenvalue
            ξ = [ComplexF64[0.5, 0.25]]
            s = 0.5 + 0.0im

            PM.compute_higher_derivative_coefficients!(
                param_coeff, red_coeff, e_dyn, s, 2, ξ; self_variable = 2)
            @test param_coeff[:, 2, 2] == s .* W1 .+ ξ[1] .+ φ .* 0.25
        end

        @testset "compute_higher_derivative_coefficients! allocation-free" begin
            # The per-monomial hot path (no self column) reads views only.
            PM = MORFE.ParametrisationMethod
            param_coeff = rand(ComplexF64, 6, 3, 4)
            red_coeff = rand(ComplexF64, 2, 4)
            e_dyn = rand(ComplexF64, 1)
            ξ = [rand(ComplexF64, 6) for _ in 1:3]
            propagate!() = PM.compute_higher_derivative_coefficients!(
                param_coeff, red_coeff, e_dyn, 1.0 + 0.5im, 4, ξ)
            propagate!()
            @test (@allocated propagate!()) == 0
        end
    end
end #@testset "ParametrisationMethod"
