using MORFE.InvarianceEquation
using StaticArrays: SVector

@testset "InvarianceEquation.jl" begin
    """Evaluate a matrix polynomial Σ B[k] s^(k-1) naively."""
    function naive_matrix_poly(linear_terms, s)
        ORDP1 = length(linear_terms)
        FOM = size(linear_terms[1], 1)
        L = zeros(eltype(linear_terms[1]), FOM, FOM)
        for k in 1:ORDP1
            L .+= linear_terms[k] .* s^(k - 1)
        end
        return L
    end

    """Evaluate the lower-order RHS naively: −Σ_j L[j](s) ξ[j]."""
    function naive_lower_order_rhs(linear_terms, lower_order_couplings, s)
        ORDP1 = length(linear_terms)
        ORD = ORDP1 - 1
        FOM = size(linear_terms[1], 1)
        rhs = zeros(eltype(linear_terms[1]), FOM)
        for j in 1:ORD
            Lj = zeros(eltype(linear_terms[1]), FOM, FOM)
            for k in (j + 1):ORDP1
                Lj .+= linear_terms[k] .* s^(k - (j + 1))
            end
            rhs .-= Lj * lower_order_couplings[j]
        end
        return rhs
    end

    """Evaluate C_r(s) = Σ_{L=1}^{ORD} C_coeffs[r][:, L] s^(L-1) naively."""
    function naive_evaluate_column(C_coeffs, r, s)
        Cr = C_coeffs[r]
        ORD = size(Cr, 2)
        FOM = size(Cr, 1)
        c = zeros(eltype(Cr), FOM)
        for L in 1:ORD
            c .+= Cr[:, L] .* s^(L - 1)
        end
        return c
    end

    """Naive external RHS: Σ_e ext[e] · E_e(s)."""
    function naive_external_rhs(E_coeffs, external_dynamics, s)
        isempty(E_coeffs) && return zeros(0)
        FOM = size(E_coeffs[1], 1)
        rhs = zeros(eltype(E_coeffs[1]), FOM)
        for e in eachindex(external_dynamics)
            iszero(external_dynamics[e]) && continue
            Ee = E_coeffs[e]
            ORD = size(Ee, 2)
            for L in 1:ORD
                rhs .+= Ee[:, L] .* s^(L - 1) .* external_dynamics[e]
            end
        end
        return rhs
    end

    function reference_column_polynomials(fom_matrices, Y, Λ, ROM)
        ORDP1 = length(fom_matrices)
        ORD = ORDP1 - 1
        FOM = size(fom_matrices[1], 1)
        NVAR = size(Y, 2)
        N_EXT = NVAR - ROM
        T = promote_type(eltype(fom_matrices[1]), eltype(Y), eltype(Λ))

        C_ref = [Matrix{T}(undef, FOM, ORD) for _ in 1:ROM]
        E_ref = [Matrix{T}(undef, FOM, ORD) for _ in 1:N_EXT]

        D = fom_matrices[ORDP1] * Y
        for r in 1:ROM
            C_ref[r][:, ORD] .= D[:, r]
        end
        for e in 1:N_EXT
            E_ref[e][:, ORD] .= -D[:, ROM + e]
        end

        for j in (ORD - 1):-1:1
            D = D * Λ + fom_matrices[j + 1] * Y
            for r in 1:ROM
                C_ref[r][:, j] .= D[:, r]
            end
            for e in 1:N_EXT
                E_ref[e][:, j] .= -D[:, ROM + e]
            end
        end
        return C_ref, E_ref
    end

    @testset "HornerEvaluator.jl" begin
        @testset "evaluate_system_matrix_and_lower_order_rhs - dense, real s" begin
            for (FOM, ORD) in [(5, 3), (4, 1), (6, 5)]
                ORDP1 = ORD + 1
                linear_terms = ntuple(k -> randn(FOM, FOM), ORDP1)
                lower_order_couplings = [randn(FOM) for _ in 1:ORD]
                L = zeros(FOM, FOM)
                L_naive = zeros(FOM, FOM)
                rhs = zeros(FOM)
                rhs_naive = zeros(FOM)
                s = randn()

                evaluate_system_matrix_and_lower_order_rhs!(
                    L,
                    rhs,
                    s,
                    lower_order_couplings,
                    linear_terms
                )

                # naive L 
                L_naive = naive_matrix_poly(linear_terms, s)
                #naive rhs
                rhs_naive = naive_lower_order_rhs(linear_terms, lower_order_couplings, s)
                @test norm(L - L_naive) <= 1e-10
                @test norm(rhs - rhs_naive) <= 1e-10
            end
        end
        @testset "evaluate_system_matrix_and_lower_order_rhs! – complex s" begin
            FOM, ORD = 4, 3
            ORDP1 = ORD + 1
            linear_terms = ntuple(k -> randn(ComplexF64, FOM, FOM), ORDP1)
            lower_order_couplings = [randn(ComplexF64, FOM) for _ in 1:ORD]
            s = 2.0 + 3.0im

            L = zeros(ComplexF64, FOM, FOM)
            rhs = zeros(ComplexF64, FOM)
            evaluate_system_matrix_and_lower_order_rhs!(
                L, rhs, s, lower_order_couplings, linear_terms)

            @test norm(L - naive_matrix_poly(linear_terms, s)) ≤ 1e-10
            @test norm(rhs -
                       naive_lower_order_rhs(linear_terms, lower_order_couplings, s)) ≤
                  1e-10
        end
        @testset "evaluate_system_matrix_and_lower_order_rhs! – ORD=1" begin
            # Minimal case: only B[1] and B[2]; L(s) = B[1] + B[2]*s
            FOM = 3
            B1 = randn(FOM, FOM)
            B2 = randn(FOM, FOM)
            xi1 = randn(FOM)
            s = 1.7

            L = zeros(FOM, FOM)
            rhs = zeros(FOM)
            evaluate_system_matrix_and_lower_order_rhs!(
                L, rhs, s, [xi1], (B1, B2))

            @test norm(L - (B1 .+ B2 .* s)) ≤ 1e-12
            # L[1](s) = B[2], so rhs = -B2 * xi1
            @test norm(rhs - (-B2 * xi1)) ≤ 1e-12
        end
        @testset "evaluate_system_matrix_and_lower_order_rhs! – zero couplings → zero rhs" begin
            FOM, ORD = 4, 3
            ORDP1 = ORD + 1
            linear_terms = ntuple(k -> randn(FOM, FOM), ORDP1)
            lower_order_couplings = [zeros(FOM) for _ in 1:ORD]
            s = randn()

            L = zeros(FOM, FOM)
            rhs = zeros(FOM)
            evaluate_system_matrix_and_lower_order_rhs!(
                L, rhs, s, lower_order_couplings, linear_terms)

            @test norm(rhs) ≤ 1e-15
            @test norm(L - naive_matrix_poly(linear_terms, s)) ≤ 1e-10
        end

        @testset "precompute_sparse_L_template" begin
            @testset "precompute_sparse_L_template basic" begin
                B1 = sparse([1, 2], [1, 2], [10.0, 20.0], 3, 3)
                B2 = sparse([1, 3], [2, 3], [30.0, 40.0], 3, 3)

                linear_terms = (B1, B2)

                L_template, mappings = precompute_sparse_L_template(linear_terms)

                # --- 1. Check union sparsity pattern ---
                union = (B1 .!= 0) .| (B2 .!= 0)
                @test nnz(L_template) == count(!iszero, union)

                # --- 2. Check template has correct structure ---
                @test size(L_template) == (3, 3)

                # --- 3. Check mapping correctness ---
                for k in 1:2
                    Bk = linear_terms[k]
                    mapping_k = mappings[k]

                    for pos_k in 1:nnz(Bk)
                        expected_val = Bk.nzval[pos_k]
                        pos_L = mapping_k[pos_k]

                        @test L_template.nzval[pos_L] == 0.0 + 0im  # initially zero
                    end
                end
            end

            @testset "precompute_sparse_L_template – overlapping patterns" begin
                # Both matrices have a nonzero at (1,1); the union should count it once.
                B1 = sparse([1, 2], [1, 1], [1.0, 2.0], 3, 3)
                B2 = sparse([1, 3], [1, 2], [3.0, 4.0], 3, 3)
                L_tmpl, _ = precompute_sparse_L_template((B1, B2))
                @test nnz(L_tmpl) == 3   # (1,1), (2,1), (3,2)
            end

            @testset "build_sparse_L_and_rhs! matches dense Horner" begin
                FOM, ORD = 6, 3
                ORDP1 = ORD + 1
                # Random sparse matrices (~50 % density)
                linear_terms_sparse = ntuple(
                    k -> sprand(FOM, FOM, 0.5) .+ 1e-3 * sparse(I, FOM, FOM), ORDP1)
                linear_terms_dense = ntuple(k -> Matrix(linear_terms_sparse[k]), ORDP1)

                lower_order_couplings = [randn(ComplexF64, FOM) for _ in 1:ORD]
                s = 1.2 + 0.5im

                L_tmpl, maps = precompute_sparse_L_template(linear_terms_sparse)
                rhs_sparse = zeros(ComplexF64, FOM)
                build_sparse_L_and_rhs!(
                    rhs_sparse, L_tmpl, maps, linear_terms_sparse, s, lower_order_couplings)

                L_dense = zeros(ComplexF64, FOM, FOM)
                rhs_dense = zeros(ComplexF64, FOM)
                evaluate_system_matrix_and_lower_order_rhs!(
                    L_dense, rhs_dense, s, lower_order_couplings, linear_terms_dense)

                @test norm(Matrix(L_tmpl) - L_dense) ≤ 1e-10
                @test norm(rhs_sparse - rhs_dense) ≤ 1e-10
            end
        end
    end #@testset HornerEvaluator.jl

    @testset "ColumnPolynomials.jl" begin
        @testset "precompute_column_polynomials – output dimensions" begin
            FOM, ROM, N_EXT, ORD = 8, 3, 2, 4
            NVAR = ROM + N_EXT
            ORDP1 = ORD + 1

            fom_matrices = ntuple(k -> randn(ComplexF64, FOM, FOM), ORDP1)
            Y = randn(ComplexF64, FOM, NVAR)
            Λ = diagm(randn(ComplexF64, NVAR))

            C, E = precompute_column_polynomials(fom_matrices, Y, Λ, ROM)

            @test length(C) == ROM
            @test length(E) == N_EXT
            @test all(size(Cmat) == (FOM, ORD) for Cmat in C)
            @test all(size(Emat) == (FOM, ORD) for Emat in E)
        end

        @testset "precompute_column_polynomials – matches reference recurrence" begin
            FOM, ROM, N_EXT, ORD = 6, 2, 2, 3
            NVAR = ROM + N_EXT
            ORDP1 = ORD + 1

            fom_matrices = ntuple(k -> randn(ComplexF64, FOM, FOM), ORDP1)
            Y = randn(ComplexF64, FOM, NVAR)
            Λ = randn(ComplexF64, NVAR, NVAR)   # general (not just diagonal)

            C, E = precompute_column_polynomials(fom_matrices, Y, Λ, ROM)
            C_ref, E_ref = reference_column_polynomials(fom_matrices, Y, Λ, ROM)

            for r in 1:ROM
                @test norm(C[r] - C_ref[r]) ≤ 1e-10
            end
            for e in 1:N_EXT
                @test norm(E[e] - E_ref[e]) ≤ 1e-10
            end
        end

        @testset "precompute_column_polynomials – N_EXT = 0" begin
            FOM, ROM, ORD = 5, 3, 2
            NVAR = ROM        # no external modes
            ORDP1 = ORD + 1

            fom_matrices = ntuple(k -> randn(ComplexF64, FOM, FOM), ORDP1)
            Y = randn(ComplexF64, FOM, NVAR)
            Λ = diagm(randn(ComplexF64, NVAR))

            C, E = precompute_column_polynomials(fom_matrices, Y, Λ, ROM)

            @test length(C) == ROM
            @test isempty(E)
        end

        @testset "precompute_column_polynomials – ORD = 1" begin
            # With ORD = 1, C_r(s) = C_r[1] (constant polynomial).
            FOM, ROM, N_EXT = 4, 2, 1
            NVAR = ROM + N_EXT
            ORDP1 = 2   # ORD = 1

            fom_matrices = ntuple(k -> randn(ComplexF64, FOM, FOM), ORDP1)
            Y = randn(ComplexF64, FOM, NVAR)
            Λ = diagm(randn(ComplexF64, NVAR))

            C, E = precompute_column_polynomials(fom_matrices, Y, Λ, ROM)

            # At ORD = 1: D = B[2] * Y; C[r][:, 1] = D[:, r]
            D_expected = fom_matrices[2] * Y
            for r in 1:ROM
                @test norm(C[r][:, 1] - D_expected[:, r]) ≤ 1e-12
            end
            for e in 1:N_EXT
                @test norm(E[e][:, 1] + D_expected[:, ROM + e]) ≤ 1e-12   # sign flip
            end
        end

        @testset "precompute_master_column_polynomials – matches monolithic C_coeffs" begin
            FOM, ROM, N_EXT, ORD = 7, 3, 2, 4
            NVAR = ROM + N_EXT
            ORDP1 = ORD + 1

            fom_matrices = ntuple(k -> randn(ComplexF64, FOM, FOM), ORDP1)
            Y = randn(ComplexF64, FOM, NVAR)
            # Upper-triangular Λ (Jordan-form convention)
            Λ = UpperTriangular(randn(ComplexF64, NVAR, NVAR))

            C_mono, _ = precompute_column_polynomials(fom_matrices, Y, Matrix(Λ), ROM)

            Λ_master = Matrix(Λ)[1:ROM, 1:ROM]
            C_split, _ = precompute_master_column_polynomials(
                fom_matrices, Y[:, 1:ROM], Λ_master)

            for r in 1:ROM
                @test norm(C_split[r] - C_mono[r]) ≤ 1e-10
            end
        end

        @testset "precompute_external_column_polynomials – matches monolithic E_coeffs" begin
            FOM, ROM, N_EXT, ORD = 7, 3, 2, 4
            NVAR = ROM + N_EXT
            ORDP1 = ORD + 1

            fom_matrices = ntuple(k -> randn(ComplexF64, FOM, FOM), ORDP1)
            Y = randn(ComplexF64, FOM, NVAR)
            Λ = UpperTriangular(randn(ComplexF64, NVAR, NVAR))

            _, E_mono = precompute_column_polynomials(fom_matrices, Y, Matrix(Λ), ROM)

            Λ_master = Matrix(Λ)[1:ROM, 1:ROM]
            _, D_steps = precompute_master_column_polynomials(
                fom_matrices, Y[:, 1:ROM], Λ_master)
            E_split = precompute_external_column_polynomials(
                fom_matrices, Y[:, (ROM + 1):end], Matrix(Λ), D_steps)

            for e in 1:N_EXT
                @test norm(E_split[e] - E_mono[e]) ≤ 1e-10
            end
        end

        @testset "precompute_external_column_polynomials – N_EXT = 0 returns empty" begin
            # The genuine no-forcing edge case: N_EXT = 0 should trigger the early
            # return guard and give back an empty coefficient vector.
            FOM, ROM, ORD = 5, 2, 3
            NVAR = ROM   # no external modes
            ORDP1 = ORD + 1

            fom_matrices = ntuple(k -> randn(ComplexF64, FOM, FOM), ORDP1)
            Y_master = randn(ComplexF64, FOM, ROM)
            Λ_master = diagm(randn(ComplexF64, ROM))
            Λ_full = Λ_master   # NVAR×NVAR = ROM×ROM, no external block

            _, D_steps = precompute_master_column_polynomials(
                fom_matrices, Y_master, Λ_master)
            E_empty = precompute_external_column_polynomials(
                fom_matrices, zeros(ComplexF64, FOM, 0), Λ_full, D_steps)

            @test isempty(E_empty)
        end

        @testset "evaluate_column! – matches naive polynomial evaluation" begin
            FOM, ROM, ORD = 5, 3, 4
            C_coeffs = [randn(ComplexF64, FOM, ORD) for _ in 1:ROM]
            s = 1.5 + 0.3im

            for r in 1:ROM
                c = zeros(ComplexF64, FOM)
                evaluate_column!(c, s, r, C_coeffs)
                @test norm(c - naive_evaluate_column(C_coeffs, r, s)) ≤ 1e-10
            end
        end

        @testset "evaluate_column! – writes into a view (no allocation check)" begin
            FOM, ROM, ORD = 4, 2, 3
            C_coeffs = [randn(ComplexF64, FOM, ORD) for _ in 1:ROM]
            s = 1.0 + 0im
            M = zeros(ComplexF64, FOM, ROM + 2)

            for r in 1:ROM
                evaluate_column!(view(M, :, r + 1), s, r, C_coeffs)
                @test norm(view(M, :, r + 1) - naive_evaluate_column(C_coeffs, r, s)) ≤
                      1e-12
            end
        end

        @testset "evaluate_external_rhs! – matches naive accumulation" begin
            FOM, N_EXT, ORD = 6, 3, 4
            E_coeffs = [randn(ComplexF64, FOM, ORD) for _ in 1:N_EXT]
            external_dynamics = randn(ComplexF64, N_EXT)
            s = 0.8 + 1.2im

            rhs = zeros(ComplexF64, FOM)
            g = zeros(ComplexF64, FOM)
            evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs, g)

            @test norm(rhs - naive_external_rhs(E_coeffs, external_dynamics, s)) ≤ 1e-10
        end

        @testset "evaluate_external_rhs! – all-zero dynamics → no change to rhs" begin
            FOM, N_EXT, ORD = 5, 3, 3
            E_coeffs = [randn(ComplexF64, FOM, ORD) for _ in 1:N_EXT]
            external_dynamics = zeros(ComplexF64, N_EXT)
            s = 0.8 + 1.2im

            rhs_init = randn(ComplexF64, FOM)
            rhs = copy(rhs_init)
            g = zeros(ComplexF64, FOM)
            evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs, g)

            @test rhs == rhs_init
        end

        @testset "evaluate_external_rhs! – sparse external dynamics (one nonzero)" begin
            FOM, N_EXT, ORD = 6, 4, 3
            E_coeffs = [randn(ComplexF64, FOM, ORD) for _ in 1:N_EXT]
            external_dynamics = zeros(ComplexF64, N_EXT)
            external_dynamics[2] = 3.0 + 0im   # only mode 2 is active
            s = 1.0 + 0im

            rhs = zeros(ComplexF64, FOM)
            g = zeros(ComplexF64, FOM)
            evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs, g)

            @test norm(rhs - naive_external_rhs(E_coeffs, external_dynamics, s)) ≤ 1e-10
        end

        @testset "evaluate_external_rhs! – N_EXT = 0 is a no-op" begin
            FOM = 5
            E_coeffs = Vector{Matrix{ComplexF64}}()
            external_dynamics = ComplexF64[]
            s = 1.0 + 0im

            rhs_init = randn(ComplexF64, FOM)
            rhs = copy(rhs_init)
            g = zeros(ComplexF64, FOM)
            evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs, g)

            @test rhs == rhs_init
        end
    end #@testset ColumnPolynomials.jl

    @testset "assemble_cohomological_matrix_and_rhs!" begin
        # Build a consistent problem fixture.
        function make_fixture(; FOM = 8, ROM = 3, N_EXT = 2, ORD = 3)
            NVAR = ROM + N_EXT
            ORDP1 = ORD + 1
            T = ComplexF64

            fom_matrices = ntuple(k -> randn(T, FOM, FOM), ORDP1)
            Y = randn(T, FOM, NVAR)
            Λ = diagm(randn(T, NVAR))   # diagonal for simplicity

            C_coeffs, E_coeffs = precompute_column_polynomials(fom_matrices, Y, Λ, ROM)

            # Pretend the first ROM/2 (at least 1) master modes are resonant.
            nR = max(1, ROM ÷ 2)
            resonance_vec = [i ≤ nR for i in 1:ROM]
            resonance = SVector{ROM, Bool}(resonance_vec...)

            lower_order_couplings = [randn(T, FOM) for _ in 1:ORD]
            external_dynamics = randn(T, N_EXT)
            s = 0.5 + 1.0im

            return (; fom_matrices, C_coeffs, E_coeffs, resonance, resonance_vec,
                lower_order_couplings, external_dynamics, s, FOM, ROM, N_EXT, nR)
        end

        @testset "output dimensions" begin
            f = make_fixture()
            (; FOM, ROM, fom_matrices, C_coeffs, E_coeffs,
            resonance, lower_order_couplings, external_dynamics, s) = f

            # Constant width: the border is ROM columns wide regardless of nR.
            n_sys = FOM + ROM
            M = zeros(ComplexF64, FOM, n_sys)
            rhs = zeros(ComplexF64, FOM)
            g = zeros(ComplexF64, FOM)

            assemble_cohomological_matrix_and_rhs!(
                M, rhs, s, fom_matrices, C_coeffs, E_coeffs,
                resonance, lower_order_couplings, external_dynamics, g)

            @test size(M) == (FOM, n_sys)
            @test length(rhs) == FOM
        end

        @testset "left block M[:,1:FOM] equals L(s)" begin
            f = make_fixture()
            (; FOM, ROM, fom_matrices, C_coeffs, E_coeffs,
            resonance, lower_order_couplings, external_dynamics, s) = f

            M = zeros(ComplexF64, FOM, FOM + ROM)
            rhs = zeros(ComplexF64, FOM)
            g = zeros(ComplexF64, FOM)
            assemble_cohomological_matrix_and_rhs!(
                M, rhs, s, fom_matrices, C_coeffs, E_coeffs,
                resonance, lower_order_couplings, external_dynamics, g)

            L_ref = naive_matrix_poly(fom_matrices, s)
            @test norm(M[:, 1:FOM] - L_ref) ≤ 1e-10
        end

        @testset "border column FOM+r is C_r(s) when resonant, zero otherwise" begin
            f = make_fixture()
            (; FOM, ROM, fom_matrices, C_coeffs, E_coeffs,
            resonance, resonance_vec, lower_order_couplings, external_dynamics, s) = f

            # Pre-fill with garbage: the assembly must overwrite every border column,
            # including the non-resonant ones it masks to zero.
            M = fill(ComplexF64(7, -3), FOM, FOM + ROM)
            rhs = zeros(ComplexF64, FOM)
            g = zeros(ComplexF64, FOM)
            assemble_cohomological_matrix_and_rhs!(
                M, rhs, s, fom_matrices, C_coeffs, E_coeffs,
                resonance, lower_order_couplings, external_dynamics, g)

            for r in 1:ROM
                if resonance_vec[r]
                    c_ref = naive_evaluate_column(C_coeffs, r, s)
                    @test norm(M[:, FOM + r] - c_ref) ≤ 1e-10
                else
                    @test all(iszero, M[:, FOM + r])
                end
            end
        end

        @testset "rhs equals lower-order + external contributions" begin
            f = make_fixture()
            (; FOM, ROM, fom_matrices, C_coeffs, E_coeffs,
            resonance, lower_order_couplings, external_dynamics, s) = f

            M = zeros(ComplexF64, FOM, FOM + ROM)
            rhs = zeros(ComplexF64, FOM)
            g = zeros(ComplexF64, FOM)
            assemble_cohomological_matrix_and_rhs!(
                M, rhs, s, fom_matrices, C_coeffs, E_coeffs,
                resonance, lower_order_couplings, external_dynamics, g)

            rhs_lo = naive_lower_order_rhs(fom_matrices, lower_order_couplings, s)
            rhs_ext = naive_external_rhs(E_coeffs, external_dynamics, s)
            @test norm(rhs - (rhs_lo .+ rhs_ext)) ≤ 1e-10
        end
        @testset "no resonant modes → border is all zeros, rhs still correct" begin
            FOM, ROM, N_EXT, ORD = 5, 3, 1, 2
            NVAR = ROM + N_EXT
            ORDP1 = ORD + 1
            T = ComplexF64

            fom_matrices = ntuple(k -> randn(T, FOM, FOM), ORDP1)
            Y = randn(T, FOM, NVAR)
            Λ = diagm(randn(T, NVAR))
            C_coeffs, E_coeffs = precompute_column_polynomials(fom_matrices, Y, Λ, ROM)

            resonance = SVector{ROM, Bool}(false, false, false)
            lower_order_couplings = [randn(T, FOM) for _ in 1:ORD]
            external_dynamics = randn(T, N_EXT)
            s = 1.0 + 0.5im

            M = fill(T(1), FOM, FOM + ROM)   # width is ROM even at nR = 0
            rhs = zeros(T, FOM)
            g = zeros(T, FOM)
            assemble_cohomological_matrix_and_rhs!(
                M, rhs, s, fom_matrices, C_coeffs, E_coeffs,
                resonance, lower_order_couplings, external_dynamics, g)

            @test size(M) == (FOM, FOM + ROM)
            @test norm(M[:, 1:FOM] - naive_matrix_poly(fom_matrices, s)) ≤ 1e-10
            @test all(iszero, M[:, (FOM + 1):(FOM + ROM)])
        end
        @testset "all modes resonant → right block has ROM columns" begin
            FOM, ROM, N_EXT, ORD = 4, 2, 1, 2
            NVAR = ROM + N_EXT
            ORDP1 = ORD + 1
            T = ComplexF64

            fom_matrices = ntuple(k -> randn(T, FOM, FOM), ORDP1)
            Y = randn(T, FOM, NVAR)
            Λ = diagm(randn(T, NVAR))
            C_coeffs, E_coeffs = precompute_column_polynomials(fom_matrices, Y, Λ, ROM)

            resonance = SVector{ROM, Bool}(true, true)
            lower_order_couplings = [randn(T, FOM) for _ in 1:ORD]
            external_dynamics = randn(T, N_EXT)
            s = 2.0 + 0im

            M = zeros(T, FOM, FOM + ROM)
            rhs = zeros(T, FOM)
            g = zeros(T, FOM)
            assemble_cohomological_matrix_and_rhs!(
                M, rhs, s, fom_matrices, C_coeffs, E_coeffs,
                resonance, lower_order_couplings, external_dynamics, g)

            for r in 1:ROM
                c_ref = naive_evaluate_column(C_coeffs, r, s)
                @test norm(M[:, FOM + r] - c_ref) ≤ 1e-10
            end
        end

        @testset "idempotency – calling twice gives the same result" begin
            f = make_fixture()
            (; FOM, ROM, fom_matrices, C_coeffs, E_coeffs,
            resonance, lower_order_couplings, external_dynamics, s) = f

            M1 = zeros(ComplexF64, FOM, FOM + ROM)
            rhs1 = zeros(ComplexF64, FOM)
            M2 = zeros(ComplexF64, FOM, FOM + ROM)
            rhs2 = zeros(ComplexF64, FOM)
            g = zeros(ComplexF64, FOM)

            assemble_cohomological_matrix_and_rhs!(
                M1, rhs1, s, fom_matrices, C_coeffs, E_coeffs,
                resonance, lower_order_couplings, external_dynamics, g)
            assemble_cohomological_matrix_and_rhs!(
                M2, rhs2, s, fom_matrices, C_coeffs, E_coeffs,
                resonance, lower_order_couplings, external_dynamics, g)

            @test M1 == M2
            @test rhs1 == rhs2
        end
    end
end