using Test
using LinearAlgebra
using SparseArrays
using Serialization

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, SpectralData
using MORFE.Resonance: ResonanceConfig
using MORFE.InvarianceEquation: precompute_sparse_L_template

@testset "Cohomological sparse backends and checkpoints" begin
    relerr(a, b) = norm(a .- b) / max(norm(a), norm(b), eps())

    @test CohomologicalSolverConfig().backend == :auto
    @test CohomologicalSolverConfig().group_superharmonics == false
    @test_throws ArgumentError CohomologicalSolverConfig(backend=:unknown)
    @test_throws ArgumentError CohomologicalSolverConfig(residual_tolerance=0.0)
    @test_throws ArgumentError CohomologicalCheckpoint(""; id="valid")
    @test_throws ArgumentError CohomologicalCheckpoint("state.jls"; id="")

    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = Matrix{Float64}(I, 2, 2)
    B1 = 0.001 .* B2
    cubic = MultilinearMap(
        (res, x1, x2, x3) -> (@. res += -x1 * x2 * x3), (3, 0))
    dense_model = NthOrderModel((B0, B1, B2), (cubic,))
    sparse_model = NthOrderModel(map(sparse, (B0, B1, B2)), (cubic,))
    ep = spectrum(dense_model; solver=DefaultEigensolver())
    sd = SpectralData(dense_model, ep; master=master_by_sorting(2))
    resonance = ResonanceConfig(
        style=:complex_normal_form, tol=0.05, warn_outer=false)

    solve_with(config; checkpoint=nothing) = parametrise(
        sparse_model, sd, 5; resonance, solver_config=config, checkpoint,
        show_progress=false, verbose=false)

    klu_plain = CohomologicalSolverConfig(
        backend=:klu, residual_tolerance=1e-10)
    klu_grouped = CohomologicalSolverConfig(
        backend=:klu, residual_tolerance=1e-10, group_superharmonics=true)
    umf_grouped = CohomologicalSolverConfig(
        backend=:umfpack, residual_tolerance=1e-10, group_superharmonics=true)

    Wk, Rk = solve_with(klu_plain)
    Wkg, Rkg = solve_with(klu_grouped)
    Wu, Ru = solve_with(umf_grouped)
    @test relerr(Wkg.poly.coefficients, Wk.poly.coefficients) <= 1e-12
    @test relerr(Rkg.poly.coefficients, Rk.poly.coefficients) <= 1e-12
    @test relerr(Wu.poly.coefficients, Wk.poly.coefficients) <= 1e-10
    @test relerr(Ru.poly.coefficients, Rk.poly.coefficients) <= 1e-10

    @testset "UMFPACK reuses one numeric factor within a group" begin
        lt = map(complex, sparse_model.linear_terms)
        L_template, L_mappings = precompute_sparse_L_template(lt)
        config = CohomologicalSolverConfig(
            backend=:umfpack, residual_tolerance=1e-10)
        ss = MORFE.CohomologicalEquations.SparseLinearSolverState{ComplexF64}(
            L_template, L_mappings, 2, 2; config)
        fill!(ss.bordered.nzval, 0)
        for i in axes(ss.bordered, 1)
            ss.bordered[i, i] = 2 + 0.1im * i
        end
        key = (0.25im, (false, false))
        for scale in (1.0, 2.0)
            rhs = fill(complex(scale), 4)
            MORFE.CohomologicalEquations._bordered_solve!(ss, rhs, key[1], key)
            @test all(isfinite, rhs)
        end
        @test ss.factorization_count == 1
        @test ss.solve_count == 2
        @test ss.max_relative_residual <= 1e-10

        # A new key must replace the C-side numeric object and reuse only the
        # unchanged symbolic analysis. Repeating this catches stale/double-freed
        # numeric pointers without relying on garbage collection timing.
        for i in 1:25
            ss.bordered[1, 1] = 2 + i / 100 + 0.1im
            rhs = ones(ComplexF64, 4)
            next_key = (complex(i), (false, false))
            MORFE.CohomologicalEquations._bordered_solve!(
                ss, rhs, next_key[1], next_key)
            @test norm(ss.bordered * rhs .- ones(ComplexF64, 4)) <= 1e-10
        end
        @test ss.factorization_count == 26
    end

    @testset "atomic degree checkpoint and exact resume contract" begin
        mktempdir() do directory
            path = joinpath(directory, "cohomological.jls")
            checkpoint = CohomologicalCheckpoint(path; id="fixture-v1")
            W0, R0 = solve_with(umf_grouped; checkpoint)
            @test isfile(path)

            saved = open(deserialize, path)
            @test saved.completed_degree == 5
            @test saved.diagnostics.backend == :umfpack
            @test saved.diagnostics.max_relative_residual <= 1e-10

            Wr, Rr = solve_with(umf_grouped; checkpoint)
            @test Wr.poly.coefficients == W0.poly.coefficients
            @test Rr.poly.coefficients == R0.poly.coefficients

            @test_throws ArgumentError solve_with(umf_grouped;
                checkpoint=CohomologicalCheckpoint(path; id="stale-id"))
            changed = CohomologicalSolverConfig(
                backend=:umfpack, residual_tolerance=1e-9,
                group_superharmonics=true)
            @test_throws ArgumentError solve_with(changed; checkpoint)

            # Simulate an interruption after degree three. Higher coefficients are
            # cleared so the resumed solve must reconstruct them, not merely accept
            # a completed result carrying a lower degree marker.
            partial_W, partial_R = deepcopy(W0), deepcopy(R0)
            mset = partial_W.poly.multiindex_set
            for idx in eachindex(mset.exponents)
                if sum(mset[idx]) > 3
                    partial_W.poly.coefficients[:, :, idx] .= 0
                    partial_R.poly.coefficients[:, idx] .= 0
                end
            end
            open(path, "w") do io
                serialize(io, merge(saved, (;
                    completed_degree=3, W=partial_W, R=partial_R)))
            end
            Wi, Ri = solve_with(umf_grouped; checkpoint)
            @test relerr(Wi.poly.coefficients, W0.poly.coefficients) <= 1e-12
            @test relerr(Ri.poly.coefficients, R0.poly.coefficients) <= 1e-12
            @test open(deserialize, path).completed_degree == 5
        end
    end
end
