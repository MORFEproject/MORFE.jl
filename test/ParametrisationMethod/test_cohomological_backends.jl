using Test
using LinearAlgebra
using SparseArrays

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, SpectralData
using MORFE.Resonance: ResonanceConfig
const _CheckpointTOML = MORFE.CohomologicalEquations.TOML

function _checkpoint_test_cubic!(res, x1, x2, x3)
    @. res += -x1 * x2 * x3
end
MORFE.checkpoint_fingerprint_data(::typeof(_checkpoint_test_cubic!)) =
    ("checkpoint-test-cubic", 1)

@testset "Cohomological sparse backends and durable checkpoints" begin
    relerr(a, b) = norm(a .- b) / max(norm(a), norm(b), eps())

    defaults = ParametrisationOptions()
    @test defaults.backend == :auto
    @test defaults.grouping == :auto
    @test defaults.residual_check == :off
    @test defaults.residual_tolerance === nothing
    @test_throws ArgumentError ParametrisationOptions(backend = :umfpack)
    @test_throws ArgumentError ParametrisationOptions(backend = :unknown)
    @test_throws ArgumentError ParametrisationOptions(grouping = :approximate)
    @test_throws ArgumentError ParametrisationOptions(residual_tolerance = 0.0)
    @test_throws ArgumentError CheckpointOptions(""; problem_id = "valid")
    @test_throws ArgumentError CheckpointOptions("state"; problem_id = "")

    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = Matrix{Float64}(I, 2, 2)
    B1 = 0.001 .* B2
    cubic = MultilinearMap(_checkpoint_test_cubic!, (3, 0))
    dense_model = NthOrderModel((B0, B1, B2), (cubic,))
    sparse_model = NthOrderModel(map(sparse, (B0, B1, B2)), (cubic,))
    ep = spectrum(dense_model; solver = DefaultEigensolver())
    sd = SpectralData(dense_model, ep; master = master_by_sorting(2))
    resonance = ResonanceConfig(
        style = :complex_normal_form, tol = 0.05, warn_outer = false)

    solve_with(options; model = sparse_model) = parametrise(
        model, sd, 5; resonance, options)

    function quiet(; kwargs...)
        defaults = (; backend = :klu, residual_check = :backward_error,
            residual_tolerance = 1e-10, show_progress = false, verbose = false)
        return ParametrisationOptions(; merge(defaults, (; kwargs...))...)
    end

    @testset "restored KLU path and exact structural grouping" begin
        Wk, Rk = solve_with(quiet(grouping = :off))
        Wkg, Rkg = solve_with(quiet(grouping = :on))
        Wka, Rka = solve_with(quiet(grouping = :auto))
        Wlegacy, Rlegacy = solve_with(quiet(
            grouping = :off, residual_check = :off,
            residual_tolerance = nothing))
        @test relerr(Wkg.poly.coefficients, Wk.poly.coefficients) <= 1e-12
        @test relerr(Rkg.poly.coefficients, Rk.poly.coefficients) <= 1e-12
        # The automatic policy must retain the direct legacy GrLex path when
        # structural reuse offers no reduction. Residual verification must not
        # perturb an already accepted solve.
        @test Wka.poly.coefficients == Wk.poly.coefficients ==
            Wlegacy.poly.coefficients
        @test Rka.poly.coefficients == Rk.poly.coefficients ==
            Rlegacy.poly.coefficients

        @test_throws ArgumentError solve_with(quiet(); model = dense_model)
        Wd, Rd = solve_with(ParametrisationOptions(
            show_progress = false, verbose = false); model = dense_model)
        @test relerr(Wd.poly.coefficients, Wk.poly.coefficients) <= 1e-11
        @test relerr(Rd.poly.coefficients, Rk.poly.coefficients) <= 1e-11
    end

    @testset "checksummed factor-group checkpoint and exact resume" begin
        mktempdir() do directory
            path = joinpath(directory, "cohomological")
            checkpoint = CheckpointOptions(path;
                problem_id = "fixture-v2", granularity = :factor_group)
            options = quiet(grouping = :on, checkpoint = checkpoint)

            opaque = MultilinearMap(
                (res,x1,x2,x3)->(@. res += -x1*x2*x3),(3,0))
            opaque_model = NthOrderModel(map(sparse,(B0,B1,B2)),(opaque,))
            @test_throws ArgumentError solve_with(options; model=opaque_model)

            W0, R0 = solve_with(options)

            manifest_path = joinpath(path, "manifest.toml")
            @test isfile(manifest_path)
            manifest = _CheckpointTOML.parsefile(manifest_path)
            @test manifest["schema_version"] == 2
            @test manifest["problem_id"] == "fixture-v2"
            @test manifest["completed_degrees"] == collect(1:5)
            @test manifest["diagnostics"]["backend"] == "klu"
            @test manifest["diagnostics"]["max_backward_error"] <= 1e-10
            @test !isempty(manifest["chunks"])
            @test all(length(chunk["sha256"]) == 64 for chunk in manifest["chunks"])

            Wr, Rr = solve_with(options)
            @test Wr.poly.coefficients == W0.poly.coefficients
            @test Rr.poly.coefficients == R0.poly.coefficients

            stale = quiet(grouping = :on, checkpoint = CheckpointOptions(path;
                problem_id = "stale-id", granularity = :factor_group))
            @test_throws ArgumentError solve_with(stale)
            changed = quiet(grouping = :on, residual_tolerance = 1e-9,
                checkpoint = checkpoint)
            @test_throws ArgumentError solve_with(changed)

            manifest = _CheckpointTOML.parsefile(manifest_path)
            retained = Any[]
            for chunk in manifest["chunks"]
                if chunk["degree"] == 5
                    rm(joinpath(path, "chunks", chunk["file"]))
                else
                    push!(retained, chunk)
                end
            end
            manifest["chunks"] = retained
            manifest["completed_degrees"] = collect(1:4)
            open(manifest_path, "w") do io
                _CheckpointTOML.print(io, manifest; sorted = true)
            end
            Wi, Ri = solve_with(options)
            @test relerr(Wi.poly.coefficients, W0.poly.coefficients) <= 1e-12
            @test relerr(Ri.poly.coefficients, R0.poly.coefficients) <= 1e-12
            @test _CheckpointTOML.parsefile(manifest_path)["completed_degrees"] == collect(1:5)

            manifest = _CheckpointTOML.parsefile(manifest_path)
            chunk_path = joinpath(path, "chunks", manifest["chunks"][1]["file"])
            open(chunk_path, "a") do io
                write(io, UInt8(0xff))
            end
            @test_throws ArgumentError solve_with(options)
        end
    end
end
