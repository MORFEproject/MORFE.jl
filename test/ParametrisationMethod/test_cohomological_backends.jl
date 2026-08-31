using Test
using LinearAlgebra
using SparseArrays
using StaticArrays: SVector

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, SpectralData
using MORFE.Resonance: ResonanceConfig
const _CheckpointTOML = MORFE.CohomologicalEquations.TOML

function _checkpoint_test_cubic!(res, x1, x2, x3)
    @. res += -x1 * x2 * x3
end
function MORFE.checkpoint_fingerprint_data(::typeof(_checkpoint_test_cubic!))
    ("checkpoint-test-cubic", 1)
end

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
    @test_throws ArgumentError ParametrisationOptions(residual_check = :relative)
    @test_throws ArgumentError ParametrisationOptions(residual_tolerance = 0.0)
    @test_throws ArgumentError ParametrisationOptions(max_refinement_steps = -1)
    @test_throws ArgumentError CheckpointOptions(""; problem_id = "valid")
    @test_throws ArgumentError CheckpointOptions("state"; problem_id = "")

    CE = MORFE.CohomologicalEquations
    @test !CE._has_structural_factor_reuse(ComplexF64[1, 2])
    @test CE._has_structural_factor_reuse(ComplexF64[1, 1])
    @test CE._has_structural_factor_reuse(ComplexF64[0, 2])

    # Mathematically equal superharmonics need not be bit-equal when equal-eigenvalue
    # contributions are distributed differently. Structural grouping deliberately
    # chooses the first job's value as the canonical superharmonic for the whole group.
    repeated = ComplexF64[-0.7021951987576804 - 0.07258186804586991im,
        -0.7021951987576804 - 0.07258186804586991im]
    resonance_mask = SVector(false)
    representatives = CE._eigenvalue_representatives(repeated)
    key_split = CE._structural_factor_key(
        SVector(12, 14), resonance_mask, representatives)
    key_combined = CE._structural_factor_key(
        SVector(26, 0), resonance_mask, representatives)
    split_superharmonic = CE._superharmonic(SVector(12, 14), repeated)
    combined_superharmonic = CE._superharmonic(SVector(26, 0), repeated)
    @test isequal(key_split, key_combined)
    @test split_superharmonic != combined_superharmonic
    canonical_group = CE._SolveGroup(
        split_superharmonic, [CE._SolveJob(1, 0), CE._SolveJob(2, 0)])
    @test canonical_group.superharmonic == split_superharmonic
    @test length(canonical_group.jobs) == 2

    rounding_mset = MultiindexSet([SVector(12, 14), SVector(26, 0)])
    rounding_resonances = MORFE.Resonance.empty_resonance_set(rounding_mset, 1)
    rounding_context = (
        lambda_diag = repeated, resonance_set = rounding_resonances)
    rounding_groups = CE._group_solve_jobs(rounding_context, rounding_mset,
        [CE._SolveJob(1, 0), CE._SolveJob(2, 0)], Val(1))
    @test length(rounding_groups) == 1
    scheduled_first_superharmonic = CE._superharmonic(rounding_mset[1], repeated)
    @test rounding_groups[1].superharmonic == scheduled_first_superharmonic
    @test rounding_groups[1].jobs ==
          [CE._SolveJob(1, 0), CE._SolveJob(2, 0)]

    @testset "solve jobs use the final skip mask" begin
        mset = all_multiindices_up_to(2, 3; min_degree = 1)
        linear = Set(CE._linear_monomial_indices(mset))
        inactive = CE._build_conjugate_symmetry(
            NoConjugatePermutation(), linear, length(mset))
        inactive_jobs = CE._build_solve_jobs(inactive)
        @test all(job.conjugate_target == 0 for job in inactive_jobs)
        @test all(!inactive.skip_bits[job.index] for job in inactive_jobs)
        @test issorted(sum(mset[job.index]) for job in inactive_jobs)

        dictionary = MORFE.Multiindices.build_exponent_index_map(mset)
        active = CE._build_conjugate_symmetry(
            SVector(2, 1), linear, mset, dictionary)
        active_jobs = CE._build_solve_jobs(active)
        paired = findfirst(job -> job.conjugate_target != 0, active_jobs)
        @test paired !== nothing
        pair = active_jobs[paired]
        @test pair.conjugate_target > pair.index
        @test active.skip_bits[pair.conjugate_target]

        # Resume and external-direction setup mark additional entries after symmetry
        # discovery. Rebuilding jobs must observe those final mutations.
        active.skip_bits[pair.index] = true
        rebuilt = CE._build_solve_jobs(active)
        @test all(job.index != pair.index for job in rebuilt)
        @test all(job.conjugate_target != pair.conjugate_target for job in rebuilt)
        @test any(job.conjugate_target == 0 for job in rebuilt)
        @test_throws ArgumentError CE._build_solve_plan(
            nothing, nothing, nothing, :invalid, Val(1))
    end

    @testset "missing external-dynamics coefficient is rejected" begin
        mset = MultiindexSet([
            SVector(1, 0, 0), SVector(0, 1, 0), SVector(0, 0, 1)])
        _, R = create_parametrisation_method_objects(mset, 2, 1, 1, 2, ComplexF64)
        ext_set = all_multiindices_up_to(2, 2; min_degree = 1)
        ext_coefficients = zeros(ComplexF64, 2, length(ext_set))
        cross = findfirst(==(SVector(1, 1)), ext_set.exponents)
        ext_coefficients[1, cross] = 1
        ext_poly = DensePolynomial(ext_coefficients, ext_set)
        @test_throws ArgumentError CE._embed_external_dynamics!(R, ext_poly, mset)
        ext_coefficients[1, cross] = 0
        @test CE._embed_external_dynamics!(R, ext_poly, mset) === nothing
    end

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

    @testset "solution storage ownership and recomputation" begin
        mset = all_multiindices_up_to(2, 5; min_degree = 1)
        rset = MORFE.Resonance.build_resonance_set(
            sparse_model, mset, sd, resonance)
        options = quiet(grouping = :off)
        W_reference, R_reference = solve_cohomological_problem(
            sparse_model, mset, sd, rset;
            conjugate_permutation = nothing, options)

        W_supplied = deepcopy(W_reference)
        R_supplied = deepcopy(R_reference)
        nonlinear_indices = findall(i -> sum(mset[i]) >= 2, eachindex(mset.exponents))
        W_supplied.poly.coefficients[:, :, nonlinear_indices] .= 17 + 3im
        R_supplied.poly.coefficients[:, nonlinear_indices] .= -11 + 5im
        W_coefficients = W_supplied.poly.coefficients
        R_coefficients = R_supplied.poly.coefficients

        W_result, R_result = solve_cohomological_problem(
            sparse_model, mset, sd, rset;
            initial_solution = (W_supplied, R_supplied),
            conjugate_permutation = nothing, options)

        @test W_result === W_supplied
        @test R_result === R_supplied
        @test W_result.poly.coefficients === W_coefficients
        @test R_result.poly.coefficients === R_coefficients
        @test W_result.poly.coefficients == W_reference.poly.coefficients
        @test R_result.poly.coefficients == R_reference.poly.coefficients
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
        Wd, Rd = solve_with(
            ParametrisationOptions(
                show_progress = false, verbose = false); model = dense_model)
        @test relerr(Wd.poly.coefficients, Wk.poly.coefficients) <= 1e-11
        @test relerr(Rd.poly.coefficients, Rk.poly.coefficients) <= 1e-11

        # Impossibly strict verification deterministically reaches the dense and KLU
        # failure paths; allowing one correction also exercises iterative refinement.
        for steps in (0, 1)
            strict_dense = ParametrisationOptions(
                residual_check = :backward_error, residual_tolerance = 1e-30,
                max_refinement_steps = steps, grouping = :off,
                show_progress = false, verbose = false)
            strict_klu = quiet(grouping = :off, residual_tolerance = 1e-30,
                max_refinement_steps = steps)
            @test_throws ErrorException solve_with(strict_dense; model = dense_model)
            @test_throws ErrorException solve_with(strict_klu)
        end
    end

    @testset "checksummed factor-group checkpoint and exact resume" begin
        mktempdir() do directory
            path = joinpath(directory, "cohomological")
            checkpoint = CheckpointOptions(path;
                problem_id = "fixture-v2", granularity = :factor_group)
            options = quiet(grouping = :on, checkpoint = checkpoint)

            opaque = MultilinearMap(
                (res, x1, x2, x3)->(@. res += -x1*x2*x3), (3, 0))
            opaque_model = NthOrderModel(map(sparse, (B0, B1, B2)), (opaque,))
            @test_throws ArgumentError solve_with(options; model = opaque_model)

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

            W_supplied = deepcopy(W0)
            R_supplied = deepcopy(R0)
            checkpoint_mset = W0.poly.multiindex_set
            checkpoint_rset = MORFE.Resonance.build_resonance_set(
                sparse_model, checkpoint_mset, sd, resonance)
            W_same, R_same = solve_cohomological_problem(
                sparse_model, checkpoint_mset, sd, checkpoint_rset;
                initial_solution = (W_supplied, R_supplied), options)
            @test W_same === W_supplied
            @test R_same === R_supplied
            @test W_same.poly.coefficients == W0.poly.coefficients
            @test R_same.poly.coefficients == R0.poly.coefficients

            W_mismatch = deepcopy(W0)
            W_mismatch.poly.coefficients[1] += 1
            @test_throws ArgumentError solve_cohomological_problem(
                sparse_model, checkpoint_mset, sd, checkpoint_rset;
                initial_solution = (W_mismatch, deepcopy(R0)), options)

            R_mismatch = deepcopy(R0)
            R_mismatch.poly.coefficients[1] += 1
            @test_throws ArgumentError solve_cohomological_problem(
                sparse_model, checkpoint_mset, sd, checkpoint_rset;
                initial_solution = (deepcopy(W0), R_mismatch), options)

            stale = quiet(grouping = :on,
                checkpoint = CheckpointOptions(path;
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
            W_partial = deepcopy(W0)
            R_partial = deepcopy(R0)
            degree_five = findall(
                i -> sum(checkpoint_mset[i]) == 5,
                eachindex(checkpoint_mset.exponents))
            W_partial.poly.coefficients[:, :, degree_five] .= 23 - 7im
            R_partial.poly.coefficients[:, degree_five] .= -19 - 2im
            W_completed, R_completed = solve_cohomological_problem(
                sparse_model, checkpoint_mset, sd, checkpoint_rset;
                initial_solution = (W_partial, R_partial), options)
            @test W_completed === W_partial
            @test R_completed === R_partial
            @test relerr(W_completed.poly.coefficients, W0.poly.coefficients) <= 1e-12
            @test relerr(R_completed.poly.coefficients, R0.poly.coefficients) <= 1e-12

            Wi, Ri = solve_with(options)
            @test relerr(Wi.poly.coefficients, W0.poly.coefficients) <= 1e-12
            @test relerr(Ri.poly.coefficients, R0.poly.coefficients) <= 1e-12
            @test _CheckpointTOML.parsefile(manifest_path)["completed_degrees"] ==
                  collect(1:5)

            manifest = _CheckpointTOML.parsefile(manifest_path)
            chunk_path = joinpath(path, "chunks", manifest["chunks"][1]["file"])
            open(chunk_path, "a") do io
                write(io, UInt8(0xff))
            end
            @test_throws ArgumentError solve_with(options)
        end
    end

    @testset "degree checkpoints" begin
        mktempdir() do directory
            checkpoint = CheckpointOptions(joinpath(directory, "degree");
                problem_id = "degree-fixture", granularity = :degree)
            Wd, Rd = solve_with(quiet(grouping = :off, checkpoint = checkpoint))
            manifest = _CheckpointTOML.parsefile(joinpath(checkpoint.path, "manifest.toml"))
            @test manifest["completed_degrees"] == collect(1:5)
            @test length(manifest["chunks"]) == 5
            Wr, Rr = solve_with(quiet(grouping = :off, checkpoint = checkpoint))
            @test Wr.poly.coefficients == Wd.poly.coefficients
            @test Rr.poly.coefficients == Rd.poly.coefficients
        end
    end

    @testset "benchmarked overloads and CSVs" begin
        mset = all_multiindices_up_to(2, 3; min_degree = 1)
        rset = MORFE.Resonance.build_resonance_set(dense_model, mset, sd, resonance)
        base_options = ParametrisationOptions(
            grouping = :off, show_progress = false, verbose = false)

        function check_benchmark(permutation)
            ordinary = solve_cohomological_problem(
                dense_model, mset, sd, rset;
                conjugate_permutation = permutation, options = base_options)
            mktempdir() do directory
                measured = solve_cohomological_problem(
                    dense_model, mset, sd, rset;
                    conjugate_permutation = permutation,
                    benchmark_dir = directory, options = base_options)
                @test measured[1].poly.coefficients ≈ ordinary[1].poly.coefficients
                @test measured[2].poly.coefficients ≈ ordinary[2].poly.coefficients

                mono = readlines(joinpath(directory, "benchmark_per_monomial.csv"))
                order = readlines(joinpath(directory, "benchmark_per_order.csv"))
                @test split(first(mono), ',') == ["order", "monomial_idx", "exponents",
                    "rhs_time_s", "rhs_alloc_bytes", "solve_time_s",
                    "solve_alloc_bytes", "monomial_total_time_s", "cumul_time_s"]
                @test length(split(first(order), ',')) == 9
                solved_rows = length(mono) - 1
                solved_from_orders = sum(parse(Int, split(row, ',')[2])
                for row in order[2:end])
                @test solved_rows == solved_from_orders
                @test solved_rows > 0
            end
        end

        check_benchmark(nothing)
        check_benchmark([2, 1])

        mktempdir() do directory
            checkpoint = CheckpointOptions(joinpath(directory, "checkpoint");
                problem_id = "benchmark-conflict")
            conflicting = ParametrisationOptions(
                checkpoint = checkpoint, grouping = :off,
                show_progress = false, verbose = false)
            @test_throws ArgumentError solve_cohomological_problem(
                dense_model, mset, sd, rset;
                benchmark_dir = joinpath(directory, "benchmark"),
                options = conflicting)
            @test !ispath(checkpoint.path)
        end
    end
end
