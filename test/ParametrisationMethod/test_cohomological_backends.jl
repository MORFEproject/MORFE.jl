using Test
using LinearAlgebra
using SparseArrays
using Random: MersenneTwister
using StaticArrays: SVector

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, SpectralData
using MORFE.Resonance: ResonanceConfig
const _CheckpointTOML = MORFE.ParametrisationSolver.TOML

function _checkpoint_test_cubic!(res, x1, x2, x3)
    @. res += -x1 * x2 * x3
end
function MORFE.checkpoint_fingerprint_data(::typeof(_checkpoint_test_cubic!))
    ("checkpoint-test-cubic", 1)
end

# A cubic term for the first-order saddle-point fixture of the automatic-backend test.
cubic_saddle = MultilinearMap((res, x1, x2, x3) -> (@. res += -x1 * x2 * x3), (3,))

# Parameter-dependent terms for the cross-degree fixture: a stiffness change linear in the
# external state, and a static load proportional to it.
const _CROSS_DEGREE_K1 = [1.0 0.2; 0.2 0.5]
const _CROSS_DEGREE_G = [0.3 -0.1; -0.2 0.4]
function _cross_degree_stiffness!(res, x, r)
    res .+= r[1] .* (_CROSS_DEGREE_K1 * x) .+ r[2] .* x
end
_cross_degree_load!(res, r) = (res .+= _CROSS_DEGREE_G * r)
function MORFE.checkpoint_fingerprint_data(::typeof(_cross_degree_stiffness!))
    ("cross-degree-stiffness", 1)
end
function MORFE.checkpoint_fingerprint_data(::typeof(_cross_degree_load!))
    ("cross-degree-load", 1)
end
# The same terms for three external coordinates (θ₁, r₁, θ₂).
const _CROSS_DEGREE_G3 = [0.3 -0.1 0.2; -0.2 0.4 0.1]
function _cross_degree_stiffness3!(res, x, r)
    res .+= r[1] .* (_CROSS_DEGREE_K1 * x) .+ r[3] .* x
end
_cross_degree_load3!(res, r) = (res .+= _CROSS_DEGREE_G3 * r)

@testset "Cohomological sparse backends and durable checkpoints" begin
    relerr(a, b) = norm(a .- b) / max(norm(a), norm(b), eps())

    defaults = ParametrisationOptions()
    @test defaults.backend == :auto
    @test defaults.grouping == :auto
    @test defaults.residual_check == :off
    @test defaults.residual_tolerance === nothing
    @test ParametrisationOptions(backend = :umfpack).backend == :umfpack
    @test_throws ArgumentError ParametrisationOptions(backend = :unknown)
    @test_throws ArgumentError ParametrisationOptions(grouping = :approximate)
    @test_throws ArgumentError ParametrisationOptions(residual_check = :relative)
    @test_throws ArgumentError ParametrisationOptions(residual_tolerance = 0.0)
    @test_throws ArgumentError ParametrisationOptions(max_refinement_steps = -1)
    @test_throws ArgumentError CheckpointOptions(""; problem_id = "valid")
    @test_throws ArgumentError CheckpointOptions("state"; problem_id = "")

    PS = MORFE.ParametrisationSolver
    CE = MORFE.CohomologicalEquations
    BLS = MORFE.BorderedLinearSolvers
    @test !PS._has_structural_factor_reuse(ComplexF64[1, 2])
    @test PS._has_structural_factor_reuse(ComplexF64[1, 1])
    @test PS._has_structural_factor_reuse(ComplexF64[0, 2])

    # Mathematically equal superharmonics need not be bit-equal when equal-eigenvalue
    # contributions are distributed differently. Structural grouping deliberately
    # chooses the first job's value as the canonical superharmonic for the whole group.
    repeated = ComplexF64[-0.7021951987576804 - 0.07258186804586991im,
        -0.7021951987576804 - 0.07258186804586991im]
    resonance_mask = SVector(false)
    representatives = PS._eigenvalue_representatives(repeated)
    key_split = PS._structural_factor_key(
        SVector(12, 14), resonance_mask, representatives)
    key_combined = PS._structural_factor_key(
        SVector(26, 0), resonance_mask, representatives)
    split_superharmonic = CE._superharmonic(SVector(12, 14), repeated)
    combined_superharmonic = CE._superharmonic(SVector(26, 0), repeated)
    @test isequal(key_split, key_combined)
    @test split_superharmonic != combined_superharmonic
    canonical_group = PS._SolveGroup(
        split_superharmonic, [PS._SolveJob(1, 0), PS._SolveJob(2, 0)])
    @test canonical_group.superharmonic == split_superharmonic
    @test length(canonical_group.jobs) == 2

    rounding_mset = MultiindexSet([SVector(12, 14), SVector(26, 0)])
    rounding_resonances = MORFE.Resonance.empty_resonance_set(rounding_mset, 1)
    rounding_context = (
        lambda_diag = repeated, resonance_set = rounding_resonances)
    rounding_groups = PS._group_solve_jobs(rounding_context, rounding_mset,
        [PS._SolveJob(1, 0), PS._SolveJob(2, 0)], [Int[], Int[]], Val(1))
    @test length(rounding_groups) == 1
    scheduled_first_superharmonic = CE._superharmonic(rounding_mset[1], repeated)
    @test rounding_groups[1].superharmonic == scheduled_first_superharmonic
    @test rounding_groups[1].jobs ==
          [PS._SolveJob(1, 0), PS._SolveJob(2, 0)]

    @testset "solve jobs use the final skip mask" begin
        mset = all_multiindices_up_to(2, 3; min_degree = 1)
        linear = Set(PS._linear_monomial_indices(mset))
        inactive = PS._build_conjugate_symmetry(
            NoConjugatePermutation(), linear, length(mset))
        inactive_jobs = PS._build_solve_jobs(inactive)
        @test all(job.conjugate_target == 0 for job in inactive_jobs)
        @test all(!inactive.skip_bits[job.index] for job in inactive_jobs)
        @test issorted(sum(mset[job.index]) for job in inactive_jobs)

        dictionary = MORFE.Multiindices.build_exponent_index_map(mset)
        active = PS._build_conjugate_symmetry(
            SVector(2, 1), linear, mset, dictionary)
        active_jobs = PS._build_solve_jobs(active)
        paired = findfirst(job -> job.conjugate_target != 0, active_jobs)
        @test paired !== nothing
        pair = active_jobs[paired]
        @test pair.conjugate_target > pair.index
        @test active.skip_bits[pair.conjugate_target]

        # Resume and external-direction setup mark additional entries after symmetry
        # discovery. Rebuilding jobs must observe those final mutations.
        active.skip_bits[pair.index] = true
        rebuilt = PS._build_solve_jobs(active)
        @test all(job.index != pair.index for job in rebuilt)
        @test all(job.conjugate_target != pair.conjugate_target for job in rebuilt)
        @test any(job.conjugate_target == 0 for job in rebuilt)
        @test_throws ArgumentError PS._build_solve_plan(
            nothing, nothing, nothing, nothing, :invalid, Val(1))
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
        @test_throws ArgumentError PS._embed_external_dynamics!(R, ext_poly, mset)
        ext_coefficients[1, cross] = 0
        @test PS._embed_external_dynamics!(R, ext_poly, mset) === nothing
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
        W_reference, R_reference = solve_parametrisation(
            sparse_model, mset, sd, rset;
            conjugate_permutation = nothing, options)

        W_supplied = deepcopy(W_reference)
        R_supplied = deepcopy(R_reference)
        nonlinear_indices = findall(i -> sum(mset[i]) >= 2, eachindex(mset.exponents))
        W_supplied.poly.coefficients[:, :, nonlinear_indices] .= 17 + 3im
        R_supplied.poly.coefficients[:, nonlinear_indices] .= -11 + 5im
        W_coefficients = W_supplied.poly.coefficients
        R_coefficients = R_supplied.poly.coefficients

        W_result, R_result = solve_parametrisation(
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
        @test_throws ArgumentError solve_with(
            quiet(backend = :umfpack); model = dense_model)
        Wd, Rd = solve_with(
            ParametrisationOptions(
                show_progress = false, verbose = false); model = dense_model)
        @test relerr(Wd.poly.coefficients, Wk.poly.coefficients) <= 1e-11
        @test relerr(Rd.poly.coefficients, Rk.poly.coefficients) <= 1e-11

        Wu, Ru = solve_with(quiet(backend = :umfpack, grouping = :off))
        Wug, Rug = solve_with(quiet(backend = :umfpack, grouping = :on))
        Wua, Rua = solve_with(quiet(backend = :umfpack, grouping = :auto))
        @test relerr(Wu.poly.coefficients, Wd.poly.coefficients) <= 1e-11
        @test relerr(Ru.poly.coefficients, Rd.poly.coefficients) <= 1e-11
        @test relerr(Wug.poly.coefficients, Wu.poly.coefficients) <= 1e-12
        @test relerr(Rug.poly.coefficients, Ru.poly.coefficients) <= 1e-12
        @test Wua.poly.coefficients == Wu.poly.coefficients
        @test Rua.poly.coefficients == Ru.poly.coefficients

        for backend in (:klu, :umfpack), _ in 1:5

            Wrepeat, Rrepeat = solve_with(quiet(backend = backend, grouping = :on))
            Wreference, Rreference = backend == :klu ? (Wkg, Rkg) : (Wug, Rug)
            @test Wrepeat.poly.coefficients == Wreference.poly.coefficients
            @test Rrepeat.poly.coefficients == Rreference.poly.coefficients
        end

        # Impossibly strict verification deterministically reaches the dense, KLU and
        # UMFPACK failure paths; allowing one correction also exercises refinement.
        for steps in (0, 1)
            strict_dense = ParametrisationOptions(
                residual_check = :backward_error, residual_tolerance = 1e-30,
                max_refinement_steps = steps, grouping = :off,
                show_progress = false, verbose = false)
            strict_klu = quiet(grouping = :off, residual_tolerance = 1e-30,
                max_refinement_steps = steps)
            strict_umfpack = quiet(backend = :umfpack, grouping = :off,
                residual_tolerance = 1e-30, max_refinement_steps = steps)
            @test_throws BLS._BorderedSolveFailure solve_with(strict_dense; model = dense_model)
            @test_throws BLS._BorderedSolveFailure solve_with(strict_klu)
            @test_throws BLS._BorderedSolveFailure solve_with(strict_umfpack)
        end
    end

    @testset "UMFPACK checkpoint diagnostics" begin
        mktempdir() do directory
            checkpoint = CheckpointOptions(joinpath(directory, "umfpack");
                problem_id = "umfpack-fixture", granularity = :degree)
            options = quiet(backend = :umfpack, grouping = :off, checkpoint = checkpoint)
            W, R = solve_with(options)
            @test all(isfinite, W.poly.coefficients)
            @test all(isfinite, R.poly.coefficients)
            manifest = _CheckpointTOML.parsefile(joinpath(checkpoint.path, "manifest.toml"))
            @test manifest["diagnostics"]["backend"] == "umfpack"
            @test manifest["completed_degrees"] == collect(1:5)
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
            W_same, R_same = solve_parametrisation(
                sparse_model, checkpoint_mset, sd, checkpoint_rset;
                initial_solution = (W_supplied, R_supplied), options)
            @test W_same === W_supplied
            @test R_same === R_supplied
            @test W_same.poly.coefficients == W0.poly.coefficients
            @test R_same.poly.coefficients == R0.poly.coefficients

            W_mismatch = deepcopy(W0)
            W_mismatch.poly.coefficients[1] += 1
            @test_throws ArgumentError solve_parametrisation(
                sparse_model, checkpoint_mset, sd, checkpoint_rset;
                initial_solution = (W_mismatch, deepcopy(R0)), options)

            R_mismatch = deepcopy(R0)
            R_mismatch.poly.coefficients[1] += 1
            @test_throws ArgumentError solve_parametrisation(
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
            W_completed, R_completed = solve_parametrisation(
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
            ordinary = solve_parametrisation(
                dense_model, mset, sd, rset;
                conjugate_permutation = permutation, options = base_options)
            mktempdir() do directory
                measured = solve_parametrisation(
                    dense_model, mset, sd, rset;
                    conjugate_permutation = permutation,
                    benchmark_dir = directory, options = base_options)
                @test measured[1].poly.coefficients ≈ ordinary[1].poly.coefficients
                @test measured[2].poly.coefficients ≈ ordinary[2].poly.coefficients

                mono = readlines(joinpath(directory, "benchmark_per_monomial.csv"))
                order = readlines(joinpath(directory, "benchmark_per_order.csv"))
                @test split(first(mono), ',') == ["order", "monomial_idx", "exponents",
                    "rhs_time_s", "rhs_alloc_bytes", "solve_time_s",
                    "solve_alloc_bytes", "factor_reused", "monomial_total_time_s",
                    "cumul_time_s"]
                @test length(split(first(order), ',')) == 10
                # The direct plan factorises every monomial.
                @test all(split(row, ',')[8] == "false" for row in mono[2:end])
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
            @test_throws ArgumentError solve_parametrisation(
                dense_model, mset, sd, rset;
                benchmark_dir = joinpath(directory, "benchmark"),
                options = conflicting)
            @test !ispath(checkpoint.path)
        end
    end

    @testset "automatic sparse backend" begin
        # UMFPACK's symbolic analysis decides: its symmetric strategy (symmetric pattern,
        # nonzero diagonal) selects UMFPACK, its unsymmetric strategy KLU.
        border(A, r) = (n = size(A, 1);
            [A sparse(ones(ComplexF64, n, r));
             sparse(ones(ComplexF64, r, n)) sparse(
                 ComplexF64(1)I, r, r)])
        m = 30
        line = spdiagm(-1 => fill(-1.0, m - 1), 0 => fill(2.0, m), 1 => fill(-1.0, m - 1))
        laplacian = ComplexF64.(kron(sparse(I, m, m), line) + kron(line, sparse(I, m, m)))
        @test BLS._auto_suitesparse_backend(border(laplacian, 2)) == :umfpack
        coupling = sparse(ComplexF64.(rand(MersenneTwister(1), 0:1, m^2 ÷ 4, m^2)))
        saddle = [laplacian transpose(coupling);
                  coupling spzeros(ComplexF64, m^2 ÷ 4, m^2 ÷ 4)]
        @test BLS._auto_suitesparse_backend(border(saddle, 2)) == :klu
        @test BLS._auto_suitesparse_backend(spzeros(ComplexF64, 3, 2)) == :klu

        # The solver state applies the choice to the operators it is built from.
        state(model, options) = PS._make_sparse_solver(
            typeof(first(model.linear_terms)), model.linear_terms,
            size(first(model.linear_terms), 1), 2, options)
        @test state(sparse_model, ParametrisationOptions()).backend isa BLS.UMFPACKBackend
        @test state(sparse_model, quiet(backend = :klu)).backend isa BLS.KLUBackend
        saddle_model = NthOrderModel(
            (sparse(real(saddle)), sparse(1.0I, size(saddle)...)), (cubic_saddle,))
        @test state(saddle_model, ParametrisationOptions()).backend isa BLS.KLUBackend
    end

    @testset "bordered matrices from order-one data" begin
        mset = all_multiindices_up_to(2, 3; min_degree = 1)
        rset = MORFE.Resonance.build_resonance_set(sparse_model, mset, sd, resonance)
        targets = [SVector(2, 0), SVector(2, 1)]
        matrices = PS._bordered_matrices(sparse_model, mset, sd, rset, targets)
        FOM, ROM = size(B0, 1), 2
        regular, resonant = matrices[SVector(2, 0)], matrices[SVector(2, 1)]
        # One constant pattern, whatever the monomial.
        @test size(regular) == size(resonant) == (FOM + ROM, FOM + ROM)
        @test regular.colptr == resonant.colptr && regular.rowval == resonant.rowval
        # z² is resonant with no master mode, so its border columns stay empty; z² z̄ is
        # resonant with exactly one, whose column C(s) is filled.
        border(M) = [norm(M[1:FOM, FOM + r]) for r in 1:ROM]
        @test all(iszero, border(regular))
        @test count(!iszero, border(resonant)) == 1
        @test_throws ArgumentError PS._bordered_matrices(
            sparse_model, mset, sd, rset, [SVector(4, 0)])
        @test_throws ArgumentError PS._bordered_matrices(
            dense_model, mset, sd, rset, targets)
    end

    @testset "cross-degree factor reuse" begin
        @test ParametrisationOptions(grouping = :across_degrees).grouping ==
              :across_degrees
        @test_throws ArgumentError ParametrisationOptions(grouping = :across_degree)

        # Equal keys run back to back when the dependencies allow, a key that can finish in
        # one run opens first, a key splits only when it must, and a cycle is reported
        # instead of looping.
        @test PS._schedule_key_runs([1, 2, 1, 2, 1], [1, 2],
            [Int[], Int[], [1], [2, 3], [3]]) == [[1, 3, 5], [2, 4]]
        split_dependencies = [Int[], Int[], [2], Int[]]
        split_runs = PS._schedule_key_runs([1, 2, 1, 2], [1, 2], split_dependencies)
        @test split_runs == [[2, 4], [1, 3]]
        @test PS._schedule_key_runs([1, 2, 1], [1, 2], [Int[], [1], [2]]) ==
              [[1], [2], [3]]
        # Whether executing job numbers in `order` runs every job after its dependencies.
        function respects_dependencies(order, dependencies)
            rank = invperm(order)
            return all(rank[d] < rank[n] for n in order for d in dependencies[n])
        end
        @test respects_dependencies(reduce(vcat, split_runs), split_dependencies)
        @test !respects_dependencies([1, 3, 2, 4], split_dependencies)
        @test PS._schedule_key_runs([1, 1], [1], [[2], [1]]) === nothing

        cubic = MultilinearMap(_checkpoint_test_cubic!, (3, 0))
        stiffness = MultilinearMap(_cross_degree_stiffness!, (1, 0), 1)
        load = MultilinearMap(_cross_degree_load!, (0, 0), 1)
        degree = 4
        mset = all_multiindices_up_to(4, degree; min_degree = 1)
        nonlinear = [α for α in mset.exponents if sum(α) >= 2]

        function fixture(external_system)
            terms = (cubic, stiffness, load)
            dense = NthOrderModel((B0, B1, B2), terms, external_system)
            sparse_model = NthOrderModel(map(sparse, (B0, B1, B2)), terms, external_system)
            spectral = SpectralData(dense,
                spectrum(dense; solver = DefaultEigensolver());
                master = master_by_sorting(2))
            return dense, sparse_model, spectral
        end

        # Solve in benchmark mode, which records whether each solve factorised.
        function measured_solve(model, spectral, permutation, grouping, backend;
                multiindices = mset)
            rset = MORFE.Resonance.build_resonance_set(
                model, multiindices, spectral, resonance)
            options = ParametrisationOptions(;
                backend, grouping, show_progress = false, verbose = false)
            return mktempdir() do directory
                W, R = solve_parametrisation(model, multiindices, spectral, rset;
                    conjugate_permutation = permutation,
                    benchmark_dir = directory, options)
                rows = readlines(joinpath(directory, "benchmark_per_monomial.csv"))[2:end]
                n_factorised = count(row -> split(row, ',')[8] == "false", rows)
                (; W = W.poly.coefficients, R = R.poly.coefficients,
                    n_jobs = length(rows), n_factorised)
            end
        end

        # Two frozen parameters: every monomial differing only in its parameter exponents
        # has the same bordered matrix. Within a degree that leaves one factorisation per
        # (degree, master exponents); across degrees, one per master exponents.
        frozen_dense, frozen_sparse, frozen_sd = fixture(ExternalSystem((0.0, 0.0)))
        for permutation in (nothing, [2, 1, 3, 4])
            primaries = permutation === nothing ? nonlinear :
                        [α for α in nonlinear if α[1] >= α[2]]
            expected = Dict(
                :off => length(primaries),
                :on => length(unique((sum(α), α[1], α[2]) for α in primaries)),
                :across_degrees => length(unique((α[1], α[2]) for α in primaries)))
            # `:auto` takes whichever plan needs the fewest factorisations: here, across.
            expected[:auto] = expected[:across_degrees]
            for (model, backend) in ((frozen_sparse, :klu), (frozen_sparse, :umfpack),
                (frozen_dense, :auto))
                direct = measured_solve(model, frozen_sd, permutation, :off, backend)
                for grouping in (:off, :on, :across_degrees, :auto)
                    result = measured_solve(
                        model, frozen_sd, permutation, grouping, backend)
                    @test result.n_jobs == length(primaries)
                    @test result.n_factorised == expected[grouping]
                    # Zero eigenvalues add exact zeros to the superharmonic, so every
                    # schedule assembles bit-identical matrices.
                    @test result.W == direct.W
                    @test result.R == direct.R
                end
            end
        end

        # `r₁` is driven by the frozen `θ` (Λ₁₂ ≠ 0, distinct eigenvalues): the monomial
        # (…, m, k) reads (…, m + 1, k − 1), which belongs to a later key. The scheduler
        # must interleave the keys, and a key-major order would read unsolved data.
        coupled_polynomial = let ms = all_multiindices_up_to(2, 1; min_degree = 1)
            A = ComplexF64[-0.5 1.0; 0.0 0.0]
            coefficients = zeros(ComplexF64, 2, length(ms))
            for (j, α) in enumerate(ms.exponents)
                column = findfirst(==(1), α)
                column === nothing || (coefficients[:, j] = A[:, column])
            end
            MORFE.Polynomials.DensePolynomial(coefficients, ms)
        end
        _, coupled_sparse, coupled_sd = fixture(ExternalSystem(coupled_polynomial))
        coupled_direct = measured_solve(coupled_sparse, coupled_sd, nothing, :off, :klu)
        # No grouping saves a factorisation here, so `:auto` keeps the direct order.
        for grouping in (:on, :across_degrees, :auto)
            result = measured_solve(coupled_sparse, coupled_sd, nothing, grouping, :klu)
            @test result.n_factorised == result.n_jobs
            @test result.W == coupled_direct.W
            @test result.R == coupled_direct.R
        end

        # Two frozen parameters, the second driving `r₁` (external order θ₁, r₁, θ₂). Within
        # a degree, e.g. (1,0,1,0,0) and (1,0,0,0,1) share a key, but the second reads
        # (1,0,0,1,0) of another key that falls between them in graded order. Grouping
        # within degrees must run that key first rather than drop the grouping: 65
        # factorisations, as many as an order that ignores the dependencies would form.
        drive = let ms = all_multiindices_up_to(3, 1; min_degree = 1)
            A = ComplexF64[0 0 0; 0 -0.5 1; 0 0 0]
            coefficients = zeros(ComplexF64, 3, length(ms))
            for (j, α) in enumerate(ms.exponents)
                column = findfirst(==(1), α)
                column === nothing || (coefficients[:, j] = A[:, column])
            end
            ExternalSystem(MORFE.Polynomials.DensePolynomial(coefficients, ms))
        end
        driven_terms = (cubic, MultilinearMap(_cross_degree_stiffness3!, (1, 0), 1),
            MultilinearMap(_cross_degree_load3!, (0, 0), 1))
        driven_dense = NthOrderModel((B0, B1, B2), driven_terms, drive)
        driven_sparse = NthOrderModel(map(sparse, (B0, B1, B2)), driven_terms, drive)
        driven_sd = SpectralData(driven_dense,
            spectrum(driven_dense; solver = DefaultEigensolver());
            master = master_by_sorting(2))
        driven_mset = all_multiindices_up_to(5, degree; min_degree = 1)
        driven(grouping) = measured_solve(
            driven_sparse, driven_sd, nothing, grouping, :klu;
            multiindices = driven_mset)
        driven_direct = driven(:off)
        driven_results = Dict(g => driven(g) for g in (:on, :across_degrees, :auto))
        for result in values(driven_results)
            @test result.W == driven_direct.W
            @test result.R == driven_direct.R
        end
        @test driven_results[:on].n_factorised == 65 < driven_direct.n_factorised
        @test driven_results[:auto].n_factorised == min(
            driven_results[:on].n_factorised, driven_results[:across_degrees].n_factorised)

        # A cross-degree checkpoint resumes from any subset of committed groups, even
        # though its chunks and degree markers are written out of degree order.
        mktempdir() do directory
            path = joinpath(directory, "cross-degree")
            checkpoint = CheckpointOptions(path;
                problem_id = "cross-degree", granularity = :factor_group)
            options = quiet(grouping = :across_degrees, checkpoint = checkpoint)
            solve_frozen() = parametrise(frozen_sparse, frozen_sd, degree;
                resonance, options)
            W0, R0 = solve_frozen()
            manifest_path = joinpath(path, "manifest.toml")
            manifest = _CheckpointTOML.parsefile(manifest_path)
            @test manifest["completed_degrees"] == collect(1:degree)
            chunks = manifest["chunks"]
            @test length(chunks) > 2
            # Drop two groups that span several degrees, and the markers of every degree
            # they touched; the resumed run must solve those groups again and re-mark
            # their degrees.
            chunk_degrees(chunk) = Set(sum(W0.poly.multiindex_set[i])
            for i in chunk["indices"])
            dropped = filter(chunk -> length(chunk_degrees(chunk)) > 1, chunks)[1:2]
            reopened = union((chunk_degrees(chunk) for chunk in dropped)...)
            @test length(reopened) > 1
            for chunk in dropped
                rm(joinpath(path, "chunks", chunk["file"]))
            end
            manifest["chunks"] = filter(chunk -> !(chunk in dropped), chunks)
            manifest["completed_degrees"] = setdiff(1:degree, reopened)
            open(manifest_path, "w") do io
                _CheckpointTOML.print(io, manifest; sorted = true)
            end
            W1, R1 = solve_frozen()
            @test relerr(W1.poly.coefficients, W0.poly.coefficients) <= 1e-12
            @test relerr(R1.poly.coefficients, R0.poly.coefficients) <= 1e-12
            @test _CheckpointTOML.parsefile(manifest_path)["completed_degrees"] ==
                  collect(1:degree)
        end
    end
end
