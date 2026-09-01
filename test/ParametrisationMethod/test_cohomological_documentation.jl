using Test
using MORFE

const _CE_DOCS = MORFE.CohomologicalEquations
const _BLS_DOCS = MORFE.BorderedLinearSolvers
const _PS_DOCS = MORFE.ParametrisationSolver

function _binding_doc_text(mod::Module, symbol::Symbol)
    metadata = Base.Docs.meta(mod)
    binding = Base.Docs.Binding(mod, symbol)
    haskey(metadata, binding) || return ""
    multidoc = metadata[binding]
    return join(
        join(part for part in docstr.text if part isa AbstractString)
    for docstr in values(multidoc.docs))
end

function _is_locally_defined_api(mod::Module, symbol::Symbol)
    value = getfield(mod, symbol)
    return (value isa Function || value isa DataType || value isa UnionAll) &&
           parentmodule(value) === mod
end

@testset "Cohomological documentation contracts" begin
    equation_module_doc = _binding_doc_text(_CE_DOCS, nameof(_CE_DOCS))
    linear_module_doc = _binding_doc_text(_BLS_DOCS, nameof(_BLS_DOCS))
    workflow_module_doc = _binding_doc_text(_PS_DOCS, nameof(_PS_DOCS))
    @test occursin("cohomological equations", equation_module_doc)
    @test occursin(r"does not\s+assemble cohomological equations", linear_module_doc)
    @test occursin("checkpoint restoration", workflow_module_doc)
    @test occursin("Source layout", equation_module_doc)
    @test occursin("Source organisation", linear_module_doc)
    @test occursin("Source organisation", workflow_module_doc)

    # Julia 1.10 exposes these language-generated module bindings as local definitions.
    language_generated_bindings = Set((:eval, :include))
    # `Base.Docs.undocumented_names` is unavailable on the supported Julia 1.10 lower
    # bound and checks exports only. Inspect every function and type in its owning module;
    # compatibility imports in `CohomologicalEquations` intentionally retain their owner's
    # documentation binding rather than duplicating it.
    for mod in (_CE_DOCS, _BLS_DOCS, _PS_DOCS)
        local_definitions = filter(
            symbol -> !startswith(string(symbol), "#") &&
                      symbol ∉ language_generated_bindings &&
                      _is_locally_defined_api(mod, symbol),
            names(mod; all = true))
        undocumented = Set(filter(
            symbol -> isempty(_binding_doc_text(mod, symbol)),
            local_definitions))
        @test isempty(undocumented)
    end

    documented_state_types = Dict(
        _PS_DOCS => (
            :CheckpointOptions,
            :CheckpointSession,
            :ParametrisationOptions,
            :ConjugateSymmetryData,
            :_SimpleProgress,
            :_SolveJob,
            :_DirectSolvePlan,
            :_SolveGroup,
            :_GroupedSolvePlan,
            :StructuralFactorKey,
            :_CompositeSolveObserver,
            :_ProgressSolveObserver,
            :_CheckpointSolveObserver,
            :_OrderAccum,
            :_BenchmarkSolveObserver),
        _CE_DOCS => (
            :InvarianceOperators,
            :OrthogonalityOperators,
            :LowerOrderResources,
            :CohomologicalBuffers,
            :CohomologicalContext),
        _BLS_DOCS => (
            :UMFPACKBackend,
            :PardisoBackend,
            :_BorderedSolveFailure,
            :SparseLinearSolverState)
    )
    for (mod, type_names) in documented_state_types
        for type_name in type_names
            type_doc = _binding_doc_text(mod, type_name)
            @test !isempty(type_doc)
            for field in fieldnames(Base.unwrap_unionall(getfield(mod, type_name)))
                @test occursin("`$(field)", type_doc)
            end
        end
    end

    key_internal_contracts = Dict(
        _PS_DOCS => (
            :_replace_options,
            :_fingerprint_value!,
            :_atomic_manifest,
            :_completed_degrees,
            :_prepare_solution_storage,
            :_restore_solution_checkpoint!,
            :_prepare_shared_resources,
            :_prepare_master_operators,
            :_prepare_external_directions!,
            :_build_complete_context,
            :_group_solve_jobs,
            :_compose_observers,
            :_execute_solve_plan!,
            :_execute_solve_schedule!,
            :_execute_benchmarked_schedule!,
            :_execute_problem_schedule!),
        _CE_DOCS => (
            :_assemble_nonlinear_rhs!,
            :_solve_prepared_system!,
            :_monomial_metrics,
            :_finalise_monomial!),
        _BLS_DOCS => (
            :_cached_klu_factor,
            :_cached_umfpack_factor,
            :_refactorise_umfpack!)
    )
    for (mod, function_names) in key_internal_contracts
        for function_name in function_names
            @test !isempty(_binding_doc_text(mod, function_name))
        end
    end

    backend_doc = _binding_doc_text(_BLS_DOCS, :AbstractSparseBackend)
    state_doc = _binding_doc_text(_BLS_DOCS, :SparseLinearSolverState)
    context_doc = _binding_doc_text(_CE_DOCS, :CohomologicalContext)
    @test occursin("Internal dispatch root", backend_doc)
    @test !occursin("pardiso::Any", state_doc)
    @test occursin("pardiso_matrix::Any", state_doc)
    @test occursin("generalised **right**", context_doc)
    @test occursin("master right eigenmodes", context_doc)
    @test occursin("left eigenmodes are represented separately", context_doc)

    problem_doc = _binding_doc_text(_PS_DOCS, :solve_parametrisation)
    @test occursin("storage reuse", problem_doc)
    @test occursin("not implicit", problem_doc)
    @test occursin("checkpoint-committed", problem_doc)

    @test !isdefined(MORFE, :CohomologicalSolverConfig)
    @test !isdefined(MORFE, :CohomologicalCheckpoint)
    @test !isdefined(_CE_DOCS, :CohomologicalSolverConfig)
    @test !isdefined(_CE_DOCS, :CohomologicalCheckpoint)

    @test !isdefined(_CE_DOCS, :_translate_legacy_options)

    # Each symbol is available only from its owning module; the former compatibility
    # namespace and workflow names have deliberately disappeared.
    for removed_name in (
        :solve_cohomological_problem,
        :solve_cohomological_equations!,
        :solve_cohomological_equations_benchmarked!,
        :_solve_cohomological_equations!,
        :_solve_cohomological_equations_typed!
    )
        @test !isdefined(_CE_DOCS, removed_name)
    end
    @test !isdefined(MORFE, :solve_cohomological_problem)
    @test !isdefined(MORFE, :solve_cohomological_equations!)
    @test !isdefined(MORFE, :solve_cohomological_equations_benchmarked!)
    @test parentmodule(MORFE.solve_single_monomial!) === _CE_DOCS
    @test parentmodule(MORFE.SparseLinearSolverState) === _BLS_DOCS
    @test parentmodule(MORFE.solve_parametrisation) === _PS_DOCS

    generated_page = read(
        joinpath(@__DIR__, "..", "..", "website",
            "documentation.html"), String)
    @test !occursin("href=\"@ref\"", generated_page)
    for module_name in (
        "BorderedLinearSolvers",
        "CohomologicalEquations",
        "ParametrisationSolver"
    )
        @test occursin("id=\"$(module_name)\" class=\"doc-module-h\"", generated_page)
        @test occursin("data-module=\"$(module_name)\"", generated_page)
    end
    for removed_name in (
        "_translate_legacy_options",
        "CohomologicalSolverConfig",
        "CohomologicalCheckpoint",
        "primary_pairs",
        "solve_cohomological_problem",
        "solve_cohomological_equations!",
        "solve_cohomological_equations_benchmarked!",
        "_solve_cohomological_equations_typed!",
        "SolveProgress.jl"
    )
        @test !occursin(removed_name, generated_page)
    end

    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    source_link_pattern = r"""href="https://github\.com/MORFEproject/MORFE\.jl/blob/main/([^"#]+)#L\d+"""
    source_paths = Set(
        normpath(joinpath(repo_root, match.captures[1]))
    for match in eachmatch(source_link_pattern, generated_page))
    @test !isempty(source_paths)
    for source_path in source_paths
        @test isfile(source_path)
    end
    documented_modules = (
        ("BorderedLinearSolvers", "BorderedLinearSolvers.jl"),
        ("CohomologicalEquations", "CohomologicalEquations.jl"),
        ("ParametrisationSolver", "ParametrisationSolver.jl")
    )
    expected_source_paths = Set{String}()
    for (directory_name, module_filename) in documented_modules
        directory = joinpath(
            repo_root, "src", "ParametrisationMethod", directory_name)
        module_path = normpath(joinpath(directory, module_filename))
        push!(expected_source_paths, module_path)
        module_source = read(module_path, String)
        for included in eachmatch(r"""include\("([^"]+\.jl)"\)""", module_source)
            push!(expected_source_paths,
                normpath(joinpath(directory, included.captures[1])))
        end
    end
    @test issubset(expected_source_paths, source_paths)
end
