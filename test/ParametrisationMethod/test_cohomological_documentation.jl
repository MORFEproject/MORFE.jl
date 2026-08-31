using Test
using MORFE

const _CE_DOCS = MORFE.CohomologicalEquations

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
    # Julia 1.10 exposes these language-generated module bindings as local definitions.
    language_generated_bindings = Set((:eval, :include))
    local_definitions = filter(
        symbol -> !startswith(string(symbol), "#") &&
                  symbol ∉ language_generated_bindings &&
                  _is_locally_defined_api(_CE_DOCS, symbol),
        names(_CE_DOCS; all = true))
    # `Base.Docs.undocumented_names` is unavailable on the supported Julia 1.10
    # lower bound and checks exports only, so inspect every locally defined function and
    # type through the binding metadata directly.
    undocumented = Set(filter(
        symbol -> isempty(_binding_doc_text(_CE_DOCS, symbol)),
        local_definitions))
    @test isempty(undocumented)

    documented_state_types = (
        :CheckpointOptions,
        :CheckpointSession,
        :ParametrisationOptions,
        :InvarianceOperators,
        :OrthogonalityOperators,
        :LowerOrderResources,
        :CohomologicalBuffers,
        :PardisoBackend,
        :SparseLinearSolverState,
        :CohomologicalContext,
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
        :_BenchmarkSolveObserver
    )
    for type_name in documented_state_types
        type_doc = _binding_doc_text(_CE_DOCS, type_name)
        @test !isempty(type_doc)
        for field in fieldnames(Base.unwrap_unionall(getfield(_CE_DOCS, type_name)))
            @test occursin("`$(field)", type_doc)
        end
    end

    key_internal_contracts = (
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
        :_solve_cohomological_equations!,
        :_solve_cohomological_equations_typed!,
        :_assemble_nonlinear_rhs!,
        :_solve_prepared_system!,
        :_monomial_metrics,
        :_finalise_monomial!,
        :_cached_klu_factor
    )
    for function_name in key_internal_contracts
        @test !isempty(_binding_doc_text(_CE_DOCS, function_name))
    end

    backend_doc = _binding_doc_text(_CE_DOCS, :AbstractSparseBackend)
    state_doc = _binding_doc_text(_CE_DOCS, :SparseLinearSolverState)
    context_doc = _binding_doc_text(_CE_DOCS, :CohomologicalContext)
    @test occursin("Internal dispatch root", backend_doc)
    @test !occursin("pardiso::Any", state_doc)
    @test occursin("pardiso_matrix::Any", state_doc)
    @test occursin("generalised **right**", context_doc)
    @test occursin("master right eigenmodes", context_doc)
    @test occursin("left eigenmodes are represented separately", context_doc)

    problem_doc = _binding_doc_text(_CE_DOCS, :solve_cohomological_problem)
    @test occursin("storage reuse", problem_doc)
    @test occursin("not implicit", problem_doc)
    @test occursin("checkpoint-committed", problem_doc)

    @test !isdefined(MORFE, :CohomologicalSolverConfig)
    @test !isdefined(MORFE, :CohomologicalCheckpoint)
    @test !isdefined(_CE_DOCS, :CohomologicalSolverConfig)
    @test !isdefined(_CE_DOCS, :CohomologicalCheckpoint)

    @test !isdefined(_CE_DOCS, :_translate_legacy_options)

    generated_page = read(
        joinpath(@__DIR__, "..", "..", "website",
            "documentation.html"), String)
    @test !occursin("href=\"@ref\"", generated_page)
    for removed_name in (
        "_translate_legacy_options",
        "CohomologicalSolverConfig",
        "CohomologicalCheckpoint",
        "primary_pairs"
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
end
