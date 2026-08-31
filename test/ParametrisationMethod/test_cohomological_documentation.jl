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
    local_exports = filter(
        symbol -> _is_locally_defined_api(_CE_DOCS, symbol),
        names(_CE_DOCS; all = false))
    undocumented = intersect(
        Set(local_exports),
        Set(Base.Docs.undocumented_names(_CE_DOCS; private = false)))
    @test isempty(undocumented)

    for type_name in (:CohomologicalBuffers, :CohomologicalContext,
        :SparseLinearSolverState)
        type_doc = _binding_doc_text(_CE_DOCS, type_name)
        @test !isempty(type_doc)
        for field in fieldnames(Base.unwrap_unionall(getfield(_CE_DOCS, type_name)))
            @test occursin("`$(field)", type_doc)
        end
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

    @test !isdefined(MORFE, :CohomologicalSolverConfig)
    @test !isdefined(MORFE, :CohomologicalCheckpoint)
    @test !isdefined(_CE_DOCS, :CohomologicalSolverConfig)
    @test !isdefined(_CE_DOCS, :CohomologicalCheckpoint)

    checkpoint = CheckpointOptions("unused-documentation-test";
        problem_id = "documentation-test")
    solver_options = ParametrisationOptions(
        backend = :klu,
        grouping = :on,
        residual_check = :backward_error,
        residual_tolerance = 1e-9)
    translated = _CE_DOCS._translate_legacy_options(
        ParametrisationOptions(),
        (; solver_config = solver_options, checkpoint,
            validate_mset = false, show_progress = false,
            verbose = false, setup_io = devnull))
    @test translated.backend == :klu
    @test translated.grouping == :on
    @test translated.residual_check == :backward_error
    @test translated.residual_tolerance == 1e-9
    @test translated.checkpoint === checkpoint
    @test !translated.validate_mset
    @test !translated.show_progress
    @test !translated.verbose
    @test translated.setup_io === devnull
    @test_throws ArgumentError _CE_DOCS._translate_legacy_options(
        ParametrisationOptions(), (; solver_config = :klu))

    generated_page = read(
        joinpath(@__DIR__, "..", "..", "website",
            "documentation.html"), String)
    @test !occursin("href=\"@ref\"", generated_page)
    @test !occursin("CohomologicalSolverConfig", generated_page)
    @test !occursin("CohomologicalCheckpoint", generated_page)
end
