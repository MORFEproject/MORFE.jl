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

    @test !isdefined(_CE_DOCS, :_translate_legacy_options)

    generated_page = read(
        joinpath(@__DIR__, "..", "..", "website",
            "documentation.html"), String)
    @test !occursin("href=\"@ref\"", generated_page)
    @test !occursin("CohomologicalSolverConfig", generated_page)
    @test !occursin("CohomologicalCheckpoint", generated_page)
end
