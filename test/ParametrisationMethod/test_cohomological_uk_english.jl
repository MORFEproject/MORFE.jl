using Test

@testset "Cohomological UK English contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    source_root = joinpath(
        repo_root, "src", "ParametrisationMethod", "CohomologicalEquations")
    test_root = joinpath(repo_root, "test", "ParametrisationMethod")

    spellings = Dict(
        "behavior" => "behaviour",
        "behaviors" => "behaviours",
        "characterization" => "characterisation",
        "characterize" => "characterise",
        "factorization" => "factorisation",
        "factorizations" => "factorisations",
        "finalization" => "finalisation",
        "finalize" => "finalise",
        "finalized" => "finalised",
        "initialization" => "initialisation",
        "initialize" => "initialise",
        "initialized" => "initialised",
        "materialization" => "materialisation",
        "materialize" => "materialise",
        "normalization" => "normalisation",
        "normalize" => "normalise",
        "optimization" => "optimisation",
        "optimize" => "optimise",
        "organization" => "organisation",
        "organize" => "organise",
        "serialization" => "serialisation",
        "serialize" => "serialise",
    )

    paths = vcat(
        filter(path -> endswith(path, ".jl"), readdir(source_root; join = true)),
        filter(path -> endswith(path, ".jl"), readdir(test_root; join = true)),
    )
    scanner_path = normpath(@__FILE__)
    violations = String[]
    for path in paths
        normpath(path) == scanner_path && continue
        for (line_number, line) in enumerate(eachline(path))
            # This is an exact quotation of Julia's externally defined error message.
            occursin("cannot be finalized because they are not mutable", line) && continue
            for (us_spelling, uk_spelling) in spellings
                pattern = Regex("(?i)(?<![A-Za-z])" * us_spelling * "(?![A-Za-z])")
                occursin(pattern, line) || continue
                relative_path = relpath(path, repo_root)
                push!(violations,
                    "$relative_path:$line_number uses '$us_spelling'; use '$uk_spelling'")
            end
        end
    end
    @test isempty(violations)
end
