"""Configuration of the sparse bordered cohomological solver."""
struct CohomologicalSolverConfig
    backend::Symbol
    residual_tolerance::Union{Nothing, Float64}
    group_superharmonics::Bool
    diagnostics_path::Union{Nothing, String}
    function CohomologicalSolverConfig(; backend::Symbol = :auto,
            residual_tolerance::Union{Nothing, Real} = nothing,
            group_superharmonics::Bool = false,
            diagnostics_path::Union{Nothing, AbstractString} = nothing)
        backend in (:auto, :klu, :umfpack, :pardiso) || throw(ArgumentError(
            "backend must be :auto, :klu, :umfpack, or :pardiso"))
        tolerance = isnothing(residual_tolerance) ? nothing : Float64(residual_tolerance)
        isnothing(tolerance) || tolerance > 0 ||
            throw(ArgumentError(
                "residual_tolerance must be positive or nothing"))
        path = isnothing(diagnostics_path) ? nothing : String(diagnostics_path)
        new(backend, tolerance, group_superharmonics, path)
    end
end

"""Atomic degree-boundary checkpoint configuration for `parametrise`."""
struct CohomologicalCheckpoint
    path::String
    id::String
    resume::Bool
    function CohomologicalCheckpoint(path::AbstractString;
            id::AbstractString, resume::Bool = true)
        isempty(path) && throw(ArgumentError("checkpoint path must not be empty"))
        isempty(id) && throw(ArgumentError("checkpoint id must not be empty"))
        new(String(path), String(id), resume)
    end
end
