"""Durable checkpoint policy for a cohomological reduction."""
struct CheckpointOptions
    path::String
    problem_id::String
    resume::Bool
    granularity::Symbol
    function CheckpointOptions(path::AbstractString;
            problem_id::AbstractString,
            resume::Bool = true,
            granularity::Symbol = :factor_group)
        isempty(path) && throw(ArgumentError("checkpoint path must not be empty"))
        isempty(problem_id) && throw(ArgumentError("checkpoint problem_id must not be empty"))
        granularity in (:degree, :factor_group) || throw(ArgumentError(
            "checkpoint granularity must be :degree or :factor_group"))
        new(String(path), String(problem_id), resume, granularity)
    end
end

"""
Operational controls for [`parametrise`](@ref).

Mathematical choices (`resonance` and `conjugate_permutation`) deliberately remain
explicit keywords of `parametrise`; this object contains validation, execution,
checkpoint, and presentation policy only.
"""
struct ParametrisationOptions{IOType <: IO}
    backend::Symbol
    grouping::Symbol
    residual_check::Symbol
    residual_tolerance::Union{Nothing, Real}
    max_refinement_steps::Int
    validate_mset::Bool
    checkpoint::Union{Nothing, CheckpointOptions}
    show_progress::Bool
    verbose::Bool
    setup_io::IOType
end

function ParametrisationOptions(;
        backend::Symbol = :auto,
        grouping::Symbol = :auto,
        residual_check::Symbol = :off,
        residual_tolerance::Union{Nothing, Real} = nothing,
        max_refinement_steps::Integer = 3,
        validate_mset::Bool = true,
        checkpoint::Union{Nothing, CheckpointOptions} = nothing,
        show_progress::Bool = true,
        verbose::Bool = true,
        setup_io::IO = stderr)
    backend in (:auto, :klu, :pardiso) || throw(ArgumentError(
        "backend must be :auto, :klu, or :pardiso; UMFPACK is not a shared MORFE backend"))
    grouping in (:auto, :off, :on) || throw(ArgumentError(
        "grouping must be :auto, :off, or :on"))
    residual_check in (:backward_error, :off) || throw(ArgumentError(
        "residual_check must be :backward_error or :off"))
    isnothing(residual_tolerance) || residual_tolerance > 0 || throw(ArgumentError(
        "residual_tolerance must be positive or nothing"))
    max_refinement_steps >= 0 || throw(ArgumentError(
        "max_refinement_steps must be non-negative"))
    return ParametrisationOptions(
        backend, grouping, residual_check, residual_tolerance,
        Int(max_refinement_steps), validate_mset, checkpoint,
        show_progress, verbose, setup_io)
end

function _replace_options(options::ParametrisationOptions;
        backend = options.backend,
        grouping = options.grouping,
        residual_check = options.residual_check,
        residual_tolerance = options.residual_tolerance,
        max_refinement_steps = options.max_refinement_steps,
        validate_mset = options.validate_mset,
        checkpoint = options.checkpoint,
        show_progress = options.show_progress,
        verbose = options.verbose,
        setup_io = options.setup_io)
    return ParametrisationOptions(;
        backend, grouping, residual_check, residual_tolerance,
        max_refinement_steps, validate_mset, checkpoint,
        show_progress, verbose, setup_io)
end

function _translate_legacy_options(options::ParametrisationOptions, kwargs)
    isempty(kwargs) && return options
    allowed = (:solver_config, :checkpoint, :validate_mset,
        :show_progress, :verbose, :setup_io)
    for key in keys(kwargs)
        key in allowed || throw(UndefKeywordError(key))
    end
    Base.depwarn(
        "operational parametrise keywords are deprecated; place them in ParametrisationOptions",
        :parametrise)
    result = options
    if haskey(kwargs, :solver_config)
        old = kwargs[:solver_config]
        old isa ParametrisationOptions || throw(ArgumentError(
            "deprecated solver_config must be constructed with CohomologicalSolverConfig"))
        result = _replace_options(result;
            backend = old.backend,
            grouping = old.grouping,
            residual_check = old.residual_check,
            residual_tolerance = old.residual_tolerance,
            max_refinement_steps = old.max_refinement_steps)
    end
    haskey(kwargs, :checkpoint) &&
        (result = _replace_options(result; checkpoint = kwargs[:checkpoint]))
    haskey(kwargs, :validate_mset) &&
        (result = _replace_options(result; validate_mset = kwargs[:validate_mset]))
    haskey(kwargs, :show_progress) &&
        (result = _replace_options(result; show_progress = kwargs[:show_progress]))
    haskey(kwargs, :verbose) &&
        (result = _replace_options(result; verbose = kwargs[:verbose]))
    haskey(kwargs, :setup_io) &&
        (result = _replace_options(result; setup_io = kwargs[:setup_io]))
    return result
end

# One compatibility release. These constructors forward into the single options
# implementation; the deprecated keywords themselves are removed from `parametrise`.
function CohomologicalSolverConfig(; backend::Symbol = :auto,
        residual_tolerance::Union{Nothing, Real} = nothing,
        group_superharmonics::Bool = false,
        diagnostics_path = nothing)
    Base.depwarn(
        "CohomologicalSolverConfig is deprecated; use ParametrisationOptions", :CohomologicalSolverConfig)
    isnothing(diagnostics_path) || Base.depwarn(
        "diagnostics_path is no longer public; diagnostics belong to developer tooling", :CohomologicalSolverConfig)
    return ParametrisationOptions(;
        backend,
        grouping = group_superharmonics ? :on : :off,
        residual_check = isnothing(residual_tolerance) ? :off : :backward_error,
        residual_tolerance)
end

function CohomologicalCheckpoint(path::AbstractString;
        id::AbstractString, resume::Bool = true)
    Base.depwarn(
        "CohomologicalCheckpoint is deprecated; use CheckpointOptions", :CohomologicalCheckpoint)
    return CheckpointOptions(path; problem_id = id, resume)
end
