# User-facing execution, validation, output, and checkpoint policy.

"""
	CheckpointOptions(path; problem_id, resume = true, granularity = :factor_group)

Configure durable checkpoints for [`parametrise`](@ref). Pass the resulting object as
the `checkpoint` field of [`ParametrisationOptions`](@ref).

# Arguments

- `path::AbstractString` — checkpoint directory. MORFE creates it and stores a manifest
  plus checksummed coefficient chunks beneath it.
- `problem_id::AbstractString` — required, non-empty identifier for the physical problem.
  It is checked when resuming so that a checkpoint cannot silently be used for a different
  run.
- `resume::Bool = true` — restore compatible completed work found at `path`. With
  `resume = false`, MORFE refuses an existing checkpoint manifest instead of overwriting
  it; use a new directory for a fresh run.
- `granularity::Symbol = :factor_group` — when data are committed:
  - `:factor_group` writes after every exact factor-reuse group (a group may contain one
	monomial), giving finer restart points and more chunk files;
  - `:degree` writes once after each completed total degree, giving fewer, larger chunks.

A checkpoint is accepted only when its fingerprint matches the model, spectral data,
resonance set, multiindex set, conjugate permutation, and numerical solver policy. Models
containing application-defined callable kernels must implement
[`checkpoint_fingerprint_data`](@ref) before they can be checkpointed.

# Example

```julia
checkpoint = CheckpointOptions("checkpoints/beam-order-9";
	problem_id = "clamped-beam-v1",
	granularity = :degree)

options = ParametrisationOptions(checkpoint = checkpoint)
W, R = parametrise(model, spectral, 9; options)
```
"""
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
        isempty(problem_id) &&
            throw(ArgumentError("checkpoint problem_id must not be empty"))
        granularity in (:degree, :factor_group) || throw(ArgumentError(
            "checkpoint granularity must be :degree or :factor_group"))
        new(String(path), String(problem_id), resume, granularity)
    end
end

"""
	ParametrisationOptions(;
		backend = :auto,
		grouping = :auto,
		residual_check = :off,
		residual_tolerance = nothing,
		max_refinement_steps = 3,
		validate_mset = true,
		checkpoint = nothing,
		show_progress = true,
		verbose = true,
		setup_io = stderr)

Example of operational controls passed to [`parametrise`](@ref) through its `options` keyword:

```julia
options = ParametrisationOptions(
	backend = :klu,
	residual_check = :backward_error,
	residual_tolerance = 1e-10,
	show_progress = false)

W, R = parametrise(model, spectral, expansion_order; options)
```

The fields below are constructor keywords of `ParametrisationOptions`; they are **not**
direct keywords of `parametrise`. Mathematical choices (`resonance` and
`conjugate_permutation`) deliberately remain explicit keywords of `parametrise`.

# Solver options

- `backend::Symbol = :auto` — linear solver for the bordered cohomological systems:
  - `:auto` uses dense LU for dense full-order matrices; for sparse matrices it uses
	Pardiso when a Pardiso extension is active and otherwise KLU;
  - `:klu` requires sparse full-order matrices and forces KLU;
  - `:umfpack` requires sparse full-order matrices and forces Julia's built-in
	SuiteSparse UMFPACK solver;
  - `:pardiso` requires sparse full-order matrices and an active Pardiso extension,
	otherwise construction of the solver fails.
- `grouping::Symbol = :auto` — exact reuse of a factorisation by monomials whose
  cohomological matrices are structurally identical:
  - `:auto` groups only when repeated or zero eigenvalues make reuse possible and grouping
	actually reduces the number of factorisations;
  - `:on` always builds and processes the exact structural groups;
  - `:off` solves directly in graded-lexicographic order without grouping.
  Grouping is exact; it never clusters approximately equal eigenvalues. Each group uses
  its first monomial's superharmonic consistently for every member.
- `residual_check::Symbol = :off` — `:backward_error` verifies every bordered solve;
  `:off` skips this additional check.
- `residual_tolerance::Union{Nothing, Real} = nothing` — positive backward-error limit.
  It is used only with `residual_check = :backward_error`. `nothing` selects the
  scalar-type-aware default `sqrt(eps(real(T))) / 100` (approximately `1.49e-10` for
  `Float64`).
- `max_refinement_steps::Integer = 3` — maximum number of iterative-refinement
  corrections after a failed backward-error check on the dense, KLU and UMFPACK paths.
  Must be non-negative. A solve still throws if it remains outside the tolerance.

# Validation and restart options

- `validate_mset::Bool = true` — validate the multiindex set before solving: dimensions,
  allowed degrees, required master unit vectors, downward closure, and (when active)
  conjugate closure. Set this to `false` only when the same set has already been validated.
- `checkpoint::Union{Nothing, CheckpointOptions} = nothing` — durable checkpoint and
  resume policy. `nothing` disables checkpoint I/O; see [`CheckpointOptions`](@ref).

# Output options

- `show_progress::Bool = true` — show the in-place monomial progress indicator on
  `stderr` when it is an interactive terminal. It remains silent in redirected output and
  CI logs.
- `verbose::Bool = true` — print the model/setup summary before building the resonance
  set. With the default `setup_io = stderr`, the summary is printed only when `stderr` is
  interactive; an explicitly supplied output stream is always honoured.
- `setup_io::IO = stderr` — destination for the setup summary controlled by `verbose`.
  This does not redirect the progress indicator, which always uses `stderr`.

# Common configurations

Quiet run:

```julia
options = ParametrisationOptions(show_progress = false, verbose = false)
```

Verified sparse solve with automatic exact factor reuse:

```julia
options = ParametrisationOptions(
	backend = :auto,
	grouping = :auto,
	residual_check = :backward_error,
	residual_tolerance = 1e-10,
	max_refinement_steps = 3)
```

All symbolic options are validated when `ParametrisationOptions` is constructed, so a
misspelling such as `backend = :suitesparse` or `grouping = :approximate` fails
immediately.
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
    backend in (:auto, :klu, :umfpack, :pardiso) || throw(ArgumentError(
        "backend must be :auto, :klu, :umfpack, or :pardiso"))
    grouping in (:auto, :off, :on) || throw(ArgumentError(
        "grouping must be :auto, :off, or :on"))
    residual_check in (:backward_error, :off) || throw(ArgumentError(
        "residual_check must be :backward_error or :off"))
    isnothing(residual_tolerance) || residual_tolerance > 0 ||
        throw(ArgumentError(
            "residual_tolerance must be positive or nothing"))
    max_refinement_steps >= 0 || throw(ArgumentError(
        "max_refinement_steps must be non-negative"))
    return ParametrisationOptions(
        backend, grouping, residual_check, residual_tolerance,
        Int(max_refinement_steps), validate_mset, checkpoint,
        show_progress, verbose, setup_io)
end

"""
	_replace_options(options; overrides...) -> ParametrisationOptions

Return a validated copy of `options`, replacing only the supplied keyword fields. This is
the internal equivalent of an immutable record update and preserves the concrete `setup_io`
type when no replacement is requested.
"""
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
