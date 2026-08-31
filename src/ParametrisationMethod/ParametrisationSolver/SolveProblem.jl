# Short top-level orchestration of the parametrisation solve phases.

# =============================================================================
# High-level driver
# =============================================================================

"""
	_prepare_problem_inputs(model, mset, spectral, conjugate_permutation, options)

Resolve concrete spectral arrays, validate dimensions and conjugate structure, and return
the data required by storage and operator preparation. This is the single setup boundary
for `SpectralData` fields whose higher-order blocks may be `nothing`.
"""
function _prepare_problem_inputs(
        model::NthOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
        mset::MultiindexSet{NVAR}, spectral::SpectralData{ORD, ROM},
        conjugate_permutation, options::ParametrisationOptions
) where {ORD, ORDP1, N_NL, N_EXT, LT, MT, NVAR, ROM}
    @assert NVAR == ROM + N_EXT "Multiindex set has $NVAR variables but ROM + N_EXT = $(ROM + N_EXT)"
    master_eigs = master_eigenvalues(spectral)
    master_right_modes = right_modes(spectral)::Matrix{ComplexF64}
    master_left_modes = left_modes(spectral)::Matrix{ComplexF64}
    master_right_mode_derivatives = right_mode_derivatives(spectral)
    master_left_mode_blocks = left_mode_blocks(spectral)
    conjugate_structure = _spectral_conjugate_permutation(
        conjugate_permutation, spectral, model.external_system)
    options.validate_mset && validate_multiindex_set(mset, NVAR, ROM;
        conjugate_permutation = conjugate_structure)
    _check_external_conjugate_block(
        conjugate_structure, model.external_system, ROM, NVAR)
    FOM = size(master_right_modes, 1)
    @assert size(master_right_modes, 2) == ROM "master right modes must have $ROM columns"
    dimensions = (ORD, FOM, ROM, N_EXT, NVAR)
    return master_eigs, master_right_modes, master_left_modes,
    master_right_mode_derivatives, master_left_mode_blocks,
    conjugate_structure, dimensions
end

"""Commit degree-one coefficients once external directions are final."""
function _commit_initial_degree!(
        session, W, R, mset, sparse_solver, completed_degree::Int)
    (session === nothing || completed_degree >= 1) && return completed_degree
    degree_one = [index for index in eachindex(mset.exponents) if sum(mset[index]) == 1]
    _write_chunk!(session, W, R, 1, degree_one, sparse_solver; degree_complete = true)
    return 1
end

"""Execute benchmark or ordinary scheduling with the appropriate typed observer."""
function _execute_problem_schedule!(W, R, ctx, symmetry, model, cache, mset,
        sparse_solver, checkpoint_session, benchmark_dir, options)
    if benchmark_dir !== nothing
        solve_cohomological_equations_benchmarked!(
            W, R, ctx, symmetry, model, cache;
            benchmark_dir, show_progress = options.show_progress)
    else
        observer = checkpoint_session === nothing ? _NO_SOLVE_OBSERVER :
                   _CheckpointSolveObserver(
            checkpoint_session, W, R, mset, sparse_solver)
        _solve_cohomological_equations!(W, R, ctx, symmetry, model, cache;
            show_progress = options.show_progress,
            grouping = options.grouping,
            observer)
    end
    return nothing
end

"""
	solve_cohomological_problem(
		model, mset, spectral::SpectralData, resonance_set;
		initial_solution = nothing,
		conjugate_permutation = :from_spectral,
		benchmark_dir = nothing,
		options = ParametrisationOptions()
	) -> (W, R)

High-level driver that assembles a [`CohomologicalContext`](@ref) from spectral data and
solves the full set of cohomological equations.

The spectral input is **one** object. It replaces five separately maintained inputs: master
eigenvalues, master right modes, master left modes, right-mode derivative blocks, and
left-mode order blocks. Every former call site had to slice and keep those arrays mutually
consistent — including the mirrored right/left block convention, where a swap is
type-correct and compiles silently. [`SpectralData`](@ref)'s explicitly named accessors own
that convention now, and it is checked numerically by `check_biorthogonality`.

The external dynamics enter through `model.external_system.first_order_dynamics`, which
`_embed_external_dynamics!` copies into the external rows of `R`; the superharmonics `s`
are then contracted against `diag(Λ)` read back from `R`, so the external part of `s` is
the diagonal of the external linear matrix.  That matrix must be upper triangular — the
`ExternalSystem` constructors guarantee it, re-basing the external coordinates when the
supplied matrix is not, as explained in the `ExternalSystems` module docstring.  When that
happens the reduced external coordinates returned here are the re-based `r′`, related to the
physical `r` by `r = Q r′` with `Q = external_basis(model.external_system)`.  The
linear-operator tuple is read from `model.linear_terms`.

## Steps

1. Select the coefficient storage: allocate and initialise `W` and `R`, or use the exact
   objects supplied through `initial_solution`.
2. Restore checkpoint-committed coefficient slices into that storage. A supplied solution
   must agree with every committed slice; non-zero coefficients alone never count as
   completed work.
3. Build symmetry and shared resources after restored indices have entered the skip mask.
4. Solve the linear cohomological equations for each fresh external forcing direction via
   a partial context in which the not-yet-solved external columns of the generalised
   right-eigenmode matrix are set to zero. Supplied or restored external directions are
   accepted rather than recomputed.
5. Build the full `generalised_right_eigenmodes` matrix by concatenating the master right
   eigenmodes with the solved external right directions. This matrix is stored as
   `CohomologicalContext.generalised_eigenmodes`; left eigenmodes enter only through the
   orthogonality operators.
6. Assemble the full context and solve every monomial not marked linear, conjugate-secondary,
   or checkpoint-complete.

## Arguments

- `model :: NthOrderModel` — full-order model; `model.linear_terms` provides `(B₀,…,B_ORD)`.
- `mset :: MultiindexSet{NVAR}` — multiindex set over all `NVAR = ROM + N_EXT` variables.
- `spectral :: SpectralData{ORD, ROM}` — master eigenvalues, the right physical modes and
  their derivative blocks, the left physical modes and their orthogonality blocks, and the
  conjugate involution. Build it with `SpectralData(model, spectrum; master = …)`.
  The orthogonality row operators are read directly off the left blocks — no eigenvalue
  folding.
- `resonance_set :: ResonanceSet`.
- `initial_solution` — optionally supply an already-initialised `(W, R)` tuple. The solver
  mutates and returns those same objects, trusts their master and external linear data, and
  recomputes scheduled nonlinear coefficients. This is storage reuse, not implicit
  checkpoint completion or an iterative warm start.
- `conjugate_permutation` — `:from_spectral` (default) takes the bundle's master-block
  involution and extends it over the external variables using the model's external system.
  Pass an `NVAR`-length vector to override it — `perm[i] = j` means mode `j` is the complex
  conjugate of mode `i` — or `nothing` to disable conjugate symmetry for this solve.
  A supplied permutation is the caller's assertion about the *eigenvectors*: two
  eigenvalues forming a conjugate pair is necessary but not sufficient (the eigenspace must
  be one-dimensional, or the eigenvectors chosen conjugately). `SpectralData`'s `:detect`
  verifies exactly that before returning one.
- `options` — execution, validation, residual-verification, and checkpoint policy in a
  [`ParametrisationOptions`](@ref). Mathematical choices remain separate arguments.
- `benchmark_dir` — `nothing` for the normal grouped/checkpoint-capable solve, or a directory
  in which [`solve_cohomological_equations_benchmarked!`](@ref) writes timing CSV files.
  Benchmark mode uses the shared direct execution plan and therefore does not perform
  factor grouping. It cannot be combined with checkpointing because benchmark timings do
  not form resumable checkpoint commits.

## Returns

`(W, R)` — the solved [`Parametrisation`](@ref) and [`ReducedDynamics`](@ref).
"""
function solve_cohomological_problem(
        model::NthOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
        mset::MultiindexSet{NVAR},
        spectral::SpectralData{ORD, ROM},
        resonance_set::ResonanceSet;
        initial_solution::Union{Nothing, Tuple{Parametrisation, ReducedDynamics}} = nothing,
        conjugate_permutation = :from_spectral,
        benchmark_dir::Union{Nothing, AbstractString} = nothing,
        options::ParametrisationOptions = ParametrisationOptions()
) where {ORD, ORDP1, N_NL, N_EXT, LT, MT, NVAR, ROM}
    checkpoint = options.checkpoint
    benchmark_dir !== nothing && checkpoint !== nothing &&
        throw(ArgumentError(
            "benchmark_dir cannot be combined with checkpointing; benchmark execution " *
            "does not create resumable checkpoint commits"))
    master_eigs, master_right_modes, master_left_modes,
    master_right_mode_derivatives, master_left_mode_blocks,
    conj_perm, dimensions = _prepare_problem_inputs(
        model, mset, spectral, conjugate_permutation, options)
    _, FOM, _, _, _ = dimensions
    T = ComplexF64
    checkpoint_session = _open_problem_checkpoint(
        checkpoint, model, spectral, resonance_set, mset, conj_perm,
        options, dimensions, T)

    linear_terms = model.linear_terms
    # Resolve and validate the concrete backend before allocating W/R. An explicit
    # incompatible dense/sparse/backend request therefore fails at the setup boundary.
    sparse_solver = _make_sparse_solver(MT, linear_terms, FOM, ROM, options)

    zero_vec = SVector{NVAR, Int}(ntuple(_ -> 0, Val(NVAR)))
    has_zero = length(mset) >= 1 && mset.exponents[1] == zero_vec
    unit_offset = has_zero ? 1 : 0

    W, R, supplied = _prepare_solution_storage(
        initial_solution, mset, model, master_right_modes, master_eigs,
        master_right_mode_derivatives, dimensions, unit_offset, T)
    completed_indices, completed_degree = _restore_solution_checkpoint!(
        checkpoint_session, W, R, supplied)

    Λ = view(R.poly.coefficients, 1:NVAR, (unit_offset + 1):(unit_offset + NVAR))
    lambda_diag = [R.poly.coefficients[i, i + unit_offset] for i in 1:NVAR]

    linear_skip_set, lower_order, sym, ml_cache, buffers = _prepare_shared_resources(
        model, W, mset, conj_perm, completed_indices,
        dimensions, MT, options)
    orthogonality_J_coeffs, right_master_blocks,
    invariance_C_coeffs, master_derivative_steps = _prepare_master_operators(
        linear_terms, W, R, master_right_modes, master_left_modes,
        master_left_mode_blocks, unit_offset, ROM)

    _prepare_external_directions!(
        W, R, supplied, completed_indices,
        model, ml_cache, sym, dimensions, unit_offset,
        Λ, lambda_diag, linear_terms, master_right_modes,
        invariance_C_coeffs, master_derivative_steps,
        orthogonality_J_coeffs, right_master_blocks,
        resonance_set, linear_skip_set, lower_order, buffers, sparse_solver)

    completed_degree = _commit_initial_degree!(
        checkpoint_session, W, R, mset, sparse_solver, completed_degree)

    ctx = _build_complete_context(
        W, dimensions, unit_offset,
        linear_terms, master_right_modes, Λ, lambda_diag,
        invariance_C_coeffs, master_derivative_steps,
        orthogonality_J_coeffs, right_master_blocks,
        resonance_set, linear_skip_set, lower_order, buffers, sparse_solver)

    _execute_problem_schedule!(W, R, ctx, sym, model, ml_cache, mset,
        sparse_solver, checkpoint_session, benchmark_dir, options)

    return W, R
end

"""
	_spectral_conjugate_permutation(request, spectral, external_system)

Resolve `:from_spectral` by taking the master permutation stored in `spectral` and, when
present, appending the conjugate block derived from `external_system`. Explicit vectors and
`nothing` pass through unchanged.
"""
function _spectral_conjugate_permutation(request, spectral::SpectralData, sys)
    request === :from_spectral || return request
    # `master_conjugate_permutation` is a field read — the restriction was settled when the
    # SpectralData was built.
    master_perm = master_conjugate_permutation(spectral)
    master_perm === nothing && return nothing
    # With no external system, NVAR == ROM and the master block already *is* the full
    # permutation. Returning the stored vector avoids rebuilding an identical one on every
    # autonomous solve — `full_conjugate_permutation` would collect, broadcast and vcat.
    sys === nothing && return master_perm
    return full_conjugate_permutation(master_perm, sys)
end
