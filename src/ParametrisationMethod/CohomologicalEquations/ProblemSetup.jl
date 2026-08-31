# End-to-end problem preparation: initialise W/R, build operators, and invoke execution.

# =============================================================================
# Private driver helpers
# =============================================================================

"""
	_embed_external_dynamics!(R, external_polynomial, mset)

Copy coefficients from the `N_EXT`-variable external polynomial into the last `N_EXT`
rows of `R`, embedding them in the full `NVAR = ROM + N_EXT` multiindex set. Throw when a
non-zero external coefficient has no destination, since silently dropping it would change
the external dynamics.
"""
function _embed_external_dynamics!(
        R::ReducedDynamics{ROM, NVAR, T},
        external_polynomial::DensePolynomial{T, N_EXT, 2},
        mset::MultiindexSet{NVAR}
) where {ROM, NVAR, T, N_EXT}
    N_EXT > 0 || return nothing
    index_map = build_exponent_index_map(mset)
    external_coefficients = external_polynomial.coefficients
    for (monomial_index, external_exponent) in enumerate(external_polynomial.multiindex_set.exponents)
        full_exponent = SVector{NVAR, Int}(ntuple(
            index -> index <= ROM ? 0 : external_exponent[index - ROM], Val(NVAR)))
        full_index = get(index_map, full_exponent, nothing)
        if full_index === nothing
            nonzero_coefficients = [(external_index,
                                        external_coefficients[external_index, monomial_index])
                                    for external_index in 1:N_EXT
                                    if !iszero(external_coefficients[external_index, monomial_index])]
            isempty(nonzero_coefficients) && continue
            throw(ArgumentError("""
                External dynamics carry the monomial r^$(Tuple(external_exponent)) with \
                non-zero coefficients $(nonzero_coefficients) (as (row, value)), but the \
                multiindex set has no entry for the corresponding full exponent \
                $(Tuple(full_exponent)).
                That term would be dropped without trace. Enlarge `mset` to contain it — \
                a re-based external system can generate additional cross monomials.
                """))
        end
        for external_index in 1:N_EXT
            coefficient = T(external_coefficients[external_index, monomial_index])
            iszero(coefficient) ||
                (R.poly.coefficients[ROM + external_index, full_index] = coefficient)
        end
    end
    return nothing
end

"""
	_linear_monomial_indices(mset) -> Vector{Int}

Return the positions of every unit-vector monomial in `mset`. These are the linear
coefficients initialised from eigenvectors or external directions rather than by the main
nonlinear solve.
"""
function _linear_monomial_indices(mset::MultiindexSet{NVAR}) where {NVAR}
    indices = Int[]
    search_limit = min(NVAR + 1, length(mset))
    for variable in 1:NVAR
        unit_exponent = SVector{NVAR, Int}(ntuple(
            index -> index == variable ? 1 : 0, Val(NVAR)))
        monomial_index = findfirst(
            ==(unit_exponent), view(mset.exponents, 1:search_limit))
        monomial_index === nothing || push!(indices, monomial_index)
    end
    return indices
end

"""
    _initialise_waveform!(W, R, master_right_modes, master_eigenvalues,
                          master_right_mode_derivatives, unit_offset, model)

Initialise the linear-monomial coefficients of `W` and `R` from spectral data:

- `W[:, 1, eᵣ] = master_right_modes[:, r]` and, for `ORD > 1`,
  `W[:, k, eᵣ] = master_right_mode_derivatives[:, k-1, r]`.
- `R[r, eᵣ] = master_eigenvalues[r]`.

Also embeds the external-system linear dynamics into the external rows of `R` via
`_embed_external_dynamics!` when `model.external_system !== nothing`.
"""
function _initialise_waveform!(
        W::Parametrisation, R::ReducedDynamics,
        master_right_modes, master_eigenvalues,
        master_right_mode_derivatives, unit_offset::Int, model
)
    ROM = length(master_eigenvalues)
    ORD_W = size(W.poly.coefficients, 2)
    for r in 1:ROM
        idx_er = r + unit_offset
        W.poly.coefficients[:, 1, idx_er] .= view(master_right_modes, :, r)
        if master_right_mode_derivatives !== nothing
            for k in 2:ORD_W
                W.poly.coefficients[:, k, idx_er] .= view(
                    master_right_mode_derivatives, :, k -
                                                      1, r)
            end
        end
        R.poly.coefficients[r, idx_er] = master_eigenvalues[r]
    end
    if model.external_system !== nothing
        _embed_external_dynamics!(R, model.external_system.first_order_dynamics, multiindex_set(W))
    end
    return nothing
end

"""
	_solve_external_directions!(W, R, partial_ctx_for, model, ml_cache, sym, N_EXT, ROM, unit_offset)

Solve the `N_EXT` external forcing directions in increasing variable order.

`partial_ctx_for(e)` returns the *partial* context for external direction `e`: one in which
the external generalised eigenvectors `Φ_ext` are known only for the directions already
solved, and zero from `e` onwards.  Solving in order is what makes that possible — column
`e` of the external column-polynomial recurrences reads only `Φ_ext,j` with `Λ_ext[j, e] ≠ 0`,
hence only `j ≤ e` for the upper-triangular `Λ_ext` that the solver requires.

When conjugate-symmetry is active, secondary external monomials are filled from their
primaries via `fill_conjugate_monomial!` in the same pass rather than afterwards, so that
`Φ_ext,k` is populated for every `k < e` regardless of how it was obtained.  This is safe
because the conjugate map always marks the larger index as the secondary.

All external linear monomials are marked in `sym.skip_bits` after this call so
the main `solve_cohomological_equations!` loop does not overwrite them.
"""
function _solve_external_directions!(
        W, R, partial_ctx_for, model, ml_cache,
        sym::ConjugateSymmetryData{NoConjugatePermutation}, N_EXT::Int, ROM::Int, unit_offset::Int
)
    for e in 1:N_EXT
        idx = ROM + e + unit_offset
        solve_single_monomial!(W, R, idx, partial_ctx_for(e), model, ml_cache)
        @inbounds sym.skip_bits[idx] = true
    end
    return nothing
end

function _solve_external_directions!(
        W, R, partial_ctx_for, model, ml_cache,
        sym::ConjugateSymmetryData{<:SVector}, N_EXT::Int, ROM::Int, unit_offset::Int
)
    N_EXT == 0 && return nothing
    for e in 1:N_EXT
        idx = ROM + e + unit_offset
        if @inbounds sym.skip_bits[idx]
            # Secondary: its source is a pre-marked primary with a smaller index, so it is
            # already available — `_build_conjugate_symmetry` only ever marks the larger
            # index of a pair, and `monomial_map` is symmetric.
            fill_conjugate_monomial!(W, R, idx, sym.monomial_map[idx], sym)
        else
            solve_single_monomial!(W, R, idx, partial_ctx_for(e), model, ml_cache)
        end
    end
    for e in 1:N_EXT
        @inbounds sym.skip_bits[ROM + e + unit_offset] = true
    end
    return nothing
end

"""
	_check_external_conjugate_block(conjugate_permutation, sys, ROM, NVAR)

Reject a `conjugate_permutation` whose external block disagrees with the external system's
own conjugate involution.

Only checked when the system was **re-based** (`external_basis(sys) !== nothing`), because
that is the only situation in which a hand-written permutation can be stale: the caller's
external indices then refer to coordinates `r′` that the constructor chose, not the ones
they wrote down.  For every system left in its own coordinates this is a no-op, so no
existing model can trip it.

Getting this wrong is silent — `fill_conjugate_monomial!` would fill external monomials from
the wrong partner — hence an error rather than a warning.  Use
[`full_conjugate_permutation`](@ref) to build the vector instead of writing it by hand.
"""
function _check_external_conjugate_block(
        conjugate_permutation, sys, ROM::Int, NVAR::Int)
    conjugate_permutation === nothing && return nothing
    external_basis(sys) === nothing && return nothing
    NVAR > ROM || return nothing

    supplied = collect(conjugate_permutation[(ROM + 1):NVAR]) .- ROM
    σ = external_conjugate_permutation(sys)

    if σ === nothing
        supplied == collect(1:(NVAR - ROM)) || throw(ArgumentError("""
             The external system was re-based onto a basis whose columns are not conjugate \
             pairs, so its external variables have no conjugate structure, but the supplied \
             `conjugate_permutation` pairs them as $(supplied).
             Either drop `conjugate_permutation`, or use an external system whose linear \
             matrix is real so the conjugate-preserving eigenvector route is taken.
             """))
        return nothing
    end

    supplied == σ || throw(ArgumentError("""
       The external block of `conjugate_permutation` is $(ROM .+ supplied), but the \
       re-based external system pairs its variables as $(ROM .+ σ).
       A change of external coordinates was applied (the supplied linear matrix was not \
       upper triangular), so an external pairing written for the original coordinates no \
       longer holds.  Build the permutation with \
       `full_conjugate_permutation(master_block, model.external_system)`.
       """))
    return nothing
end

"""
	_make_sparse_solver(MT, linear_terms, FOM, ROM,
		options = ParametrisationOptions()) -> Union{SparseLinearSolverState, Nothing}

Dispatch helper: returns a `SparseLinearSolverState` when `MT <: SparseMatrixCSC`
(sparse path), or `nothing` for all other matrix types (dense path). An explicit sparse
backend request is rejected for dense matrices; sparse backend selection follows `options`.
"""
_make_sparse_solver(::Type{<:AbstractMatrix}, _, ::Int, ::Int) = nothing
function _make_sparse_solver(
        ::Type{<:SparseMatrixCSC}, linear_terms, FOM::Int, ROM::Int,
        options::ParametrisationOptions = ParametrisationOptions()
)
    L_template, L_mappings = precompute_sparse_L_template(linear_terms)
    T = eltype(L_template)
    return SparseLinearSolverState{T}(L_template, L_mappings, FOM, ROM; options)
end
function _make_sparse_solver(::Type{<:AbstractMatrix}, _, ::Int, ::Int,
        options::ParametrisationOptions)
    options.backend in (:auto,) || throw(ArgumentError(
        "backend=$(options.backend) requires sparse full-order matrices"))
    return nothing
end

"""
	_build_context(linear_terms, generalised_right_eigenmodes, lambda_diag,
				   inv_ops, orth_ops, resonance_set, linear_skip_set,
				   lower_order, buffers, sparse_solver) -> CohomologicalContext

Construct a `CohomologicalContext` from pre-assembled operator data and shared
resources.  All type parameters are inferred from the arguments.
"""
function _build_context(
        linear_terms::NTuple{ORDP1, MT},
        generalised_right_eigenmodes::Matrix{T},
        lambda_diag::Vector{T},
        inv_ops::InvarianceOperators{T},
        orth_ops::OrthogonalityOperators{T},
        resonance_set::ResonanceSet,
        linear_skip_set::Set{Int},
        lower_order::LowerOrderResources{NVAR, T},
        buffers::CohomologicalBuffers{T},
        sparse_solver
) where {ORDP1, MT, T, NVAR}
    FOM = size(generalised_right_eigenmodes, 1)
    ORD = ORDP1 - 1
    LT = eltype(MT)
    return CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT}(
        linear_terms, generalised_right_eigenmodes, lambda_diag,
        inv_ops, orth_ops,
        resonance_set, linear_skip_set,
        lower_order, buffers, sparse_solver
    )
end

"""
	_open_problem_checkpoint(checkpoint, model, spectral, resonance_set, mset,
		conjugate_permutation, options, dimensions, T)

Return `nothing` when checkpointing is disabled. Otherwise calculate the problem
fingerprint and open or validate its checkpoint before solver and coefficient storage are
allocated. `dimensions` is `(ORD, FOM, ROM, N_EXT, NVAR)`.
"""
_open_problem_checkpoint(::Nothing, args...) = nothing
function _open_problem_checkpoint(
        checkpoint::CheckpointOptions,
        model, spectral, resonance_set, mset, conjugate_permutation,
        options::ParametrisationOptions,
        dimensions::NTuple{5, Int}, ::Type{T}
) where {T}
    ORD, FOM, ROM, N_EXT, NVAR = dimensions
    fingerprint = _problem_fingerprint(
        model, spectral, resonance_set, mset, conjugate_permutation, options)
    metadata = Dict{String, Any}(
        "ORD" => ORD, "FOM" => FOM, "ROM" => ROM, "N_EXT" => N_EXT,
        "NVAR" => NVAR, "monomial_count" => length(mset),
        "scalar_type" => string(T), "backend_request" => string(options.backend),
        "grouping" => string(options.grouping)
    )
    return _open_checkpoint(checkpoint, fingerprint, metadata)
end

"""
	_prepare_solution_storage(initial_solution, mset, model, master_right_modes,
		master_eigenvalues, master_right_mode_derivatives, dimensions, unit_offset, T)

Return `(W, R, supplied)`. Supplied objects are returned unchanged and by identity. When
no storage is supplied, allocate zeroed coefficient arrays and initialise their known
linear coefficients from the spectral and external-system data.
"""
function _prepare_solution_storage(
        initial_solution, mset, model,
        master_right_modes, master_eigenvalues, master_right_mode_derivatives,
        dimensions::NTuple{5, Int}, unit_offset::Int, ::Type{T}
) where {T}
    ORD, FOM, ROM, N_EXT, _ = dimensions
    if initial_solution !== nothing
        W, R = initial_solution
        return W, R, true
    end

    @assert ORD == 1 || master_right_mode_derivatives !== nothing """
        master right-mode derivatives must be provided for ORD > 1 systems.
        Supply a FOM × (ORD-1) × ROM array whose slice [:, k, r] = W^(k+1)[e_r].
        """
    W, R = create_parametrisation_method_objects(mset, ORD, FOM, ROM, N_EXT, T)
    _initialise_waveform!(W, R, master_right_modes, master_eigenvalues,
        master_right_mode_derivatives, unit_offset, model)
    return W, R, false
end

"""
	_restore_solution_checkpoint!(session, W, R, supplied)

Restore committed coefficient slices into the selected storage and return
`(completed_indices, completed_degree)`. When `supplied` is true, every committed slice
must already agree with the caller's storage. With no checkpoint, return `(nothing, 0)`.
"""
_restore_solution_checkpoint!(::Nothing, W, R, supplied::Bool) = (nothing, 0)
function _restore_solution_checkpoint!(
        session::CheckpointSession, W, R, supplied::Bool)
    completed_indices = _restore_checkpoint!(session, W, R;
        verify_existing = supplied)
    completed_degrees = _completed_degrees(session)
    completed_degree = isempty(completed_degrees) ? 0 : maximum(completed_degrees)
    return completed_indices, completed_degree
end

"""
	_announce_conjugate_symmetry(permutation)

Emit the conjugate-symmetry assumptions once when a concrete conjugate permutation is
active. The no-symmetry method is a compile-time no-op.
"""
_announce_conjugate_symmetry(::NoConjugatePermutation) = nothing
function _announce_conjugate_symmetry(::SVector)
    @info """
      conjugate_permutation is active — the following assumptions must hold:
        1. Real-valued FOM: all matrices in model.linear_terms and all nonlinear/force \
          terms must have real-valued entries (eltype <: Real or purely-real complex).
        2. Each mode either comes in a complex conjugate pair with another mode, or is \
          self-paired  meaning it has a real eigenvalue and a real-valued mode shape.
        3. Eigenvalue conjugacy is necessary but NOT sufficient for paired modes; \
           the right eigenvectors must satisfy \
           master_right_modes[:, perm[r]] = conj(master_right_modes[:, r]).
        4. If external modes are present (N_EXT > 0): the same pairing rules apply \
          to the external eigenvalues, encoded in the NVAR-length permutation.
      Passing an incorrect permutation silently corrupts the parametrisation and reduced-dynamics.
      """ maxlog=1
    return nothing
end

"""
	_prepare_shared_resources(model, W, mset, conjugate_permutation, completed_indices,
		dimensions, MT, options)

Build symmetry, lower-order coupling resources, the nonlinear cache, and numerical
buffers. Restored checkpoint indices are applied to the symmetry skip mask before cache
construction. Return `(linear_skip_set, lower_order, symmetry, cache, buffers)`.
"""
function _prepare_shared_resources(
        model, W, mset::MultiindexSet{NVAR}, conjugate_permutation,
        completed_indices, dimensions::NTuple{5, Int}, ::Type{MT},
        options::ParametrisationOptions
) where {NVAR, MT}
    ORD, FOM, ROM, _, _ = dimensions
    T = eltype(W.poly.coefficients)
    linear_skip_set = Set(_linear_monomial_indices(mset)[1:ROM])
    lower_order = LowerOrderResources{NVAR, T}(mset, ORD, FOM)
    permutation = conjugate_permutation === nothing ?
                  NoConjugatePermutation() :
                  SVector{NVAR, Int}(conjugate_permutation)
    _announce_conjugate_symmetry(permutation)

    symmetry = if permutation isa NoConjugatePermutation
        _build_conjugate_symmetry(permutation, linear_skip_set, length(mset))
    else
        _build_conjugate_symmetry(
            permutation, linear_skip_set, mset, lower_order.multiindex_dict)
    end

    if completed_indices !== nothing && !isempty(completed_indices)
        maximum(completed_indices) <= length(mset) || throw(ArgumentError(
            "checkpoint contains a monomial index outside the active set"))
        for index in completed_indices
            symmetry.skip_bits[index] = true
        end
    end

    cache = build_multilinear_terms_cache(model, W, symmetry.skip_bits)
    buffers = CohomologicalBuffers(T, MT, FOM, ROM, options)
    return linear_skip_set, lower_order, symmetry, cache, buffers
end

"""
	_prepare_master_operators(linear_terms, W, R, master modes and blocks,
		unit_offset, ROM)

Precompute the external-independent orthogonality and invariance operators. Return the
orthogonality row coefficients, the master order-block view, the invariance columns, and
the master derivative steps.
"""
function _prepare_master_operators(
        linear_terms, W, R, master_right_modes, master_left_modes,
        master_left_mode_blocks, unit_offset::Int, ROM::Int)
    orthogonality_J_coeffs = precompute_orthogonality_operator_coefficients(
        linear_terms, master_left_modes, master_left_mode_blocks)
    right_master_blocks = view(W.poly.coefficients, :, :,
        ((unit_offset + 1):(unit_offset + ROM)))
    lambda_master = view(
        R.poly.coefficients, 1:ROM, (unit_offset + 1):(unit_offset + ROM))
    invariance_C_coeffs, master_derivative_steps = precompute_master_column_polynomials(
        linear_terms, master_right_modes, lambda_master)
    return orthogonality_J_coeffs, right_master_blocks,
    invariance_C_coeffs, master_derivative_steps
end

"""
	_prepare_external_directions!(W, R, supplied, completed_indices, ...)

Solve fresh external linear directions in causal variable order, or accept directions from
supplied/restored storage. Every external linear monomial is marked skipped before this
function returns.
"""
function _prepare_external_directions!(
        W, R, supplied::Bool, completed_indices,
        model, cache, symmetry, dimensions::NTuple{5, Int}, unit_offset::Int,
        lambda_matrix, lambda_diag, linear_terms, master_right_modes,
        invariance_C_coeffs, master_derivative_steps,
        orthogonality_J_coeffs, right_master_blocks,
        resonance_set, linear_skip_set, lower_order, buffers, sparse_solver)
    _, FOM, ROM, N_EXT, NVAR = dimensions
    T = eltype(W.poly.coefficients)
    must_solve = !supplied &&
                 (completed_indices === nothing || isempty(completed_indices))
    if must_solve
        known_directions = zeros(T, FOM, N_EXT)
        build_partial_context = function (external_directions, blank::Int)
            partial_E_coeffs = precompute_external_column_polynomials(
                linear_terms, external_directions, lambda_matrix,
                master_derivative_steps)
            partial_orth_C_coeffs,
            partial_orth_E_coeffs = precompute_orthogonality_column_polynomials(
                orthogonality_J_coeffs, right_master_blocks,
                external_directions, lambda_matrix)
            if 1 <= blank <= N_EXT
                fill!(partial_E_coeffs[blank], zero(T))
                for r in 1:ROM
                    partial_orth_E_coeffs[r][:, blank] .= zero(T)
                end
            end
            return _build_context(
                linear_terms, hcat(master_right_modes, external_directions), lambda_diag,
                InvarianceOperators{T}(invariance_C_coeffs, partial_E_coeffs),
                OrthogonalityOperators{T}(orthogonality_J_coeffs,
                    partial_orth_C_coeffs, partial_orth_E_coeffs),
                resonance_set, linear_skip_set,
                lower_order, buffers, sparse_solver)
        end

        coupled_external = !isdiag(view(
            lambda_matrix, (ROM + 1):NVAR, (ROM + 1):NVAR))
        shared_partial_context = coupled_external ? nothing :
                                 build_partial_context(known_directions, 0)
        partial_context_for = function (external_index)
            coupled_external || return shared_partial_context
            for known_index in 1:(external_index - 1)
                known_directions[:, known_index] .= view(
                    W.poly.coefficients, :, 1,
                    ROM + known_index + unit_offset)
            end
            return build_partial_context(known_directions, external_index)
        end

        _solve_external_directions!(
            W, R, partial_context_for, model, cache, symmetry,
            N_EXT, ROM, unit_offset)
    else
        for external_index in 1:N_EXT
            @inbounds symmetry.skip_bits[
                ROM + external_index + unit_offset] = true
        end
    end
    return nothing
end

"""
	_build_complete_context(W, dimensions, unit_offset, ...)

Read the final external directions from `W`, build the complete external operators, and
return the context used by the nonlinear solve.
"""
function _build_complete_context(
        W, dimensions::NTuple{5, Int}, unit_offset::Int,
        linear_terms, master_right_modes, lambda_matrix, lambda_diag,
        invariance_C_coeffs, master_derivative_steps,
        orthogonality_J_coeffs, right_master_blocks,
        resonance_set, linear_skip_set, lower_order, buffers, sparse_solver)
    _, FOM, ROM, N_EXT, _ = dimensions
    T = eltype(W.poly.coefficients)
    external_directions = zeros(T, FOM, N_EXT)
    for external_index in 1:N_EXT
        external_directions[:, external_index] .= W.poly.coefficients[
            :, 1, ROM + external_index + unit_offset]
    end
    generalised_right_eigenmodes = hcat(master_right_modes, external_directions)
    invariance_E_coeffs = precompute_external_column_polynomials(
        linear_terms, external_directions, lambda_matrix, master_derivative_steps)
    orthogonality_C_coeffs,
    orthogonality_E_coeffs = precompute_orthogonality_column_polynomials(
        orthogonality_J_coeffs, right_master_blocks,
        external_directions, lambda_matrix)
    return _build_context(
        linear_terms, generalised_right_eigenmodes, lambda_diag,
        InvarianceOperators{T}(invariance_C_coeffs, invariance_E_coeffs),
        OrthogonalityOperators{T}(orthogonality_J_coeffs,
            orthogonality_C_coeffs, orthogonality_E_coeffs),
        resonance_set, linear_skip_set,
        lower_order, buffers, sparse_solver)
end

# =============================================================================
# High-level driver
# =============================================================================

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
    show_progress = options.show_progress
    @assert NVAR == ROM + N_EXT "Multiindex set has $NVAR variables but ROM + N_EXT = $(ROM + N_EXT)"
    # Bind to concrete locals here, at the setup boundary, and nowhere else. The bundle's
    # block fields are `Union{Nothing, Array}` so that ORD == 1 is representable, and every
    # access to them is a type-unstable branch — harmless once, unacceptable in the loop.
    master_eigs = master_eigenvalues(spectral)
    master_right_modes = right_modes(spectral)::Matrix{ComplexF64}
    master_left_modes = left_modes(spectral)::Matrix{ComplexF64}
    master_right_mode_derivatives = right_mode_derivatives(spectral)
    master_left_mode_blocks = left_mode_blocks(spectral)
    conj_perm = _spectral_conjugate_permutation(
        conjugate_permutation, spectral, model.external_system)

    # Every path into the solve lands here, so this is where the mset contract is
    # enforced. `parametrise` checks first and passes validate_mset = false.
    options.validate_mset && validate_multiindex_set(mset, NVAR, ROM;
        conjugate_permutation = conj_perm)
    _check_external_conjugate_block(conj_perm, model.external_system, ROM, NVAR)
    FOM = size(master_right_modes, 1)
    @assert size(master_right_modes, 2) == ROM "master right modes must have $ROM columns"
    T = ComplexF64
    dimensions = (ORD, FOM, ROM, N_EXT, NVAR)
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

    if checkpoint_session !== nothing && completed_degree < 1
        degree_one = [idx for idx in eachindex(mset.exponents) if sum(mset[idx]) == 1]
        _write_chunk!(checkpoint_session, W, R, 1, degree_one, sparse_solver;
            degree_complete = true)
        completed_degree = 1
    end

    ctx = _build_complete_context(
        W, dimensions, unit_offset,
        linear_terms, master_right_modes, Λ, lambda_diag,
        invariance_C_coeffs, master_derivative_steps,
        orthogonality_J_coeffs, right_master_blocks,
        resonance_set, linear_skip_set, lower_order, buffers, sparse_solver)

    # ── 7. Main solve ─────────────────────────────────────────────────────────
    if benchmark_dir !== nothing
        solve_cohomological_equations_benchmarked!(W, R, ctx, sym, model, ml_cache;
            benchmark_dir, show_progress)
    else
        observer = checkpoint_session === nothing ? _NO_SOLVE_OBSERVER :
                   _CheckpointSolveObserver(
            checkpoint_session, W, R, mset, sparse_solver)
        _solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache;
            show_progress,
            grouping = options.grouping,
            observer)
    end

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
