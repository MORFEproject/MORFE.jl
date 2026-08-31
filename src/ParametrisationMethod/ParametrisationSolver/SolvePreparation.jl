# Backend, operator, symmetry, resource, and context preparation.

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
