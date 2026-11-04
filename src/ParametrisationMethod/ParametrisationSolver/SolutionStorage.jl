# Creation, reuse, initialisation, and checkpoint restoration of `W` and `R`.

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
    _initialise_waveform!(W, R, right_blocks, master_dynamics, model)

Initialise the linear part of `W` and `R`, which is their columns `1:NVAR`
(`check_unit_columns`), from the spectral data:

- `W[:, :, 1:ROM] = right_blocks`, the master right order-blocks;
- `R[1:ROM, 1:ROM] = master_dynamics`, the master block `Λ̃` (`Diagonal` or
  `UpperTriangular`, whose diagonal is the master eigenvalues).

Also embeds the external-system linear dynamics into the external rows of `R` via
`_embed_external_dynamics!` when `model.external_system !== nothing`.
"""
function _initialise_waveform!(
        W::Parametrisation, R::ReducedDynamics, right_blocks, master_dynamics, model)
    ROM = size(master_dynamics, 1)
    W.poly.coefficients[:, :, 1:ROM] .= right_blocks
    R.poly.coefficients[1:ROM, 1:ROM] .= master_dynamics
    if model.external_system !== nothing
        _embed_external_dynamics!(R, model.external_system.first_order_dynamics, multiindex_set(W))
    end
    return nothing
end

"""
	_prepare_solution_storage(initial_solution, mset, model, spectral, dimensions, T)

Return `(W, R, supplied)`. Supplied objects are returned unchanged and by identity. When
no storage is supplied, allocate zeroed coefficient arrays and initialise their known
linear coefficients from `spectral` (the master right blocks and the master block of the
reduced linear dynamics) and the external system.
"""
function _prepare_solution_storage(
        initial_solution, mset, model, spectral, dimensions::NTuple{5, Int}, ::Type{T}
) where {T}
    ORD, FOM, ROM, N_EXT, _ = dimensions
    if initial_solution !== nothing
        W, R = initial_solution
        return W, R, true
    end
    W, R = create_parametrisation_method_objects(mset, ORD, FOM, ROM, N_EXT, T)
    _initialise_waveform!(W, R, master_bundle(spectral).right_blocks,
        master_linear_dynamics(spectral), model)
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
