# Ordered external-direction solves and conjugate-block validation.

"""
	_solve_external_directions!(W, R, partial_ctx_for, model, ml_cache,
		symmetry, N_EXT, ROM, unit_offset)

Solve external forcing directions in increasing variable order. Partial contexts contain
only directions already solved, as required by upper-triangular external dynamics.
Conjugate secondaries are reconstructed immediately, and every external linear monomial is
marked skipped before the nonlinear schedule is built.
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
