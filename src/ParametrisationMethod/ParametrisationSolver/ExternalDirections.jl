# Ordered external-direction solves and conjugate-block validation.

"""
	_solve_external_directions!(W, R, partial_ctx_for, model, ml_cache,
		symmetry, N_EXT, ROM)

Solve external forcing directions in increasing variable order. Partial contexts contain
only directions already solved, as required by upper-triangular external dynamics.
Conjugate secondaries are reconstructed immediately, and every external linear monomial is
marked skipped before the nonlinear schedule is built. A direction whose bordered matrix
equals that of the direction solved before it reuses its numeric factorisation
([`_solve_external_direction!`](@ref)).
"""
function _solve_external_directions!(
        W, R, partial_ctx_for, model, ml_cache,
        sym::ConjugateSymmetryData{NoConjugatePermutation}, N_EXT::Int, ROM::Int
)
    previous = nothing
    for e in 1:N_EXT
        idx = ROM + e
        previous = _solve_external_direction!(
            W, R, idx, partial_ctx_for(e), model, ml_cache, sym, previous, ROM)
        @inbounds sym.skip_bits[idx] = true
    end
    return nothing
end

function _solve_external_directions!(
        W, R, partial_ctx_for, model, ml_cache,
        sym::ConjugateSymmetryData{<:SVector}, N_EXT::Int, ROM::Int
)
    N_EXT == 0 && return nothing
    previous = nothing
    for e in 1:N_EXT
        idx = ROM + e
        if @inbounds sym.skip_bits[idx]
            # Secondary: its source is a pre-marked primary with a smaller index, so it is
            # already available — `_build_conjugate_symmetry` only ever marks the larger
            # index of a pair, and `monomial_map` is symmetric.
            fill_conjugate_monomial!(W, R, idx, sym.monomial_map[idx], sym)
        else
            previous = _solve_external_direction!(
                W, R, idx, partial_ctx_for(e), model, ml_cache, sym, previous, ROM)
        end
    end
    for e in 1:N_EXT
        @inbounds sym.skip_bits[ROM + e] = true
    end
    return nothing
end

"""
	_solve_external_direction!(W, R, index, ctx, model, ml_cache, sym, previous, ROM)
		-> (ctx, index)

Solve the external direction at position `index` of the multiindex set and return what the
next direction compares itself with. `previous` is the `(ctx, index)` of the direction
solved last, or `nothing`. When [`_same_bordered_matrix`](@ref) holds, the numeric
factorisation of `previous`, still cached, solves this direction as well: parameters all
have the eigenvalue zero, so they cost one factorisation instead of one each. The
factorisation of an identical matrix being identical, `W` and `R` do not change.
"""
function _solve_external_direction!(
        W, R, index::Int, ctx, model, ml_cache, sym, previous, ROM::Int)
    reuse = previous !== nothing &&
            _same_bordered_matrix(previous..., ctx, index, multiindex_set(W), ROM)
    solve_single_monomial!(W, R, index, ctx, sym, model, ml_cache, reuse)
    return (ctx, index)
end

"""
	_same_bordered_matrix(previous_ctx, previous_index, ctx, index, mset, ROM) -> Bool

Whether monomials `previous_index` and `index` have the same bordered matrix. The matrix
depends only on the context, the superharmonic and the resonance mask
([`_assemble_bordered_system!`](@ref)), so equality of all three, the superharmonics
compared bit for bit, implies equality of the matrices. A coupled external block builds a
new context for every direction, which therefore never reuses a factorisation.
"""
function _same_bordered_matrix(previous_ctx, previous_index::Int, ctx, index::Int,
        mset, ROM::Int)
    ctx === previous_ctx || return false
    isequal(_superharmonic(mset[index], ctx.lambda_diag),
        _superharmonic(mset[previous_index], ctx.lambda_diag)) || return false
    return all(
        r -> is_resonant(ctx.resonance_set, index, r) ==
             is_resonant(ctx.resonance_set, previous_index, r),
        1:ROM)
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
