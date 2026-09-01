# Typed KLU and UMFPACK factor access, refactorisation, and cache recovery.

"""
	_cached_klu_factor(state, matrix) -> KLUFactorization

Return the cached KLU factorisation with value and index types derived from the active
sparse matrix. Call only after the first successful factorisation has populated `fact`.
"""
function _cached_klu_factor(
        ss::SparseLinearSolverState{T}, ::SparseMatrixCSC{T, Ti}
) where {T, Ti}
    return ss.fact::KLUFactorization{T, Ti}
end

"""
	_cached_umfpack_factor(state, matrix) -> UmfpackLU

Return the cached UMFPACK factorisation with value and index types derived from the active
sparse matrix. Call only after a successful factorisation has populated `fact`.
"""
function _cached_umfpack_factor(
        ss::SparseLinearSolverState{T}, ::SparseMatrixCSC{T, Ti}
) where {T, Ti}
    return ss.fact::UmfpackLU{T, Ti}
end

"""Create and cache a checked fresh KLU factorisation of `A`."""
function _fresh_klu_factor!(ss::SparseLinearSolverState, A::SparseMatrixCSC)
    # Stored structural zeros in the constant border can make KLU's unchecked status
    # look successful even when the numeric factor is singular. The checked path is
    # required for both fresh and cached factors; it throws before an invalid factor can
    # reach `ldiv!`.
    F = klu(A; check = true, allowsingular = false)
    ss.fact = F
    return F
end

"""
	_discard_klu_factor!(state, factorisation) -> nothing

Remove a failed KLU factorisation from `state` and release its native symbolic and
numeric storage before another analysis begins. Merely assigning `nothing` to `state.fact`
would leave that storage alive until garbage collection, which can retain damaged KLU
state across the immediate recovery attempt on Windows.

`Base.finalize` is Julia's required external spelling; it runs the finaliser already
owned by `KLUFactorization` and prevents it from running twice later.
"""
function _discard_klu_factor!(ss::SparseLinearSolverState, factorisation::KLUFactorization)
    ss.fact = nothing
    Base.finalize(factorisation)
    return nothing
end

"""
	_refactorise_klu!(ss, A) -> KLU factorisation of `A`

Factorise the bordered matrix for the current monomial, reusing the symbolic analysis
cached in `ss.fact`.

The first call analyses and factorises; every later call redoes only the **numeric**
factorisation. Splitting them this way is what makes the constant-size formulation
pay off: the sparsity pattern is identical for every monomial, so the ordering and
symbolic phase — the expensive part — are computed once for the whole solve.

## Why `klu_factor!` and not `klu!`

The numeric phase must **re-pivot**. The bordered matrix changes value on every
monomial and its `(1,1)` block is near-singular at every resonance, where stability
depends on pivoting being free to exchange rows across the border. `klu_factor!`
re-pivots while reusing the cached symbolic analysis. KLU's exported `klu!` maps to
`klu_refactor`, which replays the pivot sequence chosen at the first monomial and so
loses accuracy exactly where it is needed.

## Preconditions and aliasing

`A`'s sparsity pattern must be identical on every call — [`SparseLinearSolverState`](@ref)
guarantees this by construction, since only `nzval` is ever written.

`klu` takes `A.nzval` by reference rather than copying it, so the factorisation and
the template share one value array: per-monomial assembly writes straight into what
KLU reads, with no copy-in step. `colptr`/`rowval` stay KLU's own (0-based) copies,
and are invariant in any case.

A factorisation is cached only if it succeeded. Both fresh and cached numeric
factorisations use KLU's checked path because stored zeros in the constant border can make
an unchecked status appear successful even when the numeric factor is singular. A failed
cached factor has its native storage released immediately and is retried once from a fresh
symbolic analysis.
"""
function _refactorise_klu!(ss::SparseLinearSolverState, A::SparseMatrixCSC)
    if ss.fact === nothing
        # The checked path is required here as well as during cached refactorisation:
        # stored structural zeros in the border can otherwise leave KLU reporting
        # success for a singular numeric factor.
        return _fresh_klu_factor!(ss, A)
    end
    F = _cached_klu_factor(ss, A)
    try
        # KLU may leave `common.status` looking successful after a failed unchecked
        # refactorisation of a bordered pattern. Requesting the checked, non-singular
        # path makes that failure observable before `ldiv!` can return NaNs.
        klu_factor!(F; check = true, allowsingular = false)
    catch error
        _is_unrecoverable_failure(error) && rethrow()
        _discard_klu_factor!(ss, F)
        return _fresh_klu_factor!(ss, A)
    end
    issuccess(F) && return F
    _discard_klu_factor!(ss, F)
    return _fresh_klu_factor!(ss, A)
end

"""
	_refactorise_umfpack!(state, matrix) -> UMFPACK factorisation

Factorise the current bordered matrix through SuiteSparse UMFPACK. Successful symbolic
analysis is reused by `lu!`. A failed cached numeric factorisation is discarded and retried
once from a fresh analysis; only a successful factorisation is retained in `state.fact`.
"""
function _refactorise_umfpack!(ss::SparseLinearSolverState, A::SparseMatrixCSC)
    if ss.fact === nothing
        F = lu(A; check = false)
        issuccess(F) && (ss.fact = F)
        return F
    end
    F = _cached_umfpack_factor(ss, A)
    try
        lu!(F, A; check = false, reuse_symbolic = true)
    catch error
        _is_unrecoverable_failure(error) && rethrow()
        ss.fact = nothing
        fresh = lu(A; check = false)
        issuccess(fresh) && (ss.fact = fresh)
        return fresh
    end
    issuccess(F) && return F
    ss.fact = nothing
    fresh = lu(A; check = false)
    issuccess(fresh) && (ss.fact = fresh)
    return fresh
end

"""Backward-compatible internal alias for [`_refactorise_klu!`](@ref)."""
_refactorise!(ss::SparseLinearSolverState, A::SparseMatrixCSC) = _refactorise_klu!(ss, A)
