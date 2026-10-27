# Sparse backend markers and reusable constant-pattern solver state.

"""
	AbstractSparseBackend

Internal dispatch root for sparse bordered solvers. [`KLUBackend`](@ref) and
[`UMFPACKBackend`](@ref) are always available; [`PardisoBackend`](@ref) is constructed
only when the extension is active.
"""
abstract type AbstractSparseBackend end

"""
	KLUBackend{VERIFY}

Marker for the built-in KLU path. `VERIFY` selects backward-error verification at compile
time so the default unchecked loop does not pay for residual bookkeeping.
"""
struct KLUBackend{VERIFY} <: AbstractSparseBackend end

"""
	UMFPACKBackend{VERIFY}

Marker for the opt-in SuiteSparse UMFPACK path. `VERIFY` selects backward-error
verification at compile time so unchecked solves retain the same specialised path as KLU.
"""
struct UMFPACKBackend{VERIFY} <: AbstractSparseBackend end

"""
	PardisoBackend{P, VERIFY}

Sparse backend state for an extension-provided Pardiso `solver`. `VERIFY` selects
backward-error verification at compile time.
"""
struct PardisoBackend{P, VERIFY} <: AbstractSparseBackend
    solver::P
end

"""
	_backend_name(backend) -> Symbol

Return the stable diagnostic name of a sparse backend (`:klu`, `:umfpack` or `:pardiso`).
"""
_backend_name(::KLUBackend) = :klu
_backend_name(::UMFPACKBackend) = :umfpack
_backend_name(::PardisoBackend) = :pardiso

"""
	SparseLinearSolverState{T, B, RT}

Sparse-path resources for the constant-size bordered cohomological system (see the
[`CohomologicalEquations`](@ref) module docstring for the system itself).

`bordered` is the `(FOM+ROM) × (FOM+ROM)` matrix actually handed to the factoriser.
**Its `colptr`/`rowval` are fixed for the entire solve — only `nzval` is rewritten
per monomial** — which is what keeps the symbolic factorisation cached in `fact`
valid throughout, and is the reason the border is masked rather than compacted.

`L_template` is the separate square `FOM × FOM` workspace on which
[`build_sparse_L_and_rhs!`](@ref) runs its fused Horner pass (it needs the transient
intermediates `L[j](s)` to accumulate the lower-order RHS); the resulting `L(s)` is
then block-copied into the `(1,1)` block of `bordered`, column by column.

Backend selection follows the caller's parametrisation options: `:klu` forces KLU,
`:umfpack` forces SuiteSparse UMFPACK, `:pardiso` requires the extension, and `:auto`
chooses between UMFPACK and KLU with [`_auto_suitesparse_backend`](@ref). Pardiso runs
only on request: it has not been benchmarked on MORFE's bordered systems. KLU and UMFPACK
reuse cached symbolic analysis while redoing numeric factorisation whenever the bordered
matrix changes. KLU uses `klu_factor!`, not the exported `klu!`: the latter freezes the
pivot sequence.

# Fields

- `bordered::SparseMatrixCSC{T}` — the `(FOM+ROM)²` matrix handed to the factoriser.
  Its `colptr`/`rowval` never change; only `nzval` is rewritten per monomial.
- `L_template::SparseMatrixCSC{T}` — `FOM × FOM` workspace carrying the union
  sparsity pattern of all `linear_terms`, on which `L(s)` is built.
- `L_mappings::Vector{Vector{Int}}` — for each `linear_terms[k]`, the position in
  `L_template.nzval` of each of its stored entries, so accumulating `s^k B_k` is an
  indexed scatter with no pattern search.
- `border_row_base::Vector{Int}` — length `FOM`; `bordered[FOM+r, c]` lives at
  `nzval[border_row_base[c] + r - 1]`.  The border rows are contiguous within each
  column, which is what makes writing them a strided copy.
- `solve_scratch::Vector{T}` — length `FOM+ROM` RHS copy for Pardiso, whose solve
  needs distinct input and output vectors. Empty on the KLU and UMFPACK paths, where
  `ldiv!` is genuinely in-place.
- `pardiso_matrix::Any` — the matrix handed to Pardiso's analysis phase; `nothing`
  until `_pardiso_prepare!` has run.
- `fact::Any` — cached KLU or UMFPACK factorisation; `nothing` until the first successful
  one.
- `backend::B` — selected [`KLUBackend`](@ref), [`UMFPACKBackend`](@ref), or
  [`PardisoBackend`](@ref).
- `residual_tolerance::Union{Nothing, RT}` — active backward-error threshold.
- `max_refinement_steps::Int` — maximum KLU or UMFPACK refinement corrections.
- `residual_work::Vector{T}` — persistent KLU or UMFPACK residual/norm workspace.
- `refinement_work::Vector{T}` — lazily allocated correction workspace.
- `max_relative_residual::RT` — largest accepted backward error observed so far.
- `refinement_count::Int` — total KLU or UMFPACK refinement corrections performed.

`mutable` is load-bearing, not incidental: the Pardiso branch attaches a finaliser to
release C-side memory, and Julia refuses to finalise an immutable object.
"""
mutable struct SparseLinearSolverState{T, B <: AbstractSparseBackend, RT <: Real}
    bordered::SparseMatrixCSC{T}         # (FOM+ROM)²; constant pattern
    L_template::SparseMatrixCSC{T}       # FOM²; workspace for L(s)
    L_mappings::Vector{Vector{Int}}      # linear_terms[k].nzval → L_template.nzval
    border_row_base::Vector{Int}         # length FOM
    solve_scratch::Vector{T}             # Pardiso only; empty on SuiteSparse paths
    pardiso_matrix::Any                  # nothing until _pardiso_prepare! has run
    fact::Any                            # nothing until the first factorisation
    backend::B
    residual_tolerance::Union{Nothing, RT}
    max_refinement_steps::Int
    residual_work::Vector{T}
    refinement_work::Vector{T}
    max_relative_residual::RT
    refinement_count::Int
end

# UMFPACK Info index (zero-based) and value, from umfpack.h: the strategy the symbolic
# analysis chose, and the code of the symmetric strategy.
const _UMFPACK_STRATEGY_USED = 18
const _UMFPACK_STRATEGY_SYMMETRIC = 3

"""
	_auto_suitesparse_backend(matrix) -> Symbol

Choose `:umfpack` or `:klu` for `backend = :auto`.

UMFPACK's symbolic analysis, which needs no numeric factorisation and takes milliseconds,
chooses a strategy on its own. It takes the symmetric strategy when the pattern is symmetric
and the diagonal nonzero, as in stiffness-dominated structural models. It takes the
unsymmetric strategy otherwise, for instance when incompressible flow leaves a zero pressure
block. Under the symmetric strategy UMFPACK's multifrontal BLAS-3 kernels factorise the
bordered matrices of volumetric finite element models several times faster than KLU, and
the gap grows with the mesh. Under the unsymmetric strategy UMFPACK's column ordering fills
far more than KLU's, and KLU is several times faster.

The function therefore returns `:umfpack` exactly when UMFPACK chooses its symmetric
strategy for `matrix`, and `:klu` otherwise, including when the analysis fails. The choice
depends only on the matrix, never on timings, so solves stay bit-for-bit reproducible.
UMFPACK itself still decides its strategy for every factorisation: nothing is imposed.
"""
function _auto_suitesparse_backend(matrix::SparseMatrixCSC)
    symmetric = try
        factorisation = UMFPACK.UmfpackLU(matrix)
        UMFPACK.umfpack_symbolic!(factorisation, nothing)
        factorisation.info[_UMFPACK_STRATEGY_USED + 1] == _UMFPACK_STRATEGY_SYMMETRIC
    catch error
        _is_unrecoverable_failure(error) && rethrow()
        false
    end
    return symmetric ? :umfpack : :klu
end

"""
	_representative_bordered(bordered, L_template, L_mappings, linear_terms) -> matrix

A copy of the bordered template filled with `L(0) = B₀` in its `(1,1)` block, zero borders,
and the identity in its corner, which is the matrix of a monomial with no resonant master
mode at zero superharmonic. [`_auto_suitesparse_backend`](@ref) reads its structure before
any monomial is assembled.
"""
function _representative_bordered(bordered::SparseMatrixCSC, L_template::SparseMatrixCSC,
        L_mappings::Vector{Vector{Int}}, linear_terms)
    L = copy(L_template)
    fill!(L.nzval, zero(eltype(L)))
    stiffness = first(linear_terms)
    for (entry, position) in enumerate(L_mappings[1])
        L.nzval[position] += nonzeros(stiffness)[entry]
    end
    matrix = copy(bordered)
    fill!(matrix.nzval, zero(eltype(matrix)))
    scatter_L_into_bordered!(matrix, L)
    FOM = size(L, 1)
    for q in 1:(size(matrix, 1) - FOM)
        matrix.nzval[matrix.colptr[FOM + q] + FOM + q - 1] = one(eltype(matrix))
    end
    return matrix
end

"""
    SparseLinearSolverState{T}(L_template, L_mappings, FOM, ROM;
        options = nothing, linear_terms = nothing) -> SparseLinearSolverState

Initialise the sparse solver state and constant-pattern bordered template. Backend
selection, residual verification, and refinement storage follow `options`. For
`backend = :auto`, `linear_terms` (the operators `B₀, …, B_ORD`) let
[`_auto_suitesparse_backend`](@ref) choose between UMFPACK and KLU; without them, `:auto`
falls back to KLU. Pardiso runs only with `backend = :pardiso`.
"""
function SparseLinearSolverState{T}(
        L_template::SparseMatrixCSC{T},
        L_mappings::Vector{Vector{Int}},
        FOM::Int,
        ROM::Int;
        options = nothing,
        linear_terms = nothing
) where {T}
    requested = options === nothing ? :auto : options.backend
    ps = requested == :pardiso ? _try_build_pardiso_solver() : nothing
    requested == :pardiso && ps === nothing &&
        error(
            "ParametrisationOptions requested Pardiso, but no Pardiso backend is active")
    bordered, border_row_base = precompute_sparse_bordered_template(L_template, ROM)
    nsys = FOM + ROM
    RT = typeof(real(zero(T)))
    residual_tolerance = _configured_residual_tolerance(T, options)
    verify = !isnothing(residual_tolerance)
    suitesparse = requested == :auto && linear_terms !== nothing ?
                  _auto_suitesparse_backend(_representative_bordered(
        bordered, L_template, L_mappings, linear_terms)) : requested
    backend = suitesparse == :umfpack ? UMFPACKBackend{verify}() :
              ps === nothing ? KLUBackend{verify}() :
              PardisoBackend{typeof(ps), verify}(ps)
    state = SparseLinearSolverState{T, typeof(backend), RT}(
        bordered, L_template, L_mappings, border_row_base,
        backend isa PardisoBackend ? Vector{T}(undef, nsys) : T[],
        nothing, nothing,
        backend, residual_tolerance,
        options === nothing ? 3 : options.max_refinement_steps,
        isnothing(residual_tolerance) || backend isa PardisoBackend ? T[] :
        Vector{T}(undef, nsys),
        T[], zero(RT), 0
    )
    # Pardiso's factorisation lives in C-side memory the GC does not track, so it has
    # to be released explicitly or every solve leaks one factorisation. This is also
    # why the struct is `mutable`: Julia will not attach a finaliser to an immutable
    # object.
    ps === nothing || finalizer(_release_pardiso!, state)
    return state
end

"""
	_release_pardiso!(state) -> nothing

Finaliser: hand the Pardiso factorisation back. No-op on the KLU and UMFPACK paths, and
never allowed to throw — a finaliser that raises would be reported out of context.
"""
function _release_pardiso!(state::SparseLinearSolverState{T, <:KLUBackend}) where {T}
    return nothing
end
function _release_pardiso!(state::SparseLinearSolverState{T, <:UMFPACKBackend}) where {T}
    return nothing
end
function _release_pardiso!(state::SparseLinearSolverState{T, <:PardisoBackend}) where {T}
    try
        _pardiso_release!(state.backend.solver, state.pardiso_matrix)
    catch
        # Nothing useful to do during finalisation.
    end
    return nothing
end
