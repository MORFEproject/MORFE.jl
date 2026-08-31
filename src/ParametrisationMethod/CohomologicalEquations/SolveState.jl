# Immutable problem data and mutable workspaces reused by every monomial solve.

# =============================================================================
# Precomputed operator coefficient bundles
# =============================================================================

"""
	InvarianceOperators{T}

Precomputed column-polynomial coefficients for the invariance equation.

Both fields hold coefficients of polynomials in the superharmonic `s`, evaluated by
[`evaluate_column!`](@ref) once per monomial: the master columns become the `C(s)`
border of the bordered system, the external ones contribute to its right-hand side.
Precomputing them per order rather than per monomial is what keeps the inner solve
loop allocation-free.

# Fields

- `column_coeffs::Vector{Matrix{T}}` — one `FOM × ORD` matrix per master mode `r`,
  the coefficients of the border *column* `C_r(s)`.  Distinct in both shape and
  role from [`OrthogonalityOperators`](@ref) `corner_coeffs`, which fills the
  `ROM × ROM` corner.
- `E_coeffs::Vector{Matrix{T}}` — one `FOM × ORD` matrix per external variable or
  direction `e`. External amplitudes are known, so these never reach the matrix.
"""
struct InvarianceOperators{T}
    column_coeffs::Vector{Matrix{T}}   # length ROM,   each FOM × ORD
    E_coeffs::Vector{Matrix{T}}        # length N_EXT, each FOM × ORD
end

"""
	OrthogonalityOperators{T}

Precomputed row and column-polynomial coefficients for the orthogonality conditions.

Together the three fields supply the bottom `ROM` rows of the bordered system: one
row per master mode, assembled by [`assemble_orthogonality_matrix_and_rhs!`](@ref).
They are read off the left eigenvector order-blocks once per order, so evaluating a
row at a given `s` costs one Horner pass and no allocation.

# Fields

- `J_coeffs::Vector{Matrix{T}}` — one `ORD × FOM` matrix per master mode `r`, the
  coefficients of the row operator `Ĵ_r(s)` acting on `W[α]`.
- `corner_coeffs::Vector{Matrix{T}}` — one `(ORD-1) × ROM` matrix per master mode,
  evaluating to the `ROM × ROM` *corner* block `Ĉ(s)` that couples the orthogonality
  rows to the unknown reduced-dynamics coefficients.  See
  [`InvarianceOperators`](@ref) `column_coeffs` for the differently-shaped border
  columns.
- `E_coeffs::Vector{Matrix{T}}` — one `(ORD-1) × N_EXT` matrix per master mode,
  contracting the known external amplitudes into the scalar right-hand side.
"""
struct OrthogonalityOperators{T}
    J_coeffs::Vector{Matrix{T}}        # length ROM, each ORD × FOM
    corner_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × ROM
    E_coeffs::Vector{Matrix{T}}        # length ROM, each (ORD-1) × N_EXT
end

# =============================================================================
# Lower-order coupling resources
# =============================================================================

"""
	LowerOrderResources{NVAR, T}

Data needed to compute lower-order coupling vectors `ξ[j]` on every monomial.

Bundled into one struct so the per-monomial buffers and the multiindex lookup are
allocated once for the whole solve rather than per call.  Everything here is a pure
function of the multiindex set, so none of it changes as the solve progresses.

# Fields

- `multiindex_dict::Dict{SVector{NVAR, Int}, Int}` — maps an exponent vector to its
  position in the multiindex set, turning the coupling lookup into a hash rather
  than a scan.
- `buffer::Vector{Vector{T}}` — `ORD` vectors of length `FOM` holding the coupling
  terms `ξ[j]`.  Reused across monomials and zeroed by the caller before each use.
- `candidate_indices::Vector{Vector{Int}}` — for each of the `L` monomial positions,
  the multiindices that can contribute a coupling to it.  Degree-1 monomials get an
  empty list, since a coupling needs two factors of degree ≥ 1.
- `unit_vectors::Vector{SVector{NVAR, Int}}` — the `NVAR` unit exponent vectors
  `eᵣ`, materialised once because the coupling recurrence subtracts them constantly.
"""
struct LowerOrderResources{NVAR, T}
    multiindex_dict::Dict{SVector{NVAR, Int}, Int}
    buffer::Vector{Vector{T}}               # length ORD, each FOM
    candidate_indices::Vector{Vector{Int}}  # length L
    unit_vectors::Vector{SVector{NVAR, Int}}
end

"""
	LowerOrderResources{NVAR, T}(mset, ORD, FOM) -> LowerOrderResources

Build the lower-order coupling resources for a multiindex set of `NVAR` variables,
ODE order `ORD`, and full-order dimension `FOM`.

`candidate_indices` is precomputed here, once per monomial position: it lists the
multiindices that can contribute a lower-order coupling to that monomial, which is a
pure function of the multiindex set and so need not be recomputed during the solve.
"""
function LowerOrderResources{NVAR, T}(
        mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int
) where {NVAR, T}
    L = length(mset)
    candidate_indices = Vector{Vector{Int}}(undef, L)
    for i in 1:L
        tdeg = sum(mset[i])
        candidate_indices[i] = tdeg < 2 ? Int[] :
                               indices_in_box_with_bounded_degree(mset, mset[i], 2, tdeg)
    end
    return LowerOrderResources{NVAR, T}(
        build_exponent_index_map(mset),
        [zeros(T, FOM) for _ in 1:ORD],
        candidate_indices,
        [SVector{NVAR, Int}(ntuple(k -> k == j ? 1 : 0, Val(NVAR))) for j in 1:NVAR]
    )
end

# =============================================================================
# Pre-allocated compute buffers
# =============================================================================

"""
	CohomologicalBuffers{T, RT}

Pre-allocated scratch buffers for the cohomological system assembly and solve.

Both paths solve the same constant-size `(FOM+ROM) × (FOM+ROM)` bordered system, but
they materialise it differently, so only one of the two matrix buffers is allocated:

- **dense path** uses `system_matrix`, the bordered matrix itself, LU'd in place;
- **sparse path** uses `orthogonality_rows` as the staging area for the `ROM`
  orthogonality rows, which are then scattered into the strided border positions of
  the sparse template's `nzval` (the sparse bordered matrix lives in
  [`SparseLinearSolverState`](@ref)).

The unused buffer is a `0×0` placeholder.

# Fields

- `system_matrix::Matrix{T}` — the `(FOM+ROM) × (FOM+ROM)` bordered matrix on the
  dense path, factorised in place by `lu!`.  `0×0` on the sparse path.
- `orthogonality_rows::Matrix{T}` — `ROM × (FOM+ROM)` staging area on the sparse
  path, holding the evaluated `Ĵ(s)` rows and corner before they are scattered into
  the template.  `0×0` on the dense path.
- `rhs::Vector{T}` — length `FOM+ROM`.  Holds the right-hand side on entry and, in
  the same memory, the solution after the solve; the unpacking step reads `W[α]`
  from its first `FOM` entries and the resonant `R[α]` from the rest.
- `external_rhs::Vector{T}` — length `FOM` scratch for `evaluate_external_rhs!`.
- `ml_result::Vector{T}` — length `FOM`, receiving the nonlinear (multilinear-term)
  contribution from `compute_multilinear_terms!`.
- `dense_solution::Vector{T}` — accepted solution retained while the dense matrix and
  right-hand side are reassembled for backward-error checks; empty when verification is
  disabled or on the sparse path.
- `dense_refinement::Vector{T}` — lazily allocated dense correction vector used only after
  a failed backward-error check.
- `residual_tolerance::Union{Nothing, RT}` — active scalar-type-aware backward-error
  threshold, or `nothing` when verification is disabled.
- `max_refinement_steps::Int` — maximum dense iterative-refinement corrections.
"""
struct CohomologicalBuffers{T, RT <: Real}
    system_matrix::Matrix{T}       # (FOM+ROM)²; dense path only
    orthogonality_rows::Matrix{T}  # ROM×(FOM+ROM); sparse path only
    rhs::Vector{T}                 # length FOM+ROM; rhs in, solution out
    external_rhs::Vector{T}        # length FOM
    ml_result::Vector{T}           # length FOM
    dense_solution::Vector{T}      # dense backward-error path only
    dense_refinement::Vector{T}    # allocated lazily after a failed first check
    residual_tolerance::Union{Nothing, RT}
    max_refinement_steps::Int
end

"""
	_configured_residual_tolerance(T, options) -> tolerance or nothing

Resolve the backward-error threshold in the real scalar type associated with `T`.
Returns `nothing` when `options.residual_check == :off`; otherwise uses the explicit
tolerance or the scalar-type-aware default.
"""
function _configured_residual_tolerance(::Type{T}, options::ParametrisationOptions) where {T}
    RT = typeof(real(zero(T)))
    options.residual_check == :off && return nothing
    return isnothing(options.residual_tolerance) ?
           sqrt(eps(RT)) / RT(100) : convert(RT, options.residual_tolerance)
end

"""
    CohomologicalBuffers(T, MT, FOM, ROM, options = ParametrisationOptions()) -> CohomologicalBuffers

Allocate all buffers for a system of full-order dimension `FOM` and `ROM` master
modes.  Dispatches on the FOM matrix type `MT`: `MT <: SparseMatrixCSC` selects the
sparse layout, everything else the dense one. `options` controls backward-error workspace
and the refinement limit.
"""
function CohomologicalBuffers(::Type{T}, ::Type{MT}, FOM::Int, ROM::Int,
        options::ParametrisationOptions = ParametrisationOptions()) where {T, MT}
    nsys = FOM + ROM
    RT = typeof(real(zero(T)))
    return CohomologicalBuffers{T, RT}(
        Matrix{T}(undef, nsys, nsys),
        Matrix{T}(undef, 0, 0),
        Vector{T}(undef, nsys),
        zeros(T, FOM),
        zeros(T, FOM),
        _configured_residual_tolerance(T, options) === nothing ? T[] :
        Vector{T}(undef, nsys),
        T[],
        _configured_residual_tolerance(T, options),
        options.max_refinement_steps
    )
end
function CohomologicalBuffers(::Type{T}, ::Type{MT}, FOM::Int, ROM::Int,
        options::ParametrisationOptions = ParametrisationOptions()) where {
        T, MT <: SparseMatrixCSC}
    RT = typeof(real(zero(T)))
    return CohomologicalBuffers{T, RT}(
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, ROM, FOM + ROM),
        Vector{T}(undef, FOM + ROM),
        zeros(T, FOM),
        zeros(T, FOM),
        T[], T[], nothing, options.max_refinement_steps
    )
end

# =============================================================================
# Sparse-path solver state
# =============================================================================

"""
	AbstractSparseBackend

Internal dispatch root for sparse bordered solvers. [`KLUBackend`](@ref) is always
available; [`PardisoBackend`](@ref) is constructed only when the extension is active.
"""
abstract type AbstractSparseBackend end

"""
	KLUBackend{VERIFY}

Marker for the built-in KLU path. `VERIFY` selects backward-error verification at compile
time so the default unchecked loop does not pay for residual bookkeeping.
"""
struct KLUBackend{VERIFY} <: AbstractSparseBackend end

"""
	PardisoBackend{P, VERIFY}

Sparse backend state for an extension-provided Pardiso `solver`. `VERIFY` selects
backward-error verification at compile time.
"""
struct PardisoBackend{P, VERIFY} <: AbstractSparseBackend
    solver::P
end

_backend_name(::KLUBackend) = :klu
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

Backend selection follows [`ParametrisationOptions`](@ref): `:klu` forces KLU, `:pardiso`
requires the extension, and `:auto` prefers an available Pardiso implementation before
falling back to KLU. KLU reuses cached symbolic analysis while redoing numeric
factorisation with partial pivoting whenever the bordered matrix changes. Note
`klu_factor!`, not the exported `klu!`: the latter freezes the pivot sequence.

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
  needs distinct input and output vectors.  Empty on the KLU path, where `ldiv!` is
  genuinely in-place.
- `pardiso_matrix::Any` — the matrix handed to Pardiso's analysis phase; `nothing`
  until `_pardiso_prepare!` has run.
- `fact::Any` — cached KLU factorisation; `nothing` until the first successful one.
- `backend::B` — selected [`KLUBackend`](@ref) or [`PardisoBackend`](@ref).
- `residual_tolerance::Union{Nothing, RT}` — active backward-error threshold.
- `max_refinement_steps::Int` — maximum KLU refinement corrections.
- `residual_work::Vector{T}` — persistent KLU residual/norm workspace.
- `refinement_work::Vector{T}` — lazily allocated correction workspace.
- `max_relative_residual::RT` — largest accepted backward error observed so far.
- `refinement_count::Int` — total KLU refinement corrections performed.

`mutable` is load-bearing, not incidental: the Pardiso branch attaches a finaliser to
release C-side memory, and Julia refuses to finalise an immutable object.
"""
mutable struct SparseLinearSolverState{T, B <: AbstractSparseBackend, RT <: Real}
    bordered::SparseMatrixCSC{T}         # (FOM+ROM)²; constant pattern
    L_template::SparseMatrixCSC{T}       # FOM²; workspace for L(s)
    L_mappings::Vector{Vector{Int}}      # linear_terms[k].nzval → L_template.nzval
    border_row_base::Vector{Int}         # length FOM
    solve_scratch::Vector{T}             # Pardiso only; empty on the KLU path
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

"""
    SparseLinearSolverState{T}(L_template, L_mappings, FOM, ROM;
        options = ParametrisationOptions()) -> SparseLinearSolverState

Initialise the sparse solver state and constant-pattern bordered template. Backend
selection, residual verification, and refinement storage follow `options`.
"""
function SparseLinearSolverState{T}(
        L_template::SparseMatrixCSC{T},
        L_mappings::Vector{Vector{Int}},
        FOM::Int,
        ROM::Int;
        options::ParametrisationOptions = ParametrisationOptions()
) where {T}
    requested = options.backend
    ps = requested in (:auto, :pardiso) ? _try_build_pardiso_solver() : nothing
    requested == :pardiso && ps === nothing &&
        error(
            "ParametrisationOptions requested Pardiso, but no Pardiso backend is active")
    bordered, border_row_base = precompute_sparse_bordered_template(L_template, ROM)
    nsys = FOM + ROM
    RT = typeof(real(zero(T)))
    residual_tolerance = _configured_residual_tolerance(T, options)
    verify = !isnothing(residual_tolerance)
    backend = ps === nothing ? KLUBackend{verify}() : PardisoBackend{typeof(ps), verify}(ps)
    state = SparseLinearSolverState{T, typeof(backend), RT}(
        bordered, L_template, L_mappings, border_row_base,
        backend isa PardisoBackend ? Vector{T}(undef, nsys) : T[],
        nothing, nothing,
        backend, residual_tolerance, options.max_refinement_steps,
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

Finaliser: hand the Pardiso factorisation back. No-op on the KLU path, and
never allowed to throw — a finaliser that raises would be reported out of context.
"""
function _release_pardiso!(state::SparseLinearSolverState{T, <:KLUBackend}) where {T}
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

# =============================================================================
# CohomologicalContext — everything the per-monomial solve needs, precomputed
# =============================================================================

"""
	CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT}

Bundles all precomputed data required to solve the cohomological equations for
every monomial.

The solve visits every multi-index in turn, and nothing in this struct varies with
the multi-index: operator coefficients, resonance look-ups, buffers and the cached
factorisation are all built once by [`solve_cohomological_problem`](@ref) and reused.
That is what allows [`solve_single_monomial!`](@ref) to run without heap allocation.
Related data is grouped into named sub-structs rather than kept flat, so each field's
provenance is visible at the call site (`ctx.orthogonality.J_coeffs`, not a bare
`ctx.J_coeffs`).

# Type parameters

| Parameter | Meaning |
|:----------|:--------|
| `T`       | Scalar type (typically `ComplexF64`) |
| `ORD`     | Differential-equation order of the full-order model |
| `ORDP1`   | `ORD + 1` |
| `NVAR`    | Total reduced variables: `ROM + N_EXT` |
| `FOM`     | Full-order state dimension |
| `LT`      | Element type of the FOM matrices |
| `MT`      | Matrix type; sparse path when `MT <: SparseMatrixCSC` |

# Fields

- `linear_terms::NTuple{ORDP1, MT}` — the `ORD+1` FOM matrices `B₀ … B_ORD` whose
  combination `L(s) = Σ_j s^j B_j` forms the `(1,1)` block of the bordered system.
- `generalised_eigenmodes::Matrix{T}` — `FOM × NVAR` matrix of generalised **right**
  eigenmodes, formed by concatenating the master right eigenmodes and the solved external
  right directions. It propagates higher derivative coefficients after the linear
  monomials have been initialised. The left eigenmodes are represented separately through
  `orthogonality` and are never stored in this field.
- `lambda_diag::Vector{T}` — the `NVAR` reduced eigenvalues; the superharmonic of a
  monomial is `s = ⟨lambda_diag, α⟩`.
- `invariance::InvarianceOperators{T}` — border-column coefficients; see
  [`InvarianceOperators`](@ref).
- `orthogonality::OrthogonalityOperators{T}` — orthogonality row, corner and
  external coefficients; see [`OrthogonalityOperators`](@ref).
- `resonance_set::ResonanceSet` — which master modes are resonant with which
  monomial, deciding per row whether the border is populated or masked out.
- `linear_monomial_skip_set::Set{Int}` — positions of master linear monomials fixed from
  the eigenvectors before the loop. External linear directions are marked separately in
  the active conjugate-symmetry skip mask after they are solved.
- `lower_order::LowerOrderResources{NVAR, T}` — coupling buffers and multiindex
  look-up; see [`LowerOrderResources`](@ref).
- `buffers::CohomologicalBuffers{T}` — assembly and solve scratch; see
  [`CohomologicalBuffers`](@ref).
- `sparse_solver::Union{Nothing, SparseLinearSolverState{T}}` — sparse-path template
  and factorisation handles, or `nothing` on the dense path.  This field, not `MT`
  alone, is what the solve branches on.
"""
struct CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT <: AbstractMatrix{LT}}
    # ── Spectral / model data ─────────────────────────────────────────────────
    linear_terms::NTuple{ORDP1, MT}
    generalised_eigenmodes::Matrix{T}
    lambda_diag::Vector{T}
    # ── Precomputed operators ─────────────────────────────────────────────────
    invariance::InvarianceOperators{T}
    orthogonality::OrthogonalityOperators{T}
    # ── Resonance bookkeeping ─────────────────────────────────────────────────
    resonance_set::ResonanceSet
    linear_monomial_skip_set::Set{Int}
    # ── Compute resources ─────────────────────────────────────────────────────
    lower_order::LowerOrderResources{NVAR, T}
    buffers::CohomologicalBuffers{T}
    sparse_solver::Union{Nothing, SparseLinearSolverState{T}}
end
