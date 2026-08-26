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
	CohomologicalBuffers{T}

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
"""
struct CohomologicalBuffers{T}
    system_matrix::Matrix{T}       # (FOM+ROM)²; dense path only
    orthogonality_rows::Matrix{T}  # ROM×(FOM+ROM); sparse path only
    rhs::Vector{T}                 # length FOM+ROM; rhs in, solution out
    external_rhs::Vector{T}        # length FOM
    ml_result::Vector{T}           # length FOM
end

"""
	CohomologicalBuffers(T, MT, FOM, ROM) -> CohomologicalBuffers

Allocate all buffers for a system of full-order dimension `FOM` and `ROM` master
modes.  Dispatches on the FOM matrix type `MT`: `MT <: SparseMatrixCSC` selects the
sparse layout, everything else the dense one.
"""
function CohomologicalBuffers(::Type{T}, ::Type{MT}, FOM::Int, ROM::Int) where {T, MT}
    return CohomologicalBuffers{T}(
        Matrix{T}(undef, FOM + ROM, FOM + ROM),
        Matrix{T}(undef, 0, 0),
        Vector{T}(undef, FOM + ROM),
        zeros(T, FOM),
        zeros(T, FOM)
    )
end
function CohomologicalBuffers(::Type{T}, ::Type{MT}, FOM::Int, ROM::Int) where {
        T, MT <: SparseMatrixCSC}
    return CohomologicalBuffers{T}(
        Matrix{T}(undef, 0, 0),
        Matrix{T}(undef, ROM, FOM + ROM),
        Vector{T}(undef, FOM + ROM),
        zeros(T, FOM),
        zeros(T, FOM)
    )
end

# =============================================================================
# Sparse-path solver state
# =============================================================================

"""
	SparseLinearSolverState{T}

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

Solver priority: Pardiso when the extension is loaded, otherwise KLU via
`klu`/`klu_factor!` — the latter reusing the cached symbolic analysis while redoing
the numeric factorisation *with* partial pivoting on every monomial, which is what
varying `s` requires.  Note `klu_factor!`, not the exported `klu!`: that one is
`klu_refactor` and freezes the pivot sequence.

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
- `pardiso::Any` — an `AbstractPardisoSolver` when the extension is loaded,
  otherwise `nothing`.  Untyped because the type only exists with the weak
  dependency present.
- `pardiso_matrix::Any` — the matrix handed to Pardiso's analysis phase; `nothing`
  until `_pardiso_prepare!` has run.
- `fact::Any` — the cached factorisation; `nothing` until the first successful one.

`mutable` is load-bearing, not incidental: the Pardiso branch attaches a finaliser to
release C-side memory, and Julia refuses to finalise an immutable object.
"""
mutable struct SparseLinearSolverState{T}
    bordered::SparseMatrixCSC{T}         # (FOM+ROM)²; constant pattern
    L_template::SparseMatrixCSC{T}       # FOM²; workspace for L(s)
    L_mappings::Vector{Vector{Int}}      # linear_terms[k].nzval → L_template.nzval
    border_row_base::Vector{Int}         # length FOM
    solve_scratch::Vector{T}             # Pardiso only; empty on the KLU path
    pardiso::Any                         # nothing, or an AbstractPardisoSolver
    pardiso_matrix::Any                  # nothing until _pardiso_prepare! has run
    fact::Any                            # nothing until the first factorisation
    backend::Symbol
    residual_tolerance::Union{Nothing, Float64}
    rhs_input::Vector{T}
    residual_work::Vector{T}
    last_factor_key::Any
    max_relative_residual::Float64
    refinement_count::Int
    factorization_count::Int
    solve_count::Int
end

"""
	SparseLinearSolverState{T}(L_template, L_mappings, FOM, ROM) -> SparseLinearSolverState

Initialise the sparse solver state: build the constant-pattern bordered template
around `L_template` and probe for Pardiso (MKL first, then open-source), falling
back to KLU.
"""
function SparseLinearSolverState{T}(
        L_template::SparseMatrixCSC{T},
        L_mappings::Vector{Vector{Int}},
        FOM::Int,
        ROM::Int;
        config::CohomologicalSolverConfig=CohomologicalSolverConfig()
) where {T}
    requested = config.backend
    ps = requested in (:auto, :pardiso) ? _try_build_pardiso_solver() : nothing
    requested == :pardiso && ps === nothing && error(
        "CohomologicalSolverConfig requested Pardiso, but no Pardiso backend is active")
    backend = requested == :auto ? (ps === nothing ? :klu : :pardiso) : requested
    bordered, border_row_base = precompute_sparse_bordered_template(L_template, ROM)
    needs_input_copy = backend in (:umfpack, :pardiso) ||
        config.residual_tolerance !== nothing
    nsys = FOM + ROM
    state = SparseLinearSolverState{T}(
        bordered, L_template, L_mappings, border_row_base,
        backend in (:pardiso, :umfpack) ? Vector{T}(undef, nsys) : T[],
        ps, nothing, nothing,
        backend, config.residual_tolerance,
        needs_input_copy ? Vector{T}(undef, nsys) : T[],
        config.residual_tolerance === nothing ? T[] : Vector{T}(undef, nsys),
        nothing, 0.0, 0, 0, 0
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
function _release_pardiso!(state::SparseLinearSolverState)
    state.pardiso === nothing && return nothing
    try
        _pardiso_release!(state.pardiso, state.pardiso_matrix)
    catch
        # Nothing useful to do during finalisation.
    end
    return nothing
end
