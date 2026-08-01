# =============================================================================
# Lower-order coupling resources
# =============================================================================

"""
	LowerOrderResources{NVAR, T}

Data needed to compute lower-order coupling vectors `ξ[j]` on every monomial.

Bundled into one struct so the per-monomial buffers and the multiindex lookup are
allocated once for the whole solve rather than per call.
"""
struct LowerOrderResources{NVAR, T}
    multiindex_dict::Dict{SVector{NVAR, Int}, Int}
    buffer::Vector{Vector{T}}               # length ORD, each FOM; zeroed before each call
    candidate_indices::Vector{Vector{Int}}  # length L; precomputed per monomial
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
"""
struct CohomologicalBuffers{T}
    system_matrix::Matrix{T}       # (FOM+ROM)×(FOM+ROM); dense path only
    orthogonality_rows::Matrix{T}  # ROM×(FOM+ROM); sparse path only — staged Ĵ(s) rows + corner
    rhs::Vector{T}                 # length FOM+ROM; holds rhs then solution after ldiv!
    external_rhs::Vector{T}        # length FOM; scratch for evaluate_external_rhs!
    ml_result::Vector{T}           # length FOM; output of compute_multilinear_terms!
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
"""
# `mutable` is load-bearing, not incidental: the Pardiso branch attaches a finaliser
# to release C-side memory, and Julia refuses to finalise an immutable object.
mutable struct SparseLinearSolverState{T}
    bordered::SparseMatrixCSC{T}         # (FOM+ROM)²; constant pattern, per-monomial nzval
    L_template::SparseMatrixCSC{T}       # FOM²; Horner workspace for L(s)
    L_mappings::Vector{Vector{Int}}      # linear_terms[k].nzval → L_template.nzval
    border_row_base::Vector{Int}         # length FOM; bordered[FOM+r, c] at base[c]+r-1
    solve_scratch::Vector{T}             # Pardiso only: RHS copy, since its solve needs
    # distinct in/out. Empty on the KLU path, whose
    # ldiv! is genuinely in-place.
    pardiso::Any                         # Nothing, or an AbstractPardisoSolver when the ext is loaded
    pardiso_matrix::Any                  # nothing until _pardiso_prepare! has run
    fact::Any                            # nothing until the first successful factorisation
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
        ROM::Int
) where {T}
    ps = _try_build_pardiso_solver()
    bordered, border_row_base = precompute_sparse_bordered_template(L_template, ROM)
    state = SparseLinearSolverState{T}(
        bordered, L_template, L_mappings, border_row_base,
        ps === nothing ? T[] : Vector{T}(undef, FOM + ROM),
        ps, nothing, nothing
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
