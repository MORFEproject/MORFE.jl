# =============================================================================
# Lower-order coupling resources
# =============================================================================

"""
	LowerOrderResources{NVAR, T}

Data needed to compute lower-order coupling vectors `ξ[j]` on every monomial.
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
ODE order `ORD`, and full-order dimension `FOM`.  Absorbs the `candidate_indices`
loop previously inlined in `solve_cohomological_problem`.
"""
function LowerOrderResources{NVAR, T}(
	mset::MultiindexSet{NVAR}, ORD::Int, FOM::Int,
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
		[SVector{NVAR, Int}(ntuple(k -> k == j ? 1 : 0, Val(NVAR))) for j in 1:NVAR],
	)
end

# =============================================================================
# Pre-allocated compute buffers
# =============================================================================

"""
	CohomologicalBuffers{T}

Pre-allocated scratch buffers for the cohomological system assembly and solve.
"""
struct CohomologicalBuffers{T}
	system_matrix::Matrix{T}   # (FOM+ROM)×(FOM+ROM); dense bordered system or Schur workspace
	rhs::Vector{T}             # length FOM+ROM; holds rhs then solution after ldiv!
	external_rhs::Vector{T}    # length FOM; scratch for evaluate_external_rhs!
	ml_result::Vector{T}       # length FOM; output of compute_multilinear_terms!
end

"""
	CohomologicalBuffers{T}(FOM, ROM) -> CohomologicalBuffers

Allocate all buffers for a system of full-order dimension `FOM` and `ROM` master modes.
"""
function CohomologicalBuffers{T}(FOM::Int, ROM::Int) where {T}
	return CohomologicalBuffers{T}(
		Matrix{T}(undef, FOM + ROM, FOM + ROM),
		Vector{T}(undef, FOM + ROM),
		zeros(T, FOM),
		zeros(T, FOM),
	)
end

# =============================================================================
# Sparse-path solver state
# =============================================================================

"""
	SparseLinearSolverState{T}

Sparse-path resources: the union-pattern `L(s)` template, its nzval index
mappings, solver handles (Pardiso priority 1; KLU priority 2), and a
pre-allocated bordered-RHS buffer that eliminates the per-monomial `hcat`
allocation on the resonant sparse path.
"""
struct SparseLinearSolverState{T}
	L_template::SparseMatrixCSC{T}
	L_mappings::Vector{Vector{Int}}
	pardiso::Any   # Nothing, or an AbstractPardisoSolver when Pardiso ext is loaded
	klu_cache::Ref{Any}                    # Ref(nothing) until first factorisation; always allocated
	rhs_extended::Matrix{T}              # FOM × (ROM+1); avoids hcat per resonant monomial
end

"""
	SparseLinearSolverState{T}(L_template, L_mappings, FOM, ROM) -> SparseLinearSolverState

Initialise the sparse solver state.  Tries MKL Pardiso first, then
open-source Pardiso, then falls back to KLU.
"""
function SparseLinearSolverState{T}(
	L_template::SparseMatrixCSC{T},
	L_mappings::Vector{Vector{Int}},
	FOM::Int,
	ROM::Int,
) where {T}
	ps = _try_build_pardiso_solver()
	return SparseLinearSolverState{T}(
		L_template, L_mappings, ps, Ref{Any}(nothing), Matrix{T}(undef, FOM, ROM + 1),
	)
end
