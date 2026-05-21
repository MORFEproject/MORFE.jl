# -----------------------------------------------------------------------
# Factorisation cache — structs
# -----------------------------------------------------------------------
#
# Use case: the parametrisation solve loop calls compute_multilinear_terms
# once per monomial (O(L) calls, where L = |mset|).  Each call invokes the
# factorisation routines, which enumerate index tuples and allocate Vectors.
# For large L this GC pressure is measurable.
#
# Since factorisation results depend only on the multiindex set and the model
# structure — not on the parametrisation coefficients W — they can be
# precomputed once and replayed on every call.
#
# What IS cached: for each (monomial, term, external-split) triple,
#   - which coefficient indices to load from W,
#   - their symmetry multipliers,
#   - which external unit vectors to pass as forcing arguments.
#
# What is NOT cached: the O(FOM) arithmetic (fill!, t.f!, axpy!).
# Those operations read from W, which changes at every solve step.

"""
	CachedSplit

Precomputed bookkeeping for one `(monomial l, term t, external-split)` triple.

- `ext_count`        — multiplicity of this external-variable split
					   (from `bounded_index_tuples`); always 1 when `me = 0`.
- `args_ext_indices` — indices into the `unit_vectors` array that reconstruct
					   the external forcing arguments; empty when `me = 0`.
- `is_asymmetric`    — true iff `t` is `FullyAsymmetric` (no scratch buffer
					   needed when replaying).
- `orders`           — derivative index for each factor slot (length = deg_internal).
- `entries`          — list of `FactorisationEntry` values, one per factorisation
					   of the remainder exponent.
"""
struct CachedSplit
	ext_count::Int
	args_ext_indices::Vector{Int}
	is_asymmetric::Bool
	orders::Vector{Int}
	entries::Vector{FactorisationEntry}
end

"""
	MultilinearTermsCache{T, QP}

All precomputed factorisation bookkeeping for a given `(model, parametrisation)`
pair.  `splits[l][t_idx]` is the list of `CachedSplit` values for monomial `l`
and term `t_idx`; an empty list means the term degree exceeds the monomial degree
and it contributes nothing.

Type parameters:
- `T`  — element type of FOM-length vectors (e.g. `ComplexF64`)
- `QP` — element type of the combined qp gradient buffer `global_∇W_qp`
		  (e.g. `Tensor{2,3,ComplexF64}` for Ferrite SVK; `Nothing` when no FEM terms)

**Build** once before the solve loop with `build_multilinear_terms_cache`.
**Use** by passing to the `(model, exp_index, parametrisation, cache)` overload of
`compute_multilinear_terms` inside the loop.  The cache is valid as long as the
multiindex set and the model structure are unchanged (i.e. across all solve steps).
"""
struct MultilinearTermsCache{T, QP}
	splits::Vector{Vector{Vector{CachedSplit}}}
	# FEM-batched splits: fem_splits[l][t_idx] is Vector{FEMCachedSplit{DEG}} for FEM terms,
	# or an empty Vector for closure terms. Element type is Any because DEG varies per term.
	fem_splits::Vector{Vector{Vector{Any}}}
	# O4 combined loop: one FEMGlobalSplit per monomial (element type Any because ENTRIES_TUPLE
	# varies by monomial degree). Accessed via _replay_global_fem! function barrier.
	global_fem_splits::Vector{Any}
	result_buffer::Vector{T}
	scratch_buffer::Vector{T}
	temp_buffer::Vector{T}
	unit_vectors::Vector   # Vector of SVector{N_EXT, Int}; empty when N_EXT == 0
	# Pre-allocated element-local residual buffer for the me>0 fallback path.
	fem_Fe::Vector{T}     # size: max_ndofs_per_cell
	# O4 shared qp gradient buffer: (max_global_unique × max_n_qp).
	# QP is a type parameter so element access inside the hot loop is type-stable.
	global_∇W_qp::Matrix{QP}
	# O4 per-term element residual buffers: global_Fe_buffers[t_idx] is sized to
	# fem_ndofs_per_cell(t) for FEM terms; empty Vector{T} for closure terms.
	global_Fe_buffers::Vector{Vector{T}}
end

# -----------------------------------------------------------------------
# FEM-specific cache structs
# -----------------------------------------------------------------------

"""
	FEMFactorisationEntry{DEG}

One factorisation entry for the FEM-batched path.

- `multiplier`           — symmetry count (same as FactorisationEntry.multiplier).
- `local_factor_indices` — NTuple of length DEG: for each factor slot, the index into
						   `FEMCachedSplit.unique_cols` for the enclosing split.
						   NTuple (not Vector) so that `ntuple(k->..., Val(DEG))` in the
						   hot loop is unrolled at compile time.
"""
struct FEMFactorisationEntry{DEG}
	multiplier::Int
	local_factor_indices::NTuple{DEG, Int}
end

"""
	FEMCachedSplit{DEG}

Precomputed bookkeeping for one (monomial, FEM-term, external-split) triple.

- `ext_count`        — external multiplicity (1 when me = 0).
- `args_ext_indices` — unit vector indices for external args (empty when me = 0).
- `unique_cols`      — deduplicated list of (derivative_order, W_col_idx) pairs across all
					   entries in this split.  Scattered to qp-level gradients once per element.
- `fem_entries`      — one FEMFactorisationEntry{DEG} per factorisation, with local indices
					   into unique_cols.
"""
struct FEMCachedSplit{DEG}
	ext_count::Int
	args_ext_indices::Vector{Int}
	unique_cols::Vector{Tuple{Int, Int}}
	fem_entries::Vector{FEMFactorisationEntry{DEG}}
end

"""
	FEMGlobalEntry{DEG}

One factorisation entry in the combined element loop for a given monomial.

- `term_idx`             — index into `model.nonlinear_terms`
- `multiplier`           — symmetry count (from `FEMFactorisationEntry`)
- `local_factor_indices` — NTuple{DEG,Int}: indices into the enclosing
						   `FEMGlobalSplit.global_unique_cols` table.
"""
struct FEMGlobalEntry{DEG}
	term_idx::Int
	multiplier::Int
	local_factor_indices::NTuple{DEG, Int}
end

"""
	FEMGlobalSplit{ENTRIES_TUPLE}

All combined-loop bookkeeping for one monomial, covering all me=0 FEM terms.

- `global_unique_cols`        — deduplicated (derivative_order, W_col_idx) pairs across
								 ALL me=0 FEM terms and their splits. Scattered once per element.
- `entries_by_deg`            — one `Vector{FEMGlobalEntry{D}}` per degree D present; stored as
								 a typed Tuple so the inner loop dispatches type-stably on DEG.
- `driver_term_idx`           — index of the FEM term that drives the element iterator; 0 when
								 the split is empty (no me=0 FEM terms for this monomial).
- `participating_term_indices` — sorted distinct `term_idx` values referenced in entries_by_deg;
								 used in the assembly step to avoid scanning all terms.
"""
struct FEMGlobalSplit{ENTRIES_TUPLE}
	global_unique_cols::Vector{Tuple{Int, Int}}
	entries_by_deg::ENTRIES_TUPLE
	driver_term_idx::Int
	participating_term_indices::Vector{Int}
end
