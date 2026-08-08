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
#   - which external arguments to pass as forcing arguments (unit vectors, or the
#     columns of the change of basis when the external system was re-based).
#
# What is NOT cached: the O(FOM) arithmetic (fill!, t.f!, axpy!).
# Those operations read from W, which changes at every solve step.

"""
	CachedSplit

Precomputed bookkeeping for one `(monomial l, term t, external-split)` triple.

# Fields

- `ext_count::Int` — multiplicity of this external-variable split (from
  `bounded_index_tuples`); always 1 when `me = 0`.
- `args_ext_indices::Vector{Int}` — indices into the cache's `external_arguments` that
  reconstruct the external forcing arguments; empty when `me = 0`.
- `is_asymmetric::Bool` — true iff the term is [`FullyAsymmetric`](@ref), in which
  case replaying it needs no scratch buffer.
- `orders::Vector{Int}` — derivative index for each factor slot; length is the
  internal degree of the term.
- `entries::Vector{FactorisationEntry}` — one entry per factorisation of the
  remainder exponent, each carrying its own symmetry multiplier.
"""
struct CachedSplit
    ext_count::Int
    args_ext_indices::Vector{Int}
    is_asymmetric::Bool
    orders::Vector{Int}
    entries::Vector{FactorisationEntry}
end

"""
	MultilinearTermsCache{T, QP, EV}

All precomputed factorisation bookkeeping for a given `(model, parametrisation)`
pair.  `splits[l][t_idx]` is the list of `CachedSplit` values for monomial `l`
and term `t_idx`; an empty list means the term degree exceeds the monomial degree
and it contributes nothing.

Type parameters:
- `T`  — element type of FOM-length vectors (e.g. `ComplexF64`)
- `QP` — element type of the combined qp gradient buffer `global_∇W_qp`
		  (e.g. `Tensor{2,3,ComplexF64}` for Ferrite SVK; `Nothing` when no FEM terms)
- `EV` — element type of `external_arguments`, i.e. `SVector{N_EXT, Int}` normally and
		  `SVector{N_EXT, eltype(Q)}` when the external system was re-based.  A type
		  parameter so the hot loop's argument tuple is inferred rather than `Any`.

**Build** once before the solve loop with `build_multilinear_terms_cache`.
**Use** by passing to the `(model, exp_index, parametrisation, cache)` overload of
`compute_multilinear_terms` inside the loop.  The cache is valid as long as the
multiindex set and the model structure are unchanged (i.e. across all solve steps).

# Fields

Three parallel split representations coexist because a model may mix closure-based
and FEM-backed nonlinear terms, and the FEM ones are far cheaper to evaluate batched
over elements than term by term:

- `splits::Vector{Vector{Vector{CachedSplit}}}` — the generic path.
  `splits[l][t_idx]` lists the [`CachedSplit`](@ref) values for monomial `l` and
  term `t_idx`; empty when the term degree exceeds the monomial degree.
- `fem_splits::Vector{Vector{Vector{Any}}}` — the same indexing for FEM terms,
  holding `FEMCachedSplit{DEG, ME}` values, empty for closure terms.  Typed `Any`
  because `DEG` varies per term; reached through a function barrier so the hot loop
  stays type-stable.
- `global_fem_splits::Vector{Any}` — one `FEMGlobalSplit` per monomial, fusing every
  FEM term into a single element loop.  `Any` for the same reason.

Buffers, all reused across monomials to keep the solve loop allocation-free:

- `result_buffer::Vector{T}` — length `FOM`, accumulates the nonlinear contribution
  returned to the caller.
- `scratch_buffer::Vector{T}` — length `FOM`, working space for symmetric terms;
  unused when a term is `FullyAsymmetric`.
- `temp_buffer::Vector{T}` — length `FOM`, holds one intermediate contraction.
- `external_arguments::Vector{EV}` — the external argument passed for each external
  variable, used to rebuild the external forcing arguments; empty when `N_EXT == 0`.
  These are the unit vectors `eⱼ` in the model's own coordinates, or the columns
  `Q[:, j]` of the change of basis when the external system was re-based — see
  `ExternalSystems.external_argument_vectors`.  Applying `Q` here, once per solve, is
  why no term ever has to know about the change of coordinates.
- `fem_Fe::Vector{T}` — element-local residual for the external-multiplicity
  fallback path, sized to the largest `ndofs_per_cell` in the model.
- `global_∇W_qp::Matrix{QP}` — shared quadrature-point gradient buffer,
  `max_global_unique × max_n_qp`.  `QP` is a type parameter precisely so element
  access inside the hot loop is statically typed.
- `global_Fe_buffers::Vector{Vector{T}}` — per-term element residual buffers, sized
  to each FEM term's `fem_ndofs_per_cell`; empty for closure terms.
"""
struct MultilinearTermsCache{T, QP, EV}
    splits::Vector{Vector{Vector{CachedSplit}}}
    fem_splits::Vector{Vector{Vector{Any}}}        # Vector{FEMCachedSplit{DEG,ME}}; Any as they vary
    global_fem_splits::Vector{Any}                 # FEMGlobalSplit per monomial
    result_buffer::Vector{T}
    scratch_buffer::Vector{T}
    temp_buffer::Vector{T}
    external_arguments::Vector{EV}                 # eⱼ, or Q[:, j]; empty when N_EXT == 0
    fem_Fe::Vector{T}                              # size: max_ndofs_per_cell
    global_∇W_qp::Matrix{QP}                       # max_global_unique × max_n_qp
    global_Fe_buffers::Vector{Vector{T}}
end

# -----------------------------------------------------------------------
# FEM-specific cache structs
# -----------------------------------------------------------------------

"""
	FEMFactorisationEntry{DEG}

One factorisation entry for the FEM-batched path.

# Fields

- `multiplier::Int` — symmetry count, as in `FactorisationEntry.multiplier`.
- `local_factor_indices::NTuple{DEG, Int}` — for each factor slot, the index into
  the enclosing [`FEMCachedSplit`](@ref)`.unique_cols`.  An `NTuple` rather than a
  `Vector` so `ntuple(k -> …, Val(DEG))` in the hot loop unrolls at compile time.
"""
struct FEMFactorisationEntry{DEG}
    multiplier::Int
    local_factor_indices::NTuple{DEG, Int}
end

"""
	FEMCachedSplit{DEG, ME}

Precomputed bookkeeping for one (monomial, FEM-term, external-split) triple.

`DEG` is the *internal* degree (`t.deg - me`) and `ME` the external multiplicity, so
`accumulate_qp!` is called with `DEG + ME == t.deg` arguments.  `ME` is a type parameter so
the external arguments are appended with a compile-time-known count in `_replay_fem_split!`.

# Fields

- `ext_count::Int` — external multiplicity; 1 when `me = 0`.
- `args_ext_indices::Vector{Int}` — indices into the cache's `external_arguments` that
  reconstruct the external arguments; empty when `me = 0`.
- `unique_cols::Vector{Tuple{Int, Int}}` — deduplicated `(derivative_order,
  W_col_idx)` pairs across every entry in this split.  Deduplication is what makes
  the batching pay: each pair is scattered to quadrature-point gradients once per
  element, however many factorisations reference it.
- `fem_entries::Vector{FEMFactorisationEntry{DEG}}` — one entry per factorisation,
  indexing into `unique_cols` rather than into `W` directly.  Keyed on the internal
  degree alone, since it only indexes `unique_cols`.
"""
struct FEMCachedSplit{DEG, ME}
    ext_count::Int
    args_ext_indices::Vector{Int}
    unique_cols::Vector{Tuple{Int, Int}}
    fem_entries::Vector{FEMFactorisationEntry{DEG}}
end

"""
	FEMGlobalEntry{DEG}

One factorisation entry in the combined element loop for a given monomial.

# Fields

- `term_idx::Int` — index into `model.nonlinear_terms`.  Present here but not in
  [`FEMFactorisationEntry`](@ref) because the combined loop mixes terms.
- `multiplier::Int` — symmetry count, as in [`FEMFactorisationEntry`](@ref).
- `local_factor_indices::NTuple{DEG, Int}` — indices into the enclosing
  [`FEMGlobalSplit`](@ref)`.global_unique_cols`.
"""
struct FEMGlobalEntry{DEG}
    term_idx::Int
    multiplier::Int
    local_factor_indices::NTuple{DEG, Int}
end

"""
	FEMGlobalSplit{ENTRIES_TUPLE}

All combined-loop bookkeeping for one monomial, covering all me=0 FEM terms.

# Fields

- `global_unique_cols::Vector{Tuple{Int, Int}}` — deduplicated `(derivative_order,
  W_col_idx)` pairs across *all* `me = 0` FEM terms and their splits, scattered once
  per element.  Deduplicating across terms, not just within one, is what the
  combined loop buys over [`FEMCachedSplit`](@ref).
- `entries_by_deg::ENTRIES_TUPLE` — one `Vector{FEMGlobalEntry{D}}` per degree `D`
  present, held in a typed tuple so the inner loop dispatches type-stably on `DEG`.
- `driver_term_idx::Int` — the FEM term whose element iterator drives the loop; `0`
  when this monomial has no `me = 0` FEM terms.
- `participating_term_indices::Vector{Int}` — sorted distinct `term_idx` values
  appearing in `entries_by_deg`, so assembly touches only the terms involved instead
  of scanning all of them.
"""
struct FEMGlobalSplit{ENTRIES_TUPLE}
    global_unique_cols::Vector{Tuple{Int, Int}}
    entries_by_deg::ENTRIES_TUPLE
    driver_term_idx::Int
    participating_term_indices::Vector{Int}
end
