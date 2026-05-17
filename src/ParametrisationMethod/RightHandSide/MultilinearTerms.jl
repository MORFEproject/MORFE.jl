"""
Module `MultilinearTerms` — efficient evaluation of the nonlinear right-hand side
of the cohomological equations.

For each monomial `α` the nonlinear contribution is

    Σₜ  Σ_{β₁+…+βₖ=α}  multiplier · t.f!(W[β₁], …, W[βₖ], r₁, …, rₘ)

where the outer sum runs over all nonlinear terms `t` of the model and the inner
sum enumerates factorisations of `α` into `k` sub-exponents from already-computed
`W` columns.  This module provides two evaluation paths:

- **Non-cached** (`compute_multilinear_terms` with an `SVector` exponent):
  calls the factorisation routines on every invocation.  Simple but allocating.

- **Cached** (`build_multilinear_terms_cache` + `compute_multilinear_terms!`):
  precomputes all factorisation bookkeeping in `MultilinearTermsCache` once before
  the solve loop, then replays it allocation-free at each monomial.

For FEM-backed terms (`FEMMultilinearMap`) an additional **O4 combined element loop**
merges all me=0 FEM contributions into a single mesh traversal per monomial,
avoiding redundant `fem_reinit!` and `scatter_qp!` calls.

Three symmetry strategies (`FullyAsymmetric`, `FullySymmetric`, `GroupwiseSymmetric`)
are dispatched at compile time from the `MultilinearMap.multiindex` field.
"""
module MultilinearTerms

using LinearAlgebra: axpy!
using StaticArrays: SVector

using ..Multiindices: indices_in_box_with_bounded_degree,
                      factorisations_asymmetric, factorisations_fully_symmetric,
                      factorisations_groupwise_symmetric,
                      bounded_index_tuples, FactorisationEntry
using ..ParametrisationMethod: Parametrisation
using ..FullOrderModel: NDOrderModel, MultilinearMap
using ..MultilinearMaps: AbstractMultilinearMap, FEMMultilinearMap,
                         fem_elements, fem_n_qp, fem_ndofs_per_cell,
                         fem_reinit!, scatter_qp!, accumulate_qp!, assemble_element!,
                         fem_getdetJdV, fem_qp_buffer

export compute_multilinear_terms, compute_multilinear_terms!, build_multilinear_terms_cache,
       MultilinearTermsCache

# -----------------------------------------------------------------------
# Symmetry classification
# -----------------------------------------------------------------------
#
# MultilinearMap.multiindex[k] is the number of factor slots that use the
# k-th derivative.  Three cases arise, each with a different accumulation strategy:
#
#   FullyAsymmetric    — all entries ≤ 1: distinct orders, t.f! goes directly
#                        into the accumulator (no scratch buffer needed).
#   FullySymmetric     — one positive entry > 1: all slots share one derivative,
#                        each factorisation carries a symmetry count.
#   GroupwiseSymmetric — multiple positive entries: slots span several derivatives,
#                        each factorisation carries a combined symmetry count.
#
# Dispatching on these tags lets Julia specialise the hot inner loop at
# compile time.

"""
    SymmetryType

Abstract tag type used to dispatch the inner factorisation accumulation strategy
at compile time.  The three concrete subtypes correspond to the three cases that
arise from `MultilinearMap.multiindex`:

- `FullyAsymmetric`    — all entries ≤ 1; each factor slot uses a different derivative.
- `FullySymmetric`     — exactly one entry > 1; all slots share one derivative.
- `GroupwiseSymmetric` — multiple entries > 0; slots span several derivatives.
"""
abstract type SymmetryType end

"""
    FullyAsymmetric <: SymmetryType

Tag for terms whose factor slots all use distinct derivatives (`multiindex` has no
repeated entry > 1).  No scratch buffer is needed; `t.f!` writes directly into
the accumulator.
"""
struct FullyAsymmetric <: SymmetryType end

"""
    FullySymmetric <: SymmetryType

Tag for terms where all factor slots share a single derivative order (exactly one
positive entry in `multiindex`).  Each factorisation carries a symmetry count.
"""
struct FullySymmetric <: SymmetryType end

"""
    GroupwiseSymmetric <: SymmetryType

Tag for terms whose factor slots span multiple derivative orders (multiple positive
entries in `multiindex`).  Uses `factorisations_groupwise_symmetric` with a combined
per-group symmetry count.
"""
struct GroupwiseSymmetric <: SymmetryType end

"""
    symmetry_type(t) -> SymmetryType

Classify a `AbstractMultilinearMap` as `FullyAsymmetric`, `FullySymmetric`,
or `GroupwiseSymmetric` based on its `multiindex` field.
"""
function symmetry_type(t::AbstractMultilinearMap)
    all(x -> x <= 1, t.multiindex) && return FullyAsymmetric()
    count(>(0), t.multiindex) == 1 && return FullySymmetric()
    return GroupwiseSymmetric()
end

# -----------------------------------------------------------------------
# Derivative order helper
# -----------------------------------------------------------------------

"""
	_derivative_orders(t) → NTuple

Map each factor slot to its 1-based derivative index.
Example: `multiindex = (2, 1)` → `(1, 1, 2)`.
"""
function _derivative_orders(t::AbstractMultilinearMap)
    deg = sum(t.multiindex)
    return ntuple(deg) do slot
        cumulative = 0
        for (k, cnt) in enumerate(t.multiindex)
            cumulative += cnt
            slot <= cumulative && return k
        end
    end
end

# -----------------------------------------------------------------------
# Per-split accumulation — one dispatch method per symmetry class
# -----------------------------------------------------------------------
#
# Each method accumulates the contributions for one (external-system) split
# of the exponent `rem` into `accum`.  `t.f!` must use += semantics.
#
# The caller is responsible for zeroing `accum` before calling when needed
# (see accumulate_multilinear_term! below).

"""
    _accumulate_split!(accum, scratch, t, sym, W, set, rem, deg, candidate_indices, args_ext)

Accumulate contributions from one (monomial, external-split) pair into `accum`,
dispatching on the `SymmetryType` tag `sym`.

- `FullyAsymmetric`: calls `t.f!` directly into `accum` for each factorisation.
- `FullySymmetric` / `GroupwiseSymmetric`: calls `t.f!` into `scratch`, then
  `axpy!(multiplier, scratch, accum)` to apply the symmetry count.
"""
function _accumulate_split!(accum, _scratch,
        t, ::FullyAsymmetric, W, set, rem, deg, candidate_indices, args_ext)
    orders = _derivative_orders(t)
    for entry in factorisations_asymmetric(set, rem, deg, candidate_indices)
        @inbounds args = ntuple(i -> @view(W[:, orders[i], entry.factor_indices[i]]), Val(deg))
        t.f!(accum, args..., args_ext...)
    end
end

# All factor slots share one derivative; each factorisation has a symmetry count.
function _accumulate_split!(accum, scratch,
        t, ::FullySymmetric, W, set, rem, deg, candidate_indices, args_ext)
    deriv_idx = findfirst(>(0), t.multiindex)::Int
    for entry in factorisations_fully_symmetric(set, rem, deg, candidate_indices)
        fill!(scratch, 0)
        @inbounds args = ntuple(i -> @view(W[:, deriv_idx, entry.factor_indices[i]]), Val(deg))
        t.f!(scratch, args..., args_ext...)
        axpy!(entry.multiplier, scratch, accum)
    end
end

# Factor slots span several derivatives; each factorisation has a combined count.
function _accumulate_split!(accum, scratch,
        t, ::GroupwiseSymmetric, W, set, rem, deg, candidate_indices, args_ext)
    orders = _derivative_orders(t)
    for entry in
        factorisations_groupwise_symmetric(set, rem, t.multiindex, candidate_indices)
        fill!(scratch, 0)
        @inbounds args = ntuple(i -> @view(W[:, orders[i], entry.factor_indices[i]]), Val(deg))
        t.f!(scratch, args..., args_ext...)
        axpy!(entry.multiplier, scratch, accum)
    end
end

# -----------------------------------------------------------------------
# Term-level accumulation
# -----------------------------------------------------------------------

"""
	accumulate_multilinear_term!(result, scratch, temp, t, parametrisation,
								  exp, candidate_indices, external_exp, unit_vectors)

Add the contribution of nonlinear term `t` for exponent `exp` to `result`.

When `t` has no external (forcing) slots (`me = 0`) there is exactly one split,
so we skip `bounded_index_tuples` and write directly into `result`, saving one
`fill!` and one `axpy!` per term.  For `me > 0` each split is accumulated into
`temp` first, then scaled into `result` via `axpy!`.
"""
function accumulate_multilinear_term!(result, scratch, temp,
        t::MultilinearMap{ORD}, parametrisation::Parametrisation{ORD, NVAR},
        exp::SVector{NVAR}, candidate_indices, external_exp, unit_vectors) where {ORD, NVAR}
    W = parametrisation.poly.coefficients
    set = parametrisation.poly.multiindex_set
    me = t.multiplicity_external
    deg = t.deg - me
    ROM = NVAR - parametrisation.external_system_size
    sym = symmetry_type(t)

    if me == 0
        _accumulate_split!(result, scratch, t, sym, W, set, exp, deg, candidate_indices, ())
    else
        for (ext_idx, ext_multiindex_external, ext_count) in
            bounded_index_tuples(me, external_exp)
            ext_multiindex = SVector(ntuple(
                i -> i <= ROM ? 0 :
                     ext_multiindex_external[i - ROM], Val(NVAR)))
            rem = exp - ext_multiindex
            args_ext = ntuple(i -> unit_vectors[ext_idx[i]], me)
            fill!(temp, 0)
            _accumulate_split!(
                temp, scratch, t, sym, W, set, rem, deg, candidate_indices, args_ext)
            axpy!(ext_count, temp, result)
        end
    end
end

# -----------------------------------------------------------------------
# Public API — non-cached
# -----------------------------------------------------------------------

"""
	compute_multilinear_terms(model, exp, parametrisation) → Vector

Return the sum of all nonlinear-term contributions for exponent `exp`.

Scratch buffers and shared data (`unit_vectors`, `candidate_indices`,
`external_exp`) are allocated once and reused across all terms.
"""
function compute_multilinear_terms(model::NDOrderModel{ORD}, exp::SVector{NVAR},
        parametrisation::Parametrisation{ORD, NVAR}) where {ORD, NVAR}
    set = parametrisation.poly.multiindex_set
    FOM = size(parametrisation)
    T = eltype(parametrisation.poly)
    deg_max = sum(exp)

    result = zeros(T, FOM)
    scratch = similar(result)   # per-factorisation scratch (symmetric branches)
    temp = similar(result)   # per-external-split accumulator (me > 0 only)

    external_system_size = parametrisation.external_system_size
    ROM = NVAR - external_system_size
    unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size))
                    for j in 1:external_system_size]
    candidate_indices = indices_in_box_with_bounded_degree(set, exp, 1, deg_max)
    external_exp = SVector(ntuple(i -> exp[ROM + i], external_system_size))

    for t in model.nonlinear_terms
        t.deg > deg_max && continue
        accumulate_multilinear_term!(result, scratch, temp, t, parametrisation,
            exp, candidate_indices, external_exp, unit_vectors)
    end
    return result
end

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

# -----------------------------------------------------------------------
# O4 — combined element loop across all FEM terms
# -----------------------------------------------------------------------
#
# For a model with N_FEM FEM terms (e.g. SVK: quadratic + cubic), the
# per-split path calls fem_reinit! N_FEM times per element per monomial,
# recomputing identical shape-function data each time.
#
# O4 merges all me=0 FEM term loops for a given monomial into a single
# element traversal: reinit! once, scatter each globally-unique W-column
# once, then accumulate contributions from all terms at every qp.

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

# -----------------------------------------------------------------------
# Cache construction helpers
# -----------------------------------------------------------------------

"""
    _orders_for_cache(sym, t, deg) -> Vector{Int}

Return the per-slot derivative order vector for storage in a `CachedSplit`.

- `FullySymmetric`: all `deg` slots share one derivative index.
- Other: read from `_derivative_orders(t)`.
"""
_orders_for_cache(::FullySymmetric, t, deg) = fill(findfirst(>(0), t.multiindex)::Int, deg)
_orders_for_cache(::SymmetryType, t, deg) = collect(Int, _derivative_orders(t))

"""
    _collect_entries(sym, t, mset, rem, deg, cands) -> Vector{FactorisationEntry}

Call the appropriate factorisation function for cache construction, routing
`FullyAsymmetric` to `factorisations_asymmetric`, `FullySymmetric` to
`factorisations_fully_symmetric`, and `GroupwiseSymmetric` to
`factorisations_groupwise_symmetric`.
"""
function _collect_entries(::FullyAsymmetric, t, mset, rem, deg, cands)
    factorisations_asymmetric(mset, rem, deg, cands)
end
function _collect_entries(::FullySymmetric, t, mset, rem, deg, cands)
    factorisations_fully_symmetric(mset, rem, deg, cands)
end
function _collect_entries(::GroupwiseSymmetric, t, mset, rem, deg, cands)
    factorisations_groupwise_symmetric(mset, rem, t.multiindex, cands)
end

"""
    _build_fem_cached_split(::Val{DEG}, cs) -> FEMCachedSplit{DEG}

Convert a `CachedSplit` to a `FEMCachedSplit{DEG}` for the FEM-batched replay path.
Deduplicates `(derivative_order, W_col_idx)` pairs across all entries and remaps
each entry's factor slots to local indices into the `unique_cols` table.
Called once at cache-build time; produces zero allocations in the hot path.
"""
function _build_fem_cached_split(::Val{DEG}, cs::CachedSplit) where {DEG}
    # 1. Enumerate unique (derivative_order, W_col_idx) pairs across all entries.
    unique_cols = Tuple{Int, Int}[]
    col_to_local = Dict{Tuple{Int, Int}, Int}()
    for entry in cs.entries
        for k in 1:DEG
            oc = (cs.orders[k], entry.factor_indices[k])
            if !haskey(col_to_local, oc)
                push!(unique_cols, oc)
                col_to_local[oc] = length(unique_cols)
            end
        end
    end
    # 2. Map each entry's factor slots to local indices in unique_cols.
    fem_entries = [FEMFactorisationEntry{DEG}(
                       entry.multiplier,
                       ntuple(k -> col_to_local[(cs.orders[k], entry.factor_indices[k])], Val(DEG))
                   )
                   for entry in cs.entries]
    return FEMCachedSplit{DEG}(cs.ext_count, cs.args_ext_indices, unique_cols, fem_entries)
end

"""
    _build_global_fem_split(model, fem_splits_l, fem_term_indices) -> FEMGlobalSplit

Build the O4 combined-element-loop bookkeeping for one monomial, merging all me=0
FEM terms into a single `FEMGlobalSplit`.

- Deduplicates `(order, col)` pairs across all me=0 FEM splits into a global table.
- Remaps each `FEMFactorisationEntry`'s local indices to global table indices.
- Groups entries by degree into a typed `Tuple` for type-stable compile-time dispatch.
- Returns an empty `FEMGlobalSplit{Tuple{}}` when no me=0 FEM terms are present.
"""
function _build_global_fem_split(model, fem_splits_l, fem_term_indices)
    isempty(fem_term_indices) && return FEMGlobalSplit{Tuple{}}([], (), 0, Int[])

    # --- Step 1: deduplicate (order, col) pairs across all me=0 FEM terms ---
    global_unique_cols = Tuple{Int, Int}[]
    col_to_global = Dict{Tuple{Int, Int}, Int}()

    for t_idx in fem_term_indices
        t = model.nonlinear_terms[t_idx]
        t.multiplicity_external == 0 || continue
        for fem_split in fem_splits_l[t_idx]
            isempty(fem_split.args_ext_indices) || continue   # skip me>0 splits
            for oc in fem_split.unique_cols
                if !haskey(col_to_global, oc)
                    push!(global_unique_cols, oc)
                    col_to_global[oc] = length(global_unique_cols)
                end
            end
        end
    end

    isempty(global_unique_cols) && return FEMGlobalSplit{Tuple{}}([], (), 0, Int[])

    # --- Step 2: remap FEMFactorisationEntry indices into global table, group by DEG ---
    entries_dict = Dict{Int, Vector{Any}}()   # DEG => Vector{FEMGlobalEntry{DEG}}

    for t_idx in fem_term_indices
        t = model.nonlinear_terms[t_idx]
        t.multiplicity_external == 0 || continue
        DEG = t.deg
        for fem_split in fem_splits_l[t_idx]
            isempty(fem_split.args_ext_indices) || continue
            for fem_entry in fem_split.fem_entries
                global_inds = ntuple(Val(DEG)) do k
                    col_to_global[fem_split.unique_cols[fem_entry.local_factor_indices[k]]]
                end
                gentry = FEMGlobalEntry{DEG}(t_idx, fem_entry.multiplier, global_inds)
                if !haskey(entries_dict, DEG)
                    entries_dict[DEG] = Any[]
                end
                push!(entries_dict[DEG], gentry)
            end
        end
    end

    # --- Step 3: build typed vectors per DEG and collect into a Tuple ---
    degs = sort!(collect(keys(entries_dict)))
    entries_by_deg = Tuple(
        [FEMGlobalEntry{d}(e.term_idx, e.multiplier, e.local_factor_indices)
         for e in entries_dict[d]] for d in degs)

    # --- Step 4: collect participating term indices ---
    participating = sort!(unique!(Int[
        e.term_idx
        for entries in values(entries_dict)
        for e in entries]))

    driver_term_idx = fem_term_indices[1]
    return FEMGlobalSplit(global_unique_cols, entries_by_deg, driver_term_idx, participating)
end

"""
	build_multilinear_terms_cache(model, parametrisation[, skip_bits]) → MultilinearTermsCache

Precompute all factorisation data for every monomial and term.
Valid as long as the multiindex set is unchanged.

When `skip_bits[l]` is `true` the cache entry for monomial `l` is left empty.
This is safe for monomials that will never be replayed (linear monomials, conjugate
secondaries): those entries are guarded by the same `skip_bits` check in the solve loop.
"""
function build_multilinear_terms_cache(
        model::NDOrderModel{ORD}, parametrisation::Parametrisation{ORD, NVAR}) where {
        ORD, NVAR}
    L = length(parametrisation.poly.multiindex_set)
    build_multilinear_terms_cache(model, parametrisation, falses(L))
end

function build_multilinear_terms_cache(
        model::NDOrderModel{ORD}, parametrisation::Parametrisation{ORD, NVAR},
        skip_bits::BitVector) where {ORD, NVAR}
    mset = parametrisation.poly.multiindex_set
    L = length(mset)
    n_terms = length(model.nonlinear_terms)
    external_system_size = parametrisation.external_system_size
    ROM = NVAR - external_system_size

    all_splits = Vector{Vector{Vector{CachedSplit}}}(undef, L)
    all_fem_splits = Vector{Vector{Vector{Any}}}(undef, L)

    for l in 1:L
        if skip_bits[l]
            all_splits[l]     = [CachedSplit[] for _ in 1:n_terms]
            all_fem_splits[l] = [[]            for _ in 1:n_terms]
            continue
        end
        exp = mset.exponents[l]
        deg_max = sum(exp)
        candidate_indices = indices_in_box_with_bounded_degree(mset, exp, 1, deg_max)
        external_exp = SVector(ntuple(i -> exp[ROM + i], external_system_size))

        term_splits = Vector{Vector{CachedSplit}}(undef, n_terms)
        fem_term_splits = Vector{Vector{Any}}(undef, n_terms)

        for (t_idx, t) in enumerate(model.nonlinear_terms)
            if t.deg > deg_max
                term_splits[t_idx] = CachedSplit[]
                fem_term_splits[t_idx] = []
                continue
            end

            me = t.multiplicity_external
            deg = t.deg - me

            if t isa FEMMultilinearMap
                # Build CachedSplit first (to reuse factorisation logic), then convert to FEMCachedSplit.
                sym = symmetry_type(t)
                is_asym = sym isa FullyAsymmetric
                orders = _orders_for_cache(sym, t, deg)
                raw_splits = CachedSplit[]
                if me == 0
                    entries = _collect_entries(sym, t, mset, exp, deg, candidate_indices)
                    push!(raw_splits, CachedSplit(1, Int[], is_asym, orders, entries))
                else
                    for (ext_idx, ext_multiindex_external, ext_count) in
                        bounded_index_tuples(me, external_exp)
                        ext_multiindex = SVector(ntuple(
                            i -> i <= ROM ? 0 : ext_multiindex_external[i - ROM], Val(NVAR)))
                        rem = exp - ext_multiindex
                        entries = _collect_entries(
                            sym, t, mset, rem, deg, candidate_indices)
                        push!(raw_splits,
                            CachedSplit(
                                ext_count, collect(Int, ext_idx), is_asym, orders, entries))
                    end
                end
                term_splits[t_idx] = raw_splits  # kept for reference / fallback
                fem_term_splits[t_idx] = [_build_fem_cached_split(Val(deg), cs)
                                          for cs in raw_splits]
            else
                sym = symmetry_type(t)
                is_asym = sym isa FullyAsymmetric
                orders = _orders_for_cache(sym, t, deg)
                splits = CachedSplit[]
                if me == 0
                    entries = _collect_entries(sym, t, mset, exp, deg, candidate_indices)
                    push!(splits, CachedSplit(1, Int[], is_asym, orders, entries))
                else
                    for (ext_idx, ext_multiindex_external, ext_count) in
                        bounded_index_tuples(me, external_exp)
                        ext_multiindex = SVector(ntuple(
                            i -> i <= ROM ? 0 : ext_multiindex_external[i - ROM], Val(NVAR)))
                        rem = exp - ext_multiindex
                        entries = _collect_entries(
                            sym, t, mset, rem, deg, candidate_indices)
                        push!(splits,
                            CachedSplit(
                                ext_count, collect(Int, ext_idx), is_asym, orders, entries))
                    end
                end
                term_splits[t_idx] = splits
                fem_term_splits[t_idx] = []
            end
        end

        all_splits[l] = term_splits
        all_fem_splits[l] = fem_term_splits
    end

    T = eltype(parametrisation.poly)
    FOM = size(parametrisation)
    unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size))
                    for j in 1:external_system_size]

    # Compute the element-residual buffer size from the FEM terms in the model.
    # The qp gradient buffer (∇W_qp) is term-specific and lives inside each FEMMultilinearMap.
    max_ndofs = 0
    for t in model.nonlinear_terms
        t isa FEMMultilinearMap || continue
        max_ndofs = max(max_ndofs, fem_ndofs_per_cell(t))
    end
    fem_Fe = zeros(T, max_ndofs)

    # --- O4: build global FEM splits and allocate shared buffers ---
    fem_term_indices = [t_idx for (t_idx, t) in enumerate(model.nonlinear_terms)
                        if t isa FEMMultilinearMap]

    global_fem_splits = Vector{Any}(undef, L)
    for l in 1:L
        global_fem_splits[l] = _build_global_fem_split(model, all_fem_splits[l], fem_term_indices)
    end

    if isempty(fem_term_indices)
        QP = Nothing
        global_∇W_qp = Matrix{Nothing}(undef, 0, 0)
    else
        driver_term = model.nonlinear_terms[fem_term_indices[1]]
        QP = eltype(fem_qp_buffer(driver_term))
        max_global_unique = maximum(
            (length(global_fem_splits[l].global_unique_cols) for l in 1:L); init = 0)
        max_n_qp = maximum(fem_n_qp(model.nonlinear_terms[i]) for i in fem_term_indices)
        global_∇W_qp = Matrix{QP}(undef, max(max_global_unique, 1), max(max_n_qp, 1))
    end

    global_Fe_buffers = Vector{Vector{T}}([
        (t isa FEMMultilinearMap ? zeros(T, fem_ndofs_per_cell(t)) : T[])
        for t in model.nonlinear_terms])

    return MultilinearTermsCache{T, QP}(all_splits, all_fem_splits, global_fem_splits,
        zeros(T, FOM), zeros(T, FOM), zeros(T, FOM),
        unit_vectors, fem_Fe, global_∇W_qp, global_Fe_buffers)
end

# -----------------------------------------------------------------------
# Public API — cached
# -----------------------------------------------------------------------

"""
    _replay_split!(result, scratch, temp, t, W, split, deg, unit_vectors)

Replay one `CachedSplit` into `result` using precomputed factorisation bookkeeping.

- `me = 0` (`args_ext_indices` empty): accumulates directly into `result`.
- `me > 0`: accumulates into `temp`, then `axpy!(ext_count, temp, result)`.

Dispatches asymmetric/symmetric accumulation via `split.is_asymmetric`.
"""
function _replay_split!(result, scratch, temp, t, W, split, deg, unit_vectors)
    if isempty(split.args_ext_indices)
        accum = result
        args_ext = ()
    else
        fill!(temp, 0)
        accum = temp
        args_ext = ntuple(i -> unit_vectors[split.args_ext_indices[i]], length(split.args_ext_indices))
    end

    if split.is_asymmetric
        for entry in split.entries
            @inbounds args = ntuple(k -> @view(W[:, split.orders[k], entry.factor_indices[k]]), Val(deg))
            t.f!(accum, args..., args_ext...)
        end
    else
        for entry in split.entries
            fill!(scratch, 0)
            @inbounds args = ntuple(k -> @view(W[:, split.orders[k], entry.factor_indices[k]]), Val(deg))
            t.f!(scratch, args..., args_ext...)
            axpy!(entry.multiplier, scratch, accum)
        end
    end

    isempty(split.args_ext_indices) || axpy!(split.ext_count, temp, result)
end

"""
    _replay_fem_split!(result, t, W, fem_split, Fe)

Replay one `FEMCachedSplit{DEG}` using the FEM-batched element loop.

For each element: calls `fem_reinit!` once, scatters each unique `(order, col)` W
column to qp-level field values via `scatter_qp!`, accumulates all qp contributions
via `accumulate_qp!`, and assembles the element residual into `result` via
`assemble_element!`.  The qp gradient buffer `∇W_qp` is obtained from the term
via `fem_qp_buffer(t)` and is owned by the term, not allocated here.
"""
function _replay_fem_split!(
        result, t::FEMMultilinearMap, W, fem_split::FEMCachedSplit{DEG},
        Fe) where {DEG}
    ∇W_qp = fem_qp_buffer(t)   # Matrix{QP_TYPE}(max_unique, n_qp) — owned by the term

    if isempty(fem_split.args_ext_indices)
        accum = result
    else
        # External-forcing case: accumulate into a temporary slice of Fe, then axpy!.
        # (Handling is analogous to _replay_split! for the me>0 branch.)
        fill!(Fe, zero(eltype(Fe)))
        accum = Fe   # will be axpy!-ed into result after the loop
    end

    n_qp = fem_n_qp(t)
    n_dofs = fem_ndofs_per_cell(t)
    n_uniq = length(fem_split.unique_cols)

    for element in fem_elements(t)

        # reinit! is called once per element, before any scatter_qp! calls for this element.
        fem_reinit!(element, t)

        # 1. Scatter each unique (order, col) W column to qp-level field quantities.
        for i in 1:n_uniq
            (order, col) = fem_split.unique_cols[i]
            scatter_qp!(@view(∇W_qp[i, 1:n_qp]), @view(W[:, order, col]), element, t)
        end

        # 2. Accumulate contributions from ALL fem_entries at each quadrature point.
        fill!(@view(Fe[1:n_dofs]), zero(eltype(Fe)))
        for q in 1:n_qp
            dΩ = fem_getdetJdV(element, q, t)
            for fem_entry in fem_split.fem_entries
                @inbounds ∇W_args = ntuple(k -> ∇W_qp[fem_entry.local_factor_indices[k], q], Val(DEG))
                accumulate_qp!(
                    @view(Fe[1:n_dofs]), ∇W_args, fem_entry.multiplier, element, q, dΩ, t)
            end
        end

        # 3. Scatter element residual to global accumulator.
        assemble_element!(accum, @view(Fe[1:n_dofs]), element, t)
    end

    isempty(fem_split.args_ext_indices) || axpy!(fem_split.ext_count, Fe, result)
end

# -----------------------------------------------------------------------
# O4 — combined element loop across all me=0 FEM terms
# -----------------------------------------------------------------------

"""
    _accumulate_global_entries!(Fe_bufs, ∇W_qp, entries_by_deg, model, element, q, dΩ)

Recursive type-stable dispatch over the degree-grouped entry tuple.  The base case
(empty tuple) is a no-op.  Each recursive step processes the head degree group —
Julia specialises a method per `DEG` so `Val(DEG)` in `ntuple` and the
`accumulate_qp!` dispatch are resolved at compile time — then recurses on the tail.
"""
@inline _accumulate_global_entries!(_, _, ::Tuple{}, _, _, _, _) = nothing

@inline function _accumulate_global_entries!(Fe_bufs, ∇W_qp,
        entries_by_deg::Tuple{Vector{FEMGlobalEntry{DEG}}, Vararg},
        model, element, q, dΩ) where {DEG}
    for gentry in first(entries_by_deg)
        t = model.nonlinear_terms[gentry.term_idx]
        ∇W_args = ntuple(k -> ∇W_qp[gentry.local_factor_indices[k], q], Val(DEG))
        accumulate_qp!(Fe_bufs[gentry.term_idx], ∇W_args, gentry.multiplier, element, q, dΩ, t)
    end
    _accumulate_global_entries!(Fe_bufs, ∇W_qp, Base.tail(entries_by_deg),
                                model, element, q, dΩ)
end

"""
    _replay_all_fem_splits!(result, model, W, global_split, global_∇W_qp, global_Fe_buffers)

Execute the O4 combined element loop: traverse the mesh once, scatter all globally-unique
W columns, then accumulate contributions from ALL me=0 FEM terms at each quadrature point.

`fem_reinit!` and `scatter_qp!` are called at most once per unique `(element, W-column)`
pair; `accumulate_qp!` dispatches per degree group via `_accumulate_global_entries!`.
"""
function _replay_all_fem_splits!(result, model, W,
        global_split::FEMGlobalSplit,
        global_∇W_qp, global_Fe_buffers)
    driver = model.nonlinear_terms[global_split.driver_term_idx]
    n_qp   = fem_n_qp(driver)
    n_uniq = length(global_split.global_unique_cols)
    participating = global_split.participating_term_indices

    for element in fem_elements(driver)
        fem_reinit!(element, driver)

        for i in 1:n_uniq
            (order, col) = global_split.global_unique_cols[i]
            scatter_qp!(@view(global_∇W_qp[i, 1:n_qp]), @view(W[:, order, col]), element, driver)
        end

        for t_idx in participating
            fill!(global_Fe_buffers[t_idx], zero(eltype(global_Fe_buffers[t_idx])))
        end

        for q in 1:n_qp
            dΩ = fem_getdetJdV(element, q, driver)
            _accumulate_global_entries!(global_Fe_buffers, global_∇W_qp,
                                        global_split.entries_by_deg, model, element, q, dΩ)
        end

        for t_idx in participating
            assemble_element!(result, global_Fe_buffers[t_idx], element,
                              model.nonlinear_terms[t_idx])
        end
    end
end

"""
    _replay_global_fem!(result, model, W, cache, exp_index)

Function barrier: retrieves `cache.global_fem_splits[exp_index]` (stored as `Any`) and
delegates to `_replay_all_fem_splits!`.  The barrier restores type stability because
`cache.global_∇W_qp::Matrix{QP}` is concretely typed at the call site even though the
`global_split` triggers runtime dispatch on `ENTRIES_TUPLE`.
"""
function _replay_global_fem!(result, model, W,
        cache::MultilinearTermsCache{T, QP}, exp_index) where {T, QP}
    gs = cache.global_fem_splits[exp_index]
    isempty(gs.global_unique_cols) && return
    _replay_all_fem_splits!(result, model, W, gs, cache.global_∇W_qp, cache.global_Fe_buffers)
end

# -----------------------------------------------------------------------
# Term-level dispatch — cached path
# -----------------------------------------------------------------------
#
# Multiple dispatch on the concrete term type selects the replay strategy.
# Adding a new AbstractMultilinearMap subtype requires only a new _replay_term! method;
# compute_multilinear_terms! never needs to be modified.

"""
    _replay_term!(result, t, W, exp_index, t_idx, cache)

Replay all cached splits for one nonlinear term `t` and monomial `exp_index` into
`result`.  Dispatches on the concrete term type:

- `MultilinearMap`: replays via `_replay_split!` (closure path).
- `FEMMultilinearMap`: me=0 splits are handled by `_replay_global_fem!` (O4 combined
  loop); only me>0 fallback splits are processed here via `_replay_fem_split!`.
"""
function _replay_term!(result, t::MultilinearMap, W, exp_index, t_idx,
        cache::MultilinearTermsCache)
    deg = t.deg - t.multiplicity_external
    for split in cache.splits[exp_index][t_idx]
        _replay_split!(result, cache.scratch_buffer, cache.temp_buffer,
                       t, W, split, deg, cache.unit_vectors)
    end
end

function _replay_term!(result, t::FEMMultilinearMap, W, exp_index, t_idx,
        cache::MultilinearTermsCache)
    # me=0 splits are handled by _replay_global_fem! (O4 combined loop).
    # Only the me>0 fallback splits need processing here.
    for fem_split in cache.fem_splits[exp_index][t_idx]
        isempty(fem_split.args_ext_indices) && continue
        _replay_fem_split!(result, t, W, fem_split, cache.fem_Fe)
    end
end

"""
	compute_multilinear_terms(model, exp_index, parametrisation, cache) → Vector

Cached variant of `compute_multilinear_terms`: replays precomputed factorisation
data instead of calling the factorisation routines.

`exp_index` is the 1-based index into `parametrisation`'s multiindex set.
"""
function compute_multilinear_terms(model::NDOrderModel{ORD}, exp_index::Int,
        parametrisation::Parametrisation{ORD, NVAR},
        cache::MultilinearTermsCache) where {ORD, NVAR}
    W = parametrisation.poly.coefficients
    FOM = size(parametrisation)
    T = eltype(parametrisation.poly)
    deg_max = sum(parametrisation.poly.multiindex_set.exponents[exp_index])

    result = zeros(T, FOM)
    scratch = similar(result)
    temp = similar(result)

    external_system_size = parametrisation.external_system_size
    unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size))
                    for j in 1:external_system_size]

    for (t_idx, t) in enumerate(model.nonlinear_terms)
        t.deg > deg_max && continue
        deg = t.deg - t.multiplicity_external
        for split in cache.splits[exp_index][t_idx]
            _replay_split!(result, scratch, temp, t, W, split, deg, unit_vectors)
        end
    end
    return result
end

"""
	compute_multilinear_terms!(result, model, exp_index, parametrisation, cache) → nothing

In-place variant: zeros `result` then accumulates all nonlinear contributions into it.
Uses `cache.scratch_buffer`, `cache.temp_buffer`, and `cache.unit_vectors` so that no
heap allocation occurs during the inner solve loop.
"""
function compute_multilinear_terms!(
        result::AbstractVector,
        model::NDOrderModel{ORD}, exp_index::Int,
        parametrisation::Parametrisation{ORD, NVAR},
        cache::MultilinearTermsCache) where {ORD, NVAR}
    fill!(result, zero(eltype(result)))
    W = parametrisation.poly.coefficients
    deg_max = sum(parametrisation.poly.multiindex_set.exponents[exp_index])

    # O4: single element loop covering all me=0 FEM terms.
    _replay_global_fem!(result, model, W, cache, exp_index)

    # Closure terms and me>0 FEM splits.
    for (t_idx, t) in enumerate(model.nonlinear_terms)
        t.deg > deg_max && continue
        _replay_term!(result, t, W, exp_index, t_idx, cache)
    end
    return nothing
end

end # module
