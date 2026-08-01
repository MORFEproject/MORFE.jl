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
    participating = sort!(unique!(Int[e.term_idx
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
            all_splits[l] = [CachedSplit[] for _ in 1:n_terms]
            all_fem_splits[l] = [[] for _ in 1:n_terms]
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
                    for (ext_idx, ext_multiindex_external, ext_count) in bounded_index_tuples(me, external_exp)
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
                    for (ext_idx, ext_multiindex_external, ext_count) in bounded_index_tuples(me, external_exp)
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
    fem_term_indices = [t_idx
                        for (t_idx, t) in enumerate(model.nonlinear_terms)
                        if t isa FEMMultilinearMap]

    if !isempty(fem_term_indices)
        @info "Using optimised FEM path: cached per-element loop" n_fem_terms = length(fem_term_indices)
    end

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

    global_Fe_buffers = Vector{Vector{T}}([(t isa FEMMultilinearMap ?
                                            zeros(T, fem_ndofs_per_cell(t)) : T[])
                                           for t in model.nonlinear_terms])

    return MultilinearTermsCache{T, QP}(all_splits, all_fem_splits, global_fem_splits,
        zeros(T, FOM), zeros(T, FOM), zeros(T, FOM),
        unit_vectors, fem_Fe, global_∇W_qp, global_Fe_buffers)
end
