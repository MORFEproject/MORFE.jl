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
	_cached_splits(t, mset, exp, candidates, ROM) -> Vector{CachedSplit}

The splits of term `t` at the monomial `exp`: one when the term has no external factor,
otherwise one per assignment of external variables to its external slots, each holding the
factorisations of the remaining exponent. Empty when the term degree exceeds the monomial
degree.
"""
function _cached_splits(t, mset, exp::SVector{NVAR}, candidates, ROM) where {NVAR}
    t.deg > sum(exp) && return CachedSplit[]
    me = t.multiplicity_external
    deg = t.deg - me
    sym = symmetry_type(t)
    orders = _orders_for_cache(sym, t, deg)
    function make_split(ext_count, ext_idx, rem)
        entries = _collect_entries(sym, t, mset, rem, deg, candidates)
        return CachedSplit(ext_count, ext_idx, sym isa FullyAsymmetric, orders, entries)
    end
    me == 0 && return [make_split(1, Int[], exp)]
    external_exp = SVector(ntuple(i -> exp[ROM + i], NVAR - ROM))
    return [make_split(ext_count, collect(Int, ext_idx),
                exp - SVector(ntuple(i -> i <= ROM ? 0 : ext_exp[i - ROM], Val(NVAR))))
            for (ext_idx, ext_exp, ext_count) in bounded_index_tuples(me, external_exp)]
end

"""
	_fem_groups(fem_terms) -> Vector{Vector{Int}}

The indices into `fem_terms` of each group of terms with `isequal` `fem_element_loop`
labels, in order of first appearance.
"""
function _fem_groups(fem_terms)
    labels = map(fem_element_loop, fem_terms)
    groups = Vector{Int}[]
    for i in eachindex(labels)
        g = findfirst(group -> isequal(labels[first(group)], labels[i]), groups)
        g === nothing ? push!(groups, [i]) : push!(groups[g], i)
    end
    return groups
end

"""
	_fem_monomial(fem_terms, fem_splits) -> (cols, entries)

Pair every FEM term of one element loop with every factorisation of each of its splits at
one monomial, for the single-pass assembly. `cols` lists the distinct
`(derivative order, W column)` pairs that the factorisations read, across the terms, so each
is scattered to the quadrature points once per element. `entries` holds, per term of
`fem_terms`, a `Vector{FEMEntry}` indexing into `cols`; its type depends only on the terms,
so it is the same at every monomial.
"""
function _fem_monomial(fem_terms::Tuple, fem_splits)
    cols = Tuple{Int, Int}[]
    rows = Dict{Tuple{Int, Int}, Int}()
    row(col) = get!(() -> (push!(cols, col); length(cols)), rows, col)
    entries = ntuple(length(fem_terms)) do i
        ME = fem_terms[i].multiplicity_external
        DEG = fem_terms[i].deg - ME
        FEMEntry{DEG, ME}[FEMEntry{DEG, ME}(s.ext_count * e.multiplier,
                              ntuple(k -> row((s.orders[k], e.factor_indices[k])), DEG),
                              ntuple(k -> s.args_ext_indices[k], ME))
                          for s in fem_splits[i] for e in s.entries]
    end
    return cols, entries
end

"""
	_fem_group(terms, cols, entries, T) -> FEMGroup

The [`FEMGroup`](@ref) of `terms`, which share one element loop, with buffers sized to it.
Its terms must agree on the quadrature points and the DOFs per element, the only part of
the shared loop MORFE can check; an `ArgumentError` is thrown otherwise.
"""
function _fem_group(terms, cols, entries, T)
    driver = first(terms)
    n_qp = fem_n_qp(driver)
    n_dofs = fem_ndofs_per_cell(driver)
    all(t -> fem_n_qp(t) == n_qp && fem_ndofs_per_cell(t) == n_dofs, terms) ||
        throw(ArgumentError("FEM terms with equal `fem_element_loop` share one element " *
                            "loop, so they must all have $n_qp quadrature points and " *
                            "$n_dofs DOFs per element"))
    qp_buffer = Matrix{eltype(fem_qp_buffer(driver))}(
        undef, maximum(length, cols; init = 0), n_qp)
    return FEMGroup(terms, cols, entries, qp_buffer, zeros(T, n_dofs))
end

"""
	build_multilinear_terms_cache(model, parametrisation[, skip_bits]) → MultilinearTermsCache

Precompute all factorisation data for every monomial and term.
Valid as long as the multiindex set is unchanged.

When `skip_bits[l]` is `true` the cache entry for monomial `l` is left empty.
This is safe for monomials that will never be replayed (linear monomials, conjugate
secondaries): those entries are guarded by the same `skip_bits` check in the solve loop.

FEM terms are grouped by `fem_element_loop`. Each group runs on the element loop of its
first term, which scatters, weights and assembles for all of them, so they must agree on
the quadrature points and the DOFs per element; an `ArgumentError` is thrown otherwise.
"""
function build_multilinear_terms_cache(
        model::NthOrderModel{ORD}, parametrisation::Parametrisation{ORD, NVAR}) where {
        ORD, NVAR}
    L = length(parametrisation.poly.multiindex_set)
    build_multilinear_terms_cache(model, parametrisation, falses(L))
end

function build_multilinear_terms_cache(
        model::NthOrderModel{ORD}, parametrisation::Parametrisation{ORD, NVAR},
        skip_bits::BitVector) where {ORD, NVAR}
    mset = parametrisation.poly.multiindex_set
    L = length(mset)
    terms = model.nonlinear_terms
    ROM = NVAR - parametrisation.external_system_size
    is_fem = [t isa FEMMultilinearMap for t in terms]
    fem_terms = Tuple(t for t in terms if t isa FEMMultilinearMap)
    members = _fem_groups(fem_terms)
    group_terms = [Tuple(fem_terms[i] for i in m) for m in members]

    splits = Vector{Vector{Vector{CachedSplit}}}(undef, L)
    fem_cols = [Vector{Vector{Tuple{Int, Int}}}(undef, L) for _ in members]
    fem_entries = [Vector{typeof(_fem_monomial(gt, map(_ -> CachedSplit[], gt))[2])}(undef, L)
                   for gt in group_terms]
    for l in 1:L
        exp = mset.exponents[l]
        candidates = skip_bits[l] ? nothing :
                     indices_in_box_with_bounded_degree(mset, exp, 1, sum(exp))
        splits_l = [skip_bits[l] ? CachedSplit[] :
                    _cached_splits(t, mset, exp, candidates, ROM)
                    for t in terms]
        fem_splits = splits_l[is_fem]
        for (g, m) in enumerate(members)
            fem_cols[g][l], fem_entries[g][l] = _fem_monomial(group_terms[g], fem_splits[m])
        end
        splits[l] = [fem ? CachedSplit[] : s for (fem, s) in zip(is_fem, splits_l)]
    end

    T = eltype(parametrisation.poly)
    FOM = size(parametrisation)
    # The external argument for each external variable: the unit vector eⱼ, or the column
    # Q[:, j] of the change of basis when the external system was re-based.  Built once per
    # solve, so a re-based system costs nothing in the inner loop and no term needs to know
    # about the change of coordinates.
    external_arguments = external_argument_vectors(
        model.external_system, parametrisation.external_system_size)

    fem_groups = ntuple(
        g -> _fem_group(group_terms[g], fem_cols[g], fem_entries[g], T), length(members))
    isempty(fem_terms) ||
        @info "Single-pass FEM assembly" n_fem_terms=length(fem_terms) element_loops=length(fem_groups)

    return MultilinearTermsCache(splits, zeros(T, FOM), zeros(T, FOM), external_arguments,
        fem_groups)
end
