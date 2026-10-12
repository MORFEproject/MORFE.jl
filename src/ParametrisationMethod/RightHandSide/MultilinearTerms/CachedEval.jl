# -----------------------------------------------------------------------
# Public API — cached
# -----------------------------------------------------------------------

"""
	_replay_split!(result, scratch, temp, t, W, split, deg, external_arguments)

Replay one `CachedSplit` into `result` using precomputed factorisation bookkeeping.

- `me = 0` (`args_ext_indices` empty): accumulates directly into `result`.
- `me > 0`: accumulates into `temp`, then `axpy!(ext_count, temp, result)`.

Dispatches asymmetric/symmetric accumulation via `split.is_asymmetric`.
"""
function _replay_split!(result, scratch, temp, t, W, split, deg, external_arguments)
    if isempty(split.args_ext_indices)
        accum = result
        args_ext = ()
    else
        fill!(temp, 0)
        accum = temp
        args_ext = ntuple(i -> external_arguments[split.args_ext_indices[i]],
            length(split.args_ext_indices))
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
	_replay_fem_split!(result, t, W, fem_split, Fe, temp, external_arguments)

Replay one `FEMCachedSplit{DEG, ME}` using the FEM-batched element loop.

For each element: calls `fem_reinit!` once, scatters each unique `(order, col)` W
column to qp-level field values via `scatter_qp!`, accumulates all qp contributions
via `accumulate_qp!`, and assembles the element residual into `result` via
`assemble_element!`.  The qp gradient buffer `∇W_qp` is obtained from the term
via `fem_qp_buffer(t)` and is owned by the term, not allocated here.

`accumulate_qp!` receives `DEG` internal arguments followed by `ME` external ones, so the
tuple has length `t.deg` — matching what `evaluate_term!`'s direct FEM overload
(`_eval_fem_term_direct!`) passes.  Before this, the external slots were omitted here and
`ext_count` was applied as a bare scalar multiplier, so an `me > 0` FEM term could not
tell one external direction from another and disagreed with the direct path.  The external
arguments are hoisted out of the element loop, exactly as in `_replay_split!`.
"""
function _replay_fem_split!(
        result, t::FEMMultilinearMap, W, fem_split::FEMCachedSplit{DEG, ME},
        Fe, temp, external_arguments) where {DEG, ME}
    ∇W_qp = fem_qp_buffer(t)   # Matrix{QP_TYPE}(max_unique, n_qp) — owned by the term

    if ME == 0
        accum = result
        args_ext = ()
    else
        # External-forcing case: accumulate into `temp`, then axpy! into `result` — exactly
        # as `_replay_split!` does for its me>0 branch.
        #
        # `temp` must be a *separate* FOM-length buffer, never `Fe`: `Fe` is the
        # element-local residual that `assemble_element!` scatters *from*, so using it as
        # the accumulator too makes the call `Fe .+= Fe` and doubles every contribution.
        fill!(temp, zero(eltype(temp)))
        accum = temp   # will be axpy!-ed into result after the loop
        args_ext = ntuple(i -> external_arguments[fem_split.args_ext_indices[i]], Val(ME))
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
                @inbounds ∇W_args = (
                    ntuple(k -> ∇W_qp[fem_entry.local_factor_indices[k], q], Val(DEG))...,
                    args_ext...)
                accumulate_qp!(
                    @view(Fe[1:n_dofs]), ∇W_args, fem_entry.multiplier, element, q, dΩ, t)
            end
        end

        # 3. Scatter element residual to global accumulator.
        assemble_element!(accum, @view(Fe[1:n_dofs]), element, t)
    end

    ME == 0 || axpy!(fem_split.ext_count, temp, result)
end

# -----------------------------------------------------------------------
# O4 — combined element loop across all me=0 FEM terms
# -----------------------------------------------------------------------

"""
	_replay_all_fem_splits!(result, model, W, global_split, global_∇W_qp, global_Fe_buffers)

Execute the O4 combined element loop: traverse the mesh once, scatter all globally-unique
W columns, then accumulate contributions from ALL me=0 FEM terms at each quadrature point.

`fem_reinit!` and `scatter_qp!` are called at most once per unique `(element, W-column)`
pair.

`model.nonlinear_terms` is an `NTuple` of abstractly typed terms, so this method is only a
function barrier. It gathers the participating terms into a tuple, whose concrete type the
method below specialises on, once per monomial. Inside, the element loop, the loop over the
terms and each term's quadrature loop ([`_accumulate_term_element!`](@ref)) dispatch at
compile time; reading a term out of `model.nonlinear_terms` inside the loop would dispatch
at run time and allocate, per element or per factorisation and quadrature point.
"""
function _replay_all_fem_splits!(result, model, W,
        global_split::FEMGlobalSplit,
        global_∇W_qp, global_Fe_buffers)
    idx = Tuple(global_split.participating_term_indices)
    terms = map(i -> model.nonlinear_terms[i], idx)
    _replay_all_fem_splits!(result, model.nonlinear_terms[global_split.driver_term_idx],
        terms, idx, W, global_split, global_∇W_qp, global_Fe_buffers)
end

function _replay_all_fem_splits!(result, driver::FEMMultilinearMap, terms::Tuple,
        idx::Tuple, W, global_split::FEMGlobalSplit, global_∇W_qp, global_Fe_buffers)
    n_qp = fem_n_qp(driver)
    n_uniq = length(global_split.global_unique_cols)

    for element in fem_elements(driver)
        fem_reinit!(element, driver)

        for i in 1:n_uniq
            (order, col) = global_split.global_unique_cols[i]
            scatter_qp!(@view(global_∇W_qp[i, 1:n_qp]), @view(W[:, order, col]), element, driver)
        end

        _accumulate_terms_element!(result, global_Fe_buffers, global_∇W_qp,
            global_split.entries_by_deg, idx, element, n_qp, driver, terms)
    end
end

# Recursion over the participating terms, unrolled at compile time.
@inline _accumulate_terms_element!(_, _, _, _, ::Tuple{}, _, _, _, ::Tuple{}) = nothing

@inline function _accumulate_terms_element!(result, Fe_buffers, ∇W_qp, entries_by_deg,
        idx::Tuple, element, n_qp, driver, terms::Tuple)
    _accumulate_term_element!(result, Fe_buffers[first(idx)], ∇W_qp, entries_by_deg,
        first(idx), element, n_qp, driver, first(terms))
    _accumulate_terms_element!(result, Fe_buffers, ∇W_qp, entries_by_deg,
        Base.tail(idx), element, n_qp, driver, Base.tail(terms))
end

"""
	_accumulate_term_element!(result, Fe, ∇W_qp, entries_by_deg, t_idx, element, n_qp, driver, t)

Integrate term `t` (index `t_idx`) over one element and add it to `result`: every
factorisation of the term at every quadrature point, with the gradients `scatter_qp!` left
in `∇W_qp`. Specialised on the concrete type of `t`, which is what keeps the quadrature
loop free of run-time dispatch. The integration weights come from the `driver` term, which
owns the element loop.
"""
function _accumulate_term_element!(result, Fe, ∇W_qp, entries_by_deg, t_idx, element,
        n_qp, driver, t)
    fill!(Fe, zero(eltype(Fe)))
    for q in 1:n_qp
        dΩ = fem_getdetJdV(element, q, driver)
        _accumulate_term_entries!(Fe, ∇W_qp, entries_by_deg, t_idx, element, q, dΩ, t)
    end
    assemble_element!(result, Fe, element, t)
end

"""
	_accumulate_term_entries!(Fe, ∇W_qp, entries_by_deg, t_idx, element, q, dΩ, t)

Add, at quadrature point `q`, every factorisation entry of term `t_idx` to `Fe`. Recurses
over the degree-grouped entry tuple, so `Val(DEG)` in `ntuple` and the `accumulate_qp!`
dispatch are resolved at compile time; groups of another degree than the term's are
skipped. The base case (empty tuple) is a no-op.
"""
@inline _accumulate_term_entries!(_, _, ::Tuple{}, _, _, _, _, _) = nothing

@inline function _accumulate_term_entries!(Fe, ∇W_qp,
        entries_by_deg::Tuple{Vector{FEMGlobalEntry{DEG}}, Vararg}, t_idx, element, q, dΩ,
        t) where {DEG}
    if t.deg == DEG
        for gentry in first(entries_by_deg)
            gentry.term_idx == t_idx || continue
            ∇W_args = ntuple(k -> ∇W_qp[gentry.local_factor_indices[k], q], Val(DEG))
            accumulate_qp!(Fe, ∇W_args, gentry.multiplier, element, q, dΩ, t)
        end
    end
    _accumulate_term_entries!(
        Fe, ∇W_qp, Base.tail(entries_by_deg), t_idx, element, q, dΩ, t)
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
    _replay_all_fem_splits!(
        result, model, W, gs, cache.global_∇W_qp, cache.global_Fe_buffers)
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
            t, W, split, deg, cache.external_arguments)
    end
end

function _replay_term!(result, t::FEMMultilinearMap, W, exp_index, t_idx,
        cache::MultilinearTermsCache)
    # me=0 splits are handled by _replay_global_fem! (O4 combined loop).
    # Only the me>0 fallback splits need processing here.
    for fem_split in cache.fem_splits[exp_index][t_idx]
        isempty(fem_split.args_ext_indices) && continue
        _replay_fem_split!(result, t, W, fem_split, cache.fem_Fe, cache.temp_buffer,
            cache.external_arguments)
    end
end

"""
	compute_multilinear_terms(model, exp_index, parametrisation, cache) → Vector

Cached variant of `compute_multilinear_terms`: replays precomputed factorisation
data instead of calling the factorisation routines.

`exp_index` is the 1-based index into `parametrisation`'s multiindex set.
"""
function compute_multilinear_terms(model::NthOrderModel{ORD}, exp_index::Int,
        parametrisation::Parametrisation{ORD, NVAR},
        cache::MultilinearTermsCache) where {ORD, NVAR}
    W = parametrisation.poly.coefficients
    FOM = size(parametrisation)
    T = eltype(parametrisation.poly)
    deg_max = sum(parametrisation.poly.multiindex_set.exponents[exp_index])

    result = zeros(T, FOM)
    scratch = similar(result)
    temp = similar(result)

    # The cache already holds these (unit vectors, or Q's columns after a re-basing);
    # rebuilding them here would both re-allocate and silently ignore the change of basis.
    external_arguments = cache.external_arguments

    for (t_idx, t) in enumerate(model.nonlinear_terms)
        t.deg > deg_max && continue
        deg = t.deg - t.multiplicity_external
        for split in cache.splits[exp_index][t_idx]
            _replay_split!(result, scratch, temp, t, W, split, deg, external_arguments)
        end
    end
    return result
end

"""
	compute_multilinear_terms!(result, model, exp_index, parametrisation, cache) → nothing

In-place variant: zeros `result` then accumulates all nonlinear contributions into it.
Uses `cache.scratch_buffer`, `cache.temp_buffer`, and `cache.external_arguments` so that no
heap allocation occurs during the inner solve loop.
"""
function compute_multilinear_terms!(
        result::AbstractVector,
        model::NthOrderModel{ORD}, exp_index::Int,
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
