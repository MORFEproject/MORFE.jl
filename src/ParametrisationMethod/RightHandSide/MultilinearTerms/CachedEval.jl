# -----------------------------------------------------------------------
# Closure terms, split by split
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

# -----------------------------------------------------------------------
# Single-pass assembly of all FEM terms
# -----------------------------------------------------------------------

"""
	_replay_fem!(result, W, group::FEMGroup, external_arguments, l)

Add the right-hand side of the FEM terms of `group` at monomial `l` to `result`, in one pass
over their element loop, driven by the group's first term.

In each element, every distinct coefficient that the factorisations read (`group.cols[l]`)
is scattered to the quadrature points once. At each quadrature point, one call to
[`accumulate_terms_qp!`](@ref) adds every factorisation of every term, external ones
included: by default, the fluxes of the terms whose flux has the type of the driver's are
summed, and the others go through `accumulate_qp!`. The flux sum is tested against the
shape functions once, with the driver's `contract_qp!`, and the element vector is assembled
once.

`group.terms` and `group.entries[l]` are concretely typed tuples aligned term by term, so
the quadrature loop dispatches nothing at run time.
"""
function _replay_fem!(result, W, group::FEMGroup, external_arguments, l)
    cols = group.cols[l]
    isempty(cols) && return nothing
    (; terms, qp_buffer, Fe) = group
    entries = group.entries[l]
    driver = first(terms)
    flux0 = fem_flux_zero(driver)
    n_qp = fem_n_qp(driver)

    for element in fem_elements(driver)
        fem_reinit!(element, driver)
        for (i, (order, col)) in enumerate(cols)
            scatter_qp!(@view(qp_buffer[i, 1:n_qp]), @view(W[:, order, col]), element, driver)
        end
        fill!(Fe, zero(eltype(Fe)))
        for q in 1:n_qp
            dΩ = fem_getdetJdV(element, q, driver)
            coefficient_values = @view qp_buffer[1:length(cols), q]
            flux = accumulate_terms_qp!(flux0, Fe, coefficient_values, external_arguments,
                entries, terms, element, q, dΩ)
            flux0 === nothing || contract_qp!(Fe, flux, element, q, dΩ, driver)
        end
        assemble_element!(result, Fe, element, driver)
    end
    return nothing
end

# One pass per element loop, unrolled at compile time: each group has its own types.
_replay_fem_groups!(result, W, ::Tuple{}, external_arguments, l) = nothing

@inline function _replay_fem_groups!(result, W, groups::Tuple, external_arguments, l)
    _replay_fem!(result, W, first(groups), external_arguments, l)
    return _replay_fem_groups!(result, W, Base.tail(groups), external_arguments, l)
end

# The default: one term at a time, unrolled at compile time. Each term's branch and methods
# resolve statically.
@inline accumulate_terms_qp!(
    flux, Fe, coefficient_values, external_arguments, ::Tuple{}, ::Tuple{},
    element, q, dΩ) = flux

@inline function accumulate_terms_qp!(flux, Fe, coefficient_values, external_arguments,
        entries::Tuple, terms::Tuple, element, q, dΩ)
    term = first(terms)
    if flux !== nothing && fem_flux_zero(term) isa typeof(flux)
        for e in first(entries)
            qp_args = _qp_args(coefficient_values, external_arguments, e)
            flux += e.multiplier * flux_qp(qp_args, element, q, term)
        end
    else
        for e in first(entries)
            qp_args = _qp_args(coefficient_values, external_arguments, e)
            accumulate_qp!(Fe, qp_args, e.multiplier, element, q, dΩ, term)
        end
    end
    return accumulate_terms_qp!(flux, Fe, coefficient_values, external_arguments,
        Base.tail(entries), Base.tail(terms), element, q, dΩ)
end

# The arguments of one factorisation: its scattered internal factors, then its external
# arguments. The indices were built against the sizes of both collections.
@inline function _qp_args(coefficient_values, external_arguments, e::FEMEntry)
    return (map(k -> @inbounds(coefficient_values[k]), e.factors)...,
        map(k -> @inbounds(external_arguments[k]), e.externals)...)
end

# -----------------------------------------------------------------------
# Public API — cached
# -----------------------------------------------------------------------

"""
	compute_multilinear_terms!(result, model, exp_index, parametrisation, cache) → nothing

Zero `result`, then add the nonlinear right-hand side of monomial `exp_index`: the FEM terms
in a single pass over each element loop ([`_replay_fem!`](@ref)), then each closure term
split by split. Uses the buffers of `cache`, so no heap allocation occurs during the inner
solve loop.
"""
function compute_multilinear_terms!(
        result::AbstractVector,
        model::NthOrderModel{ORD}, exp_index::Int,
        parametrisation::Parametrisation{ORD, NVAR},
        cache::MultilinearTermsCache) where {ORD, NVAR}
    fill!(result, zero(eltype(result)))
    W = parametrisation.poly.coefficients
    _replay_fem_groups!(result, W, cache.fem_groups, cache.external_arguments, exp_index)
    # Indexed, and only for terms with splits: iterating the abstractly typed term tuple
    # allocates, even when every split list is empty, as for the FEM terms.
    splits = cache.splits[exp_index]
    for t_idx in eachindex(splits)
        isempty(splits[t_idx]) && continue
        t = model.nonlinear_terms[t_idx]
        for split in splits[t_idx]
            _replay_split!(result, cache.scratch_buffer, cache.temp_buffer, t, W, split,
                t.deg - t.multiplicity_external, cache.external_arguments)
        end
    end
    return nothing
end

"""
	compute_multilinear_terms(model, exp_index, parametrisation, cache) → Vector

Allocating variant of [`compute_multilinear_terms!`](@ref). `exp_index` is the 1-based
index into `parametrisation`'s multiindex set.
"""
function compute_multilinear_terms(model::NthOrderModel{ORD}, exp_index::Int,
        parametrisation::Parametrisation{ORD, NVAR},
        cache::MultilinearTermsCache) where {ORD, NVAR}
    result = zeros(eltype(parametrisation.poly), size(parametrisation))
    compute_multilinear_terms!(result, model, exp_index, parametrisation, cache)
    return result
end
