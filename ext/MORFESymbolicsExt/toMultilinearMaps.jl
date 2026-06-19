"""
    _monomial_to_MultilinearMap

Defines a MultilinearMap for one monomial.

# Keyword arguments
- ` has_ext::Bool = false`: boolean thet tells the MultilinearMap wether it is an external system or not

"""
function _monomial_to_MultilinearMap(
        polarized_monomial::Vector{Num},
        polarized_variables::NTuple{ORD, Vector{Vector{Num}}},
        multiindex::NTuple{ORD, Int};
        has_ext::Bool = false) where {ORD}
    vars = reduce(vcat, polarized_variables)
    _f, _ = build_function(polarized_monomial, vars..., expression = Val(false))
    nvars = length(vars)
    function f!(res, vars...)
        @assert length(vars) == nvars
        res .+= _f(vars...)
        return nothing
    end
    deg = sum(multiindex)
    if has_ext == false
        return MultilinearMap{ORD, typeof(f!)}(
            f!, multiindex, 0, deg, false)  # Attention: Uses constructor without checking number of arguments of f!.
    else
        return MultilinearMap{ORD-1, typeof(f!)}(
            f!, multiindex[1:(end - 1)], multiindex[end], deg, false)  # Attention: Uses constructor without checking number of arguments of f!.
    end
end

"""
    all_monomials_to_MultilinearMaps(F_by_multiindex_polarized, dict_pol_vars)

Defines and collects MultilinearMaps in a Tuple for every monomal in `F_by_multiindex_polarized`.

# Arguments
- `F_by_multiindex_polarized`: Dictionary that maps multiindices to a polarized monomial vector.
- `dict_pol_vars`: Dictionary that maps multiindices to the autmatically generated variables that appear in the polarized monomial.
# Keyword arguments
- ` has_ext::Bool = false`: boolean thet tells the MultilinearMap wether it is an external system or not

"""
function all_monomials_to_MultilinearMaps(
        F_by_multiindex_polarized::Dict{NTuple{ORD, Int}, Vector{Num}},
        dict_pol_vars::Dict{NTuple{ORD, Int}, NTuple{ORD, Vector{Vector{Num}}}};
        has_ext::Bool = false) where {ORD}
    key_list = collect(keys(F_by_multiindex_polarized))
    N_NL = length(key_list)
    nonlinear_terms = Vector{MultilinearMap}(undef, N_NL)
    for i in 1:N_NL
        nonlinear_terms[i] = _monomial_to_MultilinearMap(
            F_by_multiindex_polarized[key_list[i]], dict_pol_vars[key_list[i]], key_list[i], has_ext = has_ext)
    end
    return Tuple(nonlinear_terms)
end
