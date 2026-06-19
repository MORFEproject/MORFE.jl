"""
    polarize_monomial(term, F_groups, mi=nothing)

Polarize a single monomial `term` with respect to `F_groups` (i.e. `groups[1:end-1]`).

Every factor `v^p`, where `v` is the `i`-th variable of group `g`, is replaced by
the product of `p` distinct "slot copies" of `v`: `v_1 * v_2 * ... * v_p`. Slots
are consumed group-by-group, in the order the factors appear in `term`.

Returns `(polarized_expr, slotvars)`:
- `polarized_expr :: Num` — the polarized monomial (coefficient included).
- `slotvars :: NTuple{NG, Vector{Vector{Num}}}` — `slotvars[g][s]` is the
  length-`N` vector of slot-`s` copies of the variables in `F_groups[g]`
  (this is the "`x_s`" vector you'd pass into `eval`).

If `mi` is not given, it is computed via `multidegree_of_monomial`.
"""
function polarize_monomial(term, F_groups::NTuple{NG, Vector{Num}},
        mi::Union{Nothing, NTuple{NG, Int}} = nothing) where {NG}
    # TODO Check wether term is a monome
    N = length(F_groups[1])
    for g in 1:NG
        @assert length(F_groups[g])==N "All groups must have the same number of variables."
    end

    e = Symbolics.value(term)
    coeff, info = _factor_info(e, F_groups)

    if mi === nothing
        mi = multidegree_of_monomial(term, F_groups)
    end

    # one new symbol "<original name>_<slot>" per (group, slot, variable)
    slotvars = ntuple(NG) do g
        [Num[Symbolics.variable(Symbol(Symbolics.tosymbol(F_groups[g][i]), "_", s))
             for i in 1:N]
         for s in 1:mi[g]]
    end

    # consume slots group-by-group, in the order factors appear
    counters = ones(Int, NG)
    polarized = Num(coeff)
    for (g, i, p) in info
        for _ in 1:p
            s = counters[g]
            polarized *= slotvars[g][s][i]
            counters[g] += 1
        end
    end

    @assert Tuple(counters .- 1)==mi "Monomial degree $(Tuple(counters .- 1)) does not match multidegree $mi"

    return polarized, slotvars
end

"""
    _factor_info(e, F_groups)

Decompose monomial `e` into a scalar coefficient and a list of
`(group_index, var_index_in_group, exponent)` tuples, one per symbolic factor.
For polarization.
"""
function _factor_info(e, F_groups::NTuple{NG, Vector{Num}}) where {NG}
    if Symbolics.iscall(e) && Symbolics.operation(e) == (*)
        factors = Symbolics.arguments(e)
    else
        factors = (e,)
    end

    coeff = 1
    info = Tuple{Int, Int, Int}[]

    for f in factors
        if Symbolics.value(f) isa Number
            coeff *= Symbolics.value(f)
        elseif Symbolics.iscall(f) && Symbolics.operation(f) == (^)
            base, expo = Symbolics.arguments(f)
            g, i = _findgroup_index(base, F_groups)
            push!(info, (g, i, Int(Symbolics.value(expo))))
        else
            g, i = _findgroup_index(f, F_groups)
            push!(info, (g, i, 1))
        end
    end

    return coeff, info
end

"""
    _findgroup_index(sym, groups)

Return `(group_index, var_index)` such that `groups[group_index][var_index]`
equals `sym`.
For polarization.
"""
function _findgroup_index(sym, groups::NTuple{NG, Vector{Num}}) where {NG}
    for (g, vars) in enumerate(groups)
        for (i, v) in enumerate(vars)
            if isequal(sym, Symbolics.value(v))
                return g, i
            end
        end
    end
    error("Symbol $sym not found in any of the provided groups.")
end

"""
    polarize(F_by_multiindex, F_groups, N)

Returns dictionary F_by_multiindex but polarized.

`F_groups` is the tuple of variable groups used for polarization — either
`groups[1:end-1]` (pure state case) or `(groups[1:end-1]..., ext_var)` (with
external forcing).  The multiindex length must equal `length(F_groups)`.
"""
function polarize(F_by_multiindex::Dict{NTuple{ORD, Int}, Vector{Num}},
        F_groups::NTuple{ORD, Vector{Num}},
        N::Int) where {ORD}
    F_by_multiindex_polarized = Dict{NTuple{ORD, Int}, Vector{Num}}()
    dict_slotvars = Dict{NTuple{ORD, Int}, NTuple{ORD, Vector{Vector{Num}}}}()
    for key in keys(F_by_multiindex)
        tmp = zeros(Num, N)
        slotvars_ref = nothing
        for eq_row in 1:N
            key_monomials = seperate_into_monomials(F_by_multiindex[key][eq_row])
            for index in eachindex(key_monomials)
                if !isequal(key_monomials[index], Num(0))
                    pol_mon,
                    slotvars = polarize_monomial(
                        key_monomials[index], F_groups, key)
                    if isnothing(slotvars_ref)
                        slotvars_ref = slotvars
                    elseif !isequal(slotvars, slotvars_ref)
                        error("Inconsistent slotvars detected")
                    end
                    tmp[eq_row] += pol_mon
                end
            end
        end # eq_row
        F_by_multiindex_polarized[key] = tmp
        dict_slotvars[key] = slotvars_ref
    end
    return F_by_multiindex_polarized, dict_slotvars
end