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
function polarize_monomial(term, F_groups::NTuple{NG, Vector{<:MyNum}},
        mi::Union{Nothing, NTuple{NG, Int}} = nothing) where {NG}
    N = length(F_groups[1])

    # for Complex{Num}, extract the Num part for factor analysis
    term_num = term isa Complex{Num} ? (iszero(real(term)) ? imag(term) : real(term)) : term
    e = Symbolics.value(term_num)
    coeff, info = _factor_info(e, F_groups)

    # recover the full complex coefficient
    if term isa Complex{Num}
        re_coeff = Symbolics.value(substitute(real(term), Dict(v => Num(1)
        for v in reduce(vcat, F_groups))))
        im_coeff = Symbolics.value(substitute(imag(term), Dict(v => Num(1)
        for v in reduce(vcat, F_groups))))
        coeff = re_coeff + im_coeff * im
    end

    if mi === nothing
        mi = multidegree_of_monomial(term, F_groups)
    end

    slotvars = ntuple(NG) do g
        [Num[Symbolics.variable(Symbol(Symbolics.tosymbol(F_groups[g][i]), "_", s))
             for i in 1:length(F_groups[g])]
         for s in 1:mi[g]]
    end

    counters = ones(Int, NG)
    polarized = _to_MyNum(coeff)
    for (g, i, p) in info
        for _ in 1:p
            s = counters[g]
            polarized *= slotvars[g][s][i]
            counters[g] += 1
        end
    end

    @assert Tuple(counters .- 1) == mi "Monomial degree $(Tuple(counters .- 1)) does not match multidegree $mi"
    return polarized, slotvars
end

"""
    _factor_info(e, F_groups)

Decompose monomial `e` into a scalar coefficient and a list of
`(group_index, var_index_in_group, exponent)` tuples, one per symbolic factor.
For polarization.
"""
function _factor_info(e, F_groups::NTuple{NG, Vector{<:MyNum}}) where {NG}
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
function _findgroup_index(sym, groups::NTuple{NG, Vector{<:MyNum}}) where {NG}
    for (g, vars) in enumerate(groups)
        for (i, v) in enumerate(vars)
            if isequal(sym, Symbolics.value(v))
                return g, i
            end
        end
    end
    error("Symbol $sym not found in any of the provided groups.")
end

# function _is_zero(m)
#     isequal(m, Num(0))
# end
# function _is_zero(m::Complex{Num})
#     isequal(m, Complex{Num}(0))
# end

"""
    polarize(F_by_multiindex, F_groups, N)

Returns dictionary F_by_multiindex but polarized.

`F_groups` is the tuple of variable groups used for polarization — either
`groups[1:end-1]` (pure state case) or `(groups[1:end-1]..., ext_var)` (with
external forcing).  The multiindex length must equal `length(F_groups)`.
"""
function polarize(F_by_multiindex::Dict{NTuple{ORD, Int}, Vector{MyNum}},
        F_groups::NTuple{ORD, Vector{<:MyNum}},
        N::Int) where {ORD}
    F_by_multiindex_polarized = Dict{NTuple{ORD, Int}, Vector{MyNum}}()
    dict_slotvars = Dict{NTuple{ORD, Int}, NTuple{ORD, Vector{Vector{Num}}}}()
    for key in keys(F_by_multiindex)
        tmp = Vector{MyNum}(undef, N)
        fill!(tmp, Num(0))
        slotvars_ref = nothing
        for eq_row in 1:N
            key_monomials = seperate_into_monomials(F_by_multiindex[key][eq_row])
            for index in eachindex(key_monomials)
                if !_is_zero(key_monomials[index])
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
        end
        F_by_multiindex_polarized[key] = tmp
        dict_slotvars[key] = slotvars_ref
    end
    return F_by_multiindex_polarized, dict_slotvars
end