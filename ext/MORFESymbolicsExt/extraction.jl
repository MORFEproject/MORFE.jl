"""
    extract_linear_matrices

Assemble linear_terms matrices for NDOrderModel.
Inputs are a ODE of the variables defined in `groups` that is a NTuple, where every part of the Tuple consists of the state variables or the derivatives. 
e.g: ([z1, z2, z3], [dz1, dz2, dz3], ...)
Returns tuple of matrices with B[i] is matric w.r.t. groups[i] so the ith derivative.  
    
"""
function extract_linear_matrices(
        exprs::Vector{Num}, groups::NTuple{ORDP1, Vector{Num}}, ext_var::Vector{Num} = Num[]) where {ORDP1}
    n = length(groups[1])
    all_syms = vcat(groups..., ext_var)

    B = ntuple(ORDP1) do k
        J = Symbolics.jacobian(exprs, groups[k])          # n x n, may still contain symbols
        J0 = Symbolics.substitute.(J, (Dict(s => 0 for s in all_syms),))
        Float64.(Symbolics.value.(J0)) * (-1)
    end
    return B   # B[1]=B0, B[2]=B1, ..., B[ORDP1]=B_ORD
end

"""
    nonlinear_remainder

returns the nonlinear part of the equation as the same size as `exprs`.
Also checks that the nonlinear part is not allowed to depend on the highest derivative variables.
"""
function nonlinear_remainder(
        exprs::Vector{Num}, groups::NTuple{ORDP1, Vector{Num}}, B) where {ORDP1}
    n = length(exprs)
    nl = Vector{Num}(undef, n)
    for i in 1:n
        lin = sum(B[k][i, j] * groups[k][j] for k in 1:ORDP1, j in 1:n)
        nl[i] = Symbolics.expand(exprs[i] + lin)
    end

    # F = nl, given exprs = B_ORD x^(ORD) + ... + B0 x - F
    F = nl

    # sanity check: F must not depend on the highest-derivative group
    highest = groups[end]
    for i in 1:n, s in highest

        @assert isequal(Symbolics.derivative(F[i], s), 0) "F[$i] depends on highest derivative $s — not representable in NDOrderModel"
    end

    return F
end

"""
    seperate_into_monomials

Seperates nonlinear part in to the seperate mononmials. Works on a single not vector expression!
"""
function seperate_into_monomials(expr)
    e = Symbolics.value(expand(expr))

    if Symbolics.iscall(e) && Symbolics.operation(e) == (+)
        collect(Symbolics.arguments(e))
    else
        [e]
    end
end

"""
    degree_of_monomial

Computes the degree of a monomial. Works on a single not vector expression!
"""
function degree_of_monomial(term)
    e = Symbolics.value(term)

    if Symbolics.iscall(e) && Symbolics.operation(e) == (*)
        factors = Symbolics.arguments(e)
    else
        factors = (e,)
    end

    deg = 0
    for f in factors
        if Symbolics.value(f) isa Number
            continue
        elseif Symbolics.iscall(f) && Symbolics.operation(f) == (^)
            _, expo = Symbolics.arguments(f)
            deg += Int(Symbolics.value(expo))
        else
            deg += 1
        end
    end

    return deg
end

"""
    multidegree_of_monomial(term, groups)

Computes the degree of a monomial but separated in to the derivative orders defined in `groups`.
"""
function multidegree_of_monomial(term, groups::NTuple{NG, Vector{Num}}) where {NG}
    e = Symbolics.value(term)

    if Symbolics.iscall(e) && Symbolics.operation(e) == (*)
        factors = Symbolics.arguments(e)
    else
        factors = (e,)
    end

    multideg = zeros(Int, NG)

    for f in factors
        if Symbolics.value(f) isa Number
            continue
        elseif Symbolics.iscall(f) && Symbolics.operation(f) == (^)
            base, expo = Symbolics.arguments(f)
            g = _findgroup(base, groups)
            multideg[g] += Int(Symbolics.value(expo))
        else
            g = _findgroup(f, groups)
            multideg[g] += 1
        end
    end

    return Tuple(multideg)
end

"""
    _findgroup(sym, groups)

Helper-function that returns in which group the symbol `sym` is.
"""
function _findgroup(sym, groups::NTuple{NG, Vector{Num}}) where {NG}
    for (g, vars) in enumerate(groups)
        if any(isequal(sym, Symbolics.value(v)) for v in vars)
            return g
        end
    end
    error("Symbol $sym not found in any of the provided groups — " *
          "check that `groups` includes all variables that may appear in F.")
end

"""
    extract_nonlinear_monomials(exprs, groups, linear_terms)

Calculates the nonlinear part of `exprs` by substracting `linear_terms`.
Seperates the nonlinear remander into monomials and saves them in the Vector `monomials` 
and generates additional a Vector `multideg_monomials` that contains the multiindex_degree calculated by `multidegree_monomials`. 
"""
function extract_nonlinear_monomials(
        exprs::Vector{Num}, groups::NTuple{ORDP1, Vector{Num}}, linear_terms) where {ORDP1}

    # Extract nonlinear linear_terms
    F = nonlinear_remainder(exprs, groups, linear_terms)
    N = length(F)
    @assert length(exprs)==N "length of exprs and F must match. Error in nonlinear_remainder"

    # Extract monomials and calculate degree
    F_groups = groups[1:(end - 1)]
    monomials = Vector{Any}(undef, N)
    deg_monomials = Vector{Vector{Int}}(undef, N)
    multideg_monomials = Vector{Vector{NTuple{ORDP1 - 1, Int}}}(undef, N)
    for i in eachindex(F)
        raw = seperate_into_monomials(F[i])
        # filter out zero and purely constant terms (degree 0)
        filtered = filter(raw) do m
            d = degree_of_monomial(m)
            d > 0
        end
        monomials[i] = filtered
        deg_monomials[i] = Vector{Int}(undef, length(monomials[i]))
        multideg_monomials[i] = Vector{NTuple{ORDP1 - 1, Int}}(undef, length(monomials[i]))
        for j in eachindex(monomials[i])
            deg_monomials[i][j] = degree_of_monomial(monomials[i][j])
            multideg_monomials[i][j] = multidegree_of_monomial(monomials[i][j], F_groups)
        end
    end
    return N, monomials, deg_monomials, multideg_monomials
end
"""
    extract_nonlinear_monomials(exprs, F_groups_ext, linear_terms, groups)

Variant for NDOrderModel **with external variables**.

`F_groups_ext` = `(groups[1:end-1]..., ext_var)` — the groups used to compute
multidegrees of nonlinear monomials.  `groups` is the full derivative-group
tuple and is used only to subtract the linear part (via `nonlinear_remainder`).

The returned `multideg_monomials` has tuples of length `length(F_groups_ext)`,
i.e. one entry per state-derivative group (excl. highest) plus one entry for
the external group.
"""
function extract_nonlinear_monomials(
        exprs::Vector{Num},
        F_groups_ext::NTuple{NG_EXT, Vector{Num}},
        linear_terms,
        groups::NTuple{ORDP1, Vector{Num}}) where {NG_EXT, ORDP1}

    # subtract linear part using the full groups (same logic as the base method)
    F = nonlinear_remainder(exprs, groups, linear_terms)
    N = length(F)
    @assert length(exprs) == N

    monomials = Vector{Any}(undef, N)
    deg_monomials = Vector{Vector{Int}}(undef, N)
    multideg_monomials = Vector{Vector{NTuple{NG_EXT, Int}}}(undef, N)

    for i in eachindex(F)
        monomials[i] = seperate_into_monomials(F[i])
        deg_monomials[i] = Vector{Int}(undef, length(monomials[i]))
        multideg_monomials[i] = Vector{NTuple{NG_EXT, Int}}(undef, length(monomials[i]))
        for j in eachindex(monomials[i])
            deg_monomials[i][j] = degree_of_monomial(monomials[i][j])
            multideg_monomials[i][j] = multidegree_of_monomial(monomials[i][j], F_groups_ext)
        end
    end
    return N, monomials, deg_monomials, multideg_monomials
end

function extract_nonlinear_monomials(
        exprs::Vector{Num}, groups::NTuple{ORDP1, Vector{Num}}) where {ORDP1}
    N = length(exprs)
    # Extract monomials and calculate degree
    F_groups = groups[1:(end)]
    monomials = Vector{Any}(undef, N)
    deg_monomials = Vector{Vector{Int}}(undef, N)
    multideg_monomials = Vector{Vector{NTuple{ORDP1, Int}}}(undef, N)
    for i in 1:N
        monomials[i] = seperate_into_monomials(exprs[i])
        deg_monomials[i] = Vector{Int}(undef, length(monomials[i]))
        multideg_monomials[i] = Vector{NTuple{ORDP1, Int}}(undef, length(monomials[i]))
        for j in eachindex(monomials[i])
            deg_monomials[i][j] = degree_of_monomial(monomials[i][j])
            multideg_monomials[i][j] = multidegree_of_monomial(monomials[i][j], F_groups)
        end
    end
    return N, monomials, deg_monomials, multideg_monomials
end
"""
    group_monomials(monomials, multideg_monomials)

Collects monomials of the same multiindex in different components of the vector and 
defines a mapping `F_by_multiindex` that maps from the multiindices to the grouped monomial.
"""
function group_monomials(
        monomials::Vector, multideg_monomials::Vector{Vector{NTuple{ORD, Int}}}, N::Int) where {ORD}
    multiindices = sort(unique(vcat(multideg_monomials...)))
    F_by_multiindex = Dict{NTuple{ORD, Int}, Vector{Num}}()
    for mi in multiindices
        Fmi = fill(Num(0), N)
        for i in 1:N
            for (m, md) in zip(monomials[i], multideg_monomials[i])
                if md == mi
                    Fmi[i] += m
                end
            end
        end
        F_by_multiindex[mi] = Fmi
    end
    return F_by_multiindex
end

"""
    is_polynomial(exprs, all_vars)

Check whether all expressions in `exprs` are polynomial in `all_vars`.

Uses the Taylor ansatz: substitute all vars → ε * var, Taylor-expand in ε
around 0 up to the maximum degree found in `exprs`. If the expression is
polynomial of degree ≤ d, the Taylor expansion is exact and the difference
to the original is zero. If any transcendental or rational term is present,
the Taylor expansion differs from the original.
"""
function is_polynomial(exprs::Vector{Num}, all_vars::Vector{Num})
    @variables ε

    # find maximum degree across all expressions
    d = 0
    for expr in exprs
        for m in seperate_into_monomials(Symbolics.expand(expr))
            d = max(d, degree_of_monomial(m))
        end
    end

    scaling = Dict(v => ε * v for v in all_vars)

    for expr in exprs
        # substitute vars → ε * var to make ε the grading variable
        expr_scaled = Symbolics.substitute(expr, scaling)

        # Taylor expand in ε around 0 up to degree d
        taylor_approx = Symbolics.taylor(expr_scaled, ε, 0:d)

        # substitute ε → 1 to recover the polynomial approximation
        taylor_at_1 = Symbolics.substitute(taylor_approx, Dict(ε => Num(1)))

        # if polynomial, original and Taylor must agree exactly
        diff = Symbolics.expand(expr - taylor_at_1)
        if !isequal(diff, Num(0))
            return false
        end
    end
    return true
end

"""
    check_constant_terms(exprs, all_vars)

Returns a list of indices where `exprs[i]` has a nonzero constant term.
Empty vector means all expressions vanish at the origin.
"""
function check_constant_terms(exprs::Vector{Num}, all_vars::Vector{Num})
    zero_sub = Dict(v => Num(0) for v in all_vars)
    offending = Int[]
    for (i, expr) in enumerate(exprs)
        val = Symbolics.substitute(expr, zero_sub)
        if !isequal(Symbolics.expand(val), Num(0))
            push!(offending, i)
        end
    end
    return offending
end

"""
    check_all_vars_used(exprs, all_vars)

Returns variables from `all_vars` that do not appear in any expression in `exprs`.
Empty vector means all variables are used.
"""
function check_all_vars_used(exprs::Vector{Num}, all_vars::Vector{Num})
    # get all variables that actually appear in exprs
    vars_in_exprs = Set{Num}()
    for expr in exprs
        for v in Symbolics.get_variables(expr)
            push!(vars_in_exprs, Num(v))
        end
    end
    return [v for v in all_vars if !any(isequal(v, w) for w in vars_in_exprs)]
end

"""
    check_expr(exprs, all_vars)

Make some checks wether `exprs` is correctly defined.
"""
function check_expr(exprs::Vector{Num}, all_vars::Vector{Num})
    # checks wether exprs is polynomial
    @assert is_polynomial(exprs, all_vars)==true "`eprs` must be of polynomial form!"
    # checks if there are constant terms
    offending = check_constant_terms(exprs, all_vars)
    @assert isempty(offending) "exprs has nonzero constant terms in rows $offending — the equilibrium must be at the origin"
    unused = check_all_vars_used(exprs, all_vars)
    if !isempty(unused)
        unused_names = join([string(v) for v in unused], ", ")
        @warn "The following variables appear in `groups` but not in `exprs`: $unused_names"
    end
end