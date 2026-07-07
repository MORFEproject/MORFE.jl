function _is_zero(m)
    if typeof(iszero(m)) == Bool
        return iszero(m)
    else
        return isequal(m, Num(0)) || isequal(m, Num(0.0))
    end
end
function _is_zero(m::Complex{Num})
    # isequal(m, Complex{Num}(0)) || isequal(m, Complex{Num}(0.0))
    return _is_zero(real(m)) && _is_zero(imag(m))
end

function _is_zero(m::Vector{Num})
    for m_i in m
        if !_is_zero(m_i)
            return false
        end
    end
    return true
end

function _is_zero(m::Vector{Complex{Num}})
    for m_i in m
        if !_is_zero(m_i)
            return false
        end
    end
    return true
end

"""
    _to_MyNum

Helpers that define the correct build of MyNum for numbers.
"""
_to_MyNum(c::Real) = Num(c)
_to_MyNum(c::Complex) = Num(real(c)) + Num(imag(c)) * im
_to_MyNum(m::Num) = m
_to_MyNum(m::Complex{Num}) = m
_to_MyNum(m) = Num(m)

"""
    _numeric_eltype

Helper for the extraction of coefficients.
"""
_numeric_eltype(::Type{Num}) = Float64
_numeric_eltype(::Type{Complex{Num}}) = ComplexF64

"""
    extract_linear_matrices

Assemble linear_terms matrices for NDOrderModel.
Inputs are a ODE of the variables defined in `groups` that is a NTuple, where every part of the Tuple consists of the state variables or the derivatives. 
e.g: ([z1, z2, z3], [dz1, dz2, dz3], ...)
Returns tuple of matrices with B[i] is matric w.r.t. groups[i] so the ith derivative.  
    
"""
function extract_linear_matrices(
        exprs::Vector{NT}, groups::NTuple{ORDP1, Vector{Num}}, ext_var::Vector{Num} = Num[]) where {
        NT, ORDP1}
    n = length(groups[1])
    all_syms = vcat(groups..., ext_var)
    zero_sub = Dict(s => Num(0) for s in all_syms)

    B = ntuple(ORDP1) do k
        # compute Jacobian on real and imaginary parts separately
        re_exprs = real.(complex.(exprs))
        im_exprs = imag.(complex.(exprs))

        J_re = Symbolics.jacobian(re_exprs, groups[k])
        J_im = Symbolics.jacobian(im_exprs, groups[k])

        # recombine into complex Jacobian
        J = J_re .+ im .* J_im

        # substitute all vars to zero to extract the constant (linear) coefficient
        J0 = substitute.(J, (zero_sub,))

        T = _numeric_eltype(NT)
        T.(Symbolics.value.(J0))
    end
    return B
end

"""
    nonlinear_remainder

returns the nonlinear part of the equation as the same size as `exprs` with a minus so that it holds:
    expr: linear_terms = - nonlinear_remainder
Also checks that the nonlinear part is not allowed to depend on the highest derivative variables.
"""
function nonlinear_remainder(
        exprs::Vector{NT}, groups::NTuple{ORDP1, Vector{Num}}, B) where {NT, ORDP1}
    n = length(exprs)
    nl = Vector{NT}(undef, n)
    for i in 1:n
        lin = sum(B[k][i, j] * groups[k][j] for k in 1:ORDP1, j in 1:n)
        nl[i] = Symbolics.expand(lin-exprs[i])
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
function seperate_into_monomials(expr::Num)
    e = Symbolics.value(expand(expr))
    if Symbolics.iscall(e) && Symbolics.operation(e) == (+)
        collect(Symbolics.arguments(e))
    else
        [e]
    end
end
function seperate_into_monomials(expr::Complex{Num})
    # split into real and imaginary parts — both are plain Num
    re_monomials = seperate_into_monomials(real(expr))
    im_monomials = seperate_into_monomials(imag(expr))

    # recombine: real monomials stay as Num, imaginary ones become Complex{Num}
    result = Complex{Num}[]
    for m in re_monomials
        iszero(Symbolics.expand(Num(m))) || push!(result, Num(m))
    end
    for m in im_monomials
        iszero(Symbolics.expand(Num(m))) || push!(result, im * Num(m))
    end
    return result
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

function degree_of_monomial(term::Complex{Num})
    re_deg = iszero(real(term)) ? 0 : degree_of_monomial(real(term))
    im_deg = iszero(imag(term)) ? 0 : degree_of_monomial(imag(term))
    return max(re_deg, im_deg)
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
function multidegree_of_monomial(term::Complex{Num}, groups::NTuple{
        NG, Vector{Num}}) where {NG}
    # real and imaginary parts have the same variable structure, just different coefficients
    # pick whichever part is non-zero to analyze
    if !iszero(real(term))
        return multidegree_of_monomial(real(term), groups)
    else
        return multidegree_of_monomial(imag(term), groups)
    end
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
        exprs::Vector{<:MyNum}, groups::NTuple{ORDP1, Vector{Num}}, linear_terms) where {ORDP1}

    # Extract nonlinear linear_terms
    F = nonlinear_remainder(exprs, groups, linear_terms)

    # Check wether nonlinear remainder is zero
    if _is_zero(F)
        return nothing, nothing, nothing, nothing
    end

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
        exprs::Vector{<:MyNum},
        F_groups_ext::NTuple{NG_EXT, Vector{Num}},
        linear_terms,
        groups::NTuple{ORDP1, Vector{Num}}) where {NG_EXT, ORDP1}

    # subtract linear part using the full groups (same logic as the base method)
    F = nonlinear_remainder(exprs, groups, linear_terms)

    # Check wether nonlinear remainder is zero
    if _is_zero(F)
        return nothing, nothing, nothing, nothing
    end

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
        exprs::Vector{NT}, groups::NTuple{ORDP1, Vector{Num}}) where {NT, ORDP1}
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
    F_by_multiindex = Dict{NTuple{ORD, Int}, Vector{MyNum}}()
    for mi in multiindices
        Fmi = Vector{MyNum}(fill(Num(0), N))
        for i in 1:N
            for (m, md) in zip(monomials[i], multideg_monomials[i])
                if md == mi
                    Fmi[i] += _to_MyNum(m)
                end
            end
        end
        F_by_multiindex[mi] = Fmi
    end
    return F_by_multiindex
end

"""
    _get_taylor_expansion_around_0(expr, all_vars, d)

Helper function to determine wether a expression `expr` is a monome. Used in `is_polynomial`.
Uses the function Symbolics.taylor to calulate a Taylor expansion around zero.
"""
function _get_taylor_expansion_around_0(expr::Num, all_vars::Vector{Num}, d::Int)
    @variables ε
    scaling = Dict(v => ε * v for v in all_vars)
    # substitute vars → ε * var to make ε the grading variable
    expr_scaled = Symbolics.substitute(expr, scaling)
    # Taylor expand in ε around 0 up to degree d
    taylor_approx = Symbolics.taylor(expr_scaled, ε, 0:d)
    # substitute ε → 1 to recover the polynomial approximation
    taylor_at_1 = _to_MyNum(Symbolics.substitute(taylor_approx, Dict(ε => 1)))
end

function _get_taylor_expansion_around_0(expr::Complex{Num}, all_vars::Vector{Num}, d::Int)
    taylor_real = _get_taylor_expansion_around_0(real(expr), all_vars, d)
    taylor_imag = _get_taylor_expansion_around_0(imag(expr), all_vars, d)
    return taylor_real .+ im .* taylor_imag
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
function is_polynomial(exprs::Vector{NT}, all_vars::Vector{Num}) where {NT}
    # find maximum degree across all expressions
    d = 0
    for expr in exprs
        for m in seperate_into_monomials(Symbolics.expand(expr))
            d = max(d, degree_of_monomial(m))
        end
    end

    for expr in exprs
        taylor_at_1 = _get_taylor_expansion_around_0(expr, all_vars, d)
        # polynomial, original and Taylor must agree exactly
        diff = Symbolics.expand(expr - taylor_at_1)
        if !_is_zero(diff)
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
function check_constant_terms(exprs::Vector{NT}, all_vars::Vector{Num}) where {NT}
    zero_sub = Dict(v => 0 for v in all_vars)
    offending = Int[]
    for (i, expr) in enumerate(exprs)
        val = Symbolics.substitute(expr, zero_sub)
        if !iszero(Symbolics.expand(val))
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

function check_all_vars_used(exprs::Vector{Complex{Num}}, all_vars::Vector{Num})
    vars_in_exprs = Set{Num}()
    for expr in exprs
        for part in [real(expr), imag(expr)]
            for v in Symbolics.get_variables(Num(part))
                push!(vars_in_exprs, Num(v))
            end
        end
    end
    return [v for v in all_vars if !any(isequal(v, w) for w in vars_in_exprs)]
end

"""
    check_expr(exprs, all_vars)

Make some checks wether `exprs` is correctly defined.
"""
function check_expr(exprs::Vector{NT}, all_vars::Vector{Num}) where {NT}
    # checks wether exprs is polynomial
    @assert is_polynomial(exprs, all_vars)==true "`exprs` must be of polynomial form!"
    # checks if there are constant terms
    offending = check_constant_terms(exprs, all_vars)
    @assert isempty(offending) "exprs has nonzero constant terms in rows $offending — the equilibrium must be at the origin"
    unused = check_all_vars_used(exprs, all_vars)
    if !isempty(unused)
        unused_names = join([string(v) for v in unused], ", ")
        @warn "The following variables appear in `groups` but not in `exprs`: $unused_names"
    end
end