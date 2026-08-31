"""
For the definition of MORFE.ExternalSystems via symbolic expressions.
A external system is given in the form:

	dr/dt = f(r) = A r + higher-order terms

where `r ∈ ℝ^{N_EXT}` (or ℂ^{N_EXT}),
"""

function seperate_into_monomials(exprs::Vector{<:MyNum}, var::Vector{Num})
    N = length(exprs)
    # Extract monomials and calculate degree
    monomials = Vector{Any}(undef, N)
    deg_monomials = Vector{Vector{Int}}(undef, N)
    for i in 1:N
        monomials[i] = seperate_into_monomials(exprs[i])
        deg_monomials[i] = Vector{Int}(undef, length(monomials[i]))
        for j in eachindex(monomials[i])
            deg_monomials[i][j] = degree_of_monomial(monomials[i][j])
        end
    end
    return N, monomials, deg_monomials
end

"""
    group_monomials(monomials, multideg_monomials)

Collects monomials of the same multiindex in different components of the vector and 
defines a mapping `F_by_multiindex` that maps from the multiindices to the grouped monomial.
"""
function group_monomials(
        monomials::Vector, deg_monomials::Vector{Vector{Int}}, N::Int)
    multiindices = sort(unique(vcat(deg_monomials...)))
    F_by_multiindex = Dict{Int, Vector{MyNum}}()
    for mi in multiindices
        Fmi = Vector{MyNum}(fill(Num(0), N))
        for i in 1:N
            for (m, md) in zip(monomials[i], deg_monomials[i])
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
    get_coefficients(exprs, var)

Returns the coefficient of a monomial by evaluating every variable in `var` with 1.
Attention: There is no check wether `exprs` is a monomial or not!
"""
function get_coefficients(exprs::Vector{<:MyNum}, var::Vector{Num})
    dict_to_1 = Dict(v => 1 for v in var)
    e = [Symbolics.substitute(expr, dict_to_1) for expr in exprs]
    return ComplexF64.(Symbolics.value.(e))
end

"""
    generate_polynomial(dict, var, N)

Generates a MORFE.DensePolynomial from a dictionary calculated from group_monomials.
"""
function generate_polynomial(
        dict::Dict{NTuple{ORD, Int}, Vector{MyNum}}, var::Vector{Num}, N::Int) where {ORD}
    @assert ORD == N
    n = length(keys(dict))
    # Multiindices 
    multiindices = Array{SVector{N, Int}}(undef, n)
    index = 1
    for mi in keys(dict)
        multiindices[index] = SVector(mi)
        index += 1
    end
    mset = MultiindexSet(multiindices) # get sorted lexicographically
    coefficients = Array{ComplexF64}(undef, N, n)
    index = 1
    for mi in mset.exponents
        coefficients[:, index] = get_coefficients(dict[Tuple(mi)], var)
        index += 1
    end
    return DensePolynomial(coefficients, MultiindexSet(multiindices))
end

"""
    MORFE.externalsystem_from_symbolics(exprs, var)
    
Generates an MORFE.ExternalSystem. Expects the ODE describing the external system in the form

    dr/dt var = exprs

"""
function MORFE.externalsystem_from_symbolics(exprs::Vector{<:MyNum}, var::Vector{Num})
    N, monomials,
    _, multideg_monomials = extract_nonlinear_monomials(
        exprs, tuple(([x] for x in var)...))
    F_dict = group_monomials(monomials, multideg_monomials, N)
    pol = generate_polynomial(F_dict, var, N)
    ex_system = ExternalSystem(pol)
    return ex_system
end

"""
    _differential_equations_helper_external(f, order::Int, nvars::Int; p = ())

Helper for mirroring DifferentialEquations.jl interface.
Used in MORFE.externalsystem_from_symbolics(f, nvars::Int; p = ()).
"""
function _differential_equations_helper_external(f, nvars::Int; p = ())
    @assert length(methods(f)) == 1 "f must have exactly one method — pass a plain function, not a closure with multiple dispatches"
    arity = methods(f)[1].nargs - 1 - 2   # drop f itself, then p and t
    t = Num(Symbolics.variable(:t))
    r = Num.(Symbolics.variables(:r, 1:nvars))

    if arity == 2                          # in-place: f(dr, r, p, t)
        exprs = Vector{Num}(undef, nvars)
        fill!(exprs, Num(0))
        f(exprs, r, p, t)
    elseif arity == 1                      # out-of-place: f(r, p, t) -> dr
        exprs = Vector{Num}(f(r, p, t))
    else
        error("f needs to be one of these layouts: f(dr, r, p, t) or f(r, p, t)")
    end

    @assert length(exprs) == nvars "f must produce a vector of length $nvars"
    used = reduce(union, Symbolics.get_variables.(exprs); init = Set())
    @assert !(Symbolics.value(t) in used) "f depends on t — external systems must be autonomous"
    return exprs, r
end

"""
    MORFE.externalsystem_from_symbolics(f, nvars::Int; p=())

Mirrors DifferentialEquations.jl's  convention of defining ODEs.
There are the two options to define the function f.
1) In-place, mutates the first argument:
    f(dr, r, p, t)
2) Returns dr
    f(r, p, t) -> dr
"""
function MORFE.externalsystem_from_symbolics(f, nvars::Int; p = ())
    exprs, r = _differential_equations_helper_external(f, nvars::Int; p = ())
    return MORFE.externalsystem_from_symbolics(exprs, r)
end
