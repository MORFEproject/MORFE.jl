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
    externalsystem_from_symbolics(exprs, var)
    
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