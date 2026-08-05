"""
This module is an extension using Symbolics.jl to introduce 
a clean and nice looking interface for definition and generating the NDOrdermodel
and ExternalSystem
"""
module MORFESymbolicsExt

using MORFE
using Symbolics
using StaticArrays

export model_from_symbolics
export externalsystem_from_symbolics

const MyNum = Union{Num, Complex{Num}}

include("MORFESymbolicsExt/extraction.jl")
include("MORFESymbolicsExt/polarization.jl")
include("MORFESymbolicsExt/toMultilinearMaps.jl")
include("MORFESymbolicsExt/external_system.jl")

"""
    model_from_symbolics

Generates NthOrderModel.
Inputs are a ODE of the variables defined in `groups` that is a NTuple, where every part of the Tuple consists of the state variables or the derivatives. 
e.g: ([z1, z2, z3], [dz1, dz2, dz3], ...).
The ODE is supposed to be a vector equal to zero and is inputed in the variable `exprs`. 
"""
function MORFE.model_from_symbolics(
        exprs::Vector{<:MyNum}, groups::NTuple{ORDP1, Vector{Num}}) where {ORDP1}
    # Check size of groups
    @assert ORDP1>1 "ORDP1 must be bigger than zero"
    N = length(groups[1])
    for i in 2:ORDP1
        @assert N==length(groups[i]) "Vectors in `groups` must have the same size."
    end
    check_expr(exprs, vcat(groups...))

    # Extract linear_terms
    linear_terms = extract_linear_matrices(exprs, groups)

    # Extract nonlinear linear_terms
    N, monomials,
    deg_monomials,
    multideg_monomials = extract_nonlinear_monomials(
        exprs, groups, linear_terms)

    # if nonlinear_remainder == zero
    if N===nothing && monomials===nothing && deg_monomials===nothing &&
       multideg_monomials===nothing
        return NthOrderModel(linear_terms)
    end

    F_by_multiindex = group_monomials(
        monomials, multideg_monomials, N)

    # Filter out the zero multiindex — it's just the constant remainder (should be 0)
    # filter!(kv -> !all(iszero, kv.first), F_by_multiindex)
    #If the nonlinear remainder is empty dont do the polarization
    if isempty(F_by_multiindex)
        @error "no nonlinear part used!"
    end
    F_by_multiindex_polarized,
    dict_pol_vars = polarize(F_by_multiindex, groups[1:(end - 1)], N)
    nonlinear_terms = all_monomials_to_MultilinearMaps(
        F_by_multiindex_polarized, dict_pol_vars)
    model = NthOrderModel(linear_terms, nonlinear_terms)
    return model
end

"""
    _differential_equations_helper(f!, order::Int, nvars::Int; p = ())

Helper for mirroring DifferentialEquations.jl interface.
Used in MORFE.model_from_symbolics(f, nvars::Int; p = ()).
"""
function _differential_equations_helper(f!, order::Int, nvars::Int; p = ())
    @assert length(methods(f!)) == 1 "f must have exactly one method — pass a plain function, not a closure with multiple dispatches"
    arity = methods(f!)[1].nargs - 1 - 2   # drop f itself, then p and t
    @assert arity == order + 1 "f has arity $arity, expected $(order+1) for an in-place order-$order problem"

    # ascending symbolic groups: (u, du, ..., d^{order-1}u)
    names = ["u"; ["d"^k * "u" for k in 1:(order - 1)]]
    lower_groups = ntuple(order) do k
        Num.(Symbolics.variables(Symbol(names[k]), 1:nvars))
    end
    descending_args = reverse(lower_groups)   # (d^{order-1}u, ..., du, u) — matches f's arg order

    highest = Num.(Symbolics.variables(Symbol("d" * "d"^(order - 1) * "u"), 1:nvars))
    t = Num(Symbolics.variable(:t))

    out = Vector{Num}(undef, nvars)
    fill!(out, Num(0))
    f!(out, descending_args..., p, t)

    exprs = collect(highest) .- out
    groups = (lower_groups..., highest)

    used = reduce(union, Symbolics.get_variables.(exprs); init = Set())
    @assert !(Symbolics.value(t) in used) "f depends on t — external systems must be autonomous"
    return exprs, groups
end

"""
    MORFE.model_from_symbolics(f!, order, nvars; p = ())

Mirrors DifferentialEquations.jl's  convention of defining ODEs, generalized to order 'order'.
In-place (arity == order + 1), mutates the first argument:
    f(dᵏu, dᵏ⁻¹u, ..., du, u, p, t)
Each dⁱ needst be a vector of length 'nvars'.
"""
function MORFE.model_from_symbolics(f!, order::Int, nvars::Int; p = ())
    exprs, groups = _differential_equations_helper(f!, order, nvars; p = ())
    return MORFE.model_from_symbolics(exprs, groups)
end

"""
    model_from_symbolics
 
Generates NthOrderModel where the nonlinear forcing terms may also depend on external
variables `ext_var` (the state of an ExternalSystem).
 
The multiindices of the resulting MultilinearMaps have `ORD + 1` entries:
    (d_x, d_ẋ, …, d_r)
where the last entry counts the degree in the external variables.
 
# Arguments
- `exprs`  : vector of ODE expressions (= 0), may contain both `groups` variables and `ext_var`
- `groups` : NTuple of variable groups as in `model_from_symbolics`
- `ext_var`: vector of external/forcing variables (state of the ExternalSystem)
"""
function MORFE.model_from_symbolics(
        exprs::Vector{<:MyNum},
        groups::NTuple{ORDP1, Vector{Num}},
        ext_var::Vector{Num},
        ext_exprs::Vector{<:MyNum}) where {ORDP1}
    # ---- sanity checks -------------------------------------------------------
    @assert ORDP1 > 1 "ORDP1 must be bigger than one"
    N = length(groups[1])
    for i in 2:ORDP1
        @assert N == length(groups[i]) "Vectors in `groups` must have the same size."
    end
    @assert length(ext_var) > 0 "ext_var must be non-empty; use `model_from_symbolics` otherwise"
    check_expr(exprs, vcat(groups..., ext_var))

    # ---- linear terms (only w.r.t. state groups, same as before) -------------
    linear_terms = extract_linear_matrices(exprs, groups, ext_var)

    # ---- nonlinear remainder -------------------------------------------------
    # F_groups_ext = (groups[1], …, groups[end-1], ext_var)
    # i.e. we drop the highest-derivative group and append ext_var instead.
    F_groups_ext = (groups[1:(end - 1)]..., ext_var)   # NTuple{ORDP1-1+1 = ORDP1, …}

    N_check, monomials,
    deg_monomials,
    multideg_monomials = extract_nonlinear_monomials(exprs, F_groups_ext, linear_terms, groups)

    # if nonlinear_remainder == zero
    if N_check===nothing && monomials===nothing && deg_monomials===nothing &&
       multideg_monomials===nothing
        return NthOrderModel(linear_terms)
    end

    F_by_multiindex = group_monomials(monomials, multideg_monomials, N)

    # Filter out the zero multiindex — it's just the constant remainder (should be 0)
    # filter!(kv -> !all(iszero, kv.first), F_by_multiindex)
    #If the nonlinear remainder is empty dont do the polarization
    if isempty(F_by_multiindex)
        @error "no nonlinear part used!"
    end

    # polarize over F_groups_ext (state slots + ext slot)
    F_by_multiindex_polarized,
    dict_pol_vars = polarize(F_by_multiindex, F_groups_ext, N)

    nonlinear_terms = all_monomials_to_MultilinearMaps(
        F_by_multiindex_polarized, dict_pol_vars, has_ext = true)

    # Build Morfe.ExternalSystem
    @assert is_polynomial(ext_exprs, ext_var) "External system `ext_system` must be in polynomial form!"
    ext_system = externalsystem_from_symbolics(ext_exprs, ext_var)

    model = NthOrderModel(linear_terms, nonlinear_terms, ext_system)
    return model
end

"""
    MORFE.model_from_symbolics(f!, order, nvars, f_ext, nvars_ext; p = ())

Mirrors DifferentialEquations.jl's  convention of defining ODEs, generalized to order 'order'=k.
In-place (arity == order + 1), mutates the first argument:
    f!(dᵏu, dᵏ⁻¹u, ..., du, u, p, t)
Each dⁱ needst be a vector of length 'nvars'.
f! is supposed to be a polynom and is not allowed to be dependent on `t`.
"""
function MORFE.model_from_symbolics(
        f!, order::Int, nvars::Int, f_ext, nvars_ext::Int; p = (), p_ext = ())
    exprs, groups = _differential_equations_helper(f!, order, nvars; p = ())
    exprs_ext, r_ext = _differential_equations_helper_external(f_ext, nvars_ext; p_ext = ())
    return MORFE.model_from_symbolics(exprs, groups, exprs_ext, r_ext)
end

end # module MorfeSymbolics
