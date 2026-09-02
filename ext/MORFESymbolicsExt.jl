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
function _differential_equations_helper(
        f!, order::Int, nvars::Int; p = (), ext_vars::Union{Nothing, Vector{Num}} = nothing)
    @assert length(methods(f!)) == 1 "f must have exactly one method — pass a plain function, not a closure with multiple dispatches"
    coupled = ext_vars !== nothing
    # drop f itself, then p and t; a coupled f! carries one extra argument, the external state
    arity = methods(f!)[1].nargs - 1 - 2
    expected = coupled ? order + 2 : order + 1
    @assert arity==expected "f has arity $arity, expected $expected for an in-place order-$order problem" *
                            (coupled ? " coupled to an external system" : "")

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
    if coupled
        f!(out, descending_args..., ext_vars, p, t)
    else
        f!(out, descending_args..., p, t)
    end

    exprs = collect(highest) .- out
    groups = (lower_groups..., highest)

    used = reduce(union, Symbolics.get_variables.(exprs); init = Set())
    @assert !(Symbolics.value(t) in used) "f depends on t — the model must be autonomous; route time dependence through an ExternalSystem"
    return exprs, groups
end

"""
    MORFE.model_from_symbolics(f!, order, nvars; p = ())

Mirrors DifferentialEquations.jl's convention of defining ODEs, generalised to order `order`.

`f!` must be **in-place** (arity `order + 1`) and mutates its first argument:

    f!(dᵏu, dᵏ⁻¹u, ..., du, u, p, t)

Each `dⁱu` is a vector of length `nvars`. A non-mutating `f(u, p, t)` is not accepted here —
only `externalsystem_from_symbolics` supports that layout.

`f!` must be polynomial in the state and must not depend on `t`; `p` is passed through to `f!`
unchanged, so parameters may be closed over or supplied here.
"""
function MORFE.model_from_symbolics(f!, order::Int, nvars::Int; p = ())
    exprs, groups = _differential_equations_helper(f!, order, nvars; p = p)
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
    MORFE.model_from_symbolics(f!, order, nvars, f_ext, nvars_ext; p = (), p_ext = ())

Mirrors DifferentialEquations.jl's convention of defining ODEs, generalised to order `order` = k,
for a model coupled to an [`externalsystem_from_symbolics`](@ref) driver.

`f!` must be **in-place** and takes the external state `r` as one extra argument, after `u`
and before `p` — so its arity is `order + 2`:

    f!(dᵏu, dᵏ⁻¹u, ..., du, u, r, p, t)

Each `dⁱu` is a vector of length `nvars` and `r` one of length `nvars_ext`; referencing `r`
in `f!` is how forcing enters the model. `f_ext` describes the external dynamics `ṙ = E(r)`
and may be either layout accepted by `externalsystem_from_symbolics`: in-place
`f_ext(dr, r, p, t)` or out-of-place `f_ext(r, p, t) -> dr`.

Both right-hand sides must be polynomial and independent of `t`. `p` is passed to `f!` and
`p_ext` to `f_ext`, unchanged.
"""
function MORFE.model_from_symbolics(
        f!, order::Int, nvars::Int, f_ext, nvars_ext::Int; p = (), p_ext = ())
    # The driver is built first so `f!` can be handed the *same* external symbols it must
    # reference — otherwise the coupled model would carry a driver no term ever reads.
    exprs_ext, r_ext = _differential_equations_helper_external(f_ext, nvars_ext; p = p_ext)
    exprs, groups = _differential_equations_helper(
        f!, order, nvars; p = p, ext_vars = r_ext)
    return MORFE.model_from_symbolics(exprs, groups, r_ext, exprs_ext)
end

end # module MorfeSymbolics
