"""
Module `TimeIntegrationInterface`
Provides first-order form of an `NthOrderModel` (including its
`ExternalSystem`) for time integration for e.g. with the DifferentialEquations.jl package.
 
The ORD-th order model
 
	B_ORD x^(ORD) + … + B_1 ẋ + B_0 x = F(x, …, x^(ORD-1), r),      ṙ = g(r)
 
is written with `X = [x, ẋ, …, x^(ORD-1)]` and `u = [X; r_S]` as
 
	M u̇ = rhs(u, p, t)
 
with
- `M = blockdiag(I, …, I, B_ORD, I)`   (sparse, see [`mass_matrix`](@ref))
- `rhs = [ X₂; …; X_ORD; -Σ_{i<ORD} B_i X_{i+1} + F(X, r) ; g_S(r) ]`
 
No big block matrix is ever assembled: the linear part is applied block by block.
 
# External variables as inputs
 
`r` has `N_EXT` components. With `input = [i₁, i₂, …]` those components are *not*
integrated; their values are supplied through the DifferentialEquations parameter `p`,
either as a constant vector or as a callable `t -> vector` (same order as `input`).
The remaining external variables are integrated as part of `u`. Dynamics of the
integrated variables `g_S(r)` are still evaluated with the full `r`.
 
`r` is always the *physical* external state, i.e. the coordinates the external
dynamics were originally given in. If the `ExternalSystem` was re-based to triangular
coordinates `r′` (`r = Q r′`, see `external_basis`), the physical dynamics
`g(r) = Q g′(Q⁻¹ r)` are rebuilt once at construction. For real input dynamics the
round-off imaginary parts are dropped, so `u` stays real even when `Q` is complex
(e.g. harmonic forcing). Truly complex external dynamics give a complex `u`;
[`pack_state`](@ref) promotes the element type accordingly.
 
The mass matrix is `blockdiag(I, …, I, B_ORD, I)`, so solvers have to support mass
matrices (Rosenbrock, SDIRK, BDF methods, e.g. `Rodas5P()` or `FBDF()`); explicit
methods such as `Tsit5()` do not.
 
# Usage
 
```julia
sys = FirstOrderSystem(model; input = [2])           # r₂ is a parameter
f   = ODEFunction(sys; mass_matrix = mass_matrix(sys))
u0  = pack_state(sys, (x0, v0), [r1_0])
prob = ODEProblem(f, u0, (0.0, 10.0), t -> [sin(2t)])   # p = input signal
sol  = solve(prob, Rodas5P())
```
"""

module TimeIntegrationInterface


using LinearAlgebra
using SparseArrays
 
using ..FullOrderModel: NthOrderModel
using ..MultilinearMaps: evaluate_term!
using ..ExternalSystems: ExternalSystem, external_basis
using ..Polynomials: DensePolynomial, coefficients, multiindex_set, compose_linear
 
export FirstOrderSystem, mass_matrix,
       pack_state, unpack_state, state_dimension, physical_external_dynamics
 
"""
	FirstOrderSystem
 
Callable `sys(du, u, p, t)` that evaluates `rhs` of `M u̇ = rhs(u, p, t)`.
 
Fields:
- `model`        – the `NthOrderModel`
- `n`, `nX`      – `n_fom` and `ORD * n_fom`
- `n_ext`        – number of external variables `N_EXT`
- `input`        – external indices supplied through `p`
- `integrated`   – external indices that are part of `u`
- `ext_dynamics` – the external dynamics `g` in physical coordinates (`nothing` without
                   external system)
"""
struct FirstOrderSystem{ORD, M <: NthOrderModel, P}
    model::M
    n::Int                        # n_fom
    nX::Int                       # ORD * n_fom
    n_ext::Int                    # N_EXT
    input::Vector{Int}
    integrated::Vector{Int}
    ext_dynamics::P               # Nothing or DensePolynomial (physical coordinates)
end
 
"""
	FirstOrderSystem(model::NthOrderModel; input = Int[])
 
Build the first-order system of `model`. `input` lists the (physical) external variables
supplied through the parameter `p` instead of being integrated.
"""
function FirstOrderSystem(model::NthOrderModel{ORD, ORDP1, N_NL, N_EXT};
        input::AbstractVector{<:Integer} = Int[]) where {ORD, ORDP1, N_NL, N_EXT}
    all(i -> 1 <= i <= N_EXT, input) ||
        throw(ArgumentError("input indices must lie in 1:$N_EXT"))
    allunique(input) || throw(ArgumentError("input indices must be unique"))
    isempty(input) || model.external_system !== nothing ||
        throw(ArgumentError("model has no external system, so `input` must be empty"))
    integrated = [i for i in 1:N_EXT if i ∉ input]
    n = model.n_fom
    g = physical_external_dynamics(model.external_system)
    return FirstOrderSystem{ORD, typeof(model), typeof(g)}(
        model, n, ORD * n, N_EXT, collect(Int, input), integrated, g)
end
 
"""
	physical_external_dynamics(ext) -> DensePolynomial | Nothing
 
External dynamics `ṙ = g(r)` in the physical coordinates `r`. For a re-based system
(`r = Q r′`, `ṙ′ = g′(r′)`) this is `g(r) = Q g′(Q⁻¹ r)`. If the result is real up to
round-off, a real polynomial is returned, so the conjugate pairs introduced by a complex
`Q` do not make the time integration complex.
"""
physical_external_dynamics(::Nothing) = nothing
function physical_external_dynamics(ext::ExternalSystem)
    g′ = ext.first_order_dynamics
    Q = external_basis(ext)
    Q === nothing && return g′
    Qm = Matrix(Q)
    composed = compose_linear(g′, inv(Qm))                   # g′(Q⁻¹ r)
    C = Qm * coefficients(composed)                          # Q g′(Q⁻¹ r)
    if eltype(C) <: Complex
        scale = max(maximum(abs, C; init = zero(real(eltype(C)))), one(real(eltype(C))))
        if maximum(abs ∘ imag, C; init = zero(real(eltype(C)))) <=
           sqrt(eps(real(eltype(C)))) * scale
            C = real.(C)
        end
    end
    return DensePolynomial(C, multiindex_set(composed))
end
 
"""
	state_dimension(sys)
 
Length of `u = [X; r_S]`.
"""
state_dimension(sys::FirstOrderSystem) = sys.nX + length(sys.integrated)
 
_block(x, k, n) = view(x, ((k - 1) * n + 1):(k * n))
 
# y = L x on the mechanical block X: shifts, plus last row -[B₀ … B_{ORD-1}] X.
function _apply_linear!(y, model::NthOrderModel{ORD}, x) where {ORD}
    n = model.n_fom
    for k in 1:(ORD - 1)
        copyto!(_block(y, k, n), _block(x, k + 1, n))
    end
    last = _block(y, ORD, n)
    fill!(last, zero(eltype(last)))
    for i in 1:ORD
        mul!(last, model.linear_terms[i], _block(x, i, n), -1, 1) # multiplication of last row
    end
    return y
end
 
# y = M x (mechanical part): identities, with B_ORD in the last block. Not used by the
# rhs (the solver applies M itself); kept as the matrix-free counterpart of `mass_matrix`.
function _apply_mass!(y, model::NthOrderModel{ORD}, x) where {ORD}
    n = model.n_fom
    for k in 1:(ORD - 1)
        copyto!(_block(y, k, n), _block(x, k, n))
    end
    mul!(_block(y, ORD, n), model.linear_terms[end], _block(x, ORD, n))
    return y
end
 
"""
	mass_matrix(sys) -> SparseMatrixCSC
 
Sparse mass matrix `M = blockdiag(I, …, I, B_ORD, I)` of size `state_dimension(sys)`,
e.g. for `ODEFunction(sys; mass_matrix = mass_matrix(sys))` in DifferentialEquations.
"""
function mass_matrix(sys::FirstOrderSystem{ORD}) where {ORD}
    n = sys.n
    T = eltype(sys.model.linear_terms[1])
    nS = length(sys.integrated)
    idpart = sparse(one(T) * I, (ORD - 1) * n, (ORD - 1) * n)
    return blockdiag(idpart, sparse(sys.model.linear_terms[end]),
        sparse(one(T) * I, nS, nS))
end
 
"""
	pack_state(sys, xs::Tuple, r_integrated = ()) -> Vector
 
Assemble `u₀ = [x; ẋ; …; x^(ORD-1); r_S]`. `xs` has `ORD` vectors of length `n_fom`;
`r_integrated` holds the initial (physical) values of the integrated external variables,
in increasing index order. The element type is promoted to that of the model and of the
external dynamics, so a complex system yields a complex `u₀`.
"""
function pack_state(sys::FirstOrderSystem{ORD}, xs::Tuple, r = ()) where {ORD}
    length(xs) == ORD || throw(ArgumentError("expected $ORD state vectors (x, …, x^($(ORD-1)))"))
    length(r) == length(sys.integrated) ||
        throw(ArgumentError("expected $(length(sys.integrated)) integrated external values"))
    T = promote_type(eltype(sys.model.linear_terms[1]), map(eltype, xs)...,
        eltype(collect(r)), _coeff_eltype(sys.ext_dynamics))
    return convert(Vector{T}, vcat(xs..., collect(r)))
end
 
_coeff_eltype(::Nothing) = Bool          # neutral element for promote_type
_coeff_eltype(g::DensePolynomial) = eltype(coefficients(g))
 
"""
	unpack_state(sys, u) -> (xs::NTuple{ORD}, r_integrated)
 
Inverse of [`pack_state`](@ref); returns views into `u`.
"""
function unpack_state(sys::FirstOrderSystem{ORD}, u) where {ORD}
    xs = ntuple(k -> _block(u, k, sys.n), Val(ORD))
    return xs, view(u, (sys.nX + 1):length(u))
end
 
# ---- external system ------------------------------------------------------------
 
# Values of the input variables at time t from `p`: a vector, any callable `t -> vector`
# (functions, interpolation objects, …), or nothing. Anything else (e.g.
# `SciMLBase.NullParameters()` when no `p` was passed to the ODEProblem) counts as missing.
_input_values(::Nothing, t) = nothing
_input_values(p::Union{AbstractVector, Tuple}, t) = p
_input_values(p, t) = applicable(p, t) ? p(t) : nothing
 
# Full stored external vector r′ from integrated part and inputs.
function _assemble_external(sys::FirstOrderSystem, u, p, t)
    r = Vector{eltype(u)}(undef, sys.n_ext)
    for (k, i) in enumerate(sys.integrated)
        @inbounds r[i] = u[sys.nX + k]
    end
    if !isempty(sys.input)
        vals = _input_values(p, t)
        vals === nothing &&
            throw(ArgumentError("system has input variables $(sys.input): pass their values as `p`"))
        length(vals) == length(sys.input) ||
            throw(ArgumentError("`p` must give $(length(sys.input)) values, got $(length(vals))"))
        for (k, i) in enumerate(sys.input)
            @inbounds r[i] = vals[k]
        end
    end
    return r
end
 
# Evaluate component `i` of the external polynomial g at r, for i in `rows`, into out.
# Uses only `coefficients` (N_EXT × n_monomials) and `multiindex_set(...).exponents`.
function _external_dynamics!(out, poly::DensePolynomial, rows, r)
    C = coefficients(poly)
    exps = multiindex_set(poly).exponents
    fill!(out, zero(eltype(out)))
    for (m, e) in enumerate(exps)
        mon = one(eltype(r))
        for j in eachindex(r)
            @inbounds ej = e[j]
            ej == 0 || (mon *= r[j]^ej)
        end
        for (k, i) in enumerate(rows)
            @inbounds out[k] += C[i, m] * mon
        end
    end
    return out
end
 
# ---- right-hand side ------------------------------------------------------------
 
"""
	(sys::FirstOrderSystem)(du, u, p, t)
 
In-place evaluation of `rhs` of `M u̇ = rhs(u, p, t)` (DifferentialEquations.jl signature).
 
Arguments:
- `du` – output, overwritten with `rhs(u, p, t)`
- `u`  – state `[x; ẋ; …; x^(ORD-1); r_S]`, see [`pack_state`](@ref)
- `p`  – values of the input external variables: a vector, a callable `t -> vector`,
         or `nothing` if `input` is empty
- `t`  – time; only enters through a time-dependent `p`
"""
function (sys::FirstOrderSystem{ORD})(du, u, p, t) where {ORD}
    model = sys.model
    n, nX = sys.n, sys.nX
    ext = model.external_system

    N = state_dimension(sys)
    @assert length(du) == N "Size of `du` must be $N"
    @assert length(u) == N "Size of `u` must be $N"
 
    # apply linear part: du_X = L u_X
    _apply_linear!(view(du, 1:nX), model, view(u, 1:nX))
 
    # physical external state r (integrated part from u, inputs from p)
    r = ext === nothing ? nothing : _assemble_external(sys, u, p, t)
 
    # nonlinear part: accumulate F into the last block
    if !isempty(model.nonlinear_terms)
        xs = ntuple(k -> _block(u, k, n), Val(ORD))
        F_last = _block(du, ORD, n)
        for term in model.nonlinear_terms
            evaluate_term!(F_last, term, xs, r)
        end
    end
 
    # integrated external dynamics: ṙ_S = g_S(r)
    if sys.ext_dynamics !== nothing && !isempty(sys.integrated)
        _external_dynamics!(view(du, (nX + 1):length(du)), sys.ext_dynamics,
            sys.integrated, r)
    end
    return nothing
end

end # module TimeIntegrationInterface