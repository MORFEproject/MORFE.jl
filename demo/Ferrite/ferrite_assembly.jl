"""
Ferrite.jl FEM backend for MORFE.jl geometric nonlinearity.

Implements `FEMMultilinearMap` for the St. Venant-Kirchhoff material model,
which generates nonlinearity up to cubic order in the displacement field.

SVK internal virtual work:
    W_int = ∫ S:δE dΩ
where:
    E = ε(u) + ½∇u'∇u          (Green-Lagrange strain)
    S = λ tr(E) I + 2μ E        (2nd Piola-Kirchhoff stress)
    δE = δε + sym(∇u' δ∇u)

Expanding in powers of u gives quadratic and cubic contributions only.
Higher-order material models (e.g. polynomial hyperelastic) are needed for
quartic and higher terms.
"""

using Ferrite
using LinearAlgebra
using SparseArrays

import MORFE

# -----------------------------------------------------------------------
# Concrete FEMMultilinearMap type
# -----------------------------------------------------------------------

"""
    FerriteGeometricNonlinearity{DEG, DH, CV} <: MORFE.FEMMultilinearMap{2}

FEM-backed multilinear term for St. Venant-Kirchhoff geometric nonlinearity.

- `DEG = 2` : quadratic g_quad term (two displacement inputs)
- `DEG = 3` : cubic   h_cube term (three displacement inputs)

Type parameter `ORD = 2` means the multiindex lives in an NDOrderModel of
order 2 (second-order ODE). The term only uses position-derivative inputs
(multiindex = (DEG, 0)).

# Fields
- `dh`             — DofHandler
- `cv`             — CellValues (quadrature + interpolation)
- `free_to_local`  — Dict: global DOF index → 1-based index in the free-DOF vector
- `n_free`         — number of free DOFs
- `λ`, `μ`         — Lamé constants
- `multiindex`     — NTuple{2, Int} = (DEG, 0)
- `multiplicity_external` — 0 (no external forcing)
- `deg`            — DEG
- `∇W_qp`         — pre-allocated qp gradient buffer, Matrix{Tensor{2,3,ComplexF64}}(DEG, n_qp)
- `Fe`             — pre-allocated element residual, Vector{ComplexF64}(ndofs_per_cell)
- `u_e`            — pre-allocated element DOF vector, Vector{ComplexF64}(ndofs_per_cell)
"""
struct FerriteGeometricNonlinearity{DEG, DH, CV} <: MORFE.FEMMultilinearMap{2}
    dh::DH
    cv::CV
    free_to_local::Dict{Int, Int}
    n_free::Int
    λ::Float64
    μ::Float64
    multiindex::NTuple{2, Int}
    multiplicity_external::Int
    deg::Int
    ∇W_qp::Matrix{Tensor{2, 3, ComplexF64}}
    Fe::Vector{ComplexF64}
    u_e::Vector{ComplexF64}
end

"""
    FerriteGeometricNonlinearity{DEG}(dh, cv, free_to_local, n_free, λ, μ)

Construct with pre-allocated buffers sized from `cv`.
"""
function FerriteGeometricNonlinearity{DEG}(
        dh::DH, cv::CV,
        free_to_local::Dict{Int, Int}, n_free::Int,
        λ::Float64, μ::Float64) where {DEG, DH, CV}
    n_qp   = getnquadpoints(cv)
    n_dofs = ndofs_per_cell(dh)
    ∇W_qp  = Matrix{Tensor{2, 3, ComplexF64}}(undef, DEG, n_qp)
    Fe     = Vector{ComplexF64}(undef, n_dofs)
    u_e    = Vector{ComplexF64}(undef, n_dofs)
    return FerriteGeometricNonlinearity{DEG, DH, CV}(
        dh, cv, free_to_local, n_free, λ, μ,
        (DEG, 0), 0, DEG, ∇W_qp, Fe, u_e)
end

# -----------------------------------------------------------------------
# FEMMultilinearMap interface
# -----------------------------------------------------------------------

MORFE.fem_elements(t::FerriteGeometricNonlinearity) = CellIterator(t.dh)

MORFE.fem_n_qp(t::FerriteGeometricNonlinearity) = getnquadpoints(t.cv)

MORFE.fem_ndofs_per_cell(t::FerriteGeometricNonlinearity) = ndofs_per_cell(t.dh)

MORFE.fem_qp_buffer(t::FerriteGeometricNonlinearity) = t.∇W_qp

MORFE.fem_getdetJdV(_element, q, t::FerriteGeometricNonlinearity) = getdetJdV(t.cv, q)

# Called once per element in _replay_fem_split! before the scatter loop (O1).
MORFE.fem_reinit!(element, t::FerriteGeometricNonlinearity) = reinit!(t.cv, element)

"""
    MORFE.scatter_qp!(∇W_col, W_free, element, t)

Scatter the free-DOF vector `W_free` to per-quadrature-point displacement gradients
∇W_col[q] = ∇u(ξ_q).  CellValues must already be reinit!-ed for `element` via
`fem_reinit!` before this call.
"""
function MORFE.scatter_qp!(∇W_col, W_free, element, t::FerriteGeometricNonlinearity)
    dofs = celldofs(element)
    u_e = t.u_e
    for (i, d) in enumerate(dofs)
        local_idx = get(t.free_to_local, d, 0)
        u_e[i] = local_idx == 0 ? zero(ComplexF64) : W_free[local_idx]
    end
    for q in eachindex(∇W_col)
        ∇W_col[q] = function_gradient(t.cv, q, u_e)
    end
end

# Lamé stress from a strain tensor.
@inline _σ(E, λ, μ) = λ * tr(E) * one(E) + 2μ * E

# Symmetric Green-Lagrange cross term for two gradients.
@inline _E_nl(∇u1, ∇u2) = symmetric(
    Tensor{2, 3}(0.25 * (transpose(∇u1) ⋅ ∇u2 + transpose(∇u2) ⋅ ∇u1)))

"""
    MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{2}, mult, element, q, dΩ, t)

Quadratic geometric nonlinearity integrand at one quadrature point:

    fe_r += mult * [ε(φ_r) ⊡ σ(E_nl(∇u1,∇u2))
                    + 0.5*(sym(∇u1'⋅∇φ_r) ⊡ σ(ε(∇u2))
                         + sym(∇u2'⋅∇φ_r) ⊡ σ(ε(∇u1)))] * dΩ
"""
function MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{2}, mult, _element, q, dΩ,
        t::FerriteGeometricNonlinearity{2})
    ∇u1, ∇u2 = ∇W_args
    E_nl = _E_nl(∇u1, ∇u2)
    σ_nl = _σ(E_nl, t.λ, t.μ)
    ε1 = symmetric(∇u1)
    ε2 = symmetric(∇u2)
    σ_ε1 = _σ(ε1, t.λ, t.μ)
    σ_ε2 = _σ(ε2, t.λ, t.μ)
    n_dofs = ndofs_per_cell(t.dh)
    c = ComplexF64(mult * dΩ)
    for r in 1:n_dofs
        ∂Nr = shape_gradient(t.cv, q, r)
        δε  = symmetric(∂Nr)
        Fe[r] += c * (
            δε ⊡ σ_nl
            +
            0.5 * (symmetric(Tensor{2, 3}(transpose(∇u1) ⋅ ∂Nr)) ⊡ σ_ε2
             +
             symmetric(Tensor{2, 3}(transpose(∇u2) ⋅ ∂Nr)) ⊡ σ_ε1)
        )
    end
end

"""
    MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{3}, mult, element, q, dΩ, t)

Cubic geometric nonlinearity integrand at one quadrature point:

    fe_r += mult/3 * Σ_{(i,j,k) cyclic} sym(∇ui'⋅∇φ_r) ⊡ σ(E_nl(∇uj,∇uk)) * dΩ
"""
function MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{3}, mult, _element, q, dΩ,
        t::FerriteGeometricNonlinearity{3})
    ∇u1, ∇u2, ∇u3 = ∇W_args
    E_nl_23 = _E_nl(∇u2, ∇u3);
    σ_23 = _σ(E_nl_23, t.λ, t.μ)
    E_nl_13 = _E_nl(∇u1, ∇u3);
    σ_13 = _σ(E_nl_13, t.λ, t.μ)
    E_nl_12 = _E_nl(∇u1, ∇u2);
    σ_12 = _σ(E_nl_12, t.λ, t.μ)
    n_dofs = ndofs_per_cell(t.dh)
    c = ComplexF64(mult * dΩ / 3)
    for r in 1:n_dofs
        ∂Nr = shape_gradient(t.cv, q, r)
        Fe[r] += c * (
            symmetric(Tensor{2, 3}(transpose(∇u1) ⋅ ∂Nr)) ⊡ σ_23
            + symmetric(Tensor{2, 3}(transpose(∇u2) ⋅ ∂Nr)) ⊡ σ_13
            + symmetric(Tensor{2, 3}(transpose(∇u3) ⋅ ∂Nr)) ⊡ σ_12
        )
    end
end

"""
    MORFE.assemble_element!(accum, Fe, element, t)

Scatter element residual `Fe` (indexed by local DOF) into the global free-DOF
accumulator `accum` (indexed by free-DOF position).
"""
function MORFE.assemble_element!(accum, Fe, element, t::FerriteGeometricNonlinearity)
    dofs = celldofs(element)
    for (r, d) in enumerate(dofs)
        local_idx = get(t.free_to_local, d, 0)
        local_idx != 0 && (accum[local_idx] += Fe[r])
    end
end

# -----------------------------------------------------------------------
# Linear matrix assembly
# -----------------------------------------------------------------------

"""
    assemble_KM!(K, M, dh, cv, λ, μ, ρ)

Assemble the global stiffness matrix `K` and mass matrix `M` into pre-allocated
sparse matrices using standard Galerkin FEM.

    K_rs = ∫ ε(φ_r) ⊡ (λ tr(ε(φ_s)) I + 2μ ε(φ_s)) dΩ
    M_rs = ∫ ρ φ_r · φ_s dΩ
"""
function assemble_KM!(K, M, dh, cv, λ::Float64, μ::Float64, ρ::Float64)
    n_dpc = ndofs_per_cell(dh)
    Ke = zeros(n_dpc, n_dpc)
    Me = zeros(n_dpc, n_dpc)
    asm_K = start_assemble(K)
    asm_M = start_assemble(M)

    for element in CellIterator(dh)
        fill!(Ke, 0.0)
        fill!(Me, 0.0)
        reinit!(cv, element)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            for r in 1:n_dpc
                δε = shape_symmetric_gradient(cv, q, r)
                Nr = shape_value(cv, q, r)
                for s in 1:n_dpc
                    ε = shape_symmetric_gradient(cv, q, s)
                    σ = λ * tr(ε) * one(ε) + 2μ * ε
                    Ke[r, s] += (δε ⊡ σ) * dΩ
                    Ns = shape_value(cv, q, s)
                    Me[r, s] += ρ * (Nr ⋅ Ns) * dΩ
                end
            end
        end
        assemble!(asm_K, celldofs(element), Ke)
        assemble!(asm_M, celldofs(element), Me)
    end
end
