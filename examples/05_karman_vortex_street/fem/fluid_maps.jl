"""
    fluid_maps.jl — FEMMultilinearMap and MultilinearMap terms for the cylinder-flow DPIM.

Three nonlinear terms are implemented:

1. FluidConvection — quadratic convective term f₂(s, s):
       f₂(s₁, s₂) = −[(u₁·∇)u₂ + (u₂·∇)u₁]  (symmetrised for ORD=1)
   Implements FEMMultilinearMap{1} with multiindex=(2,), multiplicity_external=0.

2. make_param_coupling — parametric viscous coupling g₁(s, η′):
       g₁(s, η′) = −D·η′ · K_raw · u′
   A simple MultilinearMap{1} backed by a pre-assembled sparse matrix; main.jl
   passes K_visc = −D·K_raw (raw stiffness scaled by −_CYL_D in place).
   multiindex=(1,), multiplicity_external=1.

3. make_base_forcing — base-flow parametric forcing h₀(η′):
       h₀(η′) = −D·η′ · K_raw[free, ALL] · s₀
   A constant-in-state MultilinearMap{1} that provides the forcing direction when
   Re departs from Re₀.  multiindex=(0,), multiplicity_external=1.
   The rectangular free×ALL block is essential: the full base flow s₀ carries the
   Poiseuille profile on the prescribed inlet DOFs.
   This term drives the external direction Φ_ext in the cohomological solve: without
   it the (0,0,1) monomial has zero RHS and Φ_ext is computed incorrectly, making all
   η′-dependent reduced-dynamics coefficients wrong.
"""

using Ferrite
using LinearAlgebra
using SparseArrays
using MORFE

# ─────────────────────────────────────────────────────────────────────────────
# QP data type — velocity value + gradient at one quadrature point
# ─────────────────────────────────────────────────────────────────────────────

"""
    FluidVelQP{T}

Stores velocity value and gradient at one quadrature point, for one column of
the DPIM parametrisation W.  Used as the element type of the FEM qp buffer.

Fields:
  val  — Vec{2, T}: velocity value (u_x, u_y)
  grad — Tensor{2, 2, T}: velocity gradient, grad[i, j] = ∂_j u_i
"""
struct FluidVelQP{T}
    val ::Vec{2, T}
    grad::Tensor{2, 2, T}
end

# ─────────────────────────────────────────────────────────────────────────────
# FluidConvection — FEMMultilinearMap{1} for f₂(s, s) = −(u·∇)u symmetrised
# ─────────────────────────────────────────────────────────────────────────────

"""
    FluidConvection{DH, CV_VEL} <: MORFE.FEMMultilinearMap{1}

FEM-backed quadratic convective term for 2D incompressible flow.

Assembles the half-symmetrised bilinear form (so that f₂(s, s) = −∫ φ·(u·∇)u dΩ):
    f₂(s₁, s₂) = −½ ∫ φ · [(u₁·∇)u₂ + (u₂·∇)u₁] dΩ
               = −½ ∫ φ · [∇u₂ · u₁ + ∇u₁ · u₂] dΩ

where ∇u[i,j] = ∂_j u_i so that (∇u · v)[i] = Σ_j ∂_j u_i · v_j = (v·∇u)_i.

The QP buffer stores `FluidVelQP{ComplexF64}` (value + gradient) at each QP.
Only velocity DOF rows of Fe are written; pressure DOF rows remain zero.
"""
struct FluidConvection{DH, CV_VEL} <: MORFE.FEMMultilinearMap{1}
    dh              ::DH
    cv_vel          ::CV_VEL
    free_to_local   ::Dict{Int, Int}
    n_free          ::Int
    dof_range_u     ::UnitRange{Int}    # local vel DOF indices within a cell
    dof_range_p     ::UnitRange{Int}    # local pres DOF indices (needed for total size)
    n_vel_dofs_per_cell::Int

    # AbstractMultilinearMap interface fields
    multiindex          ::NTuple{1, Int}          # (2,)
    multiplicity_external::Int                     # 0
    deg                 ::Int                      # 2
    fully_asymmetric    ::Union{Nothing, Bool}     # false

    # Pre-allocated buffers
    qp_buffer ::Matrix{FluidVelQP{ComplexF64}}    # (max_unique_cols, n_qp)
    Fe        ::Vector{ComplexF64}                 # (ndofs_per_cell,)
    u_e       ::Vector{ComplexF64}                 # (n_vel_dofs_per_cell,)
end

"""
    FluidConvection(fom; max_unique_cols)

Construct a `FluidConvection` term from the FEM setup named tuple returned by
`setup_fem`.  `max_unique_cols` should equal the total number of monomials in the
DPIM multiindex set (passed after calling `all_multiindices_up_to`).
"""
function FluidConvection(fom; max_unique_cols::Int)
    n_qp     = getnquadpoints(fom.cv_vel)
    n_dpc    = ndofs_per_cell(fom.dh)
    n_vel    = fom.n_vel_dofs_per_cell

    zero_qp  = FluidVelQP(zero(Vec{2, ComplexF64}), zero(Tensor{2, 2, ComplexF64}))
    qp_buf   = fill(zero_qp, max_unique_cols, n_qp)
    Fe_buf   = zeros(ComplexF64, n_dpc)
    u_e_buf  = zeros(ComplexF64, n_vel)

    return FluidConvection(
        fom.dh, fom.cv_vel, fom.free_to_local_dpim, fom.n_free_dpim,
        fom.dof_range_u, fom.dof_range_p, n_vel,
        (2,), 0, 2, false,
        qp_buf, Fe_buf, u_e_buf,
    )
end

# ── FEMMultilinearMap interface ────────────────────────────────────────────

MORFE.fem_elements(t::FluidConvection)       = CellIterator(t.dh)
MORFE.fem_n_qp(t::FluidConvection)           = getnquadpoints(t.cv_vel)
MORFE.fem_ndofs_per_cell(t::FluidConvection) = ndofs_per_cell(t.dh)
MORFE.fem_qp_buffer(t::FluidConvection)      = t.qp_buffer
MORFE.fem_getdetJdV(_, q, t::FluidConvection) = getdetJdV(t.cv_vel, q)
MORFE.fem_reinit!(element, t::FluidConvection)  = reinit!(t.cv_vel, element)

"""
    MORFE.scatter_qp!(∇W_col, W_free, element, t::FluidConvection)

Extract velocity DOFs from the free-DOF state vector `W_free`, compute velocity
value and gradient at each QP, and store as `FluidVelQP` in `∇W_col`.

Pressure DOFs in `W_free` are ignored — only velocity DOFs feed into the
convective nonlinearity.
"""
function MORFE.scatter_qp!(∇W_col, W_free, element, t::FluidConvection)
    dofs     = celldofs(element)
    vel_dofs = dofs[t.dof_range_u]   # global velocity DOF indices for this cell
    u_e      = t.u_e

    # Lift velocity DOF values from free-DOF vector (0 for prescribed DOFs)
    for (i, d) in enumerate(vel_dofs)
        local_idx = get(t.free_to_local, d, 0)
        u_e[i] = local_idx == 0 ? zero(ComplexF64) : W_free[local_idx]
    end

    # Store (value, gradient) at each QP
    for q in eachindex(∇W_col)
        ∇W_col[q] = FluidVelQP(
            function_value(t.cv_vel, q, u_e),
            function_gradient(t.cv_vel, q, u_e),
        )
    end
end

"""
    MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{2}, mult, element, q, dΩ, t::FluidConvection)

Accumulate convective integrand at one quadrature point:

    Fe[k] += mult · φᵢ · (−½)[∇u₂·u₁ + ∇u₁·u₂] · dΩ   for velocity DOF k

∇u ⋅ v computes (v·∇u) via Tensor{2,2} × Vec{2} → Vec{2} contraction:
    (∇u ⋅ v)[i] = Σ_j ∂_j u_i · v_j = (v·∇u)_i
which is exactly the (v·∇)u material-derivative direction.
"""
function MORFE.accumulate_qp!(
    Fe, ∇W_args::NTuple{2, FluidVelQP{ComplexF64}}, mult, _, q, dΩ,
    t::FluidConvection,
)
    qp1, qp2 = ∇W_args

    # Symmetric bilinear form f₂(s₁,s₂) satisfying f₂(s,s) = f(s) = −(u·∇u).
    # MORFE assumes fully_asymmetric=false → calls f₂(W_col₁, W_col₂) once and
    # multiplies by the multinomial coefficient (e.g. ×2 for the z₁z₂ cross-term).
    # Therefore f₂ must be the HALF-symmetrised form:
    #   f₂(u₁,u₂) = −½[(u₁·∇)u₂ + (u₂·∇)u₁]
    # Check: f₂(u,u) = −½[2(u·∇u)] = −(u·∇u) ✓
    conv = -0.5 * (qp2.grad ⋅ qp1.val + qp1.grad ⋅ qp2.val)   # Vec{2, ComplexF64}

    c = mult * dΩ

    # Assemble into velocity DOF rows only
    for i in 1:t.n_vel_dofs_per_cell
        ri = t.dof_range_u[i]
        φᵢ = shape_value(t.cv_vel, q, i)   # Vec{2, Float64}
        Fe[ri] += c * (φᵢ ⋅ conv)
    end
    # Pressure rows of Fe remain at their initial (zero) value
end

"""
    MORFE.assemble_element!(accum, Fe, element, t::FluidConvection)

Scatter element residual into the global free-DOF accumulator.
Pressure-DOF entries of Fe are zero so only velocity-DOF contributions are added.
"""
function MORFE.assemble_element!(accum, Fe, element, t::FluidConvection)
    dofs = celldofs(element)
    for (r, d) in enumerate(dofs)
        local_idx = get(t.free_to_local, d, 0)
        local_idx != 0 && (accum[local_idx] += Fe[r])
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# FluidParamCoupling — MultilinearMap{1} for g₁(s, η′) = η′ · K_visc · s
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_param_coupling(K_visc_free) -> MultilinearMap{1}

Create a `MultilinearMap{1}` for the parametric viscous coupling

    g₁(s, η′) = η′ · K_visc · u′        (main.jl passes K_visc = −D·K_raw)

with multiindex=(1,) and multiplicity_external=1 (one state input, one η′ input).

K_visc_free is the viscosity stiffness restricted to free DOFs, pre-scaled by
−_CYL_D in main.jl so that the term equals −D·η′·K_raw·u′ = −(ν−ν₀)·K_raw·u′.
Its pressure rows and columns are zero, so the multiplication correctly
targets only velocity DOFs in both input and output.

MORFE passes the external argument as a unit SVector{1,Int}([1]) — the actual η′
scaling is handled by the polynomial factorisation in the cohomological solver.
Hence f!(accum, s, r_ext) adds r_ext[1] · K_visc · s = K_visc · s to accum.
"""
function make_param_coupling(K_visc_free::SparseMatrixCSC)
    K = K_visc_free   # close over the matrix

    # deg = 1 (state) + 1 (external) = 2;  f! takes (accum, s, r_ext) → nargs = 4
    function g₁!(accum, s, r_ext)
        mul!(accum, K, s, eltype(accum)(r_ext[1]), one(eltype(accum)))
    end

    return MultilinearMap(g₁!, (1,), 1; fully_asymmetric = false)
end

"""
    make_base_forcing(h₀_vec_free) -> MultilinearMap{1}

Create a `MultilinearMap{1}` for the base-flow parametric forcing

    h₀(η′) = −D·η′ · K_raw[free, ALL] · s₀

with multiindex=(0,) and multiplicity_external=1 (no state input, one η′ input).

h₀_vec_free = −D · K_raw[free, ALL] · s₀_full is a fixed vector encoding the
viscous forcing direction when Re departs from Re₀: the rectangular block acts on
the FULL base-flow vector, whose prescribed inlet DOFs carry the Poiseuille
profile.  It determines the external direction Φ_ext via the cohomological
equation for monomial (0,0,1).

MORFE passes the external argument as a unit SVector{1,Int}([1]), so
f!(accum, r_ext) adds r_ext[1] · h₀_vec to accum (η′ scaling is handled
by the polynomial factorisation framework).
"""
function make_base_forcing(h₀_vec_free::AbstractVector)
    h₀ = ComplexF64.(h₀_vec_free)   # close over a complex copy
    function h₀!(accum, r_ext)
        axpy!(eltype(accum)(r_ext[1]), h₀, accum)
    end
    return MultilinearMap(h₀!, (0,), 1; fully_asymmetric = false)
end

# ─────────────────────────────────────────────────────────────────────────────
# Assemble the raw viscosity stiffness (without 1/Re factor)
# ─────────────────────────────────────────────────────────────────────────────

"""
    assemble_K_visc(fom) -> (K_visc_free, K_visc_rect)

Assemble the raw viscosity stiffness matrix K_visc (full DOF space):

    K_visc_kl = ∫ 2 ε(φ^k) : ε(φ^l) dΩ    (velocity test and trial, P2)

Pressure DOF rows and columns are zero.  Two restrictions are returned:

  K_visc_free — square free×free block, for the parametric coupling g₁ acting on
                the perturbation (which vanishes on prescribed DOFs);
  K_visc_rect — rectangular free×ALL block, for the base-flow forcing h₀ = K·u₀,
                where u₀ is nonzero on the prescribed inlet DOFs (Poiseuille).
"""
function assemble_K_visc(fom)
    K_full = allocate_matrix(fom.dh)
    fill!(K_full, 0.0)

    n_dpc = ndofs_per_cell(fom.dh)
    Ke    = zeros(n_dpc, n_dpc)
    n_vel = fom.n_vel_dofs_per_cell
    asm   = start_assemble(K_full)

    for element in CellIterator(fom.dh)
        fill!(Ke, 0.0)
        reinit!(fom.cv_vel, element)

        for q in 1:getnquadpoints(fom.cv_vel)
            dΩ = getdetJdV(fom.cv_vel, q)
            for i in 1:n_vel
                ri   = fom.dof_range_u[i]
                ∇φᵢ = shape_gradient(fom.cv_vel, q, i)
                for j in 1:n_vel
                    rj   = fom.dof_range_u[j]
                    ∇φⱼ = shape_gradient(fom.cv_vel, q, j)
                    Ke[ri, rj] += 2 * (symmetric(∇φᵢ) ⊡ symmetric(∇φⱼ)) * dΩ
                end
            end
        end

        assemble!(asm, celldofs(element), Ke)
    end

    K_visc_free = K_full[fom.free_dpim, fom.free_dpim]
    K_visc_rect = K_full[fom.free_dpim, :]
    @info "K_visc assembled: $(nnz(K_visc_free)) nonzeros in free subspace"
    return K_visc_free, K_visc_rect
end
