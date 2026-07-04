"""
    linear_operators.jl — Assemble B₀ and B₁ for the DPIM.

The linearised NSE around the steady state (u₀, p₀) at Re₀ reads:

    B₁ ṡ = −B₀ s + nonlinear terms

where s = [u′; p′] ∈ ℝ^FOM and

    B₁ = [M    0 ]        (singular — no time derivative for pressure)
         [0    0 ]

    B₀ = −A_lin = [(D/Re₀)K + J_conv(u₀)    −D^T]
                  [−D                           0 ]

A_lin is the linearised RHS operator:

    A_lin = [−(D/Re₀)K − J_conv(u₀)    D^T ]
            [ D                          0  ]

with:
  K_kl      = ∫ ∇φ^k : ∇φ^l dΩ                               (viscosity stiffness)
  J_conv_kl = ∫ φ^k · [(u₀·∇)φ^l + (φ^l·∇)u₀] dΩ            (linearised convection)

Note: viscosity ν = D·U_mean/Re₀ = _CYL_D/Re₀  (physical units, D = 0.1 m, U_mean = 1)
  D_ml      = ∫ ψ^m (∇·φ^l) dΩ                                (divergence)
  D^T_km    = ∫ (∇·φ^k) ψ^m dΩ                                (gradient)
  M_kl      = ∫ φ^k · φ^l dΩ                                  (velocity mass)

All matrices are assembled in the full DOF space (ndofs × ndofs) and then
restricted to the free-DOF subspace: B₀_free = B₀[free, free].

Note: A_lin is assembled as a NEGATIVE of the Newton Jacobian for the vel–vel and
vel–pres blocks, and identical to the Newton Jacobian for the pres–vel block.
"""

using Ferrite
using SparseArrays

# ─────────────────────────────────────────────────────────────────────────────
# Element-level assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    _assemble_linear_element!(Me, ALe, u₀_e, cv_vel, cv_pres,
                              dof_range_u, dof_range_p, Re0)

Compute element mass matrix block `Me` (velocity DOFs only) and element
linearised-operator matrix `ALe` (full element DOF space).

`ALe` contributes to A_lin; the caller negates it for B₀ = −A_lin.
"""
function _assemble_linear_element!(
    Me, ALe,
    u₀_e,                   # steady-state velocity DOFs for this element
    cv_vel, cv_pres,
    dof_range_u, dof_range_p,
    Re0::Float64,
)
    fill!(Me,  0.0)
    fill!(ALe, 0.0)

    n_vel  = length(dof_range_u)
    n_pres = length(dof_range_p)
    inv_Re = _CYL_D / Re0

    for q in 1:getnquadpoints(cv_vel)
        dΩ = getdetJdV(cv_vel, q)

        # Base-flow velocity and gradient at QP
        u₀_q  = function_value(cv_vel,    q, u₀_e)   # Vec{2}
        ∇u₀_q = function_gradient(cv_vel, q, u₀_e)   # Tensor{2,2}: ∂_j u₀_i

        # ── velocity–velocity block ──────────────────────────────────
        for i in 1:n_vel
            ri    = dof_range_u[i]
            φᵢ   = shape_value(cv_vel,    q, i)   # Vec{2}
            ∇φᵢ  = shape_gradient(cv_vel, q, i)   # Tensor{2,2}
            divφᵢ = tr(∇φᵢ)

            for j in 1:n_vel
                rj   = dof_range_u[j]
                φⱼ  = shape_value(cv_vel,    q, j)
                ∇φⱼ = shape_gradient(cv_vel, q, j)

                # Mass matrix entry (velocity block only)
                Me[ri, rj] += (φᵢ ⋅ φⱼ) * dΩ

                # A_lin vel–vel = −K/Re₀ − J_conv(u₀)
                # J_conv_kl = ∫ φ^k · [(u₀·∇)φ^l + (φ^l·∇)u₀] dΩ
                ALe[ri, rj] += (
                    -2 * inv_Re * (symmetric(∇φᵢ) ⊡ symmetric(∇φⱼ))   # −viscous: −2ν ε(φᵢ):ε(φⱼ)
                    - φᵢ ⋅ (∇φⱼ ⋅ u₀_q)                                # −(u₀·∇)φ^j
                    - φᵢ ⋅ (∇u₀_q ⋅ φⱼ)                                # −(φ^j·∇)u₀
                ) * dΩ
            end

            # ── velocity–pressure block (D^T) ──────────────────────
            for m in 1:n_pres
                rm  = dof_range_p[m]
                ψₘ = shape_value(cv_pres, q, m)
                ALe[ri, rm] += divφᵢ * ψₘ * dΩ   # +D^T
            end
        end

        # ── pressure–velocity block (D) ──────────────────────────────
        for m in 1:n_pres
            rm  = dof_range_p[m]
            ψₘ = shape_value(cv_pres, q, m)

            for j in 1:n_vel
                rj    = dof_range_u[j]
                divφⱼ = tr(shape_gradient(cv_vel, q, j))
                ALe[rm, rj] += ψₘ * divφⱼ * dΩ   # +D
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Global assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    assemble_linear_operators(s0_full, fom; Re0) -> (B0_free, B1_free, A_lin_free)

Assemble:
  B1_free  — velocity mass matrix restricted to free DOFs (singular)
  A_lin_free — linearised NSE operator restricted to free DOFs
  B0_free  = −A_lin_free

All three are sparse matrices of size n_free × n_free.

`s0_full` is the full-DOF steady-state solution (from `solve_steady_state`).
"""
function assemble_linear_operators(s0_full, fom; Re0::Float64)
    N     = ndofs(fom.dh)
    n_dpc = ndofs_per_cell(fom.dh)

    # Allocate full-space sparse matrices (same sparsity pattern as K)
    M_full   = allocate_matrix(fom.dh)
    AL_full  = allocate_matrix(fom.dh)   # A_lin

    fill!(M_full,  0.0)
    fill!(AL_full, 0.0)

    Me  = zeros(n_dpc, n_dpc)
    ALe = zeros(n_dpc, n_dpc)

    asm_M  = start_assemble(M_full)
    asm_AL = start_assemble(AL_full)

    for element in CellIterator(fom.dh)
        reinit!(fom.cv_vel,  element)
        reinit!(fom.cv_pres, element)

        dofs  = celldofs(element)
        u₀_e  = s0_full[dofs[fom.dof_range_u]]

        _assemble_linear_element!(
            Me, ALe, u₀_e,
            fom.cv_vel, fom.cv_pres,
            fom.dof_range_u, fom.dof_range_p,
            Re0,
        )

        assemble!(asm_M,  dofs, Me)
        assemble!(asm_AL, dofs, ALe)
    end

    # ── Restrict to DPIM free-DOF subspace (same prescribed set as the base
    #    flow — inlet perturbation frozen to zero, so free_dpim == free) ────
    B1_free   = M_full[fom.free_dpim, fom.free_dpim]
    A_lin_free = AL_full[fom.free_dpim, fom.free_dpim]
    B0_free    = -A_lin_free                   # B₀ = −A_lin

    @info "Linear operators assembled (Re₀ = $Re0)"
    @info "  B₁ nnz = $(nnz(B1_free)),  B₀ nnz = $(nnz(B0_free))"
    @info "  B₁ rank check: $(count(!iszero, diag(B1_free))) of $(size(B1_free,1)) diagonal entries nonzero"

    return B0_free, B1_free, A_lin_free
end

# ─────────────────────────────────────────────────────────────────────────────
# Sanity check: verify B₀, B₁ are consistent with the Newton Jacobian
# ─────────────────────────────────────────────────────────────────────────────

"""
    check_linearisation(s0_full, fom, B0_free; Re0, ε = 1e-5)

Finite-difference check: compare A_lin s ≈ [R(s₀ + ε·e_i) − R(s₀)] / ε
for a few random free DOF directions.  Prints max relative error.
"""
function check_linearisation(s0_full, fom, B0_free; Re0::Float64, ε::Float64 = 1e-5)
    N_free = fom.n_free_dpim
    A_lin_free = -B0_free

    # Evaluate residual at steady state (should be ≈ 0)
    K0 = allocate_matrix(fom.dh)
    R0 = zeros(ndofs(fom.dh))
    assemble_steady_nse!(K0, R0, s0_full, fom, Re0)
    apply_zero!(K0, R0, fom.ch_hom)
    R0_free = R0[fom.free_dpim]

    @info "Linearisation check: ‖R(u₀)‖ = $(norm(R0_free)) (should be ≈ 0)"

    # Reuse K0 as scratch for perturbed assembly (assemble_steady_nse! zeroes it internally)
    Rp   = zeros(ndofs(fom.dh))
    errs = Float64[]
    for _ in 1:10
        i   = rand(1:N_free)
        e_i = zeros(N_free);  e_i[i] = 1.0

        # Perturbed full solution
        s_pert = copy(s0_full)
        s_pert[fom.free_dpim] .+= ε .* e_i

        assemble_steady_nse!(K0, Rp, s_pert, fom, Re0)
        apply_zero!(K0, Rp, fom.ch_hom)

        fd_col = (Rp[fom.free_dpim] .- R0_free) ./ ε  # FD approximation of J·e_i

        # Newton Jacobian column: -A_lin e_i (since J_newton = -A_lin)
        J_col = A_lin_free[:, i]   # A_lin column i → should match FD
        push!(errs, norm(J_col .- (-fd_col)) / (norm(J_col) + 1e-15))
    end

    @info "  Max relative linearisation error: $(maximum(errs))"
    return maximum(errs)
end

# ─────────────────────────────────────────────────────────────────────────────
# Pressure lift weight vector
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_pressure_lift_weights(fom) -> Vector{Float64}

Assemble `l ∈ ℝ^N` such that `F_L^pres = l ⋅ u_full = ∫_Γ_cyl (-p n_y) dΓ`.

Sign convention matches `compute_drag_lift`: `n` is the outward normal from the
fluid, so the pressure traction is `-p·n` and `l[i] = -∫ n_y ψ_i^p dΓ` for
each pressure DOF `i` on the cylinder boundary (zero elsewhere).

Note: this captures the PRESSURE contribution only.  The viscous shear traction
`(2ν ε(u))·n` that `compute_drag_lift` integrates is deliberately omitted from
the lift polynomial L(z); at Re ≈ 50 it contributes a few percent of the total
lift, so L(z) slightly underestimates the physical lift amplitude.
"""
function compute_pressure_lift_weights(fom)
    ndofs_total = Ferrite.ndofs(fom.dh)
    l = zeros(Float64, ndofs_total)

    qr_face   = FacetQuadratureRule{RefTriangle}(4)
    ip_pres_f = Lagrange{RefTriangle, 1}()
    ip_geo    = Lagrange{RefTriangle, 1}()
    fv_pres   = FacetValues(qr_face, ip_pres_f, ip_geo)

    cyl_set = getfacetset(fom.grid, "Cylinder")

    for (cell_idx, local_facet_idx) in cyl_set
        cell = CellCache(fom.dh)
        reinit!(cell, cell_idx)
        gdofs = celldofs(cell)
        reinit!(fv_pres, cell, local_facet_idx)

        for q in 1:getnquadpoints(fv_pres)
            dΓ  = getdetJdV(fv_pres, q)
            n_y = getnormal(fv_pres, q)[2]

            for (i, li) in enumerate(fom.dof_range_p)
                l[gdofs[li]] += (-n_y) * shape_value(fv_pres, q, i) * dΓ
            end
        end
    end

    return l
end
