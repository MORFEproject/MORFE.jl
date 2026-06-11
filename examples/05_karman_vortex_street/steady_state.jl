"""
    steady_state.jl — Newton iteration for the steady incompressible NSE.

Solves the non-dimensional steady Navier-Stokes equations at Reynolds number Re₀:

    (1/Re₀) ∫ ∇v:∇u dΩ + ∫ v·(u·∇u) dΩ - ∫ (∇·v) p dΩ = 0   ∀v (momentum)
    ∫ q (∇·u) dΩ = 0                                              ∀q (incompressibility)

using Newton–Raphson iteration in the full DOF space (velocity + pressure).

Residual and consistent tangent are assembled per element at each iteration.
The reduced (free-DOF) linear system is solved with KLU sparse direct.

Convective Jacobian includes BOTH terms:
    dN/du [φ^l] = ∫ φ^i · [(φ^l·∇)u + (u·∇)φ^l] dΩ
"""

using Ferrite
using LinearAlgebra
using SparseArrays
using KLU

# ─────────────────────────────────────────────────────────────────────────────
# Element-level assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    _assemble_element!(Ke, Re_e, u_e, p_e, cv_vel, cv_pres, dof_range_u, dof_range_p, Re0)

Compute element Jacobian `Ke` and residual `Re_e` for the steady NSE at `Re0`.

Quadrature loop fills three blocks:
  • vel–vel  : viscous + convective tangent (two terms)
  • vel–pres : −G^T  (pressure gradient term in momentum)
  • pres–vel :  G    (divergence constraint)
"""
function _assemble_element!(
    Ke, Re_e,
    u_e, p_e,
    cv_vel, cv_pres,
    dof_range_u, dof_range_p,
    Re0::Float64,
)
    fill!(Ke,   0.0)
    fill!(Re_e, 0.0)

    n_vel  = length(dof_range_u)
    n_pres = length(dof_range_p)
    inv_Re = _CYL_D / Re0

    for q in 1:getnquadpoints(cv_vel)
        dΩ = getdetJdV(cv_vel, q)

        # Current field values at QP
        u_q  = function_value(cv_vel,  q, u_e)   # Vec{2}
        ∇u_q = function_gradient(cv_vel, q, u_e) # Tensor{2,2}: [i,j] = ∂_j u_i
        p_q  = function_value(cv_pres, q, p_e)   # Float64

        # ── Velocity rows ─────────────────────────────────────────────
        for i in 1:n_vel
            ri     = dof_range_u[i]
            φᵢ    = shape_value(cv_vel,     q, i)   # Vec{2}
            ∇φᵢ   = shape_gradient(cv_vel,  q, i)   # Tensor{2,2}
            divφᵢ = tr(∇φᵢ)                          # scalar div(φ_i) = ∂_x u_x + ∂_y u_y

            # Residual
            Re_e[ri] += (
                inv_Re * (∇φᵢ ⊡ ∇u_q)    # viscous:     ∇φᵢ : ∇u
                + φᵢ ⋅ (∇u_q ⋅ u_q)       # convective:  φᵢ · (u·∇u)
                - divφᵢ * p_q              # pressure:   −div(φᵢ) p
            ) * dΩ

            # vel–vel Jacobian block
            for j in 1:n_vel
                rj    = dof_range_u[j]
                φⱼ   = shape_value(cv_vel,    q, j)
                ∇φⱼ  = shape_gradient(cv_vel, q, j)

                Ke[ri, rj] += (
                    inv_Re * (∇φᵢ ⊡ ∇φⱼ)   # viscous tangent
                    + φᵢ ⋅ (∇u_q ⋅ φⱼ)      # (φ^j·∇)u :  convective by trial fn
                    + φᵢ ⋅ (∇φⱼ ⋅ u_q)      # (u·∇)φ^j :  advection  of trial fn
                ) * dΩ
            end

            # vel–pres Jacobian block (−G^T)
            for m in 1:n_pres
                rm   = dof_range_p[m]
                ψₘ  = shape_value(cv_pres, q, m)
                Ke[ri, rm] -= divφᵢ * ψₘ * dΩ   # −div(φᵢ) ψₘ
            end
        end

        # ── Pressure rows (divergence constraint) ─────────────────────
        divu_q = tr(∇u_q)   # ∇·u at QP
        for m in 1:n_pres
            rm  = dof_range_p[m]
            ψₘ = shape_value(cv_pres, q, m)

            # Residual
            Re_e[rm] += ψₘ * divu_q * dΩ

            # pres–vel Jacobian block (G)
            for j in 1:n_vel
                rj    = dof_range_u[j]
                divφⱼ = tr(shape_gradient(cv_vel, q, j))
                Ke[rm, rj] += ψₘ * divφⱼ * dΩ
            end
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Global assembly
# ─────────────────────────────────────────────────────────────────────────────

"""
    assemble_steady_nse!(K, R, s_full, fom, Re0)

Fill `K` (tangent) and `R` (residual) from the current full-DOF solution `s_full`.
Both `K` and `R` must be pre-allocated (e.g. via `allocate_matrix(dh)` and `zeros`).
"""
function assemble_steady_nse!(K, R, s_full, fom, Re0::Float64)
    fill!(K, 0.0)
    fill!(R, 0.0)

    n_dpc = ndofs_per_cell(fom.dh)
    Ke    = zeros(n_dpc, n_dpc)
    Re_e  = zeros(n_dpc)

    asm = start_assemble(K, R)

    for element in CellIterator(fom.dh)
        reinit!(fom.cv_vel,  element)
        reinit!(fom.cv_pres, element)

        dofs = celldofs(element)
        u_e  = s_full[dofs[fom.dof_range_u]]
        p_e  = s_full[dofs[fom.dof_range_p]]

        _assemble_element!(
            Ke, Re_e, u_e, p_e,
            fom.cv_vel, fom.cv_pres,
            fom.dof_range_u, fom.dof_range_p,
            Re0,
        )

        assemble!(asm, dofs, Ke, Re_e)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Newton iteration
# ─────────────────────────────────────────────────────────────────────────────

"""
    solve_steady_state(fom; Re0, tol = 1e-10, max_iter = 30) -> (u0_free, p0_free, s_full)

Newton–Raphson solver for the steady NSE at Reynolds number `Re0`.

Returns:
  u0_free  — free-DOF velocity perturbation base flow (length n_u_free)
  p0_free  — free-DOF pressure base flow             (length n_p_free)
  s0_full  — full-DOF solution vector (including prescribed BCs)

The initial guess is the exact Poiseuille profile at the inlet with zero in the
interior.  Convergence is monitored by the free-DOF residual ℓ²-norm.
"""
function solve_steady_state(fom; Re0::Float64, tol::Float64 = 1e-10, max_iter::Int = 30)
    N = ndofs(fom.dh)

    # Allocate once outside the loop
    K_full = allocate_matrix(fom.dh)
    R_full = zeros(N)

    # Initial guess: Poiseuille BCs everywhere, zero interior
    s_full = zeros(N)
    apply!(s_full, fom.ch_full)   # sets prescribed DOFs to Poiseuille / no-slip

    converged = false
    for iter in 1:max_iter
        assemble_steady_nse!(K_full, R_full, s_full, fom, Re0)

        # Apply homogeneous BCs to the UPDATE (δs = 0 at prescribed DOFs)
        apply_zero!(K_full, R_full, fom.ch_hom)

        # Restrict to free DOF subspace
        R_free = R_full[fom.free]
        res_norm = norm(R_free)
        @info "  Newton iter $iter (Re₀ = $Re0): ‖R‖ = $(round(res_norm; sigdigits=4))"

        if res_norm < tol
            converged = true
            break
        end

        K_free = K_full[fom.free, fom.free]
        F      = klu(K_free)
        δs     = -(F \ R_free)

        s_full[fom.free] .+= δs
    end

    converged || @warn "Newton did not converge in $max_iter iterations"

    # Split the free-DOF solution into velocity and pressure
    # Determine which free DOFs belong to each field
    u0_free, p0_free = _split_free_solution(s_full, fom)

    return u0_free, p0_free, s_full
end

"""
    _split_free_solution(s_full, fom) -> (u0_free, p0_free)

Extract velocity and pressure components from the full solution vector,
restricted to free DOFs.
"""
function _split_free_solution(s_full, fom)
    # Collect all global DOF indices belonging to :u (velocity) and :p (pressure)
    u_global = Set{Int}()
    p_global = Set{Int}()
    for element in CellIterator(fom.dh)
        dofs = celldofs(element)
        union!(u_global, dofs[fom.dof_range_u])
        union!(p_global, dofs[fom.dof_range_p])
    end

    # Free velocity DOFs (sorted global index → local free index)
    u_free_global = sort(collect(intersect(u_global, Set(fom.free))))
    p_free_global = sort(collect(intersect(p_global, Set(fom.free))))

    u0_free = s_full[u_free_global]
    p0_free = s_full[p_free_global]

    return u0_free, p0_free
end

# ─────────────────────────────────────────────────────────────────────────────
# Drag / lift validation (Turek–Schäfer benchmark values at Re = 20)
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_drag_lift(s_full, fom; Re0) -> (Cd, Cl)

Compute drag and lift coefficients by integrating the stress tensor over the
cylinder boundary.  Reference values (Turek–Schäfer benchmark, Re = 20):
  Cd ≈ 5.57,  Cl ≈ 0.011.
"""
function compute_drag_lift(s_full, fom; Re0::Float64)
    D   = 2.0 * _CYL_R
    U   = U_MEAN
    ref = 0.5 * U^2 * D   # reference force (per unit depth, ρ = 1)

    Fd = 0.0
    Fl = 0.0

    # Facet integration over cylinder boundary
    ip_face  = Lagrange{RefTriangle, 2}()^2
    ip_geo   = Lagrange{RefTriangle, 1}()
    qr_face  = FacetQuadratureRule{RefTriangle}(4)
    fv_vel   = FacetValues(qr_face, ip_face, ip_geo)

    ip_pres_f = Lagrange{RefTriangle, 1}()
    fv_pres   = FacetValues(qr_face, ip_pres_f, ip_geo)

    cyl_set = getfacetset(fom.grid, "Cylinder")

    for (cell_idx, local_facet_idx) in cyl_set
        element = CellIterator(fom.dh)
        # seek to the correct cell (Ferrite v1: use element index directly)
        # This is a simplified approach; a more efficient one uses a FacetIterator
        cell = CellCache(fom.dh)
        reinit!(cell, cell_idx)
        dofs = celldofs(cell)
        u_e  = s_full[dofs[fom.dof_range_u]]
        p_e  = s_full[dofs[fom.dof_range_p]]

        reinit!(fv_vel,  cell, local_facet_idx)
        reinit!(fv_pres, cell, local_facet_idx)

        for q in 1:getnquadpoints(fv_vel)
            dΓ   = getdetJdV(fv_vel, q)
            n    = getnormal(fv_vel, q)               # outward normal on cylinder

            ∇u_q = function_gradient(fv_vel, q, u_e)
            p_q  = function_value(fv_pres, q, p_e)

            # Stress tensor: σ = -p I + (1/Re₀)(∇u + ∇u^T)
            σ = -p_q * one(∇u_q) + (_CYL_D/Re0) * (∇u_q + transpose(∇u_q))
            traction = σ ⋅ n   # Vec{2}

            Fd += traction[1] * dΓ   # x-component → drag
            Fl += traction[2] * dΓ   # y-component → lift
        end
    end

    Cd = -2.0 * Fd / ref   # negative: Turek convention (force on fluid → on cylinder)
    Cl = -2.0 * Fl / ref

    return Cd, Cl
end
