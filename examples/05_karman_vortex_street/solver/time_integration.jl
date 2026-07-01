"""
    time_integration.jl — IMEX θ-method FOM integrator for the cylinder-flow DPIM demo.

Integrates the perturbation NSE

    B₁ ṡ = −B₀ s + f₂(s,s) + η′ K_visc s + η′ h₀_vec

in the free_dpim DOF subspace using an implicit-explicit θ-method:
  • Linear part (−B₀ s) : implicit with weight θ
  • Nonlinear part N(s)  : explicit (evaluated at current step)

LHS matrix L = B₁/Δt + θ B₀ is constant → factorised once with KLU.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Element-level assembly of f₂(s, s) = −∫ φ · (u·∇u) dΩ
# ─────────────────────────────────────────────────────────────────────────────

"""
    eval_perturbation_convection!(accum, s_free, fom)

Assemble the quadratic perturbation convection into `accum` (length n_free_dpim):

    accum[k] += −∫_Ω  φᵢ · (∇u · u)  dΩ     (velocity DOFs k only)

where u is the velocity part of `s_free` extracted element-by-element via
`fom.free_to_local_dpim`.  `accum` is zeroed on entry.
"""
function eval_perturbation_convection!(
    accum::Vector{Float64},
    s_free::Vector{Float64},
    fom,
)
    fill!(accum, 0.0)

    n_dpc = ndofs_per_cell(fom.dh)
    n_vel = fom.n_vel_dofs_per_cell

    Fe  = zeros(Float64, n_dpc)
    u_e = zeros(Float64, n_vel)

    for element in CellIterator(fom.dh)
        reinit!(fom.cv_vel, element)
        dofs     = celldofs(element)
        vel_dofs = dofs[fom.dof_range_u]   # global velocity DOF indices for this cell

        # Extract velocity values from the free-DOF state vector
        fill!(u_e, 0.0)
        for (i, d) in enumerate(vel_dofs)
            li = get(fom.free_to_local_dpim, d, 0)
            li != 0 && (u_e[i] = s_free[li])
        end

        fill!(Fe, 0.0)
        for q in 1:getnquadpoints(fom.cv_vel)
            dΩ   = getdetJdV(fom.cv_vel, q)
            u_q  = function_value(fom.cv_vel,    q, u_e)   # Vec{2,Float64}
            ∇u_q = function_gradient(fom.cv_vel, q, u_e)   # Tensor{2,2,Float64}

            # f₂(s,s) = −(u·∇)u;  in Ferrite: (∇u_q ⋅ u_q)[i] = Σⱼ ∂_j uᵢ · uⱼ = (u·∇u)ᵢ
            conv = -(∇u_q ⋅ u_q)   # Vec{2,Float64}

            for i in 1:n_vel
                ri = fom.dof_range_u[i]
                φᵢ = shape_value(fom.cv_vel, q, i)   # Vec{2,Float64}
                Fe[ri] += (φᵢ ⋅ conv) * dΩ
            end
        end

        # Scatter element residual into free-DOF accumulator
        for (r, d) in enumerate(dofs)
            li = get(fom.free_to_local_dpim, d, 0)
            li != 0 && (accum[li] += Fe[r])
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# IMEX θ-method integrator
# ─────────────────────────────────────────────────────────────────────────────

"""
    integrate_fom(s_init, η_prime, T, fom, B₀, B₁, K_visc, h₀_vec;
                  Δt=1e-3, θ=0.5, verbose=false, n_spinup=10, γ_spinup=5.0)
    → (s_T, E_M_norm, X_max)

Integrate the perturbation NSE from t=0 to t=T using an IMEX θ-method with
an optional damped spin-up phase.

## Arguments
- `s_init`   : initial perturbation state, `Vector{Float64}` of length `fom.n_free_dpim`
- `η_prime`  : parameter deviation `1/Re − 1/Re₀`
- `T`        : integration period (seconds)
- `fom`      : FEM setup named tuple from `setup_fem()`
- `B₀`, `B₁`: sparse linear operators from `assemble_linear_operators()`
- `K_visc`   : viscosity stiffness scaled by `−_CYL_D`
- `h₀_vec`   : base-flow forcing vector `K_visc * s₀_full[fom.free_dpim]`

## Keyword arguments
- `Δt`       : time-step size (default `1e-3`)
- `θ`        : implicit weight (0.5 = Crank-Nicolson, 1 = backward Euler)
- `verbose`  : print progress if `true`
- `n_spinup` : number of spin-up orbits with artificial damping before measurement (default 10)
- `γ_spinup` : artificial damping coefficient for spin-up; adds `−γ B₁ s` to shift all
               eigenvalues leftward by γ, suppressing transients (default 5.0)

## Returns
- `s_T`      : state at t=T of the **measurement** orbit
- `E_M_norm` : `sqrt((s_T − s_start)ᵀ B₁ (s_T − s_start))` — mass-weighted periodicity error
- `X_max`    : `max_t sqrt(s(t)ᵀ B₁ s(t))` over the measurement orbit
"""
function integrate_fom(
    s_init  ::Vector{Float64},
    η_prime ::Float64,
    T       ::Float64,
    fom,
    B₀, B₁,
    K_visc,
    h₀_vec  ::Vector{Float64};
    Δt      ::Float64 = 1e-3,
    θ       ::Float64 = 0.5,
    verbose ::Bool    = false,
    n_spinup::Int     = 10,
    γ_spinup::Float64 = 5.0,
)
    n_steps  = max(1, round(Int, T / Δt))
    Δt_exact = T / n_steps

    inv_dt = 1.0 / Δt_exact
    A_imp  = B₀ .- η_prime .* K_visc   # linear implicit operator (K_visc moved in)

    # ── Pre-allocated work buffers ────────────────────────────────────────────
    s   = copy(s_init)
    rhs = similar(s)
    f2  = similar(s)
    tmp = similar(s)

    # ── Spin-up: n_spinup orbits with artificial damping −γ B₁ s ─────────────
    # Adding γ B₁ to the implicit operator shifts all eigenvalues left by γ,
    # suppressing transients without changing the steady periodic solution's shape
    # (it shifts the limit cycle slightly; the measurement orbit corrects for this).
    if n_spinup > 0
        A_damp     = A_imp .+ γ_spinup .* B₁
        LHS_damp   = inv_dt .* B₁ .+ θ           .* A_damp
        RHS_M_damp = inv_dt .* B₁ .- (1.0 - θ)  .* A_damp
        L_klu_damp = klu(LHS_damp)

        for orbit in 1:n_spinup
            verbose && @printf("  spin-up orbit %d / %d ...\n", orbit, n_spinup)
            for step in 1:n_steps
                eval_perturbation_convection!(f2, s, fom)
                mul!(rhs, RHS_M_damp, s)
                axpy!(1.0,     f2,     rhs)
                axpy!(η_prime, h₀_vec, rhs)
                ldiv!(s, L_klu_damp, rhs)
                if !isfinite(dot(s, s))
                    @warn "integrate_fom: non-finite state during spin-up orbit " *
                          "$orbit/$n_spinup at step $step/$n_steps — try larger γ_spinup or smaller Δt"
                    return s, NaN, NaN
                end
            end
        end
    end

    # ── Measurement orbit: no artificial damping ──────────────────────────────
    LHS_clean   = inv_dt .* B₁ .+ θ           .* A_imp
    RHS_M_clean = inv_dt .* B₁ .- (1.0 - θ)  .* A_imp
    L_klu_clean = klu(LHS_clean)

    s_start = copy(s)   # start of measurement orbit (after spin-up)

    mul!(tmp, B₁, s)
    X_max = sqrt(max(dot(s, tmp), 0.0))

    for step in 1:n_steps
        verbose && step % 100 == 0 &&
            @printf("  measurement: step %d / %d  (t = %.4f s)\n",
                    step, n_steps, step * Δt_exact)

        eval_perturbation_convection!(f2, s, fom)
        mul!(rhs, RHS_M_clean, s)
        axpy!(1.0,     f2,     rhs)
        axpy!(η_prime, h₀_vec, rhs)
        ldiv!(s, L_klu_clean, rhs)

        if !isfinite(dot(s, s))
            @warn "integrate_fom: non-finite state during measurement orbit at step $step/$n_steps"
            return s, NaN, NaN
        end

        mul!(tmp, B₁, s)
        X_max = max(X_max, sqrt(max(dot(s, tmp), 0.0)))
    end

    # ── Periodicity error ‖s(T) − s_start‖_M ────────────────────────────────
    E = s .- s_start
    mul!(tmp, B₁, E)
    E_norm = sqrt(max(dot(E, tmp), 0.0))

    return s, E_norm, X_max
end
