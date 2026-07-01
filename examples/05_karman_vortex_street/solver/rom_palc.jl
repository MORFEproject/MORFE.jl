"""
    rom_palc.jl — Pseudo-Arclength Continuation on the ROM limit-cycle branch.

The ROM limit cycle for the autonomous conjugate pair (z₁, z₂=conj(z₁), η') satisfies:

    ż₁ = R₁(z₁, z₂, η')  with  z₁(t) = ρ e^{iΩt}

Evaluating at the canonical phase z₁=ρ (real, positive) and z₂=ρ:

    R₁(ρ, ρ, η') = iΩρ   (purely imaginary × ρ at a periodic orbit)

So the periodic orbit condition is:

    F(ρ, η') = Re(R₁(ρ, ρ, η')) = 0        [growth rate vanishes]
    Ω(ρ, η') = Im(R₁(ρ, ρ, η')) / ρ        [angular frequency]

PALC continuation of the curve F(ρ, η') = 0:
  predictor  : (ρ, η') + Δs × τ  along the unit tangent τ
  corrector  : 2×2 Newton on [F; arclength condition] = 0
  new tangent: null vector of ∇F normalised consistently with previous τ
"""

"""
    _rom_R1(ρ, η, R) → ComplexF64

Evaluate the first component of the reduced dynamics at the canonical phase
z₁ = ρ (real), z₂ = ρ, η' = η.
"""
@inline function _rom_R1(ρ::Float64, η::Float64, R)
    z = ComplexF64[ρ, ρ, η]
    return evaluate(R.poly, z)[1]::ComplexF64
end

"""
    rom_po_residual(ρ, η, R) → Float64

Periodic-orbit residual F(ρ, η') = Re(R₁(ρ, ρ, η')).
Vanishes when (ρ, η') lies on the limit-cycle branch.
"""
rom_po_residual(ρ::Float64, η::Float64, R) = real(_rom_R1(ρ, η, R))

"""
    rom_po_frequency(ρ, η, R) → Float64

Angular frequency Ω = Im(R₁(ρ, ρ, η')) / ρ at the periodic orbit.
"""
rom_po_frequency(ρ::Float64, η::Float64, R) = imag(_rom_R1(ρ, η, R)) / ρ

"""
    rom_hopf_eta(R; ε=1e-6, η0=0.0, tol=1e-12, max_iter=50) → Float64

Find η′ such that the real part of the linearised ROM eigenvalue
λ(η′) = lim_{ρ→0} R₁(ρ, ρ, η′) / ρ vanishes — the true Hopf bifurcation
point. Re₀ is only the FOM's linear-expansion point, not necessarily the
critical Reynolds number, so this root is generally nonzero (`rom_po_residual`
at ρ=ε is, to leading order in ε, ε·Re(λ(η′))).
"""
function rom_hopf_eta(
    R;
    ε::Float64 = 1e-6,
    η0::Float64 = 0.0,
    tol::Float64 = 1e-12,
    max_iter::Int = 50,
)
    η = η0
    for _ in 1:max_iter
        F = rom_po_residual(ε, η, R)
        abs(F) < tol * ε && return η
        ε_η = 1e-6 * max(abs(η), 1e-6)
        dF = (rom_po_residual(ε, η + ε_η, R) - F) / ε_η
        abs(dF) < 1e-300 && break
        η -= F / dF
    end
    return η
end

"""
    rom_palc_tangent(ρ, η, R, τ_prev) → Vector{Float64}

Unit tangent to the branch F(ρ, η') = 0 at (ρ, η'), consistent with τ_prev.
"""
function rom_palc_tangent(ρ::Float64, η::Float64, R, τ_prev::Vector{Float64})
    ε_ρ = 1e-7 * max(ρ,   1.0)
    ε_η = 1e-7 * max(abs(η), 1e-8)
    F0   = rom_po_residual(ρ, η, R)
    dF_dρ = (rom_po_residual(ρ + ε_ρ, η, R) - F0) / ε_ρ
    dF_dη = (rom_po_residual(ρ, η + ε_η, R) - F0) / ε_η
    # tangent ⊥ gradient: rotate gradient 90°
    τ = [-dF_dη, dF_dρ]
    nrm = norm(τ)
    nrm < 1e-300 && return copy(τ_prev)   # degenerate: keep old
    τ ./= nrm
    dot(τ, τ_prev) < 0.0 && (τ .*= -1.0)  # consistent orientation
    return τ
end

"""
    rom_palc_step(ρ, η, τ, Δs, R; tol=1e-10, max_iter=20)
    → (ρ_new, η_new, T_new, τ_new, n_iter, converged)

One pseudo-arclength continuation step:
  predictor  : move Δs along the branch tangent τ
  corrector  : Newton on [F(ρ,η); τ·(p−last)−Δs] = 0

Returns the new branch point (ρ_new, η_new), the ROM period T_new,
the updated unit tangent τ_new, the number of Newton iterations, and
a convergence flag.
"""
function rom_palc_step(
    ρ::Float64, η::Float64, τ::Vector{Float64}, Δs::Float64, R;
    tol::Float64     = 1e-10,
    max_iter::Int    = 20,
)
    # ── Predictor ────────────────────────────────────────────────────────────
    ρ_p = max(ρ + Δs * τ[1], 1e-12)
    η_p = η + Δs * τ[2]

    # ── Newton corrector ─────────────────────────────────────────────────────
    converged = false
    n_iter    = 0

    for iter in 1:max_iter
        n_iter = iter
        F = rom_po_residual(ρ_p, η_p, R)
        N = τ[1] * (ρ_p - ρ) + τ[2] * (η_p - η) - Δs

        if abs(F) < tol && abs(N) < tol
            converged = true
            break
        end

        # 2×2 Jacobian (F row by finite differences, N row is exact)
        ε_ρ = 1e-7 * max(ρ_p,   1.0)
        ε_η = 1e-7 * max(abs(η_p), 1e-8)
        dF_dρ = (rom_po_residual(ρ_p + ε_ρ, η_p, R) - F) / ε_ρ
        dF_dη = (rom_po_residual(ρ_p, η_p + ε_η, R) - F) / ε_η

        J = @SMatrix [dF_dρ  dF_dη
                      τ[1]   τ[2] ]
        δ = J \ SVector(-F, -N)

        ρ_p = max(ρ_p + δ[1], 1e-12)
        η_p = η_p + δ[2]
    end

    # ── Period and new tangent ────────────────────────────────────────────────
    Ω     = rom_po_frequency(ρ_p, η_p, R)
    T_new = 2π / abs(Ω)
    τ_new = rom_palc_tangent(ρ_p, η_p, R, τ)

    return ρ_p, η_p, T_new, τ_new, n_iter, converged
end
