"""
bias.jl — static equilibrium (bias point) and derived model constants (Phase 4).

Bias problem (V = V₀, exact):
	K x₀ = b Q₀²            →  x₀ = Q₀² x_b,  x_b = K⁻¹ b
	c₀ Q₀ (1 − gᵀx₀) = V₀   →  scalar cubic  c₀ Q₀ − c₀ (gᵀx_b) Q₀³ = V₀

Derived constants:
	ĉ  = c₀ (1 − gᵀx₀)               effective electrical stiffness
	β  = 2 Q₀ b                       linear electromechanical coupling
	ℓ₀ = K b/(2Q₀ bᵀb),  ℓ₁ = D b/(2Q₀ bᵀb),  ℓ₂ = M b/(2Q₀ bᵀb)
so that the charge proxy σ = ℓ₀ᵀu + ℓ₁ᵀu̇ + ℓ₂ᵀü satisfies σ = q + q²/(2Q₀) exactly
on trajectories of the shifted system (mech′)/(elec′).
"""

using LinearAlgebra

function bias_point(p, fe)
    xb = fe.K \ fe.b
    a = p.c0 * dot(fe.g, xb)               # cubic:  c₀Q₀ − a·Q₀³ = V₀

    # Newton on φ(Q₀) = c₀Q₀ − aQ₀³ − V₀, start from the uncoupled estimate
    Q0 = p.V0 / p.c0
    converged = false
    for _ in 1:100
        φ = p.c0 * Q0 - a * Q0^3 - p.V0
        dφ = p.c0 - 3a * Q0^2
        @assert dφ > 0 "bias Newton: dφ ≤ 0 — beyond pull-in, lower V0"
        Q0 -= φ / dφ
        if abs(p.c0 * Q0 - a * Q0^3 - p.V0) < 1e-14 * max(p.V0, 1e-12)
            converged = true
            break
        end
    end
    @assert converged "bias Newton did not converge"

    x0 = Q0^2 .* xb
    ĉ = p.c0 * (1 - dot(fe.g, x0))
    @assert ĉ > 0.5 * p.c0 "bias strain too large (near pull-in) — lower V0 or m_b"

    btb = dot(fe.b, fe.b)
    ℓ0 = (fe.K * fe.b) ./ (2Q0 * btb)
    ℓ1 = (fe.D * fe.b) ./ (2Q0 * btb)
    ℓ2 = (fe.M * fe.b) ./ (2Q0 * btb)

    return (; Q0, x0, ĉ, ℓ0, ℓ1, ℓ2)
end

"""
	bias_checks(p, fe, bp) — Phase 4 acceptance tests A1–A3. Throws on failure.
"""
function bias_checks(p, fe, bp)
    # A1: equilibrium residuals of the unshifted system (relative — K is ill-conditioned)
    r1 = norm(fe.K * bp.x0 .- fe.b .* bp.Q0^2)
    r2 = abs(p.c0 * bp.Q0 * (1 - dot(fe.g, bp.x0)) - p.V0)
    @assert r1 < 1e-8 * norm(fe.b) * bp.Q0^2 "A1: mech equilibrium residual $r1"
    @assert r2 < 1e-12 "A1: elec equilibrium residual $r2"

    # A2: exactness of the charge-proxy identity on a random consistent point.
    # Pick (u, u̇, q), then make ü consistent with (mech′); check σ = q + q²/(2Q₀).
    # Tolerance is relative to the term magnitudes (the ℓₖᵀ products cancel heavily).
    β = 2 * bp.Q0 .* fe.b
    for _ in 1:5
        u = 1e-3 .* randn(fe.n)
        u̇ = 1e-3 .* randn(fe.n)
        q = 1e-3 * randn()
        ü = fe.M \ (β .* q .+ fe.b .* q^2 .- fe.D * u̇ .- fe.K * u)
        σ = dot(bp.ℓ0, u) + dot(bp.ℓ1, u̇) + dot(bp.ℓ2, ü)
        scale = abs(dot(bp.ℓ0, u)) + abs(dot(bp.ℓ1, u̇)) + abs(dot(bp.ℓ2, ü)) + abs(q)
        @assert abs(σ - (q + q^2 / (2 * bp.Q0))) < 1e-12 * scale "A2: proxy identity violated"
    end

    # A3: bias strain in a sensible window
    s = dot(fe.g, bp.x0)
    @assert 0.005 ≤ s ≤ 0.2 "A3: bias strain gᵀx₀ = $s outside [0.005, 0.2] — tune V0/m_b"

    println("Phase 4 (bias) checks passed:  Q₀ = $(bp.Q0),  ĉ = $(bp.ĉ),  gᵀx₀ = $s")
    return nothing
end
