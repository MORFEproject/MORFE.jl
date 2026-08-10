"""
coupling_terms.jl — third-order model assembly (Phase 5).

Implements the closed third-order system (implementation_plan_detailed.md §1.5):

	B₃ u⁽³⁾ + B₂ ü + B₁ u̇ + B₀ u = F(u, u̇, ü, v)

	B₃ = R M,  B₂ = R D + ĉ M,  B₁ = R K + ĉ D,  B₀ = ĉ K − 2c₀Q₀² (b gᵀ)

	F =  2Q₀ b v  +  2 b σv  +  4c₀Q₀ b σγ  −  ĉ b σ²  +  (ĉ/Q₀) b σ³  −  (1/Q₀) b σ²v

with σ = ℓ₀ᵀu + ℓ₁ᵀu̇ + ℓ₂ᵀü, γ = gᵀu, and v(t) = v_a cos Ωt = (v_a/2)(r₁ + r₂).

Each monomial in (u, u̇, ü, r) becomes one `MultilinearMap` (term catalogue §6.3–6.5).
Multinomial factors of the σ-power expansions are included inside each `f!`
(diagonal-evaluation convention). All maps accumulate `scalar .* b` into `res`.

NOTE: the functionals ℓ₀, ℓ₁, ℓ₂, g, b are real vectors, so `dot(ℓ, x)` performs no
unwanted conjugation even when `x` is complex (as it is inside the cohomological solve).
"""

using LinearAlgebra

function build_linear_matrices(p, fe, bp)
    B3 = p.R .* fe.M
    B2 = p.R .* fe.D .+ bp.ĉ .* fe.M
    B1 = p.R .* fe.K .+ bp.ĉ .* fe.D
    B0 = bp.ĉ .* fe.K .- (2 * p.c0 * bp.Q0^2) .* (fe.b * fe.g')  # nonsymmetric rank-one
    return B0, B1, B2, B3
end

function build_nonlinear_terms(p, fe, bp; forced::Bool, include_cubic_forcing::Bool = true)
    bvec = fe.b
    gv = fe.g
    ℓ0, ℓ1, ℓ2 = bp.ℓ0, bp.ℓ1, bp.ℓ2
    c2 = 4 * p.c0 * bp.Q0          # σγ coefficient
    cs = bp.ĉ                      # σ² coefficient (with minus sign in the maps)
    c3 = bp.ĉ / bp.Q0              # σ³ coefficient
    cf0 = bp.Q0 * p.v_a            # 2Q₀b·(v_a/2)
    cf1 = p.v_a                    # 2b σ·(v_a/2)
    cf2 = -p.v_a / (2 * bp.Q0)     # −(1/Q₀)b σ²·(v_a/2)

    # ── degree 2, autonomous (6 maps): + c2·σγ − cs·σ² ────────────────────────
    q1! = (res, a1, a2) -> (
        res .+= ((c2 / 2) * (dot(ℓ0, a1) * dot(gv, a2) +
                             dot(gv, a1) * dot(ℓ0, a2)) -
                 cs * dot(ℓ0, a1) * dot(ℓ0, a2)) .* bvec;
        nothing)
    q2! = (res, a, y) -> (
        res .+= (c2 * dot(gv, a) * dot(ℓ1, y) -
                 2cs * dot(ℓ0, a) * dot(ℓ1, y)) .* bvec;
        nothing)
    q3! = (res, a, z) -> (
        res .+= (c2 * dot(gv, a) * dot(ℓ2, z) -
                 2cs * dot(ℓ0, a) * dot(ℓ2, z)) .* bvec;
        nothing)
    q4! = (res, y1, y2) -> (res .+= (-cs * dot(ℓ1, y1) * dot(ℓ1, y2)) .* bvec; nothing)
    q5! = (res, y, z) -> (res .+= (-2cs * dot(ℓ1, y) * dot(ℓ2, z)) .* bvec; nothing)
    q6! = (res, z1, z2) -> (res .+= (-cs * dot(ℓ2, z1) * dot(ℓ2, z2)) .* bvec; nothing)

    # ── degree 3, autonomous (10 maps): + c3·σ³ ───────────────────────────────
    t1! = (res, a1, a2, a3) -> (
        res .+= (c3 * dot(ℓ0, a1) * dot(ℓ0, a2) * dot(ℓ0, a3)) .* bvec; nothing)
    t2! = (res, a1, a2, y) -> (
        res .+= (3c3 * dot(ℓ0, a1) * dot(ℓ0, a2) * dot(ℓ1, y)) .* bvec; nothing)
    t3! = (res, a1, a2, z) -> (
        res .+= (3c3 * dot(ℓ0, a1) * dot(ℓ0, a2) * dot(ℓ2, z)) .* bvec; nothing)
    t4! = (res, a, y1, y2) -> (
        res .+= (3c3 * dot(ℓ0, a) * dot(ℓ1, y1) * dot(ℓ1, y2)) .* bvec; nothing)
    t5! = (res, a, y, z) -> (
        res .+= (6c3 * dot(ℓ0, a) * dot(ℓ1, y) * dot(ℓ2, z)) .* bvec; nothing)
    t6! = (res, a, z1, z2) -> (
        res .+= (3c3 * dot(ℓ0, a) * dot(ℓ2, z1) * dot(ℓ2, z2)) .* bvec; nothing)
    t7! = (res, y1, y2, y3) -> (
        res .+= (c3 * dot(ℓ1, y1) * dot(ℓ1, y2) * dot(ℓ1, y3)) .* bvec; nothing)
    t8! = (res, y1, y2, z) -> (
        res .+= (3c3 * dot(ℓ1, y1) * dot(ℓ1, y2) * dot(ℓ2, z)) .* bvec; nothing)
    t9! = (res, y, z1, z2) -> (
        res .+= (3c3 * dot(ℓ1, y) * dot(ℓ2, z1) * dot(ℓ2, z2)) .* bvec; nothing)
    t10! = (res, z1, z2, z3) -> (
        res .+= (c3 * dot(ℓ2, z1) * dot(ℓ2, z2) * dot(ℓ2, z3)) .* bvec; nothing)

    terms = AbstractMultilinearMap{3}[
        MultilinearMap(q1!, (2, 0, 0); fully_asymmetric = false),
        MultilinearMap(q2!, (1, 1, 0)),
        MultilinearMap(q3!, (1, 0, 1)),
        MultilinearMap(q4!, (0, 2, 0); fully_asymmetric = false),
        MultilinearMap(q5!, (0, 1, 1)),
        MultilinearMap(q6!, (0, 0, 2); fully_asymmetric = false),
        MultilinearMap(t1!, (3, 0, 0); fully_asymmetric = false),
        MultilinearMap(t2!, (2, 1, 0); fully_asymmetric = false),
        MultilinearMap(t3!, (2, 0, 1); fully_asymmetric = false),
        MultilinearMap(t4!, (1, 2, 0); fully_asymmetric = false),
        MultilinearMap(t5!, (1, 1, 1)),
        MultilinearMap(t6!, (1, 0, 2); fully_asymmetric = false),
        MultilinearMap(t7!, (0, 3, 0); fully_asymmetric = false),
        MultilinearMap(t8!, (0, 2, 1); fully_asymmetric = false),
        MultilinearMap(t9!, (0, 1, 2); fully_asymmetric = false),
        MultilinearMap(t10!, (0, 0, 3); fully_asymmetric = false)
    ]

    if forced
        # ── degree 1, external (1 map): 2Q₀b·v ───────────────────────────────
        f0! = (res, r) -> (res .+= (cf0 * sum(r)) .* bvec; nothing)
        push!(terms, MultilinearMap(f0!, (0, 0, 0), 1))

        # ── degree 2, external (3 maps): 2b·σv ───────────────────────────────
        f1a! = (res, a, r) -> (res .+= (cf1 * dot(ℓ0, a) * sum(r)) .* bvec; nothing)
        f1b! = (res, y, r) -> (res .+= (cf1 * dot(ℓ1, y) * sum(r)) .* bvec; nothing)
        f1c! = (res, z, r) -> (res .+= (cf1 * dot(ℓ2, z) * sum(r)) .* bvec; nothing)
        push!(terms, MultilinearMap(f1a!, (1, 0, 0), 1))
        push!(terms, MultilinearMap(f1b!, (0, 1, 0), 1))
        push!(terms, MultilinearMap(f1c!, (0, 0, 1), 1))

        # ── degree 3, external (6 maps): −(1/Q₀)b·σ²v (exactness terms) ──────
        if include_cubic_forcing
            f2a! = (res, a1, a2, r) -> (
                res .+= (cf2 * dot(ℓ0, a1) * dot(ℓ0, a2) * sum(r)) .* bvec; nothing)
            f2b! = (res, a, y, r) -> (
                res .+= (2cf2 * dot(ℓ0, a) * dot(ℓ1, y) * sum(r)) .* bvec; nothing)
            f2c! = (res, a, z, r) -> (
                res .+= (2cf2 * dot(ℓ0, a) * dot(ℓ2, z) * sum(r)) .* bvec; nothing)
            f2d! = (res, y1, y2, r) -> (
                res .+= (cf2 * dot(ℓ1, y1) * dot(ℓ1, y2) * sum(r)) .* bvec; nothing)
            f2e! = (res, y, z, r) -> (
                res .+= (2cf2 * dot(ℓ1, y) * dot(ℓ2, z) * sum(r)) .* bvec; nothing)
            f2f! = (res, z1, z2, r) -> (
                res .+= (cf2 * dot(ℓ2, z1) * dot(ℓ2, z2) * sum(r)) .* bvec; nothing)
            push!(terms, MultilinearMap(f2a!, (2, 0, 0), 1; fully_asymmetric = false))
            push!(terms, MultilinearMap(f2b!, (1, 1, 0), 1))
            push!(terms, MultilinearMap(f2c!, (1, 0, 1), 1))
            push!(terms, MultilinearMap(f2d!, (0, 2, 0), 1; fully_asymmetric = false))
            push!(terms, MultilinearMap(f2e!, (0, 1, 1), 1))
            push!(terms, MultilinearMap(f2f!, (0, 0, 2), 1; fully_asymmetric = false))
        end
    end

    return (terms...,)
end

"""
	build_model(p, fe, bp; forced, Ω = 0.0, include_cubic_forcing = true) -> NthOrderModel

`forced = false`: autonomous ORD = 3 model (NVAR = 2 reduction path).
`forced = true` : adds the external system ṙ = diag(+iΩ, −iΩ) r and all forcing maps.
"""
function build_model(p, fe, bp; forced::Bool, Ω::Float64 = 0.0,
        include_cubic_forcing::Bool = true)
    B0, B1, B2, B3 = build_linear_matrices(p, fe, bp)
    terms = build_nonlinear_terms(p, fe, bp; forced, include_cubic_forcing)
    if forced
        @assert Ω > 0 "forced model requires Ω > 0"
        ext = ExternalSystem((im * Ω, -im * Ω))
        return NthOrderModel((B0, B1, B2, B3), terms, ext)
    else
        return NthOrderModel((B0, B1, B2, B3), terms)
    end
end

"""
	F_thirdorder(u, u̇, ü, t, p, fe, bp; Ω, forced, include_cubic_forcing) -> Vector

Independent literal re-implementation of the right-hand side F (used by the acceptance
checks and the validation integrators — deliberately NOT built from the MultilinearMaps).
"""
function F_thirdorder(u, u̇, ü, t, p, fe, bp; Ω = 0.0, forced::Bool = true,
        include_cubic_forcing::Bool = true)
    σ = dot(bp.ℓ0, u) + dot(bp.ℓ1, u̇) + dot(bp.ℓ2, ü)
    γ = dot(fe.g, u)
    v = forced ? p.v_a * cos(Ω * t) : 0.0
    s = 2 * bp.Q0 * v + 2 * σ * v + 4 * p.c0 * bp.Q0 * σ * γ - bp.ĉ * σ^2 +
        (bp.ĉ / bp.Q0) * σ^3
    if include_cubic_forcing
        s -= (1 / bp.Q0) * σ^2 * v
    end
    return s .* fe.b
end

"""
	coupling_checks(p, fe, bp) — Phase 5 acceptance tests A1–A3. Throws on failure.
"""
function coupling_checks(p, fe, bp)
    B0, B1, B2, B3 = build_linear_matrices(p, fe, bp)

    # A1: both models construct (and the implicit-symmetry @info must not fire —
    # watch the terminal; every multiindex entry > 1 carries fully_asymmetric = false).
    model_a = build_model(p, fe, bp; forced = false)
    Ωtest = p.ω1_target
    model_f = build_model(p, fe, bp; forced = true, Ω = Ωtest)

    # A2: MultilinearMaps ≡ literal F (at t = 0, where v = v_a·cos(0) = (v_a/2)·sum((1,1)))
    r = ComplexF64[1.0, 1.0]
    for _ in 1:5
        u = 1e-2 .* randn(fe.n)
        u̇ = 1e-2 .* randn(fe.n)
        ü = 1e-2 .* randn(fe.n)
        res = zeros(ComplexF64, fe.n)
        for deg in 1:3
            evaluate_nonlinear_terms!(res, model_f, deg, (u, u̇, ü), r)
        end
        Flit = F_thirdorder(u, u̇, ü, 0.0, p, fe, bp; Ω = Ωtest, forced = true)
        @assert norm(res .- Flit) < 1e-12 * max(norm(Flit), 1e-12) "A2: maps ≠ literal F"
        # autonomous model: forcing terms absent
        res .= 0
        for deg in 2:3
            evaluate_nonlinear_terms!(res, model_a, deg, (u, u̇, ü))
        end
        Flit_a = F_thirdorder(u, u̇, ü, 0.0, p, fe, bp; forced = false)
        @assert norm(res .- Flit_a) < 1e-12 * max(norm(Flit_a), 1e-12) "A2(auton) failed"
    end

    # A3(i): closure is algebraically exact BEFORE the proxy-series truncation.
    # Build a consistent state, evaluate the exact RHS scalar in the true charge q,
    #   s_exact = 2Q₀v + 2qv + 4c₀Q₀qγ − ĉq² + 2c₀q²γ,
    # and check that B₃u⁽³⁾ + B₂ü + B₁u̇ + B₀u = s_exact·b reproduces d/dt(mech′) to roundoff.
    β = 2 * bp.Q0 .* fe.b
    s_exact(q, γ, v) = 2 * bp.Q0 * v + 2 * q * v + 4 * p.c0 * bp.Q0 * q * γ -
                       bp.ĉ * q^2 + 2 * p.c0 * q^2 * γ
    u_dir, v_dir = randn(fe.n), randn(fe.n)
    q_dir = randn()
    let ε = 1e-2, vlt = p.v_a
        u = ε .* u_dir
        u̇ = ε .* v_dir
        q = ε * q_dir
        γ = dot(fe.g, u)
        ü = fe.M \ (β .* q .+ fe.b .* q^2 .- fe.D * u̇ .- fe.K * u)
        q̇ = (vlt + p.c0 * bp.Q0 * γ + p.c0 * q * γ - bp.ĉ * q) / p.R
        u3_ref = fe.M \ (β .* q̇ .+ 2 .* fe.b .* (q * q̇) .- fe.D * ü .- fe.K * u̇)
        u3_full = B3 \ (s_exact(q, γ, vlt) .* fe.b .- B2 * ü .- B1 * u̇ .- B0 * u)
        res = norm(u3_full .- u3_ref) / max(norm(u3_ref), 1e-300)
        @assert res < 1e-11 "A3(i): closure not exact pre-truncation (res = $res)"
    end

    # A3(ii): truncation residual equals its EXACT degree-6 polynomial in q.
    #   d(q) ≡ s_tr − s_exact = −(v_a/Q₀²)q³ + (5ĉ/4Q₀² − v_a/4Q₀³)q⁴ + (3ĉ/4Q₀³)q⁵ + (ĉ/8Q₀⁴)q⁶
    # Proof: σ = q + q²/(2Q₀) exactly (bias.jl §A2); expand σ², σ³ and observe γ drops out.
    # We use σ directly from the identity rather than from the ü path, because the K-stiffness
    # cancellation in ℓ₀ᵀu + ℓ₂ᵀü loses ~7 digits for a 50-element mesh (K entries ~h⁻³).
    # The proxy identity σ = q + q²/(2Q₀) is verified separately by bias_checks A2.
    for ε in (0.05, 0.025)
        q = ε * q_dir
        σ = q + q^2 / (2 * bp.Q0)     # exact proxy identity, no cancellation
        s_tr = 2 * bp.Q0 * p.v_a + 2 * σ * p.v_a -
               bp.ĉ * σ^2 + (bp.ĉ / bp.Q0) * σ^3 - (1 / bp.Q0) * σ^2 * p.v_a
        actual = s_tr - s_exact(q, 0.0, p.v_a)    # γ = 0 (drops out analytically)
        predicted = -(p.v_a / bp.Q0^2) * q^3 +
                    (5 * bp.ĉ / (4 * bp.Q0^2) - p.v_a / (4 * bp.Q0^3)) * q^4 +
                    (3 * bp.ĉ / (4 * bp.Q0^3)) * q^5 +
                    (bp.ĉ / (8 * bp.Q0^4)) * q^6
        tol_scale = abs(p.v_a / bp.Q0^2) * abs(q)^3 + (5 * bp.ĉ / (4 * bp.Q0^2)) * abs(q)^4
        @assert abs(actual - predicted) < 1e-10 * max(tol_scale, 1e-30) "A3(ii): truncation formula mismatch at ε=$ε (rel=$(round(abs(actual-predicted)/max(tol_scale,1e-300), sigdigits=3)))"
    end

    println("Phase 5 (coupling) checks passed:  s_tr − s_exact = exact degree-6 polynomial in q ✓")
    return nothing
end
