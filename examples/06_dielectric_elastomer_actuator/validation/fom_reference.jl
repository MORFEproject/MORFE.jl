"""
validation/fom_reference.jl — ground-truth integration of the coupled (u, q) system.

Integrates the SHIFTED coupled equations literally (never went through the third-order
closure — independent ground truth):

	M ü = 2Q₀ b q + b q² − D u̇ − K u
	R q̇ = v_a cos(Ωt) + c₀Q₀ (gᵀu) + c₀ q (gᵀu) − ĉ q

State s = (u, u̇, q) ∈ ℝ^{2n+1}. The Kelvin–Voigt model makes the high FE modes very
stiff, so the linear part is treated with Crank–Nicolson (prefactored) and the mild
rank-one nonlinearity with AB2 — see `imex_cn_ab2` in rom_utils.jl.
"""

using LinearAlgebra

"""
	fom_linear_operator(p, fe, bp) -> Matrix (2n+1) × (2n+1)
"""
function fom_linear_operator(p, fe, bp)
    n = fe.n
    Minv = inv(fe.M)
    L = zeros(2n + 1, 2n + 1)
    L[1:n, (n + 1):2n] .= I(n)
    L[(n + 1):2n, 1:n] .= -Minv * fe.K
    L[(n + 1):2n, (n + 1):2n] .= -Minv * fe.D
    L[(n + 1):2n, 2n + 1] .= Minv * (2 * bp.Q0 .* fe.b)
    L[2n + 1, 1:n] .= (p.c0 * bp.Q0 / p.R) .* fe.g
    L[2n + 1, 2n + 1] = -bp.ĉ / p.R
    return L
end

"""
	integrate_fom(p, fe, bp; Ω, T_end, dt, s0 = zeros, forced = true) -> (ts, wtip)

Returns times and tip displacement u_tip(t) (shifted coordinates, i.e. about the bias).
"""
function integrate_fom(p, fe, bp; Ω, T_end, dt, s0 = nothing, forced::Bool = true)
    n = fe.n
    Lmat = fom_linear_operator(p, fe, bp)
    Mfac = lu(fe.M)
    function Nfun(s, t)
        u = view(s, 1:n)
        q = s[2n + 1]
        γ = dot(fe.g, u)
        v = forced ? p.v_a * cos(Ω * t) : 0.0
        out = zeros(2n + 1)
        out[(n + 1):2n] .= Mfac \ (fe.b .* q^2)
        out[2n + 1] = (v + p.c0 * q * γ) / p.R
        return out
    end
    s0 === nothing && (s0 = zeros(2n + 1))
    ts, S = imex_cn_ab2(Lmat, Nfun, s0, 0.0, T_end, dt)
    wtip = [s[fe.idx_wtip] for s in S]
    return ts, wtip, S
end

"""
	integrate_thirdorder(p, fe, bp; Ω, T_end, dt, s0 = nothing, forced = true,
						 include_cubic_forcing = true) -> (ts, wtip)

Direct integration of the CLOSED third-order system (state (u, u̇, ü) ∈ ℝ^{3n}) —
validates the closure independently of MORFE.
"""
function integrate_thirdorder(p, fe, bp; Ω, T_end, dt, s0 = nothing, forced::Bool = true,
        include_cubic_forcing::Bool = true)
    n = fe.n
    B0, B1, B2, B3 = build_linear_matrices(p, fe, bp)
    B3fac = lu(B3)
    Lmat = zeros(3n, 3n)
    Lmat[1:n, (n + 1):2n] .= I(n)
    Lmat[(n + 1):2n, (2n + 1):3n] .= I(n)
    Lmat[(2n + 1):3n, 1:n] .= -(B3fac \ B0)
    Lmat[(2n + 1):3n, (n + 1):2n] .= -(B3fac \ B1)
    Lmat[(2n + 1):3n, (2n + 1):3n] .= -(B3fac \ B2)
    function Nfun(s, t)
        u = view(s, 1:n)
        u̇ = view(s, (n + 1):2n)
        ü = view(s, (2n + 1):3n)
        Fv = F_thirdorder(u, u̇, ü, t, p, fe, bp; Ω, forced, include_cubic_forcing)
        out = zeros(3n)
        out[(2n + 1):3n] .= B3fac \ Fv
        return out
    end
    if s0 === nothing
        s0 = zeros(3n)        # consistent with (u,u̇,q) = 0:  ü(0) = 0
    end
    ts, S = imex_cn_ab2(Lmat, Nfun, s0, 0.0, T_end, dt)
    wtip = [s[fe.idx_wtip] for s in S]
    return ts, wtip, S
end
