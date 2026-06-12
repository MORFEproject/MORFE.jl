"""
hermite_beam.jl — cubic-Hermite Euler–Bernoulli FE assembly for the DEA cantilever.

Produces (Phase 3 of the implementation plan):
- `M`, `K`  : mass / bending stiffness (dense, free DOFs only)
- `D`       : Kelvin–Voigt damping, D = η_over_E · K
- `b`       : electrostatic load vector of the linearly tapered electrode
			  m_b(x) = m_b(1 − x/L); assembled as the (globally identical) consistent
			  load vector of the uniform transverse load p = m_b/L
- `b_weak`  : same vector assembled directly from the weak form ∫ m_b N′ dx (check A5)
- `g`       : taper-weighted mean-curvature functional; collapses exactly to
			  gᵀx = 2α_c·w_tip/L² (tip displacement functional)

DOF numbering: node i (0-based) → global DOFs (2i+1, 2i+2) = (w_i, θ_i).
Node 0 clamped → free DOFs are 3:2(n_elem+1); n = 2·n_elem.
The last two free DOFs are (w_tip, θ_tip); idx_wtip = n − 1.
"""

using LinearAlgebra

hermite_Ke(EI, h) = (EI / h^3) .* [
	12.0   6h    -12.0   6h;
	6h     4h^2  -6h     2h^2;
	-12.0  -6h   12.0    -6h;
	6h     2h^2  -6h     4h^2]

hermite_Me(ρA, h) = (ρA * h / 420) .* [
	156.0   22h     54.0    -13h;
	22h     4h^2    13h     -3h^2;
	54.0    13h     156.0   -22h;
	-13h    -3h^2   -22h    4h^2]

function assemble_beam(p)
	n_elem = p.n_elem
	L, h = p.L, p.L / p.n_elem
	ndof_full = 2 * (n_elem + 1)

	Kf = zeros(ndof_full, ndof_full)
	Mf = zeros(ndof_full, ndof_full)
	bf = zeros(ndof_full)        # uniform-load route
	bwf = zeros(ndof_full)       # weak-form taper route (acceptance A5)

	Ke = hermite_Ke(p.EI, h)
	Me = hermite_Me(p.ρA, h)

	# Consistent nodal vector of uniform transverse load p_load = m_b/L
	p_load = p.m_b / L
	fe_uniform = p_load .* [h / 2, h^2 / 12, h / 2, -h^2 / 12]

	# 2-point Gauss on local coordinate ξ ∈ [0,1] (exact: linear taper × quadratic N′)
	ξg = (0.5 - 0.5 / sqrt(3.0), 0.5 + 0.5 / sqrt(3.0))
	wg = (0.5, 0.5)

	for e in 0:(n_elem - 1)
		dofs = (2e + 1):(2e + 4)
		Kf[dofs, dofs] .+= Ke
		Mf[dofs, dofs] .+= Me
		bf[dofs] .+= fe_uniform

		xe = e * h
		for (ξ, w) in zip(ξg, wg)
			x = xe + ξ * h
			mb = p.m_b * (1 - x / L)
			# dN/dx of the four Hermite shape functions at ξ
			dN = ((-6ξ + 6ξ^2) / h,
				1 - 4ξ + 3ξ^2,
				(6ξ - 6ξ^2) / h,
				-2ξ + 3ξ^2)
			for i in 1:4
				bwf[dofs[i]] += w * mb * dN[i] * h    # ∫ m_b N_i′ dx,  dx = h dξ
			end
		end
	end

	free = 3:ndof_full
	K = Kf[free, free]
	M = Mf[free, free]
	D = p.η_over_E .* K
	b = bf[free]
	b_weak = bwf[free]
	n = length(free)

	# Taper-weighted mean curvature ⟨w″⟩ = 2 w_tip / L² (exact identity, §4.3)
	g = zeros(n)
	idx_wtip = n - 1                 # w-DOF of the tip node
	g[idx_wtip] = 2 * p.α_c / L^2

	return (; M, K, D, b, b_weak, g, n, h, idx_wtip)
end

"""
	beam_checks(p, fe) — Phase 3 acceptance tests A1–A6. Throws on failure.
"""
function beam_checks(p, fe)
	# A1: symmetry & positive definiteness
	@assert norm(fe.K - fe.K') < 1e-12 * norm(fe.K) "A1: K not symmetric"
	@assert norm(fe.M - fe.M') < 1e-12 * norm(fe.M) "A1: M not symmetric"
	@assert isposdef(Symmetric(fe.K)) "A1: K not positive definite"
	@assert isposdef(Symmetric(fe.M)) "A1: M not positive definite"

	# A2: fundamental frequency vs analytic cantilever value
	ω1 = sqrt(minimum(real(eigvals(fe.K, fe.M))))
	relerr = abs(ω1 - p.ω1_target) / p.ω1_target
	@assert relerr < 1e-3 "A2: ω₁ = $ω1 vs $(p.ω1_target) (relerr = $relerr)"

	# A3: static tip-force test  u_tip = PL³/(3EI)
	f = zeros(fe.n)
	f[fe.idx_wtip] = 1.0
	u = fe.K \ f
	@assert abs(u[fe.idx_wtip] - p.L^3 / (3 * p.EI)) < 1e-6 * p.L^3 / (3 * p.EI) "A3 failed"

	# A4: static distributed-load test  u_tip = pL⁴/(8EI), p = m_b/L;  sign convention
	u = fe.K \ fe.b
	u_ref = (p.m_b / p.L) * p.L^4 / (8 * p.EI)
	@assert abs(u[fe.idx_wtip] - u_ref) < 1e-6 * abs(u_ref) "A4: tip deflection mismatch"
	@assert dot(fe.g, u) > 0 "A4: gᵀ(K⁻¹b) ≤ 0 — flip the sign of m_b"

	# A5: uniform-load and weak-form taper assemblies agree globally
	@assert norm(fe.b - fe.b_weak) < 1e-12 * norm(fe.b) "A5: b assemblies disagree"

	# A6: actuation is distributed — every w-DOF (odd indices) must be loaded.
	# Interior θ-couples cancel between adjacent elements by design (h²/12 pairs).
	@assert all(!iszero, fe.b[1:2:end]) "A6: some w-DOFs unloaded"
	frac = count(!iszero, fe.b) / fe.n
	@assert frac ≥ 0.45 "A6: b populated on only $(round(100frac))% of DOFs"

	println("Phase 3 (FE assembly) checks passed:  n = $(fe.n),  ω₁ = $ω1")
	return ω1
end
