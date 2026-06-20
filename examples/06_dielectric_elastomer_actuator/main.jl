"""
	main.jl — Top-level driver for the dielectric elastomer actuator DPIM demo.

Pipeline
────────
1.  Cubic-Hermite FE assembly of the cantilever (M, K, D, tapered load b, strain g)
2.  Static bias point (x₀, Q₀) at V₀ and derived constants (ĉ, ℓ₀, ℓ₁, ℓ₂)
3.  Third-order model:  B₃u⁽³⁾ + B₂ü + B₁u̇ + B₀u = F(u, u̇, ü, v)   [ORD = 3]
4.  Dense companion eigenanalysis of the cubic pencil → master bending pair
5.  Milestone 1: autonomous reduction (NVAR = 2) — backbone of the biased actuator
6.  Milestone 2: forced reduction (NVAR = 4, ExternalSystem ±iΩ) at primary resonance
7.  Realification + reduced-dynamics report
8.  ROM time response of the tip

All parameters in config.jl. Acceptance checks (implementation_plan_detailed.md
Phases 3–7) run inline and `@assert`; set `RUN_CHECKS = false` to skip.
Validation against the coupled (u, q) ground truth: validation/compare_rom_fom.jl.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml")) || !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
	Pkg.add(["StaticArrays", "Printf", "Plots"])
end
Pkg.instantiate()

using MORFE
using LinearAlgebra
using StaticArrays
using Printf

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "fem", "hermite_beam.jl"))
include(joinpath(@__DIR__, "model", "bias.jl"))
include(joinpath(@__DIR__, "model", "coupling_terms.jl"))
include(joinpath(@__DIR__, "solver", "eigensolver.jl"))
include(joinpath(@__DIR__, "solver", "rom_utils.jl"))

const RUN_CHECKS = get(ENV, "DEA_RUN_CHECKS", "1") == "1"

# ─────────────────────────────────────────────────────────────────────────────
# 1–2. FE assembly and bias point
# ─────────────────────────────────────────────────────────────────────────────
p = dea_parameters()
fe = assemble_beam(p)
RUN_CHECKS && beam_checks(p, fe)
bp = bias_point(p, fe)
RUN_CHECKS && bias_checks(p, fe, bp)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Third-order model (autonomous variant; forcing added per Ω in step 6)
# ─────────────────────────────────────────────────────────────────────────────
RUN_CHECKS && coupling_checks(p, fe, bp)
B0, B1, B2, B3 = build_linear_matrices(p, fe, bp)
model_auto = build_model(p, fe, bp; forced = false)
n = fe.n

# ─────────────────────────────────────────────────────────────────────────────
# 4. Eigenanalysis of the cubic pencil (dense companion form)
# ─────────────────────────────────────────────────────────────────────────────
println("\nEigenanalysis (dense companion form, 3n = $(3n)) …")
eig = dea_eigenanalysis(model_auto, B0, B1, B2, B3, fe.idx_wtip, bp.ĉ / p.R)
(; λ1, master_eigenvalues, master_modes, mmd, left_eigenmodes) = eig

# ─────────────────────────────────────────────────────────────────────────────
# 5. Milestone 1 — autonomous reduction (backbone)
# ─────────────────────────────────────────────────────────────────────────────
println("\nMilestone 1 — autonomous reduction (NVAR = 2, degree ≤ 7) …")
mset2 = all_multiindices_up_to(2, 7; min_degree = 1)
res2 = resonance_set_from_complex_normal_form_style(
	mset2, Vector{ComplexF64}(master_eigenvalues), 0.05)
@time W_a, R_a = solve_cohomological_problem(
	model_auto, mset2, master_eigenvalues, master_modes, left_eigenmodes, res2;
	master_modes_derivatives = mmd, conjugate_permutation = [2, 1])

# Realify the master equation ż₁ = ẋ₁ + i ẏ₁ in real variables (x₁, y₁):
# Re(c) feeds ẋ₁, Im(c) feeds ẏ₁ (imaginary parts carry the frequency content).
Rr_a = realify(extract_component(R_a.poly, 1), [2, 1])
println("Autonomous reduced dynamics ż₁ in real variables (x₁, y₁):")
println("  (x₁,y₁) exponents :  ẋ₁-coeff      ẏ₁-coeff")
for (m, mi) in enumerate(Rr_a.multiindex_set.exponents)
	c = Rr_a.coefficients[m]
	abs(c) > 1e-10 &&
		@printf "  %-18s: %+12.5e  %+12.5e\n" string(Tuple(mi)) real(c) imag(c)
end

# Backbone coefficient (resonant cubic monomial z₁²z̄₁ ↦ exponent (2,1))
let idx = findfirst(α -> Tuple(α) == (2, 1), [Tuple(a) for a in mset2.exponents])
	γ_bb = R_a.poly.coefficients[1, idx]
	@printf "Backbone coefficient c(z₁²z̄₁) = %+.6e %+.6eim\n" real(γ_bb) imag(γ_bb)
	@assert isfinite(γ_bb) "backbone coefficient not finite"
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Milestone 2 — forced reduction at primary resonance
# ─────────────────────────────────────────────────────────────────────────────
Ω = imag(λ1)
println("\nMilestone 2 — forced reduction (NVAR = 4, degree ≤ 5),  Ω = $Ω …")
model_f = build_model(p, fe, bp; forced = true, Ω = Ω)
mset4 = all_multiindices_up_to(4, 5; min_degree = 1)
res4 = resonance_set_from_complex_normal_form_style(
	mset4, Vector{ComplexF64}(master_eigenvalues), 0.05;
	external_eigenvalues = ComplexF64[im * Ω, -im * Ω])
@time W, R = solve_cohomological_problem(
	model_f, mset4, master_eigenvalues, master_modes, left_eigenmodes, res4;
	master_modes_derivatives = mmd, conjugate_permutation = [2, 1, 4, 3])

# ─────────────────────────────────────────────────────────────────────────────
# 7. Acceptance checks + realification
# ─────────────────────────────────────────────────────────────────────────────
if RUN_CHECKS
	@assert all(isfinite, W.poly.coefficients) "NaN/Inf in W"
	@assert all(isfinite, R.poly.coefficients) "NaN/Inf in R"
	let exps = [Tuple(a) for a in mset4.exponents]
		targets = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
		lin = [findfirst(==(t), exps) for t in targets]
		Λdiag = [R.poly.coefficients[i, lin[i]] for i in 1:4]
		target = [λ1, conj(λ1), im * Ω, -im * Ω]
		@assert all(abs.(Λdiag .- target) .< 1e-8 .* abs.(target)) "linear part of R wrong"
	end
end

# Conjugate symmetry of the reduced dynamics: R₂(z) must equal conj(R₁) under the
# exponent permutation (z₁↔z₂, r₁↔r₂). This is the invariant that makes the realified
# (ẋ, ẏ) system real — the realified ż₁ coefficients themselves are complex by design
# (Re → ẋ₁, Im → ẏ₁).
let exps = [Tuple(a) for a in mset4.exponents]
	lookup = Dict(e => i for (i, e) in enumerate(exps))
	permc = (2, 1, 4, 3)
	worst = 0.0
	for (m, e) in enumerate(exps)
		mbar = lookup[ntuple(j -> e[permc[j]], 4)]
		worst = max(worst,
			abs(R.poly.coefficients[2, m] - conj(R.poly.coefficients[1, mbar])))
	end
	@assert worst < 1e-8 * maximum(abs.(R.poly.coefficients)) "conjugate symmetry violated ($worst)"
end

# Realified master equation in real variables (x₁, x₂, y₁, y₂) where (x₁,y₁) ↔ z-pair
# and (x₂,y₂) ↔ forcing pair r.
Rr = realify(extract_component(R.poly, 1), [2, 1, 4, 3])
println("Forced reduced dynamics ż₁ in real variables (x₁, x₂, y₁, y₂):")
println("  exponents          :  ẋ₁-coeff      ẏ₁-coeff")
for (m, mi) in enumerate(Rr.multiindex_set.exponents)
	c = Rr.coefficients[m]
	abs(c) > 1e-6 &&
		@printf "  %-18s: %+12.5e  %+12.5e\n" string(Tuple(mi)) real(c) imag(c)
end
println("Phase 7 checks passed.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. ROM time response of the tip at primary resonance
# ─────────────────────────────────────────────────────────────────────────────
T_period = 2π / Ω
T_end = 40 * T_period
z0 = ComplexF64[0, 0, 1, 1]
ts, Z = integrate_rom(R, mset4, z0, T_end, T_period / 200)
wtip = reconstruct_tip(W, mset4, Z, fe.idx_wtip)
amp = steady_amplitude(ts, wtip, T_period)
@printf "\nROM steady-state tip amplitude at Ω = Im λ₁:  %.6e  (bias deflection %.6e)\n" amp bp.x0[fe.idx_wtip]
println("\nDone. Run validation/compare_rom_fom.jl for the FOM comparison.")
