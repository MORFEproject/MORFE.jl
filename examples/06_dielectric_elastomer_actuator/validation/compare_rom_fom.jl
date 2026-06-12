"""
validation/compare_rom_fom.jl — three-way validation (Phase 9).

1. CLOSURE   : coupled (u,q) system  vs  direct integration of the third-order system.
			   Mismatch must shrink ≈8× when v_a is halved (cubic-order closure).
2. ROM       : steady tip amplitude of the DPIM ROM vs the coupled FOM at Ω = Im λ₁.
3. mini-FRF  : ROM amplitude curve over Ω/ω₁ ∈ [0.97, 1.03] (re-reduction per Ω),
			   FOM long-time integration at a few points.

Run:  julia --project=.. validation/compare_rom_fom.jl     (or include after dea_demo.jl)
"""

# Run the full pipeline first (defines p, fe, bp, eigen data, W, R, mset4, Ω, …)
if !(@isdefined W)
	include(joinpath(@__DIR__, "..", "main.jl"))
end
include(joinpath(@__DIR__, "fom_reference.jl"))

using Printf

T_period = 2π / Ω
dt_fom = T_period / 500

# ──────────────────────────────────────────────────────────────────────────────
# 1. Closure validation: coupled vs third-order, error order in v_a
# ──────────────────────────────────────────────────────────────────────────────
println("\n[1] Closure validation (coupled (u,q) vs closed third-order system)")
T_cl = 20 * T_period
closure_errs = Float64[]
for va in (p.v_a, p.v_a / 2)
	pl = merge(p, (; v_a = va))
	_, w_cpl, _ = integrate_fom(pl, fe, bp; Ω, T_end = T_cl, dt = dt_fom)
	_, w_3rd, _ = integrate_thirdorder(pl, fe, bp; Ω, T_end = T_cl, dt = dt_fom)
	err = norm(w_cpl .- w_3rd) / max(norm(w_cpl), 1e-300)
	push!(closure_errs, err)
	@printf "    v_a = %-8.4g  rel. L2 mismatch = %.3e\n" va err
end
ratio = closure_errs[1] / max(closure_errs[2], 1e-300)
@printf "    error ratio (v_a halved) = %.2f   (cubic closure ⇒ expect ≈ 4–8)\n" ratio
@assert closure_errs[1] < 5e-2 "closure mismatch too large — check term transcription"

# ──────────────────────────────────────────────────────────────────────────────
# 2. ROM vs FOM at primary resonance
# ──────────────────────────────────────────────────────────────────────────────
println("\n[2] ROM vs coupled FOM, steady amplitude at Ω = Im λ₁ = $Ω")
T_ss = 60 * T_period
ts_f, w_f, _ = integrate_fom(p, fe, bp; Ω, T_end = T_ss, dt = dt_fom)
amp_fom = steady_amplitude(ts_f, w_f, T_period)

ts_r, Z_r = integrate_rom(R, mset4, ComplexF64[0, 0, 1, 1], T_ss, T_period / 200)
w_r = reconstruct_tip(W, mset4, Z_r, fe.idx_wtip)
amp_rom = steady_amplitude(ts_r, w_r, T_period)

@printf "    FOM amplitude = %.6e\n    ROM amplitude = %.6e\n    rel. error    = %.3e\n" amp_fom amp_rom abs(amp_rom - amp_fom) / amp_fom
@assert abs(amp_rom - amp_fom) / amp_fom < 0.05 "ROM amplitude error > 5%"

# ──────────────────────────────────────────────────────────────────────────────
# 3. mini-FRF: re-reduce per Ω (the external eigenvalues change with Ω)
# ──────────────────────────────────────────────────────────────────────────────
println("\n[3] mini-FRF around primary resonance (re-reduction per forcing frequency)")

function rom_amplitude_at(Ωk)
	model_k = build_model(p, fe, bp; forced = true, Ω = Ωk)
	res_k = resonance_set_from_complex_normal_form_style(
		mset4, Vector{ComplexF64}(master_eigenvalues), 0.05;
		external_eigenvalues = ComplexF64[im * Ωk, -im * Ωk])
	W_k, R_k = solve_cohomological_problem(
		model_k, mset4, master_eigenvalues, master_modes, left_eigenmodes, res_k;
		master_modes_derivatives = mmd, conjugate_permutation = [2, 1, 4, 3],
		show_progress = false)
	Tk = 2π / Ωk
	ts_k, Z_k = integrate_rom(R_k, mset4, ComplexF64[0, 0, 1, 1], 60 * Tk, Tk / 200)
	w_k = reconstruct_tip(W_k, mset4, Z_k, fe.idx_wtip)
	return steady_amplitude(ts_k, w_k, Tk)
end

function fom_amplitude_at(Ωk)
	Tk = 2π / Ωk
	ts_k, w_k, _ = integrate_fom(p, fe, bp; Ω = Ωk, T_end = 60 * Tk, dt = Tk / 500)
	return steady_amplitude(ts_k, w_k, Tk)
end

Ωs = Ω .* (0.97:0.01:1.03)
Ω_fom_check = Ω .* (0.97, 1.0, 1.03)
println("    Ω/ω₁      ROM amp         FOM amp        rel.err")
for Ωk in Ωs
	a_rom = rom_amplitude_at(Ωk)
	if any(isapprox.(Ωk, Ω_fom_check; rtol = 1e-12))
		a_fom = fom_amplitude_at(Ωk)
		@printf "    %.3f   %.6e   %.6e   %.2e\n" Ωk / Ω a_rom a_fom abs(a_rom - a_fom) / a_fom
		@assert abs(a_rom - a_fom) / a_fom < 0.05 "FRF point error > 5% at Ω/ω₁ = $(Ωk/Ω)"
	else
		@printf "    %.3f   %.6e\n" Ωk / Ω a_rom
	end
end

println("\nValidation complete.")
