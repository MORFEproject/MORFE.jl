"""
validation/compare_rom_fom.jl — three-way validation (Phase 9).

1. CLOSURE   : coupled (u,q) system  vs  direct integration of the third-order system.
               Mismatch must shrink ≈8× when v_a is halved (cubic-order closure).
2. ROM       : steady tip amplitude of the DPIM ROM vs the coupled FOM at Ω = Im λ₁.
3. mini-FRF  : ROM amplitude curve over Ω/ω₁ ∈ [0.97, 1.03] (re-reduction per Ω),
               FOM long-time integration at a few points.

Outputs (saved to validation/figures/):
  closure_comparison.png  — tip displacement time history (coupled vs third-order)
  frf.png                 — FRF: ROM curve + FOM check points
  rom_vs_fom_timehist.png — ROM vs FOM tip time history at Ω = Im λ₁

Run:  julia --project=.. validation/compare_rom_fom.jl     (or include after main.jl)
"""

# Run the full pipeline first (defines p, fe, bp, eigen data, W, R, mset4, Ω, …)
if !(@isdefined W)
    include(joinpath(@__DIR__, "..", "main.jl"))
end
include(joinpath(@__DIR__, "fom_reference.jl"))

using Printf
using Plots
ENV["GKSwstype"] = "nul"   # headless rendering (no display required)

const _figs = joinpath(@__DIR__, "figures")
mkpath(_figs)

T_period = 2π / Ω
dt_fom = T_period / 500

# ──────────────────────────────────────────────────────────────────────────────
# 1. Closure validation: coupled vs third-order, error order in v_a
# ──────────────────────────────────────────────────────────────────────────────
println("\n[1] Closure validation (coupled (u,q) vs closed third-order system)")
T_cl = 20 * T_period
# Run both integrations explicitly to avoid Julia soft-scope ambiguity
ts_cl, wcpl_cl, _ = integrate_fom(p, fe, bp; Ω, T_end = T_cl, dt = dt_fom)
_, w3rd_cl, _ = integrate_thirdorder(p, fe, bp; Ω, T_end = T_cl, dt = dt_fom)
err1 = norm(wcpl_cl .- w3rd_cl) / max(norm(wcpl_cl), 1e-300)
@printf "    v_a = %-8.4g  rel. L2 mismatch = %.3e\n" p.v_a err1

let p_half = merge(p, (; v_a = p.v_a / 2))
    _, w_cpl_h, _ = integrate_fom(p_half, fe, bp; Ω, T_end = T_cl, dt = dt_fom)
    _, w_3rd_h, _ = integrate_thirdorder(p_half, fe, bp; Ω, T_end = T_cl, dt = dt_fom)
    err2 = norm(w_cpl_h .- w_3rd_h) / max(norm(w_cpl_h), 1e-300)
    @printf "    v_a = %-8.4g  rel. L2 mismatch = %.3e\n" (p.v_a / 2) err2
    global closure_errs = [err1, err2]
end
ratio = closure_errs[1] / max(closure_errs[2], 1e-300)
@printf "    error ratio (v_a halved) = %.2f   (cubic closure ⇒ expect ≈ 4–8)\n" ratio
@assert closure_errs[1] < 5e-2 "closure mismatch too large — check term transcription"

# Plot last 5 periods of the closure comparison
t_cut = ts_cl[end] - 5 * T_period
idx_cl = findall(>=(t_cut), ts_cl)
plt_cl = plot(ts_cl[idx_cl], wcpl_cl[idx_cl];
    lw = 2.0, color = :black, label = "coupled FOM  (u,q)",
    xlabel = "t", ylabel = "tip displacement",
    title = "Closure check: coupled FOM vs third-order system  (v_a = $(p.v_a))",
    size = (900, 400), dpi = 150)
plot!(plt_cl, ts_cl[idx_cl], w3rd_cl[idx_cl];
    lw = 1.5, ls = :dash, color = :crimson, label = "third-order closure")
savefig(plt_cl, joinpath(_figs, "closure_comparison.png"))
println("    Saved → $(joinpath(_figs, "closure_comparison.png"))")

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

@printf "    FOM amplitude = %.6e\n    ROM amplitude = %.6e\n    rel. error    = %.3e\n" amp_fom amp_rom abs(amp_rom -
                                                                                                             amp_fom) /
                                                                                                         amp_fom
@assert abs(amp_rom - amp_fom) / amp_fom < 0.05 "ROM amplitude error > 5%"

# Plot last 5 periods of the ROM vs FOM tip history
t_cut2 = ts_f[end] - 5 * T_period
idx_f = findall(>=(t_cut2), ts_f)
t_cut2r = ts_r[end] - 5 * T_period
idx_r = findall(>=(t_cut2r), ts_r)
plt_th = plot(ts_f[idx_f], w_f[idx_f];
    lw = 2.0, color = :black, label = "coupled FOM",
    xlabel = "t", ylabel = "tip displacement",
    title = "ROM vs FOM at Ω = Im λ₁  (steady state)",
    size = (900, 400), dpi = 150)
plot!(plt_th, ts_r[idx_r], w_r[idx_r];
    lw = 1.5, ls = :dash, color = :dodgerblue, label = "ROM (DPIM)")
savefig(plt_th, joinpath(_figs, "rom_vs_fom_timehist.png"))
println("    Saved → $(joinpath(_figs, "rom_vs_fom_timehist.png"))")

# ──────────────────────────────────────────────────────────────────────────────
# 3. mini-FRF: re-reduce per Ω (the external eigenvalues change with Ω)
# ──────────────────────────────────────────────────────────────────────────────
println("\n[3] mini-FRF around primary resonance (re-reduction per forcing frequency)")

function rom_amplitude_at(Ωk)
    model_k = build_model(p, fe, bp; forced = true, Ω = Ωk)
    res_k = resonance_set_from_complex_normal_form_style(
        mset4, Vector{ComplexF64}(master_eigenvalues), 0.05;
        external_eigenvalues = ComplexF64[im * Ωk, -im * Ωk])
    W_k, R_k = solve_cohomological_problem(model_k, mset4, spectral, res_k;
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
rom_amps = Float64[]
fom_check_Ω = Float64[]
fom_check_amp = Float64[]

println("    Ω/ω₁      ROM amp         FOM amp        rel.err")
for Ωk in Ωs
    a_rom = rom_amplitude_at(Ωk)
    push!(rom_amps, a_rom)
    if any(isapprox.(Ωk, Ω_fom_check; rtol = 1e-12))
        a_fom = fom_amplitude_at(Ωk)
        push!(fom_check_Ω, Ωk)
        push!(fom_check_amp, a_fom)
        @printf "    %.3f   %.6e   %.6e   %.2e\n" Ωk / Ω a_rom a_fom abs(a_rom - a_fom) /
                                                                     a_fom
        @assert abs(a_rom - a_fom) / a_fom < 0.05 "FRF point error > 5% at Ω/ω₁ = $(Ωk/Ω)"
    else
        @printf "    %.3f   %.6e\n" Ωk / Ω a_rom
    end
end

# FRF plot
plt_frf = plot(Ωs ./ Ω, rom_amps;
    lw = 2.0, color = :dodgerblue, label = "ROM (DPIM, degree 5)",
    xlabel = "Ω / ω₁", ylabel = "Steady tip amplitude",
    title = "Frequency response around primary resonance",
    size = (800, 500), dpi = 150)
scatter!(plt_frf, fom_check_Ω ./ Ω, fom_check_amp;
    ms = 7, color = :black, markershape = :circle, label = "coupled FOM")
savefig(plt_frf, joinpath(_figs, "frf.png"))
println("    Saved → $(joinpath(_figs, "frf.png"))")

println("\nValidation complete.")
