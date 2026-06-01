"""
	compute_backbone.jl

Post-processing for the two-parameter parametric beam ROM in (z₁,z₂,θ₁,θ₂).

Loads R from results/, realifies the z₁ equation, and for each (θ₁,θ₂)
in a user-defined grid:

  (1) Extracts ω₀(θ₁,θ₂) = Im[∂R1/∂x(0,0,θ₁,θ₂)]  (linear eigenfrequency).
  (2) Computes the analytical backbone Ω(r,θ₁,θ₂) = Im[R1(r,0,θ₁,θ₂)] / r.
  (3) Validates with BifurcationKit (continuation of G(r,Ω)=0).

Figures produced:
  backbone_curves.png  — Ω vs |z₁| for various (θ₁,θ₂)
  backbone_shift.png   — (Ω − ω₀) vs |z₁|
  omega0_slope_minus2.png — ω₀(θ₁,0) vs (1+θ₁) log-log, slope -2 reference

Run after parametric_beam_demo.jl has produced results/R.jls.
"""

using Pkg: Pkg
const _backbone_env = joinpath(@__DIR__, "backbone_env")
Pkg.activate(_backbone_env)
if !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
end
Pkg.instantiate()

using Serialization
using StaticArrays: SVector
using MORFE
using MORFE.Polynomials: DensePolynomial, evaluate, extract_component,
	each_term, similar_poly
using MORFE.Realification: realify
using BifurcationKit
using Plots
ENV["GKSwstype"] = "nul"   # suppress GR pop-up window; plots are saved to file only
using Printf

# ------------------------------------------------------------------
# 1.  Load ROM
# ------------------------------------------------------------------
const _results = joinpath(@__DIR__, "results")
isfile(joinpath(_results, "R.jls")) ||
	error("results/R.jls not found.  Run parametric_beam_demo.jl first.")

println("Loading ROM …")
R = deserialize(joinpath(_results, "R.jls"))

# ------------------------------------------------------------------
# 2.  R1_cplx : ℝ⁴→ℂ   (realified z₁ equation in (x, y, θ₁, θ₂))
# ------------------------------------------------------------------
conj_map = [2, 1, 3, 4]    # z₁↔z₂; θ₁,θ₂ real → self-conjugate
R1_cplx = extract_component(realify(R.poly, conj_map), 1)

# ------------------------------------------------------------------
# 3.  Symbolic partial derivatives of R1_cplx
# ------------------------------------------------------------------

"""
	poly_deriv(p, var_idx)

Partial derivative of scalar polynomial `p` w.r.t. variable `var_idx`.
"""
function poly_deriv(p::DensePolynomial{T, NVAR, 1}, var_idx::Int) where {T, NVAR}
	dict = Dict{SVector{NVAR, Int}, T}()
	for (α, c) in each_term(p)
		n = α[var_idx]
		iszero(n) && continue
		new_α = SVector{NVAR, Int}(ntuple(j -> j == var_idx ? α[j] - 1 : α[j], Val(NVAR)))
		dict[new_α] = get(dict, new_α, zero(T)) + T(n) * c
	end
	return similar_poly(dict)
end

const dR1dx = poly_deriv(R1_cplx, 1)  # ∂R1_cplx/∂x

# Drop all monomials whose total parametric degree (sum of θ₁,θ₂ exponents) exceeds N.
function truncate_param_degree(p::DensePolynomial{T, NVAR, 1}, N::Int) where {T, NVAR}
	dict = Dict{SVector{NVAR, Int}, T}()
	for (α, c) in each_term(p)
		sum(α[i] for i in 3:NVAR) ≤ N || continue
		dict[α] = c
	end
	return similar_poly(dict)
end

# Order-3 variants: drop θ-degree ≥ 4 terms (the ones that diverge for |θ₁| ≳ 0.2).
const R1_cplx_3 = truncate_param_degree(R1_cplx, 3)
const dR1dx_3   = poly_deriv(R1_cplx_3, 1)

# ------------------------------------------------------------------
# 4.  Linear eigenfrequency ω₀(θ₁,θ₂) and backbone  Ω(r,θ₁,θ₂)
# ------------------------------------------------------------------
ω₀_of(θ₁, θ₂) = imag(evaluate(dR1dx, [0.0, 0.0, θ₁, θ₂]))
ω₀_of_3(θ₁, θ₂) = imag(evaluate(dR1dx_3, [0.0, 0.0, θ₁, θ₂]))
backbone_Ω(r, θ₁, θ₂) = imag(evaluate(R1_cplx, [max(r, 1e-12), 0.0, θ₁, θ₂])) /
						max(r, 1e-12)

# Announce reference frequency
let ω₀ = ω₀_of(0.0, 0.0)
	println(@sprintf("Linear frequency ω₀(0,0) = %.6f rad/s   T₀ = %.6f s", ω₀, 2π / ω₀))
end

# ------------------------------------------------------------------
# 5.  Parameters
# ------------------------------------------------------------------
const r_max = 5.0
const r_seed = 0.5

# Parameter grid: fix θ₂ at several arch levels, vary θ₁ (axial stretch)
const θ₁_values = [-0.25, -0.12, 0.0, 0.12, 0.25]
const θ₂_values = [0.0]   # arch amplitudes

# For backbone figure: all (θ₁, θ₂) combinations
all_params = [(θ₁, θ₂) for θ₂ in θ₂_values for θ₁ in θ₁_values]

# ------------------------------------------------------------------
# 6.  BifurcationKit continuation for one (θ₁,θ₂)
# ------------------------------------------------------------------
function run_bk_backbone(θ₁_val, θ₂_val)
	Ω₀ = backbone_Ω(r_seed, θ₁_val, θ₂_val)

	G(x, p) = [backbone_Ω(x[1], θ₁_val, θ₂_val) - p[1]]

	J_fun(x, _) = begin
		r  = max(x[1], 1e-10)
		f0 = imag(evaluate(R1_cplx, [r, 0.0, θ₁_val, θ₂_val]))
		df = imag(evaluate(dR1dx, [r, 0.0, θ₁_val, θ₂_val]))
		reshape([(df * r - f0) / r^2], 1, 1)
	end

	prob = BifurcationProblem(
		G, [r_seed], [Ω₀], @optic(_[1]);
		J = J_fun,
		record_from_solution = (x, p; k...) -> (r = x[1], Ω = p),
	)

	ω₀ = ω₀_of(θ₁_val, θ₂_val)
	opts = ContinuationPar(
		p_min = min(ω₀ * 0.75, ω₀ * 1.60),
		p_max = max(ω₀ * 0.75, ω₀ * 1.60),
		max_steps = 2000,
		ds = 0.01,
		dsmax = 0.2,
	)

	return try
		continuation(prob, PALC(), opts; verbosity = 0)
	catch e
		@warn "Continuation failed for (θ₁=$θ₁_val, θ₂=$θ₂_val): $e"
		nothing
	end
end

# ------------------------------------------------------------------
# 7.  Run all branches
# ------------------------------------------------------------------
println("Running BifurcationKit backbone continuations …")
branches = map(all_params) do (θ₁, θ₂)
	br = run_bk_backbone(θ₁, θ₂)
	if !isnothing(br)
		r_bk = [s.r for s in br.branch]
		Ω_bk = [s.Ω for s in br.branch]
		@printf "  (θ₁,θ₂)=(%.2f,%.2f): %d pts, r ∈ [%.2f,%.2f], Ω ∈ [%.4f,%.4f]\n" θ₁ θ₂ length(br.branch) extrema(r_bk)... extrema(Ω_bk)...
	end
	br
end

# ------------------------------------------------------------------
# 8.  Analytical backbone curves
# ------------------------------------------------------------------
r_range = range(1e-6, r_max, 400)

Ω_curves = [(θ₁, θ₂, [backbone_Ω(r, θ₁, θ₂) for r in r_range])
			for (θ₁, θ₂) in all_params]
ω₀_vals = [ω₀_of(θ₁, θ₂) for (θ₁, θ₂) in all_params]

# ------------------------------------------------------------------
# 9.  Figure 1: absolute backbone Ω vs |z₁|
# ------------------------------------------------------------------
println("Plotting …")

clrs = palette(:tab10)

plt1 = plot(;
	xlabel = "Backbone frequency  Ω  (rad/s)",
	ylabel = "Modal amplitude  |z₁|",
	title = "Two-parameter backbone curves",
	ylims = (0, r_max * 1.05),
	size = (800, 600),
	dpi = 150,
)

for (k, ((θ₁, θ₂, Ωcurve), br)) in enumerate(zip(Ω_curves, branches))
	lbl = "(θ₁,θ₂)=($(θ₁),$(θ₂))"
	plot!(plt1, Ωcurve, collect(r_range); lw = 2.0, color = clrs[k], label = lbl)
	if !isnothing(br) && length(br.branch) > 1
		r_bk = [s.r for s in br.branch]
		Ω_bk = [s.Ω for s in br.branch]
		mask = r_bk .<= r_max
		scatter!(plt1, Ω_bk[mask], r_bk[mask];
			color = clrs[k], ms = 3, markerstrokewidth = 0, label = nothing)
	end
end

# ------------------------------------------------------------------
# 10.  Figure 2: nonlinear shift (Ω − ω₀) vs |z₁|
# ------------------------------------------------------------------
plt2 = plot(;
	xlabel = "Nonlinear frequency shift  Ω − ω₀(θ₁,θ₂)  (rad/s)",
	ylabel = "Modal amplitude  |z₁|",
	title = "Two-parameter backbone shift",
	ylims = (0, r_max * 1.05),
	size = (800, 600),
	dpi = 150,
)

for (k, ((θ₁, θ₂, Ωcurve), br, ω₀)) in enumerate(zip(Ω_curves, branches, ω₀_vals))
	lbl = "(θ₁,θ₂)=($(θ₁),$(θ₂))"
	plot!(plt2, Ωcurve .- ω₀, collect(r_range); lw = 2.0, color = clrs[k], label = lbl)
	if !isnothing(br) && length(br.branch) > 1
		r_bk = [s.r for s in br.branch]
		Ω_bk = [s.Ω for s in br.branch]
		mask = r_bk .<= r_max
		scatter!(plt2, Ω_bk[mask] .- ω₀, r_bk[mask];
			color = clrs[k], ms = 3, markerstrokewidth = 0, label = nothing)
	end
end

# ------------------------------------------------------------------
# 11.  Save figures 1 and 2
# ------------------------------------------------------------------
mkpath(_results)
savefig(plt1, joinpath(_results, "backbone_curves.png"))
savefig(plt2, joinpath(_results, "backbone_shift.png"))
println("Saved → $(_results)/backbone_curves.png")
println("Saved → $(_results)/backbone_shift.png")

# ------------------------------------------------------------------
# 12.  ω₀(θ₁,0) vs (1+θ₁) — log-log with slope -2 reference
#      (axial-stretch effect, arch fixed at θ₂=0)
# ------------------------------------------------------------------
const _L₀ = 1000.0;
const _W = 10.0;
const _H = 24.0
const _E = 160e3;
const _ρ = 2.32e-3
const _A = _W * _H;
const _I = _H * _W^3 / 12
const _β₁L = 4.730040744862704
const _ω₁_EB₀ = (_β₁L / _L₀)^2 * sqrt(_E * _I / (_ρ * _A))

ω₁_EB(θ₁) = _ω₁_EB₀ * (1 + θ₁)^(-2)

let ω_m = ω₀_of(0.0, 0.0)
	@printf "EB analytical ω₁(0,0) = %.6f rad/ms\n" _ω₁_EB₀
	@printf "MORFE ROM    ω₀(0,0)  = %.6f rad/ms\n" ω_m
	@printf "Relative error        = %.4f %%\n" 100 * abs(_ω₁_EB₀ - ω_m) / _ω₁_EB₀
end

θ₁_fine  = range(-0.8, 1.0, 200)
ω_fine   = [ω₀_of(θ₁, 0.0) for θ₁ in θ₁_fine]
ω_fine_3 = [ω₀_of_3(θ₁, 0.0) for θ₁ in θ₁_fine]
ω₀_ref   = ω₀_of(0.0, 0.0) .* (1 .+ θ₁_fine) .^ (-2)
ω_EB     = ω₁_EB.(θ₁_fine)

# Mask: only plot ROM values where the polynomial gives a physically valid (positive)
# frequency.  The degree-5 expansion diverges for |θ₁| ≳ 0.2; wrapping with abs()
# would hide that failure rather than expose it.
mask   = ω_fine .> 0
mask_3 = ω_fine_3 .> 0

# Analytical references are always positive — plot them over the full range.
plt3 = plot(1 .+ θ₁_fine, ω₀_ref;
	xscale = :log10, yscale = :log10,
	lw = 2.0, ls = :dash, color = :red, label = "ω₀(0)·(1+θ₁)⁻²  (slope –2)",
	xlabel = "1 + θ₁", ylabel = "ω₀ (rad/ms)",
	title = "Axial-stretch eigenfrequency scaling (θ₂=0)",
	legend = :bottomright,
	size = (600, 500), dpi = 150,
)
plot!(plt3, 1 .+ θ₁_fine, ω_EB;
	lw = 1.5, ls = :dot, color = :blue, label = "EB: (β₁/L₀)²√(EI/ρA)·(1+θ₁)⁻²")
# Order-3 truncation (θ-degree ≤ 3): uses only the directly-assembled K/M coefficients.
plot!(plt3, (1 .+ θ₁_fine)[mask_3], ω_fine_3[mask_3];
	lw = 2.0, ls = :dashdot, color = :darkorange, label = "ω₀(θ₁,0) MORFE order-3")
# Full degree-5 ROM restricted to the valid region where the polynomial hasn't diverged.
plot!(plt3, (1 .+ θ₁_fine)[mask], ω_fine[mask];
	lw = 2.5, color = :black, label = "ω₀(θ₁,0) MORFE order-5")

savefig(plt3, joinpath(_results, "omega0_slope_minus2.png"))
println("Saved → $(_results)/omega0_slope_minus2.png")
