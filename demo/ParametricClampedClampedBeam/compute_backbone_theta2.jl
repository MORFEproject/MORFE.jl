"""
	compute_backbone_theta2.jl

Backbone curves for the two-parameter parametric beam ROM, sweeping θ₂
(arch pre-deformation amplitude) at fixed θ₁ = 0 (nominal axial length).

Loads R from results/ and for each θ₂ in a fine grid:
  (1) Extracts ω₀(0, θ₂) = Im[∂R1/∂x(0,0,0,θ₂)]  (linear eigenfrequency).
  (2) Computes the analytical backbone Ω(r, 0, θ₂) = Im[R1(r,0,0,θ₂)] / r.
  (3) Validates with BifurcationKit continuation.

Figures produced in results/:
  backbone_theta2_curves.png  — Ω vs |z₁| for each θ₂
  backbone_theta2_shift.png   — (Ω − ω₀) vs |z₁|

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
ENV["GKSwstype"] = "nul"
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
conj_map = [2, 1, 3, 4]
R1_cplx = extract_component(realify(R.poly, conj_map), 1)

# ------------------------------------------------------------------
# 3.  Symbolic partial derivative ∂R1/∂x
# ------------------------------------------------------------------
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

const dR1dx = poly_deriv(R1_cplx, 1)

function truncate_param_degree(p::DensePolynomial{T, NVAR, 1}, N::Int) where {T, NVAR}
	dict = Dict{SVector{NVAR, Int}, T}()
	for (α, c) in each_term(p)
		sum(α[i] for i in 3:NVAR) ≤ N || continue
		dict[α] = c
	end
	return similar_poly(dict)
end

const R1_cplx_3 = truncate_param_degree(R1_cplx, 3)
const dR1dx_3   = poly_deriv(R1_cplx_3, 1)

# ------------------------------------------------------------------
# 4.  Linear eigenfrequency and backbone
# ------------------------------------------------------------------
ω₀_of(θ₂)    = imag(evaluate(dR1dx,   [0.0, 0.0, 0.0, θ₂]))
ω₀_of_3(θ₂)  = imag(evaluate(dR1dx_3, [0.0, 0.0, 0.0, θ₂]))
backbone_Ω(r, θ₂) = imag(evaluate(R1_cplx,   [max(r, 1e-12), 0.0, 0.0, θ₂])) /
					max(r, 1e-12)
backbone_Ω_3(r, θ₂) = imag(evaluate(R1_cplx_3, [max(r, 1e-12), 0.0, 0.0, θ₂])) /
					max(r, 1e-12)

let ω₀ = ω₀_of(0.0)
	println(@sprintf("Linear frequency ω₀(0,0) = %.6f rad/s   T₀ = %.6f s", ω₀, 2π / ω₀))
end

# ------------------------------------------------------------------
# 4b.  Symmetry check: ω₀(+θ₂) ≈ ω₀(-θ₂)
# ------------------------------------------------------------------
println("\nSymmetry check: ω₀(+θ₂) vs ω₀(-θ₂)")
println(@sprintf("  %-8s  %-14s  %-14s  %-12s", "θ₂", "ω₀(+θ₂)", "ω₀(-θ₂)", "|Δω₀|/ω₀(0)"))
let ω₀_ref = ω₀_of(0.0)
	for θ₂ in [0.25, 0.5, 0.75, 1.0]
		ωp = ω₀_of(+θ₂)
		ωm = ω₀_of(-θ₂)
		rel = abs(ωp - ωm) / ω₀_ref
		println(@sprintf("  %-8.3f  %-14.6f  %-14.6f  %-12.2e", θ₂, ωp, ωm, rel))
	end
end
println()

# ------------------------------------------------------------------
# 4c.  Plot ω₀(θ₂) vs θ₂ for small arch pre-deformations
# ------------------------------------------------------------------
let θ₂_fine = range(-20.00, 20.00, 300)
	ω₀_fine   = ω₀_of.(θ₂_fine)
	ω₀_fine_3 = ω₀_of_3.(θ₂_fine)
	ω₀_ref    = ω₀_of(0.0)
	mask5 = ω₀_fine   .> 0
	mask3 = ω₀_fine_3 .> 0

	plt_ω0 = plot(θ₂_fine[mask5], ω₀_fine[mask5];
		xlabel = "Arch parameter  θ₂",
		ylabel = "Linear eigenfrequency  ω₀(0, θ₂)  (rad/ms)",
		title  = "Eigenfrequency vs arch pre-deformation (θ₁ = 0)",
		lw     = 2.5,
		color  = :royalblue,
		label  = "ω₀(0, θ₂)  (ROM order-5)",
		legend = :top,
		size   = (700, 480),
		dpi    = 150,
	)
	plot!(plt_ω0, θ₂_fine[mask3], ω₀_fine_3[mask3];
		lw = 2.0, ls = :dashdot, color = :darkorange,
		label = "ω₀(0, θ₂)  (ROM order-3)")
	hline!(plt_ω0, [ω₀_ref]; ls = :dash, color = :gray, lw = 1.2,
		label = @sprintf("ω₀(0,0) = %.5f", ω₀_ref))
	vline!(plt_ω0, [0.0]; ls = :dot, color = :black, lw = 0.8, label = nothing)

	mkpath(_results)
	savefig(plt_ω0, joinpath(_results, "omega0_vs_theta2.png"))
	println("Saved → $(_results)/omega0_vs_theta2.png")
end

# ------------------------------------------------------------------
# 5.  θ₂ sweep parameters
# ------------------------------------------------------------------
const r_max  = 5.0
const r_seed = 0.5

const θ₂_values = [-0.1, -0.05, 0.0, 0.05, 0.1]

# ------------------------------------------------------------------
# 6.  BifurcationKit continuation for one θ₂
# ------------------------------------------------------------------
function run_bk_backbone(θ₂_val)
	Ω₀ = backbone_Ω(r_seed, θ₂_val)

	G(x, p) = [backbone_Ω(x[1], θ₂_val) - p[1]]

	J_fun(x, _) = begin
		r  = max(x[1], 1e-10)
		f0 = imag(evaluate(R1_cplx, [r, 0.0, 0.0, θ₂_val]))
		df = imag(evaluate(dR1dx, [r, 0.0, 0.0, θ₂_val]))
		reshape([(df * r - f0) / r^2], 1, 1)
	end

	prob = BifurcationProblem(
		G, [r_seed], [Ω₀], @optic(_[1]);
		J = J_fun,
		record_from_solution = (x, p; k...) -> (r = x[1], Ω = p),
	)

	ω₀ = ω₀_of(θ₂_val)
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
		@warn "Continuation failed for θ₂=$θ₂_val: $e"
		nothing
	end
end

# ------------------------------------------------------------------
# 7.  Run all branches
# ------------------------------------------------------------------
println("Running BifurcationKit backbone continuations …")
branches = map(θ₂_values) do θ₂
	br = run_bk_backbone(θ₂)
	if !isnothing(br)
		r_bk = [s.r for s in br.branch]
		Ω_bk = [s.Ω for s in br.branch]
		@printf "  θ₂=%.3f: %d pts, r ∈ [%.2f,%.2f], Ω ∈ [%.4f,%.4f]\n" θ₂ length(br.branch) extrema(r_bk)... extrema(Ω_bk)...
	end
	br
end

# ------------------------------------------------------------------
# 8.  Analytical backbone curves
# ------------------------------------------------------------------
r_range   = range(1e-6, r_max, 400)
Ω_curves  = [backbone_Ω.(r_range, θ₂)   for θ₂ in θ₂_values]
Ω_curves_3 = [backbone_Ω_3.(r_range, θ₂) for θ₂ in θ₂_values]
ω₀_vals   = [ω₀_of(θ₂)   for θ₂ in θ₂_values]
ω₀_vals_3 = [ω₀_of_3(θ₂) for θ₂ in θ₂_values]

# ------------------------------------------------------------------
# 9.  Colour palette — diverging blue→red through white at θ₂=0
# ------------------------------------------------------------------
println("Plotting …")

clrs = palette(:Blues_4, length(θ₂_values) + 1)[2:end]

# ------------------------------------------------------------------
# 10.  Figure 1: absolute backbone Ω vs |z₁|
# ------------------------------------------------------------------
plt1 = plot(;
	xlabel = "Backbone frequency  Ω  (rad/s)",
	ylabel = "Modal amplitude  |z₁|",
	title  = "θ₂ sweep (θ₁ = 0): backbone curves",
	ylims  = (0, r_max * 1.05),
	size   = (800, 600),
	dpi    = 150,
	legend = :outertopright,
)

for (k, (θ₂, Ωcurve, Ωcurve_3, br)) in
		enumerate(zip(θ₂_values, Ω_curves, Ω_curves_3, branches))
	lbl = @sprintf("θ₂ = %.2f", θ₂)
	plot!(plt1, Ωcurve, collect(r_range); lw = 2.0, color = clrs[k], label = lbl)
	plot!(plt1, Ωcurve_3, collect(r_range);
		lw = 1.2, ls = :dash, color = clrs[k], label = nothing)
	if !isnothing(br) && length(br.branch) > 1
		r_bk = [s.r for s in br.branch]
		Ω_bk = [s.Ω for s in br.branch]
		mask = r_bk .<= r_max
		scatter!(plt1, Ω_bk[mask], r_bk[mask];
			color = clrs[k], ms = 3, markerstrokewidth = 0, label = nothing)
	end
end

# ------------------------------------------------------------------
# 11.  Figure 2: nonlinear shift (Ω − ω₀) vs |z₁|
# ------------------------------------------------------------------
plt2 = plot(;
	xlabel = "Nonlinear frequency shift  Ω − ω₀(0,θ₂)  (rad/s)",
	ylabel = "Modal amplitude  |z₁|",
	title  = "θ₂ sweep (θ₁ = 0): backbone shift",
	ylims  = (0, r_max * 1.05),
	size   = (800, 600),
	dpi    = 150,
	legend = :outertopright,
)

for (k, (θ₂, Ωcurve, Ωcurve_3, br, ω₀, ω₀_3)) in
		enumerate(zip(θ₂_values, Ω_curves, Ω_curves_3, branches, ω₀_vals, ω₀_vals_3))
	lbl = @sprintf("θ₂ = %.2f", θ₂)
	plot!(plt2, Ωcurve .- ω₀, collect(r_range); lw = 2.0, color = clrs[k], label = lbl)
	plot!(plt2, Ωcurve_3 .- ω₀_3, collect(r_range);
		lw = 1.2, ls = :dash, color = clrs[k], label = nothing)
	if !isnothing(br) && length(br.branch) > 1
		r_bk = [s.r for s in br.branch]
		Ω_bk = [s.Ω for s in br.branch]
		mask = r_bk .<= r_max
		scatter!(plt2, Ω_bk[mask] .- ω₀, r_bk[mask];
			color = clrs[k], ms = 3, markerstrokewidth = 0, label = nothing)
	end
end

# ------------------------------------------------------------------
# 12.  Save
# ------------------------------------------------------------------
mkpath(_results)
savefig(plt1, joinpath(_results, "backbone_theta2_curves.png"))
savefig(plt2, joinpath(_results, "backbone_theta2_shift.png"))
println("Saved → $(_results)/backbone_theta2_curves.png")
println("Saved → $(_results)/backbone_theta2_shift.png")
