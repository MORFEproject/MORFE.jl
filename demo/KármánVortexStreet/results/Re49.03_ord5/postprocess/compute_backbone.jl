"""
	compute_backbone.jl

Post-processing for the KVS DPIM ROM.

Loads R and W from the parent results directory, realifies ż₁, and produces
three figures showing the supercritical Hopf bifurcation structure:

  bifurcation_diagram.png — Amplitude ρ*(Re) vs Re: fixed-point branch
	  (solid = stable, dashed = unstable) + limit-cycle branch.
	  ρ ∝ √(Re − Re_c) near Re_c, giving perpendicular branches.
  hopf_3d.png             — 3D (Re, a₁, a₂) Hopf tube: Re-axis as fixed-point
	  line, circles of radius ρ*(Re) in the perpendicular plane.
  strouhal_vs_Re.png      — Strouhal angular frequency Ω*(Re) vs Re.

Run after main.jl has produced R.jls and W.jls.
"""

using Pkg: Pkg
const _backbone_env = joinpath(@__DIR__, "../../../backbone_env")
Pkg.activate(_backbone_env)
if !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../../../../..")))
	Pkg.add(["Roots", "Plots", "StaticArrays"])
end
Pkg.instantiate()

using Serialization
using StaticArrays: SVector
using MORFE
using MORFE.Polynomials: DensePolynomial, evaluate, extract_component, each_term, similar_poly
using MORFE.Realification: realify
using Roots
using Plots
ENV["GKSwstype"] = "nul"
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load ROM
# ─────────────────────────────────────────────────────────────────────────────

const _results = joinpath(@__DIR__, "..")
isfile(joinpath(_results, "R.jls")) ||
	error("R.jls not found — run main.jl first.")
isfile(joinpath(_results, "W.jls")) ||
	error("W.jls not found — run main.jl first.")

println("Loading ROM …")
R = deserialize(joinpath(_results, "R.jls"))
W = deserialize(joinpath(_results, "W.jls"))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Parse Re₀ from header
# ─────────────────────────────────────────────────────────────────────────────

function _parse_Re₀(path)
	for line in eachline(path)
		m = match(r"Re₀\s*=\s*([\d.]+)", line)
		isnothing(m) || return parse(Float64, m.captures[1])
	end
	error("Re₀ not found in $path")
end

const Re₀ = _parse_Re₀(joinpath(_results, "reduced_dynamics.txt"))

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Realified ż₁ polynomial in (a₁, a₂, η′)
# ─────────────────────────────────────────────────────────────────────────────

const conj_map = [2, 1, 3]   # z₁ ↔ z₂; η′ maps to itself
const R1_full  = extract_component(realify(R.poly, conj_map), 1)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

function poly_truncate(p::DensePolynomial{T, NVAR, 1}, max_order::Int) where {T, NVAR}
	dict = Dict{SVector{NVAR, Int}, T}()
	for (α, c) in each_term(p)
		sum(α) <= max_order || continue
		dict[α] = c
	end
	return similar_poly(dict)
end

function find_limit_cycle(R1_k, η)
	f(ρ) = real(evaluate(R1_k, SVector(ρ, 0.0, η)))
	# Log-scan [1e-15, 1e4] to locate sign change (robust to any eigenvector normalisation)
	rhos = [10.0^k for k in range(-15, 4, 60)]
	vals = f.(rhos)
	for i in 1:(length(rhos)-1)
		vals[i] * vals[i+1] < 0.0 || continue
		return find_zero(f, (rhos[i], rhos[i+1]), Bisection())
	end
	return NaN
end

# Find Re_c by bisecting on the linear growth rate σ(Re) = Re(R₁(ε,0,η))/ε
function find_Re_c(R1_k; bracket = (47.0, 57.0))
	ε = 1e-12
	g(Re) = real(evaluate(R1_k, SVector(ε, 0.0, 1.0 / Re - 1.0 / Re₀))) / ε
	g(bracket[1]) * g(bracket[2]) >= 0.0 && return NaN
	return find_zero(g, bracket, Bisection())
end

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Backbone at one truncation order
# ─────────────────────────────────────────────────────────────────────────────

const ORDERS = [3]
const ORDER_COLORS = [:darkorange]#, :steelblue]
const RE_RANGE = range(47.0, 57.0, 10000)

function backbone_at_order(ord)
	R1_k    = poly_truncate(R1_full, ord)
	Re_pts  = Float64[]
	rho_pts = Float64[]
	Ω_pts  = Float64[]
	for Re in RE_RANGE
		η = 1.0 / Re - 1.0 / Re₀
		ρ = find_limit_cycle(R1_k, η)
		isnan(ρ) && continue
		Ω = imag(evaluate(R1_k, SVector(ρ, 0.0, η))) / ρ
		push!(Re_pts, Re)
		push!(rho_pts, ρ)
		push!(Ω_pts, Ω)
	end
	Re_c = find_Re_c(R1_k)
	return (; Re_pts, rho_pts, Ω_pts, Re_c)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Compute
# ─────────────────────────────────────────────────────────────────────────────

@printf("Re₀ = %.4f\n", Re₀)
let ε = 1e-12
	λ_check = evaluate(R1_full, SVector(ε, 0.0, 0.0)) / ε
	@printf("  λ₁ (from R₁) ≈ %+.6f %+.6f·i  (should match header)\n",
		real(λ_check), imag(λ_check))
end

println("\nComputing backbone …")
all_results = map(ORDERS) do ord
	print("  order $ord … ")
	r = backbone_at_order(ord)
	n = length(r.Re_pts)
	if n > 0
		@printf("%d pts,  Re ∈ [%.3f, %.3f],  Re_c ≈ %.4f\n",
			n, minimum(r.Re_pts), maximum(r.Re_pts), isnan(r.Re_c) ? NaN : r.Re_c)
	else
		@printf("no limit cycles found  (Re_c ≈ %.4f)\n",
			isnan(r.Re_c) ? NaN : r.Re_c)
	end
	(order = ord, r...)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Figure 1 — bifurcation_diagram.png  (ρ vs Re)
# ─────────────────────────────────────────────────────────────────────────────

Re_lo = Float64(RE_RANGE[1])
Re_hi = Float64(RE_RANGE[end])

plt_bif = plot(;
	xlabel = "Reynolds number  Re",
	ylabel = "Modal amplitude  ρ",
	title  = "Kármán vortex street — Hopf bifurcation",
	size   = (700, 500),
	dpi    = 150,
	legend = :topleft,
)

# Fixed-point branch: ρ = 0, drawn per order (Re_c is order-dependent)
for (res, clr) in zip(all_results, ORDER_COLORS)
	Re_c = res.Re_c
	isnan(Re_c) && continue
	# Stable segment (Re ≤ Re_c)
	plot!(plt_bif, [Re_lo, Re_c], [0.0, 0.0];
		color = clr, lw = 2, ls = :solid, label = false)
	# Unstable segment (Re ≥ Re_c)
	plot!(plt_bif, [Re_c, Re_hi], [0.0, 0.0];
		color = clr, lw = 2, ls = :dash, label = false)
	# Bifurcation point marker
	scatter!(plt_bif, [Re_c], [0.0];
		color = clr, ms = 6, markerstrokewidth = 0, label = false)
end

# Limit-cycle branch
for (res, clr) in zip(all_results, ORDER_COLORS)
	isempty(res.Re_pts) && continue
	plot!(plt_bif, res.Re_pts, res.rho_pts;
		lw = 2, color = clr, label = "order $(res.order)")
end

savefig(plt_bif, joinpath(@__DIR__, "bifurcation_diagram.png"))
println("\nSaved → bifurcation_diagram.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Figure 2 — hopf_3d.png  (Re, a₁, a₂) Hopf tube
# ─────────────────────────────────────────────────────────────────────────────

plt3d = plot3d(;
	xlabel = "Re",
	ylabel = "a₁  (Re z₁)",
	zlabel = "a₂  (Im z₁)",
	title  = "Kármán vortex street — Hopf tube",
	size   = (800, 600),
	dpi    = 150,
	legend = :topleft,
	camera = (30, 20),
)

N_θ = 60
θ_circle = range(0, 2π, N_θ + 1)[1:N_θ]

for (res, clr) in zip(all_results, ORDER_COLORS)
	Re_c = res.Re_c
	isnan(Re_c) && continue

	# Fixed-point line: stable (Re ≤ Re_c)
	Re_stable = range(Re_lo, Re_c, 40)
	plot3d!(plt3d, collect(Re_stable), zeros(40), zeros(40);
		color = clr, lw = 2, ls = :solid,
		label = "fixed point — order $(res.order)")

	# Fixed-point line: unstable (Re ≥ Re_c)
	Re_unstable = range(Re_c, Re_hi, 80)
	plot3d!(plt3d, collect(Re_unstable), zeros(80), zeros(80);
		color = clr, lw = 2, ls = :dash, label = false)

	# Bifurcation point
	scatter3d!(plt3d, [Re_c], [0.0], [0.0];
		color = clr, ms = 5, markerstrokewidth = 0, label = false)

	# Limit-cycle circles: ~25 evenly spaced
	if !isempty(res.Re_pts)
		n_pts = length(res.Re_pts)
		stride = max(1, n_pts ÷ 25)
		for i in 1:stride:n_pts
			Re_i = res.Re_pts[i]
			ρ_i = res.rho_pts[i]
			plot3d!(plt3d,
				fill(Re_i, N_θ),
				ρ_i .* cos.(θ_circle),
				ρ_i .* sin.(θ_circle);
				color = clr, lw = 0.8, alpha = 0.7, label = false)
		end
	end
end

savefig(plt3d, joinpath(@__DIR__, "hopf_3d.png"))
println("Saved → hopf_3d.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Figure 3 — strouhal_vs_Re.png  (Ω vs Re)
# ─────────────────────────────────────────────────────────────────────────────

plt_str = plot(;
	xlabel = "Reynolds number  Re",
	ylabel = "Strouhal angular frequency  Ω  (rad/s)",
	title  = "Kármán vortex street — Strouhal frequency",
	size   = (700, 500),
	dpi    = 150,
	legend = :topleft,
)
vline!(plt_str, [Re₀]; ls = :dash, color = :gray, lw = 1, label = "Re₀ = $Re₀")

for (res, clr) in zip(all_results, ORDER_COLORS)
	isempty(res.Re_pts) && continue
	plot!(plt_str, res.Re_pts, res.Ω_pts; lw = 2, color = clr, label = "order $(res.order)")
end

savefig(plt_str, joinpath(@__DIR__, "strouhal_vs_Re.png"))
println("Saved → strouhal_vs_Re.png")
