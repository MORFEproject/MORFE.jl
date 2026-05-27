"""
	compute_backbone.jl

Post-processing script for the parametric cantilever beam ROM.

Loads R from results/, realifies the z₁ equation, constructs the symbolic
Jacobian polynomial, then for each θ value in θ_values:

  (1) Extracts ω₀(θ) = Im[∂R1_cplx/∂x(0,0,θ)]  (linear eigenfrequency).
  (2) Computes the analytical backbone  Ω(r, θ) = Im[R1_cplx(r,0,θ)] / r.
  (3) Validates with BifurcationKit (equilibrium continuation of
	  G(r,Ω)=0 with analytical Jacobian from the symbolic derivative).

Figure 1: absolute backbone  Ω vs |z₁|  for each θ.
Figure 2: nonlinear shift  (Ω − ω₀(θ)) vs |z₁|  (all start at 0).

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
# 2.  R1_cplx : ℝ³→ℂ   (realified z₁ equation in (x, y, θ))
# ------------------------------------------------------------------
conj_map = [2, 1, 3]
R1_cplx = extract_component(realify(R.poly, conj_map), 1)

# ------------------------------------------------------------------
# 3.  Symbolic partial derivatives of R1_cplx
# ------------------------------------------------------------------

"""
	poly_deriv(p, var_idx)

Return the partial derivative of scalar polynomial `p` w.r.t. variable `var_idx`
as a new `DensePolynomial`.  Built from `each_term` + `similar_poly`.
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
const dR1dy = poly_deriv(R1_cplx, 2)  # ∂R1_cplx/∂y

# ------------------------------------------------------------------
# 4.  Linear eigenfrequency as a function of θ
#     ω₀(θ) = Im[∂R1_cplx/∂x(0, 0, θ)]  =  Im[λ₁(θ)]
# ------------------------------------------------------------------
ω₀_of_θ(θ_val) = imag(evaluate(dR1dx, [0.0, 0.0, θ_val]))

# Announce the θ=0 value for reference
let ω₀ = ω₀_of_θ(0.0)
	println(@sprintf("Linear frequency ω₀(0) = %.6f rad/s   T₀ = %.6f s", ω₀, 2π / ω₀))
end

# ------------------------------------------------------------------
# 5.  Parameters
# ------------------------------------------------------------------
const r_max = 5.0
const r_seed = 0.5    # BK seed — large enough that dΩ/dr is non-negligible
const θ_values = [-0.5, -0.25, 0.0, 0.25, 0.5]

# ------------------------------------------------------------------
# 6.  BifurcationKit backbone continuation for a single θ
#
# Backbone condition:
#   G(r, Ω) = Im[R1_cplx(r, 0, θ)] / r  −  Ω  =  0
#
# Analytical Jacobian (quotient rule, using the symbolic dR1dx):
#   dG/dr = (Im[dR1dx(r, 0, θ)] · r  −  Im[R1_cplx(r, 0, θ)]) / r²
# ------------------------------------------------------------------
function run_bk_backbone(θ_val)
	ω₀ = ω₀_of_θ(θ_val)
	Ω₀ = imag(evaluate(R1_cplx, [r_seed, 0.0, θ_val])) / r_seed

	G(x, p) = begin
		r = max(x[1], 1e-10)
		[imag(evaluate(R1_cplx, [r, 0.0, θ_val])) / r - p[1]]
	end

	J(x, _) = begin
		r = max(x[1], 1e-10)
		val = imag(evaluate(R1_cplx, [r, 0.0, θ_val]))
		dval = imag(evaluate(dR1dx, [r, 0.0, θ_val]))
		reshape([(dval * r - val) / r^2], 1, 1)
	end

	prob = BifurcationProblem(
		G, [r_seed], [Ω₀], @optic(_[1]);
		J = J,
		record_from_solution = (x, p; k...) -> (r = x[1], Ω = p),
	)

	opts = ContinuationPar(
		p_min = ω₀ * 0.80,
		p_max = ω₀ * 1.50,
		max_steps = 2000,
		ds = 0.01,
		dsmax = 0.2,
	)

	return try
		continuation(prob, PALC(), opts; verbosity = 0)
	catch e
		@warn "Continuation failed for θ=$θ_val: $e"
		nothing
	end
end

# ------------------------------------------------------------------
# 7.  Run all branches
# ------------------------------------------------------------------
println("Running BifurcationKit backbone continuations …")
branches = map(θ_values) do θ_val
	br = run_bk_backbone(θ_val)
	if !isnothing(br)
		r_bk = [s.r for s in br.branch]
		Ω_bk = [s.Ω for s in br.branch]
		@printf "  θ=%.3f: %d pts, r ∈ [%.3f, %.3f], Ω ∈ [%.6f, %.6f]\n" θ_val length(br.branch) extrema(r_bk)... extrema(Ω_bk)...
	end
	br
end

# ------------------------------------------------------------------
# 8.  Analytical curves
# ------------------------------------------------------------------
r_range = range(1e-6, r_max, 400)

Ω_curves = map(θ_values) do θ_val
	[imag(evaluate(R1_cplx, [r, 0.0, θ_val])) / r for r in r_range]
end

ω₀_vals = ω₀_of_θ.(θ_values)

# ------------------------------------------------------------------
# 9.  Figure 1: absolute backbone  Ω vs |z₁|
# ------------------------------------------------------------------
println("Plotting …")

all_Ω = vcat(Ω_curves...)
Ω_lo = minimum(all_Ω) - 0.10 * (maximum(all_Ω) - minimum(all_Ω) + 1e-12)
Ω_hi = maximum(all_Ω) + 0.20 * (maximum(all_Ω) - minimum(all_Ω) + 1e-12)

clrs = palette(:tab10)

plt1 = plot(;
	xlabel = "Backbone frequency  Ω  (rad/s)",
	ylabel = "Modal amplitude  |z₁|",
	title = "Backbone curves — beam ROM",
	xlims = (Ω_lo, Ω_hi),
	ylims = (0, r_max * 1.05),
	size = (700, 600),
	dpi = 150,
)

for (k, θ_val) in enumerate(θ_values)
	plot!(plt1, Ω_curves[k], collect(r_range);
		lw = 2.5, color = clrs[k], label = "θ=$(θ_val) analytical")
	br = branches[k]
	if !isnothing(br) && length(br.branch) > 1
		r_bk = [s.r for s in br.branch]
		Ω_bk = [s.Ω for s in br.branch]
		mask = r_bk .<= r_max
		scatter!(plt1, Ω_bk[mask], r_bk[mask];
			color = clrs[k], ms = 3, markerstrokewidth = 0,
			label = "θ=$(θ_val) BK")
	end
end

# ------------------------------------------------------------------
# 10.  Figure 2: nonlinear shift  (Ω − ω₀(θ)) vs |z₁|
# ------------------------------------------------------------------
all_shift = vcat([Ω_curves[k] .- ω₀_vals[k] for k in eachindex(θ_values)]...)
shift_lo = minimum(all_shift) - 0.10 * (maximum(all_shift) - minimum(all_shift) + 1e-12)
shift_hi = maximum(all_shift) + 0.20 * (maximum(all_shift) - minimum(all_shift) + 1e-12)

plt2 = plot(;
	xlabel = "Nonlinear frequency shift  Ω − ω₀(θ)  (rad/s)",
	ylabel = "Modal amplitude  |z₁|",
	title = "Nonlinear backbone shift — beam ROM",
	xlims = (shift_lo, shift_hi),
	ylims = (0, r_max * 1.05),
	size = (700, 600),
	dpi = 150,
)

for (k, θ_val) in enumerate(θ_values)
	ω₀ = ω₀_vals[k]
	plot!(plt2, Ω_curves[k] .- ω₀, collect(r_range);
		lw = 2.5, color = clrs[k], label = "θ=$(θ_val) analytical")
	br = branches[k]
	if !isnothing(br) && length(br.branch) > 1
		r_bk = [s.r for s in br.branch]
		Ω_bk = [s.Ω for s in br.branch]
		mask = r_bk .<= r_max
		scatter!(plt2, Ω_bk[mask] .- ω₀, r_bk[mask];
			color = clrs[k], ms = 3, markerstrokewidth = 0,
			label = "θ=$(θ_val) BK")
	end
end

# ------------------------------------------------------------------
# 11.  Save
# ------------------------------------------------------------------
mkpath(_results)
savefig(plt1, joinpath(_results, "backbone_curves.png"))
savefig(plt2, joinpath(_results, "backbone_shift.png"))
println("Saved → $(_results)/backbone_curves.png")
println("Saved → $(_results)/backbone_shift.png")
