"""
	compute_backbone.jl

Post-processing for the arch_2_force ROM in (z₁, z₂).

Loads R and W from results/, realifies the z₁ equation, and computes backbone
curves via BifurcationKit orthogonal collocation with Doedel regularisation at
orders 3, 5, 7, and 9.  Both the ROM field f and the parametrisation map W are
truncated to the same order at each level for a consistent comparison.

NOTE: This backbone computation assumes an AUTONOMOUS ROM (no external forcing).
Run arch_2_force.jl with forces = [] in the config to produce the conservative W
and R before using this script.  For the forced-response FRF, a separate parameter
sweep is needed.

Figures produced in results/:
  backbone_curves.png   — Ω vs |z₁|          (four orders overlaid)
  backbone_shift.png    — (Ω − ω₀) vs |z₁|   (four orders overlaid)
  backbone_physical.png — physical amplitude |u(dof_idx)| vs Ω

Amplitude scale (ε, r_max): set in mm/kg/s units.  Inspect _phi1_amps after running
arch_2_force.jl to choose appropriate values for the arch geometry.

Run after arch_2_force.jl (autonomous config) has produced results/R.jls and results/W.jls.
"""

using Pkg: Pkg
const _backbone_env = joinpath(@__DIR__, "../../../backbone_env")
Pkg.activate(_backbone_env)
if !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../../../../../..")))
	Pkg.add(["BifurcationKit", "Plots", "StaticArrays"])
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
const _results = joinpath(@__DIR__, "..")
isfile(joinpath(_results, "R.jls")) ||
	error("results/R.jls not found.  Run arch_2_force.jl (autonomous config) first.")
isfile(joinpath(_results, "W.jls")) ||
	error("results/W.jls not found.  Run arch_2_force.jl (autonomous config) first.")

println("Loading ROM …")
R = deserialize(joinpath(_results, "R.jls"))
W = deserialize(joinpath(_results, "W.jls"))

# ------------------------------------------------------------------
# 2.  Realify full R → R1_cplx in (x, y) coordinates
# ------------------------------------------------------------------
conj_map = [2, 1]    # z₁↔z₂
R1_cplx = extract_component(realify(R.poly, conj_map), 1)

# ------------------------------------------------------------------
# 3.  Polynomial helpers
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

function poly_truncate(p::DensePolynomial{T, NVAR, 1}, max_order::Int) where {T, NVAR}
	dict = Dict{SVector{NVAR, Int}, T}()
	for (α, c) in each_term(p)
		sum(α) <= max_order || continue
		dict[α] = c
	end
	return similar_poly(dict)
end

# Coordinate-rescaled field  dζ/dτ = R1(εζ)/(εω₀) = R1_scaled(ζ),
# obtained by scaling each coefficient  cₐ → cₐ · ε^(|α|−1) / ω₀.
function poly_scale_coords(p::DensePolynomial{T, NVAR, 1}, ε, denom) where {T, NVAR}
	dict = Dict{SVector{NVAR, Int}, T}()
	for (α, c) in each_term(p)
		dict[α] = c * ε^(sum(α) - 1) / denom
	end
	return similar_poly(dict)
end

# ------------------------------------------------------------------
# 4.  Shared constants
# ------------------------------------------------------------------
const ORDERS = [3, 5, 7, 9]
const ORDER_COLORS = [:darkorange, :steelblue, :crimson, :darkgreen]

# ω₀ is the linear coefficient — order-independent (purely imaginary in this ROM)
const ω₀_bk = imag(evaluate(poly_deriv(poly_truncate(R1_cplx, 1), 1), [0.0, 0.0]))
const σ₀_bk = real(evaluate(poly_deriv(poly_truncate(R1_cplx, 1), 1), [0.0, 0.0]))

# Amplitude scale in mm/kg/s units.
# ε:     reference modal amplitude (ζ = z₁/ε); tune after inspecting eigenvector amplitudes.
# r_max: physical |z₁| range to plot.
# Set ε ≈ characteristic amplitude where cubic term ≈ linear term.
const ε = 1e-4                       # reference modal amplitude (ζ = z₁/ε)
const r_max = 5e-4                   # physical |z₁| range to plot

@printf("  ω₀ = %.6e rad/s   f₀ = %.4f Hz   σ₀ = %.4e\n",
	ω₀_bk, ω₀_bk / (2π), σ₀_bk)

# ------------------------------------------------------------------
# 5.  BifurcationKit — Doedel-regularised orthogonal collocation
# ------------------------------------------------------------------
#
# Doedel regularisation replaces the conservative ODE  ζ′ = R1_scaled(ζ)
# with the augmented system
#
#   ζ′ = (1 + iλ) · R1_scaled(ζ)
#
# In realified (x, y) coordinates  (ζ = x + iy):
#
#   x′ = Re[R1] − λ · Im[R1]
#   y′ = Im[R1] + λ · Re[R1]
#
# Because the field is exactly conservative (σ₀ = 0), the backbone branch is a
# vertical line {λ = 0, r free}.  We seed PALC with two orbits at λ = 0
# and increasing |ζ| so the initial tangent points along amplitude.

# ── Shared BK parameters ──────────────────────────────────────────
const T̂₀ = 2π                 # linear period in rescaled time
const Ntst = 25
const m_col = 4
const n_unknowns = 2 * (1 + m_col * Ntst)
const r_cap = r_max             # amplitude stop for continuation

opts_bk = ContinuationPar(
	p_min = -1.0, p_max = 1.0,
	max_steps = 200,
	ds = 0.005, dsmax = 0.04, dsmin = 1e-5,
	detect_bifurcation = 0,
	detect_fold = false,
)

make_orbit_col(coll, rζ) =
	BifurcationKit.generate_solution(coll, t -> SVector(rζ * cos(t), rζ * sin(t)), T̂₀)

r_range_ana = range(1e-10, r_max, 500)

amp_phys(uvec) = ε * maximum(sqrt.(@views(uvec[1:(end-1)])[1:2:end] .^ 2 .+
								   @views(uvec[1:(end-1)])[2:2:end] .^ 2))

function record_col(u, p; k...)
	T̂ = u[end]
	(T = T̂, Ω = ω₀_bk * (2π / T̂), r = amp_phys(u), λ = p.p)
end

stop_at_amp(z, tau, step, contResult; kwargs...) = amp_phys(z.u) < r_cap

function continuation_coll_2pts(coll, u0, u1, alg, _cp, linear_algo;
	record_from_solution = nothing, finalise_solution = nothing, verbosity = 1)
	jacPO = BifurcationKit.generate_jacobian(coll, u1, BifurcationKit.getparams(coll))
	linear_algo = @set linear_algo.solver = BifurcationKit.FloquetWrapperLS(linear_algo.solver)
	cp = @set _cp.newton_options.linsolver =
		BifurcationKit.FloquetWrapperLS(_cp.newton_options.linsolver)
	alg = BifurcationKit.update(alg, cp, linear_algo)
	_finsol = BifurcationKit.modify_po_finalise(coll,
		(finalise_solution = finalise_solution,), coll.update_section_every_step)
	pars = BifurcationKit.getparams(coll.prob_vf)
	lens = BifurcationKit.getlens(coll.prob_vf)
	_rec = BifurcationKit.modify_po_record(coll, pars, lens;
		record_from_solution = record_from_solution, plot_solution = nothing)
	_plt = BifurcationKit.modify_po_plot(coll, pars, lens;
		record_from_solution = record_from_solution, plot_solution = nothing)
	wrap = BifurcationKit.WrapPOColl(coll, jacPO, u1, BifurcationKit.getparams(coll),
		BifurcationKit.getlens(coll), _plt, _rec)
	it = BifurcationKit.ContIterable(wrap, alg, cp;
		kind = BifurcationKit.PeriodicOrbitCont(),
		finalise_solution = _finsol, verbosity = verbosity)
	return BifurcationKit.continuation(it, u0, 0.0, u1, 0.0)
end

# ── W/DOF setup ─────────────────────────────────────────────────────────
const W_disp = DensePolynomial(@view(W.poly.coefficients[:, 1, :]),
	W.poly.multiindex_set)
const _exps_W = W.poly.multiindex_set.exponents
const _l10 = findfirst(α -> α[1] == 1 && α[2] == 0, _exps_W)
const _phi1_amps = abs.(@view W.poly.coefficients[:, 1, _l10])

# User-editable: override to pin a specific free-DOF index (1…FOM).
# Default: DOF with largest mode-1 amplitude (typically mid-arch z-direction).
const dof_idx = argmax(_phi1_amps)
# ────────────────────────────────────────────────────────────────────────

const W_i_full = extract_component(W_disp, dof_idx)  # RANK=1 scalar poly, full order
@printf("  DOF %d selected (eigenvector amplitude = %.4e mm)\n",
	dof_idx, _phi1_amps[dof_idx])

# ------------------------------------------------------------------
# 6.  Per-order loop
# ------------------------------------------------------------------
println("Running BifurcationKit Doedel backbone …")
all_results = map(ORDERS) do ord
	R1_top_k = poly_truncate(R1_cplx, ord)
	R1_scaled_k = poly_scale_coords(R1_top_k, ε, ω₀_bk)
	dR1dx_k = poly_deriv(R1_scaled_k, 1)
	dR1dy_k = poly_deriv(R1_scaled_k, 2)
	W_i_k = poly_truncate(W_i_full, ord)

	Ω_ana_k = [imag(evaluate(R1_top_k, [r, 0.0])) / r for r in r_range_ana]
	phys_ana_k = [abs(evaluate(W_i_k, SVector(r, r))) for r in r_range_ana]

	f_k = (u, p) -> begin
		λ = p[1]
		fv = evaluate(R1_scaled_k, [u[1], u[2]])
		SVector(real(fv) - λ * imag(fv), imag(fv) + λ * real(fv))
	end
	J_k = (u, p) -> begin
		λ = p[1]
		G = (1 + im * λ) * evaluate(dR1dx_k, [u[1], u[2]])
		H = (1 + im * λ) * evaluate(dR1dy_k, [u[1], u[2]])
		[real(G) real(H); imag(G) imag(H)]
	end

	prob_bif_k = BifurcationProblem(
		f_k, SVector(0.3, 0.0), [0.0], (@optic _[1]); J = J_k)
	prob_col_k = PeriodicOrbitOCollProblem(Ntst, m_col;
		N = 2,
		prob_vf = prob_bif_k,
		jacobian = BifurcationKit.DenseAnalytical(),
		update_section_every_step = 1,
		ϕ = zeros(n_unknowns),
		xπ = zeros(n_unknowns),
		∂ϕ = zeros(2, Ntst * m_col),
	)
	u0_k = make_orbit_col(prob_col_k, 0.15)
	u1_k = make_orbit_col(prob_col_k, 0.30)
	BifurcationKit.updatesection!(prob_col_k, u0_k, nothing)

	print("  order $ord … ")
	br_k = try
		continuation_coll_2pts(prob_col_k, u0_k, u1_k, PALC(tangent = Bordered()),
			opts_bk, MatrixBLS();
			record_from_solution = record_col,
			finalise_solution = stop_at_amp,
			verbosity = 0)
	catch e
		@warn "order $ord failed: $e"
		nothing
	end

	r_bk = Float64[]
	Ω_bk = Float64[]
	phys_bk = Float64[]
	if !isnothing(br_k)
		_m = [s.r <= r_max for s in br_k.branch]
		r_bk   = [s.r for s in br_k.branch][_m]
		Ω_bk   = [s.Ω for s in br_k.branch][_m]
		z1_bk  = r_bk ./ ε
		phys_bk = [abs(evaluate(W_i_k, SVector(r, r))) for r in z1_bk]
		@printf("%d pts, |z₁| ∈ [%.2e, %.2e]\n", length(r_bk), extrema(r_bk)...)
	else
		println("failed")
	end
	(order = ord, Ω_ana = Ω_ana_k, phys_ana = phys_ana_k,
		br = br_k, r_bk = r_bk, Ω_bk = Ω_bk, phys_bk = phys_bk)
end

# ------------------------------------------------------------------
# 7.  Figure 1: Ω vs |z₁|
# ------------------------------------------------------------------
println("Plotting …")

plt1 = plot(;
	xlabel = "Backbone frequency  Ω  (rad/s)",
	ylabel = "Modal amplitude  |z₁|",
	title  = "arch_2_force backbone (BK Doedel)",
	ylims  = (0, r_max * 1.05),
	size   = (800, 600),
	dpi    = 150,
)
for (res, clr) in zip(all_results, ORDER_COLORS)
	lbl = "order $(res.order)"
	plot!(plt1, res.Ω_ana, collect(r_range_ana);
		lw = 2, color = clr, label = "Analytical $lbl")
	!isempty(res.r_bk) && plot!(plt1, res.Ω_bk, res.r_bk;
		color = clr, lw = 1.5, ls = :dash,
		marker = :circle, ms = 3, markerstrokewidth = 0, label = "BK $lbl")
end

# ------------------------------------------------------------------
# 8.  Figure 2: nonlinear shift (Ω − ω₀) vs |z₁|
# ------------------------------------------------------------------
plt2 = plot(;
	xlabel = "Nonlinear frequency shift  Ω − ω₀  (rad/s)",
	ylabel = "Modal amplitude  |z₁|",
	title  = "arch_2_force backbone shift",
	ylims  = (0, r_max * 1.05),
	size   = (800, 600),
	dpi    = 150,
)
for (res, clr) in zip(all_results, ORDER_COLORS)
	lbl = "order $(res.order)"
	plot!(plt2, res.Ω_ana .- ω₀_bk, collect(r_range_ana);
		lw = 2, color = clr, label = "Analytical $lbl")
	!isempty(res.r_bk) && plot!(plt2, res.Ω_bk .- ω₀_bk, res.r_bk;
		color = clr, lw = 1.5, ls = :dash,
		marker = :circle, ms = 3, markerstrokewidth = 0, label = "BK $lbl")
end

# ------------------------------------------------------------------
# 9.  Figure 3: physical amplitude |u(dof_idx)| vs frequency
# ------------------------------------------------------------------
plt3 = plot(;
	xlabel = "Backbone frequency  Ω  (rad/s)",
	ylabel = @sprintf("|u(DOF %d)|  (mm)", dof_idx),
	title  = @sprintf("arch_2_force — physical amplitude at DOF %d", dof_idx),
	size   = (800, 600),
	dpi    = 150,
)
for (res, clr) in zip(all_results, ORDER_COLORS)
	lbl = "order $(res.order)"
	plot!(plt3, res.Ω_ana, res.phys_ana;
		lw = 2, color = clr, label = "Analytical $lbl")
	!isempty(res.Ω_bk) && plot!(plt3, res.Ω_bk, res.phys_bk;
		color = clr, lw = 1.5, ls = :dash,
		marker = :circle, ms = 3, markerstrokewidth = 0, label = "BK $lbl")
end

# ------------------------------------------------------------------
# 10.  Save figures
# ------------------------------------------------------------------
mkpath(_results)
savefig(plt1, joinpath(_results, "backbone_curves.png"))
savefig(plt2, joinpath(_results, "backbone_shift.png"))
savefig(plt3, joinpath(_results, "backbone_physical.png"))
println("Saved → $(_results)/backbone_curves.png")
println("Saved → $(_results)/backbone_shift.png")
println("Saved → $(_results)/backbone_physical.png")
