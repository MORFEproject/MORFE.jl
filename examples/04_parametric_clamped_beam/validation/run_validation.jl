"""
	run_validation.jl

Non-parametric validation of the two-parameter beam ROM.

For each test point (θ₁*, θ₂*):
  1. Deform the reference mesh to the exact geometry x*(θ₁*, θ₂*).
  2. Assemble exact K, M, C and nonlinear maps on the deformed mesh.
  3. Solve the DPIM cohomological equations → exact (W★, R★).
  4. Compare backbone and ω₀ against the parametric ROM evaluated at (θ₁*, θ₂*).

Outputs:
  results/figures/validation_backbone_t1=*.png  — per test-point overlay
  results/figures/validation_omega0.png          — ω₀ comparison bar chart
  results/data/validation_metrics.csv            — numerical metrics table
"""

using Pkg: Pkg
const _val_env = joinpath(@__DIR__, "validation_env")
Pkg.activate(_val_env)
if !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "..")))
	# Add all packages except Arpack first
	Pkg.add(["Ferrite", "FerriteGmsh", "LinearMaps", "Tensors", "StaticArrays", "Plots"])
	# Then add Arpack with version constraint (this will install/downgrade to 0.5.3)
	Pkg.add(Pkg.PackageSpec(name = "Arpack", version = "0.5.3"))

end
Pkg.pin("Arpack")

using MORFE
using Ferrite, FerriteGmsh
using SparseArrays, LinearAlgebra, Statistics
using Arpack, LinearMaps
using Tensors, StaticArrays
using MORFE.Polynomials: DensePolynomial, evaluate, extract_component, each_term, similar_poly
using MORFE.Realification: realify
using Serialization
using Printf
using Plots
ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "exact_geometry_assembly.jl"))
include(joinpath(@__DIR__, "exact_nonlinear_maps.jl"))
include(joinpath(@__DIR__, "..", "plotting", "backbone_plots.jl"))

# ------------------------------------------------------------------
# Constants  (match main.jl)
# ------------------------------------------------------------------
const _msh = joinpath(@__DIR__, "..", "..", "..", "benchmark", "ferrite", "beam_h27_10x2x2.msh")
const _data = joinpath(@__DIR__, "..", "results", "data")
const _figs = joinpath(@__DIR__, "..", "results", "figures")
const E_val = 160e3
const ν_val = 0.22
const ρ_val = 2.32e-3
const λ_lame = (E_val * ν_val) / ((1 + ν_val) * (1 - 2ν_val))
const μ_lame = E_val / (2(1 + ν_val))
const α_damp = 0.5369754008568333 / 500.0
const β_damp = 0.0
const J₁_def = Tensor{2, 3, Float64}((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0)
const ROM_VAL = 2
const MAX_DEG = 5
const CONJ_PARAM = [2, 1, 3, 4]
const CONJ_EXACT = [2, 1]

# ------------------------------------------------------------------
# Test points
# ------------------------------------------------------------------
const test_points = [
	(0.0, 0.0),    # trivial: Δω₀_rel ≈ 0 (smoke test)
	(0.05, 0.0),
	(0.10, 0.0),
	(0.20, 0.0),
	(-0.05, 0.0),
	(-0.1, 0.0),
	(-0.2, 0.0),
	(0.0, 1.5),
	(0.05, 1.5),
	(0.0, 3.0),
	(0.0, 4.5),
	(0.0, 6.0),
	(0.0, 7.0),
]

# ------------------------------------------------------------------
# Reference FE setup  (built once from reference mesh)
# ------------------------------------------------------------------
isfile(_msh) || error("Mesh not found at $_msh.  Run generate_beam_mesh.jl first.")
grid_ref = togrid(_msh)
ip       = Lagrange{RefHexahedron, 2}()^3
geo_ip   = Lagrange{RefHexahedron, 2}()
qr       = QuadratureRule{RefHexahedron}(3)

dh_ref = DofHandler(grid_ref)
add!(dh_ref, :u, ip)
close!(dh_ref)
ch_ref = ConstraintHandler(dh_ref)
add!(ch_ref, Dirichlet(:u, getfacetset(grid_ref, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch_ref)
update!(ch_ref, 0.0)
free_ref          = sort(setdiff(1:ndofs(dh_ref), ch_ref.prescribed_dofs))
free_to_local_ref = Dict(d => i for (i, d) in enumerate(free_ref))

# ------------------------------------------------------------------
# Load parametric ROM + arch mode
# ------------------------------------------------------------------
isfile(joinpath(_data, "R.jls")) ||
	error("results/data/R.jls not found.  Run main.jl first.")

R_param = deserialize(joinpath(_data, "R.jls"))
W_param = deserialize(joinpath(_data, "W.jls"))

arch_mode_free = if isfile(joinpath(_data, "arch_mode.jls"))
	deserialize(joinpath(_data, "arch_mode.jls"))
else
	# Fallback: recompute from the reference eigenproblem
	println("arch_mode.jls not found — recomputing from reference eigenproblem …")
	_cv0 = CellValues(qr, ip, geo_ip)
	_K0f = allocate_matrix(dh_ref)
	_M0f = allocate_matrix(dh_ref)
	assemble_KM!(_K0f, _M0f, dh_ref, _cv0, λ_lame, μ_lame, ρ_val)
	_K0 = _K0f[free_ref, free_ref]
	_M0 = _M0f[free_ref, free_ref]
	_slvr = StructureModalDampingEigensolver(4, α_damp, β_damp)
	_ep = solve_eigenproblem(_K0, _M0, _slvr; sorter! = (args...) -> nothing)
	(_, _Y, _) = get_eigenpairs(_ep)
	_arch = real(_Y[:, 1, 1])
	_arch ./= maximum(abs, _arch)
	_arch
end

# ------------------------------------------------------------------
# Parametric-side backbone helpers
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

R1_param    = extract_component(realify(R_param.poly, CONJ_PARAM), 1)
dR1dx_param = poly_deriv(R1_param, 1)

ω₀_param(θ₁, θ₂) = imag(evaluate(dR1dx_param, [0.0, 0.0, θ₁, θ₂]))
backbone_Ω_param(r, θ₁, θ₂) =
	imag(evaluate(R1_param, [max(r, 1e-12), 0.0, θ₁, θ₂])) / max(r, 1e-12)

# ------------------------------------------------------------------
# Validation loop
# ------------------------------------------------------------------
const r_range = range(1e-6, 50.0, 400)
results_table = NamedTuple[]
mkpath(_data)
mkpath(_figs)

for (θ₁_star, θ₂_star) in test_points
	println("\n── Validation at (θ₁*, θ₂*) = ($θ₁_star, $θ₂_star) ──")

	# ── Exact side ────────────────────────────────────────────────────
	grid_def = deform_grid(grid_ref, dh_ref, θ₁_star, θ₂_star,
		J₁_def, arch_mode_free, free_to_local_ref)
	ex = assemble_exact_KM(grid_def, ip, geo_ip, qr,
		λ_lame, μ_lame, ρ_val, α_damp, β_damp)
	(; dh, cv, free_to_local, n_free, K, M, C) = ex

	mset_ex = all_multiindices_up_to(ROM_VAL, MAX_DEG; min_degree = 1)
	n_ex = length(mset_ex)
	term_quad★, term_cube★ = build_exact_nonlinear_maps(
		dh, cv, free_to_local, n_free, λ_lame, μ_lame; max_unique_cols = n_ex)

	model★ = NDOrderModel((K, C, M), (term_quad★, term_cube★))

	solver★ = StructureModalDampingEigensolver(ROM_VAL + 2, α_damp, β_damp)
	ep★     = solve_eigenproblem(K, M, solver★; sorter! = (args...) -> nothing)
	select_master_modes_by_sorting(ep★, ROM_VAL)
	(eigs★, Y★, X★) = get_eigenpairs(ep★)

	master_eigs★  = SVector{ROM_VAL, ComplexF64}(eigs★[1:ROM_VAL])
	master_modes★ = Y★[:, 1, 1:ROM_VAL]
	left_modes★   = X★[:, 1:ROM_VAL]
	# ORD=2 system (K,C,M) → 1 derivative level
	derivs★ = zeros(ComplexF64, n_free, 1, ROM_VAL)
	for r in 1:ROM_VAL
		derivs★[:, 1, r] .= Y★[:, 2, r]
	end

	res_set★ = resonance_set_from_complex_normal_form_style(
		mset_ex, Vector{ComplexF64}(master_eigs★), 0.05)
	print("  Cohomological solve: ")
	@time W★, R★ = solve_cohomological_problem(
		model★, mset_ex,
		master_eigs★, master_modes★, left_modes★, res_set★;
		master_modes_derivatives = derivs★,
		conjugate_permutation    = CONJ_EXACT)

	R1★ = extract_component(realify(R★.poly, CONJ_EXACT), 1)
	dR1dx★ = poly_deriv(R1★, 1)
	ω₀★ = imag(evaluate(dR1dx★, [0.0, 0.0]))
	Ω_ex_vec = [imag(evaluate(R1★, [max(r, 1e-12), 0.0])) / max(r, 1e-12) for r in r_range]

	# ── Parametric side at (θ₁*, θ₂*) ────────────────────────────────
	ω₀_p = ω₀_param(θ₁_star, θ₂_star)
	Ω_p_vec = [backbone_Ω_param(r, θ₁_star, θ₂_star) for r in r_range]

	# ── Metrics ───────────────────────────────────────────────────────
	# NOTE: exact and parametric modes are independently normalised; |z|=r is not
	# invariant across models. Δω₀_rel compares the invariant linear eigenvalue.
	# Backbone RMS compares frequency at same r — valid as a truncation-error indicator.
	Δω₀_rel = abs(ω₀★ - ω₀_p) / abs(ω₀★)
	Δω_vec = abs.(Ω_ex_vec .- Ω_p_vec)
	rms_err = sqrt(mean(Δω_vec .^ 2))
	max_err = maximum(Δω_vec)

	push!(results_table, (;
		θ₁ = θ₁_star, θ₂ = θ₂_star,
		ω₀_exact = ω₀★, ω₀_param = ω₀_p,
		Δω₀_rel, rms_err, max_err))

	@printf "  ω₀ exact = %.6f   param = %.6f   Δω₀_rel = %.2e\n" ω₀★ ω₀_p Δω₀_rel
	@printf "  Backbone RMS err = %.2e   max = %.2e  (rad/ms)\n" rms_err max_err

	# ── Overlay figure ────────────────────────────────────────────────
	lbl_ex  = @sprintf("exact  (θ₁=%.3f, θ₂=%.3f)", θ₁_star, θ₂_star)
	plt_bk  = plot_backbone_absolute(
	r_range, [Ω_ex_vec, Ω_p_vec], [nothing, nothing],
	[lbl_ex, "parametric ROM"];
	colors = [:black, :red],
	title  = @sprintf("Exact vs parametric  (θ₁=%.3f, θ₂=%.3f)", θ₁_star, θ₂_star))
	figname = @sprintf("validation_backbone_t1=%.3f_t2=%.3f.png", θ₁_star, θ₂_star)
	savefig(plt_bk, joinpath(_figs, figname))
	println("  Saved → $(joinpath(_figs, figname))")
end

# ------------------------------------------------------------------
# Summary: ω₀ comparison plot + CSV
# ------------------------------------------------------------------
ω₀_ex_all = [row.ω₀_exact for row in results_table]
ω₀_p_all  = [row.ω₀_param for row in results_table]
tick_labels  = [@sprintf("(%.2f,%.2f)", row.θ₁, row.θ₂) for row in results_table]
idx          = 1:length(test_points)

plt_ω0 = plot(idx, ω₀_ex_all;
	label = "exact",
	lw = 2.0, marker = :circle,
	xlabel = "Test point  (θ₁, θ₂)",
	ylabel = "Linear eigenfrequency  ω₀  (rad/ms)",
	title = "ω₀: exact vs parametric ROM",
	xticks = (idx, tick_labels),
	size = (900, 500), dpi = 150)
plot!(plt_ω0, idx, ω₀_p_all;
	label = "parametric", lw = 2.0, ls = :dash, marker = :square)
savefig(plt_ω0, joinpath(_figs, "validation_omega0.png"))

open(joinpath(_data, "validation_metrics.csv"), "w") do io
	println(io, "theta1_star,theta2_star,omega0_exact,omega0_param," *
				"delta_omega0_rel,backbone_rms_err,backbone_max_err")
	for row in results_table
		println(io, "%.6f,%.6f,%.6f,%.6f,%.4e,%.4e,%.4e\n",
			row.θ₁, row.θ₂, row.ω₀_exact, row.ω₀_param, row.Δω₀_rel, row.rms_err, row.max_err)
	end
end

println("\nValidation complete.")
println("  Figures : $_figs/validation_*.png")
println("  Metrics : $(joinpath(_data, "validation_metrics.csv"))")
