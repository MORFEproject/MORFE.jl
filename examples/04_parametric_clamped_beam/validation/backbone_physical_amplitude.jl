"""
	backbone_physical_amplitude.jl

Backbone-shift validation in PHYSICAL amplitude, in the style of
backbone_shift.png but with the y-axis given by the maximum physical
displacement reconstructed through the parametrisation W.

Cases: the full 3×3 product
	θ₁ ∈ {-0.1, 0.0, 0.1}  ×  θ₂ ∈ {0.0, 3.0, 6.0}

For each case:
  • Parametric side — the two-parameter ROM (W, R) from main.jl evaluated
	at (θ₁, θ₂).
  • Exact side — deform the mesh to the exact geometry, assemble exact
	K, M, C and nonlinear maps, run the full (non-parametric) DPIM solve
	→ (W★, R★).

The modal amplitude r = |z₁| is NOT comparable across the two models
(master modes are independently normalised), so each backbone is plotted
against the physical amplitude

	a(r) = max_φ  max_i  | u_i(r, φ) − u_i^static |,
	u(r, φ) = Re W_disp(r·e^{iφ}, r·e^{-iφ}, θ₁, θ₂),

i.e. the peak oscillation amplitude over one period, with the static
θ-induced offset u^static = Re W_disp(0, 0, θ₁, θ₂) subtracted
(W★ has no static part: its multiindex set starts at degree 1).
The maximum is attained at the beam midpoint — verified and printed.

Outputs:
  results/data/backbone_curves.csv                       — per-case (model, amplitude, shift) long-format data
  results/data/backbone_physical_metrics.csv             — numerical metrics

Figures are produced by a separate Python script:
  validation/plot_backbone.py

Run after main.jl has produced results/data/W.jls and R.jls.
"""

using Pkg: Pkg
const _val_env = joinpath(@__DIR__, "validation_env")
Pkg.activate(_val_env)
if !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "..")))
	Pkg.add(["Ferrite", "FerriteGmsh", "LinearMaps", "Tensors", "StaticArrays"])
	Pkg.add(Pkg.PackageSpec(name = "Arpack", version = "0.5.3"))
end
Pkg.instantiate()

using MORFE
using Ferrite, FerriteGmsh
using SparseArrays, LinearAlgebra, Statistics
using Arpack, LinearMaps
using Tensors, StaticArrays
using MORFE.Polynomials: DensePolynomial, evaluate, extract_component, each_term, similar_poly
using MORFE.Realification: realify
using Serialization
using Printf

include(joinpath(@__DIR__, "exact_geometry_assembly.jl"))
include(joinpath(@__DIR__, "exact_nonlinear_maps.jl"))

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
const MAX_DEG = 9   # z-expansion order of the exact reference — matches the
# anisotropic parametric ROM (deg_z ≤ 9, deg_θ ≤ 4) in main.jl
const CONJ_PARAM = [2, 1, 3, 4]
const CONJ_EXACT = [2, 1]

# ------------------------------------------------------------------
# Cases and sweep parameters
# ------------------------------------------------------------------
const θ₁_cases = (-0.1, 0.0, 0.1)
const θ₂_cases = (0.0, 3.0, 6.0)
const cases = [(θ₁, θ₂) for θ₂ in θ₂_cases for θ₁ in θ₁_cases]

const A_MAX   = 10.0    # physical amplitude cap for the plots (beam width = 10)
const R_MAX   = 80.0    # generous modal-amplitude sweep; curves truncated at A_MAX
const N_R     = 240     # points along each backbone
const N_PHASE = 24      # orbit phases for the max-over-period amplitude

# ------------------------------------------------------------------
# Reference FE setup
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

# free-local-DOF → (node, component) for locating the amplitude argmax.
# deform_grid preserves topology and dof construction order, so this map is
# valid for the exact (deformed-mesh) parametrisations as well.
dof_to_nodecomp = Dict{Int, Tuple{Int, Int}}()
for cell in CellIterator(dh_ref)
	dofs  = celldofs(cell)
	nodes = grid_ref.cells[Ferrite.cellid(cell)].nodes
	for l in eachindex(nodes), c in 1:3
		gdof = dofs[3(l-1)+c]
		haskey(free_to_local_ref, gdof) &&
			(dof_to_nodecomp[free_to_local_ref[gdof]] = (nodes[l], c))
	end
end

# ------------------------------------------------------------------
# Load parametric ROM + arch mode
# ------------------------------------------------------------------
isfile(joinpath(_data, "R.jls")) && isfile(joinpath(_data, "W.jls")) ||
	error("results/data/{R,W}.jls not found.  Run main.jl first.")
R_param = deserialize(joinpath(_data, "R.jls"))
W_param = deserialize(joinpath(_data, "W.jls"))

arch_mode_free = if isfile(joinpath(_data, "arch_mode.jls"))
	deserialize(joinpath(_data, "arch_mode.jls"))
else
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
# Polynomial helpers
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

"Displacement level of a Parametrisation as a rank-2 (FOM × L) polynomial."
displacement_poly(W) = DensePolynomial(
	Matrix(W.poly.coefficients[:, 1, :]), W.poly.multiindex_set)

"""
Keep only monomials where the sum of the first two variable exponents equals 1.
Variables 1 and 2 are z₁ and z̄₁ (or z₁★ and z̄₁★ for the exact ROM).
This retains the fundamental harmonic only, discarding DC (z₁z̄₁, etc.) and
second/higher harmonics (z₁², z̄₁², …) that arise from quadratic coupling at
large arch parameter θ₂. Both the parametric and exact amplitudes are then
proportional to r (∝ r) so the backbone shapes are directly comparable.
The fundamental-harmonic polynomial vanishes at z₁=0, so u_static = 0.
"""
function fundamental_harmonic(p::DensePolynomial{T, NVAR, 2}) where {T, NVAR}
	dict = Dict{SVector{NVAR, Int}, Vector{T}}()
	for (α, c) in each_term(p)
		α[1] + α[2] == 1 || continue
		dict[α] = collect(c)
	end
	return similar_poly(dict)
end

R1_param = extract_component(realify(R_param.poly, CONJ_PARAM), 1)
dR1dx_param = poly_deriv(R1_param, 1)
ω₀_param(θ₁, θ₂) = imag(evaluate(dR1dx_param, [0.0, 0.0, θ₁, θ₂]))
Ω_param(r, θ₁, θ₂) = imag(evaluate(R1_param, [max(r, 1e-12), 0.0, θ₁, θ₂])) /
					 max(r, 1e-12)

const Wdisp_param      = displacement_poly(W_param)
const Wdisp_param_fund = fundamental_harmonic(Wdisp_param)
const n_free_param     = size(Wdisp_param.coefficients, 1)

# ------------------------------------------------------------------
# Physical amplitude via W
# ------------------------------------------------------------------
"""
	physical_amplitude(Wdisp, r, θ_extra, u_static) -> (a, dof_argmax)

Peak oscillation amplitude over one orbit period:
	max over phase φ and DOF i of |Re Wdisp(r·e^{iφ}, r·e^{-iφ}, θ…)_i − u_static_i|.
Evaluating the complex W at (z, z̄) is equivalent to evaluating the realified
W at (x, y) = (r cos φ, r sin φ)  —  realify uses z = x + i y.
"""
function physical_amplitude(Wdisp::DensePolynomial{T, NVAR, 2}, r::Float64,
	θ_extra, u_static::Vector{Float64}) where {T, NVAR}
	a = 0.0
	i_max = 1
	for φ in range(0.0, 2π; length = N_PHASE + 1)[1:(end-1)]
		z = r * cis(φ)
		vals = ComplexF64[z, conj(z), θ_extra...]
		u = evaluate(Wdisp, vals)
		for i in eachindex(u)
			d = abs(real(u[i]) - u_static[i])
			d > a && (a = d; i_max = i)
		end
	end
	return a, i_max
end

"Amplitude curve a(r) over r_range; returns (a_vec, dof of overall argmax)."
function amplitude_curve(Wdisp, r_range, θ_extra, u_static)
	a_vec = zeros(length(r_range))
	i_glob = 1
	for (k, r) in enumerate(r_range)
		a_vec[k], i_k = physical_amplitude(Wdisp, r, θ_extra, u_static)
		k == length(r_range) && (i_glob = i_k)
	end
	return a_vec, i_glob
end

"Linear interpolation of y(x) at xq (x sorted ascending). Returns NaN outside."
function lerp(x::AbstractVector, y::AbstractVector, xq::Float64)
	(xq < x[1] || xq > x[end]) && return NaN
	k = searchsortedlast(x, xq)
	k == length(x) && return y[end]
	t = (xq - x[k]) / (x[k+1] - x[k])
	return (1 - t) * y[k] + t * y[k+1]
end

# ------------------------------------------------------------------
# Exact DPIM solve at one (θ₁, θ₂)   (same procedure as run_validation.jl)
# ------------------------------------------------------------------
function exact_rom(θ₁_star, θ₂_star)
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
	derivs★       = zeros(ComplexF64, n_free, 1, ROM_VAL)
	for r in 1:ROM_VAL
		derivs★[:, 1, r] .= Y★[:, 2, r]
	end

	left_derivs★ = ep★.left_eigenmodes_orders[:, 1:1, 1:ROM_VAL]

	res_set★ = resonance_set_from_complex_normal_form_style(
		mset_ex, Vector{ComplexF64}(master_eigs★), 0.05)
	print("  Cohomological solve: ")
	@time W★, R★ = solve_cohomological_problem(
		model★, mset_ex,
		master_eigs★, master_modes★, left_modes★, res_set★;
		master_modes_derivatives = derivs★,
		left_modes_derivatives = left_derivs★,
		conjugate_permutation    = CONJ_EXACT)
	return W★, R★
end

# ------------------------------------------------------------------
# Main loop over the 3×3 cases
# ------------------------------------------------------------------
const r_range = range(1e-6, R_MAX, N_R)
mkpath(_data)

curves = NamedTuple[]    # one entry per case
for (θ₁_star, θ₂_star) in cases
	println("\n── Case (θ₁, θ₂) = ($θ₁_star, $θ₂_star) ──")

	# ── Exact side ────────────────────────────────────────────────────
	W★, R★ = exact_rom(θ₁_star, θ₂_star)
	R1★      = extract_component(realify(R★.poly, CONJ_EXACT), 1)
	dR1dx★   = poly_deriv(R1★, 1)
	ω₀_ex   = imag(evaluate(dR1dx★, [0.0, 0.0]))
	Ω_ex      = [imag(evaluate(R1★, [max(r, 1e-12), 0.0])) / max(r, 1e-12)
	for r in r_range]
	Wdisp★      = displacement_poly(W★)
	n_free★     = size(Wdisp★.coefficients, 1)
	Wdisp★_fund = fundamental_harmonic(Wdisp★)
	# Fundamental harmonic vanishes at z=0, so u_static = 0 for both models.
	a_ex, i_ex = amplitude_curve(Wdisp★_fund, r_range, (), zeros(n_free★))

	# ── Parametric side ───────────────────────────────────────────────
	ω₀_p = ω₀_param(θ₁_star, θ₂_star)
	Ω_p = [Ω_param(r, θ₁_star, θ₂_star) for r in r_range]
	a_p, i_p = amplitude_curve(Wdisp_param_fund, r_range,
		(θ₁_star, θ₂_star), zeros(n_free_param))

	# Report where the maximum displacement occurs (expect beam midpoint)
	for (tag, i_dof) in (("exact", i_ex), ("param", i_p))
		node, comp = dof_to_nodecomp[i_dof]
		@printf "  max |u| (%s) at node %d, comp %d, x₀ = %s\n" tag node comp string(grid_ref.nodes[node].x)
	end

	# ── Truncate both curves at the amplitude cap ─────────────────────
	# Prefix truncation (not a mask): keeps the curve contiguous and the
	# amplitude axis sorted even if the degree-5 polynomial misbehaves
	# at large r.
	prefix_below(a, cap) = 1:(something(findfirst(>(cap), a), length(a)+1)-1)
	m_ex = prefix_below(a_ex, A_MAX)
	m_p = prefix_below(a_p, A_MAX)

	# ── Metrics: compare shift at matched physical amplitude ──────────
	a_common = range(0.0, min(maximum(a_ex[m_ex]), maximum(a_p[m_p])), 100)
	Δ = [lerp(a_ex[m_ex], (Ω_ex .- ω₀_ex)[m_ex], a) -
		 lerp(a_p[m_p], (Ω_p .- ω₀_p)[m_p], a) for a in a_common]
	Δ = filter(!isnan, Δ)
	rms_shift = sqrt(mean(Δ .^ 2))
	max_shift = maximum(abs.(Δ))
	Δω₀_rel = abs(ω₀_ex - ω₀_p) / abs(ω₀_ex)
	@printf "  ω₀ exact = %.6f  param = %.6f  Δω₀_rel = %.2e\n" ω₀_ex ω₀_p Δω₀_rel
	@printf "  Shift error at matched amplitude: RMS = %.3e  max = %.3e (rad/ms)\n" rms_shift max_shift

	push!(curves, (; θ₁ = θ₁_star, θ₂ = θ₂_star,
		ω₀_ex, ω₀_p, Δω₀_rel, rms_shift, max_shift,
		shift_ex = (Ω_ex .- ω₀_ex)[m_ex], a_ex = a_ex[m_ex],
		shift_p = (Ω_p .- ω₀_p)[m_p], a_p = a_p[m_p]))
end

# ------------------------------------------------------------------
# Metrics CSV
# ------------------------------------------------------------------
open(joinpath(_data, "backbone_physical_metrics.csv"), "w") do io
	println(io, "theta1,theta2,omega0_exact,omega0_param,delta_omega0_rel," *
				"shift_rms_err,shift_max_err")
	for c in curves
		@printf io "%.6f,%.6f,%.8f,%.8f,%.4e,%.4e,%.4e\n" c.θ₁ c.θ₂ c.ω₀_ex c.ω₀_p c.Δω₀_rel c.rms_shift c.max_shift
	end
end
println("Saved → $(joinpath(_data, "backbone_physical_metrics.csv"))")

# ------------------------------------------------------------------
# Backbone curves CSV  (long format: one row per data point)
# ------------------------------------------------------------------
open(joinpath(_data, "backbone_curves.csv"), "w") do io
	println(io, "theta1,theta2,model,amplitude,shift")
	for c in curves
		for (a, s) in zip(c.a_ex, c.shift_ex)
			@printf io "%.6f,%.6f,exact,%.10f,%.10f\n" c.θ₁ c.θ₂ a s
		end
		for (a, s) in zip(c.a_p, c.shift_p)
			@printf io "%.6f,%.6f,param,%.10f,%.10f\n" c.θ₁ c.θ₂ a s
		end
	end
end
println("Saved → $(joinpath(_data, "backbone_curves.csv"))")
