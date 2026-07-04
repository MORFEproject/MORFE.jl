"""
	main.jl — Kármán vortex street DPIM run (STEP 1 of 3).

What this computes
──────────────────
The 2D incompressible flow past a cylinder loses stability at Re_c ≈ 49 through a Hopf
bifurcation — the Kármán vortex street. This script reduces the ~58 000-DOF
finite-element model to a SINGLE complex ODE (a Stuart–Landau equation)

	ż₁ = λ z₁ + c₁₀₁ z₁ η′ + c₂₁₀ z₁|z₁|² + …

by the Direct Parametrisation of Invariant Manifolds (DPIM). The Reynolds number is
carried along as an extra parametric coordinate η′ = 1/Re − 1/Re₀, so ONE run at the
expansion point Re₀ describes the whole bifurcation neighbourhood.

How to run
──────────
	julia --project=. main.jl          # ≈ 7 min (Apple Silicon): 35 s setup + order-9 solve

Outputs land in results/Re{Re₀}_ord{MAX_ORD}/ (ROM, lift polynomial, TKE Gram, CSVs;
see README.md). One run at MAX_ORD suffices for the whole order-convergence study —
the cohomological solve is graded, so lower orders are exact truncations.

Then:   julia --project=. solve_rom.jl     (limit-cycle branch per truncation order)
		python3 compare_orders.py          (lift & avg-TKE vs Re, one curve per order)

All parameters in config.jl.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
	Pkg.add([
		"Ferrite", "FerriteGmsh", "Gmsh",
		"Arpack", "LinearMaps",
		"StaticArrays", "KLU",
	])
end
Pkg.instantiate()

using MORFE
using Ferrite
using FerriteGmsh
using Gmsh
using Arpack
using LinearMaps
using StaticArrays
using LinearAlgebra
using SparseArrays
using Printf
using Serialization
using DelimitedFiles

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "fem", "mesh.jl"))
include(joinpath(@__DIR__, "fem", "fem_setup.jl"))
include(joinpath(@__DIR__, "solver", "steady_state.jl"))
include(joinpath(@__DIR__, "fem", "linear_operators.jl"))
include(joinpath(@__DIR__, "fem", "fluid_maps.jl"))
include(joinpath(@__DIR__, "fem", "energy_gram.jl"))
include(joinpath(@__DIR__, "solver", "eigensolver.jl"))
include(joinpath(@__DIR__, "exports.jl"))

# Results directory (deterministic name — the same config overwrites the previous run)
const RESULTS_DIR = joinpath(@__DIR__, "results", @sprintf("Re%.2f_ord%d", Re₀, MAX_ORD))
const DATA_DIR = joinpath(RESULTS_DIR, "data")
mkpath(DATA_DIR)
mkpath(joinpath(RESULTS_DIR, "figures"))

_log = open(joinpath(RESULTS_DIR, "summary.log"), "w")
_out = TeeIO(stdout, _log)   # everything printed to _out also lands in summary.log

println(_out, _sep)
println(_out, "Kármán Vortex Street DPIM  (Re₀ = $Re₀,  order = $MAX_ORD)")
println(_out, "  results → $RESULTS_DIR")
println(_out, _sep)

# ─────────────────────────────────────────────────────────────────────────────
# 1 — Mesh: Turek–Schäfer channel (2.2 × 0.41 m) with a Ø 0.1 m cylinder
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[1/10] Generating Turek–Schäfer mesh ...")
r_mesh = @timed generate_mesh(;
	h_cyl = MESH_H_CYL,
	h_wake = MESH_H_WAKE,
	h_bulk = MESH_H_BULK,
)
meshfile = r_mesh.value

# ─────────────────────────────────────────────────────────────────────────────
# 2 — FEM setup: P2/P1 Taylor-Hood velocity/pressure spaces.
#     Dirichlet BCs (Poiseuille inlet, no-slip walls/cylinder) are eliminated;
#     the DPIM operates on the remaining `free_dpim` DOFs, where the perturbation
#     around the base flow vanishes on every prescribed boundary.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[2/10] Ferrite P2/P1 Taylor-Hood FEM setup ...")
r_fem = @timed setup_fem(meshfile)
fom = r_fem.value
println(_out, "  Free DOFs (steady state): $(fom.n_free)")
println(_out, "  Free DOFs (DPIM): $(fom.n_free_dpim)")

# ─────────────────────────────────────────────────────────────────────────────
# 3 — Base flow: steady Navier-Stokes solution u₀ at Re₀ (Newton iteration).
#     The manifold is expanded around this fixed point.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[3/10] Newton steady-state at Re₀ = $Re₀ ...")
r_ss = @timed solve_steady_state(fom; Re0 = Re₀)
(_, _, s₀_full) = r_ss.value

# ─────────────────────────────────────────────────────────────────────────────
# 4 — Linearised operators. The perturbation s = [u′; p′] obeys the descriptor
#     system  B₁ ṡ + B₀ s = F(s, η′):  B₁ is the velocity mass matrix (singular —
#     pressure has no time derivative), B₀ = −A_lin collects viscosity, base-flow
#     convection and the pressure/incompressibility coupling.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[4/10] Assembling linearised NSE operators ...")
r_ops = @timed assemble_linear_operators(s₀_full, fom; Re0 = Re₀)
(B₀, B₁) = r_ops.value
println(_out, "  B₁ nnz = $(nnz(B₁)),  B₀ nnz = $(nnz(B₀))")

# ─────────────────────────────────────────────────────────────────────────────
# 5 — The η′ trick. With ν = D/Re, the parameter η′ = 1/Re − 1/Re₀ enters the
#     equations LINEARLY through the viscous term:
#       g₁(s, η′) = −D·η′·K_raw·u′        (operator change on the perturbation)
#       h₀(η′)    = −D·η′·K_raw·u₀        (base-flow forcing; u₀ is the FULL base
#     flow — its prescribed inlet DOFs carry the Poiseuille profile, so h₀ needs
#     the rectangular free×ALL block of K_raw, not the free×free one.)
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[5/10] Assembling K_visc and base-flow forcing h₀ ...")
r_kvisc = @timed assemble_K_visc(fom)
(K_visc, K_visc_rect) = r_kvisc.value
K_visc .*= -_CYL_D
h₀_vec = -_CYL_D .* (K_visc_rect * s₀_full)
println(_out, "  K_visc nnz = $(nnz(K_visc))")

# ─────────────────────────────────────────────────────────────────────────────
# 6 — Master modes: the Hopf pair. Shift-invert ARPACK on A_lin y = λ B₁ y finds
#     the least-damped complex-conjugate eigenpair λ₁,₂ = σ₀ ± iω₀ (σ₀ ≈ 0 near
#     Re_c, ω₀ ≈ 16.86 rad/s — the shedding frequency). Its right/left
#     eigenvectors seed the two master coordinates z₁, z̄₁ of the manifold.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[6/10] Shift-invert ARPACK eigenproblem ...")
r_eig = @timed solve_hopf_eigenproblem(
	-B₀, B₁;
	nev = EIG_NEV,
	sigma_re = EIG_SIGMA_RE,
	sigma_im = EIG_SIGMA_IM,
	target_freq = EIG_TARGET_FREQ,
)
(master_eigenvalues, master_modes, left_eigenmodes, all_eigenvalues, all_modes) = r_eig.value

# ─────────────────────────────────────────────────────────────────────────────
# 7 — Full-order model for the DPIM: the quadratic convection f₂(s,s) = −(u·∇)u
#     (FEM-assembled on the fly), the two η′ terms from stage 5, and the
#     parameter itself as a trivial external system η̇′ = 0. The multiindex set
#     enumerates every monomial z₁^a z̄₁^b η′^c up to total degree MAX_ORD.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[7/10] Building NDOrderModel and multiindex set ...")
mset = all_multiindices_up_to(NVAR, MAX_ORD; min_degree = 1)
convection = FluidConvection(fom; max_unique_cols = length(mset))
g₁ = make_param_coupling(K_visc)
h₀ = make_base_forcing(h₀_vec)
ext_sys = ExternalSystem((0.0 + 0.0im,))

model = NDOrderModel((B₀, B₁), (convection, g₁, h₀), ext_sys)
println(_out, "  $(length(mset)) monomials (NVAR=$NVAR, order ≤ $MAX_ORD)")

# ─────────────────────────────────────────────────────────────────────────────
# 8 — DPIM solve. The normal-form-style resonance set keeps only the resonant
#     monomials z₁|z₁|^{2k} η′^j in the reduced dynamics; everything else is
#     absorbed into the manifold map W. Solving the cohomological equations
#     order-by-order yields
#       W : (z₁, z̄₁, η′) ↦ full state      (the invariant-manifold embedding)
#       R : ż = R(z)                        (the reduced dynamics on it)
#     The solve is graded — degree-N coefficients never depend on degrees > N —
#     which is why lower orders are exact truncations of this run.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[8/10] Solving cohomological equations (order $MAX_ORD) ...")
lambda_im = ComplexF64[complex(0.0, imag(λ)) for λ in master_eigenvalues]
resonance_set = resonance_set_from_complex_normal_form_style(
	mset, Vector{ComplexF64}(lambda_im), 0.05 * abs(imag(master_eigenvalues[1]));
	external_eigenvalues = ComplexF64[0.0 + 0.0im])

println(_out, "\nResonance set  (NVAR=$NVAR, max_degree=$MAX_ORD)")
for t in 1:NVAR
	cols = resonant_multiindices(resonance_set, t)
	@printf(_out, "     Target %d:  %d monomials\n", t, length(cols))
	isempty(cols) || println(_out, "       ", join(["$(mset.exponents[k])" for k in cols], "  "))
end

conj_map = [2, 1, 3]   # mode 1 (Im>0) ↔ mode 2 (Im<0); η′ self-conjugate
r_dpim = @timed solve_cohomological_problem(
	model, mset,
	master_eigenvalues,
	master_modes .* 1e-2, left_eigenmodes .* 1e-2,   # scale modes for better numerical stability (see discussion in #48)
	resonance_set;
	conjugate_permutation = conj_map,
)
(W, R) = r_dpim.value

# ─────────────────────────────────────────────────────────────────────────────
# 9 — Report the reduced dynamics: the first row ż₁ = R₁(z₁, z̄₁, η′) is the
#     Stuart-Landau equation (row 2 is its conjugate, row 3 is η̇′ = 0).
#     Supercritical Hopf ⇔ Re(c₂₁₀) < 0 for the [2,1,0] monomial.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[9/10] Reduced dynamics — first row ż₁ = R₁(z₁, z̄₁, η′) ...")
export_reduced_dynamics(_out, DATA_DIR, R, master_eigenvalues;
	re0 = Re₀, ord = MAX_ORD, nvar = NVAR)

# ─────────────────────────────────────────────────────────────────────────────
# 10 — Save the ROM and the two physical observables that compare_orders.py
#      evaluates along the limit cycle:
#        · lift polynomial  L(z) = L0 + Σ (lᵀW_α) z^α   (pressure lift on the cylinder)
#        · TKE Gram matrix  G = W_velᵀ M_vel W_vel / |Ω| (period-averaged kinetic energy)
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[10/10] Saving ROM and exporting observables ...")

export_rom(DATA_DIR, W, R)

# Lift functional: l picks the pressure traction −p·n_y on the cylinder boundary.
l_free = compute_pressure_lift_weights(fom)[fom.free_dpim]
L0_lift = dot(l_free, real.(s₀_full[fom.free_dpim]))   # base-flow lift
L_coeffs_lift = export_lift_polynomial(_out, DATA_DIR, W, l_free, L0_lift)

# TKE Gram: the L×L matrix that lets Python evaluate ⟨TKE⟩ without FOM-sized data.
(M_vel, vel_rows, area) = prepare_energy_gram(fom)
write_energy_gram(DATA_DIR, W, M_vel, vel_rows, area)
println(_out, "  tke_gram_re.csv, tke_gram_im.csv, tke_avector.csv  written to data/")

export_vtk_bundle(DATA_DIR, fom, s₀_full, master_eigenvalues, all_eigenvalues, all_modes)

csv_path = export_coefficient_csvs(_out, DATA_DIR, R, mset, L0_lift, L_coeffs_lift)
EXPORT_MATLAB && export_matlab_model(_out, DATA_DIR, csv_path; re0 = Re₀, ord = MAX_ORD)

write_summary(_out, RESULTS_DIR, fom, master_eigenvalues,
	r_mesh, r_fem, r_ss, r_ops, r_kvisc, r_eig, r_dpim;
	re0 = Re₀, ord = MAX_ORD, nvar = NVAR)

close(_log)
