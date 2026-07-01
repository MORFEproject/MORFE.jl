"""
	follow_limit_cycle.jl — FOM Hopf limit-cycle branch via FOM pseudo-arclength continuation.

Algorithm
─────────
Starting from the Hopf bifurcation point (s₀ = 0, T = T_hopf, η′ = 0), the limit-cycle
branch is followed by pseudo-arclength continuation entirely in FOM state space
(s₀, T, η′). The ROM is used only once, to extract the Hopf eigenvector (φ₁_im) and
eigenfrequency (T_hopf) for the initial tangent — all subsequent continuation is driven
by FOM Newton-Krylov shooting with η′ as a free variable (`find_periodic_orbit_newton_palc`).

Per PALC step:
  1. predictor : (s₀, T, η′) ← last + Δs × unit tangent (scaled metric)
  2. corrector : Newton-GMRES bordered system [F; g; N] = 0 (shooting, phase, arclength)
  3. new tangent: secant of consecutive converged branch points
  4. Δs adaptation: too few outer Newton iters → double; too many → halve and retry

Output: DATA_DIR/limit_cycle_branch.csv
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
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
using KLU

const EXAMPLE_DIR = realpath(joinpath(@__DIR__, ".."))

include(joinpath(EXAMPLE_DIR, "config.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "mesh.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "fem_setup.jl"))
include(joinpath(EXAMPLE_DIR, "solver", "steady_state.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "linear_operators.jl"))
include(joinpath(EXAMPLE_DIR, "fem", "fluid_maps.jl"))
include(joinpath(EXAMPLE_DIR, "solver", "time_integration.jl"))
include(joinpath(EXAMPLE_DIR, "solver", "shooting.jl"))
include(joinpath(EXAMPLE_DIR, "solver", "rom_palc.jl"))   # only rom_po_frequency is used

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

const RE_END = 56.0          # target Reynolds number
const Δs_INIT = 1e-6        # initial arclength step (scaled FOM state space)
const Δs_MIN = 1e-20
const Δs_MAX = 0.005
const N_NEWTON_FAST = 3            # fewer outer Newton iters → double Δs
const N_NEWTON_SLOW = 5            # unused directly; MAX_NEWTON_FOM governs halving
const Δt_INTEG = 1e-2
const θ_INTEG = 0.5
const MAX_NEWTON_FOM = 15
const TOL_FOM = 1e-3         # ‖F‖/‖s‖ convergence tolerance
const KRYLOV_DIM = 20
const MAX_STEPS = 5000

const DATA_DIR = joinpath(EXAMPLE_DIR, "results",
	@sprintf("Re%.2f_ord%d", Re₀, MAX_ORD), "data")
isdir(DATA_DIR) || error("DATA_DIR not found: $DATA_DIR — run main.jl first.")

const η_end = 1.0 / RE_END - 1.0 / Re₀    # most negative η′ on the branch

@printf("follow_limit_cycle.jl\n")

# ─────────────────────────────────────────────────────────────────────────────
# Load ROM
# ─────────────────────────────────────────────────────────────────────────────

R_path = joinpath(DATA_DIR, "R.jls")
W_path = joinpath(DATA_DIR, "W.jls")
isfile(R_path) || error("R.jls not found in $DATA_DIR — run main.jl first.")
isfile(W_path) || error("W.jls not found in $DATA_DIR — run main.jl first.")

@info "Loading ROM ..."
R = deserialize(R_path)
W = deserialize(W_path)
@info "  ROM loaded"

# ─────────────────────────────────────────────────────────────────────────────
# FOM setup
# ─────────────────────────────────────────────────────────────────────────────

meshfile = joinpath(EXAMPLE_DIR, "fem", "cylinder_flow.msh")
isfile(meshfile) || error("Mesh not found — run main.jl first.")
@info "Setting up FEM ..."
fom = Base.invokelatest(setup_fem, meshfile)
@info "  n_free_dpim = $(fom.n_free_dpim)"

linear_ops_file = joinpath(DATA_DIR, "linear_ops.jls")
if isfile(linear_ops_file)
	@info "Loading cached operators ..."
	ops = deserialize(linear_ops_file)
	B₀ = ops.B₀;
	B₁ = ops.B₁
	K_visc = ops.K_visc;
	h₀_vec = ops.h₀_vec
else
	@info "Computing operators — this takes a few minutes ..."
	(_, _, s₀_full) = Base.invokelatest(solve_steady_state, fom; Re0 = Re₀)
	B₀, B₁ = Base.invokelatest(assemble_linear_operators, s₀_full, fom; Re0 = Re₀)
	K_visc = Base.invokelatest(assemble_K_visc, fom)
	K_visc .*= -_CYL_D
	h₀_vec = K_visc * s₀_full[fom.free_dpim]
	serialize(linear_ops_file, (; B₀, B₁, K_visc, h₀_vec))
end

@info "Computing lift weights and eigenvector directions ..."
l_free = Base.invokelatest(compute_pressure_lift_weights, fom)[fom.free_dpim]

const Re_c = 49.03               # Hopf bifurcation Re (from linear stability)
const η_c = 1.0 / Re_c - 1.0 / Re₀

@printf("  FOM PALC from Hopf bifurcation (Re_c = %.2f) to Re = %.2f\n\n", Re_c, RE_END)
@printf("  η′_c = %+.6e\n", η_c)

# Extract φ₁_im (Hopf eigenmode) and φ₁_re (phase condition) from W's linear
# coefficient at η=0.  Do NOT use η=η_c here: evaluate(W,[ε,0,η_c])/ε adds
# c₀₀₁·η_c/ε which is O(100) times larger than the eigenmode for η_c/ε≈100.
let ε_ev = 1e-6
	# Use η=0: evaluate at η_c would add c₀₀₁·η_c/ε_ev (≈100× the eigenmode)
	ψ₁ = evaluate(W.poly, ComplexF64[ε_ev, 0.0, 0.0])[:, 1] ./ ε_ev
	global φ₁_im = imag.(ψ₁)
	global φ₁_re = real.(ψ₁)
	φ₁_re_nrm = norm(φ₁_re)
	φ₁_re_nrm > 0 && (φ₁_re ./= φ₁_re_nrm)
	@printf("  ‖φ₁_im‖ = %.4e   ‖φ₁_re‖(normalised) = %.4f\n", norm(φ₁_im), norm(φ₁_re))
end

# ─────────────────────────────────────────────────────────────────────────────
# Initialise at the Hopf bifurcation (s₀ = 0, η′ = η_c, T = T_hopf)
# ─────────────────────────────────────────────────────────────────────────────

T_hopf = 2π / abs(rom_po_frequency(1e-6, 0.0, R))
@printf("  T_hopf = %.6f s\n\n", T_hopf)

const s_scale = norm(φ₁_im)
const T_scale = T_hopf
const η_scale = abs(η_end - η_c)

# Scaled initial tangent: pure amplitude growth, orthogonal to η′
τ̃_s = φ₁_im ./ s_scale     # unit L2 vector
τ̃_T = 0.0
τ̃_η = 0.0

τ_s_eff = τ̃_s ./ s_scale
τ_T_eff = τ̃_T / T_scale
τ_η_eff = τ̃_η / η_scale

s_cur = zeros(fom.n_free_dpim)
T_cur = T_hopf
η_cur = η_c
Δs = Δs_INIT

# ─────────────────────────────────────────────────────────────────────────────
# Continuation loop
# ─────────────────────────────────────────────────────────────────────────────

Re_vec = Float64[];
η_vec = Float64[];
s0nrm_vec = Float64[]
FL_vec = Float64[];
T_vec = Float64[];
Δs_vec = Float64[]
nintg_v = Int[];
conv_v = Bool[]

sep = "─" ^ 104
@printf("\n%s\n", sep)
@printf("  %5s  %8s  %9s  %10s  %10s  %10s  %9s  %7s  %5s\n",
	"step", "Re", "η′", "‖s₀‖", "max|FL|", "T_fom (s)", "|Δs|", "n_intg", "conv")
@printf("%s\n", sep)

n_step = 0
while n_step < MAX_STEPS
	global s_cur, T_cur, η_cur, τ̃_s, τ̃_T, τ̃_η, τ_s_eff, τ_T_eff, τ_η_eff, Δs, n_step

	Re_cur = 1.0 / (1.0 / Re₀ + η_cur)
	Re_cur >= RE_END && break

	# ── Predictor ──────────────────────────────────────────────────────────
	s_pred = s_cur .+ (Δs * s_scale) .* τ̃_s
	T_pred = T_cur + Δs * T_scale * τ̃_T
	η_pred = η_cur + Δs * η_scale * τ̃_η

	# ── FOM PALC corrector ────────────────────────────────────────────────
	# Near the bifurcation the orbit is x(t)=ε·φ₁·cos(ωt): the predictor
	# is already the eigenmode to O(ε²) — skip Newton entirely.
	# The 2×2 bordered system is ill-conditioned at O(1/ρ²) when τ̃_η≈0;
	# delay Newton until ρ≈0.01 so the condition number is manageable.
	if norm(s_pred) < s_scale * 1e-2
		s_new, T_new, η_new = s_pred, T_pred, η_pred
		n_nwt, n_intg, fom_conv = 1, 0, true
	else
		s_new, T_new, η_new, n_nwt, n_intg, fom_conv = Base.invokelatest(
			find_periodic_orbit_newton_palc,
			s_pred, T_pred, η_pred,
			s_cur, T_cur, η_cur,
			τ_s_eff, τ_T_eff, τ_η_eff, Δs,
			fom, B₀, B₁, K_visc, h₀_vec, φ₁_re;
			Δt = Δt_INTEG, θ = θ_INTEG,
			max_newton = MAX_NEWTON_FOM, tol = TOL_FOM,
			krylov_dim = KRYLOV_DIM,
			lift_weights = l_free, verbose = true,
			max_dη = 0.5 * abs(η_end - η_c),
		)
	end

	if !fom_conv || n_nwt > MAX_NEWTON_FOM
		Δs = max(Δs * 0.5, Δs_MIN)
		if Δs <= Δs_MIN
			@warn "Δs hit minimum at η′=$η_cur — stopping."
			break
		end
		continue   # retry from the same branch point with a smaller step
	end

	Re_new = 1.0 / (1.0 / Re₀ + η_new)

	# ── Lift amplitude over the converged orbit ───────────────────────────
	n_steps_po = max(1, round(Int, T_new / Δt_INTEG))
	Δt_ex = T_new / n_steps_po
	inv_dt = 1.0 / Δt_ex
	A_imp = B₀ .- η_new .* K_visc
	LHS_po = inv_dt .* B₁ .+ θ_INTEG .* A_imp
	RHS_po = inv_dt .* B₁ .- (1.0 - θ_INTEG) .* A_imp
	L_po = klu(LHS_po)
	_, FL_hist = Base.invokelatest(
		_integrate_one_orbit,
		s_new, η_new, T_new, fom, L_po, RHS_po, h₀_vec, Δt_ex, l_free)
	FL_max = FL_hist !== nothing ? maximum(abs, FL_hist) : NaN

	@printf("  %5d  %8.4f  %+9.5f  %10.4e  %10.4e  %10.5f  %9.2e  %7d  %5s\n",
		n_step + 1, Re_new, η_new, norm(s_new), FL_max, T_new, Δs, n_nwt,
		fom_conv ? "✓" : "✗")

	push!(Re_vec, Re_new);
	push!(η_vec, η_new);
	push!(s0nrm_vec, norm(s_new))
	push!(FL_vec, FL_max);
	push!(T_vec, T_new);
	push!(Δs_vec, Δs)
	push!(nintg_v, n_intg);
	push!(conv_v, fom_conv)

	# ── New tangent from secant (scaled metric), sign-corrected ───────────
	Δỹ_s = (s_new .- s_cur) ./ s_scale
	Δỹ_T = (T_new - T_cur) / T_scale
	Δỹ_η = (η_new - η_cur) / η_scale
	Δnorm = sqrt(dot(Δỹ_s, Δỹ_s) + Δỹ_T^2 + Δỹ_η^2)

	if Δnorm > 1e-300
		τ̃_s_new = Δỹ_s ./ Δnorm
		τ̃_T_new = Δỹ_T / Δnorm
		τ̃_η_new = Δỹ_η / Δnorm
		if dot(τ̃_s_new, τ̃_s) + τ̃_T_new * τ̃_T + τ̃_η_new * τ̃_η < 0.0
			τ̃_s_new .*= -1.0
			τ̃_T_new *= -1.0
			τ̃_η_new *= -1.0
		end
		τ̃_s = τ̃_s_new
		τ̃_T = τ̃_T_new
		τ̃_η = τ̃_η_new
		τ_s_eff = τ̃_s ./ s_scale
		τ_T_eff = τ̃_T / T_scale
		τ_η_eff = τ̃_η / η_scale
	end

	s_cur = s_new;
	T_cur = T_new;
	η_cur = η_new
	n_step += 1

	# ── Adapt Δs from outer Newton iteration count ─────────────────────────
	n_nwt < N_NEWTON_FAST && (Δs = min(Δs * 2.0, Δs_MAX))
end

@printf("%s\n\n", sep)
n_step >= MAX_STEPS && @warn "Reached MAX_STEPS=$MAX_STEPS."

# ─────────────────────────────────────────────────────────────────────────────
# Output CSV
# ─────────────────────────────────────────────────────────────────────────────

out_path = joinpath(DATA_DIR, "limit_cycle_branch.csv")
open(out_path, "w") do io
	println(io, "Re,eta,s0_norm,max_abs_FL,T_fom,d_s,n_integrations,converged")
	for i in eachindex(Re_vec)
		@printf(io, "%.10f,%.15e,%.15e,%.15e,%.15e,%.15e,%d,%s\n",
			Re_vec[i], η_vec[i], s0nrm_vec[i], FL_vec[i], T_vec[i],
			Δs_vec[i], nintg_v[i], conv_v[i])
	end
end
@info "Results written to $out_path  ($(length(Re_vec)) branch points)"
