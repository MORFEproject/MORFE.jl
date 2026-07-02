"""
	main.jl — Top-level driver for the Kármán vortex street DPIM demo.

Pipeline
────────
1.  Generate Turek–Schäfer mesh (Gmsh)
2.  Ferrite P2/P1 Taylor-Hood FEM setup
3.  Newton steady-state solve at Re₀
4.  Assemble linearised NSE operators B₀, B₁
5.  Assemble K_visc (parametric coupling) and h₀ (base-flow forcing)
6.  Shift-invert ARPACK eigenproblem → Hopf pair (λ₁, λ₂)
7.  Build NDOrderModel + multiindex set
8.  Solve cohomological equations (DPIM)
9.  Realify reduced dynamics → Stuart-Landau coefficients
10. Export R + lift polynomial → CSV → MATLAB (COCO format)

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

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "fem", "mesh.jl"))
include(joinpath(@__DIR__, "fem", "fem_setup.jl"))
include(joinpath(@__DIR__, "solver", "steady_state.jl"))
include(joinpath(@__DIR__, "fem", "linear_operators.jl"))
include(joinpath(@__DIR__, "fem", "fluid_maps.jl"))
include(joinpath(@__DIR__, "solver", "eigensolver.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# Results directory (deterministic name — same config overwrites previous run)
# ─────────────────────────────────────────────────────────────────────────────

const RESULTS_DIR = joinpath(@__DIR__, "results", @sprintf("Re%.2f_ord%d", Re₀, MAX_ORD))
const DATA_DIR = joinpath(RESULTS_DIR, "data")
const FIGS_DIR = joinpath(RESULTS_DIR, "figures")
mkpath(DATA_DIR)
mkpath(FIGS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Logging: tee all output to stdout and summary.log simultaneously
# ─────────────────────────────────────────────────────────────────────────────

_log = open(joinpath(RESULTS_DIR, "summary.log"), "w")

struct TeeIO <: IO
	a::IO
	b::IO
end
Base.unsafe_write(t::TeeIO, p::Ptr{UInt8}, n::UInt) =
	(unsafe_write(t.a, p, n); unsafe_write(t.b, p, n); n)
Base.flush(t::TeeIO) = (flush(t.a); flush(t.b))

_out = TeeIO(stdout, _log)

const _sep = "=" ^ 60
const _dash = "-" ^ 60

println(_out, _sep)
println(_out, "Kármán Vortex Street DPIM  (Re₀ = $Re₀,  order = $MAX_ORD)")
println(_out, "  results → $RESULTS_DIR")
println(_out, _sep)

# ─────────────────────────────────────────────────────────────────────────────
# 1 — Mesh
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[1/10] Generating Turek–Schäfer mesh ...")
r_mesh = @timed generate_mesh(;
	h_cyl = MESH_H_CYL,
	h_wake = MESH_H_WAKE,
	h_bulk = MESH_H_BULK,
)
meshfile = r_mesh.value

# ─────────────────────────────────────────────────────────────────────────────
# 2 — FEM setup
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[2/10] Ferrite P2/P1 Taylor-Hood FEM setup ...")
r_fem = @timed setup_fem(meshfile)
fom = r_fem.value
println(_out, "  Free DOFs (steady state): $(fom.n_free)")
println(_out, "  Free DOFs (DPIM, inlet free): $(fom.n_free_dpim)")

# ─────────────────────────────────────────────────────────────────────────────
# 3 — Steady-state Newton solve
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[3/10] Newton steady-state at Re₀ = $Re₀ ...")
r_ss = @timed solve_steady_state(fom; Re0 = Re₀)
(_, _, s₀_full) = r_ss.value

# ─────────────────────────────────────────────────────────────────────────────
# 4 — Linear operators B₀, B₁
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[4/10] Assembling linearised NSE operators ...")
r_ops = @timed assemble_linear_operators(s₀_full, fom; Re0 = Re₀)
(B₀, B₁) = r_ops.value
println(_out, "  B₁ nnz = $(nnz(B₁)),  B₀ nnz = $(nnz(B₀))")

# ─────────────────────────────────────────────────────────────────────────────
# 5 — K_visc (parametric coupling) + h₀ (base-flow forcing)
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[5/10] Assembling K_visc and base-flow forcing h₀ ...")
r_kvisc = @timed assemble_K_visc(fom)
(K_visc, K_visc_rect) = r_kvisc.value
K_visc .*= -_CYL_D                            # physical sign: ΔA_lin = -D·η·K, so g₁ = -D·η·K·s
# h₀(η′) = -D·η′·K_raw·u₀ — u₀ is the FULL base flow: the prescribed inlet DOFs carry the
# Poiseuille profile, so the rectangular free×ALL block is required here (free×free would
# silently drop the K_raw[free, inlet]·u₀[inlet] contribution next to the inlet).
h₀_vec = -_CYL_D .* (K_visc_rect * s₀_full)
println(_out, "  K_visc nnz = $(nnz(K_visc))")

# ─────────────────────────────────────────────────────────────────────────────
# 6 — Hopf eigenpair
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
# 7 — NDOrderModel + multiindex set
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
# 8 — Resonance set + cohomological equations
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[8/10] Solving cohomological equations (order $MAX_ORD) ...")
lambda_im = ComplexF64[complex(0.0, imag(λ)) for λ in master_eigenvalues]
resonance_set = resonance_set_from_complex_normal_form_style(
	mset, Vector{ComplexF64}(lambda_im), 0.05 * abs(imag(master_eigenvalues[1]));
	external_eigenvalues=ComplexF64[0.0 + 0.0im])

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
# 9 — Realify + write results
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[9/10] Realifying reduced dynamics ...")
Rr = ReducedDynamics(realify(R.poly, conj_map), R.external_system_size)

rdyn_path = joinpath(DATA_DIR, "reduced_dynamics.txt")
open(rdyn_path, "w") do io
	println(io, "Kármán Vortex Street — Reduced Dynamics (real form)")
	@printf(io, "Re₀ = %.4f,  DPIM order = %d,  NVAR = %d\n", Re₀, MAX_ORD, NVAR)
	println(io, "")
	println(io, "Hopf eigenvalues:")
	for (i, λ) in enumerate(master_eigenvalues)
		@printf(io, "  λ[%d] = %+.10f %+.10f i\n", i, real(λ), imag(λ))
	end
	println(io, "")
	println(io, "Nonzero reduced-dynamics monomials:")
	for m in eachindex(Rr.poly.multiindex_set.exponents)
		mi = Rr.poly.multiindex_set.exponents[m]
		c = Rr.poly.coefficients[:, m]
		any(abs.(real.(c)) .> 1e-14) || continue
		@printf(io, "  %-20s : %s\n", string(mi), string(round.(real.(c); sigdigits = 8)))
	end
end

println(_out, "\nReduced dynamics (real form) — nonzero monomials:")
for m in eachindex(Rr.poly.multiindex_set.exponents)
	mi = Rr.poly.multiindex_set.exponents[m]
	c = Rr.poly.coefficients[1, m]
	any(abs.(real.(c)) .> 1e-12) || continue
	@printf(_out, "  %-20s : %s\n", string(mi), string(round.(c; sigdigits = 6)))
end

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

to_gb(b) = round(b / 1024^3; digits = 2)

println(_out)
println(_out, _sep)
println(_out, "Kármán Vortex Street DPIM — Summary")
println(_out, "  Re₀ = $Re₀,  order = $MAX_ORD,  NVAR = $NVAR,  FOM = $(fom.n_free)")
@printf(_out, "  Hopf eigenvalue:  λ₁ = %+.6f %+.6f·i\n",
	real(master_eigenvalues[1]), imag(master_eigenvalues[1]))
println(_out, _dash)
@printf(_out, "  %-36s  %9.3f s  %8.2f GB\n",
	"[1] Mesh generation", r_mesh.time, to_gb(r_mesh.bytes))
@printf(_out, "  %-36s  %9.3f s  %8.2f GB\n",
	"[2] FEM setup", r_fem.time, to_gb(r_fem.bytes))
@printf(_out, "  %-36s  %9.3f s  %8.2f GB\n",
	"[3] Newton steady-state", r_ss.time, to_gb(r_ss.bytes))
@printf(_out, "  %-36s  %9.3f s  %8.2f GB\n",
	"[4] Linear operators", r_ops.time, to_gb(r_ops.bytes))
@printf(_out, "  %-36s  %9.3f s  %8.2f GB\n",
	"[5] K_visc + h₀", r_kvisc.time, to_gb(r_kvisc.bytes))
@printf(_out, "  %-36s  %9.3f s  %8.2f GB\n",
	"[6] Eigenproblem", r_eig.time, to_gb(r_eig.bytes))
@printf(_out, "  %-36s  %9.3f s  %8.2f GB\n",
	"[8] Cohomological solve", r_dpim.time, to_gb(r_dpim.bytes))
println(_out, _sep)

# ─────────────────────────────────────────────────────────────────────────────
# Save ROM
# ─────────────────────────────────────────────────────────────────────────────

serialize(joinpath(DATA_DIR, "W.jls"), W)
serialize(joinpath(DATA_DIR, "R.jls"), R)

# ── Pressure lift polynomial L(z) ─────────────────────────────────────────────
L0_lift, L_coeffs_lift = let
	l = compute_pressure_lift_weights(fom)
	l_free = l[fom.free_dpim]

	C = MORFE.ParametrisationMethod.coefficients(W)   # (FOM, 1, L)
	W1_coeffs = @view(C[:, 1, :])                             # (FOM, L)
	mset_l = MORFE.ParametrisationMethod.multiindex_set(W)

	L_coeffs_l = vec(transpose(W1_coeffs) * l_free)           # (L,) ComplexF64 — bilinear lᵀW (adjoint would conjugate)
	L0_l = dot(l_free, real.(s₀_full[fom.free_dpim]))   # scalar: base-flow lift

	lift_rom = (; L0 = L0_l, L_coeffs = L_coeffs_l, mset = mset_l)
	serialize(joinpath(DATA_DIR, "lift_polynomial.jls"), lift_rom)
	@printf(_out, "  Lift polynomial: L0 = %.6f, %d coefficients\n", L0_l, length(L_coeffs_l))
	(L0_l, L_coeffs_l)
end

# ── VTK data bundle (plain arrays, no Ferrite types) for visualise_paraview.jl ─
let _nn = Ferrite.getnnodes(fom.grid)
	vtk_data = (;
		node_coords = Float64[fom.grid.nodes[i].x[c] for c in 1:2, i in 1:_nn],
		cell_connectivity = [collect(Int32, c.nodes) for c in fom.grid.cells],
		cell_dofs = [collect(Ferrite.celldofs(cell)) for cell in Ferrite.CellIterator(fom.dh)],
		free_dpim = fom.free_dpim,
		ndofs_total = Ferrite.ndofs(fom.dh),
		n_vel_dofs_per_cell = fom.n_vel_dofs_per_cell,
		s0_free_dpim = s₀_full[fom.free_dpim],
		master_eigenvalues = master_eigenvalues,
		all_eigenvalues = all_eigenvalues,
		all_modes = Matrix{ComplexF64}(all_modes),
	)
	serialize(joinpath(DATA_DIR, "vtk_data.jls"), vtk_data)
end

# ─────────────────────────────────────────────────────────────────────────────
# 10 — Export R → CSV → MATLAB (COCO format)
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[10/10] Exporting R coefficients and generating MATLAB files ...")

r_export = @timed let
	csv_path = joinpath(DATA_DIR, "R_coefficients.csv")
	exps = R.poly.multiindex_set.exponents
	coeffs = R.poly.coefficients   # (NVAR, L) ComplexF64
	NVAR_R = size(coeffs, 1)
	n_rows = 0
	open(csv_path, "w") do io
		header = join(["exp_$i" for i in 1:length(exps[1])], ",") * "," *
				 join(["R$(i)_re,R$(i)_im" for i in 1:NVAR_R], ",")
		println(io, header)
		for (m, ex) in enumerate(exps)
			c = coeffs[:, m]
			any(abs.(c) .> 1e-14) || continue
			row = join(string.(Int.(ex)), ",") * "," *
				  join(["$(real(c[i])),$(imag(c[i]))" for i in 1:NVAR_R], ",")
			println(io, row)
			n_rows += 1
		end
	end
	println(_out, "  R_coefficients.csv  ($n_rows rows)")

	# L_coefficients.csv — lift polynomial (constant row + polynomial terms)
	lift_csv_path = joinpath(DATA_DIR, "L_coefficients.csv")
	L_exps = mset.exponents
	n_L_rows = 0
	open(lift_csv_path, "w") do io
		header = join(["exp_$i" for i in 1:length(L_exps[1])], ",") * ",L_re,L_im"
		println(io, header)
		println(io, join(zeros(Int, length(L_exps[1])), ",") * ",$(L0_lift),0.0")   # constant (base-flow lift)
		for (m, ex) in enumerate(L_exps)
			c = L_coeffs_lift[m]
			abs(c) > 1e-14 || continue
			println(io, join(string.(Int.(ex)), ",") * ",$(real(c)),$(imag(c))")
			n_L_rows += 1
		end
	end
	println(_out, "  L_coefficients.csv  ($n_L_rows polynomial rows, L0 = $(round(L0_lift; sigdigits=6)))")

	py_script = joinpath(@__DIR__, "validation", "generate_matlab.py")
	py3 = Sys.which("python3")
	if py3 !== nothing && isfile(py_script)
		run(Cmd([py3, py_script, csv_path,
			"--output-dir", DATA_DIR,
			"--re0", string(Re₀),
			"--max-ord", string(MAX_ORD)]))
		println(_out, "  vec_fields_karman.m + vec_fields_karman_DFDX.m  written to data/")
	else
		println(_out, "  Warning: python3 not found — run validation/generate_matlab.py manually")
	end
end

println(_out, "\nResults saved to: $RESULTS_DIR")
println(_out, "  summary.log, summary.txt")
println(_out, "  data/: reduced_dynamics.txt, W.jls, R.jls, vtk_data.jls")
println(_out, "         R_coefficients.csv, vec_fields_karman.m, vec_fields_karman_DFDX.m")
println(_out, "         L_coefficients.csv, lift_karman.m")

close(_log)

open(joinpath(RESULTS_DIR, "summary.txt"), "w") do io
	println(io, "example: 05_karman_vortex_street")
	@printf(io, "run_name: Re%.2f_ord%d\n", Re₀, MAX_ORD)
	println(io, "model: 2D Navier-Stokes, Kármán vortex street, Ferrite P2/P1 Taylor-Hood")
	println(io, "n_free: $(fom.n_free)")
	println(io, "Re0: $Re₀")
	println(io, "master_modes: 2  (Hopf pair)")
	println(io, "master_eigenvalues: $(collect(master_eigenvalues))")
	println(io, "parametrisation_order: $MAX_ORD")
	@printf(io, "cohomological_solve_time_s: %.3f\n", r_dpim.time)
	println(io, "julia_version: $(VERSION)")
	commit = try
		readchomp(`git rev-parse --short HEAD`)
	catch
		;
		"unknown"
	end
	println(io, "morfe_commit: $commit")
	println(io, "timestamp: $(time())")
end
