"""
	main.jl — Kármán vortex street DPIM order sweep (STEP 1 of the pipeline).

Shared stages (order-independent, run ONCE):
1. Generate Turek–Schäfer mesh (Gmsh)
2. Ferrite P2/P1 Taylor-Hood FEM setup
3. Newton steady-state solve at Re₀
4. Assemble linearised NSE operators B₀, B₁
5. Assemble K_visc (parametric coupling) and h₀ (base-flow forcing)
6. Shift-invert ARPACK eigenproblem → Hopf pair (λ₁, λ₂)

Per-order stages (loop over ORDERS from config.jl):
7. Build NDOrderModel + multiindex set
8. Solve cohomological equations (DPIM)
9. Realify reduced dynamics → Stuart-Landau coefficients
10. Export R + lift polynomial + TKE Gram → results/Re{Re₀}_ord{N}/data/

Next steps:  solve_rom.jl (ROM limit-cycle branch)  →  compare_orders.py.
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

# ─────────────────────────────────────────────────────────────────────────────
# Logging: per-order summary.log tee'd with stdout
# ─────────────────────────────────────────────────────────────────────────────

struct TeeIO <: IO
	a::IO
	b::IO
end
Base.unsafe_write(t::TeeIO, p::Ptr{UInt8}, n::UInt) =
	(unsafe_write(t.a, p, n); unsafe_write(t.b, p, n); n)
Base.flush(t::TeeIO) = (flush(t.a); flush(t.b))

const _sep = "=" ^ 60
const _dash = "-" ^ 60

to_gb(b) = round(b / 1024^3; digits = 2)

println(_sep)
println("Kármán Vortex Street DPIM sweep  (Re₀ = $Re₀,  orders = $ORDERS)")
println(_sep)

# ─────────────────────────────────────────────────────────────────────────────
# 1 — Mesh
# ─────────────────────────────────────────────────────────────────────────────

println("\n[1/6] Generating Turek–Schäfer mesh ...")
r_mesh = @timed generate_mesh(;
	h_cyl = MESH_H_CYL,
	h_wake = MESH_H_WAKE,
	h_bulk = MESH_H_BULK,
)
meshfile = r_mesh.value

# ─────────────────────────────────────────────────────────────────────────────
# 2 — FEM setup
# ─────────────────────────────────────────────────────────────────────────────

println("\n[2/6] Ferrite P2/P1 Taylor-Hood FEM setup ...")
r_fem = @timed setup_fem(meshfile)
fom = r_fem.value
println("  Free DOFs (steady state): $(fom.n_free)")
println("  Free DOFs (DPIM): $(fom.n_free_dpim)")

# ─────────────────────────────────────────────────────────────────────────────
# 3 — Steady-state Newton solve
# ─────────────────────────────────────────────────────────────────────────────

println("\n[3/6] Newton steady-state at Re₀ = $Re₀ ...")
r_ss = @timed solve_steady_state(fom; Re0 = Re₀)
(_, _, s₀_full) = r_ss.value

# ─────────────────────────────────────────────────────────────────────────────
# 4 — Linear operators B₀, B₁
# ─────────────────────────────────────────────────────────────────────────────

println("\n[4/6] Assembling linearised NSE operators ...")
r_ops = @timed assemble_linear_operators(s₀_full, fom; Re0 = Re₀)
(B₀, B₁) = r_ops.value
println("  B₁ nnz = $(nnz(B₁)),  B₀ nnz = $(nnz(B₀))")

# ─────────────────────────────────────────────────────────────────────────────
# 5 — K_visc (parametric coupling) + h₀ (base-flow forcing)
# ─────────────────────────────────────────────────────────────────────────────

println("\n[5/6] Assembling K_visc and base-flow forcing h₀ ...")
r_kvisc = @timed assemble_K_visc(fom)
(K_visc, K_visc_rect) = r_kvisc.value
K_visc .*= -_CYL_D                            # physical sign: ΔA_lin = -D·η·K, so g₁ = -D·η·K·s
# h₀(η′) = -D·η′·K_raw·u₀ — u₀ is the FULL base flow: the prescribed inlet DOFs carry the
# Poiseuille profile, so the rectangular free×ALL block is required here (free×free would
# silently drop the K_raw[free, inlet]·u₀[inlet] contribution next to the inlet).
h₀_vec = -_CYL_D .* (K_visc_rect * s₀_full)
println("  K_visc nnz = $(nnz(K_visc))")

# ─────────────────────────────────────────────────────────────────────────────
# 6 — Hopf eigenpair
# ─────────────────────────────────────────────────────────────────────────────

println("\n[6/6] Shift-invert ARPACK eigenproblem ...")
r_eig = @timed solve_hopf_eigenproblem(
	-B₀, B₁;
	nev = EIG_NEV,
	sigma_re = EIG_SIGMA_RE,
	sigma_im = EIG_SIGMA_IM,
	target_freq = EIG_TARGET_FREQ,
)
(master_eigenvalues, master_modes, left_eigenmodes, all_eigenvalues, all_modes) = r_eig.value

# ─────────────────────────────────────────────────────────────────────────────
# Shared postprocessing operators (order-independent)
# ─────────────────────────────────────────────────────────────────────────────

println("\nPreparing shared postprocessing operators ...")
g₁ = make_param_coupling(K_visc)
h₀ = make_base_forcing(h₀_vec)
ext_sys = ExternalSystem((0.0 + 0.0im,))
conj_map = [2, 1, 3]   # mode 1 (Im>0) ↔ mode 2 (Im<0); η′ self-conjugate
l_lift = compute_pressure_lift_weights(fom)
l_free = l_lift[fom.free_dpim]
L0_lift = dot(l_free, real.(s₀_full[fom.free_dpim]))
(M_vel, vel_rows, area) = prepare_energy_gram(fom)
println("  lift weights, energy Gram operators ready  (|Ω| = $(round(area; sigdigits=6)))")

# ─────────────────────────────────────────────────────────────────────────────
# Per-order pipeline (stages 7–10)
# ─────────────────────────────────────────────────────────────────────────────

function run_order(ord, fom, s₀_full, B₀, B₁, g₁, h₀, ext_sys, conj_map,
		master_eigenvalues, master_modes, left_eigenmodes, all_eigenvalues, all_modes,
		l_free, L0_lift, M_vel, vel_rows, area)

	results_dir = joinpath(@__DIR__, "results", @sprintf("Re%.2f_ord%d", Re₀, ord))
	data_dir = joinpath(results_dir, "data")
	figs_dir = joinpath(results_dir, "figures")
	mkpath(data_dir)
	mkpath(figs_dir)

	log = open(joinpath(results_dir, "summary.log"), "w")
	out = TeeIO(stdout, log)
	local mset, r_dpim
	try
		println(out, "\n" * _sep)
		println(out, "Kármán Vortex Street DPIM  (Re₀ = $Re₀,  order = $ord)")
		println(out, "  results → $results_dir")
		println(out, _sep)

		# Shared-stage recap (each summary.log is self-contained)
		println(out, "  Free DOFs (steady state): $(fom.n_free)")
		println(out, "  Free DOFs (DPIM): $(fom.n_free_dpim)")
		for (i, λ) in enumerate(master_eigenvalues)
			@printf(out, "  λ[%d] = %+.10f %+.10f i\n", i, real(λ), imag(λ))
		end

		# ── 7 — NDOrderModel + multiindex set ─────────────────────────────
		println(out, "\n[7/10] Building NDOrderModel and multiindex set ...")
		mset = all_multiindices_up_to(NVAR, ord; min_degree = 1)
		convection = FluidConvection(fom; max_unique_cols = length(mset))
		model = NDOrderModel((B₀, B₁), (convection, g₁, h₀), ext_sys)
		println(out, "  $(length(mset)) monomials (NVAR=$NVAR, order ≤ $ord)")

		# ── 8 — Resonance set + cohomological equations ───────────────────
		println(out, "\n[8/10] Solving cohomological equations (order $ord) ...")
		lambda_im = ComplexF64[complex(0.0, imag(λ)) for λ in master_eigenvalues]
		resonance_set = resonance_set_from_complex_normal_form_style(
			mset, Vector{ComplexF64}(lambda_im), 0.05 * abs(imag(master_eigenvalues[1]));
			external_eigenvalues = ComplexF64[0.0 + 0.0im])

		println(out, "\nResonance set  (NVAR=$NVAR, max_degree=$ord)")
		for t in 1:NVAR
			cols = resonant_multiindices(resonance_set, t)
			@printf(out, "     Target %d:  %d monomials\n", t, length(cols))
			isempty(cols) || println(out, "       ", join(["$(mset.exponents[k])" for k in cols], "  "))
		end

		r_dpim = @timed solve_cohomological_problem(
			model, mset,
			master_eigenvalues,
			master_modes .* 1e-2, left_eigenmodes .* 1e-2,   # scale modes for better numerical stability (see discussion in #48)
			resonance_set;
			conjugate_permutation = conj_map,
		)
		(W, R) = r_dpim.value

		# ── 9 — Realify + write results ───────────────────────────────────
		println(out, "\n[9/10] Realifying reduced dynamics ...")
		Rr = ReducedDynamics(realify(R.poly, conj_map), R.external_system_size)

		rdyn_path = joinpath(data_dir, "reduced_dynamics.txt")
		open(rdyn_path, "w") do io
			println(io, "Kármán Vortex Street — Reduced Dynamics (real form)")
			@printf(io, "Re₀ = %.4f,  DPIM order = %d,  NVAR = %d\n", Re₀, ord, NVAR)
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

		println(out, "\nReduced dynamics (real form) — nonzero monomials:")
		for m in eachindex(Rr.poly.multiindex_set.exponents)
			mi = Rr.poly.multiindex_set.exponents[m]
			c = Rr.poly.coefficients[1, m]
			any(abs.(real.(c)) .> 1e-12) || continue
			@printf(out, "  %-20s : %s\n", string(mi), string(round.(c; sigdigits = 6)))
		end

		# ── Save ROM ───────────────────────────────────────────────────────
		serialize(joinpath(data_dir, "W.jls"), W)
		serialize(joinpath(data_dir, "R.jls"), R)

		# ── Pressure lift polynomial L(z) ──────────────────────────────────
		C_W = MORFE.ParametrisationMethod.coefficients(W)   # (FOM, 1, L)
		W1_coeffs = @view(C_W[:, 1, :])                     # (FOM, L)
		mset_l = MORFE.ParametrisationMethod.multiindex_set(W)
		L_coeffs_lift = vec(transpose(W1_coeffs) * l_free)  # (L,) ComplexF64 — bilinear lᵀW (adjoint would conjugate)
		lift_rom = (; L0 = L0_lift, L_coeffs = L_coeffs_lift, mset = mset_l)
		serialize(joinpath(data_dir, "lift_polynomial.jls"), lift_rom)
		@printf(out, "  Lift polynomial: L0 = %.6f, %d coefficients\n", L0_lift, length(L_coeffs_lift))

		# ── TKE energy Gram (for compare_orders.py / run_tke.py) ──────────
		write_energy_gram(data_dir, W, M_vel, vel_rows, area)
		println(out, "  tke_gram_re.csv, tke_gram_im.csv, tke_avector.csv  written to data/")

		# ── VTK data bundle (plain arrays, no Ferrite types) ──────────────
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
			serialize(joinpath(data_dir, "vtk_data.jls"), vtk_data)
		end

		# ── 10 — Export R + L → CSV (→ MATLAB when EXPORT_MATLAB) ─────────
		println(out, "\n[10/10] Exporting R and L coefficients ...")

		csv_path = joinpath(data_dir, "R_coefficients.csv")
		let
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
			println(out, "  R_coefficients.csv  ($n_rows rows)")
		end

		let
			lift_csv_path = joinpath(data_dir, "L_coefficients.csv")
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
			println(out, "  L_coefficients.csv  ($n_L_rows polynomial rows, L0 = $(round(L0_lift; sigdigits=6)))")
		end

		if EXPORT_MATLAB
			py_script = joinpath(@__DIR__, "validation", "generate_matlab.py")
			py3 = Sys.which("python3")
			if py3 !== nothing && isfile(py_script)
				run(Cmd([py3, py_script, csv_path,
					"--output-dir", data_dir,
					"--re0", string(Re₀),
					"--max-ord", string(ord)]))
				println(out, "  vec_fields_karman.m + vec_fields_karman_DFDX.m  written to data/")
			else
				println(out, "  Warning: python3 not found — run validation/generate_matlab.py manually")
			end
		end

		# ── Per-order summary ──────────────────────────────────────────────
		println(out)
		println(out, _dash)
		@printf(out, "  order %d:  %d monomials,  cohomological solve %.3f s  %.2f GB\n",
			ord, length(mset), r_dpim.time, to_gb(r_dpim.bytes))
		println(out, "  results saved to: $results_dir")
		println(out, _dash)

		open(joinpath(results_dir, "summary.txt"), "w") do io
			println(io, "example: 05_karman_vortex_street")
			@printf(io, "run_name: Re%.2f_ord%d\n", Re₀, ord)
			println(io, "model: 2D Navier-Stokes, Kármán vortex street, Ferrite P2/P1 Taylor-Hood")
			println(io, "n_free: $(fom.n_free)")
			println(io, "Re0: $Re₀")
			println(io, "master_modes: 2  (Hopf pair)")
			println(io, "master_eigenvalues: $(collect(master_eigenvalues))")
			println(io, "parametrisation_order: $ord")
			@printf(io, "cohomological_solve_time_s: %.3f\n", r_dpim.time)
			println(io, "julia_version: $(VERSION)")
			commit = try
				readchomp(`git rev-parse --short HEAD`)
			catch
				"unknown"
			end
			println(io, "morfe_commit: $commit")
			println(io, "timestamp: $(time())")
		end
	finally
		close(log)
	end

	return (; ord, n_monomials = length(mset), dpim_time = r_dpim.time)
end

sweep_stats = NamedTuple[]
for ord in ORDERS
	stats = run_order(ord, fom, s₀_full, B₀, B₁, g₁, h₀, ext_sys, conj_map,
		master_eigenvalues, master_modes, left_eigenmodes, all_eigenvalues, all_modes,
		l_free, L0_lift, M_vel, vel_rows, area)
	push!(sweep_stats, stats)
end

# ─────────────────────────────────────────────────────────────────────────────
# Sweep summary
# ─────────────────────────────────────────────────────────────────────────────

println()
println(_sep)
println("Kármán Vortex Street DPIM sweep — Summary  (Re₀ = $Re₀, FOM = $(fom.n_free))")
println(_dash)
@printf("  %-36s  %9.3f s  %8.2f GB\n", "[1] Mesh generation", r_mesh.time, to_gb(r_mesh.bytes))
@printf("  %-36s  %9.3f s  %8.2f GB\n", "[2] FEM setup", r_fem.time, to_gb(r_fem.bytes))
@printf("  %-36s  %9.3f s  %8.2f GB\n", "[3] Newton steady-state", r_ss.time, to_gb(r_ss.bytes))
@printf("  %-36s  %9.3f s  %8.2f GB\n", "[4] Linear operators", r_ops.time, to_gb(r_ops.bytes))
@printf("  %-36s  %9.3f s  %8.2f GB\n", "[5] K_visc + h₀", r_kvisc.time, to_gb(r_kvisc.bytes))
@printf("  %-36s  %9.3f s  %8.2f GB\n", "[6] Eigenproblem", r_eig.time, to_gb(r_eig.bytes))
println(_dash)
for s in sweep_stats
	@printf("  order %-2d  %4d monomials   cohomological solve  %9.3f s\n",
		s.ord, s.n_monomials, s.dpim_time)
end
println(_sep)
println("Next:  julia --project=. solve_rom.jl   →   python3 compare_orders.py")
