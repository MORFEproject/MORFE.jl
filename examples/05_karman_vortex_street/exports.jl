"""
	exports.jl — logging and output plumbing for the Kármán DPIM demo.

Everything in this file is I/O — log tee, serialisation, CSV export, summary tables.
No physics lives here; main.jl stays a readable narrative of the actual pipeline.
"""

using Printf
using Serialization
using DelimitedFiles

# ─────────────────────────────────────────────────────────────────────────────
# Logging: tee all output to stdout and summary.log simultaneously
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

# ─────────────────────────────────────────────────────────────────────────────
# ROM serialisation
# ─────────────────────────────────────────────────────────────────────────────

"""
	export_rom(data_dir, W, R)

Serialise the parametrisation `W.jls` and reduced dynamics `R.jls` (input for
solve_rom.jl and the validation scripts).
"""
function export_rom(data_dir::AbstractString, W, R)
	serialize(joinpath(data_dir, "W.jls"), W)
	serialize(joinpath(data_dir, "R.jls"), R)
	return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Reduced-dynamics report (complex form, equation 1)
# ─────────────────────────────────────────────────────────────────────────────

"""
	export_reduced_dynamics(out, data_dir, R, master_eigenvalues; re0, ord, nvar)

Write `reduced_dynamics.txt` with the first row ż₁ = R₁(z₁, z̄₁, η′) in complex form
(equation 2 is the conjugate, equation 3 the parameter η̇′ = 0) and echo the nonzero
monomials to `out`.
"""
function export_reduced_dynamics(out::IO, data_dir::AbstractString, R,
		master_eigenvalues; re0, ord, nvar)
	exps = R.poly.multiindex_set.exponents

	open(joinpath(data_dir, "reduced_dynamics.txt"), "w") do io
		println(io, "Kármán Vortex Street — Reduced Dynamics (complex form, equation 1)")
		@printf(io, "Re₀ = %.4f,  DPIM order = %d,  NVAR = %d\n", re0, ord, nvar)
		println(io, "")
		println(io, "Hopf eigenvalues:")
		for (i, λ) in enumerate(master_eigenvalues)
			@printf(io, "  λ[%d] = %+.10f %+.10f i\n", i, real(λ), imag(λ))
		end
		println(io, "")
		println(io, "ż₁ = R₁(z₁, z̄₁, η′) — nonzero monomials  [z₁-pow, z̄₁-pow, η′-pow]:")
		for m in eachindex(exps)
			c = R.poly.coefficients[1, m]
			abs(c) > 1e-14 || continue
			@printf(io, "  %-14s : %+.10e %+.10e·i\n", string(exps[m]), real(c), imag(c))
		end
		println(io, "")
		println(io, "Equation 2 is the complex conjugate; equation 3 is the parameter, η̇′ = 0.")
	end

	println(out, "\nReduced dynamics ż₁ = R₁ — nonzero monomials:")
	for m in eachindex(exps)
		c = R.poly.coefficients[1, m]
		abs(c) > 1e-12 || continue
		@printf(out, "  %-14s : %s\n", string(exps[m]), string(round(c; sigdigits = 6)))
	end
	return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Lift polynomial
# ─────────────────────────────────────────────────────────────────────────────

"""
	export_lift_polynomial(out, data_dir, W, l_free, L0) -> L_coeffs

Project the lift weight vector onto the parametrisation, L_α = lᵀW_α (bilinear —
the adjoint would conjugate), serialise `lift_polynomial.jls`, and return the
coefficient vector for the CSV export.
"""
function export_lift_polynomial(out::IO, data_dir::AbstractString, W, l_free, L0)
	C = MORFE.ParametrisationMethod.coefficients(W)   # (FOM, 1, L)
	W1_coeffs = @view(C[:, 1, :])                     # (FOM, L)
	mset_l = MORFE.ParametrisationMethod.multiindex_set(W)
	L_coeffs = vec(transpose(W1_coeffs) * l_free)     # (L,) ComplexF64
	serialize(joinpath(data_dir, "lift_polynomial.jls"),
		(; L0 = L0, L_coeffs = L_coeffs, mset = mset_l))
	@printf(out, "  Lift polynomial: L0 = %.6f, %d coefficients\n", L0, length(L_coeffs))
	return L_coeffs
end

# ─────────────────────────────────────────────────────────────────────────────
# Coefficient CSVs (consumed by compare_orders.py and generate_matlab.py)
# ─────────────────────────────────────────────────────────────────────────────

"""
	export_coefficient_csvs(out, data_dir, R, mset, L0, L_coeffs) -> csv_path

Write `R_coefficients.csv` (all rows of the complex reduced dynamics, one monomial per
row) and `L_coefficients.csv` (constant base-flow row + lift polynomial). Returns the
R csv path (input for the optional MATLAB export).
"""
function export_coefficient_csvs(out::IO, data_dir::AbstractString, R, mset, L0, L_coeffs)
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
			println(io, join(zeros(Int, length(L_exps[1])), ",") * ",$(L0),0.0")   # constant (base-flow lift)
			for (m, ex) in enumerate(L_exps)
				c = L_coeffs[m]
				abs(c) > 1e-14 || continue
				println(io, join(string.(Int.(ex)), ",") * ",$(real(c)),$(imag(c))")
				n_L_rows += 1
			end
		end
		println(out, "  L_coefficients.csv  ($n_L_rows polynomial rows, L0 = $(round(L0; sigdigits=6)))")
	end

	return csv_path
end

"""
	export_matlab_model(out, data_dir, csv_path; re0, ord)

Optional matcont/COCO export: run validation/generate_matlab.py on the R CSV.
Called only when `EXPORT_MATLAB` is set in config.jl.
"""
function export_matlab_model(out::IO, data_dir::AbstractString, csv_path; re0, ord)
	py_script = joinpath(@__DIR__, "validation", "generate_matlab.py")
	py3 = Sys.which("python3")
	if py3 !== nothing && isfile(py_script)
		run(Cmd([py3, py_script, csv_path,
			"--output-dir", data_dir,
			"--re0", string(re0),
			"--max-ord", string(ord)]))
		println(out, "  vec_fields_karman.m + vec_fields_karman_DFDX.m  written to data/")
	else
		println(out, "  Warning: python3 not found — run validation/generate_matlab.py manually")
	end
	return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# VTK bundle + run summary
# ─────────────────────────────────────────────────────────────────────────────

"""
	export_vtk_bundle(data_dir, fom, s0_full, master_eigenvalues, all_eigenvalues, all_modes)

Serialise a plain-array bundle (mesh, DOF maps, base flow, eigenmodes — no Ferrite
types) for external ParaView/VTU export tooling.
"""
function export_vtk_bundle(data_dir::AbstractString, fom, s0_full,
		master_eigenvalues, all_eigenvalues, all_modes)
	_nn = Ferrite.getnnodes(fom.grid)
	vtk_data = (;
		node_coords = Float64[fom.grid.nodes[i].x[c] for c in 1:2, i in 1:_nn],
		cell_connectivity = [collect(Int32, c.nodes) for c in fom.grid.cells],
		cell_dofs = [collect(Ferrite.celldofs(cell)) for cell in Ferrite.CellIterator(fom.dh)],
		free_dpim = fom.free_dpim,
		ndofs_total = Ferrite.ndofs(fom.dh),
		n_vel_dofs_per_cell = fom.n_vel_dofs_per_cell,
		s0_free_dpim = s0_full[fom.free_dpim],
		master_eigenvalues = master_eigenvalues,
		all_eigenvalues = all_eigenvalues,
		all_modes = Matrix{ComplexF64}(all_modes),
	)
	serialize(joinpath(data_dir, "vtk_data.jls"), vtk_data)
	return nothing
end

"""
	write_summary(out, results_dir, fom, master_eigenvalues,
	              r_mesh, r_fem, r_ss, r_ops, r_kvisc, r_eig, r_dpim; re0, ord, nvar)

Print the timing table to `out` and write the structured `summary.txt`.
"""
function write_summary(out::IO, results_dir::AbstractString, fom, master_eigenvalues,
		r_mesh, r_fem, r_ss, r_ops, r_kvisc, r_eig, r_dpim; re0, ord, nvar)
	println(out)
	println(out, _sep)
	println(out, "Kármán Vortex Street DPIM — Summary")
	println(out, "  Re₀ = $re0,  order = $ord,  NVAR = $nvar,  FOM = $(fom.n_free)")
	@printf(out, "  Hopf eigenvalue:  λ₁ = %+.6f %+.6f·i\n",
		real(master_eigenvalues[1]), imag(master_eigenvalues[1]))
	println(out, _dash)
	for (label, r) in (
		("[1] Mesh generation", r_mesh), ("[2] FEM setup", r_fem),
		("[3] Newton steady-state", r_ss), ("[4] Linear operators", r_ops),
		("[5] K_visc + h₀", r_kvisc), ("[6] Eigenproblem", r_eig),
		("[8] Cohomological solve", r_dpim))
		@printf(out, "  %-36s  %9.3f s  %8.2f GB\n", label, r.time, to_gb(r.bytes))
	end
	println(out, _sep)
	println(out, "Next:  julia --project=. solve_rom.jl   →   python3 compare_orders.py")

	open(joinpath(results_dir, "summary.txt"), "w") do io
		println(io, "example: 05_karman_vortex_street")
		@printf(io, "run_name: Re%.2f_ord%d\n", re0, ord)
		println(io, "model: 2D Navier-Stokes, Kármán vortex street, Ferrite P2/P1 Taylor-Hood")
		println(io, "n_free: $(fom.n_free)")
		println(io, "Re0: $re0")
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
	return nothing
end
