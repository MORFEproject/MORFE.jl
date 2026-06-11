"""
Logging helpers for arch_2_force.jl.

Provides TeeIO (stdout + file) and structured print functions so the
main driver can delegate all verbose output in one call per section.
"""

using Printf

# -----------------------------------------------------------------------
# TeeIO
# -----------------------------------------------------------------------

struct TeeIO <: IO
	a::IO
	b::IO
end
Base.unsafe_write(t::TeeIO, p::Ptr{UInt8}, n::UInt) =
	(unsafe_write(t.a, p, n); unsafe_write(t.b, p, n); n)
Base.write(t::TeeIO, x::UInt8) = (write(t.a, x); write(t.b, x);)
Base.flush(t::TeeIO) = (flush(t.a); flush(t.b))

"""
	open_log(results_dir) -> (out::TeeIO, log::IOStream)

Open `results_dir/summary.log` for writing and return a TeeIO that mirrors
all output to both stdout and the log file.  Call `close_log(log)` at the end.
"""
function open_log(results_dir::String)
	mkpath(results_dir)
	log = open(joinpath(results_dir, "summary.log"), "w")
	return TeeIO(stdout, log), log
end

close_log(log::IO) = close(log)

# -----------------------------------------------------------------------
# Section helpers
# -----------------------------------------------------------------------

const _SEP  = "=" ^ 70
const _DASH = "-" ^ 70

function print_header(out, cfg, ROM, N_EXT, NVAR, E, ν, ρ, λ, μ, results_dir)
	println(out);
	println(out, _SEP)
	println(out, "MORFE.jl — arch_2_force  |  $(cfg.label)")
	println(out, _DASH)
	@printf(out, "  Phys. master modes : %s   ROM = %d,  N_EXT = %d,  NVAR = %d\n",
		cfg.phys_modes, ROM, N_EXT, NVAR)
	@printf(out, "  max_degree = %d,  neig = %d\n", cfg.max_degree, cfg.neig)
	@printf(out, "  Rayleigh:  α = %.4g,  β = %.4g\n", cfg.rayleigh_α, cfg.rayleigh_β)
	if isempty(cfg.forces)
		println(out, "  External forcing: none (autonomous)")
	else
		for f in cfg.forces
			@printf(out, "  Force: Ω from mode %d, shape from mode %d, amp = %.4g\n",
				f.frequency_mode, f.shape_mode, f.amplitude)
		end
	end
	@printf(out, "  Resonance: style = %s,  tol_rel = %g\n",
		cfg.resonance.style, cfg.resonance.tolerance_rel)
	println(out, "  Results → $results_dir");
	println(out, _SEP)
	println(out)
	println(out, "Material: isotropic polysilicon (mm·kg·s)")
	@printf(out, "  E = %.4g MPa,  ν = %.2f,  ρ = %.4g kg/mm³  →  λ = %.4g,  μ = %.4g MPa\n",
		E, ν, ρ, λ, μ)
end

function print_header(out, E, ν, ρ, λ, μ, results_dir)
	println(out);
	println(out, _SEP)
	println(out, "MORFE.jl — arch_2_force")
	println(out, _DASH)
	println(out, "  Results → $results_dir");
	println(out, _SEP)
	println(out)
	println(out, "Material: isotropic polysilicon (mm·kg·s)")
	@printf(out, "  E = %.4g MPa,  ν = %.2f,  ρ = %.4g kg/mm³  →  λ = %.4g,  μ = %.4g MPa\n",
		E, ν, ρ, λ, μ)
end

function print_mesh_info(out, mesh_path, n_cells, n_nodes, n_dofs, n_constrained, n_free)
	println(out, "\n§1  Mesh: $(basename(mesh_path))")
	@printf(out, "     %d P18 cells,  %d nodes\n", n_cells, n_nodes)
	println(out, "\n§2  FEM: P18 quadratic prism (54 DOFs/cell)")
	@printf(out, "     %d total DOFs,  %d constrained,  %d free\n", n_dofs, n_constrained, n_free)
end

function print_mode_table(out, eigenvalues, master_indices)
	n_phys = length(eigenvalues) ÷ 2
	println(out, "\n§3  Eigenproblem  ($n_phys physical mode pairs)")
	println(out, "     " * "-"^62)
	@printf(out, "     %3s  %8s  %-20s  %-14s  %s\n",
		"Phys", "EV idx", "σ (decay)", "f (Hz)", "ω (rad/s)")
	println(out, "     " * "-"^62)
	for n in 1:n_phys
		λ = eigenvalues[2n-1]
		mark = (2n-1) ∈ master_indices ? "  ← MASTER ★" : ""
		@printf(out, "     %3d   %2d,%2d   %+.6e   %12.4f   %14.4f%s\n",
			n, 2n-1, 2n, real(λ), imag(λ)/(2π), imag(λ), mark)
	end
	println(out, "     " * "-"^62)
end

function print_mode_table(out, eigenvalues)
	n_phys = length(eigenvalues) ÷ 2
	println(out, "\n§3  Eigenproblem  ($n_phys physical mode pairs)")
	println(out, "     " * "-"^62)
	@printf(out, "     %3s  %8s  %-20s  %-14s  %s\n",
		"Phys", "EV idx", "σ (decay)", "f (Hz)", "ω (rad/s)")
	println(out, "     " * "-"^62)
	for n in 1:n_phys
		λ = eigenvalues[2n-1]
		@printf(out, "     %3d   %2d,%2d   %+.6e   %12.4f   %14.4f\n",
			n, 2n-1, 2n, real(λ), imag(λ)/(2π), imag(λ))
	end
	println(out, "     " * "-"^62)
end

function print_resonance_summary(out, resonance_set, mset, master_eigs, ext_eigs, tol_rel, NVAR, max_degree)
	_n_int = n_internal(resonance_set)
	_n_out = resonance_set.outer_resonances === nothing ? 0 : size(resonance_set.outer_resonances, 1)
	n_targets = _n_int + _n_out
	n_mono = length(mset)
	n_total = count(resonance_set.inner_resonances) +
			  (_n_out > 0 ? count(resonance_set.outer_resonances) : 0)
	println(out, "\n§4  Resonance set  (NVAR=$NVAR, max_degree=$max_degree, tol_rel=$tol_rel)")
	@printf(out, "     %d resonant / %d total  across %d targets\n",
		n_total, n_targets * n_mono, n_targets)
	for t in 1:n_targets
		cols = resonant_multiindices(resonance_set, t)
		λt = t ≤ _n_int ? master_eigs[t] : ext_eigs[t-_n_int]
		@printf(out, "     Target %d  (λ = %+.4e %+.4ei):  %d monomials\n",
			t, real(λt), imag(λt), length(cols))
		isempty(cols) || println(out, "       ", join(["$(mset.exponents[k])" for k in cols], "  "))
	end
end

function print_R_coefficients(out, R)
	println(out, "\n§5  Reduced dynamics R — nonzero coefficients:")
	for m in 1:length(R.poly.multiindex_set.exponents)
		c = R.poly.coefficients[:, m]
		any(abs.(c) .> 1e-12) || continue
		println(out, "     ", R.poly.multiindex_set.exponents[m], "   ", c)
	end
end

function print_summary(out, cfg, n_free, eigenvalues, master_indices, max_degree, NVAR, n_mono,
	t_eig, t_solve, results_dir)
	to_gb(b) = round(b / 1024^3; digits = 2)
	master_freqs = join(
		[@sprintf("%.4f", abs(eigenvalues[master_indices[2i-1]])/(2π)) for i in 1:length(cfg.phys_modes)],
		", ")
	println(out);
	println(out, _SEP)
	println(out, "MORFE.jl — arch_2_force (isotropic polysilicon, P18 prisms)")
	@printf(out, "  Config: %s   FOM: %d free DOFs\n", cfg.label, n_free)
	@printf(out, "  Master modes: phys. %s  →  %s Hz\n", cfg.phys_modes, master_freqs)
	@printf(out, "  max_degree = %d,  NVAR = %d,  monomials = %d\n", max_degree, NVAR, n_mono)
	println(out, _DASH)
	@printf(out, "  %-30s  %9.3f s  %6.2f GB\n", "§3  Eigenproblem", t_eig.time, to_gb(t_eig.bytes))
	@printf(out, "  %-30s  %9.3f s  %6.2f GB\n", "§5  Cohomological solve", t_solve.time, to_gb(t_solve.bytes))
	println(out, _SEP)
	println(out, "\nROM saved → $results_dir")
	println(out, "  W.jls,  R.jls,  summary.log")
end
