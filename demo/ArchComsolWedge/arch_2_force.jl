"""
MORFE.jl demo — arch_2_force (isotropic polysilicon, COMSOL P18 mesh, forced).

Runs the full DPIM pipeline: mesh → FEM → eigenproblem → model → ROM.
All verbose output goes to both terminal and results/<config>/summary.log.

  arch_2_force.jl          ← this file (pipeline driver)
  setup/mesh.jl            ← mesh loading (P18 QuadraticWedge, Dirichlet BCs)
  setup/assembly.jl        ← SVK FEM assembler (isotropic, Ferrite backend)
  setup/logging.jl         ← TeeIO + structured print helpers
  tools/visualise_modes.jl ← export eigenmodes to VTK/browser viewer
  tools/node_dof_table.jl  ← build node → free-DOF mapping table
  results/<config>/        ← config.jl, W.jls, R.jls, summary.log
							  postprocess/compute_backbone.jl
							  postprocess/visualise_deformation.jl

Usage:  julia --project arch_2_force.jl [results/<config>/config.jl]
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
	Pkg.add(["Ferrite", "Arpack", "LinearMaps", "StaticArrays", "WriteVTK"])
end
Pkg.instantiate()

using MORFE, Ferrite, SparseArrays, LinearAlgebra, Arpack, LinearMaps, Serialization, StaticArrays, Printf

include(joinpath(@__DIR__, "setup/mesh.jl"))
include(joinpath(@__DIR__, "setup/assembly.jl"))
include(joinpath(@__DIR__, "setup/logging.jl"))

# ── Config ────────────────────────────────────────────────────────────────────
config_path = get(ARGS, 1, joinpath(@__DIR__, "results/mode_1_order_5_cnf/config.jl"))
cfg = include(config_path)
results_dir = dirname(abspath(config_path))
master_indices = vcat([[2n-1, 2n] for n in cfg.phys_modes]...)
conjugate_permutation = vcat([[2i, 2i-1] for i in 1:length(cfg.phys_modes)]...)
ROM = length(master_indices);
N_EXT = 2 * length(cfg.forces);
NVAR = ROM + N_EXT

out, _log = open_log(results_dir)

# ── Material (isotropic polysilicon, mm·kg·s) ─────────────────────────────────
const E = 160e3
const ν = 0.22
const ρ = 2.32e-3
const λ = E*ν / ((1+ν)*(1-2ν))
const μ = E / (2(1+ν))
print_header(out, cfg, ROM, N_EXT, NVAR, E, ν, ρ, λ, μ, results_dir)

# ── Mesh + DOF handler ────────────────────────────────────────────────────────
const mesh_file = joinpath(@__DIR__, "arch_2_force.mphtxt")
isfile(mesh_file) || error("Mesh not found: $mesh_file")
grid, constrained = load_arch_mesh(mesh_file)
ip = Lagrange{RefPrism, 2}()^3;
geo_ip = Lagrange{RefPrism, 2}()
cv = CellValues(QuadratureRule{RefPrism}(4), ip, geo_ip)
dh = DofHandler(grid);
add!(dh, :u, ip);
close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, constrained, (x, t) -> zeros(3), [1, 2, 3]));
close!(ch);
update!(ch, 0.0)
free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free));
n_free = length(free)
print_mesh_info(out, mesh_file, length(grid.cells), length(grid.nodes), ndofs(dh),
	length(ch.prescribed_dofs), n_free)

# ── Stiffness, mass, damping ──────────────────────────────────────────────────
K_full = allocate_matrix(dh);
M_full = allocate_matrix(dh)
assemble_KM!(K_full, M_full, dh, cv, λ, μ, ρ)
K = K_full[free, free]
M = M_full[free, free]
C = cfg.rayleigh_α .* M .+ cfg.rayleigh_β .* K

# ── Eigenproblem ──────────────────────────────────────────────────────────────
t_eig = @timed solve_eigenproblem(K, M,
	StructureModalDampingEigensolver(cfg.neig, cfg.rayleigh_α, cfg.rayleigh_β);
	sorter! = (args...) -> nothing)
eigenproblem = t_eig.value;
eigenvalues, Y, X = get_eigenpairs(eigenproblem)
print_mode_table(out, eigenvalues, master_indices)

# ── NL terms + model ──────────────────────────────────────────────────────────
mset  = all_multiindices_up_to(NVAR, cfg.max_degree; min_degree = 1)
ncols = length(mset)

select_master_modes_by_sorting(eigenproblem, ROM)
master_modes = Y[:, 1, master_indices];
left_eigenmodes = X[:, master_indices]
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[master_indices])
n_deriv = size(eigenproblem.eigenmodes, 2) - 1
master_modes_derivatives = zeros(ComplexF64, n_free, n_deriv, ROM)
for (r, idx) in enumerate(master_indices), k in 1:n_deriv
	master_modes_derivatives[:, k, r] .= Y[:, k+1, idx];
end

term_quad  = FerriteGeometricNonlinearity{2}(dh, cv, free_to_local, n_free, λ, μ; max_unique_cols = ncols)
term_cubic = FerriteGeometricNonlinearity{3}(dh, cv, free_to_local, n_free, λ, μ; max_unique_cols = ncols)

ext_freqs = ComplexF64[]
for f in cfg.forces
	Ω = abs(eigenvalues[2*f.frequency_mode-1])
	append!(ext_freqs, [complex(0.0, Ω), complex(0.0, -Ω)]);
end
model = if isempty(cfg.forces)
	NDOrderModel((K, C, M), (term_quad, term_cubic))
else
	forcing = Tuple(map(cfg.forces) do f
		fv = real((f.amplitude/2) .* (M * Y[:, 1, 2*f.shape_mode-1]))
		MultilinearMap((res, r) -> (res .+= fv * sum(r)), (0, 0), 1);
	end)
	NDOrderModel((K, C, M), (term_quad, term_cubic, forcing...), ExternalSystem(Tuple(ext_freqs)))
end

# ── Resonance set ─────────────────────────────────────────────────────────────
master_eigs = Vector{ComplexF64}(master_eigenvalues);
ext_eigs = Vector{ComplexF64}(ext_freqs)
tol_rel = cfg.resonance.tolerance_rel
tol_vec = [[tol_rel * abs(master_eigs[j]) for j in 1:ROM] for _ in 1:length(mset.exponents)]
resonance_set = if cfg.resonance.style == :cnf
	resonance_set_from_complex_normal_form_style(mset, master_eigs, tol_vec; external_eigenvalues = ext_eigs)
elseif cfg.resonance.style == :rnf
	resonance_set_from_real_normal_form_style(mset, master_eigs, conjugate_permutation, tol_vec;
		external_eigenvalues = ext_eigs)
else
	resonance_set_from_graph_style(mset, master_eigs, ext_eigs, ext_eigs, tol_rel)
end
print_resonance_summary(out, resonance_set, mset, master_eigs, ext_eigs, tol_rel, NVAR, cfg.max_degree)

# ── Cohomological solve ───────────────────────────────────────────────────────
if get(cfg, :check, false)
	print(out, "\nProceed with cohomological solve? [y/N]: ")
	readline() == "y" || (close_log(_log); exit(0))
end
t_solve = @timed solve_cohomological_problem(model, mset, master_eigenvalues,
	master_modes, left_eigenmodes, resonance_set;
	master_modes_derivatives = master_modes_derivatives,
	conjugate_permutation    = conjugate_permutation)
W, R = t_solve.value
print_R_coefficients(out, R)

# ── Save ──────────────────────────────────────────────────────────────────────
serialize(joinpath(results_dir, "W.jls"), W);
serialize(joinpath(results_dir, "R.jls"), R)
print_summary(out, cfg, n_free, eigenvalues, master_indices, cfg.max_degree, NVAR, ncols,
	t_eig, t_solve, results_dir)
close_log(_log)
