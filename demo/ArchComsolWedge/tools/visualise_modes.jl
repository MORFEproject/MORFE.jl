"""
Paraview visualisation for the arch_2_force mesh and eigenmodes.

Exports to results/paraview/:
  mesh.vtu          — undeformed geometry (open in Paraview to inspect the mesh)
  mode_01.vtu …    — one file per eigenmode; fields Re_u and Im_u
  modes.pvd         — PVD collection: open this and step through modes with the
					  time slider.  Apply Filters → Warp by Vector → Re_u to see
					  the deformed shape.

Alternatively, open results/paraview/mesh_viewer.html in a browser for an
in-browser WebGL view of the mesh without Paraview.

Usage:
  julia --project visualise_modes.jl
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../../..")))
	Pkg.add(["Ferrite", "Arpack", "LinearMaps", "StaticArrays", "WriteVTK"])
end
if !haskey(Pkg.project().dependencies, "WriteVTK")
	Pkg.add("WriteVTK")
end
Pkg.instantiate()

# loading Ferrite + WriteVTK activates MORFEWriteVTKExt automatically
using MORFE
using Ferrite
using WriteVTK
using SparseArrays
using LinearAlgebra
using Arpack
using LinearMaps
using StaticArrays
using Printf

include(joinpath(@__DIR__, "../setup/mesh.jl"))
include(joinpath(@__DIR__, "../setup/assembly.jl"))
include(joinpath(@__DIR__, "../setup/logging.jl"))

# ── Config ────────────────────────────────────────────────────────────────────
result_dir = joinpath(@__DIR__, "../results/visualise_geometry_and_modes")
out, _log = open_log(result_dir)
const N_MODES = 20

# ── Material (isotropic polysilicon, mm·kg·s) ─────────────────────────────────
const E = 160e3
const ν = 0.22
const ρ = 2.32e-3
const λ = E*ν / ((1+ν)*(1-2ν))
const μ = E / (2(1+ν))
print_header(out, E, ν, ρ, λ, μ, result_dir)

# ── Mesh + DOF handler ────────────────────────────────────────────────────────
const mesh_file = joinpath(@__DIR__, "../arch_2_force.mphtxt")
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

# ── Eigenproblem ──────────────────────────────────────────────────────────────
t_eig = @timed solve_eigenproblem(K, M,
	StructureModalDampingEigensolver(N_MODES, 0.0, 0.0);
	sorter! = (args...) -> nothing)
eigenproblem = t_eig.value;
eigenvalues, Y, X = get_eigenpairs(eigenproblem)
print_mode_table(out, eigenvalues)

# -----------------------------------------------------------------------
# §4  Export to Paraview
# -----------------------------------------------------------------------

const _out = joinpath(result_dir, "paraview")
println("\n§4  Writing Paraview files to $(_out)/ …")

write_paraview_mesh(joinpath(_out, "mesh"), grid;
	dh = dh, prescribed_dofs = ch.prescribed_dofs)

write_paraview_modes(_out, grid, dh, eigenvalues, Y, free; n_modes = N_MODES)

println("\nDone.  Open $(_out)/modes.pvd in Paraview.")
println("  → time slider steps through modes")
println("  → Filters → Warp by Vector → Re_u shows the deformed shape")
println("Or open $(_out)/mesh_viewer.html in a browser.")
