"""
Paraview visualisation for the arch_2_force mesh and eigenmodes.

Exports to results/visualise_structure_and_modes/paraview/:
  mesh.vtu               — undeformed geometry (open in Paraview to inspect the mesh)
  boundary_entities.vtu  — boundary faces with cell data `entity_id` = COMSOL
						   surface number. Colour by entity_id, or select a face
						   ('s' + click) and read entity_id in the Selection
						   Inspector — this is the number to use in
						   dirichlet = Set([...]) in mechanical_model / load_comsol_grid.
  mode_01.vtu …          — one file per eigenmode; fields Re_u and Im_u
  modes.pvd              — PVD collection: open this and step through modes with the
						   time slider.  Apply Filters → Warp by Vector → Re_u to see
						   the deformed shape.

Usage:
  julia --project visualise_structure_and_modes.jl
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../../..")))
	Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays", "WriteVTK"])
end
if !haskey(Pkg.project().dependencies, "WriteVTK")
	Pkg.add("WriteVTK")
end
if !haskey(Pkg.project().dependencies, "FerriteGmsh")
	Pkg.add("FerriteGmsh")
end
Pkg.instantiate()

# loading Ferrite + FerriteGmsh + Arpack + LinearMaps activates MORFEStructuralSVK automatically;
# loading Ferrite + WriteVTK activates MORFEWriteVTKExt automatically.
using MORFE
using Ferrite
using FerriteGmsh
using WriteVTK
using Arpack
using LinearMaps
using StaticArrays
using SparseArrays
using LinearAlgebra
using Printf

SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)

# ── Config ────────────────────────────────────────────────────────────────────
const result_dir = joinpath(@__DIR__, "../results/visualise_structure_and_modes")
mkpath(result_dir)
const N_MODES = 20

# ── Material (isotropic polysilicon, mm·kg·s) ─────────────────────────────────
const E = 160e3
const ν = 0.22
const ρ = 2.32e-3
const λ = E * ν / ((1 + ν) * (1 - 2ν))
const μ = E / (2(1 + ν))

println("Material: E=$E  ν=$ν  ρ=$ρ  →  λ=$(round(λ; digits=1))  μ=$(round(μ; digits=1))")

# ── Mesh + DOF handler ────────────────────────────────────────────────────────
const mesh_file = joinpath(@__DIR__, "../arch_2_force.mphtxt")
isfile(mesh_file) || error("Mesh not found: $mesh_file")

println("\n§1  Loading mesh …")
grid, constrained = SVK.load_comsol_grid(mesh_file, Set([1, 11]))

ip = Lagrange{RefPrism, 2}()^3
geo_ip = Lagrange{RefPrism, 2}()
cv = CellValues(QuadratureRule{RefPrism}(4), ip, geo_ip)
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, constrained, (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)
free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
n_free = length(free)

@printf("  Cells: %d   Nodes: %d   Total DOFs: %d   Constrained: %d   Free: %d\n",
	length(grid.cells), length(grid.nodes), ndofs(dh), length(ch.prescribed_dofs), n_free)

# ── Stiffness + mass assembly ─────────────────────────────────────────────────
println("\n§2  Assembling K, M …")
K_full = allocate_matrix(dh)
M_full = allocate_matrix(dh)
ferrite_assemble_KM!(K_full, M_full, dh, cv, λ, μ, ρ)
K = K_full[free, free]
M = M_full[free, free]

# ── Eigenproblem ──────────────────────────────────────────────────────────────
println("\n§3  Eigenproblem ($(N_MODES) modes) …")
t_eig = @timed solve_eigenproblem(K, M,
	StructureModalDampingEigensolver(N_MODES, 0.0, 0.0);
	sorter! = (args...) -> nothing)
eigenproblem = t_eig.value
eigenvalues, Y, X = get_eigenpairs(eigenproblem)

println("  Eigenfrequencies (Hz):")
for i in 1:2:length(eigenvalues)
	@printf("    mode %2d: %10.4f Hz\n", (i + 1) ÷ 2, abs(eigenvalues[i]) / (2π))
end
@printf("  (solved in %.1f s)\n", t_eig.time)

# ── §4  Export modes ──────────────────────────────────────────────────────────
const _out = joinpath(result_dir, "paraview")
mkpath(_out)
println("\n§4  Writing Paraview files to $(_out)/ …")

write_paraview_mesh(joinpath(_out, "mesh"), grid;
	dh = dh, prescribed_dofs = ch.prescribed_dofs)

write_paraview_modes(_out, grid, dh, eigenvalues, Y, free; n_modes = N_MODES)

# ── §4b  Boundary faces with COMSOL surface numbers ──────────────────────────
# Every T6/Q9 boundary face is exported as a linear tri/quad with cell data
# `entity_id` = COMSOL geometric surface number (1-based, as in the COMSOL GUI
# and dirichlet = Set([...])).  Selecting a face in Paraview shows its number.
println("\n§4b  Boundary entity map …")
(nn, n2c,
	e2nT6, e2gT6, e2nQ9, e2gQ9,
	_, _, _, _, _, _) = SVK._read_mesh(mesh_file)

points_b = reshape(n2c, 3, nn)   # arch mesh coords are already in SI — no scaling

neT6 = length(e2nT6) ÷ 6
neQ9 = length(e2nQ9) ÷ 9
bcells = Vector{WriteVTK.MeshCell}(undef, neT6 + neQ9)
entity_id = Vector{Int32}(undef, neT6 + neQ9)

for i in 1:neT6
	base = (i - 1) * 6
	bcells[i] = MeshCell(VTKCellTypes.VTK_TRIANGLE,
		[e2nT6[base+1], e2nT6[base+2], e2nT6[base+3]])
	entity_id[i] = e2gT6[i]
end
for i in 1:neQ9
	base = (i - 1) * 9
	bcells[neT6+i] = MeshCell(VTKCellTypes.VTK_QUAD,
		[e2nQ9[base+1], e2nQ9[base+2], e2nQ9[base+4], e2nQ9[base+3]])
	entity_id[neT6+i] = e2gQ9[i]
end

vtk_grid(joinpath(_out, "boundary_entities"), points_b, bcells) do vtk
	vtk_cell_data(vtk, entity_id, "entity_id")
	vtk_point_data(vtk, Int32.(1:nn), "node_id")
end
println("  Boundary faces → $(joinpath(_out, "boundary_entities")).vtu")
println("  ($neT6 tris + $neQ9 quads, entity_id ∈ [$(minimum(entity_id)), $(maximum(entity_id))])")

println("\nDone.  In Paraview:")
println("  • modes.pvd → time slider steps through modes")
println("    → Filters → Warp by Vector → Re_u to see the deformed shape")
println("  • boundary_entities.vtu → colour by entity_id to identify COMSOL surface numbers")
println("    → select a face ('s' + click) and read entity_id in the Selection Inspector")
