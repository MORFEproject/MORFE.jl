"""
Generate results/node_dof_table.txt.

Columns: node_number  x_coordinate  y_coordinate  z_coordinate
		 x_dof_number  y_dof_number  z_dof_number

DOF numbers are free (unconstrained) DOF indices — the same indexing used in
the reduced K/M/C matrices.  Constrained DOFs are written as 0.

Run after the main environment is set up (Manifest.toml must exist).
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Ferrite
using Printf

include(joinpath(@__DIR__, "../setup/mesh.jl"))

# -----------------------------------------------------------------------
# Mesh + DOF setup (mirrors arch_2_force.jl §1–§2)
# -----------------------------------------------------------------------
const _mesh_path = joinpath(
	@__DIR__, "../arch_2_force.mphtxt")
isfile(_mesh_path) || error("Mesh not found: $_mesh_path")

println("Loading mesh …")
(grid, constrained_nodes) = load_arch_mesh(_mesh_path)

const ip = Lagrange{RefPrism, 2}()^3

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, constrained_nodes, (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)

free          = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))

println("  Total nodes : ", Ferrite.getnnodes(grid))
println("  Total DOFs  : ", ndofs(dh))
println("  Free DOFs   : ", length(free))

# -----------------------------------------------------------------------
# Build node → global DOFs map (single cell pass)
# -----------------------------------------------------------------------
println("Building node→DOF map …")
nn = Ferrite.getnnodes(grid)
node_global_dofs = Vector{NTuple{3, Int}}(undef, nn)
seen = falses(nn)

for cell_id in 1:Ferrite.getncells(grid)
	cell_nodes = grid.cells[cell_id].nodes       # NTuple{18,Int}
	cell_dofs  = Ferrite.celldofs(dh, cell_id)   # length 54, interleaved x/y/z
	for (k, node_id) in enumerate(cell_nodes)
		seen[node_id] && continue
		base = 3 * (k - 1)
		node_global_dofs[node_id] = (cell_dofs[base+1],
			cell_dofs[base+2],
			cell_dofs[base+3])
		seen[node_id] = true
	end
end

# -----------------------------------------------------------------------
# Write table
# -----------------------------------------------------------------------
fmt_dof(d) = d == 0 ? "-" : string(d)

mkpath(joinpath(@__DIR__, "../results/visualise_geometry_and_modes"))
out_path = joinpath(@__DIR__, "../results/visualise_geometry_and_modes/node_dof_table.txt")

println("Writing table → ", out_path)
open(out_path, "w") do io
	println(io, "node_number x_coordinate y_coordinate z_coordinate x_dof_number y_dof_number z_dof_number")
	for node_id in 1:nn
		coord = grid.nodes[node_id].x
		gdofs = node_global_dofs[node_id]
		fdof_x = get(free_to_local, gdofs[1], 0)
		fdof_y = get(free_to_local, gdofs[2], 0)
		fdof_z = get(free_to_local, gdofs[3], 0)
		coord_str = @sprintf("%.10e %.10e %.10e", coord[1], coord[2], coord[3])
		println(io, "$node_id $coord_str $(fmt_dof(fdof_x)) $(fmt_dof(fdof_y)) $(fmt_dof(fdof_z))")
	end
end

println("Done.  Rows written: ", nn)
