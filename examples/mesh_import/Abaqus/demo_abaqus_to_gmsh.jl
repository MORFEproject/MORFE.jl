# -------------------------------------------------------------------
# Demo: Abaqus → Gmsh mesh conversion (AbaqusToGmsh)
#
# Generates structured 3-D hexahedral meshes in Abaqus .inp format,
# converts them to Gmsh .msh, and verifies correctness.
#
# Demonstrated element types
#   C3D8  – 8-node linear hexahedron      → Gmsh type 5
#   C3D20 – 20-node quadratic hexahedron  → Gmsh type 17
#   abaqus_to_gmsh_linear – quadratic mesh downgraded to linear
# -------------------------------------------------------------------

import Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "Gmsh")
    Pkg.add("Gmsh")
end
Pkg.instantiate()

using Printf

include(joinpath(@__DIR__, "../../../ext/FEMUtility/AbaqusToGmsh.jl"))
using .AbaqusToGmsh
using Gmsh

const OUTDIR = joinpath(@__DIR__, "output")
mkpath(OUTDIR)

# ===================================================================
# 1.  Structured mesh generators
# ===================================================================

"""Write a structured nx×ny×nz C3D8 (linear hex) mesh to an Abaqus .inp file."""
function write_c3d8_inp(path, nx, ny, nz; Lx = 1.0, Ly = 1.0, Lz = 1.0)
	hx, hy, hz = Lx/nx, Ly/ny, Lz/nz
	nid(i, j, k) = i + j*(nx+1) + k*(nx+1)*(ny+1) + 1
	open(path, "w") do io
		println(io, "** Structured C3D8 mesh $(nx)×$(ny)×$(nz)")
		println(io, "*Node")
		for k in 0:nz, j in 0:ny, i in 0:nx
			@printf(io, "%d, %.6g, %.6g, %.6g\n", nid(i, j, k), i*hx, j*hy, k*hz)
		end
		println(io, "*Element, type=C3D8, elset=VOL")
		eid = 1
		for k in 0:(nz-1), j in 0:(ny-1), i in 0:(nx-1)
			ns = [nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k), nid(i, j+1, k),
				nid(i, j, k+1), nid(i+1, j, k+1), nid(i+1, j+1, k+1), nid(i, j+1, k+1)]
			println(io, "$eid, ", join(ns, ", "))
			eid += 1
		end
	end
	return (nx+1)*(ny+1)*(nz+1), nx*ny*nz
end

"""
Write a structured nx×ny×nz C3D20 (quadratic serendipity hex) mesh to .inp.

Nodes are placed at element corners and edge midpoints only — no face or
body-centre nodes (serendipity family, not full Lagrange).
"""
function write_c3d20_inp(path, nx, ny, nz; Lx = 1.0, Ly = 1.0, Lz = 1.0)
	hx, hy, hz = Lx/(2nx), Ly/(2ny), Lz/(2nz)

	# A node at fine-grid position (i,j,k) (i ∈ 0..2nx etc.) belongs to the
	# serendipity set iff at most one of i%2, j%2, k%2 is non-zero.
	node_coords = NTuple{3, Float64}[]
	node_map = Dict{NTuple{3, Int}, Int}()
	for k in 0:2nz, j in 0:2ny, i in 0:2nx
		(i%2 + j%2 + k%2) <= 1 || continue
		push!(node_coords, (i*hx, j*hy, k*hz))
		node_map[(i, j, k)] = length(node_coords)
	end

	open(path, "w") do io
		println(io, "** Structured C3D20 mesh $(nx)×$(ny)×$(nz)")
		println(io, "*Node")
		for (id, (x, y, z)) in enumerate(node_coords)
			@printf(io, "%d, %.6g, %.6g, %.6g\n", id, x, y, z)
		end
		println(io, "*Element, type=C3D20, elset=VOL")
		eid = 1
		for k in 0:(nz-1), j in 0:(ny-1), i in 0:(nx-1)
			I, J, K = 2i, 2j, 2k   # fine-grid origin of this element
			# Abaqus C3D20 ordering: 8 corners, then 12 edge midpoints
			ns = [
				node_map[(I, J, K)], node_map[(I+2, J, K)],
				node_map[(I+2, J+2, K)], node_map[(I, J+2, K)],
				node_map[(I, J, K+2)], node_map[(I+2, J, K+2)],
				node_map[(I+2, J+2, K+2)], node_map[(I, J+2, K+2)],
				node_map[(I+1, J, K)], node_map[(I+2, J+1, K)],
				node_map[(I+1, J+2, K)], node_map[(I, J+1, K)],
				node_map[(I+1, J, K+2)], node_map[(I+2, J+1, K+2)],
				node_map[(I+1, J+2, K+2)], node_map[(I, J+1, K+2)],
				node_map[(I, J, K+1)], node_map[(I+2, J, K+1)],
				node_map[(I+2, J+2, K+1)], node_map[(I, J+2, K+1)],
			]
			println(io, "$eid, ", join(ns, ", "))
			eid += 1
		end
	end
	return length(node_coords), nx*ny*nz
end

# ===================================================================
# 2.  Conversion with stats report
# ===================================================================

const GMSH_TYPE_NAMES = Dict(
	1 => "Line2", 8 => "Line3", 2 => "Tri3", 9 => "Tri6",
	3 => "Quad4", 16 => "Quad8", 4 => "Tet4", 11 => "Tet10",
	5 => "Hex8", 17 => "Hex20", 6 => "Wed6", 18 => "Wed15",
)

function convert_and_report(inp, msh, label; linear = false)
	println("\n", "─"^60)
	println("  $label")
	println("  Input:  $(basename(inp))")
	println("  Output: $(basename(msh))")
	linear ? abaqus_to_gmsh_linear(inp, msh) : abaqus_to_gmsh(inp, msh)

	gmsh.initialize()
	gmsh.open(msh)
	node_tags, _, _ = gmsh.model.mesh.getNodes(-1, -1)
	elem_types, _, _ = gmsh.model.mesh.getElements(-1, -1)
	@printf("  Nodes: %d\n", length(node_tags))
	for et in elem_types
		_, node_flat = gmsh.model.mesh.getElementsByType(et)
		_, _, _, npe, _, _ = gmsh.model.mesh.getElementProperties(et)
		ne = length(node_flat) ÷ npe
		name = get(GMSH_TYPE_NAMES, Int(et), "type$(et)")
		@printf("  Elements: %d × %s (%d nodes/elem)\n", ne, name, npe)
	end
	gmsh.finalize()
end

# ===================================================================
# 3.  Demo
# ===================================================================

println("="^60)
println("  AbaqusToGmsh demo — 3D hexahedral meshes")
println("="^60)

# --- C3D8: 3×3×2 linear hex mesh ----------------------------------
inp_c3d8 = joinpath(OUTDIR, "cube_c3d8.inp")
msh_c3d8 = joinpath(OUTDIR, "cube_c3d8.msh")

nn, ne = write_c3d8_inp(inp_c3d8, 3, 3, 2; Lx = 3.0, Ly = 3.0, Lz = 2.0)
@printf("\nGenerated C3D8 inp:  %d nodes, %d elements\n", nn, ne)
convert_and_report(inp_c3d8, msh_c3d8, "C3D8 → Gmsh type 5 (Hex8)")

# --- C3D20: 3×3×2 quadratic hex mesh ------------------------------
inp_c3d20 = joinpath(OUTDIR, "cube_c3d20.inp")
msh_c3d20 = joinpath(OUTDIR, "cube_c3d20.msh")

nn, ne = write_c3d20_inp(inp_c3d20, 3, 3, 2; Lx = 3.0, Ly = 3.0, Lz = 2.0)
@printf("\nGenerated C3D20 inp: %d nodes, %d elements\n", nn, ne)
convert_and_report(inp_c3d20, msh_c3d20, "C3D20 → Gmsh type 17 (Hex20)")

# --- C3D20 downgraded to linear -----------------------------------
msh_lin = joinpath(OUTDIR, "cube_c3d20_linear.msh")
convert_and_report(inp_c3d20, msh_lin,
	"C3D20 → abaqus_to_gmsh_linear → Gmsh type 5 (Hex8)"; linear = true)

# ===================================================================
# 4.  Geometric verification on a single C3D20 element
#
#  Check that PERM_C3D20 maps nodes to the correct Gmsh positions.
#  For a unit-cube element, Gmsh Hex20 node 10 must sit at (0, 0.5, 0)
#  (mid-point of corners 1=(0,0,0) and 4=(0,1,0), i.e. edge mid of the
#  bottom-face y-edge).  That is Abaqus node 12 = mid(4,1), and
#  PERM_C3D20[10] = 12.
# ===================================================================

println("\n", "─"^60)
println("  Geometric verification: single C3D20 unit-cube element")

inp_single = joinpath(OUTDIR, "single_c3d20.inp")
msh_single = joinpath(OUTDIR, "single_c3d20.msh")
write_c3d20_inp(inp_single, 1, 1, 1)
abaqus_to_gmsh(inp_single, msh_single)

gmsh.initialize()
gmsh.open(msh_single)

# Build tag → coordinate map
all_tags, all_coords, _ = gmsh.model.mesh.getNodes(-1, -1)
tag_to_coord = Dict{Int64, NTuple{3, Float64}}(
	Int64(all_tags[i]) => (all_coords[3i-2], all_coords[3i-1], all_coords[3i])
	for i in eachindex(all_tags)
)

# Retrieve the one Hex20 element's node connectivity
_, node_flat = gmsh.model.mesh.getElementsByType(17)
@assert length(node_flat) == 20 "Expected exactly 20 nodes for single Hex20"

# Gmsh position 10 should be mid(corner1, corner4) = (0, 0.5, 0)
gmsh_pos10 = tag_to_coord[Int64(node_flat[10])]
expected = (0.0, 0.5, 0.0)
err = maximum(abs(gmsh_pos10[k] - expected[k]) for k in 1:3)
@printf("  Gmsh Hex20 node at position 10: (%.4f, %.4f, %.4f)  expected (0, 0.5, 0)\n",
	gmsh_pos10...)
@assert err < 1e-12 "Node-ordering mismatch: PERM_C3D20 is incorrect"
println("  Node-ordering check passed ✓")

# Jacobian determinant must be positive at the element centroid
_, dets, _ = gmsh.model.mesh.getJacobian(node_flat[1] > 0 ? 1 : -1, [0.0, 0.0, 0.0])
# getJacobian takes an element tag, not node tag — use getElementsByType result
elem_tags20, _ = gmsh.model.mesh.getElementsByType(17)
_, dets, _ = gmsh.model.mesh.getJacobian(Int(elem_tags20[1]), [0.0, 0.0, 0.0])
@assert dets[1] > 0 "Negative Jacobian — element orientation is wrong"
@printf("  Jacobian at centroid: %.6g  (must be > 0) ✓\n", dets[1])

gmsh.finalize()

# ===================================================================
println("\n", "="^60)
println("  Output files written to examples/mesh_import/Abaqus/:")
for f in [inp_c3d8, msh_c3d8, inp_c3d20, msh_c3d20, msh_lin,
	inp_single, msh_single]
	println("    ", basename(f))
end
println("="^60)
println("\nDemo finished successfully.")
