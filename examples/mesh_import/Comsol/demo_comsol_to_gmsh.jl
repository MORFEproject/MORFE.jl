# -------------------------------------------------------------------
# Demo: COMSOL → Gmsh mesh conversion (ComsolToGmsh / GmshToComsol)
#
# Generates structured 3-D meshes in COMSOL .mphtxt format and converts
# them to Gmsh .msh.  Also demonstrates the inverse (gmsh_to_comsol) and
# a full round-trip.
#
# Demonstrated element types
#   hex2 (H27) – 27-node quadratic hexahedron  → Gmsh type 12
#   tet2 (T10) – 10-node quadratic tetrahedron → Gmsh type 11
#   comsol_to_gmsh_linear – quadratic mesh downgraded to linear
# -------------------------------------------------------------------

import Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "Gmsh")
    Pkg.add("Gmsh")
end
Pkg.instantiate()

using Printf

include(joinpath(@__DIR__, "../../../ext/FEMUtility/ComsolToGmsh.jl"))
include(joinpath(@__DIR__, "../../../ext/FEMUtility/GmshToComsol.jl"))
using .ComsolToGmsh
using .GmshToComsol
using Gmsh

const OUTDIR = joinpath(@__DIR__, "output")
mkpath(OUTDIR)

# ===================================================================
# 1.  COMSOL .mphtxt file generators
# ===================================================================

# COMSOL hex2 (H27) node offsets from element fine-grid origin (I,J,K).
# Derived from INV_PERM_H27 (the COMSOL→Gmsh permutation inverse) and
# the Gmsh type-12 reference-element coordinates.
const H27_OFFSETS = NTuple{3, Int}[
	(0, 0, 0), (2, 0, 0), (
		0, 2, 0), (2, 2, 0),   # corners 1-4  (COMSOL: 3↔4 swapped vs Gmsh)
	(0, 0, 2), (2, 0, 2), (
		0, 2, 2), (2, 2, 2),   # corners 5-8  (COMSOL: 7↔8 swapped vs Gmsh)
	(1, 0, 0), (2, 1, 0), (
		2, 0, 1), (0, 1, 0),   # edge mids 9-12
	(1, 2, 0), (0, 2, 1), (
		1, 0, 1), (1, 2, 2),   # edge mids 13-16
	(0, 1, 1), (1, 1, 2), (
		1, 1, 1), (2, 1, 1),   # face/body 17-20 (x−,z+,body,x+)
	(0, 0, 1), (1, 0, 2), (
		2, 2, 1), (1, 2, 1),   # 21-24 (edge mid, edge mid, edge mid, y+ face)
	(2, 1, 2), (1, 1, 0), (
		0, 1, 2),           # 25-27 (edge mid, z− face, edge mid)
]

"""
Write a structured nx×ny×nz COMSOL hex2 (H27) mesh to a .mphtxt file.

The node layout uses the full 3×3×3 Lagrange grid per element (corners,
edge midpoints, face centres, body centre).  Nodes are shared between
adjacent elements.
"""
function write_h27_mphtxt(path, nx, ny, nz; Lx = 1.0, Ly = 1.0, Lz = 1.0)
	hx, hy, hz = Lx/(2nx), Ly/(2ny), Lz/(2nz)

	# All grid points: i ∈ 0..2nx, j ∈ 0..2ny, k ∈ 0..2nz (full Lagrange)
	nid(i, j, k) = i + j*(2nx+1) + k*(2nx+1)*(2ny+1)   # 0-indexed

	nn = (2nx+1)*(2ny+1)*(2nz+1)
	ne = nx*ny*nz

	open(path, "w") do io
		# --- header ---
		println(io, "# Created by MORFE_jl.")
		println(io)
		println(io, "# Major & minor version")
		println(io, "0 1")
		println(io)
		println(io, "1 # number of tags")
		println(io, "# Tags")
		println(io, "5 mesh1")
		println(io)
		println(io, "1 # number of types")
		println(io, "# Types")
		println(io, "3 obj")
		println(io)
		println(io)
		println(io, "# --------- Object 0 ----------")
		println(io)
		println(io)
		println(io, "0 1 # version")
		println(io, "4 Mesh # class")
		println(io, "4 # version")
		println(io, "3 # sdim")
		println(io, "$nn # number of mesh vertices")
		println(io, "0 # lowest mesh vertex index")
		println(io)
		println(io, "# Mesh vertex coordinates")
		for k in 0:2nz, j in 0:2ny, i in 0:2nx
			@printf(io, "%.6g %.6g %.6g\n", i*hx, j*hy, k*hz)
		end
		println(io)
		println(io, "1 # number of element types")

		# --- hex2 element block ---
		println(io)
		println(io, "# --------- Type 0 ----------")
		println(io)
		println(io, "hex2 # type name")
		println(io, "# number of nodes per element")
		println(io, "27")
		println(io, "# number of elements")
		println(io, "$ne")
		println(io, "# Elements")
		for k in 0:(nz-1), j in 0:(ny-1), i in 0:(nx-1)
			I, J, K = 2i, 2j, 2k
			ns = [nid(I + di, J + dj, K + dk) for (di, dj, dk) in H27_OFFSETS]
			println(io, join(ns, " "))
		end
		println(io, "# number of parameter values per element")
		println(io, "0")
		println(io, "# Geometric entity indices")
		for eid in 1:ne
			println(io, "$eid")   # unique per-element IDs used as Gmsh element tags
		end
	end
	return nn, ne
end

"""
Write a single COMSOL tet2 (T10) element (unit tetrahedron) to a .mphtxt file.

COMSOL tet2 corner ordering swaps corners 3↔4 and edge-mids 6↔7 vs Gmsh.
Corners: 1=(0,0,0), 2=(1,0,0), 3=(0,0,1) [Gmsh c4], 4=(0,1,0) [Gmsh c3].
"""
function write_t10_single_mphtxt(path)
	# Node coordinates in COMSOL tet2 ordering (0-indexed)
	coords = [
		(0.0, 0.0, 0.0),    # COMSOL 1  = Gmsh corner 1
		(1.0, 0.0, 0.0),    # COMSOL 2  = Gmsh corner 2
		(0.0, 0.0, 1.0),    # COMSOL 3  = Gmsh corner 4
		(0.0, 1.0, 0.0),    # COMSOL 4  = Gmsh corner 3
		(0.5, 0.0, 0.0),    # COMSOL 5  = mid(1,2)
		(0.0, 0.5, 0.0),    # COMSOL 6  = mid(1,4) in COMSOL = Gmsh mid(1,3)
		(0.5, 0.5, 0.0),    # COMSOL 7  = mid(2,4) in COMSOL = Gmsh mid(2,3)
		(0.0, 0.0, 0.5),    # COMSOL 8  = mid(1,3) in COMSOL = Gmsh mid(1,4)
		(0.0, 0.5, 0.5),    # COMSOL 9  = mid(3,4) in COMSOL = Gmsh mid(3,4)
		(0.5, 0.0, 0.5),    # COMSOL 10 = mid(2,3) in COMSOL = Gmsh mid(2,4)
	]
	nn, ne = 10, 1

	open(path, "w") do io
		println(io, "# Created by MORFE_jl.")
		println(io)
		println(io, "# Major & minor version")
		println(io, "0 1")
		println(io)
		println(io, "1 # number of tags")
		println(io, "# Tags")
		println(io, "5 mesh1")
		println(io)
		println(io, "1 # number of types")
		println(io, "# Types")
		println(io, "3 obj")
		println(io)
		println(io)
		println(io, "# --------- Object 0 ----------")
		println(io)
		println(io)
		println(io, "0 1 # version")
		println(io, "4 Mesh # class")
		println(io, "4 # version")
		println(io, "3 # sdim")
		println(io, "$nn # number of mesh vertices")
		println(io, "0 # lowest mesh vertex index")
		println(io)
		println(io, "# Mesh vertex coordinates")
		for (x, y, z) in coords
			@printf(io, "%.6g %.6g %.6g\n", x, y, z)
		end
		println(io)
		println(io, "1 # number of element types")
		println(io)
		println(io, "# --------- Type 0 ----------")
		println(io)
		println(io, "tet2 # type name")
		println(io, "# number of nodes per element")
		println(io, "10")
		println(io, "# number of elements")
		println(io, "$ne")
		println(io, "# Elements")
		println(io, "0 1 2 3 4 5 6 7 8 9")
		println(io, "# number of parameter values per element")
		println(io, "0")
		println(io, "# Geometric entity indices")
		println(io, "1")
	end
	return nn, ne
end

# ===================================================================
# 2.  Conversion with stats report
# ===================================================================

const GMSH_TYPE_NAMES = Dict(
	1=>"Line2", 8=>"Line3", 2=>"Tri3", 9=>"Tri6",
	3=>"Quad4", 16=>"Quad8", 4=>"Tet4", 11=>"Tet10",
	5=>"Hex8", 12=>"Hex27", 6=>"Wed6", 13=>"Wed18",
)

function convert_and_report(inp, out, label; linear = false)
	println("\n", "─"^60)
	println("  $label")
	println("  Input:  $(basename(inp))")
	println("  Output: $(basename(out))")
	linear ? comsol_to_gmsh_linear(inp, out) : comsol_to_gmsh(inp, out)

	gmsh.initialize()
	gmsh.open(out)
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
println("  ComsolToGmsh demo — 3D hexahedral and tetrahedral meshes")
println("="^60)

# --- H27: 3×3×2 quadratic hex mesh --------------------------------
comsol_h27 = joinpath(OUTDIR, "cube_h27.mphtxt")
gmsh_h27 = joinpath(OUTDIR, "cube_h27.msh")

nn, ne = write_h27_mphtxt(comsol_h27, 3, 3, 2; Lx = 3.0, Ly = 3.0, Lz = 2.0)
@printf("\nGenerated H27 mphtxt: %d nodes, %d elements\n", nn, ne)
convert_and_report(comsol_h27, gmsh_h27,
	"COMSOL hex2 → Gmsh type 12 (Hex27)")

# --- H27 downgraded to linear Hex8 --------------------------------
gmsh_h27_lin = joinpath(OUTDIR, "cube_h27_linear.msh")
convert_and_report(comsol_h27, gmsh_h27_lin,
	"COMSOL hex2 → comsol_to_gmsh_linear → Gmsh type 5 (Hex8)";
	linear = true)

# --- T10: single quadratic tet element ----------------------------
comsol_t10 = joinpath(OUTDIR, "single_t10.mphtxt")
gmsh_t10 = joinpath(OUTDIR, "single_t10.msh")

nn, ne = write_t10_single_mphtxt(comsol_t10)
@printf("\nGenerated T10 mphtxt: %d nodes, %d elements\n", nn, ne)
convert_and_report(comsol_t10, gmsh_t10,
	"COMSOL tet2 → Gmsh type 11 (Tet10)")

# ===================================================================
# 4.  Geometric verification
#
#  For H27 (perm_H27 = [1,2,4,3,...]):
#    Gmsh[3] = COMSOL[4] → in the unit-cube element, COMSOL corner 4
#    is at (1,1,0), which should be Gmsh corner 3 = (1,1,0). ✓
#
#  For T10 (perm_T10 = [1,2,4,3,5,7,6,8,9,10]):
#    Gmsh[3] = COMSOL[4] = (0,1,0)   (Gmsh corner 3) ✓
#    Gmsh[4] = COMSOL[3] = (0,0,1)   (Gmsh corner 4) ✓
#    Gmsh[6] = COMSOL[7] = (0.5,0.5,0) (Gmsh mid(2,3)) ✓
# ===================================================================

println("\n", "─"^60)
println("  Geometric verification: single H27 unit-cube element")

comsol_h27_1 = joinpath(OUTDIR, "single_h27.mphtxt")
gmsh_h27_1 = joinpath(OUTDIR, "single_h27.msh")
write_h27_mphtxt(comsol_h27_1, 1, 1, 1)
comsol_to_gmsh(comsol_h27_1, gmsh_h27_1)

gmsh.initialize()
gmsh.open(gmsh_h27_1)

all_tags, all_coords, _ = gmsh.model.mesh.getNodes(-1, -1)
tag_to_coord = Dict{Int64, NTuple{3, Float64}}(
	Int64(all_tags[i]) => (all_coords[3i-2], all_coords[3i-1], all_coords[3i])
	for i in eachindex(all_tags)
)
_, node_flat = gmsh.model.mesh.getElementsByType(12)
@assert length(node_flat) == 27 "Expected 27 nodes for single Hex27"

# Gmsh position 3 = COMSOL[perm_H27[3]] = COMSOL[4] = corner at (1,1,0)
check_pos, expected = 3, (1.0, 1.0, 0.0)
got = tag_to_coord[Int64(node_flat[check_pos])]
err = maximum(abs(got[k] - expected[k]) for k in 1:3)
@printf("  Gmsh H27 node at position 3: (%.4f, %.4f, %.4f)  expected (1, 1, 0)\n", got...)
@assert err < 1e-12 "H27 node-ordering mismatch"
println("  Node-ordering check passed ✓")

_, dets,
_ = gmsh.model.mesh.getJacobian(
	Int(gmsh.model.mesh.getElementsByType(12)[1][1]), [0.0, 0.0, 0.0])
@assert dets[1] > 0 "Negative Jacobian — element orientation wrong"
@printf("  Jacobian at centroid: %.6g  (must be > 0) ✓\n", dets[1])

gmsh.finalize()

# ---- T10 node-ordering check ------------------------------------
println()
println("  Geometric verification: single T10 unit-tet element")

comsol_to_gmsh(comsol_t10, gmsh_t10)   # already written above
gmsh.initialize()
gmsh.open(gmsh_t10)

all_tags, all_coords, _ = gmsh.model.mesh.getNodes(-1, -1)
tag_to_coord = Dict{Int64, NTuple{3, Float64}}(
	Int64(all_tags[i]) => (all_coords[3i-2], all_coords[3i-1], all_coords[3i])
	for i in eachindex(all_tags)
)
_, node_flat = gmsh.model.mesh.getElementsByType(11)
@assert length(node_flat) == 10 "Expected 10 nodes for single Tet10"

checks = [(3, (0.0, 1.0, 0.0), "corner 3 (Gmsh) = COMSOL[4]"),
	(4, (0.0, 0.0, 1.0), "corner 4 (Gmsh) = COMSOL[3]"),
	(6, (0.5, 0.5, 0.0), "mid(2,3) (Gmsh) = COMSOL[7]")]
for (pos, exp, desc) in checks
	local got = tag_to_coord[Int64(node_flat[pos])]
	local err = maximum(abs(got[k] - exp[k]) for k in 1:3)
	@printf("  Position %2d %-28s → (%.2f, %.2f, %.2f)\n", pos, desc, got...)
	@assert err < 1e-12 "T10 mismatch at position $pos"
end
println("  Node-ordering checks passed ✓")

gmsh.finalize()

# ===================================================================
# 5.  Round-trip: COMSOL → Gmsh → COMSOL → Gmsh
# ===================================================================

println("\n", "─"^60)
println("  Round-trip: H27  COMSOL → Gmsh → COMSOL → Gmsh")

comsol_rt = joinpath(OUTDIR, "rt_h27.mphtxt")
gmsh_rt = joinpath(OUTDIR, "rt_h27.msh")

gmsh_to_comsol(gmsh_h27_1, comsol_rt)       # Gmsh → COMSOL
comsol_to_gmsh(comsol_rt, gmsh_rt)         # COMSOL → Gmsh

gmsh.initialize()
gmsh.open(gmsh_rt)
all_tags_rt, all_coords_rt, _ = gmsh.model.mesh.getNodes(-1, -1)
_, node_flat_rt = gmsh.model.mesh.getElementsByType(12)
gmsh.finalize()

# Compare connectivity from the first and second Gmsh files
gmsh.initialize()
gmsh.open(gmsh_h27_1)
_, node_flat_orig = gmsh.model.mesh.getElementsByType(12)
gmsh.finalize()

@assert node_flat_rt == node_flat_orig "Round-trip connectivity mismatch!"
println("  Element connectivity preserved across round-trip ✓")

# ===================================================================
println("\n", "="^60)
println("  Output files written to examples/mesh_import/Comsol/:")
for f in [comsol_h27, gmsh_h27, gmsh_h27_lin, comsol_t10, gmsh_t10,
	comsol_h27_1, gmsh_h27_1, comsol_rt, gmsh_rt]
	println("    ", basename(f))
end
println("="^60)
println("\nDemo finished successfully.")
