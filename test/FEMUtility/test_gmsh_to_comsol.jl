using Gmsh

# ── helpers ────────────────────────────────────────────────────────────────
const GMSH_TYPE_NAME = Dict(
	1 => "Line2", 2 => "Tri3", 3 => "Quad4", 4 => "Tet4",
	5 => "Hex8", 6 => "Prism6", 8 => "Line3", 9 => "Tri6",
	10 => "Quad9", 11 => "Tet10", 12 => "Hex27", 13 => "Prism18",
	15 => "Point",
)

function inspect_mesh(gmsh_file)
	println("─── Inspecting: $gmsh_file ───")
	gmsh.initialize()
	gmsh.open(gmsh_file)

	nodeTags, _, _ = gmsh.model.mesh.getNodes(-1, -1)
	println("  Nodes: $(length(nodeTags))")

	for dim in 0:3
		entities = gmsh.model.getEntities(dim)
		for (d, tag) in entities
			types, etags, _ = gmsh.model.mesh.getElements(d, tag)
			for (t, et) in zip(types, etags)
				tname = get(GMSH_TYPE_NAME, t, "type$t")
				println("  dim=$d tag=$tag  type=$t ($tname)  count=$(length(et))")
			end
		end
	end

	gmsh.finalize()
end

# ── roundtrip test with a synthetic P18 + Q9 mesh ──────────────────────────
function make_synthetic_p18_mesh(msh_out)
	# Manually build a tiny Gmsh mesh with:
	#   - 1 P18 prism volume element
	#   - 1 Q9 quadrangle surface element
	# Node positions for one P18 prism (18 nodes):
	# Bottom tri: n1=(0,0,0), n2=(1,0,0), n3=(0,1,0)
	# Top tri:    n4=(0,0,1), n5=(1,0,1), n6=(0,1,1)
	# Edge mids bottom: n7=(0.5,0,0), n8=(0.5,0.5,0), n9=(0,0.5,0)
	# Edge mids top:    n10=(0.5,0,1), n11=(0.5,0.5,1), n12=(0,0.5,1)
	# Vertical edges:   n13=(0,0,0.5), n14=(1,0,0.5), n15=(0,1,0.5)
	# Internal edge mids (prism18 specific):
	#   n16=(0.5,0,0.5), n17=(0.5,0.5,0.5), n18=(0,0.5,0.5)
	coords_P18 = [
		0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,  # 1-3
		0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,  # 4-6
		0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0,  # 7-9
		0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5, 1.0,  # 10-12
		0.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0, 0.5,  # 13-15
		0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5,  # 16-18
	]
	# Q9 face on z=0: nodes 1,2,3 + mids 7,8,9
	# But Q9 needs 9 nodes (2D quad), so let's use separate Q9 coords.
	# Q9: (-1,-1), (1,-1), (1,1), (-1,1), (0,-1), (1,0), (0,1), (-1,0), (0,0)
	coords_Q9 = [
		-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	]

	gmsh.initialize()
	gmsh.model.add("synthetic")

	# Volume entity
	gmsh.model.addDiscreteEntity(3, 1)
	gmsh.model.addDiscreteEntity(2, 2)

	# Nodes for P18 (tags 1..18)
	n18 = length(coords_P18) ÷ 3
	gmsh.model.mesh.addNodes(3, 1, collect(1:n18), coords_P18)

	# Nodes for Q9 (tags 19..27, on entity surf=2)
	n9 = length(coords_Q9) ÷ 3
	gmsh.model.mesh.addNodes(2, 2, collect((n18+1):(n18+n9)), coords_Q9)

	# P18 element: tag=1, nodes 1..18
	gmsh.model.mesh.addElements(3, 1, [13], [[1]], [collect(Int64, 1:18)])

	# Q9 element: tag=2, nodes 19..27
	gmsh.model.mesh.addElements(2, 2, [10], [[2]], [collect(Int64, (n18+1):(n18+n9))])

	gmsh.write(msh_out)
	gmsh.finalize()
	println("  Wrote synthetic mesh → $msh_out")
end

# ── read back a COMSOL file and dump summary ────────────────────────────────
function inspect_comsol(mphtxt_file)
	println("─── Inspecting COMSOL output: $mphtxt_file ───")
	open(mphtxt_file) do f
		for line in eachline(f)
			if any(k -> contains(line, k), [
				"# number of mesh vertices",
				"# Mesh vertex coordinates",
				"# type name",
				"# number of elements",
				"# Geometric entity indices",
				"# number of element types",
			])
				println("  ", line)
			end
		end
	end
end

# ─── main ──────────────────────────────────────────────────────────────────
beam_msh = joinpath(@__DIR__, "..", "..", "demo", "Gridap", "clamped_clamped_beam.msh")
synth_msh = joinpath(tempdir(), "synthetic_p18q9.msh")
comsol_out = joinpath(tempdir(), "out.mphtxt")
gmsh2_out = joinpath(tempdir(), "roundtrip.msh")

# 1. Inspect the provided beam mesh
inspect_mesh(beam_msh)
println()
println("  → The beam mesh has only LINEAR elements (Hex8, Quad4).")
println("    gmsh_to_comsol handles quadratic types (T6/Q9/P18) only.")
println()

# 2. Build a synthetic quadratic mesh and test gmsh → comsol
println("─── Building synthetic P18+Q9 mesh ───")
make_synthetic_p18_mesh(synth_msh)
println()

# 3. gmsh_to_comsol
include(joinpath(@__DIR__, "..", "..", "src", "FEMUtility", "GmshToComsol.jl"))
using .GmshToComsol
GmshToComsol.gmsh_to_comsol(synth_msh, comsol_out)
println("  gmsh_to_comsol done → $comsol_out")
println()

inspect_comsol(comsol_out)
println()

# 4. comsol → gmsh roundtrip
include(joinpath(@__DIR__, "..", "..", "src", "FEMUtility", "ComsolToGmsh.jl"))
using .ComsolToGmsh
ComsolToGmsh.comsol_to_gmsh(comsol_out, gmsh2_out)
println("  comsol_to_gmsh done → $gmsh2_out")
println()
inspect_mesh(gmsh2_out)

println("\n=== DONE ===")
