"""
Generate a pure-H27 GMSH mesh for the clamped-clamped beam benchmark.

Produces two output files:
  beam_h27.msh    — GMSH format, used by benchmark_ferrite.jl (Ferrite path)
  beam_h27.mphtxt — COMSOL format, drop-in replacement for beam.mphtxt in the
					MORFE2.0 benchmark (re-run with the same mesh)

Geometry: 1000 × 10 × 24 mm clamped-clamped beam (same as Gridap/Ferrite demo).
Mesh: 40 × 2 × 2 H27 elements (quadratic Lagrange hex).
  Nodes:      81 × 5 × 5 = 2025
  Constrained (two x-ends): 2 × 25 = 50
  Free DoFs:  (2025 − 50) × 3 = 5925  (≈ MORFE2.0's 5952)

Physical groups in beam_h27.msh:
  "Volume"     — volume entity (3D H27 cells)
  "Dirichlet"  — two x-end faces (clamped)
  "FreeSurface" — remaining four faces

Usage:
  julia --project demo/BenchmarkFerrite/generate_beam_mesh.jl
"""

using Gmsh

include(joinpath(@__DIR__, "../../src/FEMUtility/GmshToComsol.jl"))
using .GmshToComsol

const L  = 1000.0   # beam length (mm)
const W  = 10.0   # width
const H  = 24.0   # height/thickness
const NX = 40     # elements along x
const NY = 2     # elements along y
const NZ = 2     # elements along z

const msh_path    = joinpath(@__DIR__, "beam_h27.msh")
const mphtxt_path = joinpath(@__DIR__, "../../legacy_morfe/MORFE2.0/input/beam_h27.mphtxt")

# -----------------------------------------------------------------------
# 1. GMSH geometry and mesh
# -----------------------------------------------------------------------

gmsh.initialize()
gmsh.model.add("beam_h27")
gmsh.option.setNumber("General.Verbosity", 2)

v = gmsh.model.occ.addBox(0.0, 0.0, 0.0, L, W, H)
gmsh.model.occ.synchronize()

# -----------------------------------------------------------------------
# 2. Transfinite structured hex mesh
# -----------------------------------------------------------------------

# Set number of nodes (= elements + 1) on each curve according to its direction.
for (_, tag) in gmsh.model.getEntities(1)
	xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
	dx = xmax - xmin;
	dy = ymax - ymin;
	dz = zmax - zmin
	n = if dx > dy && dx > dz
		NX + 1   # along beam length
	elseif dy > dz
		NY + 1   # along width
	else
		NZ + 1   # along thickness
	end
	gmsh.model.mesh.setTransfiniteCurve(tag, n)
end

# Classify end faces (Dirichlet) vs side/top/bottom faces (FreeSurface).
dirichlet_tags = Int[]
freesurface_tags = Int[]
for (_, tag) in gmsh.model.getEntities(2)
	xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(2, tag)
	if isapprox(xmax - xmin, 0.0; atol = 1.0)   # flat in x → end face
		push!(dirichlet_tags, tag)
	else
		push!(freesurface_tags, tag)
	end
	gmsh.model.mesh.setTransfiniteSurface(tag)
	gmsh.model.mesh.setRecombine(2, tag)
end

gmsh.model.mesh.setTransfiniteVolume(v)

# -----------------------------------------------------------------------
# 3. Physical groups
# -----------------------------------------------------------------------

gmsh.model.addPhysicalGroup(3, [v], -1, "Volume")
gmsh.model.addPhysicalGroup(2, dirichlet_tags, -1, "Dirichlet")
gmsh.model.addPhysicalGroup(2, freesurface_tags, -1, "FreeSurface")

# -----------------------------------------------------------------------
# 4. Generate and export
# -----------------------------------------------------------------------

gmsh.model.mesh.generate(3)
gmsh.model.mesh.setOrder(2)   # H27 quadratic Lagrange

println("Nodes: ", length(gmsh.model.mesh.getNodes(-1, -1)[1]))
gmsh.write(msh_path)
println("Wrote: ", msh_path)

gmsh.finalize()

# -----------------------------------------------------------------------
# 5. Cross-export to COMSOL (.mphtxt) for the MORFE2.0 benchmark
# -----------------------------------------------------------------------

gmsh_to_comsol(msh_path, mphtxt_path)
println("Wrote: ", mphtxt_path)
println()
println("Free DoFs (expected): ", (2025 - 50) * 3, "  (target ≈ 5952)")
