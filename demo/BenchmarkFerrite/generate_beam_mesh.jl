"""
Generate a pure-H27 GMSH mesh for the clamped-clamped beam benchmark.

Exposes `generate_beam_mesh(nx, ny, nz; output_dir, legacy_dir, max_order)`
which writes:
  beam_h27_{nx}x{ny}x{nz}.msh    — GMSH format (Ferrite path)
  beam_h27_{nx}x{ny}x{nz}.mphtxt — COMSOL format (legacy MORFE2.0 path)
  beam_h27_{nx}x{ny}x{nz}.jl     — legacy MORFE2.0 input configuration

and returns a NamedTuple with file paths, free DoF count, and element index
ranges needed for the legacy `.jl` input.

Geometry: 1000 × 10 × 24 mm clamped-clamped beam.
Element type: H27 quadratic Lagrange hexahedron (Gmsh type 12).

Called at the bottom of this file with (nx=40, ny=2, nz=2) to reproduce the
canonical beam_h27.{msh,mphtxt} files used by benchmark_ferrite.jl and the
existing legacy benchmark.
"""

using Gmsh

include(joinpath(@__DIR__, "../../src/FEMUtility/GmshToComsol.jl"))
using .GmshToComsol

const _BEAM_L = 1000.0   # mm
const _BEAM_W = 10.0
const _BEAM_H = 24.0

"""
	generate_beam_mesh(nx, ny, nz; output_dir, legacy_dir, max_order) -> NamedTuple

Generate a clamped-clamped beam mesh with `nx × ny × nz` H27 elements.

Keyword arguments:
- `output_dir`: directory for the `.msh` file (default: this script's directory)
- `legacy_dir`: directory for the `.mphtxt` and `.jl` files (default: legacy_morfe/MORFE2.0/input)
- `max_order`: polynomial degree written into the legacy `.jl` input (default: 5)

Returns `(; msh_path, mphtxt_path, legacy_jl_path, free_dofs,
			dirichlet_q9_indices, h27_range)`.
"""
function generate_beam_mesh(
	nx::Int, ny::Int, nz::Int;
	output_dir::AbstractString = @__DIR__,
	legacy_dir::AbstractString = joinpath(@__DIR__, "../../legacy_morfe/MORFE2.0/input"),
	max_order::Int = 5,
)
	mesh_name = "beam_h27_$(nx)x$(ny)x$(nz)"
	msh_path = joinpath(output_dir, "$(mesh_name).msh")
	mphtxt_path = joinpath(legacy_dir, "$(mesh_name).mphtxt")
	legacy_jl_path = joinpath(legacy_dir, "$(mesh_name).jl")

	# -----------------------------------------------------------------------
	# 1. GMSH geometry and transfinite hex mesh
	# -----------------------------------------------------------------------
	gmsh.initialize()
	gmsh.model.add(mesh_name)
	gmsh.option.setNumber("General.Verbosity", 2)

	v = gmsh.model.occ.addBox(0.0, 0.0, 0.0, _BEAM_L, _BEAM_W, _BEAM_H)
	gmsh.model.occ.synchronize()

	for (_, tag) in gmsh.model.getEntities(1)
		xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
		dx = xmax - xmin
		dy = ymax - ymin
		dz = zmax - zmin
		n = if dx > dy && dx > dz
			nx + 1
		elseif dy > dz
			ny + 1
		else
			nz + 1
		end
		gmsh.model.mesh.setTransfiniteCurve(tag, n)
	end

	dirichlet_tags   = Int[]
	freesurface_tags = Int[]
	for (_, tag) in gmsh.model.getEntities(2)
		xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(2, tag)
		if isapprox(xmax - xmin, 0.0; atol = 1.0)
			push!(dirichlet_tags, tag)
		else
			push!(freesurface_tags, tag)
		end
		gmsh.model.mesh.setTransfiniteSurface(tag)
		gmsh.model.mesh.setRecombine(2, tag)
	end

	gmsh.model.mesh.setTransfiniteVolume(v)

	gmsh.model.addPhysicalGroup(3, [v], -1, "Volume")
	gmsh.model.addPhysicalGroup(2, dirichlet_tags, -1, "Dirichlet")
	gmsh.model.addPhysicalGroup(2, freesurface_tags, -1, "FreeSurface")

	gmsh.model.mesh.generate(3)
	gmsh.model.mesh.setOrder(2)

	n_nodes = length(gmsh.model.mesh.getNodes(-1, -1)[1])

	# -----------------------------------------------------------------------
	# 2. Compute legacy element indices (before finalize clears the model)
	# -----------------------------------------------------------------------
	# All Q9 (quad2, Gmsh type 10) element tags in mphtxt order (sorted by tag).
	all_q9_tags  = sort(Int.(gmsh.model.mesh.getElementsByType(10)[1]))
	all_h27_tags = sort(Int.(gmsh.model.mesh.getElementsByType(12)[1]))
	n_q9         = length(all_q9_tags)
	n_h27        = length(all_h27_tags)

	# Element tags in the "Dirichlet" physical group (x-end faces).
	dirichlet_pg_tag = only(
		tag for (dim, tag) in gmsh.model.getPhysicalGroups(2)
				if gmsh.model.getPhysicalName(2, tag) == "Dirichlet"
	)
	dirichlet_entity_tags = gmsh.model.getEntitiesForPhysicalGroup(2, dirichlet_pg_tag)
	dirichlet_q9_tags = Int[]
	for s in dirichlet_entity_tags
		elem_types, elem_tag_groups, _ = gmsh.model.mesh.getElements(2, s)
		for (etype, etags) in zip(elem_types, elem_tag_groups)
			etype == 10 && append!(dirichlet_q9_tags, Int.(etags))
		end
	end
	sort!(dirichlet_q9_tags)

	# Map Gmsh tag → 1-based position in the sorted Q9 list (= mphtxt index).
	q9_pos = Dict(t => i for (i, t) in enumerate(all_q9_tags))
	dirichlet_q9_indices = sort([q9_pos[t] for t in dirichlet_q9_tags])

	# H27 section in mphtxt comes after all Q9 elements.
	h27_range = (n_q9+1):(n_q9+n_h27)

	# -----------------------------------------------------------------------
	# 3. Export .msh
	# -----------------------------------------------------------------------
	gmsh.write(msh_path)
	println("Wrote: ", msh_path)
	gmsh.finalize()

	# -----------------------------------------------------------------------
	# 4. Cross-export to COMSOL (.mphtxt)
	# -----------------------------------------------------------------------
	mkpath(legacy_dir)
	gmsh_to_comsol(msh_path, mphtxt_path)
	println("Wrote: ", mphtxt_path)

	# -----------------------------------------------------------------------
	# 5. Write legacy MORFE2.0 input .jl file
	# -----------------------------------------------------------------------
	n_constrained = 2 * (2ny + 1) * (2nz + 1)
	free_dofs     = (n_nodes - n_constrained) * 3

	open(legacy_jl_path, "w") do io
		println(io, "# Auto-generated by generate_beam_mesh.jl — do not edit by hand.")
		println(io, "# Mesh: $(mesh_name).mphtxt   FOM = $(free_dofs)")
		println(io)
		println(io, "domains_list = [collect($(h27_range.start):$(h27_range.stop))]")
		println(io, "boundaries_list = [$(dirichlet_q9_indices)]")
		println(io, "constrained_dof = [[1, 1, 1]]")
		println(io, "bc_vals = [[0.0, 0.0, 0.0]]")
		println(io)
		println(io, "materials_list = [\"polysilicon\"]")
		println(io, "density = 2.32e-3")
		println(io, "young_modulus = 160e3")
		println(io, "poisson_ratio = 0.22")
		println(io, "mat = MORFE_newmaterial(\"polysilicon\", density, young_modulus, poisson_ratio)")
		println(io, "materials_dict = Dict(\"polysilicon\" => mat)")
		println(io)
		println(io, "info.α = 0.5369754008568333 / 500.0")
		println(io, "info.β = 0.0")
		println(io, "info.Φ = [1]")
		println(io, "info.neig = 10")
		println(io, "info.Ffreq = 1")
		println(io, "info.Fmodes = [1]")
		println(io, "info.Fmult = 0.5 * [5.0]")
		println(io, "info.omega_mul = 1.0")
		println(io, "info.style = 'c'")
		println(io, "info.max_order = $(max_order)")
		println(io, "info.max_orderNA = $(max_order)")
		println(io, "dirout = compose_name_output_dir(\"$(mesh_name)\", info)")
		println(io, "info.output_dir = dirout")
	end
	println("Wrote: ", legacy_jl_path)

	println()
	println("  $(mesh_name):")
	println("    Nodes         : $(n_nodes)")
	println("    Free DoFs     : $(free_dofs)  (expected: $(75 * (2nx - 1)))")
	println("    Q9 face elems : $(n_q9)")
	println("    H27 vol elems : $(n_h27)")
	println("    Dirichlet Q9  : $(dirichlet_q9_indices)")
	println("    H27 range     : $(h27_range)")

	return (;
		msh_path,
		mphtxt_path,
		legacy_jl_path,
		free_dofs,
		dirichlet_q9_indices,
		h27_range,
	)
end

# -----------------------------------------------------------------------
# Script entry-point: reproduce the canonical beam_h27.{msh,mphtxt} files.
# -----------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
	println("Generating canonical beam_h27 mesh (40×2×2) …")
	canonical_msh    = joinpath(@__DIR__, "beam_h27.msh")
	canonical_mphtxt = joinpath(@__DIR__, "../../legacy_morfe/MORFE2.0/input/beam_h27.mphtxt")

	# Generate via the function (writes beam_h27_40x2x2.* files).
	meta = generate_beam_mesh(40, 2, 2)

	# Also copy to the canonical names expected by benchmark_ferrite.jl and
	# the existing legacy benchmark.
	cp(meta.msh_path, canonical_msh; force = true)
	println("Copied to: ", canonical_msh)
	cp(meta.mphtxt_path, canonical_mphtxt; force = true)
	println("Copied to: ", canonical_mphtxt)

	println()
	println("Free DoFs (expected): ", meta.free_dofs, "  (target ≈ 5952)")
end
