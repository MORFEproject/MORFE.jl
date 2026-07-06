"""
MORFE.jl demo — 08_mems_micromirror : STEP → Gmsh mesh

Reads a neutral CAD file (`mirror_gimbal.step`) exported from the SolidWorks part
`Mirror-Gimbal-v1.2.SLDPRT`, generates a 3-D tetrahedral mesh, tags the anchor
surface(s) as the physical group "clamp", and writes `mirror_gimbal.msh` for
`main.jl`.

Native `.SLDPRT` cannot be read by any open-source tool — first export a STEP from
a CAD tool that opens it (see README.md).  Run in the folder:

    julia --project mesh.jl

Workflow:
  1. Run once and read the printed "Surface tags / bounding boxes" table.
  2. Fill in the `CLAMP` block below with the anchor face selection.
  3. Re-run — `mirror_gimbal.msh` is written with a non-empty "clamp" facetset.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
    Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays", "Gmsh"])
end
Pkg.instantiate()

import Gmsh: gmsh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
const STEP_FILE = joinpath(@__DIR__, "mirror_gimbal.step")
const MSH_FILE  = joinpath(@__DIR__, "mirror_gimbal.msh")

# STEP files are usually in millimetres. MORFE's SVK example convention (see
# example 03) works in micrometres (coords span ~0–1000 for a MEMS device), so
# the material constants E = 170e3 MPa, ρ = 2.33e-3 in main.jl are µm-consistent.
# Set LENGTH_SCALE so the meshed coordinates come out in µm.
#   STEP in mm → µm : 1000.0     STEP already in µm : 1.0
# This choice directly sets the eigenfrequencies — confirm against the real part size.
const LENGTH_SCALE = 1000.0

# Target element size, in the SAME units as the scaled geometry (µm).
const MESH_SIZE_MIN = 2.0
const MESH_SIZE_MAX = 10.0

# ---------------------------------------------------------------------------
gmsh.initialize()
gmsh.model.add("mirror_gimbal")

isfile(STEP_FILE) || error("Missing $STEP_FILE — export a STEP from the .SLDPRT first (see README.md).")
gmsh.model.occ.importShapes(STEP_FILE)
gmsh.model.occ.synchronize()

# Scale geometry into µm.
if LENGTH_SCALE != 1.0
    gmsh.model.occ.dilate(gmsh.model.getEntities(3), 0.0, 0.0, 0.0,
        LENGTH_SCALE, LENGTH_SCALE, LENGTH_SCALE)
    gmsh.model.occ.synchronize()
end

# ---------------------------------------------------------------------------
# Diagnostic: list every surface with its bounding box so the anchor faces can
# be identified. `bbox = (xmin, ymin, zmin, xmax, ymax, zmax)`.
# ---------------------------------------------------------------------------
surfaces = gmsh.model.getEntities(2)
println("\nSurface tags / bounding boxes (units: µm after scaling):")
println("  tag |      xmin      ymin      zmin  →      xmax      ymax      zmax")
for (_, tag) in surfaces
    b = gmsh.model.getBoundingBox(2, tag)
    println(rpad("  $tag", 6), " | ",
        join((round.(b; digits = 2)), "  "))
end

# Model-wide bounding box (handy for the clamp selector below).
gbox = gmsh.model.getBoundingBox(-1, -1)
(xmin, ymin, zmin, xmax, ymax, zmax) = gbox
println("\nModel bounding box: x[$xmin, $xmax]  y[$ymin, $ymax]  z[$zmin, $zmax]\n")

# ---------------------------------------------------------------------------
# Helper: select surface tags whose bounding box lies on a coordinate plane.
#   axis :x/:y/:z, value the plane coordinate, tol the matching tolerance.
# ---------------------------------------------------------------------------
function select_by_plane(axis::Symbol, value::Real; tol = 1e-6)
    idx = axis === :x ? (1, 4) : axis === :y ? (2, 5) : (3, 6)
    tags = Int[]
    for (_, tag) in gmsh.model.getEntities(2)
        b = gmsh.model.getBoundingBox(2, tag)
        if abs(b[idx[1]] - value) < tol && abs(b[idx[2]] - value) < tol
            push!(tags, tag)
        end
    end
    return tags
end

# ===========================================================================
# CLAMP — anchor faces (EDIT THIS after the first run)
#
# Pick ONE approach:
#   (a) explicit tags read from the table above:
#         const CLAMP = [7, 12]
#   (b) all faces on the substrate plane (e.g. z = zmin, tol in µm):
#         const CLAMP = select_by_plane(:z, zmin; tol = 1e-3)
# ===========================================================================
const CLAMP = select_by_plane(:z, zmin; tol = 1e-3)   # TODO: confirm this is the anchor plane

isempty(CLAMP) && error("CLAMP is empty — no anchor faces selected. Edit the CLAMP block in mesh.jl.")

# ---------------------------------------------------------------------------
# Physical groups: "domain" (volume, so cells are written) and "clamp" (faces).
# ---------------------------------------------------------------------------
volumes = [tag for (_, tag) in gmsh.model.getEntities(3)]
gmsh.model.addPhysicalGroup(3, volumes, -1, "domain")
gmsh.model.addPhysicalGroup(2, CLAMP, -1, "clamp")

# Linear tets — robust in FerriteGmsh.togrid; main.jl adds the quadratic
# displacement field via fe_order = 2 (field order ⟂ geometric order).
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE_MIN)
gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE_MAX)

gmsh.model.mesh.generate(3)
gmsh.write(MSH_FILE)
println("Wrote $MSH_FILE  (clamp faces: $CLAMP)")

gmsh.finalize()
