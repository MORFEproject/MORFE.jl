"""
    mesh.jl — Turek–Schäfer 2D cylinder-flow mesh via Gmsh

Geometry (DFG benchmark):
  Channel  : [0, 2.2] × [0, 0.41]
  Cylinder : centre (0.2, 0.2), diameter D = 0.1 (radius r = 0.05)

Boundary physical groups (→ FacetSets in FerriteGmsh):
  "Inlet"    — left edge  (x = 0)
  "Outlet"   — right edge (x = 2.2)
  "Walls"    — top + bottom edges
  "Cylinder" — circle arcs

Cell physical group:
  "Domain"   — the fluid 2-D region

Mesh: linear triangles (Ferrite Triangle cells).  P2/P1 Taylor-Hood DOFs are
added by Ferrite's DofHandler — no need for quadratic triangles in the geometry.

Refinement strategy:
  h = h_cyl  (≈ 0.005) within distance r of the cylinder
  h = h_wake (≈ 0.015) in the wake corridor [cx, L] × [cy±3r]
  h = h_bulk (≈ 0.04)  elsewhere

With the default h settings in config.jl the P2/P1 Ferrite setup yields
n_free ≈ 57 860 — finer than the paper's 17 973-DOF mesh.
"""

using Gmsh

const _CHANNEL_L = 2.2
const _CHANNEL_H = 0.41
const _CYL_CX    = 0.2
const _CYL_CY    = 0.2
const _CYL_R     = 0.05
const _CYL_D     = 2.0 * _CYL_R   # cylinder diameter = 0.1 m (reference length)

"""
    generate_mesh(; meshfile = "cylinder_flow.msh",
                    h_cyl  = 0.005,
                    h_wake = 0.015,
                    h_bulk = 0.04) -> meshfile

Generate the Turek–Schäfer mesh and write it to `meshfile`.
Returns the path so callers can `togrid(generate_mesh())`.
"""
function generate_mesh(;
    meshfile::String = joinpath(@__DIR__, "cylinder_flow.msh"),
    h_cyl::Float64  = 0.005,
    h_wake::Float64 = 0.015,
    h_bulk::Float64 = 0.04,
)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)   # suppress Gmsh console spam
    gmsh.model.add("TurekSchaefer")

    L, H    = _CHANNEL_L, _CHANNEL_H
    cx, cy  = _CYL_CX, _CYL_CY
    r       = _CYL_R

    # ── Channel corners ──────────────────────────────────────────────────
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h_bulk)   # bottom-left
    p2 = gmsh.model.geo.addPoint(L,   0.0, 0.0, h_bulk)   # bottom-right
    p3 = gmsh.model.geo.addPoint(L,   H,   0.0, h_bulk)   # top-right
    p4 = gmsh.model.geo.addPoint(0.0, H,   0.0, h_bulk)   # top-left

    l_bot    = gmsh.model.geo.addLine(p1, p2)   # bottom wall
    l_outlet = gmsh.model.geo.addLine(p2, p3)   # outlet (right)
    l_top    = gmsh.model.geo.addLine(p3, p4)   # top wall
    l_inlet  = gmsh.model.geo.addLine(p4, p1)   # inlet (left)

    channel_loop = gmsh.model.geo.addCurveLoop([l_bot, l_outlet, l_top, l_inlet])

    # ── Cylinder: 4 circle arcs through cardinal points ──────────────────
    # Centre point (only used as arc reference, NOT meshed)
    pc = gmsh.model.geo.addPoint(cx,     cy,     0.0, h_cyl)
    pe = gmsh.model.geo.addPoint(cx + r, cy,     0.0, h_cyl)   # east
    pn = gmsh.model.geo.addPoint(cx,     cy + r, 0.0, h_cyl)   # north
    pw = gmsh.model.geo.addPoint(cx - r, cy,     0.0, h_cyl)   # west
    ps = gmsh.model.geo.addPoint(cx,     cy - r, 0.0, h_cyl)   # south

    arc_ne = gmsh.model.geo.addCircleArc(pe, pc, pn)
    arc_nw = gmsh.model.geo.addCircleArc(pn, pc, pw)
    arc_ws = gmsh.model.geo.addCircleArc(pw, pc, ps)
    arc_se = gmsh.model.geo.addCircleArc(ps, pc, pe)

    # Curve loop must be consistently oriented (counter-clockwise viewed from +z).
    # The channel loop is CCW; the cylinder hole loop must be CW to define a "hole".
    cyl_loop = gmsh.model.geo.addCurveLoop([arc_ne, arc_nw, arc_ws, arc_se])

    # Domain: channel minus cylinder hole
    domain_surf = gmsh.model.geo.addPlaneSurface([channel_loop, cyl_loop])

    gmsh.model.geo.synchronize()

    # ── Mesh size fields ─────────────────────────────────────────────────
    # Field 1: distance from cylinder arcs
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [arc_ne, arc_nw, arc_ws, arc_se])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 300)

    # Field 2: threshold based on distance → graded h from h_cyl to h_bulk
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin",  h_cyl)
    gmsh.model.mesh.field.setNumber(2, "SizeMax",  h_bulk)
    gmsh.model.mesh.field.setNumber(2, "DistMin",  r)
    gmsh.model.mesh.field.setNumber(2, "DistMax",  8.0 * r)

    # Field 3: box covering the wake corridor
    # The wake extends downstream from the cylinder; refine to h_wake there.
    gmsh.model.mesh.field.add("Box", 3)
    gmsh.model.mesh.field.setNumber(3, "VIn",   h_wake)
    gmsh.model.mesh.field.setNumber(3, "VOut",  h_bulk)
    gmsh.model.mesh.field.setNumber(3, "XMin",  cx - r)
    gmsh.model.mesh.field.setNumber(3, "XMax",  L)
    gmsh.model.mesh.field.setNumber(3, "YMin",  cy - 4.0 * r)
    gmsh.model.mesh.field.setNumber(3, "YMax",  cy + 4.0 * r)

    # Field 4: take the minimum of fields 2 and 3
    gmsh.model.mesh.field.add("Min", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
    gmsh.model.mesh.field.setAsBackgroundMesh(4)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # ── Physical groups (names become FacetSet/CellSet names in FerriteGmsh) ──
    gmsh.model.addPhysicalGroup(1, [l_inlet],             -1, "Inlet")
    gmsh.model.addPhysicalGroup(1, [l_outlet],            -1, "Outlet")
    gmsh.model.addPhysicalGroup(1, [l_bot, l_top],        -1, "Walls")
    gmsh.model.addPhysicalGroup(1, [arc_ne, arc_nw, arc_ws, arc_se], -1, "Cylinder")
    gmsh.model.addPhysicalGroup(2, [domain_surf],         -1, "Domain")

    # ── Mesh generation ──────────────────────────────────────────────────
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")  # improve triangle quality

    gmsh.write(meshfile)
    n_tri = length(gmsh.model.mesh.getElementsByType(2)[2]) ÷ 3   # 3 nodes/triangle
    @info "Mesh written to $meshfile ($n_tri triangles)"

    gmsh.finalize()
    return meshfile
end
