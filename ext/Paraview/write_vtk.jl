# Implementation of MORFE.ParaviewExport functions.
# Included by ext/MORFEWriteVTKExt.jl; requires Ferrite and WriteVTK in scope.

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Ferrite Lagrange{RefPrism,2} stores nodes in this order (see reference_coordinates):
#   1-6  : corners  (3 bottom, 3 top)
#   7    : mid(0-1) bottom        10: mid(1-2) bottom    8: mid(0-2) bottom
#   9    : mid(0-3) vertical      11: mid(1-4) vertical  12: mid(2-5) vertical
#   13   : mid(3-4) top           15: mid(4-5) top        14: mid(3-5) top
#   16-18: quad-face centres
#
# VTK_BIQUADRATIC_QUADRATIC_WEDGE (type 32) expects:
#   0-5  : corners (same mapping)
#   6    : mid(0-1) bottom   7: mid(1-2) bottom   8: mid(2-0) bottom
#   9    : mid(3-4) top     10: mid(4-5) top      11: mid(5-3) top
#   12   : mid(0-3) vert    13: mid(1-4) vert     14: mid(2-5) vert
#   15-17: quad-face centres in same order
#
# VTK position p needs Ferrite index PERM_FERRITE_TO_VTK[p]:
const PERM_FERRITE_TO_VTK = [1, 2, 3, 4, 5, 6,   # corners — identical
                              7, 10, 8,            # bottom edge mids: VTK 7,8,9
                              13, 15, 14,          # top    edge mids: VTK 10,11,12
                              9, 11, 12,           # vert   edge mids: VTK 13,14,15
                              16, 18, 17]          # face centres:     VTK 16,17,18

"""
    _vtk_geometry(grid) -> (points, cells)

Extract a 3×n_nodes coordinate matrix and a `MeshCell` array from a
Ferrite 3-D grid.  Dispatches on the number of nodes per cell:
  - 18 nodes → VTK_BIQUADRATIC_QUADRATIC_WEDGE (18-node quadratic prism, type 32)
               Applies PERM_FERRITE_TO_VTK to reorder Ferrite nodes to VTK convention.
  - 6  nodes → VTK_WEDGE (linear wedge, type 13) — no reordering needed.
"""
function _vtk_geometry(grid::Ferrite.Grid{3})
    points = reduce(hcat, [collect(n.x) for n in grid.nodes])
    cells  = map(grid.cells) do cell
        nn = length(cell.nodes)
        if nn == 18
            MeshCell(VTKCellTypes.VTK_BIQUADRATIC_QUADRATIC_WEDGE,
                     [cell.nodes[i] for i in PERM_FERRITE_TO_VTK])
        elseif nn == 6
            MeshCell(VTKCellTypes.VTK_WEDGE, collect(cell.nodes))
        else
            error("_vtk_geometry: unsupported cell size $nn (expected 6 or 18).")
        end
    end
    return points, cells
end

"""
    _scatter_to_nodes(u_full, dh) -> 3 × n_nodes matrix

Scatter a full DOF vector (length `ndofs(dh)`) to a component-major
nodal array, assuming a single 3-component `:u` field with node-major
DOF ordering (standard for `Lagrange{RefX, p}()^3`).
"""
function _scatter_to_nodes(u_full::AbstractVector, dh::Ferrite.DofHandler)
    n_nodes = Ferrite.getnnodes(dh.grid)
    out     = zeros(eltype(u_full), 3, n_nodes)
    visited = falses(n_nodes)
    for cell in CellIterator(dh)
        gdofs  = celldofs(cell)
        cnodes = cell.nodes
        for (i, node) in enumerate(cnodes)
            visited[node] && continue
            visited[node] = true
            for c in 1:3
                out[c, node] = u_full[gdofs[(i - 1) * 3 + c]]
            end
        end
    end
    return out
end

"""
    _build_node_dof_map(dh) -> 3 × n_nodes Int matrix

For each node and each spatial component (1=x, 2=y, 3=z), return the
global DOF index (1-indexed, same as the full system row/column number).
Assumes a single 3-component `:u` field with node-major DOF ordering.
"""
function _build_node_dof_map(dh::Ferrite.DofHandler)
    n_nodes = Ferrite.getnnodes(dh.grid)
    out     = zeros(Int, 3, n_nodes)
    visited = falses(n_nodes)
    for cell in CellIterator(dh)
        gdofs  = celldofs(cell)
        cnodes = cell.nodes
        for (i, node) in enumerate(cnodes)
            visited[node] && continue
            visited[node] = true
            for c in 1:3
                out[c, node] = gdofs[(i - 1) * 3 + c]
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# write_paraview_mesh
# ---------------------------------------------------------------------------

"""
    write_paraview_mesh(filename, grid; dh=nothing, prescribed_dofs=nothing)

Write the undeformed mesh to `filename.vtu`.

Optional kwargs — when both are supplied, the following point-data arrays are
added so that clicking any node in Paraview shows its metadata:

  node_id  (Int32) — 1-indexed mesh node number
  x, y, z  (Float64) — node coordinates as named scalar fields
  dof_x, dof_y, dof_z  (Int32) — free DOF indices (1-indexed, same as the
                                   reduced K/M vectors); -1 for constrained

  dh              — Ferrite DofHandler
  prescribed_dofs — ch.prescribed_dofs (vector of constrained global DOF indices)
"""
function MORFE.ParaviewExport.write_paraview_mesh(
    filename::AbstractString,
    grid::Ferrite.Grid{3};
    dh::Union{Ferrite.DofHandler, Nothing} = nothing,
    prescribed_dofs = nothing,
)
    mkpath(dirname(filename))
    points, cells = _vtk_geometry(grid)
    n_nodes = Ferrite.getnnodes(grid)

    if isnothing(dh)
        vtk_grid(filename, points, cells) do _
        end
    else
        pset = isnothing(prescribed_dofs) ? Set{Int}() : Set{Int}(prescribed_dofs)
        dof_map = _build_node_dof_map(dh)

        # Build free-DOF index (same convention as the reduced K/M vectors and
        # node_dof_table.txt: 1…n_free, -1 for constrained directions).
        n_dofs = Ferrite.ndofs(dh)
        free = sort(setdiff(1:n_dofs, pset))
        free_to_local = Dict(d => i for (i, d) in enumerate(free))

        node_ids = Int32.(1:n_nodes)
        xs = Float64[grid.nodes[n].x[1] for n in 1:n_nodes]
        ys = Float64[grid.nodes[n].x[2] for n in 1:n_nodes]
        zs = Float64[grid.nodes[n].x[3] for n in 1:n_nodes]
        dof_x = Int32[get(free_to_local, dof_map[1, n], -1) for n in 1:n_nodes]
        dof_y = Int32[get(free_to_local, dof_map[2, n], -1) for n in 1:n_nodes]
        dof_z = Int32[get(free_to_local, dof_map[3, n], -1) for n in 1:n_nodes]

        vtk_grid(filename, points, cells) do vtk
            vtk_point_data(vtk, node_ids, "node_id")
            vtk_point_data(vtk, xs,       "x")
            vtk_point_data(vtk, ys,       "y")
            vtk_point_data(vtk, zs,       "z")
            vtk_point_data(vtk, dof_x,    "dof_x")
            vtk_point_data(vtk, dof_y,    "dof_y")
            vtk_point_data(vtk, dof_z,    "dof_z")
        end
    end
    println("  Mesh → $(filename).vtu")
end

# ---------------------------------------------------------------------------
# write_paraview_modes
# ---------------------------------------------------------------------------

function MORFE.ParaviewExport.write_paraview_modes(
    outdir::AbstractString,
    grid::Ferrite.Grid{3},
    dh::Ferrite.DofHandler,
    eigenvalues::AbstractVector{<:Complex},
    Y::AbstractArray{<:Complex, 3},
    free::AbstractVector{Int};
    n_modes::Int = 10,
    prefix::AbstractString = "mode",
)
    mkpath(outdir)
    points, cells = _vtk_geometry(grid)
    n_dofs = Ferrite.ndofs(dh)
    n_eig  = size(Y, 3)
    n_out  = min(n_modes, n_eig)

    pvd = paraview_collection(joinpath(outdir, "modes"))

    println("\n  Mode shapes → $(outdir)/")
    println("  " * "-"^58)
    @printf("  %-6s  %-22s  %s\n", "Mode", "Freq (Hz)", "File")
    println("  " * "-"^58)

    for k in 1:n_out
        fname = joinpath(outdir, @sprintf("%s_%02d", prefix, k))

        u_full_re = zeros(Float64, n_dofs)
        u_full_im = zeros(Float64, n_dofs)
        u_full_re[free] .= real.(Y[:, 1, k])
        u_full_im[free] .= imag.(Y[:, 1, k])

        u_re = _scatter_to_nodes(u_full_re, dh)
        u_im = _scatter_to_nodes(u_full_im, dh)

        # pvd[time] = vtk must be inside the do-block (WriteVTK API requirement)
        vtk_grid(fname, points, cells) do vtk
            vtk_point_data(vtk, u_re, "Re_u")
            vtk_point_data(vtk, u_im, "Im_u")
            pvd[Float64(k)] = vtk
        end

        freq_hz = abs(eigenvalues[k]) / (2π)
        @printf("  %-6d  %-22.6g  %s.vtu\n", k, freq_hz, basename(fname))
    end

    println("  " * "-"^58)
    vtk_save(pvd)
    println("  Collection → $(joinpath(outdir, "modes.pvd"))")
end

# ---------------------------------------------------------------------------
# write_paraview_deformation
# ---------------------------------------------------------------------------

function MORFE.ParaviewExport.write_paraview_deformation(
    outdir::AbstractString,
    grid::Ferrite.Grid{3},
    dh::Ferrite.DofHandler,
    W::MORFE.ParametrisationMethod.Parametrisation,
    free::AbstractVector{Int},
    r_amplitude::Real;
    theta::Real = 0.0,
    label::AbstractString = "deformation",
)
    C    = MORFE.ParametrisationMethod.coefficients(W)   # (FOM, ORD, L)
    mset = MORFE.ParametrisationMethod.multiindex_set(W)
    W_disp = MORFE.Polynomials.DensePolynomial(@view(C[:, 1, :]), mset)

    z1 = r_amplitude * cis(float(theta))
    sv = [z1, conj(z1)]

    u_free     = MORFE.Polynomials.evaluate(W_disp, sv)   # FOM ComplexF64 (BLAS gemv)
    A_lin      = MORFE.Polynomials.linear_matrix_of_polynomial(W_disp)  # FOM × 2
    u_free_lin = A_lin * sv
    u_free_nl  = u_free .- u_free_lin

    n_dofs = Ferrite.ndofs(dh)
    function to_vtk(u_cplx)
        buf = zeros(Float64, n_dofs)
        buf[free] .= real.(u_cplx)
        return _scatter_to_nodes(buf, dh)   # (3, n_nodes)
    end

    mkpath(outdir)
    points, cells = _vtk_geometry(grid)
    fname = joinpath(outdir, label)
    vtk_grid(fname, points, cells) do vtk
        vtk_point_data(vtk, to_vtk(u_free),    "u_total")
        vtk_point_data(vtk, to_vtk(u_free_nl), "u_nonlinear")
    end
    println("  Deformation → $(fname).vtu")
end

# ---------------------------------------------------------------------------
# write_paraview_manifold
# ---------------------------------------------------------------------------

function MORFE.ParaviewExport.write_paraview_manifold(
    outdir::AbstractString,
    grid::Ferrite.Grid{3},
    dh::Ferrite.DofHandler,
    W::MORFE.ParametrisationMethod.Parametrisation,
    free::AbstractVector{Int};
    alpha_indices = nothing,
)
    mkpath(outdir)
    points, cells = _vtk_geometry(grid)
    n_dofs = Ferrite.ndofs(dh)

    mset    = MORFE.ParametrisationMethod.multiindex_set(W)
    C       = MORFE.ParametrisationMethod.coefficients(W)   # FOM × ORD × L
    exps    = mset.exponents
    n_mono  = length(exps)
    indices = isnothing(alpha_indices) ? (1:n_mono) : alpha_indices

    pvd = paraview_collection(joinpath(outdir, "manifold"))

    println("\n  Manifold slices → $(outdir)/")

    for m in indices
        alpha = exps[m]
        label = join(alpha, "_")
        fname = joinpath(outdir, "manifold_$(label)")

        v = C[:, 1, m]   # position slice (first ORD level)

        u_full_re = zeros(Float64, n_dofs)
        u_full_im = zeros(Float64, n_dofs)
        u_full_re[free] .= real.(v)
        u_full_im[free] .= imag.(v)

        u_re = _scatter_to_nodes(u_full_re, dh)
        u_im = _scatter_to_nodes(u_full_im, dh)

        vtk_grid(fname, points, cells) do vtk
            vtk_point_data(vtk, u_re, "Re_W_alpha")
            vtk_point_data(vtk, u_im, "Im_W_alpha")
            pvd[Float64(m)] = vtk
        end
    end

    vtk_save(pvd)
    println("  Collection → $(joinpath(outdir, "manifold.pvd"))")
end
