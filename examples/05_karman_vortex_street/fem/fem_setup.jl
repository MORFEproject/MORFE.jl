"""
    fem_setup.jl — Ferrite P2/P1 Taylor-Hood setup for the cylinder-flow mesh.

Returns a NamedTuple with:
  grid, dh, ch_full, ch_hom, cv_vel, cv_pres,
  free, free_to_local, n_free, dof_range_u, dof_range_p, n_vel_dofs_per_cell

Two ConstraintHandlers over the SAME prescribed-DOF set (different inlet values):
  ch_full  — inhomogeneous Poiseuille inlet + no-slip walls + cylinder;
             used for the steady-state Newton solve
  ch_hom   — homogeneous inlet + no-slip walls + cylinder;
             used for the linearised (perturbation / DPIM) problem.
             The inlet perturbation is frozen to zero (u = u₀ imposed → u' = 0),
             matching the base-flow BC set and the reference DPIM2D_NS code.

Consequently 'free' == 'free_dpim' as index sets:
  free      = setdiff(all, ch_full) — excludes inlet + walls + cylinder
  free_dpim = setdiff(all, ch_hom)  — excludes inlet + walls + cylinder
"""

using Ferrite
using FerriteGmsh

# Mean inflow velocity (Ū = 1 → parabolic max = 1.5)
const U_MEAN = 1.0
const U_MAX  = 1.5 * U_MEAN  # max of Poiseuille profile

"""
    setup_fem(meshfile) -> NamedTuple

Load the mesh at `meshfile` and build all Ferrite FEM objects for the
P2/P1 Taylor-Hood cylinder-flow problem.
"""
function setup_fem(meshfile::String)
    grid = togrid(meshfile)
    @info "Grid: $(getncells(grid)) cells, $(getnnodes(grid)) nodes"

    # ── Interpolations ────────────────────────────────────────────────────
    # Geometric mapping: linear (mesh has 3-node triangles)
    ip_geo  = Lagrange{RefTriangle, 1}()

    # Velocity: P2 vector (6 scalar nodes × 2 components = 12 DOFs/cell)
    ip_vel  = Lagrange{RefTriangle, 2}()^2

    # Pressure: P1 scalar (3 corner nodes = 3 DOFs/cell)
    ip_pres = Lagrange{RefTriangle, 1}()

    # ── Quadrature ────────────────────────────────────────────────────────
    # order=4: integrates polynomials of degree ≤ 4 exactly.
    # This covers ∇P2·∇P2 (degree 2) and P2·P2 mass (degree 4).
    # For the nonlinear convection P2·∇P2·P2 (degree 5), minor quadrature
    # error is acceptable — the spatial truncation error dominates anyway.
    qr = QuadratureRule{RefTriangle}(4)

    # Sub-parametric CellValues: P2/P1 fields on linear-triangle geometry
    cv_vel  = CellValues(qr, ip_vel,  ip_geo)
    cv_pres = CellValues(qr, ip_pres, ip_geo)

    # ── DofHandler: :u first, :p second ──────────────────────────────────
    dh = DofHandler(grid)
    add!(dh, :u, ip_vel)
    add!(dh, :p, ip_pres)
    close!(dh)

    dof_range_u = dof_range(dh, :u)   # local DOF indices for :u in a cell
    dof_range_p = dof_range(dh, :p)   # local DOF indices for :p in a cell
    n_vel_dofs_per_cell  = length(dof_range_u)
    n_pres_dofs_per_cell = length(dof_range_p)

    @info "DOFs: $(ndofs(dh)) total"
    @info "  Velocity per cell : $n_vel_dofs_per_cell  (range $dof_range_u)"
    @info "  Pressure per cell : $n_pres_dofs_per_cell (range $dof_range_p)"

    H = _CHANNEL_H

    # ── Inhomogeneous BCs (for steady-state Newton solve) ─────────────────
    ch_full = ConstraintHandler(dh)
    # Inlet: Poiseuille profile [u_x(y), 0] as a Vec{2} (no components arg → all DOFs)
    add!(ch_full, Dirichlet(:u, getfacetset(grid, "Inlet"),
        (x, _) -> Vec{2}((U_MAX * 4.0 * x[2] * (H - x[2]) / H^2, 0.0))))
    # No-slip walls and cylinder
    add!(ch_full, Dirichlet(:u, getfacetset(grid, "Walls"),
        (x, _) -> Vec{2}((0.0, 0.0))))
    add!(ch_full, Dirichlet(:u, getfacetset(grid, "Cylinder"),
        (x, _) -> Vec{2}((0.0, 0.0))))
    close!(ch_full)
    update!(ch_full, 0.0)

    # ── Homogeneous BCs (for perturbation / DPIM eigenproblem) ───────────
    # Inlet, walls and cylinder — all where the base flow has a Dirichlet BC.
    # The inflow profile is imposed (u = u₀), so the perturbation must vanish
    # there (u' = 0); leaving the inlet free would inject spurious inlet velocity
    # into the modes and the convective quadratic. Matches the reference code.
    ch_hom = ConstraintHandler(dh)
    add!(ch_hom, Dirichlet(:u, getfacetset(grid, "Inlet"),
        (x, _) -> Vec{2}((0.0, 0.0))))
    add!(ch_hom, Dirichlet(:u, getfacetset(grid, "Walls"),
        (x, _) -> Vec{2}((0.0, 0.0))))
    add!(ch_hom, Dirichlet(:u, getfacetset(grid, "Cylinder"),
        (x, _) -> Vec{2}((0.0, 0.0))))
    close!(ch_hom)
    update!(ch_hom, 0.0)

    # ── Free DOFs ─────────────────────────────────────────────────────────
    # free      : for steady-state Newton solve (inlet prescribed to Poiseuille)
    # free_dpim : for DPIM operators (inlet frozen → u' = 0 for perturbation)
    # Both exclude inlet + walls + cylinder, so they coincide as index sets.
    free      = sort(setdiff(1:ndofs(dh), ch_full.prescribed_dofs))
    free_to_local = Dict{Int, Int}(d => i for (i, d) in enumerate(free))
    n_free    = length(free)

    free_dpim = sort(setdiff(1:ndofs(dh), ch_hom.prescribed_dofs))
    free_to_local_dpim = Dict{Int, Int}(d => i for (i, d) in enumerate(free_dpim))
    n_free_dpim = length(free_dpim)

    @info "  Free DOFs (steady state) : $n_free"
    @info "  Free DOFs (DPIM)         : $n_free_dpim  (inlet frozen)"

    return (;
        grid, dh, ch_full, ch_hom, cv_vel, cv_pres, ip_vel, ip_pres, qr,
        free, free_to_local, n_free,
        free_dpim, free_to_local_dpim, n_free_dpim,
        dof_range_u, dof_range_p, n_vel_dofs_per_cell,
    )
end
