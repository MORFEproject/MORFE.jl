# 05 — Kármán vortex street

Incompressible Navier-Stokes flow past a cylinder. The steady base flow is computed
first; then DPIM is applied to the linearised (non-symmetric) first-order FOM to
parametrise the unstable manifold associated with the Hopf bifurcation that gives
rise to the Kármán vortex street.

**Entry script:** `main.jl`

**Mesh:** `cylinder_flow.msh` (GMSH; 2D cylinder-wake geometry)

**Expected output:** base-flow solution, eigenvalues of the linearised flow,
reduced dynamics R on the SSM, reduced_dynamics.txt summary.

## How to run

```bash
julia --project=examples/05_karman_vortex_street -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/05_karman_vortex_street/main.jl")'
```

## Files

| File | Purpose |
|------|---------|
| `main.jl` | Top-level driver |
| `config.jl` | Problem parameters (Re, mesh, ROM size) |
| `mesh.jl` | Mesh loading and DOF setup |
| `fem_setup.jl` | FEM spaces and boundary conditions |
| `linear_operators.jl` | Linearised Navier-Stokes operators |
| `steady_state.jl` | Newton solve for the base flow |
| `fluid_maps.jl` | Nonlinear multilinear maps |
| `eigensolver.jl` | Custom eigensolver for the non-symmetric problem |
