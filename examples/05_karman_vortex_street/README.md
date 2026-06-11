# 05 — Kármán vortex street

## Model

Incompressible Navier-Stokes flow past a cylinder (2D, Ferrite P2/P1 Taylor-Hood).
The steady base flow is computed first; then DPIM parametrises the unstable SSM
associated with the Hopf bifurcation that gives rise to the Kármán vortex street.
Parameters are set in `config.jl` (default: Re₀ = 49.03, order 5).

## How to run

```bash
julia --project=examples/05_karman_vortex_street -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/05_karman_vortex_street/main.jl")'
```

## Expected outputs

```text
results/
  Re49.03_ord5/
    summary.log              — verbose tee'd run log
    summary.txt              — structured key:value summary with environment info
    data/
      W.jls                  — parametrisation (serialised)
      R.jls                  — reduced dynamics (serialised)
      reduced_dynamics.txt   — realified Stuart-Landau coefficients (human-readable)
      vtk_data.jls           — mesh + mode data bundle for ParaView export
    figures/                 — post-processing figures (from postprocess scripts)
```

## Reference results

Curated reference outputs live in `results/reference/Re49.03_ord5/` (tracked in git).

## Historical results

A full archived run is restored untracked from the archive repo into:

```text
results/Re49.03_ord5/
  W.jls, R.jls, vtk_data.jls    — parametrisation and VTK bundle
  reduced_dynamics.txt           — Stuart-Landau coefficients
  summary.log                    — run log
  paraview/                      — VTU files for mode and vortex visualisation
```

These are not version-controlled here; the durable copy is `MORFE_results_archive`.

## Approximate runtime

~30 minutes on a modern workstation (order-5 parametrisation, sparse Ferrite assembly,
~25k-DOF non-symmetric system). VTK export adds a few minutes.

## Files

| File | Purpose |
| ---- | ------- |
| `main.jl` | Top-level driver |
| `config.jl` | Problem parameters (Re, mesh, ROM size) |
| `cylinder_flow.msh` | Input mesh (Turek-Schäfer cylinder geometry) |
| `fem/mesh.jl` | Mesh loading and DOF setup |
| `fem/fem_setup.jl` | FEM spaces and boundary conditions |
| `fem/linear_operators.jl` | Linearised Navier-Stokes operators |
| `fem/fluid_maps.jl` | Nonlinear multilinear maps |
| `solver/steady_state.jl` | Newton solve for the base flow |
| `solver/eigensolver.jl` | Custom eigensolver for the non-symmetric problem |
| `backbone/backbone_env/` | Julia environment for post-processing scripts |
