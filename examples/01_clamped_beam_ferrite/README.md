# 01 — Clamped-clamped beam (Ferrite.jl backend)

Structural mechanical problem: clamped-clamped beam (St. Venant-Kirchhoff material,
cubic geometric nonlinearity) using Ferrite.jl as the FEM backend.

**Entry script:** `demo_mechanical_problem.jl`

**FOM size:** ~4977 free DOFs (40×3×1 Hex27 mesh, quadratic Lagrange)

**Mesh:** `examples/02_clamped_beam_gridap/clamped_clamped_beam.msh` (shared with the Gridap example)

**Expected output:** eigenvalues, reduced dynamics coefficients (realified), timing for the cohomological solve.

## How to run

```bash
julia --project=examples/01_clamped_beam_ferrite -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/01_clamped_beam_ferrite/demo_mechanical_problem.jl")'
```

## Notes

The Ferrite backend is loaded automatically via the `MORFEFerriteExt` package extension
when `using Ferrite` is executed. The assembly functions `ferrite_assemble_KM!` and
`ferrite_nonlinearity` are the public entry points defined in `ext/FerriteBackend/`.
