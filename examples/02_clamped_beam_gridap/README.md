# 02 — Clamped-clamped beam (Gridap.jl backend)

Same clamped-clamped beam as example 01, assembled with Gridap.jl as the FEM backend.
Use this example to compare results or to use Gridap as the reference implementation
for building a new FEM backend.

**Entry script:** `main.jl`

**Mesh:** `clamped_clamped_beam.msh` (40×3×1 Hex8, quadratic order-2 Lagrange; also used by example 01)

**Expected output:** eigenvalues, reduced dynamics coefficients.

## How to run

```bash
julia --project=examples/02_clamped_beam_gridap -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/02_clamped_beam_gridap/main.jl")'
```
