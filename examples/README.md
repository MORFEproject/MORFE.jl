# Examples

Self-contained, runnable examples demonstrating the MORFE.jl pipeline.

## Structure

| Folder | Model | Demonstrates | Approx. runtime |
|--------|-------|--------------|-----------------|
| `01_clamped_beam_ferrite/` | Clamped-clamped beam (St. Venant-Kirchhoff) | Full DPIM pipeline with Ferrite.jl FEM backend via `MORFEFerriteExt` | ~5 min |
| `02_clamped_beam_gridap/` | Same beam | Full DPIM pipeline with Gridap.jl FEM backend | ~5 min |
| `03_arch_comsol_wedge/` | Arch wedge (COMSOL mesh import) | COMSOL `.mphtxt` → MORFE pipeline | ~10 min |
| `04_parametric_clamped_beam/` | Clamped beam with axial-stretch parameter θ | Parametric ROM in (z₁, z₂, θ) with N_EXT=1 | ~10 min |
| `05_karman_vortex_street/` | Cylinder wake flow (Kármán vortex street) | Fluid DPIM with Ferrite.jl, non-symmetric FOM | ~30 min |
| `mesh_import/` | Test meshes | Abaqus/COMSOL → GMSH format conversion | seconds |
| `internals/` | Synthetic models | Low-level API: polynomials, multiindices, parametrisation method | seconds–1 min |

## How to run an example

From the repository root:

```julia
using Pkg
Pkg.activate("examples/01_clamped_beam_ferrite")
Pkg.develop(path=".")        # use local MORFE
Pkg.instantiate()
include("examples/01_clamped_beam_ferrite/demo_mechanical_problem.jl")
```

Or equivalently from the shell:

```bash
julia --project=examples/01_clamped_beam_ferrite -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/01_clamped_beam_ferrite/demo_mechanical_problem.jl")'
```

Each example has its own `Project.toml`. The MORFE package is developed in-place
(`Pkg.develop`) so the local source is always used.

## Advanced users

- `examples/internals/` contains low-level API demos (polynomial algebra, multiindex
  factorisations, eigensolver, invariance equation) that are fast to run and useful
  when debugging or extending MORFE internals.
- `ext/FerriteBackend/` contains the Ferrite FEM backend implementation loaded
  automatically via the `MORFEFerriteExt` package extension when `using Ferrite`.
