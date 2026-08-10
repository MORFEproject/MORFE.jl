# Examples

Self-contained, runnable examples demonstrating the MORFE.jl pipeline.
Each example manages its own Julia environment (`Project.toml`) and writes
all outputs under a `results/` subfolder that is git-ignored (except for
`results/reference/` which is tracked as the blessed reference output).

---

## Example contract

Every runnable example satisfies:

1. **Entry point** is `main.jl` at the example root. Running it end-to-end
   produces files on disk — a run that only prints to the terminal is non-compliant.
2. **Outputs** go under `results/` (never the CWD):
   - *Single-run examples* (02): `results/data/` (W.jls, R.jls,
     R_coefficients.csv), `results/figures/`, `results/summary.txt`.
   - *Config-driven examples*: `results/<run_name>/data/`,
     `figures/`, `summary.txt`; `<run_name>` is derived from the config.
3. **`summary.txt`** contains at minimum: model description, master modes,
   eigenfrequencies, parametrisation order, wall-clock time, Julia version,
   MORFE git commit, timestamp.
4. **Reference results** live in `results/reference/` — small, curated, tracked
   in git. Everything else under `results/` is git-ignored.
5. **`validate.jl`** compares a fresh run's `R_coefficients.csv` against the
   reference. Run after completing a fresh `main.jl` run.

---

## Example table

| Folder | Model | Demonstrates | Status | Approx. runtime |
| ------ | ----- | ------------ | ------ | --------------- |
| `02_clamped_beam_gridap/` | Clamped-clamped beam (St. Venant-Kirchhoff) | Full DPIM pipeline with Gridap.jl FEM backend | runnable | 8 min |
| `06_dielectric_elastomer_actuator/` | Dielectric elastomer actuator (pure-Julia Hermite beam) | Coupled electrostatic-mechanical ROM, order-3 `NthOrderModel` | runnable | minutes |
| `mesh_import/` | Test meshes | Abaqus/COMSOL → GMSH format conversion | utility | seconds |
| `internals/` | Synthetic models | Low-level API: polynomials, multiindices, parametrisation method | utility | seconds–1 min |

All **Ferrite.jl-backed examples** moved to the companion package
[MORFEFerrite.jl](https://github.com/MORFEproject/MORFEFerrite.jl/tree/main/examples):

| Folder (in MORFEFerrite.jl) | Model |
| --------------------------- | ----- |
| `01_clamped_beam_ferrite/` | Clamped-clamped SVK beam — high-level `StructuralSVK` UI + low-level pipeline |
| `03_arch_comsol_wedge/` | Arch wedge, COMSOL `.mphtxt` P18 mesh |
| `04_parametric_clamped_beam/` | Two-parameter ROM (axial stretch θ₁ + bending-mode arch θ₂), general `ParametricStructural` engine |
| `05_karman_vortex_street/` | Cylinder wake flow (Kármán), `FluidNavierStokes` backend |
| `07_parametric_arch/` | Single-parameter sinusoidal arch, `ParametricStructural` |
| `08_mems_micromirror/` | MEMS scanning micromirror, `StructuralSVK` |

---

## How to run an example

From the repository root:

```bash
julia --project=examples/02_clamped_beam_gridap -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/02_clamped_beam_gridap/main.jl")'
```

Each example's own `README.md` has the exact command.

## Validation

After a fresh run, compare against the reference with the example's
`validate.jl` (where provided). Reference results are regenerated only by
deliberately blessing a verified run into the tracked reference folder.

## Notes on CI

Examples call `Pkg.activate` and `Pkg.instantiate` at startup — they manage
their own environments and cannot be `include()`d directly from the MORFE test
suite (which uses a different environment). Run them as standalone processes or
in a dedicated CI job. The lightweight internals demos do work from the test
suite via `GROUP=examples julia --project test/runtests.jl`.

## Historical result sets and large binaries

Large binaries (`.jls`, `.h5`, `.vtu`, meshes, IFX data) and historical result
sets live in the `MORFE_results_archive` repository (sibling folder — see its
`INDEX.md`). They are never re-tracked here.
