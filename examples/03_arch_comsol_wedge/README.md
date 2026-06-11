# 03 — Arch COMSOL wedge

## Model

Arch structure (isotropic polysilicon, St. Venant-Kirchhoff) loaded from a COMSOL `.mphtxt`
mesh (`arch_2_force.mphtxt`, P18 quadratic wedge elements). Demonstrates the COMSOL mesh
import pipeline and forced/unforced DPIM.

## How to run

```bash
julia --project=examples/03_arch_comsol_wedge -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/03_arch_comsol_wedge/main.jl")'
```

Pass an alternative config as a positional argument:

```bash
julia --project=examples/03_arch_comsol_wedge main.jl configs/mode_1_order_5_cnf.jl
```

## Expected outputs

```text
results/
  mode_1_order_5_cnf/
    summary.log              — verbose run log (tee'd to terminal)
    summary.txt              — structured key:value summary with environment info
    data/
      W.jls                  — parametrisation (serialised)
      R.jls                  — reduced dynamics (serialised)
    figures/                 — (empty; use tools/ scripts for post-processing)
```

## Reference results

Curated reference outputs live in `results/reference/mode_1_order_5_cnf/` (tracked in git).

## Approximate runtime

~10–20 minutes depending on hardware (order-5 parametrisation, sparse Ferrite assembly).

## Subdirectories

| Folder | Contents |
| ------ | -------- |
| `configs/` | Run configurations (tracked inputs; one `.jl` file per run) |
| `setup/` | Assembly, mesh, and logging helpers included by the main script |
| `tools/` | Post-processing utilities (node-DOF table, mode visualisation) |
