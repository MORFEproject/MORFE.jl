# 03 — Arch COMSOL wedge

Arch structure loaded from a COMSOL `.mphtxt` mesh file, demonstrating the
COMSOL → GMSH import pipeline and the full DPIM solve.

**Entry script:** `arch_2_force.jl`

**Mesh:** `arch_2_force.mphtxt` (COMSOL export; converted to GMSH at runtime via `comsol_to_gmsh`)

**Expected output:** eigenvalues, parametrisation W and reduced dynamics R, summary log.

## How to run

```bash
julia --project=examples/03_arch_comsol_wedge -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/03_arch_comsol_wedge/arch_2_force.jl")'
```

## Subdirectories

| Folder | Contents |
|--------|----------|
| `setup/` | Assembly, mesh, and logging helpers included by the main script |
| `tools/` | Post-processing utilities (node-DOF table, mode visualisation) |
