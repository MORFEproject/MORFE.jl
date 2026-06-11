# mesh_import — FEM mesh format conversion utilities

Demonstrates converting Abaqus and COMSOL mesh files to GMSH format via the
`FEMUtility` module exported from MORFE.

**Entry scripts:**
- `Abaqus/demo_abaqus_to_gmsh.jl` — Abaqus `.inp` → GMSH `.msh`
- `Comsol/demo_comsol_to_gmsh.jl` — COMSOL `.mphtxt` → GMSH `.msh`

Test fixture meshes (cube, wedge) are included in both subdirectories.

## How to run

```bash
julia --project -e 'include("examples/mesh_import/Abaqus/demo_abaqus_to_gmsh.jl")'
julia --project -e 'include("examples/mesh_import/Comsol/demo_comsol_to_gmsh.jl")'
```
