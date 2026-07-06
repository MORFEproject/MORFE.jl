# 08 — MEMS micromirror (gimbal)

## Model

Single-crystal silicon MEMS mirror-gimbal (St. Venant-Kirchhoff), reduced with DPIM.
The geometry starts as the SolidWorks part `Mirror-Gimbal-v1.2.SLDPRT` and is meshed
through Gmsh, then loaded via the SVK `mechanical_model` Gmsh-`.msh` path.

## Why the mesh is not committed

`Mirror-Gimbal-v1.2.SLDPRT` is **native SolidWorks binary** (proprietary, compressed).
No open-source tool — meshio, Gmsh, FreeCAD — can read its geometry, so it cannot be
converted inside this repo. You must first export a neutral **STEP** file from a CAD
tool that opens SolidWorks parts. This is a one-time manual step.

## Step 1 — export a STEP from the `.SLDPRT`

- **macOS (no SolidWorks): Autodesk Fusion 360** — free for personal use. Upload/insert
  `Mirror-Gimbal-v1.2.SLDPRT`, then **File → Export → STEP (`.step`, AP214)**.
- **Windows: SolidWorks** — **File → Save As → STEP AP214 (`.step`)**.
- Save the result as `mirror_gimbal.step` in this folder.

> Avoid online SLDPRT→STEP converters: they upload proprietary geometry to a third
> party (may be cached/indexed). Prefer a local CAD tool.

**Alternative — skip Gmsh entirely** if you can already export a *volume mesh*:
- COMSOL `.mphtxt` → load directly like [example 03](../03_arch_comsol_wedge/main.jl)
  (`dirichlet` = `Set{Int}` of boundary entity IDs).
- Abaqus `.inp` → convert with `MORFE.FEMUtility.abaqus_to_gmsh(inp, msh)`.
- Gmsh `.msh` → point `main.jl` at it directly.

## Step 2 — mesh it

```bash
julia --project mesh.jl
```

First run prints a **Surface tags / bounding boxes** table and the model bounding
box. Use it to identify the anchor faces, then edit the `CLAMP` block near the bottom
of `mesh.jl`:

- explicit tags: `const CLAMP = [7, 12]`, or
- a substrate plane: `const CLAMP = select_by_plane(:z, zmin; tol = 1e-3)`.

Also confirm `LENGTH_SCALE` (STEP is usually mm → `1000.0` gives µm) and
`MESH_SIZE_MIN/MAX`. Re-run to write `mirror_gimbal.msh` with a non-empty `"clamp"`
facetset.

## Step 3 — build the ROM

```bash
julia --project main.jl
```

Uses single-crystal silicon (`E = 170e3 MPa`, `ν = 0.28`, `ρ = 2.33e-3`) in the
µm-consistent unit convention of example 03, clamps the `"clamp"` facetset, and runs
order-5 DPIM about master mode 1.

## Expected outputs

```text
results/
  data/     — W.jls (parametrisation), R.jls (reduced dynamics)
  summary.txt
```

## Notes

- Elements are **linear tets**; `fe_order = 2` in `main.jl` adds the quadratic
  displacement field (field order is independent of geometric order).
- Units are critical: the first eigenfrequency should land in a sane MEMS range
  (kHz–MHz). If it is off by powers of ten, revisit `LENGTH_SCALE`.
- If `FerriteGmsh.togrid` rejects the `.msh`, set `Mesh.MshFileVersion = 2.2` in
  `mesh.jl` before `gmsh.write`, or use one of the direct-mesh alternatives above.
