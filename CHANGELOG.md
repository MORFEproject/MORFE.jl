# Changelog

## Next minor release

### Breaking

- Removed the `FEMUtility` module and the `abaqus_to_gmsh`,
  `abaqus_to_gmsh_linear`, `comsol_to_gmsh`, `comsol_to_gmsh_linear`, and
  `gmsh_to_comsol` exports. Mesh loading and conversion now belong to
  `MORFEFerrite.Common.MeshIO` and are exported directly by MORFEFerrite.
