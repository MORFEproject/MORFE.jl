module FEMUtility

export abaqus_to_gmsh, abaqus_to_gmsh_linear,
       comsol_to_gmsh, comsol_to_gmsh_linear,
       gmsh_to_comsol

"""
    abaqus_to_gmsh(inp_file::String, gmsh_file::String)

Read an Abaqus `.inp` mesh and write a Gmsh `.msh` file.

Supported element types: C3D4/10/10M, C3D8/8R/8I, C3D20/20R, C3D6, C3D15,
S3/3R, S4/4R, S6, S8/8R, CPS3/4/4R/6/8, CPE3/4/4R/6/8, CAX3/4/4R/6/8/8R,
B31, B32, T3D2, T3D3. Unknown element types are silently skipped.

Requires `Gmsh.jl` — load it with `using Gmsh` to activate the MORFE extension.
"""
function abaqus_to_gmsh end

"""
    abaqus_to_gmsh_linear(inp_file::String, gmsh_file::String)

Like [`abaqus_to_gmsh`](@ref) but downgrades quadratic elements to their
linear (corner-nodes-only) counterparts before writing the Gmsh `.msh` file.

Requires `Gmsh.jl` — load it with `using Gmsh` to activate the MORFE extension.
"""
function abaqus_to_gmsh_linear end

"""
    comsol_to_gmsh(comsol_file::String, gmsh_file::String)

Read a COMSOL `.mphtxt` mesh and write a Gmsh `.msh` file.  Node re-ordering
is applied for T6, Q9, and P18 element types; other types are passed through
as-is.

Requires `Gmsh.jl` — load it with `using Gmsh` to activate the MORFE extension.
"""
function comsol_to_gmsh end

"""
    comsol_to_gmsh_linear(comsol_file::String, gmsh_file::String)

Like [`comsol_to_gmsh`](@ref) but downgrades quadratic elements to their
linear counterparts before writing.

Requires `Gmsh.jl` — load it with `using Gmsh` to activate the MORFE extension.
"""
function comsol_to_gmsh_linear end

"""
    gmsh_to_comsol(gmsh_file::String, comsol_file::String)

Read a Gmsh `.msh` file and write a COMSOL `.mphtxt` mesh.  This is the
inverse of [`comsol_to_gmsh`](@ref): it undoes the node-reordering
permutations and converts 1-based Gmsh indexing back to 0-based COMSOL
indexing.

Requires `Gmsh.jl` — load it with `using Gmsh` to activate the MORFE extension.
"""
function gmsh_to_comsol end

end # module FEMUtility
