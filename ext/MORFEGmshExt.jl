module MORFEGmshExt

using MORFE
using MORFE.FEMUtility
using Gmsh
using Printf

_ext(f) = joinpath(pkgdir(MORFE), "ext", "FEMUtility", f)
include(_ext("AbaqusToGmsh.jl"))
include(_ext("ComsolToGmsh.jl"))
include(_ext("GmshToComsol.jl"))

using .AbaqusToGmsh: abaqus_to_gmsh, abaqus_to_gmsh_linear
using .ComsolToGmsh: comsol_to_gmsh, comsol_to_gmsh_linear
using .GmshToComsol: gmsh_to_comsol

MORFE.FEMUtility.abaqus_to_gmsh(args...; kwargs...) =
    AbaqusToGmsh.abaqus_to_gmsh(args...; kwargs...)

MORFE.FEMUtility.abaqus_to_gmsh_linear(args...; kwargs...) =
    AbaqusToGmsh.abaqus_to_gmsh_linear(args...; kwargs...)

MORFE.FEMUtility.comsol_to_gmsh(args...; kwargs...) =
    ComsolToGmsh.comsol_to_gmsh(args...; kwargs...)

MORFE.FEMUtility.comsol_to_gmsh_linear(args...; kwargs...) =
    ComsolToGmsh.comsol_to_gmsh_linear(args...; kwargs...)

MORFE.FEMUtility.gmsh_to_comsol(args...; kwargs...) =
    GmshToComsol.gmsh_to_comsol(args...; kwargs...)

end # module MORFEGmshExt
