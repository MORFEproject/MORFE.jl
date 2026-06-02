module MORFEWriteVTKExt

using MORFE
using MORFE.ParaviewExport
using MORFE.ParametrisationMethod
using Ferrite
using WriteVTK
using Printf

_ext(f) = joinpath(pkgdir(MORFE), "ext", "Paraview", f)
include(_ext("write_vtk.jl"))

end # module MORFEWriteVTKExt
