module MORFEFerriteExt

using MORFE: MORFE
using Ferrite
using LinearAlgebra
using SparseArrays

_ext(file) = joinpath(@__DIR__, "FerriteBackend", file)
include(_ext("ferrite_assembly.jl"))

MORFE.ferrite_nonlinearity(degree::Integer, args...; kwargs...) =
    FerriteGeometricNonlinearity{Int(degree)}(args...; kwargs...)

MORFE.ferrite_assemble_KM!(args...; kwargs...) = assemble_KM!(args...; kwargs...)

end
