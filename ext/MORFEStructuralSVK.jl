"""
MORFEStructuralSVK — high-level UI for St. Venant-Kirchhoff structural models
with the Ferrite backend: mesh → `mechanical_model` → `parametrise` → ROM,
autonomous or with near-resonant harmonic forcing.

Access:

    using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps
    SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)

(Direct `MORFE.parametrise` access would require stubs in `src/`, which is
intentionally avoided; see HIGH_LEVEL_API_PLAN.md.)
"""
module MORFEStructuralSVK

using MORFE: MORFE
using MORFE
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using LinearAlgebra, SparseArrays, Serialization, Printf
using StaticArrays

_svk(file) = joinpath(@__DIR__, "StructuralSVK", file)
_femu(file) = joinpath(@__DIR__, "FEMUtility", file)
include(_svk("types.jl"))
include(_svk("rayleigh_solver.jl"))
include(_femu("comsol_ferrite.jl"))
include(_svk("mechanical_model.jl"))
include(_svk("parametrise.jl"))
include(_svk("postprocess.jl"))

end
