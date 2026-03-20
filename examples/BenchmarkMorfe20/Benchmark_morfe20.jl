"""
Implementation of the MORFE module to reproduce and compare the system from Morfe2.0
Different Morfe2.0 examples can be used by exchanging 'info_file' and 'mesh_file'.
"""

include(joinpath(@__DIR__, "../../src/MORFE.jl"))
using .MORFE
include("Morfe_2_0/Morfe_2_0.jl")
using .Morfe_2_0

using LinearAlgebra
using SparseArrays

#Make info
info = Infostruct()
info_file = "beam_damp.jl"
include(info_file)

#Import mesh
mesh_file = "./examples/BenchmarkMorfe20/beam.mphtxt"
mesh = read_mesh(mesh_file, domains_list, materials_list, materials_dict,
    boundaries_list, constrained_dof, bc_vals)

# initialise a dummy field to store dofs ordering and static solutions
U = Field(mesh, Morfe_2_0.dim)

info.nm = length(info.Φ)   # master modes
info.nz = 2 * info.nm
info.nzforce = 2  # imposes only two nonautonomous
if info.Ffreq == 0
    info.nzforce = 0
end
info.nrom = info.nz + info.nzforce
info.nK = U.neq   # dim of FEM problem
info.nA = 2 * info.nK  # dim of first order sys
info.nMat = info.nA + info.nz  # dim of system to be solved

#Assemble matrices
colptr, rowval = Morfe_2_0.assembler_dummy_MK(mesh, U)
val = zeros(Float64, length(rowval))
K = SparseMatrixCSC(U.neq, U.neq, colptr, rowval, val)
M = deepcopy(K)
Morfe_2_0.assembler_MK!(mesh, U, K, M)
C = info.α * M + info.β * K

"""
NDOrderModel

N = 2

M U'' + C U' + K U = F(U)
"""
function quadratic_2d!(res, Ψ₁, Ψ₂)
    Morfe_2_0.assembly_G!(res, Ψ₁, Ψ₂, U, mesh)
end
quadratic_term_2d = FullOrderModel.MultilinearMap(quadratic_2d!, (2, 0))
function cubic_2d!(res, Ψ₁, Ψ₂, Ψ₃)
    Morfe_2_0.assembly_H!(res, Ψ₁, Ψ₂, Ψ₃, fem.U, fem.mesh)
end
cubic_term_2d = FullOrderModel.MultilinearMap(cubic_2d!, (3, 0))
nonlinear_terms_2d = (quadratic_term_2d, cubic_term_2d)
model_2d = FullOrderModel.NDOrderModel((K, C, M), nonlinear_terms_2d)

"""
FirstOrderModel

Use U'=V instead of MU'=MV
"""
A, B = FullOrderModel.linear_first_order_matrices(model_2d)

function quadratic_1d!(res, Ψ₁, Ψ₂)
    Morfe_2_0.assembly_G!(res, Ψ₁, Ψ₂, U, mesh; one_d = true)
end
quadratic_term_1d = FullOrderModel.MultilinearMap(quadratic_1d!)
function cubic_1d!(res, Ψ₁, Ψ₂, Ψ₃)
    Morfe_2_0.assembly_H!(res, Ψ₁, Ψ₂, Ψ₃, fem.U, fem.mesh; one_d = true)
end
cubic_term_1d = FullOrderModel.MultilinearMap(cubic_1d!)
nonlinear_terms_1d = (quadratic_term_1d, cubic_term_1d)

model_1d = FullOrderModel.FirstOrderModel((-A, B), nonlinear_terms_1d)