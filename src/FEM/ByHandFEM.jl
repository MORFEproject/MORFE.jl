# Implementation of MORFE 2.0 https://github.com/MORFEproject/MORFE2.0
include("Morfe_2_0/Morfe_2_0.jl")
using .Morfe_2_0
using SparseArrays: SparseMatrixCSC

mutable struct ByHandFEM <: AbstractFEM
    mesh::Grid
    U::Field
    dim
    # FEM Storage
    M
    K
    f
end

function ByHandFEM(info_file::String, mesh_file::String)
  include(info_file)
  dim = 3
  mesh = read_mesh(mesh_file,domains_list,materials_list,materials_dict,
                 boundaries_list,constrained_dof,bc_vals)
  U = Field(mesh, dim)

  return ByHandFEM(
    mesh,
    U,
    dim,
    nothing, 
    nothing,
    nothing
  )
end

function assemble_system!(fem::ByHandFEM)
    colptr, rowval = Morfe_2_0.assembler_dummy_MK(fem.mesh, fem.U)
    val = zeros(Float64,length(rowval))
    fem.K = SparseMatrixCSC(fem.U.neq, fem.U.neq, colptr, rowval, val)
    fem.M = deepcopy(fem.K)
    Morfe_2_0.assembler_MK!(fem.mesh, fem.U, fem.K, fem.M)
end

function assemble_mass_matrix!(fem::ByHandFEM)
    error("For $(typeof(fem)) use assemble_system!")
end

function assemble_stiffness_matrix!(fem::ByHandFEM)
    error("For $(typeof(fem)) use assemble_system!")
end

function assemble_load_vector!(fem::ByHandFEM)
    error("Use eigenmode as load vector!")
end

function evaluate_quadratic_nonlinearity(fem::ByHandFEM, Ψ₁,Ψ₂)
    res = zeros(2*fem.U.neq)
    Morfe_2_0.assembly_G!(res, Ψ₁, Ψ₂, fem.U, fem.mesh)
    return res
end

function evaluate_cubic_nonlinearity(fem::ByHandFEM, Ψ₁,Ψ₂,Ψ₃)
    res = zeros(2*fem.U.neq)
    Morfe_2_0.assembly_H!(res, Ψ₁, Ψ₂, Ψ₃, fem.U, fem.mesh)
    return res
end

function mass_matrix(fem::ByHandFEM)
    if isnothing(fem.M)
        assemble_mass_matrix!(fem)
    end
    return fem.M
end

function stiffness_matrix(fem::ByHandFEM)
     if isnothing(fem.K)
        assemble_system!(fem)
    end
    return fem.K
end

function load_vector(fem::ByHandFEM)
     if isnothing(fem.f)
        assemble_system!(fem)
    end
    return fem.f
end

function ndofs(fem::ByHandFEM)
    return fem.U.neq
end