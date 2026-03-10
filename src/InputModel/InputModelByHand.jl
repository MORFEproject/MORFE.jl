# Implementation of MORFE 2.0 https://github.com/MORFEproject/MORFE2.0
include("Morfe_2_0/Morfe_2_0.jl")
using .Morfe_2_0
using SparseArrays: SparseMatrixCSC

#=
InputModelByHand implements an ODE system of second order:
    M d_t² U  +  C d_t U + KU  +  F_nl(U) = F(t) 
with the nonlinearity up to order 3. C has to be calculated by modal damping, and F(t) as linearcombination of eigenmodes.
=#
mutable struct InputModelByHand <: InputModelAbstractSecondOrder
    mesh::Grid
    U::Field
    dim
    # FEM Storage
    M
    K
    F
end

function InputModelByHand(info_file::String, mesh_file::String)
  include(info_file)
  dim = 3
  mesh = read_mesh(mesh_file,domains_list,materials_list,materials_dict,
                 boundaries_list,constrained_dof,bc_vals)
  U = Field(mesh, dim)

  return InputModelByHand(
    mesh,
    U,
    dim,
    nothing, 
    nothing,
    nothing
  )
end

function assemble!(fem::InputModelByHand)
    colptr, rowval = Morfe_2_0.assembler_dummy_MK(fem.mesh, fem.U)
    val = zeros(Float64,length(rowval))
    fem.K = SparseMatrixCSC(fem.U.neq, fem.U.neq, colptr, rowval, val)
    fem.M = deepcopy(fem.K)
    Morfe_2_0.assembler_MK!(fem.mesh, fem.U, fem.K, fem.M)
end

function assemble_mass_matrix!(fem::InputModelByHand)
    error("For $(typeof(fem)) use assemble_system!")
end

function assemble_stiffness_matrix!(fem::InputModelByHand)
    error("For $(typeof(fem)) use assemble_system!")
end

function assemble_load_vector!(fem::InputModelByHand)
    error("Use eigenmode as load vector!")
end

function evaluate_quadratic_nonlinearity(fem::InputModelByHand, Ψ₁,Ψ₂)
    res = zeros(2*fem.U.neq)
    Morfe_2_0.assembly_G!(res, Ψ₁, Ψ₂, fem.U, fem.mesh)
    return res
end

function evaluate_cubic_nonlinearity(fem::InputModelByHand, Ψ₁,Ψ₂,Ψ₃)
    res = zeros(2*fem.U.neq)
    Morfe_2_0.assembly_H!(res, Ψ₁, Ψ₂, Ψ₃, fem.U, fem.mesh)
    return res
end

function mass_matrix(fem::InputModelByHand)
    if isnothing(fem.M)
        assemble_mass_matrix!(fem)
    end
    return fem.M
end

function stiffness_matrix(fem::InputModelByHand)
     if isnothing(fem.K)
        assemble_system!(fem)
    end
    return fem.K
end

function load_vector(fem::InputModelByHand)
     if isnothing(fem.F)
        assemble_system!(fem)
    end
    return fem.F
end

function ndofs(fem::InputModelByHand)
    return fem.U.neq
end