module FEM

include("FEMInterface.jl")

include("GridapFEM.jl")

include("ByHandFEM.jl")

export AbstractFEM, GridapFEM, ByHandFEM

export assemble_mass_matrix!, assemble_stiffness_matrix!, assemble_load_vector!, assemble_system!, 
    mass_matrix, stiffness_matrix, load_vector, 
    evaluate_cubic_nonlinearity, evaluate_quadratic_nonlinearity,
    ndofs, 
    field_from_vector, visualize

end #module