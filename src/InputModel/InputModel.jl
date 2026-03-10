module InputModel

include("InputModelInterface.jl")

include("InputModelGridapMechanical.jl")

include("InputModelByHand.jl")

export InputModelAbstract, InputModelAbstractSecondOrder, InputModelGridapMechanical, InputModelByHand

export assemble!, 
    a_matrix, b_matrix, f_vector, 
    mass_matrix, damping_matrix, stiffness_matrix, load_vector,
    evaluate_nonlinearity,
    ndofs, visualize

end #module