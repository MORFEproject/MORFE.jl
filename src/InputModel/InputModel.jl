module InputModel

include("InputModelInterface.jl")

include("InputModelGridapMechanical.jl")

include("InputModelByHand.jl")

export InputModelAbstract, InputModelAbstractSecondOrder, InputModelGridapMechanical, InputModelByHand

export assemble!, 
    get_a_matrix, get_b_matrix, get_f_vector,
    mass_matrix, damping_matrix, stiffness_matrix, load_vector,
    evaluate_nonlinearity,
    ndofs, visualize
end #module