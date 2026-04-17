module MORFE

include("Multiindices.jl")
include("Polynomials.jl")
include("FullOrderModel/MultilinearMaps.jl")
include("FullOrderModel/ExternalSystems.jl")
include("FullOrderModel/FullOrderModel.jl")
include("SpectralDecomposition/Eigensolvers.jl")
include("SpectralDecomposition/JordanChain.jl")
include("Realification.jl")
include("ParametrisationMethod/Resonance.jl")
include("ParametrisationMethod/InvarianceEquation.jl")
include("ParametrisationMethod/MasterModeOrthogonality.jl")
include("ParametrisationMethod/ParametrisationMethod.jl")
include("SpectralDecomposition/PropagateEigenmodes.jl")
include("ParametrisationMethod/RightHandSide/MultilinearTerms.jl")
include("ParametrisationMethod/RightHandSide/LowerOrderCouplings.jl")
include("ParametrisationMethod/CohomologicalEquations.jl")

# Re‑export public API from submodules
using .Multiindices
using .Polynomials: DensePolynomial, evaluate
using .MultilinearMaps
using .ExternalSystems
using .FullOrderModel
using .Eigensolvers
using .JordanChain
using .Resonance
using .InvarianceEquation
using .MasterModeOrthogonality
using .ParametrisationMethod
using .PropagateEigenmodes
using .MultilinearTerms: compute_multilinear_terms
using .LowerOrderCouplings
using .CohomologicalEquations

export MultiindexSet, zero_multiindex,
	all_multiindices_up_to, multiindices_with_total_degree,
	all_multiindices_in_box, indices_in_box_with_bounded_degree
export DensePolynomial, evaluate
export MultilinearMap, ExternalSystem
export FullOrderModel, FirstOrderModel, NDOrderModel,
	linear_first_order_matrices, evaluate_nonlinear_terms!
export ResonanceSet,
	resonance_set_from_graph_style,
	resonance_set_from_complex_normal_form_style,
	resonance_set_from_real_normal_form_style,
	resonance_set_from_condition_number_estimate
export Parametrisation, ReducedDynamics, create_parametrisation_method_objects
export compute_multilinear_terms
export solve_cohomological_equations!, solve_single_monomial!, solve_cohomological_problem

end # module
