module MORFE

include("Multiindices.jl")
include("Polynomials.jl")
include("FullOrderModel/MultilinearMaps.jl")
include("FullOrderModel/ExternalSystems.jl")
include("FullOrderModel/FullOrderModel.jl")
include("SpectralDecomposition/Eigensolvers.jl")
include("Realification.jl")
include("ParametrisationMethod/Resonance.jl")
include("ParametrisationMethod/InvarianceEquation.jl")
include("ParametrisationMethod/MasterModeOrthogonality.jl")
include("ParametrisationMethod/ParametrisationMethod.jl")
include("ParametrisationMethod/RightHandSide/MultilinearTerms.jl")
include("ParametrisationMethod/RightHandSide/LowerOrderCouplings.jl")

# Re‑export public API from submodules
using .Multiindices
using .Polynomials: DensePolynomial, evaluate
using .MultilinearMaps
using .ExternalSystems
using .FullOrderModel
using .Eigensolvers: generalized_eigenpairs
using .Resonance
using .InvarianceEquation
using .MasterModeOrthogonality
using .ParametrisationMethod
using .MultilinearTerms: compute_multilinear_terms
using .LowerOrderCouplings

export MultiindexSet, zero_multiindex,
	all_multiindices_up_to, multiindices_with_total_degree,
	all_multiindices_in_box, indices_in_box_with_bounded_degree
export DensePolynomial, evaluate
export MultilinearMap, ExternalSystem
export FullOrderModel, FirstOrderModel, NDOrderModel, MultilinearMap,
	linear_first_order_matrices, evaluate_nonlinear_terms!
export SingleResonance, ResonanceSet, resonance_set, resonance_set_from_eigenvalues
export Parametrisation
export compute_multilinear_terms

end # module
