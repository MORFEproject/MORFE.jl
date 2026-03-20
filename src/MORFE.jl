module MORFE

include("Multiindices.jl")
include("Polynomials.jl")
include("FullOrderModel.jl")
include("Eigensolvers.jl")
include("Realification.jl")
include("ParametrisationMethod/Parametrisation.jl")
include("ParametrisationMethod/RightHandSide/MultilinearTerms.jl")
include("ParametrisationMethod/RightHandSide/LowerOrderCouplings.jl")

# Re‑export public API from submodules
using .Multiindices
using .Polynomials: DensePolynomial, evaluate
using .FullOrderModel: FullOrderModel, FirstOrderModel, NDOrderModel
using .Eigensolvers: generalized_eigenpairs
using .ParametrisationModule: Parametrisation
using .MultilinearTerms: compute_multilinear_terms
using .LowerOrderCouplings

export MultiindexSet
export DensePolynomial, evaluate
export FullOrderModel, FirstOrderModel, NDOrderModel
export Parametrisation
export compute_multilinear_terms

end # module