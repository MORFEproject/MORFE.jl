module MORFE

include("Multiindices.jl")
include("Polynomials.jl")
include("FullOrderModel/MultilinearMaps.jl")
include("FullOrderModel/ExternalSystems.jl")
include("FullOrderModel/FullOrderModel.jl")
include("SpectralDecomposition/Eigensolvers.jl")
include("SpectralDecomposition/Eigenproblems.jl")
include("Realification.jl")
include("ParametrisationMethod/Resonance.jl")
include("ParametrisationMethod/InvarianceEquation/InvarianceEquation.jl")
include("ParametrisationMethod/MasterModeOrthogonality/MasterModeOrthogonality.jl")
include("ParametrisationMethod/ParametrisationMethod.jl")
include("ParametrisationMethod/RightHandSide/MultilinearTerms/MultilinearTerms.jl")
include("ParametrisationMethod/RightHandSide/LowerOrderCouplings.jl")
include("ParametrisationMethod/CohomologicalEquations/CohomologicalEquations.jl")
include("FEMUtility.jl")
include("BifurcationSolvers/BifurcationKitInterface.jl")
include("Validation/InvarianceError.jl")

# Re‑export public API from submodules
using .Multiindices
using .Polynomials: DensePolynomial, evaluate, extract_component
using .MultilinearMaps
using .ExternalSystems
using .FullOrderModel
using .Eigensolvers
using .Eigenproblems
using .Realification
using .Resonance
using .InvarianceEquation
using .MasterModeOrthogonality
using .ParametrisationMethod
using .MultilinearTerms: compute_multilinear_terms
using .LowerOrderCouplings
using .CohomologicalEquations
using .FEMUtility
using .BifurcationKitInterface
using .InvarianceError

# Multiindices
export MultiindexSet, zero_multiindex,
       all_multiindices_up_to, multiindices_with_total_degree,
       all_multiindices_in_box, indices_in_box_with_bounded_degree

# Polynomials
export DensePolynomial, evaluate, extract_component

# MultilinearMaps
export AbstractMultilinearMap, FEMMultilinearMap, MultilinearMap, ExternalSystem
# FEMMultilinearMap interface methods (to be extended by FEM backends)
export fem_elements, fem_n_qp, fem_ndofs_per_cell,
       scatter_qp!, accumulate_qp!, assemble_element!, fem_getdetJdV, fem_qp_buffer,
       fem_reinit!

# FullOrderModel
export FullOrderModel, FirstOrderModel, NDOrderModel,
       linear_first_order_matrices, evaluate_nonlinear_terms!

# Eigenproblems
export AbstractEigensolver, DefaultEigensolver, ArpackEigensolver, MorfeEigensolver,
       StructureModalDampingEigensolver
export solve, solve_left, sort_by_magnitude!, normalize_biorthogonal!
export Eigenproblem, solve_eigenproblem, get_eigenpairs, select_master_modes_by_hand,
       select_master_modes_by_sorting, select_master_modes_by_target_frequency

# Realification
export realify, compose_linear, realify_via_linear

#Resonance
export ResonanceSet,
       resonance_set_from_graph_style,
       resonance_set_from_complex_normal_form_style,
       resonance_set_from_real_normal_form_style,
       resonance_set_from_condition_number_estimate

# ParametrisationMethod
export Parametrisation, ReducedDynamics, create_parametrisation_method_objects
export compute_multilinear_terms
export CohomologicalContext,
       InvarianceOperators, OrthogonalityOperators,
       LowerOrderResources, CohomologicalBuffers, SparseLinearSolverState
export NoConjugatePermutation, ConjugateSymmetryData, fill_conjugate_monomial!,
       detect_conjugate_permutation
export solve_cohomological_equations!, solve_cohomological_equations_benchmarked!,
       solve_single_monomial!, solve_cohomological_problem

# FEMUtility
export abaqus_to_gmsh, abaqus_to_gmsh_linear,
       comsol_to_gmsh, comsol_to_gmsh_linear,
       gmsh_to_comsol

# BifurcationKit interface
export make_bk_problem

# Validation
export invariance_error_norms, invariance_error_convergence, plot_invariance_convergence

end # module
