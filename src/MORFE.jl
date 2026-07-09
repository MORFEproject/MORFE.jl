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
include("Export/ParaviewExport.jl")

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
using .ParaviewExport

# High-level `parametrise(model, order, eigenproblem; …)` method. Included here
# (after the using-block above) so its body can resolve `solve_cohomological_problem`,
# `build_resonance_set`, `all_multiindices_up_to`, etc. from the re-exported scope.
include("ParametrisationMethod/parametrise_entry.jl")

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
# Ferrite extension entry points (populated by MORFEFerriteExt when Ferrite is loaded)
export ferrite_nonlinearity, ferrite_assemble_KM!

# FullOrderModel
export FullOrderModel, FirstOrderModel, NDOrderModel,
	linear_first_order_matrices, evaluate_nonlinear_terms!

# Eigenproblems
export AbstractEigensolver, DefaultEigensolver, ArpackEigensolver, MorfeEigensolver,
	StructureModalDampingEigensolver
export solve, solve_left, sort_by_magnitude!, normalise_biorthogonal!
export Eigenproblem, solve_eigenproblem, get_eigenpairs, select_master_modes_by_hand,
	select_master_modes_by_sorting, select_master_modes_by_target_frequency
export left_eigenmode_orders_from_slice

# Realification
export realify, compose_linear, realify_via_linear

#Resonance
export ResonanceSet,
	n_internal,
	empty_resonance_set,
	resonance_set_from_graph_style,
	resonance_set_from_complex_normal_form_style,
	resonance_set_from_real_normal_form_style,
	resonance_set_from_condition_number_estimate,
	resonant_multiindices,
	resonant_targets

# ParametrisationMethod
export Parametrisation, ReducedDynamics, create_parametrisation_method_objects,
       restrict_ReducedDynamics_to_degree, restrict_Parametrisation_to_degree
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

# Paraview export
export write_paraview_mesh, write_paraview_modes, write_paraview_manifold,
	write_paraview_deformation

# MORFESymbolicsExt
function model_from_symbolics end
function externalsystem_from_symbolics end
export model_from_symbolics, externalsystem_from_symbolics

# ParametrisationMethod high-level entry (also re-exported from the submodule)
export parametrise

end # module
