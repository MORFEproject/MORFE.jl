module MORFE

include("Multiindices.jl")
include("Polynomials.jl")
include("FullOrderModel/MultilinearMaps.jl")
include("FullOrderModel/ExternalSystems.jl")
include("FullOrderModel/FullOrderModel.jl")
include("SpectralDecomposition/SpectralDecomposition.jl")
include("SpectralDecomposition/ConjugatePermutation.jl")
include("SpectralDecomposition/SpectralData.jl")
include("Realification.jl")
include("ParametrisationMethod/Resonance.jl")
include("ParametrisationMethod/InvarianceEquation/InvarianceEquation.jl")
include("ParametrisationMethod/MasterModeOrthogonality/MasterModeOrthogonality.jl")
# Coefficient containers + the mset contract. Must precede CohomologicalEquations,
# which imports them; ParametrisationMethod (which owns `parametrise`) follows the
# solver instead, because it *calls* it.
include("ParametrisationMethod/ParametrisationObjects.jl")
include("ParametrisationMethod/RightHandSide/MultilinearTerms/MultilinearTerms.jl")
include("ParametrisationMethod/RightHandSide/LowerOrderCouplings.jl")
include("ParametrisationMethod/CohomologicalEquations/CohomologicalEquations.jl")
include("ParametrisationMethod/ParametrisationMethod.jl")
include("FEMUtility.jl")
include("BifurcationSolvers/BifurcationKitInterface.jl")
include("Validation/InvarianceError.jl")
include("Export/RomIO.jl")
include("Validation/RomComparison.jl")

# Re‑export public API from submodules
using .Multiindices
using .Polynomials: DensePolynomial, evaluate, extract_component
using .MultilinearMaps
using .ExternalSystems
using .FullOrderModel
using .SpectralDecomposition
using .ConjugatePermutation
using .SpectralDataTypes
using .Realification
using .Resonance
using .InvarianceEquation
using .MasterModeOrthogonality
using .ParametrisationObjects
using .MultilinearTerms: compute_multilinear_terms
using .LowerOrderCouplings
using .CohomologicalEquations
using .ParametrisationMethod
using .FEMUtility
using .BifurcationKitInterface
using .InvarianceError
using .RomIO
using .RomComparison

# Multiindices
export MultiindexSet, zero_multiindex,
       all_multiindices_up_to, multiindices_with_total_degree,
       all_multiindices_in_box, indices_in_box_with_bounded_degree,
       delete_multiindices, is_downward_closed, is_conjugate_closed

# Polynomials
export DensePolynomial, evaluate, extract_component

# MultilinearMaps
export AbstractMultilinearMap, FEMMultilinearMap, MultilinearMap, ExternalSystem
# ExternalSystems — change of external coordinates
export external_basis, external_argument_vectors, to_physical_external
# FEMMultilinearMap interface methods (to be extended by FEM backends,
# e.g. MORFEFerrite's StructuralSVK / FluidNavierStokes)
export fem_elements, fem_n_qp, fem_ndofs_per_cell,
       scatter_qp!, accumulate_qp!, assemble_element!, fem_getdetJdV, fem_qp_buffer,
       fem_reinit!

# FullOrderModel
export FullOrderModel, FirstOrderModel, NDOrderModel,
       linear_first_order_matrices, evaluate_nonlinear_terms!

# SpectralDecomposition
export AbstractEigensolver, DefaultEigensolver, ArpackEigensolver, MorfeEigensolver,
       StructureModalDampingEigensolver
export eigensolve, eigensolve_left, generalised_eigenpairs,
       sort_by_magnitude!, normalise_biorthogonal!, sort_left_eigenmodes
export Spectrum, spectrum, left_eigenmode_orders_from_slice,
       select_master_modes_by_hand, select_master_modes_by_sorting,
       select_master_modes_by_target_frequency
# SpectralData
export SpectralData, ModeBundle, check_biorthogonality,
       right_modes, left_modes, right_mode_derivatives, left_mode_blocks,
       master_eigenvalues, outer_eigenvalues
# Resonance configuration
export ResonanceConfig

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
       detect_conjugate_permutation,
       external_conjugate_permutation, full_conjugate_permutation
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
export compare_rom_coefficients

# ROM persistence
export save_rom, read_rom_coefficients, write_rom_coefficients_csv

# MORFESymbolicsExt
function model_from_symbolics end
function externalsystem_from_symbolics end
export model_from_symbolics, externalsystem_from_symbolics

# ParametrisationMethod high-level entry (also re-exported from the submodule)
export parametrise, build_multiindex_set, validate_multiindex_set

end # module
