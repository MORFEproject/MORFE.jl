"""
	ParametrisationSolver

Operational workflow for constructing a parametrisation and reduced dynamics.

This internal submodule owns the work surrounding the mathematical cohomological
equation: configuration, `W`/`R` storage preparation, checkpoint restoration and commits,
conjugate reconstruction, causal monomial scheduling, progress, and benchmarking. It
delegates each individual cohomological equation to `CohomologicalEquations` and delegates
sparse factorisation to `BorderedLinearSolvers`.

# Source organisation

| File | Purpose |
|:-----|:--------|
| `Configuration.jl` | User-facing execution and checkpoint options |
| `Checkpointing.jl` | Validated, atomic checkpoint persistence and restoration |
| `ConjugateSymmetry.jl` | Conjugate-pair bookkeeping and coefficient reconstruction |
| `SolveProgress.jl` | Allocation-conscious terminal progress reporting |
| `SolveSchedule.jl` | Causal jobs and exact structural factor groups |
| `SolveExecution.jl` | Typed observers and plan execution |
| `Benchmarking.jl` | Timing instrumentation and CSV reporting |
| `SolutionStorage.jl` | Creation, reuse, initialisation, and checkpoint restoration of `W`/`R` |
| `ExternalDirections.jl` | Ordered external solves and conjugate-block validation |
| `SolvePreparation.jl` | Backend, operator, symmetry, resources, and complete context |
| `SolveProblem.jl` | Short top-level phase orchestration |

The module root contains only dependencies, includes, and exports so that these
responsibilities remain visible rather than accumulating in one driver file.
"""
module ParametrisationSolver

using LinearAlgebra
using SparseArrays
using SHA
using TOML
using StaticArrays: SVector, MVector

using ..Multiindices: MultiindexSet, build_exponent_index_map
using ..Polynomials: DensePolynomial
using ..ParametrisationObjects: validate_multiindex_set,
                                Parametrisation, ReducedDynamics,
                                create_parametrisation_method_objects,
                                compute_higher_derivative_coefficients!,
                                multiindex_set
using ..FullOrderModel: NthOrderModel
using ..ExternalSystems: external_basis
using ..SpectralDecomposition: SpectralData, right_modes, left_modes,
                               right_mode_derivatives, left_mode_blocks,
                               master_eigenvalues, master_conjugate_permutation,
                               detect_conjugate_permutation,
                               external_conjugate_permutation,
                               full_conjugate_permutation
using ..MultilinearTerms: compute_multilinear_terms!, build_multilinear_terms_cache,
                          MultilinearTermsCache
using ..Resonance: ResonanceSet
using ..InvarianceEquation: precompute_master_column_polynomials,
                            precompute_external_column_polynomials,
                            precompute_sparse_L_template
using ..MasterModeOrthogonality: precompute_orthogonality_operator_coefficients,
                                 precompute_orthogonality_column_polynomials
using ..CohomologicalEquations: CohomologicalContext, InvarianceOperators,
                                OrthogonalityOperators, LowerOrderResources,
                                CohomologicalBuffers, solve_single_monomial!,
                                _run_single_monomial!, _solve_monomial!,
                                _NO_MONOMIAL_INSTRUMENTATION, _superharmonic,
                                _resonance_vector
import ..CohomologicalEquations: _assemble_nonlinear_rhs!, _solve_prepared_system!,
                                 _monomial_metrics
using ..BorderedLinearSolvers: SparseLinearSolverState, _backend_name

include("Configuration.jl")
include("Checkpointing.jl")
include("ConjugateSymmetry.jl")
include("SolutionStorage.jl")
include("ExternalDirections.jl")
include("SolvePreparation.jl")
include("SolveProgress.jl")
include("SolveSchedule.jl")
include("SolveExecution.jl")
include("Benchmarking.jl")
include("SolveProblem.jl")

export ParametrisationOptions, CheckpointOptions, checkpoint_fingerprint_data,
       NoConjugatePermutation, ConjugateSymmetryData, fill_conjugate_monomial!,
       detect_conjugate_permutation, external_conjugate_permutation,
       full_conjugate_permutation, solve_cohomological_equations!,
       solve_cohomological_equations_benchmarked!, solve_cohomological_problem

end # module ParametrisationSolver
