# Compatibility bindings for the former all-in-one `CohomologicalEquations` namespace.
#
# New code should use the owning modules. These imports preserve existing qualified names
# without moving scheduling, checkpointing, or factorisation implementations back into the
# mathematical equation module.
@eval CohomologicalEquations begin
    using ..BorderedLinearSolvers: SparseLinearSolverState,
                                   AbstractSparseBackend, KLUBackend,
                                   UMFPACKBackend, PardisoBackend,
                                   _BorderedSolveFailure, _backend_name,
                                   _cached_klu_factor, _cached_umfpack_factor,
                                   _refactorise!, _refactorise_klu!,
                                   _refactorise_umfpack!, _bordered_solve!,
                                   _try_build_pardiso_solver, _pardiso_prepare!,
                                   _pardiso_factorise_solve!, _pardiso_solve!,
                                   _pardiso_release!
    using ..ParametrisationSolver: ParametrisationOptions, CheckpointOptions,
                                   CheckpointSession, checkpoint_fingerprint_data,
                                   _replace_options, NoConjugatePermutation,
                                   ConjugateSymmetryData, fill_conjugate_monomial!,
                                   detect_conjugate_permutation,
                                   external_conjugate_permutation,
                                   full_conjugate_permutation,
                                   solve_cohomological_equations!,
                                   solve_cohomological_equations_benchmarked!,
                                   solve_cohomological_problem,
                                   _SimpleProgress, _SolveJob, _DirectSolvePlan,
                                   _SolveGroup, _GroupedSolvePlan, StructuralFactorKey,
                                   _CompositeSolveObserver, _ProgressSolveObserver,
                                   _CheckpointSolveObserver, _OrderAccum,
                                   _BenchmarkSolveObserver, _build_conjugate_symmetry,
                                   _build_solve_jobs, _build_solve_plan,
                                   _eigenvalue_representatives,
                                   _has_structural_factor_reuse,
                                   _structural_factor_key, _group_solve_jobs,
                                   _embed_external_dynamics!,
                                   _linear_monomial_indices,
                                   _prepare_solution_storage,
                                   _restore_solution_checkpoint!,
                                   _prepare_shared_resources,
                                   _prepare_master_operators,
                                   _prepare_external_directions!,
                                   _build_complete_context,
                                   _compose_observers, _execute_solve_plan!,
                                   _solve_cohomological_equations!,
                                   _solve_cohomological_equations_typed!, TOML

    export SparseLinearSolverState,
           ParametrisationOptions, CheckpointOptions,
           checkpoint_fingerprint_data,
           NoConjugatePermutation, ConjugateSymmetryData,
           fill_conjugate_monomial!, detect_conjugate_permutation,
           external_conjugate_permutation, full_conjugate_permutation,
           solve_cohomological_equations!,
           solve_cohomological_equations_benchmarked!,
           solve_cohomological_problem,
           _try_build_pardiso_solver, _pardiso_prepare!,
           _pardiso_factorise_solve!, _pardiso_solve!, _pardiso_release!
end
