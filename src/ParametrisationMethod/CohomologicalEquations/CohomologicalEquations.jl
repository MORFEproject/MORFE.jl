"""
	CohomologicalEquations

Solve the cohomological equations that arise in the parametrisation method for
computing Spectral Submanifolds (SSMs) of high-dimensional dynamical systems.

---

# Problem statement

Given a full-order model of order `ORD` in `FOM` degrees of freedom, the
parametrisation method seeks a change of coordinates

```
x(t) = W(z(t)),          z ∈ ℂᴺᵛᵃʳ
```

such that the reduced dynamics `ż = R(z)` is simpler.  Expanding both `W` and
`R` as dense polynomials in the `NVAR = ROM + N_EXT` reduced variables (ROM
master-mode amplitudes plus N_EXT external forcing amplitudes) and matching
coefficients monomial by monomial yields the **cohomological equations**: one
linear system per multi-index `α`.

---

# System structure for multi-index `α`

Let `s = ⟨λ, α⟩` be the superharmonic frequency and let `P = diag(ρ)` be the
diagonal `0/1` matrix of the master modes resonant at `α`, `Q = I − P`.  The
cohomological linear system is **bordered** and of **constant size** `FOM + ROM`:

```
┌                          ┐ ┌         ┐   ┌             ┐
│  L(s)     C(s) P         │ │  W[α]   │ = │  RHS_inv    │   FOM rows  (invariance)
│  P Ĵ(s)   P Ĉ(s) P + τQ  │ │  R[α]   │   │  P RHS_ort  │   ROM rows  (orthogonality)
└                          ┘ └         ┘   └             ┘
```

where:
- `L(s)`  (`FOM × FOM`) is the parametrisation operator,
- `C(s)`  (`FOM × ROM`) acts on the unknown reduced-dynamics coefficients,
- `Ĵ(s)` (`ROM × FOM`) is the orthogonality row operator, stacking the rows `Ĵ_r(s)`,
- `Ĉ(s)` (`ROM × ROM`) is the orthogonality joint operator,
- `W[α] ∈ ℂᶠᵒᵐ` is the parametrisation coefficient,
- `R[α] ∈ ℂᴿᴼᴹ` are the master reduced-dynamics coefficients,
- `τ = 1` on the non-resonant diagonal, so those rows read `R[r, α] = 0`.

Non-resonant master modes stay in the system as trivial equations rather than being
compacted away.  That is what keeps the size — and on the sparse path the sparsity
pattern — independent of `α`, so a single symbolic factorisation serves every
monomial.  It costs nothing in accuracy: permuting the unknowns as
`[W; R_res; R_non]` makes the matrix block triangular with an exactly decoupled `τI`
block, and each trivial row is a singleton in both its row and its column.  External
forcing modes are *known* and appear only on the right-hand side.

## Why the system is bordered rather than reduced

The border is not an optimisation but a conditioning requirement.  Inner resonance is
flagged by `|λ_r − s| < tol`, and `det L(λ_r) = 0` for every master eigenvalue, so
"monomial `α` is resonant" means precisely "`L(s_α)` is numerically singular" — and a
border is present only on exactly those monomials.  Eliminating `L(s)` first, by
forming `L(s)⁻¹·RHS` and `L(s)⁻¹C(s)`, would therefore apply an inverse that does not
usefully exist, with backward error growing like `κ(L(s_α)) → ∞`.

Factorising the bordered matrix as a whole keeps the backward error at `κ` of the
*bordered* matrix, which stays bounded because the border spans the near-null
directions of `L(s)` (Keller's bordering lemma).  This is why the linear algebra 
below never inverts `L(s)` alone, and why its numeric
factorisation must re-pivot on every monomial.

---

# Module contents

| Symbol | Description |
|:-------|:------------|
| [`InvarianceOperators`](@ref)       | Precomputed invariance-equation operator coefficients |
| [`OrthogonalityOperators`](@ref)    | Precomputed orthogonality-condition operator coefficients |
| [`LowerOrderResources`](@ref)       | Lower-order coupling data and buffers |
| [`CohomologicalBuffers`](@ref)      | Pre-allocated system-assembly scratch buffers |
| [`CohomologicalContext`](@ref)      | Composed struct bundling all precomputed operators and resources |
| [`solve_single_monomial!`](@ref)    | Solve the cohomological system for one multi-index |

# Source layout

The module is deliberately organised by responsibility:

| File | Responsibility |
|:-----|:---------------|
| `CohomologicalEquations.jl` | Focused module composition, imports, and exports |
| `SolveState.jl` | Equation operators, lower-order resources, and reusable assembly buffers |
| `EquationAssembly.jl` | Bordered equation assembly and the dense equation solve |
| `MonomialSolve.jl` | Nonlinear right-hand side, bordered-solver call, and `W`/`R` finalisation |

Linear factorisation belongs to `BorderedLinearSolvers`; scheduling, checkpointing,
symmetry, progress, and benchmarking belong to `ParametrisationSolver`.
"""
module CohomologicalEquations

using ..Multiindices: MultiindexSet, indices_in_box_with_bounded_degree,
                      build_exponent_index_map
using ..Polynomials: DensePolynomial
using ..ParametrisationObjects: validate_multiindex_set,
                                Parametrisation, ReducedDynamics,
                                create_parametrisation_method_objects,
                                compute_higher_derivative_coefficients!,
                                multiindex_set
using ..LowerOrderCouplings: compute_lower_order_couplings
using ..InvarianceEquation: assemble_cohomological_matrix_and_rhs!,
                            precompute_master_column_polynomials,
                            precompute_external_column_polynomials,
                            build_sparse_L_and_rhs!,
                            precompute_sparse_L_template,
                            precompute_sparse_bordered_template,
                            scatter_L_into_bordered!,
                            evaluate_column!,
                            evaluate_external_rhs!
using ..MasterModeOrthogonality: assemble_orthogonality_matrix_and_rhs!,
                                 precompute_orthogonality_operator_coefficients,
                                 precompute_orthogonality_column_polynomials
using ..FullOrderModel: NthOrderModel
using ..MultilinearTerms: compute_multilinear_terms, compute_multilinear_terms!,
                          MultilinearTermsCache
using ..Resonance: ResonanceSet, is_resonant
using ..BorderedLinearSolvers: SparseLinearSolverState, _bordered_solve!,
                               _configured_residual_tolerance,
                               _throw_bordered_failure, _is_unrecoverable_failure
using LinearAlgebra
using SparseArrays
using StaticArrays: SVector, MVector
include("SolveState.jl")

export CohomologicalContext,
       InvarianceOperators,
       OrthogonalityOperators,
       LowerOrderResources,
       CohomologicalBuffers,
       solve_single_monomial!

"""
	_resonance_vector(resonance_set, monomial_idx, ::Val{ROM}) -> SVector{ROM, Bool}

Return a compile-time-sized boolean vector indicating which of the `ROM` master
modes are resonant with the monomial at position `monomial_idx` in the multiindex
set.  Using `Val{ROM}` enables the compiler to emit a fully unrolled ntuple loop.
"""
@inline function _resonance_vector(
        resonance_set::ResonanceSet,
        monomial_idx::Int,
        ::Val{ROM}
) where {ROM}
    return SVector{ROM, Bool}(ntuple(r -> is_resonant(resonance_set, monomial_idx, r), Val(ROM)))
end

"""Return the monomial superharmonic `sum(alpha[i] * lambda_diag[i])`."""
@inline _superharmonic(multi, lambda_diag) = sum(multi[i] * lambda_diag[i]
for i in eachindex(lambda_diag))

# Per-monomial preparation, assembly, solution, and finalisation follow. Causal
# scheduling and all-solve entry points belong to `ParametrisationSolver`.

include("EquationAssembly.jl")
include("MonomialSolve.jl")

end # module CohomologicalEquations
