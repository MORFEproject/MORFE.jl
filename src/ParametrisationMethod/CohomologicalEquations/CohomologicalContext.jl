# =============================================================================
# CohomologicalContext — everything the per-monomial solve needs, precomputed
# =============================================================================

"""
	CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT}

Bundles all precomputed data required to solve the cohomological equations for
every monomial.

The solve visits every multi-index in turn, and nothing in this struct varies with
the multi-index: operator coefficients, resonance look-ups, buffers and the cached
factorisation are all built once by [`solve_cohomological_problem`](@ref) and reused.
That is what allows [`solve_single_monomial!`](@ref) to run without heap allocation.
Related data is grouped into named sub-structs rather than kept flat, so each field's
provenance is visible at the call site (`ctx.orthogonality.J_coeffs`, not a bare
`ctx.J_coeffs`).

# Type parameters

| Parameter | Meaning |
|:----------|:--------|
| `T`       | Scalar type (typically `ComplexF64`) |
| `ORD`     | Polynomial order of the full-order ODE |
| `ORDP1`   | `ORD + 1` |
| `NVAR`    | Total reduced variables: `ROM + N_EXT` |
| `FOM`     | Full-order state dimension |
| `LT`      | Element type of the FOM matrices |
| `MT`      | Matrix type; sparse path when `MT <: SparseMatrixCSC` |

# Fields

- `linear_terms::NTuple{ORDP1, MT}` — the `ORD+1` FOM matrices `B₀ … B_ORD` whose
  combination `L(s) = Σ_j s^j B_j` forms the `(1,1)` block of the bordered system.
- `generalised_eigenmodes::Matrix{T}` — right eigenmode order-blocks, used to
  initialise the linear monomials and to propagate higher derivative coefficients.
- `lambda_diag::Vector{T}` — the `NVAR` reduced eigenvalues; the superharmonic of a
  monomial is `s = ⟨lambda_diag, α⟩`.
- `invariance::InvarianceOperators{T}` — border-column coefficients; see
  [`InvarianceOperators`](@ref).
- `orthogonality::OrthogonalityOperators{T}` — orthogonality row, corner and
  external coefficients; see [`OrthogonalityOperators`](@ref).
- `resonance_set::ResonanceSet` — which master modes are resonant with which
  monomial, deciding per row whether the border is populated or masked out.
- `linear_monomial_skip_set::Set{Int}` — positions of monomials already fixed
  before the loop (linear ones from eigenvectors, external forcing directions), so
  the driver skips rather than re-solves them.
- `lower_order::LowerOrderResources{NVAR, T}` — coupling buffers and multiindex
  look-up; see [`LowerOrderResources`](@ref).
- `buffers::CohomologicalBuffers{T}` — assembly and solve scratch; see
  [`CohomologicalBuffers`](@ref).
- `sparse_solver::Union{Nothing, SparseLinearSolverState{T}}` — sparse-path template
  and factorisation handles, or `nothing` on the dense path.  This field, not `MT`
  alone, is what the solve branches on.
"""
struct CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT <: AbstractMatrix{LT}}
    # ── Spectral / model data ─────────────────────────────────────────────────
    linear_terms::NTuple{ORDP1, MT}
    generalised_eigenmodes::Matrix{T}
    lambda_diag::Vector{T}
    # ── Precomputed operators ─────────────────────────────────────────────────
    invariance::InvarianceOperators{T}
    orthogonality::OrthogonalityOperators{T}
    # ── Resonance bookkeeping ─────────────────────────────────────────────────
    resonance_set::ResonanceSet
    linear_monomial_skip_set::Set{Int}
    # ── Compute resources ─────────────────────────────────────────────────────
    lower_order::LowerOrderResources{NVAR, T}
    buffers::CohomologicalBuffers{T}
    sparse_solver::Union{Nothing, SparseLinearSolverState{T}}
end
