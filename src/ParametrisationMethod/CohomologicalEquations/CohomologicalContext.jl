# =============================================================================
# CohomologicalContext — composed struct replacing the 22-field flat layout
# =============================================================================

"""
	CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT}

Bundles all precomputed data required to solve the cohomological equations for
every monomial.  The 22 flat fields of the previous design are composed into five
named sub-structs, making data provenance explicit at every call site:

```
ctx.invariance.column_coeffs (was: ctx.invariance_C_coeffs)
ctx.orthogonality.J_coeffs   (was: ctx.orthogonality_J_coeffs)
ctx.buffers.rhs              (was: ctx.rhs_buffer)
ctx.lower_order.buffer       (was: ctx.lower_order_buffer)
ctx.sparse_solver.pardiso    (was: ctx.pardiso_solver)
```

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
