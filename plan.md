# MORFE_jl — Performance Optimisation Plan

## Context

The parametrisation method solves a cohomological equation for each of **L monomials** in graded-lexicographic order. The current implementation is architecturally sound (good precomputation, caches, fused Horner passes, in-place BLAS) but has several per-monomial allocation and redundant-work patterns that add up significantly as `L` grows with polynomial degree and number of variables. The demo system (`FOM=2, NVAR=3, degree=3`) is small, but typical engineering models have `FOM ≥ 10`, `NVAR ≥ 4`, `degree ≥ 5`, pushing `L` into the hundreds.

---

## Identified Inefficiencies (by impact)

### 1 — LU Factorisation Rebuilt Every Monomial *(CRITICAL)*
**Location:** `CohomologicalEquations.jl` — `solve_single_monomial!`  
```
ldiv!(lu!(A_sys), ...)   # called L times
```
The system matrix `A_sys` is `(FOM+nR) × (FOM+nR)`. It is factorised from scratch for every monomial, even though the resonance pattern `resonance::SVector{ROM,Bool}` — which determines `nR` and the column structure — changes rarely (only when transitioning between degree shells or at isolated resonant monomials). For most monomials the matrix is identical up to the superharmonic scalar `s`, and the operator columns `C_r(s)` and `J_r(s)` change smoothly.

**Cost:** `O(L × (FOM+nR)³)` — dominates for `FOM ≥ 5`.

### 2 — Per-Monomial Temporary Allocations in `compute_multilinear_terms` *(HIGH)*
**Location:** `MultilinearTerms.jl:370-372`  
```julia
result  = zeros(T, FOM)
scratch = similar(result)
temp    = similar(result)
```
Three `FOM`-length vectors are heap-allocated on every call, one per monomial. The factorisation bookkeeping is cached but the evaluation buffers are not.

### 3 — System Assembly Matrices Allocated Per Monomial *(HIGH)*
**Location:** `InvarianceEquation.jl` (`assemble_cohomological_matrix_and_rhs`) and `MasterModeOrthogonality.jl` (`assemble_orthogonality_matrix_and_rhs`)
```julia
M   = Matrix{T}(undef, FOM, FOM + nR)   # per monomial
rhs = zeros(T, FOM)                      # per monomial
M   = Matrix{T}(undef, nR, FOM + nR)    # per monomial (orthogonality)
rhs = Vector{T}(undef, nR)              # per monomial (orthogonality)
```
These matrices change every monomial but their *maximum size* is fixed: `(FOM + ROM) × (FOM + ROM)`. They should be pre-allocated once and sub-viewed.

### 4 — Unit Vectors Reconstructed on Every Helper Call *(MEDIUM)*
**Location:** `LowerOrderCouplings.jl:23-24` inside `_sum_degree_one_terms!` and `_sum_higher_degree_terms!`  
```julia
unit_vectors = [SVector{NVAR, Int}(ntuple(k -> k==j ? 1 : 0, Val(NVAR))) for j in 1:NVAR]
```
These are constants; they are rebuilt on every call into the lower-order coupling functions (i.e., every monomial). They should live in `CohomologicalContext`.

### 5 — Recursive `vcat()` in Multiindex Generation *(MEDIUM)*
**Location:** `Multiindices.jl:138-150`  
The `recurse` function uses `vcat(prefix, e)` in every recursive branch, producing `O(L × order)` allocations during the one-time construction of `MultiindexSet`. For `(NVAR=4, degree=7)` this is tens of thousands of transient vectors.

### 6 — `collect()` Calls on Hot-Path Adjacent Code *(MEDIUM)*
- `CohomologicalEquations.jl:623` — `collect(multi_i)` for each of L monomials during candidate precomputation
- `Polynomials.jl:129,487` — `collect(keys(dict))` in polynomial constructors
- `MultilinearTerms.jl:306` — `collect(Int, ext_idx)` per cache-build entry

### 7 — Matrix Copy in `PropagateEigenmodes.jl` Per Jordan Vector *(MEDIUM)*
**Location:** `PropagateEigenmodes.jl:60,135`  
```julia
tmp_mat = copy(linear_terms[1])
```
A full `FOM × FOM` matrix is copied to start each Horner evaluation of `P(λ)`. This is called per Jordan vector index. A pre-allocated scratch buffer would eliminate these copies.

### 8 — Missing Buffer Reuse in External Context Partial Assembly *(LOW)*
**Location:** `CohomologicalEquations.jl:638-640`  
```julia
zeros(ComplexF64, FOM, N_EXT)  # + hcat allocation
```
These one-time (per external-mode loop iteration) allocations can be folded into the context.

---

## Recommended Architecture Changes

### A — Expand `CohomologicalContext` to Hold All Reusable Buffers

`CohomologicalContext` already pre-allocates `system_matrix_buffer`, `rhs_buffer`, `lower_order_buffer`. Extend it to also carry:

| New field | Type | Replaces |
|---|---|---|
| `ml_result_buffer` | `Vector{T}(FOM)` | `result` in `compute_multilinear_terms` |
| `ml_scratch_buffer` | `Vector{T}(FOM)` | `scratch` in `compute_multilinear_terms` |
| `ml_temp_buffer` | `Vector{T}(FOM)` | `temp` in `_replay_split!` |
| `invariance_M_buffer` | `Matrix{T}(FOM, FOM+ROM)` | `M` in `assemble_cohomological_matrix_and_rhs` |
| `invariance_rhs_buffer` | `Vector{T}(FOM)` | `rhs` in same |
| `orth_M_buffer` | `Matrix{T}(ROM, FOM+ROM)` | `M` in orthogonality assembly |
| `orth_rhs_buffer` | `Vector{T}(ROM)` | `rhs` in same |
| `unit_vectors` | `NTuple{NVAR, SVector{NVAR,Int}}` | per-call reconstruction |

Pass the relevant buffer slices into `compute_multilinear_terms`, `assemble_cohomological_matrix_and_rhs`, and `assemble_orthogonality_matrix_and_rhs` as extra arguments. The public API of `solve_cohomological_problem` does not need to change.

### B — Cache LU Factorisation by Resonance Pattern

Group monomials that share the same `resonance::SVector{ROM,Bool}` value. For each group, the system matrix structure is the same. Reuse the LU object (or explicitly flag when a refactorisation is needed vs. when only `s` and the RHS change).

Concretely:
1. Before the main loop, cluster monomial indices by resonance pattern (the resonance set is already precomputed).
2. Within a cluster, compare consecutive `s` values; if the matrix changes, refactorise; otherwise only update the RHS.

This is the single largest algorithmic saving. For a typical model with few resonances, the vast majority of monomials share the non-resonant pattern (`nR = 0`), making the system matrix invariant up to the scalar `s` shift in the diagonal block.

> **Note:** When `nR = 0` the system is purely `L(s) W = rhs` with no R unknowns. In this case the operator `L(s)` can often be factorised symbolically (Horner precomputed into LU), reused across all non-resonant monomials and updated cheaply.

### C — Fix Multiindex Generation Allocations

Replace the recursive `vcat()` strategy in `all_multiindices_up_to` with a pre-allocated `Matrix{Int}` column buffer passed through the recursion by index. Each call writes into a pre-allocated column rather than allocating a new vector. This is a standard DFS-with-stack pattern.

### D — Accept `AbstractVector` / `SVector` in `indices_in_box_with_bounded_degree`

Removes the `collect(multi_i)` on every monomial during candidate precomputation (`CohomologicalEquations.jl:623`).

### E — Pre-allocate Horner Scratch in `PropagateEigenmodes`

Pass a single `FOM × FOM` scratch matrix through the two call sites (`propagate_right_eigenmode!` and `propagate_left_eigenmode!`) so the `copy()` is replaced by a `copyto!` into the pre-allocated buffer.

---

## Verification

After each change:
1. Run the demo: `julia demo/ParametrisationMethod/demo_parametrisation_method.jl` — output must match previous `output.h5` to machine precision.
2. Profile with `@profile` / `Profile.print()` or `BenchmarkTools.@benchmark` on `solve_cohomological_problem` to confirm allocations decrease.
3. Use `--track-allocation=user` to identify any remaining per-monomial heap activity.

---

## Priority Order

| # | Change | Effort | Expected gain |
|---|--------|--------|---------------|
| 1 | Expand context buffers (A) | Low — additive | Eliminates ~6 allocations/monomial |
| 2 | Fix `compute_multilinear_terms` to accept buffers | Low | Eliminates 3 allocs/monomial × L |
| 3 | Fix assembly functions to use context buffers | Medium | Eliminates 4 allocs/monomial × L |
| 4 | Cache LU by resonance pattern (B) | Medium | O(L × FOM³) → O(patterns × FOM³) |
| 5 | Fix multiindex generation (C) | Low | One-time construction cost |
| 6 | Accept SVector in `indices_in_box` (D) | Trivial | L `collect()` calls gone |
| 7 | Pre-allocate Horner scratch (E) | Trivial | Minor (one-time per eigenproblem) |
