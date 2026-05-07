# MORFE_jl — Performance Optimisation Plan

## Context

The parametrisation method solves a cohomological equation for each of **L monomials** in graded-lexicographic order. The current implementation is architecturally sound (good precomputation, caches, fused Horner passes, in-place BLAS) but has several per-monomial allocation and redundant-work patterns that add up significantly as `L` grows with polynomial degree and number of variables. The demo system (`FOM=2, NVAR=3, degree=3`) is small, but typical engineering models have `FOM ≥ 10`, `NVAR ≥ 4`, `degree ≥ 5`, pushing `L` into the hundreds.

---

## Status Legend
- ✅ Done
- ❌ Not yet done
- ⚠️ Partially done

---

## Identified Inefficiencies

### 1 — LU Factorisation Rebuilt Every Monomial *(CRITICAL)* ❌
**Location:** `CohomologicalEquations.jl` — `solve_single_monomial!` line ~332  
```julia
F = lu!(view(ctx.system_matrix_buffer, 1:n_sys, 1:n_sys), check = false)  # called L times
```
The dense system matrix `A_sys` is factorised from scratch for every monomial even though the resonance pattern `resonance::SVector{ROM,Bool}` — which determines `nR` and the column structure — changes rarely (only when transitioning between degree shells or at isolated resonant monomials). For most monomials the matrix is identical up to the superharmonic scalar `s`.

**Note:** The sparse path already has symbolic factorisation caching via KLU/Pardiso. This gap applies to the **dense path only**.

**Cost:** `O(L × (FOM+nR)³)` — dominates for `FOM ≥ 5`.

### 2 — Per-Monomial Temporary Allocations in `compute_multilinear_terms` *(HIGH)* ✅
**Fixed.** The cached overload (`compute_multilinear_terms!` accepting a `MultilinearTermsCache`) retrieves `scratch_buffer`, `temp_buffer`, and `unit_vectors` directly from the cache struct — zero per-monomial allocations on the hot path.

The non-cached overload (convenience fallback) still allocates, but it is not called from the solve loop.

### 3 — System Assembly Matrices Allocated Per Monomial *(HIGH)* ✅
**Fixed.** Both `assemble_cohomological_matrix_and_rhs!` and `assemble_orthogonality_matrix_and_rhs!` now have in-place variants that accept pre-allocated `M` and `rhs` buffers from the caller (`solve_single_monomial!` passes `ctx.system_matrix_buffer`). The allocating variants are retained as convenience wrappers.

### 4 — Unit Vectors Reconstructed on Every Helper Call *(MEDIUM)* ⚠️
**Location:** `LowerOrderCouplings.jl:23-24` inside `_sum_degree_one_terms!` and `_sum_higher_degree_terms!`  
```julia
unit_vectors = [SVector{NVAR, Int}(ntuple(k -> k==j ? 1 : 0, Val(NVAR))) for j in 1:NVAR]
```
These are constants rebuilt on every call (i.e., every monomial). The `unit_vectors` field **has been added to `MultilinearTermsCache`** and is correctly used in `compute_multilinear_terms!`, but `LowerOrderCouplings.jl` still reconstructs its own copy independently. These should be passed from `CohomologicalContext` or `MultilinearTermsCache` instead.

### 5 — Recursive `vcat()` in Multiindex Generation *(MEDIUM)* ❌
**Location:** `Multiindices.jl:140,144`  
The `recurse` function inside `_generate_ascending_lex_fixed!` uses `vcat(prefix, e)` in every recursive branch, producing `O(L × order)` allocations during the one-time construction of `MultiindexSet`. Replace with a pre-allocated column buffer passed by index (standard DFS-with-stack pattern).

### 6 — `collect()` Call on Hot-Path Adjacent Code *(MEDIUM)* ❌
**Location:** `CohomologicalEquations.jl:777`
```julia
indices_in_box_with_bounded_degree(mset, collect(multi_i), 2, tdeg)
```
`collect(multi_i)` converts an `SVector` to a `Vector` for every monomial during candidate precomputation. Fix by making `indices_in_box_with_bounded_degree` accept `AbstractVector` / `SVector` directly.

**Other `collect` calls** (lower priority):
- `Polynomials.jl:129,487` — `collect(keys(dict))` in polynomial constructors
- `MultilinearTerms.jl:306` — `collect(Int, ext_idx)` per cache-build entry (setup-time only)

### 7 — Matrix Copy in `PropagateEigenmodes.jl` Per Jordan Vector *(MEDIUM)* ❌
**Location:** `PropagateEigenmodes.jl:60,135`  
```julia
tmp_mat = copy(linear_terms[1])
```
A full `FOM × FOM` matrix is copied to start each Horner evaluation of `P(λ)`. Pass a single pre-allocated scratch buffer through both call sites so `copy()` becomes `copyto!`.

### 8 — Missing Buffer Reuse in External Context Partial Assembly *(LOW)* ✅
**Fixed.** `external_rhs_buffer` and `ml_result_buffer` are pre-allocated in `CohomologicalContext`; no per-monomial `zeros(ComplexF64, FOM, N_EXT)` or `hcat` on the hot path.

---

## RHS of the Cohomological Equations — Specific Inefficiencies

### RHS-A — External Unit Vectors Recreated Per Monomial in Cached Path *(HIGH)* ✅
**Fixed.** `unit_vectors` is constructed once in `build_multilinear_terms_cache` and stored in `MultilinearTermsCache`. The cached overload of `compute_multilinear_terms!` reads it from the cache at line 572 of `MultilinearTerms.jl`.

### RHS-B — Scalar Inner Loop in Lower-Order Couplings Instead of BLAS `axpy!` *(HIGH)* ❌
**Location:** `LowerOrderCouplings.jl:51-56` and `LowerOrderCouplings.jl:95-99`  
```julia
for k in 1:ORD
    acc_vec = accumulator[k]
    @inbounds for l in eachindex(acc_vec)
        acc_vec[l] += factor * param_coeff[l, k]
    end
end
```
This manual scalar loop is equivalent to `axpy!(factor, view(param_coeff, :, k), accumulator[k])` for each `k`. Using BLAS `axpy!` exploits SIMD and cache-blocking for larger FOM. This is the innermost loop of the lower-order coupling computation, called for every candidate sub-monomial of every monomial.

### RHS-C — FEM Batched Assembly *(CRITICAL — large FOM / FEM)* ✅
**Implemented.** The `FEMMultilinearMap` abstract type is defined in `MultilinearMaps.jl` with the required interface (`fem_elements`, `fem_n_qp`, `scatter_qp!`, `accumulate_qp!`, `assemble_element!`, `fem_getdetJdV`, `fem_qp_buffer`). The batched path in `MultilinearTerms.jl` uses `FEMCachedSplit` / `FEMFactorisationEntry` structs and `_replay_fem_split!`, which traverses the mesh **once per (monomial, term, split)** rather than once per factorisation entry. Pre-allocated element-residual buffer `fem_Fe` lives in `MultilinearTermsCache`.

### RHS-D — Fallback: Closure Maps Without Elemental Access *(STRUCTURAL)* ✅
**Retained by design.** The opaque `MultilinearMap` closure path is unchanged and remains the default for small/analytical systems. `FEMMultilinearMap` is opt-in; users who do not implement it continue using the existing closure path with no changes.

---

## Recommended Architecture Changes

### A — Expand `CohomologicalContext` to Hold All Reusable Buffers ⚠️

Most buffers have been added. What remains:

| Field | Status | Notes |
|---|---|---|
| `lower_order_buffer` | ✅ Done | `Vector{Vector{T}}(ORD)`, zeroed before each call |
| `external_rhs_buffer` | ✅ Done | Pre-allocated FOM buffer |
| `ml_result_buffer` | ✅ Done | Pre-allocated FOM buffer |
| `ml_scratch_buffer` / `ml_temp_buffer` | ✅ Done | In `MultilinearTermsCache` |
| `invariance_M_buffer` / `invariance_rhs_buffer` | ✅ Done | Passed from `ctx.system_matrix_buffer` |
| `orth_M_buffer` / `orth_rhs_buffer` | ✅ Done | In-place variant exists |
| `unit_vectors` (multilinear) | ✅ Done | In `MultilinearTermsCache` |
| `unit_vectors` (lower-order couplings) | ❌ Todo | Still rebuilt in `LowerOrderCouplings.jl:23,73` |

**Remaining action:** pass the NVAR unit vectors into `_sum_degree_one_terms!` and `_sum_higher_degree_terms!` from the context instead of rebuilding them.

### B — Cache LU Factorisation by Resonance Pattern ❌

Dense path still refactorises every monomial. Plan:
1. Before the main loop, cluster monomial indices by resonance pattern (already precomputed).
2. Within a cluster, compare consecutive `s` values; refactorise only when the matrix changes.

For the non-resonant pattern (`nR = 0`, which covers the vast majority of monomials), the operator `L(s)` is constant up to the scalar `s` shift; a single LU suffices for all such monomials.

Sparse path (KLU/Pardiso) already has symbolic caching — this change targets dense only.

### C — Fix Multiindex Generation Allocations ❌

Replace the recursive `vcat()` strategy in `_generate_ascending_lex_fixed!` (`Multiindices.jl:140,144`) with a pre-allocated `Vector{Int}` column buffer mutated in-place through the recursion. One-time construction cost; not a hot-loop concern, but relevant for large `(NVAR, degree)`.

### D — Accept `AbstractVector` / `SVector` in `indices_in_box_with_bounded_degree` ❌

Removes the `collect(multi_i)` call at `CohomologicalEquations.jl:777`, eliminating L per-monomial `Vector` allocations during candidate precomputation.

### E — Pre-allocate Horner Scratch in `PropagateEigenmodes` ❌

Pass a single `FOM × FOM` scratch matrix through `propagate_right_jordan_vector` and `propagate_left_jordan_vector` (call sites at `PropagateEigenmodes.jl:60,135`) so `copy(linear_terms[1])` becomes `copyto!(scratch, linear_terms[1])`. This is called per Jordan vector index, not per monomial, so the impact is minor.

---

## Verification

After each change:
1. Run the demo: `julia demo/ParametrisationMethod/demo_parametrisation_method.jl` — output must match previous `output.h5` to machine precision.
2. Profile with `@profile` / `Profile.print()` or `BenchmarkTools.@benchmark` on `solve_cohomological_problem` to confirm allocations decrease.
3. Use `--track-allocation=user` to identify any remaining per-monomial heap activity.

---

## Priority Order (remaining work)

| # | Change | Effort | Expected gain |
|---|--------|--------|---------------|
| 1 | Cache LU by resonance pattern — dense path (B) | Medium | O(L × FOM³) → O(patterns × FOM³) |
| 2 | BLAS `axpy!` in lower-order couplings (RHS-B) | Low | SIMD speedup in innermost loop |
| 3 | Pass unit vectors to `LowerOrderCouplings` (A) | Low | Eliminates 2 small allocs/monomial |
| 4 | Accept SVector in `indices_in_box` (D) | Trivial | Eliminates L `collect()` calls |
| 5 | Pre-allocate Horner scratch (E) | Trivial | Minor (one-time per eigenproblem) |
| 6 | Fix multiindex generation vcat (C) | Low | One-time construction cost |
