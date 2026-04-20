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

## RHS of the Cohomological Equations — Specific Inefficiencies

The RHS `g_α` assembled for each monomial α has two sources: the **multilinear nonlinear terms** (evaluation of f(W(z)) at order α) and the **lower-order couplings** (the DW·R term). Both are computed in the hot loop.

### RHS-A — External Unit Vectors Recreated Per Monomial in Cached Path *(HIGH)*
**Location:** `MultilinearTerms.jl:375`  
```julia
unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0, external_system_size)) for j in 1:external_system_size]
```
This is inside `compute_multilinear_terms` (cached overload) and allocates a fresh vector every monomial, independently of the NVAR unit vectors already discussed. Should be stored in `CohomologicalContext` alongside the multilinear result/scratch/temp buffers (see change A).

### RHS-B — Scalar Inner Loop in Lower-Order Couplings Instead of BLAS `axpy!` *(HIGH)*
**Location:** `LowerOrderCouplings.jl:51-55` and `LowerOrderCouplings.jl:95-99`  
```julia
for k in 1:ORD
    acc_vec = accumulator[k]
    @inbounds for l in eachindex(acc_vec)
        acc_vec[l] += factor * param_coeff[l, k]
    end
end
```
This manual scalar loop is equivalent to `axpy!(factor, view(param_coeff, :, k), accumulator[k])` for each `k`. Using BLAS `axpy!` lets the runtime exploit SIMD and cache-blocking, especially for larger FOM. This is the innermost loop of the lower-order coupling computation, called for every candidate sub-monomial of every monomial.

### RHS-C — n_entries FEM Assemblies Per Monomial *(CRITICAL — large FOM / FEM)*
**Location:** `MultilinearTerms.jl` — `_accumulate_split!` / `_replay_split!`

When `t.f!` is a FEM assembly routine, each call traverses the entire mesh. The current design calls it once per factorisation entry — `n_entries` separate mesh traversals per (monomial, term, split):

```julia
for entry in split.entries          # n_entries mesh traversals
    t.f!(scratch, W[:,i1], W[:,i2], W[:,i3])
    axpy!(entry.multiplier, scratch, accum)
end
```

`n_entries` grows combinatorially with polynomial degree. Total cost: `O(L × n_entries_avg × n_elem × n_qp)`.

**Fix: restructure around element-local evaluation.**

In FEM the assembly is element-local:
```
result[r] += Σ_e Σ_q  w_q · |J_e| · φ_r(ξ_q)  · g(u_e(ξ_q), v_e(ξ_q), w_e(ξ_q))
```
where `u_e(ξ_q) = Σ_{a ∈ dofs(e)} u_a · φ_a(ξ_q)` is the local interpolation at the quadrature point. The sum over all factorisation entries can be pulled inside the element/quadrature loop:

```
field_e(ξ_q) = Σ_entries mult · g(W_{i1,e}(ξ_q), W_{i2,e}(ξ_q), W_{i3,e}(ξ_q))
```

This scalar field is then assembled once:
```
result[r] += Σ_e Σ_q  w_q · |J_e| · φ_r(ξ_q) · field_e(ξ_q)
```

The element loop runs **once per (monomial, term, split)** instead of `n_entries` times.

**New interface: `FEMMultilinearMap`**

Replace the single opaque `t.f!(result, args...)` with four primitive operations:

```julia
abstract type FEMMultilinearMap <: AbstractMultilinearMap end

# Iterate over elements
elements(t::FEMMultilinearMap)                         # → element iterator

# Scatter a DOF vector to quadrature-point values for one element
# Writes: W_qp[q] = Σ_{a ∈ dofs(e)} W_global[a] · φ_a(ξ_q)
scatter_to_qp!(W_qp::AbstractVector, W_global::AbstractVector,
               element, t::FEMMultilinearMap)

# Evaluate the pointwise nonlinear function at one quadrature point
# Returns a scalar: g(vals[1], vals[2], ..., vals[deg])
pointwise(vals::NTuple, element, qp::Int, t::FEMMultilinearMap) → Number

# Assemble a scalar field back to global DOFs for one element
# Adds: result[r] += Σ_q  w_q · |J_e| · φ_r(ξ_q) · field_qp[q]
assemble_element!(result::AbstractVector, field_qp::AbstractVector,
                  element, t::FEMMultilinearMap)
```

**Batched accumulation loop (replaces `_accumulate_split!` for FEM terms):**

```julia
function _accumulate_split_fem!(accum, W, t::FEMMultilinearMap,
                                  split, unique_cols, W_local_qp, field_qp)
    # unique_cols: pre-computed list of (order_k, monomial_idx_k) for this split
    # W_local_qp:  pre-allocated Matrix{T}(n_unique_cols, n_qp_per_elem)  [context buffer]
    # field_qp:    pre-allocated Vector{T}(n_qp_per_elem)                  [context buffer]

    for element in elements(t)

        # 1. Scatter each unique W column to quadrature-point values (element-local)
        for (k, (ord, col)) in enumerate(unique_cols)
            scatter_to_qp!(@view(W_local_qp[k, :]), @view(W[:, ord, col]), element, t)
        end

        # 2. Accumulate scalar field: all factorisation entries, all quadrature points
        fill!(field_qp, 0)
        for entry in split.entries
            local_indices = entry.local_unique_col_indices  # pre-computed in cache (see below)
            for q in eachindex(field_qp)
                vals = ntuple(k -> W_local_qp[local_indices[k], q], Val(deg))
                field_qp[q] += entry.multiplier * pointwise(vals, element, q, t)
            end
        end

        # 3. Assemble scalar field into global result (one element-level assembly)
        assemble_element!(accum, field_qp, element, t)
    end
end
```

**Cache change required:** `CachedSplit.entries` currently stores global monomial indices. Add `local_unique_col_indices::NTuple` to each `FactorisationEntry` (or a parallel field in `CachedSplit`) mapping each factor slot to its position in `unique_cols`. This is computed once during `build_multilinear_terms_cache`.

**Pre-allocated context buffers (added to `CohomologicalContext`):**
- `fem_W_local_qp::Matrix{T}(max_unique_cols, max_qp_per_elem)` — element-local mode values
- `fem_field_qp::Vector{T}(max_qp_per_elem)` — scalar field at quadrature points

**Backward compatibility:** `MultilinearMap` (the current closure type) retains the existing per-entry path unchanged. `FEMMultilinearMap` opts into the batched path.

**Expected gain:** FEM mesh traversal count drops from `n_entries` to `1` per split. For degree ≥ 3 with typical `n_entries = 10–100`, this is a **10–100× speedup** on the RHS evaluation, which is the dominant cost for large FOM.

### RHS-D — Fallback: Closure Maps Without Elemental Access *(STRUCTURAL)*

The current opaque `t.f!(result, x1, ..., xk)` cannot be batched because the solver has no access to element-level operations. For small FOM (analytical / toy systems), this path remains and should be kept as the default. The `FEMMultilinearMap` interface (RHS-C) is opt-in; users who do not implement it continue using the existing closure path with no changes.

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
