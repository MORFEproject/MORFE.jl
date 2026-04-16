# MORFE.jl — Architecture & Performance Improvement Plan

## Context

The codebase implements the parametrisation method for computing Spectral Submanifolds (SSMs).
It is functionally correct for the non-sparse case but has accumulated several architectural
inconsistencies, unnecessary allocations in the per-monomial hot loop, and API surface issues.
This plan addresses all of them in a prioritised order: bugs first, hot-path performance second,
type-system cleanup third, API last.

---

## Group 1 — Bug Fixes

### 1.1 Undefined variable `N` in sparse `linear_first_order_matrices`

**File:** `src/FullOrderModel/FullOrderModel.jl:205, 214`

**Problem:** The sparse overload uses `N` (undefined) instead of `ORD` in two loop bounds:

```julia
for i in 1:(N - 1)   # line 205 — should be ORD - 1
for i in 1:N         # line 214 — should be ORD
```

**Fix:** Replace both `N` with `ORD`.

**Test:** Run `linear_first_order_matrices` on a sparse `NDOrderModel` (e.g., `SparseMatrixCSC`
linear terms) and verify A, B match the dense output.

---

## Group 2 — Hot-Path Performance (per-monomial loop)

These affect every call to `solve_single_monomial!` (called L times, where L = |mset|).

### 2.1 Use `MultilinearTermsCache` in the main solve loop

**Files:** `src/ParametrisationMethod/CohomologicalEquations.jl`,
`src/ParametrisationMethod/RightHandSide/MultilinearTerms.jl`

**Problem:** `solve_single_monomial!:287` calls `compute_multilinear_terms(model, multi, W)`
(non-cached path), which allocates per call:

- 3 × FOM-vectors (`result`, `scratch`, `temp`) — lines 167–169 of MultilinearTerms.jl
- N_EXT `SVector` unit vectors — line 173
- Fresh `candidate_indices` via `indices_in_box_with_bounded_degree` — line 174

`MultilinearTermsCache` already exists and eliminates all these allocations; it is just not
wired in.

**Fix:**

1. Build the cache once in `solve_cohomological_problem` before the main solve:
   ```julia
   ml_cache = build_multilinear_terms_cache(model, W)
   ```
2. Pass it through `solve_cohomological_equations!(W, R, ctx, model, ml_cache)` and down to
   `solve_single_monomial!`.
3. In `solve_single_monomial!:287`, replace:
   ```julia
   nonlinear_rhs = T.(compute_multilinear_terms(model, multi, W))
   ```
   with the cached variant (uses `idx` integer, not `multi` SVector):
   ```julia
   nonlinear_rhs = compute_multilinear_terms(model, idx, W, ml_cache)
   ```
4. Drop the `T.(...)` wrapper — the cached path already returns `T`-typed.

**Note:** The cache is built once on W with zero coefficients (only mset structure matters for
factorisation bookkeeping). It remains valid for the entire solve.

### 2.2 Pre-allocate buffers in `compute_lower_order_couplings`

**File:** `src/ParametrisationMethod/RightHandSide/LowerOrderCouplings.jl`

**Problem:** Every call (L times) allocates:

- `build_exponent_index_map(mset)` — a fresh `Dict{SVector{NVAR,Int},Int}` (line 119)
- `accumulator = [zeros(T, FOM) for _ in 1:ORD]` — ORD fresh FOM-vectors (line 125)
- `indices_in_box_with_bounded_degree(...)` inside `_sum_higher_degree_terms!` — fresh `Int[]`
  per call (line 70–71)

**Fix:**

1. Compute `multiindex_dict = build_exponent_index_map(mset)` once in
   `solve_cohomological_problem` and store it in `CohomologicalContext` as a new field
   `multiindex_dict::Dict{SVector{NVAR,Int},Int}`.
2. Add a pre-allocated accumulator `lower_order_buffer::Vector{Vector{T}}` (length ORD, each
   FOM) to `CohomologicalContext`. Zero it before each `compute_lower_order_couplings` call.
3. Update `compute_lower_order_couplings` signature to accept the dict and buffer; remove
   internal allocations. Return the buffer directly as a plain `Vector{Vector{T}}`.
4. For `indices_in_box_with_bounded_degree`: pre-build a per-monomial candidate index list
   during context construction (similar to how `MultilinearTermsCache` works). Store as
   `candidate_indices_by_monomial::Vector{Vector{Int}}` in context.

### 2.3 Pre-allocate the stacked system in `solve_single_monomial!`

**File:** `src/ParametrisationMethod/CohomologicalEquations.jl:317`

**Problem:**

```julia
sol = [M_inv; M_orth] \ [rhs_inv; rhs_orth]
```

`vcat` allocates new matrices every call. `nR = count(resonance)` varies per monomial, so the
system size changes.

**Fix:** Add to `CohomologicalContext`:

- `system_matrix_buffer::Matrix{T}` — pre-allocated `(FOM + ROM) × (FOM + ROM)` (worst-case
  nR = ROM)
- `rhs_buffer::Vector{T}` — pre-allocated `FOM + ROM`

At each monomial, write the blocks in-place using views into these buffers up to the actual
`(FOM + nR)` size, then call `\` on the view.

### 2.4 Eliminate per-call allocations in `evaluate_external_rhs!`

**File:** `src/ParametrisationMethod/InvarianceEquation.jl:531, 537`

**Problem:**

- `active = findall(!iszero, external_dynamics)` allocates a fresh `Int[]` per call (line 531)
- `g = zeros(T, FOM)` allocates a fresh FOM-vector per call (line 537)

**Fix:**

1. Replace `findall` with an inline loop over `eachindex(external_dynamics)`.
2. Add a `external_rhs_buffer::Vector{T}` (length FOM) to `CohomologicalContext`; pass it as
   an argument and `fill!(buffer, zero(T))` at the start. Update `evaluate_external_rhs!`
   signature to accept the buffer.

---

## Group 3 — Cold-Path Performance (one-time setup)

### 3.1 Remove redundant sort; precompute skip set

**File:** `src/ParametrisationMethod/CohomologicalEquations.jl:381, 385`

**Problem:** `solve_cohomological_equations!` does two redundant/repeated operations on every
call:

```julia
indices_by_order = sort(1:nterms, by = i -> sum(mset[i]))  # O(L log L) — always a no-op
skip_set = Set(_linear_monomial_indices(mset))              # O(NVAR) + findfirst × NVAR
```

The multiindex set is already stored in GrLex order (graded lex), so `sum(mset[i])` is
non-decreasing by construction — the `sort` is always a no-op. Remove it and just iterate
`1:nterms` directly. The skip set computation is not free and never changes for a given `mset`;
move it into `CohomologicalContext`.

**Fix:**

1. Delete the `sort` line; replace `indices_by_order` with plain `1:nterms`.
2. Store `linear_monomial_skip_set::Set{Int}` in `CohomologicalContext`; compute once during
   construction.
3. Fix `_linear_monomial_indices`: replace `zeros(Int, NVAR)` + mutation with `SVector`
   `setindex` to avoid heap allocation.

### 3.2 Single-pass context build (eliminate duplicate precompute)

**File:** `src/ParametrisationMethod/CohomologicalEquations.jl:537–588`

**Problem:** Two calls each to `precompute_column_polynomials` and
`precompute_orthogonality_column_polynomials` — the first with `Φ_ext = 0`, the second with the
solved `Φ_ext`. The master-mode C-coeffs and J-coeffs are identical in both passes.

**Why C_coeffs and J_coeffs are independent of `Φ_ext`:**

`J_coeffs` (`precompute_orthogonality_operator_coefficients`) takes only `fom_matrices`,
`left_eigenmodes`, and `master_eigenvalues` — no `generalised_right_eigenmodes` at all.

`C_coeffs` come from `precompute_column_polynomials`, whose Horner recurrence multiplies the
working buffer as `D ← D · Λ`. Since Λ is **upper triangular**:
`(D·Λ)[:,r] = Σ_{k≤r} D[:,k]·Λ[k,r]`. For master-mode columns `r ≤ ROM`, all `k ≤ r` are
also `≤ ROM`, so the master columns of D form a **closed subsystem** under the recurrence —
they never read external columns of D (i.e., never read Φ_ext). Therefore C_coeffs depend only
on `master_modes` and `Λ[1:ROM, 1:ROM]`.

**Note on E_coeffs:** They are NOT independent of `master_modes`. Because Λ has off-diagonal
blocks `Λ[1:ROM, ROM+1:NVAR]` (upper-right), the external columns of D·Λ are
`(D·Λ)[:,ROM+e] = Σ_{k≤ROM+e} D[:,k]·Λ[k,ROM+e]`, which includes master-column contributions
`D[:,1:ROM]`. Therefore E_coeffs depend on both `master_modes` AND `Φ_ext`, and cannot be
factored into a function of `Φ_ext` alone.

**Fix:** Split `precompute_column_polynomials` into two functions that share intermediate
master-column buffers:

- `precompute_master_column_polynomials(fom_matrices, master_modes, Λ)` → returns
  `(C_coeffs, D_master_steps)` where `D_master_steps[j]` = `D[:,1:ROM]` at Horner step `j`
  (FOM×ROM matrix saved at each step)
- `precompute_external_column_polynomials(fom_matrices, external_directions, Λ, D_master_steps)`
  → E_coeffs only, reusing the saved master intermediates instead of recomputing them

Then:

1. Before the external linear solve: compute `(C_coeffs, D_master_steps)` and `J_coeffs` once.
2. Build partial context with C_coeffs, J_coeffs, empty E_coeffs for the external monomial
   solve.
3. After solving: compute E_coeffs from `(fom_matrices, Φ_ext, Λ, D_master_steps)` — no
   master-mode work repeated.

This eliminates one full O(ORD·FOM·ROM) pass for C_coeffs and avoids re-running the master
Horner steps. The extra storage cost is `(ORD-1)` matrices of size FOM×ROM.

---

## Group 4 — Type System Cleanup

### 4.1 Replace `left_eigenmodes::SVector{ROM, Vector{T}}` with `Matrix{T}`

**Files:** `src/ParametrisationMethod/CohomologicalEquations.jl:465`,
`src/ParametrisationMethod/MasterModeOrthogonality.jl:230`

**Problem:** `SVector{ROM, Vector{ComplexF64}}` is a static array of heap-allocated dynamic
vectors. It's structurally inconsistent with `master_modes::Matrix{ComplexF64}` (FOM×ROM).

**Fix:**

- Change the argument type in `solve_cohomological_problem` and
  `precompute_orthogonality_operator_coefficients` to `AbstractMatrix{T}` (FOM × ROM).
- Update the demo: `left_eigenmodes = master_modes` (same matrix, used as-is).
- Inside `MasterModeOrthogonality`, replace `left_eigenmodes[r]` column access with
  `view(left_eigenmodes, :, r)`.

### 4.2 Unify `C_coeffs`/`E_coeffs`/`J_coeffs` storage type

**File:** `src/ParametrisationMethod/CohomologicalEquations.jl:184–189`

**Problem:** Mixed types in `CohomologicalContext`:

- `invariance_C_coeffs::Vector{Matrix{T}}` ← consistent
- `invariance_E_coeffs::Vector{Matrix{T}}` ← consistent
- `orthogonality_J_coeffs::NTuple{ROM, Matrix{T}}` ← **inconsistent** (NTuple while others are Vector)
- `orthogonality_C_coeffs::Vector{Matrix{T}}` ← consistent
- `orthogonality_E_coeffs::Vector{Matrix{T}}` ← consistent

**Fix:** Change `orthogonality_J_coeffs` to `Vector{Matrix{T}}`. Update
`precompute_orthogonality_operator_coefficients` return type and all downstream call sites (in
`MasterModeOrthogonality.jl` and `CohomologicalEquations.jl`).

### 4.3 Remove unnecessary `T.(...)` broadcast conversion

**File:** `src/ParametrisationMethod/CohomologicalEquations.jl:287`

**Problem:** `T.(compute_multilinear_terms(model, multi, W))` — redundant once 2.1 is done
(cached path already returns correct type).

**Fix:** After 2.1 is implemented, remove the `T.(...)` wrapper.

### 4.4 Remove unnecessary `Vector{ComplexF64}(...)` conversion on SVector eigenvalues

**File:** `src/ParametrisationMethod/CohomologicalEquations.jl:474–475`

**Problem:** `Vector{ComplexF64}(model.external_system.eigenvalues)` converts `SVector → Vector`
unnecessarily. Downstream code uses it in simple indexed access, which works on both.

**Fix:** Remove the conversion; use the `SVector` directly.

---

## Group 5 — API Cleanup

### 5.1 Fix `MORFE.jl` exports

**File:** `src/MORFE.jl`

**Problems:**

- `resonance_set_from_graph_style`, `resonance_set_from_complex_normal_form_style`,
  `resonance_set_from_real_normal_form_style` are not re-exported at top level — the demo
  imports them directly with `using .MORFE.Resonance:...`.
- `CohomologicalContext` is exported but is an internal implementation detail; user-facing API
  is `solve_cohomological_problem`.

**Fix:** Add resonance constructors to the top-level `using .Resonance:` re-export block.
Remove `CohomologicalContext` from top-level exports (keep it exported within the
`CohomologicalEquations` submodule for advanced users).

### 5.2 Remove `PropagateEigenmodes` dead stub

**File:** `src/SpectralDecomposition/PropagateEigenmodes.jl`, `src/MORFE.jl:15, 32`

**Problem:** `PropagateEigenmodes` is an empty module (3 lines), included and re-exported,
contributing nothing.

**Fix:** Delete the file; remove include and export lines from `MORFE.jl`.

### 5.3 Add resonance-style selection guidance

**File:** `src/ParametrisationMethod/Resonance.jl`

**Problem:** No guidance on when to use graph vs CNF vs real NF vs condition-number styles.

**Fix:** Add a module-level docstring section "Choosing a resonance style" with 4–6 lines
covering: graph style (default for non-autonomous SSMs), CNF (for autonomous ROMs), real NF
(real coefficient systems), condition-number (when near-resonance detection is uncertain).

### 5.4 Clean up demo

**File:** `demo/ParametrisationMethod/demo_parametrisation_method.jl`

**Problems:**

- Lines 160–161: dead computation `s = ...; M = ...` (never used)
- Lines 119–122: `left_eigenmodes` placeholder using right modes — should match updated type
  (Matrix) after 4.1

**Fix:** Remove lines 160–161. After 4.1, simplify `left_eigenmodes = master_modes`.

---

## Execution Schedule

| #   | Task                                                | Depends on | Effort  |
| --- | --------------------------------------------------- | ---------- | ------- |
| 1.1 | Fix `N` → `ORD` in sparse FullOrderModel            | —          | trivial |
| 4.2 | Unify J_coeffs to Vector                            | —          | small   |
| 4.1 | `left_eigenmodes` → Matrix                          | 4.2        | small   |
| 2.1 | Wire `MultilinearTermsCache` into main solve        | —          | small   |
| 2.2 | Pre-allocate in `compute_lower_order_couplings`     | —          | medium  |
| 2.3 | Pre-allocate stacked system buffer                  | —          | small   |
| 2.4 | Pre-allocate `evaluate_external_rhs!` buffer        | —          | small   |
| 3.1 | Remove sort; precompute skip set into context       | —          | small   |
| 3.2 | Single-pass context build                           | —          | medium  |
| 4.3 | Remove `T.(...)` wrapper                            | 2.1        | trivial |
| 4.4 | Remove `Vector(eigenvalues)` conversion             | —          | trivial |
| 5.1 | Fix exports                                         | —          | trivial |
| 5.2 | Remove PropagateEigenmodes stub                     | —          | trivial |
| 5.3 | Add resonance style guidance                        | —          | small   |
| 5.4 | Clean up demo                                       | 4.1, 4.3   | trivial |

**Execution order:** 1.1 → 4.2 → 4.1 → 2.1 → 2.2 → 2.3 → 2.4 → 3.1 → 3.2 → 4.3 → 4.4 → 5.1 → 5.2 → 5.3 → 5.4

Start with type fixes (4.2, 4.1) because they change function signatures that performance fixes
depend on. Then do hot-path fixes in order of impact (2.1 is highest, requires no
preconditions).

---

## Verification Schedule

**Rule:** after each task, run every demo listed for that task plus its tests (if any). Fix
regressions before proceeding. The full-pipeline demo is always included as a final gate.

**Numerical invariants** (checked after every task via `demo_parametrisation_method.jl`):

- Linear monomial W/R coefficients match eigenvalue/eigenvector values
- Degree-2 W coefficients = 0 (cubic-only model, no quadratic terms)
- Degree-3 R coefficients non-zero, magnitude unchanged

| Task | Demos to run                                                                                                                 | Extra checks                                                                                     |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1.1  | `demo/FullOrderModel/demo_NDOrderModel.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl`                   | Construct sparse `NDOrderModel`; compare A,B to dense output — must be equal.                   |
| 4.2  | `demo/ParametrisationMethod/demo_master_mode_orthogonality.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl` | J_coeffs values unchanged from NTuple variant.                                                   |
| 4.1  | `demo/ParametrisationMethod/demo_master_mode_orthogonality.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl` | Demo `left_eigenmodes = master_modes`; no type error; J_coeffs unchanged.                        |
| 2.1  | `demo/ParametrisationMethod/demo_multilinear_terms.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl`       | `@allocated solve_cohomological_problem(...)` reduced; nonlinear RHS values identical.           |
| 2.2  | `demo/ParametrisationMethod/demo_lower_order_couplings.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl`   | `@allocated` further reduced; lower-order coupling vectors unchanged.                            |
| 2.3  | `demo/ParametrisationMethod/demo_parametrisation_method.jl`                                                                 | `@allocated` further reduced; solved W,R coefficients identical.                                 |
| 2.4  | `demo/ParametrisationMethod/demo_invariance_equation.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl`     | `@allocated` further reduced; external RHS contribution unchanged.                               |
| 3.1  | `demo/ParametrisationMethod/demo_parametrisation_method.jl`                                                                 | Skip set contains exactly the NVAR linear monomial indices; iteration order unchanged.            |
| 3.2  | `demo/ParametrisationMethod/demo_invariance_equation.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl`     | C_coeffs, E_coeffs, J_coeffs match previous two-call output at several monomials.               |
| 4.3  | `demo/ParametrisationMethod/demo_multilinear_terms.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl`       | No type error; results unchanged.                                                                |
| 4.4  | `demo/ParametrisationMethod/demo_parametrisation_method.jl`                                                                 | No type error; external eigenvalue indexing unchanged.                                           |
| 5.1  | All demos (exports change affects every `using .MORFE.*`)                                                                    | Demo no longer needs `using .MORFE.Resonance:...`; `CohomologicalContext` not in top namespace. |
| 5.2  | `demo/Eigensolver/demo_propagation.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl`                       | No include/export errors after file deletion.                                                    |
| 5.3  | `demo/ParametrisationMethod/demo_resonances.jl`<br>`demo/ParametrisationMethod/demo_parametrisation_method.jl`              | Docstring renders correctly; resonance set unchanged.                                            |
| 5.4  | `demo/ParametrisationMethod/demo_parametrisation_method.jl`                                                                 | Dead lines removed; `left_eigenmodes = master_modes`; demo output unchanged.                    |
