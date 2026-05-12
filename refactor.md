# Refactoring plan: `CohomologicalEquations.jl`, `InvarianceEquation.jl`, `MasterModeOrthogonality.jl`

## Diagnosis

All three files are oversized because they each carry multiple unrelated concerns:

| File | Lines | Concerns bundled together |
|:-----|------:|:--------------------------|
| `CohomologicalEquations.jl` | 872 | (1) 22-field god-object struct, (2) dense+sparse solver implementations, (3) buffer-allocation helpers, (4) 200-line high-level driver |
| `InvarianceEquation.jl` | 979 | (1) operator precomputation (dense Horner), (2) sparse Horner + template management, (3) column/external polynomial evaluation, (4) three near-duplicate assembly overloads |
| `MasterModeOrthogonality.jl` | 811 | (1) row-operator precomputation, (2) joint-operator precomputation, (3) fused row Horner evaluation, (4) column/external polynomial evaluation, (5) allocating + in-place assembly overloads |

`InvarianceEquation.jl` and `MasterModeOrthogonality.jl` are structurally parallel and
receive identical treatment.  The split is summarised in the table below; §3 gives details
for each file.

| Responsibility | `InvarianceEquation` | `MasterModeOrthogonality` |
|:---------------|:---------------------|:--------------------------|
| Operator evaluation via Horner | `HornerEvaluator.jl` | `HornerEvaluator.jl` |
| Precomputation + column/external evaluation | `ColumnPolynomials.jl` | `ColumnPolynomials.jl` |
| Assembly (module file, trimmed) | `InvarianceEquation.jl` | `MasterModeOrthogonality.jl` |

The root cause in `CohomologicalEquations.jl` is `CohomologicalContext`: a single flat
struct with 22 fields spanning six conceptually distinct categories of data:

| Category | Current field names |
|:---------|:--------------------|
| Spectral / model data | `linear_terms`, `generalised_eigenmodes`, `lambda_diag` |
| Invariance operators | `invariance_C_coeffs`, `invariance_E_coeffs` |
| Orthogonality operators | `orthogonality_J_coeffs`, `orthogonality_C_coeffs`, `orthogonality_E_coeffs` |
| Resonance bookkeeping | `resonance_set`, `linear_monomial_skip_set` |
| Lower-order coupling | `multiindex_dict`, `lower_order_buffer`, `candidate_indices_by_monomial`, `unit_vectors` |
| Solve buffers + sparse state | `system_matrix_buffer`, `rhs_buffer`, `external_rhs_buffer`, `ml_result_buffer`, `sparse_L_template`, `sparse_L_mappings`, `pardiso_solver`, `klu_cache` |

---

## 1. Target folder structure

Three new subdirectories are introduced under `src/ParametrisationMethod/`.  Each
contains the module file that `MORFE.jl` already includes by name, plus the extracted
sub-files that module `include`s internally.  The two files per module are named
identically across both operator modules — the shared naming makes the parallel
structure explicit.

```
src/ParametrisationMethod/
│
├── ParametrisationMethod.jl              # unchanged
├── Resonance.jl                          # unchanged
│
├── RightHandSide/                        # existing — unchanged
│   ├── LowerOrderCouplings.jl
│   └── MultilinearTerms.jl
│
├── InvarianceEquation/                   # NEW folder
│   ├── InvarianceEquation.jl             # module file (trimmed to ~60 lines)
│   ├── HornerEvaluator.jl                # L(s) dense + sparse Horner evaluation
│   └── ColumnPolynomials.jl              # C/E precomputation + column/external evaluation
│
├── MasterModeOrthogonality/              # NEW folder
│   ├── MasterModeOrthogonality.jl        # module file (trimmed to ~60 lines)
│   ├── HornerEvaluator.jl                # L_r(s) row Horner evaluation + scalar lower-order RHS
│   └── ColumnPolynomials.jl              # J/C/E precomputation + column/external evaluation
│
└── CohomologicalEquations/               # NEW folder
    ├── CohomologicalEquations.jl         # module file (public API, ~180 lines)
    ├── OperatorData.jl                   # InvarianceOperators, OrthogonalityOperators structs
    ├── SolverResources.jl                # LowerOrderResources, CohomologicalBuffers, SparseLinearSolverState
    ├── CohomologicalContext.jl           # CohomologicalContext struct (composed from the above)
    ├── CohomologicalSolver.jl            # _sparse_solve, _solve_monomial! (dense + sparse)
    └── CohomologicalDriver.jl            # solve_cohomological_problem + private sub-functions
```

### How the include chain changes

**`src/MORFE.jl`** — three lines change, everything else is untouched:
```julia
# Before:
include("ParametrisationMethod/InvarianceEquation.jl")
include("ParametrisationMethod/MasterModeOrthogonality.jl")
include("ParametrisationMethod/CohomologicalEquations.jl")

# After:
include("ParametrisationMethod/InvarianceEquation/InvarianceEquation.jl")
include("ParametrisationMethod/MasterModeOrthogonality/MasterModeOrthogonality.jl")
include("ParametrisationMethod/CohomologicalEquations/CohomologicalEquations.jl")
```

**`InvarianceEquation/InvarianceEquation.jl`** (module file):
```julia
module InvarianceEquation
  include("HornerEvaluator.jl")      # path is relative to this file's directory
  include("ColumnPolynomials.jl")
  # ... assemble_cohomological_matrix_and_rhs! (in-place only)
  export ...
end
```

**`MasterModeOrthogonality/MasterModeOrthogonality.jl`** (module file):
```julia
module MasterModeOrthogonality
  include("HornerEvaluator.jl")
  include("ColumnPolynomials.jl")
  # ... assemble_orthogonality_matrix_and_rhs! (in-place only)
  export ...
end
```

**`CohomologicalEquations/CohomologicalEquations.jl`** (module file):
```julia
module CohomologicalEquations
  # ... using declarations ...
  include("OperatorData.jl")
  include("SolverResources.jl")
  include("CohomologicalContext.jl")
  include("CohomologicalSolver.jl")
  include("CohomologicalDriver.jl")
  # ... solve_single_monomial!, solve_cohomological_equations!, helpers
  export ...
end
```

Julia resolves each `include()` path relative to the file containing the call, so
sub-files need no knowledge of where they sit in the project tree.

---

## 2. New data structures

Replace the one flat `CohomologicalContext` struct with a four-layer composition.

### 2a. `InvarianceOperators{T}` — lives in `OperatorData.jl`
```julia
struct InvarianceOperators{T}
    C_coeffs::Vector{Matrix{T}}   # length ROM,   each FOM × ORD
    E_coeffs::Vector{Matrix{T}}   # length N_EXT, each FOM × ORD
end
```
Owns the precomputed column-polynomial coefficients produced by
`precompute_master_column_polynomials` / `precompute_external_column_polynomials`.
Constructor wraps the two-pass precomputation so callers never see `D_master_steps`:
```julia
function InvarianceOperators(linear_terms, master_modes, external_directions, Λ, ROM)
    C, D_steps = precompute_master_column_polynomials(linear_terms, master_modes, view(Λ, 1:ROM, 1:ROM))
    E = precompute_external_column_polynomials(linear_terms, external_directions, Λ, D_steps)
    return InvarianceOperators(C, E)
end
```

### 2b. `OrthogonalityOperators{T}` — lives in `OperatorData.jl`
```julia
struct OrthogonalityOperators{T}
    J_coeffs::Vector{Matrix{T}}   # length ROM, each ORD × FOM
    C_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × ROM
    E_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × N_EXT
end
```
Owns the three coefficient families from
`precompute_orthogonality_operator_coefficients` /
`precompute_orthogonality_column_polynomials`.

### 2c. `LowerOrderResources{NVAR, T}` — lives in `SolverResources.jl`
```julia
struct LowerOrderResources{NVAR, T}
    multiindex_dict::Dict{SVector{NVAR, Int}, Int}
    buffer::Vector{Vector{T}}               # length ORD, each FOM; zeroed per monomial
    candidate_indices::Vector{Vector{Int}}  # length L; precomputed per monomial
    unit_vectors::Vector{SVector{NVAR, Int}}
end
```
Constructor absorbs the `candidate_indices_by_monomial` loop currently inlined
in `solve_cohomological_problem`:
```julia
function LowerOrderResources{NVAR, T}(mset, ORD, FOM) where {NVAR, T}
    L = length(mset)
    candidates = [ tdeg < 2 ? Int[] : indices_in_box_with_bounded_degree(mset, mset[i], 2, tdeg)
                   for (i, tdeg) in enumerate(sum.(mset.exponents)) ]
    LowerOrderResources{NVAR, T}(
        build_exponent_index_map(mset),
        [zeros(T, FOM) for _ in 1:ORD],
        candidates,
        [SVector{NVAR,Int}(ntuple(k -> k==j ? 1 : 0, Val(NVAR))) for j in 1:NVAR],
    )
end
```

### 2d. `SparseLinearSolverState{T}` — lives in `SolverResources.jl`
```julia
struct SparseLinearSolverState{T}
    L_template::SparseMatrixCSC{T}
    L_mappings::Vector{Vector{Int}}
    pardiso::Union{Nothing, AbstractPardisoSolver}
    klu_cache::Ref{Any}             # Ref(nothing) until first factorisation
    rhs_extended::Matrix{T}         # FOM × (ROM+1); pre-allocated for the Schur hcat
end
```
Constructor calls `precompute_sparse_L_template`, `_alloc_pardiso_solver`, and
`_alloc_klu_cache` — currently scattered across three helpers in
`CohomologicalEquations.jl`.  The `rhs_extended` field eliminates the
`hcat(Vector(rhs), C_r)` allocation on every resonant monomial in the sparse path.

> **Cross-module dependency**: `precompute_sparse_L_template` is defined in
> `InvarianceEquation` and must remain there (it belongs to the Horner evaluator
> for `L(s)`).  `CohomologicalEquations/SolverResources.jl` therefore needs
> `using ..InvarianceEquation: precompute_sparse_L_template` in its module preamble.
> Alternatively, the call can be kept in the outer `CohomologicalContext` constructor
> and the resulting `(template, mappings)` pair passed into `SparseLinearSolverState`
> as plain arguments, keeping `SolverResources.jl` dependency-free.

### 2e. `CohomologicalBuffers{T}` — lives in `SolverResources.jl`
```julia
struct CohomologicalBuffers{T}
    system_matrix::Matrix{T}   # (FOM+ROM)×(FOM+ROM) dense OR (ROM+1)×(ROM+1) Schur
    rhs::Vector{T}             # length FOM+ROM
    external_rhs::Vector{T}    # length FOM
    ml_result::Vector{T}       # length FOM
end
```
Constructor absorbs `_alloc_system_buffer` (currently dispatching on `MT`).

### 2f. `CohomologicalContext` (10 named fields) — lives in `CohomologicalContext.jl`
```julia
struct CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT}
    # Spectral / model data
    linear_terms::NTuple{ORDP1, MT}
    generalised_eigenmodes::Matrix{T}
    lambda_diag::Vector{T}
    # Precomputed operators
    invariance::InvarianceOperators{T}
    orthogonality::OrthogonalityOperators{T}
    # Resonance bookkeeping
    resonance_set::ResonanceSet
    linear_monomial_skip_set::Set{Int}
    # Compute resources
    lower_order::LowerOrderResources{NVAR, T}
    buffers::CohomologicalBuffers{T}
    sparse_solver::Union{Nothing, SparseLinearSolverState{T}}   # nothing on dense path
end
```
**22 flat fields → 10 named fields.**  Every call site becomes self-documenting:
`ctx.invariance.C_coeffs` replaces `ctx.invariance_C_coeffs`;
`ctx.buffers.rhs` replaces `ctx.rhs_buffer`;
`ctx.sparse_solver.pardiso` replaces `ctx.pardiso_solver`.

---

## 3. What goes in each file

### `InvarianceEquation/HornerEvaluator.jl`

- `evaluate_system_matrix_and_lower_order_rhs!` — dense fused Horner pass for `L(s)`
- `precompute_sparse_L_template` — build union-pattern template + index mappings
- `build_sparse_L_and_rhs!` — sparse fused Horner pass for `L(s)`

These three form a self-contained unit: "evaluate the parametrisation operator
polynomial via Horner, accumulating the lower-order RHS simultaneously."

### `InvarianceEquation/ColumnPolynomials.jl`

- `precompute_column_polynomials` (kept as a shim for backward compat)
- `precompute_master_column_polynomials`
- `precompute_external_column_polynomials`
- `evaluate_column!`
- `evaluate_external_rhs!` (in-place variant with pre-allocated buffer only)

### `InvarianceEquation/InvarianceEquation.jl` (trimmed module)

Retains only:
- `assemble_cohomological_matrix_and_rhs!` (in-place; the only variant called in `_solve_monomial!`)
- Module `using` / `include` / `export` declarations

The two allocating overloads `assemble_cohomological_matrix_and_rhs` (lines 766–833)
and the allocating `evaluate_external_rhs!` overload (lines 688–697) are **deleted**:
they are not called anywhere inside the library.

Target: ≈ 60 lines.

---

### `MasterModeOrthogonality/HornerEvaluator.jl`

- `evaluate_orthogonality_row_and_lower_order_rhs!` — fused Horner pass for the row
  operator `L_r(s)` and the scalar lower-order RHS contribution.

The function is structurally parallel to `evaluate_system_matrix_and_lower_order_rhs!`
in `InvarianceEquation`: it traverses the same kind of Horner recurrence while
accumulating a scalar dot-product instead of a dense matrix-vector product.

### `MasterModeOrthogonality/ColumnPolynomials.jl`

- `precompute_orthogonality_operator_coefficients` — builds `J_coeffs` (row operators `L_r(s)`)
- `precompute_orthogonality_column_polynomials` — builds `C_coeffs`, `E_coeffs` (joint operator)
- `evaluate_orthogonality_column_row!` — evaluates the resonant block of `C_r(s)` into one matrix row
- `evaluate_orthogonality_external_rhs` — computes the scalar external-forcing RHS for mode `r`

**Minor cleanup during extraction**: `evaluate_orthogonality_external_rhs` currently
calls `findall(!iszero, external_dynamics)` (allocates a temporary index vector).
Replace with an inline scan identical to the pattern already used in
`evaluate_external_rhs!` in `InvarianceEquation`, eliminating a small but unnecessary
allocation on every resonant monomial call.

### `MasterModeOrthogonality/MasterModeOrthogonality.jl` (trimmed module)

Retains only:
- `assemble_orthogonality_matrix_and_rhs!` (in-place; the only variant called in `_solve_monomial!`)
- Module `using` / `include` / `export` declarations

The allocating overload `assemble_orthogonality_matrix_and_rhs` (lines 733–770) is
**deleted**: it is not called inside the library.

Target: ≈ 60 lines.

---

### `CohomologicalEquations/OperatorData.jl`

- Struct definitions: `InvarianceOperators{T}`, `OrthogonalityOperators{T}`
- Their constructors (described in §2a–2b above)

### `CohomologicalEquations/SolverResources.jl`

- Struct definitions: `LowerOrderResources{NVAR,T}`, `CohomologicalBuffers{T}`,
  `SparseLinearSolverState{T}`
- Their constructors (absorb `_alloc_system_buffer`, `_alloc_pardiso_solver`,
  `_alloc_klu_cache`, `precompute_sparse_L_template`)

### `CohomologicalEquations/CohomologicalContext.jl`

- Struct definition of `CohomologicalContext` (composed from the sub-structs)
- A single outer constructor with a clear parameter list

### `CohomologicalEquations/CohomologicalSolver.jl`

- `_sparse_solve` (three dispatch methods: Pardiso / KLU / dense fallback)
- `_solve_monomial!` dense path
- `_solve_monomial!` sparse path

**Data-flow improvement**: replace `RHS_mat = hcat(Vector(rhs), C_r)` (CE.jl line 377)
with an in-place write into `ctx.sparse_solver.rhs_extended`, eliminating the last
per-monomial allocation on the sparse resonant path.

### `CohomologicalEquations/CohomologicalDriver.jl`

- `solve_cohomological_problem` broken into three private helpers:
  - `_initialise_waveform!(W, R, master_modes, master_eigenvalues, master_modes_derivatives, unit_offset, model)` — create W/R, init linear monomials, embed external dynamics
  - `_solve_external_directions!(W, R, partial_ctx, model, ml_cache, N_EXT, unit_offset)` — partial-context external monomial loop
  - `_build_context(linear_terms, eigenmodes, lambda_diag, ...) -> CohomologicalContext` — operator precomputation + context assembly via sub-struct constructors

After extraction, `solve_cohomological_problem` becomes a ≈ 40-line orchestrator:
```julia
function solve_cohomological_problem(model, mset, ...)
    W, R = _initialise_waveform!(...)
    ml_cache = build_multilinear_terms_cache(model, W)
    partial_ctx = _build_context(linear_terms, partial_eigenmodes, ...)
    _solve_external_directions!(W, R, partial_ctx, model, ml_cache, N_EXT, unit_offset)
    ctx = _build_context(linear_terms, full_eigenmodes, ...)
    solve_cohomological_equations!(W, R, ctx, model, ml_cache)
    return W, R
end
```

### `CohomologicalEquations/CohomologicalEquations.jl` (trimmed public API)

Retains:
- `_embed_external_dynamics!`
- `_linear_monomial_indices`
- `_resonance_vector`
- `solve_single_monomial!`
- `solve_cohomological_equations!`
- Module `using`, `include`, and `export` declarations

Target: ≈ 180 lines.

---

## 4. Redundancy and dead code to remove

| Item | Source location | Action |
|:-----|:----------------|:-------|
| `assemble_cohomological_matrix_and_rhs` (2 allocating overloads) | IE.jl lines 766–833 | Delete |
| `evaluate_external_rhs!` allocating overload | IE.jl lines 688–697 | Delete |
| `assemble_orthogonality_matrix_and_rhs` (allocating overload) | MMO.jl lines 733–770 | Delete |
| `findall(!iszero, ...)` in `evaluate_orthogonality_external_rhs` | MMO.jl line 640 | Replace with inline scan (no alloc) |
| `_alloc_system_buffer` (two dispatch methods) | CE.jl lines 569–578 | Fold into `CohomologicalBuffers` constructor |
| `_alloc_sparse_L_data` (two dispatch methods) | CE.jl lines 583–586 | Fold into `SparseLinearSolverState` constructor |
| `_alloc_pardiso_solver` (two dispatch methods) | CE.jl lines 589–605 | Fold into `SparseLinearSolverState` constructor |
| `_alloc_klu_cache` (two dispatch methods) | CE.jl lines 609–610 | Fold into `SparseLinearSolverState` constructor |
| `hcat(Vector(rhs), C_r)` per-monomial allocation | CE.jl line 381 | Replace with `rhs_extended` field in `SparseLinearSolverState` |

---

## 5. Migration order

Each step compiles and passes tests before the next begins.

1. **Create `InvarianceEquation/` folder.** Move the existing file there as
   `InvarianceEquation/InvarianceEquation.jl`; stub out the two `include` calls.
   Update `MORFE.jl` path.  Tests pass — no logic change.

2. **Create `InvarianceEquation/HornerEvaluator.jl`.** Cut
   `evaluate_system_matrix_and_lower_order_rhs!`, `precompute_sparse_L_template`,
   `build_sparse_L_and_rhs!` from the module file.

3. **Create `InvarianceEquation/ColumnPolynomials.jl`.** Cut the five precomputation /
   evaluation functions.  Delete the two dead allocating overloads.

4. **Create `MasterModeOrthogonality/` folder.** Move the existing file there as
   `MasterModeOrthogonality/MasterModeOrthogonality.jl`; stub out the two `include`
   calls.  Update `MORFE.jl` path.

5. **Create `MasterModeOrthogonality/HornerEvaluator.jl`.** Cut
   `evaluate_orthogonality_row_and_lower_order_rhs!`.

6. **Create `MasterModeOrthogonality/ColumnPolynomials.jl`.** Cut the four
   precomputation / evaluation functions.  Fix the `findall` allocation.
   Delete the dead allocating assembly overload.

7. **Create `CohomologicalEquations/` folder.** Move the existing file there; update
   `MORFE.jl` path.

8. **Create `OperatorData.jl` and `SolverResources.jl`.** Define the five new structs
   with their constructors.

9. **Create `CohomologicalContext.jl`.** Rewrite the struct using sub-structs.  Update
   all field accesses: `ctx.invariance.C_coeffs`, `ctx.buffers.rhs`,
   `ctx.sparse_solver.pardiso`, etc.

10. **Create `CohomologicalSolver.jl`.** Move `_sparse_solve` and both `_solve_monomial!`
    variants.  Apply the `rhs_extended` buffer optimisation.

11. **Create `CohomologicalDriver.jl`.** Extract `solve_cohomological_problem` and its
    three private sub-functions.  Delete the four dead `_alloc_*` helpers.

12. **Trim `CohomologicalEquations/CohomologicalEquations.jl`** to the public API +
    `include`/`export` declarations only.

13. **Audit and clean**: verify line counts match targets, run the full test suite, check
    that no public symbol has been accidentally un-exported.

---

## 6. Invariants to preserve

- The public API (`solve_cohomological_problem`, `solve_cohomological_equations!`,
  `solve_single_monomial!`, `CohomologicalContext`) stays in the
  `CohomologicalEquations` module namespace — existing call sites require no changes.
- All public symbols from `InvarianceEquation` and `MasterModeOrthogonality` remain
  exported from those module namespaces — downstream `using` statements are unaffected.
- Type parameters of `CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT}` are
  unchanged; sub-struct constructors are type-inferred, not manually annotated.
- No allocations are introduced on the per-monomial hot path; two are removed
  (`hcat` in the sparse resonant path; `findall` in `evaluate_orthogonality_external_rhs`).
- Dense/sparse dispatch continues to resolve at compile time via `MT <: SparseMatrixCSC`.
