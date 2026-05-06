# MORFE.jl Profiling Analysis — Detailed

**Source**: `profile_morfe.html` (ProfileCanvas, Julia profiler)  
**Entry point**: `solve_cohomological_problem` — `CohomologicalEquations.jl:515`  
**Total samples**: 15,596 across 2 threads  
**Benchmark**: `benchmark_morfe20.jl`

---

## 1. Overall cost distribution

| Rank | Leaf function | Source | Samples | % Total |
| ---- | ------------- | ------ | -------: | -------: |
| 1 | `getrf!` (LAPACK LU kernel) | `lapack.jl:565` | 9,810 | **62.9%** |
| 2 | `_unsafe_getindex` (slice copy) | `multidimensional.jl:903` | 1,465 | **9.4%** |
| 3 | `copyto!` (system matrix init) | `InvarianceEquation.jl:508` | 1,389 | **8.9%** |
| 4 | `materialize!` (stacking .= copies) | `broadcast.jl` | 469 | **3.0%** |
| 5 | `materialize!` (broadcast in assembly) | `broadcast.jl` | 581 | **3.7%** |
| 6 | `rmul!` (scalar × matrix) | `generic.jl:182` | 417 | **2.7%** |
| 7 | `ldiv!` (triangular solve) | `lu.jl:430` | 180 | **1.2%** |
| 8 | `mul!` (matrix-vector) | `matmul.jl:71` | 141 | **0.9%** |
| 9 | `chkfinite` (NaN guard) | `lapack.jl:88` | 95 | **0.6%** |
| 10 | `assembly_G!` (quadratic FEM) | `assembler.jl:126` | 116 | **0.7%** |
| 11 | `assembly_H!` (cubic FEM) | `assembler.jl:27` | 59 | **0.4%** |

---

## 2. Call-tree structure

```
solve_cohomological_problem          [15,539 | 99.6%]
└─ solve_cohomological_equations!    [14,939 | 95.8%]   ← main loop over monomials
   ├─ solve_single_monomial! :373    [10,100 | 64.8%]   LU solve
   │   └─ lu!
   │       ├─ getrf!                 [ 9,810 | 62.9%]   ★★★ top hotspot
   │       ├─ ldiv!                  [   180 |  1.2%]
   │       └─ chkfinite              [    95 |  0.6%]
   ├─ solve_single_monomial! :341    [ 2,641 | 16.9%]   matrix assembly
   │   └─ assemble_cohomological_matrix_and_rhs
   │       ├─ copyto!                [ 1,389 |  8.9%]   system matrix init (intrinsic)
   │       ├─ materialize! (bcast)   [   581 |  3.7%]   RHS broadcast (intrinsic)
   │       ├─ rmul!                  [   417 |  2.7%]   scalar multiply (intrinsic)
   │       └─ mul! / matvecmul!      [   141 |  0.9%]   mat-vec product (intrinsic)
   ├─ solve_single_monomial! :372    [ 1,554 | 10.0%]   ★★ index/copy hotspot
   │   └─ getindex → _unsafe_getindex[ 1,465 |  9.4%]   slice copy alloc  ← ELIMINATED
   ├─ solve_single_monomial! :367    [   469 |  3.0%]   ★ stacking .= copies
   │   └─ materialize!              [   469 |  3.0%]   ← ELIMINATED
   └─ solve_single_monomial! :331    [   175 |  1.1%]   multilinear terms
       └─ compute_multilinear_terms
           ├─ quadratic! / assembly_G![  116 |  0.7%]
           └─ cubic!    / assembly_H! [   59 |  0.4%]
```

The monomial loop in `solve_cohomological_equations!` is inherently serial (causal/ascending-degree order).

---

## 3. Hotspot analysis

### 3.1  LU factorisation — 62.9%

**Code**: `CohomologicalEquations.jl` (line 373 in original, rewritten)

`lu!` calls LAPACK `getrf!` — O(n³) dense factorisation. Runs once per monomial with an
`n_sys × n_sys` system where `n_sys = FOM + nR` (`nR` = resonant master modes at this
monomial). This is the unavoidable dominant cost: the system matrix `L(s)` changes at
every monomial because `s = ⟨λ, α⟩` is monomial-specific.

| Sub-call | Samples | % of branch | Notes |
| -------- | -------: | ----------: | ----- |
| `getrf!` kernel | 9,810 | 97.1% | LAPACK dense LU |
| `chkfinite` | 95 | 0.9% | NaN/Inf guard — **eliminated** |
| `ldiv!` | 180 | 1.8% | Back-substitution (intrinsic) |

---

### 3.2  Slice copy — 10.0%

**Original code** (`CohomologicalEquations.jl:372`):

```julia
# Range indexing always allocates a new Matrix — this is a copy, not a view.
A_sys = ctx.system_matrix_buffer[1:n_sys, 1:n_sys]
ldiv!(lu!(A_sys), view(ctx.rhs_buffer, 1:n_sys))
```

`ctx.system_matrix_buffer` is sized `(FOM+ROM) × (FOM+ROM)` worst-case; taking a range
index extracts a fresh `n_sys × n_sys` heap allocation every monomial. The copy is then
factored in-place and discarded, so it was never necessary.

**Key insight**: the buffer is written fresh at the start of every monomial call, so it
is never reused after the LU step. There is no reason to preserve it, and therefore no
reason to copy it.

---

### 3.3  Stacking copies — 3.0%

**Original code** (`CohomologicalEquations.jl:367–370`):

```julia
ctx.system_matrix_buffer[1:FOM, 1:n_sys]         .= M_inv    # materialize!
ctx.system_matrix_buffer[(FOM+1):n_sys, 1:n_sys]  .= M_orth  # materialize!
ctx.rhs_buffer[1:FOM]                             .= rhs_inv
ctx.rhs_buffer[(FOM+1):n_sys]                     .= rhs_orth
```

`M_inv` and `M_orth` were freshly allocated inside `assemble_cohomological_matrix_and_rhs`
and `assemble_orthogonality_matrix_and_rhs`, then immediately copied into the context
buffers. These two allocation-then-copy round-trips account for the 3.0% `materialize!`
cost at line 367.

---

### 3.4  Matrix / RHS allocations inside assembly

**Original code** (`InvarianceEquation.jl:810–813`, `MasterModeOrthogonality.jl:743–744`):

```julia
# InvarianceEquation — called every monomial:
M   = Matrix{T}(undef, FOM, FOM + nR)   # ← heap alloc
rhs = zeros(T, FOM)                      # ← heap alloc

# MasterModeOrthogonality — called every resonant monomial:
M   = Matrix{T}(undef, nR, FOM + nR)    # ← heap alloc
rhs = Vector{T}(undef, nR)              # ← heap alloc
```

These allocations do not show up as large sample counts in the statistical profiler
(malloc is fast) but create GC pressure proportional to problem size and polynomial
order. At large FOM (e.g. FOM = 2000 for a 1000-DOF beam), each `Matrix{ComplexF64}(undef, 2000, 2000)` is a 64 MB allocation — per monomial call.

---

### 3.5  `compute_multilinear_terms` allocations — 1.1%

**Original code** (`MultilinearTerms.jl:370–375`):

```julia
result  = zeros(T, FOM)          # ← alloc every call
scratch = similar(result)        # ← alloc every call
temp    = similar(result)        # ← alloc every call
unit_vectors = [SVector(ntuple(k -> k == j ? 1 : 0,
    external_system_size)) for j in 1:external_system_size]   # ← rebuilt every call
```

Three FOM-length work vectors and the constant `unit_vectors` array were re-created on
every monomial. The `unit_vectors` entries are identical across all monomials.

---

## 4. Optimisations implemented

### OPT-1 + OPT-2  In-place LU on `system_matrix_buffer` with `check=false`

**Files**: `CohomologicalEquations.jl`

The slice copy and `chkfinite` overhead are both eliminated in a single change.
Because `system_matrix_buffer` is always overwritten at the start of each monomial and
is never read back after the LU step, it can be factored directly in-place. `lu!`
accepts any `StridedMatrix`, and `view(A, 1:n, 1:n)` of a `Matrix` is strided (column
stride = `size(A, 1)`), so LAPACK is called with the correct `lda` parameter — no
internal copy occurs inside LAPACK either.

```julia
# BEFORE — allocates a new n_sys × n_sys matrix every monomial:
A_sys = ctx.system_matrix_buffer[1:n_sys, 1:n_sys]   # ← 10% of runtime
ldiv!(lu!(A_sys), view(ctx.rhs_buffer, 1:n_sys))

# AFTER — zero-allocation, factors the buffer in-place:
F = lu!(view(ctx.system_matrix_buffer, 1:n_sys, 1:n_sys), check = false)
ldiv!(F, view(ctx.rhs_buffer, 1:n_sys))
```

`check = false` skips `LAPACK.chkfinite` (the NaN/Inf scan). The assembled system
matrix comes entirely from `linear_terms` (validated FEM matrices) and precomputed
eigenvalue combinations, so no user-supplied data can introduce NaN at this point.

**Saving**: ~10.0% (slice copy) + ~0.6% (chkfinite) = **~10.6%**

---

### OPT-3  In-place assembly — direct writes into context buffers

**Files**: `InvarianceEquation.jl`, `MasterModeOrthogonality.jl`, `CohomologicalEquations.jl`

New `!`-suffixed variants of both assembly functions accept pre-allocated `M` and `rhs`
views as arguments and write directly into them. The original allocating variants are
preserved for external use.

```julia
# InvarianceEquation.jl — new variant (no M or rhs allocation):
function assemble_cohomological_matrix_and_rhs!(M, rhs, s, linear_terms,
        C_coeffs, E_coeffs, resonance, lower_order_couplings,
        external_dynamics, g_buffer)
    fill!(rhs, zero(eltype(rhs)))
    evaluate_system_matrix_and_lower_order_rhs!(view(M, :, 1:FOM), rhs, ...)
    ...
    return nothing
end

# MasterModeOrthogonality.jl — new variant with nR=0 fast-path:
function assemble_orthogonality_matrix_and_rhs!(M, rhs, s, ...)
    isempty(rhs) && return nothing   # non-resonant monomial: skip immediately
    ...
    return nothing
end
```

`solve_single_monomial!` now assembles directly into `ctx.system_matrix_buffer` and
`ctx.rhs_buffer`, eliminating both the intermediate allocations and the subsequent
stacking copies:

```julia
# BEFORE — two assembly allocations + two stacking copies:
M_inv, rhs_inv = assemble_cohomological_matrix_and_rhs(...)   # alloc M + rhs
M_orth, rhs_orth = assemble_orthogonality_matrix_and_rhs(...) # alloc M + rhs
ctx.system_matrix_buffer[1:FOM, 1:n_sys]         .= M_inv    # copy (3.0%)
ctx.system_matrix_buffer[(FOM+1):n_sys, 1:n_sys]  .= M_orth  # copy
ctx.rhs_buffer[1:FOM]                             .= rhs_inv
ctx.rhs_buffer[(FOM+1):n_sys]                     .= rhs_orth

# AFTER — zero intermediate allocations, no stacking copies:
assemble_cohomological_matrix_and_rhs!(
    view(ctx.system_matrix_buffer, 1:FOM, 1:n_sys),
    view(ctx.rhs_buffer, 1:FOM), s, ...)
view(ctx.rhs_buffer, 1:FOM) .+= ctx.ml_result_buffer

assemble_orthogonality_matrix_and_rhs!(
    view(ctx.system_matrix_buffer, (FOM+1):n_sys, 1:n_sys),
    view(ctx.rhs_buffer, (FOM+1):n_sys), s, ...)
```

**Saving**: ~3.0% direct (stacking copies eliminated) + GC pressure relief
(scales with FOM and polynomial order)

---

### OPT-4  Pre-allocated buffers in `MultilinearTermsCache`

**Files**: `MultilinearTerms.jl`, `CohomologicalEquations.jl`

`MultilinearTermsCache` is now parameterised by `T` and carries three reusable work
vectors and the precomputed `unit_vectors`:

```julia
# BEFORE:
struct MultilinearTermsCache
    splits::Vector{Vector{Vector{CachedSplit}}}
end

# AFTER:
struct MultilinearTermsCache{T}
    splits::Vector{Vector{Vector{CachedSplit}}}
    result_buffer::Vector{T}    # length FOM
    scratch_buffer::Vector{T}   # length FOM
    temp_buffer::Vector{T}      # length FOM
    unit_vectors::Vector        # length N_EXT; built once, never rebuilt
end
```

A new in-place variant `compute_multilinear_terms!` writes into a caller-supplied
`result` buffer using `cache.scratch_buffer` and `cache.temp_buffer` internally:

```julia
# BEFORE — 3 allocs + 1 vector-of-SVectors per monomial:
result  = zeros(T, FOM)
scratch = similar(result)
temp    = similar(result)
unit_vectors = [SVector(...) for j in 1:external_system_size]

# AFTER — zero allocations, writes into ctx.ml_result_buffer:
compute_multilinear_terms!(ctx.ml_result_buffer, model, idx, W, ml_cache)
```

`ctx.ml_result_buffer` (length FOM, `ComplexF64`) is added to `CohomologicalContext`
and allocated once in `solve_cohomological_problem`.

**Saving**: ~1.1% + zero GC from these 4 per-monomial allocations

---

## 5. Optimisation impact summary

| # | Optimisation | Mechanism | Est. saving | Status |
| - | ------------ | --------- | ----------: | ------ |
| OPT-1 | Factor `system_matrix_buffer` in-place | Eliminate `A_sys = buf[1:n, 1:n]` copy | ~10.0% | **Implemented** |
| OPT-2 | `lu!(..., check=false)` | Skip `chkfinite` NaN scan | ~0.6% | **Implemented** |
| OPT-3 | In-place assembly (`!` variants) | Eliminate M/rhs allocs + stacking copies | ~3.0% + GC | **Implemented** |
| OPT-4 | Buffer `MultilinearTermsCache` | Eliminate 4 per-monomial allocs | ~1.1% | **Implemented** |
| OPT-5 | Schur complement for resonant block | Factor FOM×FOM instead of (FOM+nR)×(FOM+nR) | ~10–30%† | Not implemented |
| OPT-6 | Parallel assembly of M_inv / M_orth | Overlap two independent assemblies on 2 threads | ~5–10%‡ | Not implemented |

† OPT-5 impact depends on how many monomials are resonant; most significant at high SSM order or large master dimension.  
‡ OPT-6 requires thread-safe buffer management; applicable since the profile was collected with 2 threads.

**Combined OPT-1–4 (implemented)**: expected ~14–15% direct profiler saving plus
elimination of all per-monomial heap allocations in the inner loop. The remaining 62.9%
(`getrf!`) is the irreducible mathematical cost of dense LU.

---

## 6. What the inner loop looks like after optimisation

```julia
# solve_single_monomial! — hot path after OPT-1 through OPT-4
# (CohomologicalEquations.jl)

# Step 4: nonlinear RHS — zero allocation (uses ml_cache buffers)
compute_multilinear_terms!(ctx.ml_result_buffer, model, idx, W, ml_cache)

# Step 6: assemble invariance block directly into system_matrix_buffer
assemble_cohomological_matrix_and_rhs!(
    view(ctx.system_matrix_buffer, 1:FOM, 1:n_sys),
    view(ctx.rhs_buffer, 1:FOM),
    s, ctx.linear_terms, ctx.invariance_C_coeffs, ctx.invariance_E_coeffs,
    resonance, lower_order_couplings, external_dynamics, ctx.external_rhs_buffer,
)
view(ctx.rhs_buffer, 1:FOM) .+= ctx.ml_result_buffer

# Assemble orthogonality block (returns immediately if nR == 0)
assemble_orthogonality_matrix_and_rhs!(
    view(ctx.system_matrix_buffer, (FOM+1):n_sys, 1:n_sys),
    view(ctx.rhs_buffer, (FOM+1):n_sys),
    s, ctx.orthogonality_J_coeffs, ...,
)

# Factor in-place — no copy (buffer is refilled next iteration)
F = lu!(view(ctx.system_matrix_buffer, 1:n_sys, 1:n_sys), check = false)
ldiv!(F, view(ctx.rhs_buffer, 1:n_sys))
```

The only remaining allocation per monomial is the `LU` struct returned by `lu!` itself
(~400 B for the factorisation metadata + pivot vector). This is unavoidable as it must
be passed to `ldiv!`.

---

## 7. Measurement recommendations

```julia
using BenchmarkTools

# Verify zero-allocation for non-resonant monomials after OPT-1–4
# (expect only the ~400 B LU struct)
@allocated solve_single_monomial!(W, R, idx_non_resonant, ctx, model, ml_cache)

# Measure total GC pressure
@time solve_cohomological_equations!(W, R, ctx, model, ml_cache)
# → look at "X allocations: Y MiB"; should now be O(n_monomials) × 400 B

# Regression guard in tests:
@assert @allocated(solve_single_monomial!(...)) < 1000
```
