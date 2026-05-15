# MORFE.jl — Next Steps

## Context and scaling assumptions

These plans are written with the following regime in mind:

- **FOM is very large** (10³–10⁶ DOF): each monomial solve is `O(FOM³)` for dense LU
  or `O(FOM·nnz)` for sparse — this is always the dominant cost.
- **Expansion order is moderate**: typically 7–11, at most ~30. For NVAR=4, degree=9
  gives `L ≈ 715` monomials; with conjugate symmetry ~350 primaries. L is in the
  hundreds to low thousands, not millions.
- **Consequence**: the number of monomials is small enough that per-monomial loop
  overhead is irrelevant next to the per-monomial FOM-scale linear solve. The right
  optimisations are those that reduce the cost _of each solve_ (threading) or _the number
  of solves_ (conjugate symmetry, already done), not micro-loop bookkeeping.

---

## Current state snapshot

| Category | Status |
|:---------|:-------|
| In-place LU on `system_matrix_buffer`, `check=false` | ✅ |
| In-place system assembly (invariance + orthogonality) | ✅ |
| Pre-allocated `MultilinearTermsCache` buffers | ✅ |
| `CohomologicalContext` sub-struct decomposition | ✅ |
| Conjugate-symmetry skip/fill (secondary monomials skipped, filled from primary) | ✅ |
| Micro-optimisations O1–O6 (`@inbounds`, `primary_pairs`, merged setup loop, dict lookup, …) | ✅ |
| BLAS `axpy!` in `LowerOrderCouplings` | ✅ |
| FEM batched assembly O1–O3 (`FEMMultilinearMap`, `FEMCachedSplit`, `_replay_fem_split!`) | ✅ |

---

## Priority table

| # | Item | Impact | Effort | Status | Section |
|:--|:-----|:-------|:-------|:-------|:--------|
| 1 | Thread-parallel solve by degree group | **Critical** | Large | ⬜ | §1 |
| 2 | `solve_jobs` flat work list | Medium | Small | ⬜ | §2 |
| 3 | `detect_conjugate_permutation` standalone utility | Medium (UX) | Trivial | ✅ | §3 |
| 4a | `_replay_term!` multiple dispatch (prerequisite to §4) | Cleanup | Trivial | ✅ | §4a |
| 4 | FEM O4 — combined element loop | **Critical** (FEM) | Large | ⬜ | §4 |
| 5 | Dense-path Schur complement for resonant block | Low–Medium | Small | ⬜ | §5 |
| 6 | Multiindex generation: eliminate `vcat()` | Low | Small | ⬜ | §6 |
| 7 | Accept `SVector` in `indices_in_box_with_bounded_degree` | Low | Trivial | ⬜ | §7 |
| 8 | Pre-allocate Horner scratch in `PropagateEigenmodes` | Negligible | Trivial | ⬜ | §8 |
| 9 | Remove `monomial_map` from `ConjugateSymmetryData` | Cleanup | Trivial | ⬜ | §9 |

**Not pursued**: Real arithmetic for self-conjugate monomials. Self-conjugate monomials
are a small fraction of L (which is itself small), so their 4× LU speedup is swamped by
the cost of the non-self-conjugate primaries. The implementation complexity is not
justified.

---

## §1 — Thread-parallel solve by degree group *(Critical)*

### Why this is the right target

At FOM = 5 000, a single non-resonant dense-path monomial solve is dominated by `lu!`
(`getrf!`) at `O(FOM³) ≈ 125 × 10⁹` flops. Profiling confirms `getrf!` accounts for
**62.9%** of total wall time. For degree-9, NVAR-4 expansions with conjugate symmetry,
there are ~350 primary solves. Each is independent of the others within the same degree
shell. Parallelising across 8 threads gives an 8× reduction in total wall time — the
single most impactful change available.

### Correctness of intra-degree parallelism

The GrLex ordering groups monomials by total degree `d` before splitting within a degree.
Lower-order couplings `ξ[α]` at degree `d` use only `W[β]` with `|β| < d`. Therefore:

> **All primary monomials at the same degree `d` are mutually independent.**

The fill step for degree-`d` secondaries (`fill_conjugate_monomial!`) depends on the
corresponding primary being solved — so fills must happen after all same-degree primary
solves complete. This is a degree-level barrier, not a monomial-level one.

### Required infrastructure

#### 1a. `degree_boundaries` in `ConjugateSymmetryData`

Add a `degree_boundaries::Vector{Int}` field. Entry `k` is the first index into
`solve_jobs` (see §2) that belongs to degree `k`. Built while populating `solve_jobs`:

```julia
degree_boundaries = Int[1]
current_degree = sum(mset.exponents[solve_jobs[1][1]])
for k in 2:length(solve_jobs)
    d = sum(mset.exponents[solve_jobs[k][1]])
    if d > current_degree
        push!(degree_boundaries, k)
        current_degree = d
    end
end
push!(degree_boundaries, length(solve_jobs) + 1)   # sentinel
```

This requires §2 (`solve_jobs`) to be in place first.

#### 1b. Per-thread compute buffers

The following are written to during each monomial solve and must not be shared:

| Mutable resource | Refactoring |
|:-----------------|:------------|
| `ctx.buffers` (`CohomologicalBuffers`) | `Vector{CohomologicalBuffers{T}}` of length `n_threads` |
| `ctx.lower_order.buffer` (zeroed before each monomial) | `buffer::Vector{Vector{Vector{T}}}` — outer index = thread |
| `ss.klu_cache` in `SparseLinearSolverState` (KLU numeric factor) | `klu_caches::Vector{Ref{Any}}` of length `n_threads` |

All other fields in `ctx` are read-only after construction; no change needed.

**`CohomologicalBuffers` multi-allocation helper** (`SolverResources.jl`):

```julia
function alloc_thread_buffers(FOM::Int, ROM::Int, n_threads::Int, ::Type{T}) where {T}
    return [CohomologicalBuffers{T}(FOM, ROM) for _ in 1:n_threads]
end
```

**`LowerOrderResources` per-thread buffer** (`SolverResources.jl`):

```julia
struct LowerOrderResources{NVAR, T}
    multiindex_dict   ::Dict{SVector{NVAR, Int}, Int}
    buffer            ::Vector{Vector{Vector{T}}}  # [thread_id][order] → FOM vector
    candidate_indices ::Vector{Vector{Int}}
    unit_vectors      ::Vector{SVector{NVAR, Int}}
end
# Constructor gains an n_threads argument; defaults to 1 (serial path unchanged)
```

#### 1c. Thread-parallel solve loop

New function `solve_cohomological_equations_threaded!` in `CohomologicalEquations.jl`:

```julia
function solve_cohomological_equations_threaded!(
        W, R, ctx, sym, model, ml_cache
)
    jobs = sym.solve_jobs
    bounds = sym.degree_boundaries
    n_degrees = length(bounds) - 1

    for d in 1:n_degrees
        lo = bounds[d];  hi = bounds[d + 1] - 1

        # All degree-d primary solves are independent — run in parallel
        Threads.@threads for k in lo:hi
            idx, _ = jobs[k]
            _solve_single_threaded!(W, R, idx, ctx, Threads.threadid(), sym, model, ml_cache)
        end

        # Degree barrier: all primaries at degree d are done; fill their secondaries
        for k in lo:hi
            idx, dst = jobs[k]
            iszero(dst) || fill_conjugate_monomial!(W, R, dst, idx, sym)
        end
    end
    return nothing
end
```

`_solve_single_threaded!` is identical to `solve_single_monomial!` except it indexes
into `ctx.buffers[tid]` and `ctx.lower_order.buffer[tid]` instead of scalar fields.

#### 1d. Integration strategy

The existing serial `solve_cohomological_equations!` is left **completely unchanged**.
`solve_cohomological_problem` gains a keyword argument:

```julia
function solve_cohomological_problem(...; n_threads::Int = 1, ...)
```

When `n_threads > 1`, the driver allocates per-thread buffers and dispatches to the
threaded overload. The default `n_threads = 1` keeps the serial path allocation-identical
to the current state.

### Files to modify

| File | Change |
|:-----|:-------|
| `ConjugateSymmetry.jl` | Add `degree_boundaries` field; populate during build |
| `SolverResources.jl` | Per-thread constructors for `CohomologicalBuffers` and `LowerOrderResources` |
| `CohomologicalEquations.jl` | Add `solve_cohomological_equations_threaded!` + `_solve_single_threaded!` |
| `CohomologicalDriver.jl` | Add `n_threads` kwarg; conditional threaded dispatch |

### Verification

1. `n_threads = 1` output must be bit-identical to the serial path.
2. `n_threads = 2`: W, R coefficients must match serial output to `1e-12 * norm(W)`.
3. `@benchmark solve_cohomological_problem(...; n_threads=8)` on the 20×3×3 Ferrite beam
   should show close to 8× speedup in the solve-loop component.
4. `@allocated solve_cohomological_problem(...; n_threads=1)` must be unchanged from
   the serial baseline.

---

## §2 — `solve_jobs` flat work list *(Medium)*

### Motivation

The active-symmetry solve loop currently iterates all `L` monomial indices:

```julia
for idx in 1:nterms                            # L iterations
    @inbounds sym.skip_bits[idx] && continue   # BitVector read + branch on ~half
    solve_single_monomial!(...)
    if ptr <= length(pairs) && pairs[ptr][1] == idx
        fill_conjugate_monomial!(...)
        ptr += 1
    end
end
```

With `L ≈ 700` and FOM ≫ 1, the loop overhead is negligible. The main benefit of
this change is **structural clarity** and enabling §1 (threading needs `degree_boundaries`
which is naturally built while constructing `solve_jobs`) rather than raw performance.

### Implementation

Replace `primary_pairs::Vector{NTuple{2,Int}}` with `solve_jobs::Vector{NTuple{2,Int}}`
in `ConjugateSymmetryData`. Each entry `(idx, dst)`:

- `dst = 0` → solve `idx`, no fill (self-symmetric or unpaired)
- `dst > 0` → solve `idx`, then fill `dst`

Built in `_build_conjugate_symmetry`:

```julia
solve_jobs = NTuple{2, Int}[]
for i in eachindex(monomial_map)
    skip_bits[i] && continue
    j = monomial_map[i]
    push!(solve_jobs, j > i ? (i, j) : (i, 0))
    j > i && (skip_bits[j] = true)
end
```

The solve loop becomes:

```julia
for (idx, dst) in sym.solve_jobs
    solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
    iszero(dst) || fill_conjugate_monomial!(W, R, dst, idx, sym)
end
```

`skip_bits` is kept in the struct — it is consumed externally by
`build_multilinear_terms_cache`.

### Files to modify

| File | Change |
|:-----|:-------|
| `ConjugateSymmetry.jl` | Replace `primary_pairs` with `solve_jobs`; both factory overloads |
| `CohomologicalEquations.jl` | Rewrite active-symmetry `solve_cohomological_equations!` |

---

## §3 — `detect_conjugate_permutation` utility function *(Medium — UX)*

### Design constraint

Conjugate-eigenvalue pairing (`λ_j ≈ conj(λ_i)`) is **necessary but not sufficient** for
the symmetry relation `W_j = conj(W_i)` to hold. Two modes can share the same frequency
yet have eigenvectors that are not complex conjugates of each other (e.g. degenerate
eigenspaces, non-standard eigenvector normalisation, Jordan chains). Silently enabling
conjugate symmetry based on eigenvalues alone would produce wrong results in such cases.

**Consequence for the driver**: the default `conjugate_permutation = nothing` continues
to mean "no conjugate-symmetry exploitation". Auto-detection is **not** wired into the
driver. `detect_conjugate_permutation` is a standalone utility the caller can use as a
starting point and then validate before passing the result to `solve_cohomological_problem`.

### Implementation

New public function in `ConjugateSymmetry.jl`:

```julia
"""
    detect_conjugate_permutation(lambda; atol=1e-8)
        -> Union{SVector{NVAR,Int}, NoConjugatePermutation}

Build a candidate conjugate permutation from eigenvalues `lambda` (length NVAR):

  - perm[i] = j  if lambda[j] ≈ conj(lambda[i]) and j ≠ i
  - perm[i] = i  if lambda[i] is numerically real (|imag| ≤ atol)
  - perm[i] = 0  if no conjugate partner found in lambda

Returns `NoConjugatePermutation()` when all entries map to themselves or zero
(no pairs to exploit).

**Warning**: eigenvalue pairing is necessary but not sufficient.  The caller must
verify that the corresponding eigenvectors satisfy W_j = conj(W_i) before passing
the returned permutation to `solve_cohomological_problem`.
"""
function detect_conjugate_permutation(lambda::AbstractVector{<:Number}; atol=1e-8)
    NVAR = length(lambda)
    perm = zeros(Int, NVAR)
    assigned = falses(NVAR)
    for i in 1:NVAR
        assigned[i] && continue
        if abs(imag(lambda[i])) ≤ atol
            perm[i] = i;  assigned[i] = true;  continue
        end
        found = false
        for j in (i+1):NVAR
            (assigned[j] || abs(imag(lambda[j])) ≤ atol) && continue
            if abs(lambda[j] - conj(lambda[i])) ≤ atol
                perm[i] = j;  perm[j] = i
                assigned[i] = assigned[j] = true
                found = true;  break
            end
        end
        found || (perm[i] = 0;  assigned[i] = true)
    end
    svec = SVector{NVAR, Int}(perm)
    all(i -> svec[i] == i || svec[i] == 0, 1:NVAR) && return NoConjugatePermutation()
    return svec
end
```

**Driver** — no change. `conjugate_permutation = nothing` continues to mean
`NoConjugatePermutation()`:

```julia
_conj_perm = conjugate_permutation !== nothing ?
    SVector{NVAR, Int}(conjugate_permutation) :
    NoConjugatePermutation()
```

The caller's responsibility is to call `detect_conjugate_permutation`, check the result
against the actual eigenvectors, and pass only a validated permutation.

### Files to modify

| File | Change |
|:-----|:-------|
| `ConjugateSymmetry.jl` | Add `detect_conjugate_permutation` with warning docstring |
| `src/MORFE.jl` | Export `detect_conjugate_permutation` |

`CohomologicalDriver.jl` is **not modified**.

### Verification

1. ROM=2 conjugate pair `(λ, conj(λ))` → result is `SVector(2, 1)`.
2. ROM=2 two real eigenvalues → result is `NoConjugatePermutation()`.
3. ROM=2 two complex eigenvalues with `λ_1 ≈ conj(λ_2)` but eigenvectors unrelated →
   `detect_conjugate_permutation` returns the pairing, but passing it to the solver
   without first verifying eigenvector conjugacy is a user error. The function's docstring
   documents this explicitly.

---

## §4a — `_replay_term!` multiple dispatch *(Cleanup — prerequisite to §4)* ✅

Replaced the `if t isa FEMMultilinearMap ... else ... end` branch in
`compute_multilinear_terms!` with two `_replay_term!` methods dispatching on
`MultilinearMap` and `FEMMultilinearMap` respectively.  `compute_multilinear_terms!`
is now a flat loop with no `isa` checks.  Adding further term types requires only a
new `_replay_term!` method.

**Files modified**: `MultilinearTerms.jl` — added `_replay_term!` pair; simplified
`compute_multilinear_terms!`.

---

## §4 — FEM O4: combined element loop across all FEM terms *(Critical for FEM)*

### Motivation

After O1–O3, `_replay_fem_split!` is called once per `(term, split)` pair per monomial.
For SVK (quadratic + cubic terms), this triggers two separate element loops and therefore
two `fem_reinit!` calls per element — the second recomputes identical shape-function
data. O4 merges all FEM term loops for a given monomial into a **single element
traversal**: `reinit!` once, scatter each globally-unique W-column once, then dispatch
`accumulate_qp!` for every entry from every term.

For SVK the saving is `reinit!` cost × element count × primary monomials. For
higher-degree material models with `n_fem_terms > 2` the saving scales linearly.

### Dispatch structure — closure path is unchanged

`compute_multilinear_terms!` dispatches on term type via `_replay_term!` (implemented
in §4a, now done):

```julia
# In compute_multilinear_terms!:
for (t_idx, t) in enumerate(model.nonlinear_terms)
    t.deg > deg_max && continue
    _replay_term!(result, t, W, exp_index, t_idx, cache)
end

# MultilinearMap (closure path — Gridap demo):
function _replay_term!(result, t::MultilinearMap, W, exp_index, t_idx, cache)
    deg = t.deg - t.multiplicity_external
    for split in cache.splits[exp_index][t_idx]
        _replay_split!(result, cache.scratch_buffer, cache.temp_buffer,
                       t, W, split, deg, cache.unit_vectors)
    end
end

# FEMMultilinearMap (batched element-loop path — Ferrite demo):
function _replay_term!(result, t::FEMMultilinearMap, W, exp_index, t_idx, cache)
    for fem_split in cache.fem_splits[exp_index][t_idx]
        _replay_fem_split!(result, t, W, fem_split, cache.fem_Fe)
    end
end
```

The Gridap demo uses `MultilinearMap` where `quadratic_nonlinearity!` /
`cubic_nonlinearity!` call `assemble_vector` internally — these route through
`_replay_term!(::MultilinearMap, ...)` and are **completely unaffected** by §4.
Adding a new `AbstractMultilinearMap` subtype in the future requires only a new
`_replay_term!` method; `compute_multilinear_terms!` never changes.

§4 adds a third `_replay_term!` method for the combined-loop case, or reorganises
the existing `FEMMultilinearMap` method to call `_replay_all_fem_splits!` instead of
the per-term loop.  The combined element loop is built only for `FEMMultilinearMap`
terms; `global_fem_splits[l].global_unique_cols` is empty when no `FEMMultilinearMap`
terms are present, and the loop body is skipped via `isempty(gs.global_unique_cols)`.
A model using only `MultilinearMap` (e.g. the Gridap demo) sees zero change.

### Key new types

**`FEMGlobalEntry{DEG}`** — one factorisation entry in the combined loop, with factor
indices remapped into the global unique-column table:

```julia
struct FEMGlobalEntry{DEG}
    term_idx            ::Int
    multiplier          ::Int
    local_factor_indices::NTuple{DEG, Int}
end
```

**`FEMGlobalSplit{ENTRIES_TUPLE}`** — all combined-loop data for one monomial:

```julia
struct FEMGlobalSplit{ENTRIES_TUPLE}
    global_unique_cols  ::Vector{Tuple{Int, Int}}
    entries_by_deg      ::ENTRIES_TUPLE
    driver_term_idx     ::Int
end
```

`ENTRIES_TUPLE` is a concrete type parameter (e.g.
`Tuple{Vector{FEMGlobalEntry{2}}, Vector{FEMGlobalEntry{3}}}` for SVK) enabling fully
type-stable dispatch in the inner loop.

### Cache build

`_build_global_fem_split(model, fem_term_splits_l, fem_term_indices)` is called once
per monomial during `build_multilinear_terms_cache`:

1. Collect all `(order, W_col_idx)` pairs from all `me=0` FEM splits across all terms;
   deduplicate into `global_unique_cols` via a `Dict`.
2. Remap each `FEMFactorisationEntry` into a `FEMGlobalEntry{DEG}` using global indices.
3. Group by `DEG`; materialise as a typed `Tuple` of typed `Vector`s.

Validate that all FEM terms share the same mesh (element count and type).

### Combined replay function

`_replay_all_fem_splits!(result, model, W, global_split, global_∇W_qp, global_Fe_buffers)`:

```
for element in fem_elements(driver_term)
    fem_reinit!(element, driver_term)                  ← once per element (was n_fem_terms times)
    for each col in global_unique_cols
        scatter_qp!(global_∇W_qp[col, :], W_col, element, driver_term)
    end
    fill all global_Fe_buffers with zeros
    for each quadrature point q
        dΩ = fem_getdetJdV(element, q, driver_term)
        _accumulate_global_entries!(global_Fe_buffers, global_∇W_qp,
                                    global_split.entries_by_deg, model, element, q, dΩ)
    end
    for each participating term
        assemble_element!(result, global_Fe_buffers[term], element, term)
    end
end
```

### Type-stable inner dispatch via recursive tuple peeling

```julia
@inline _accumulate_global_entries!(_, _, ::Tuple{}, _, _, _, _) = nothing

@inline function _accumulate_global_entries!(Fe_bufs, ∇W_qp,
        entries_by_deg::Tuple{Vector{FEMGlobalEntry{DEG}}, Vararg},
        model, el, q, dΩ) where {DEG}
    for gentry in first(entries_by_deg)
        t = model.nonlinear_terms[gentry.term_idx]
        ∇W_args = ntuple(k -> ∇W_qp[gentry.local_factor_indices[k], q], Val(DEG))
        accumulate_qp!(Fe_bufs[gentry.term_idx], ∇W_args, gentry.multiplier, el, q, dΩ, t)
    end
    _accumulate_global_entries!(Fe_bufs, ∇W_qp, Base.tail(entries_by_deg), model, el, q, dΩ)
end
```

For SVK Julia specialises two methods (DEG=2 and DEG=3); no dynamic dispatch in either.

### Extended `MultilinearTermsCache`

Add three fields:

```julia
global_fem_splits  ::Vector{FEMGlobalSplit}
global_∇W_qp       ::Matrix             # (max_global_unique × max_n_qp)
global_Fe_buffers  ::Vector{Vector{T}}  # one per FEM term
```

### Integration in `compute_multilinear_terms!`

```julia
# Combined FEM loop (me=0 terms only)
gs = cache.global_fem_splits[exp_index]
isempty(gs.global_unique_cols) ||
    _replay_all_fem_splits!(result, model, W, gs, cache.global_∇W_qp, cache.global_Fe_buffers)

# Closure terms and me>0 FEM terms (unchanged)
for (t_idx, t) in enumerate(model.nonlinear_terms)
    ...
end
```

### Files to modify

| File | Change |
|:-----|:-------|
| `MultilinearTerms.jl` | Add `FEMGlobalEntry{DEG}`, `FEMGlobalSplit`; extend `MultilinearTermsCache`; add `_build_global_fem_split`, `_replay_all_fem_splits!`, `_accumulate_global_entries!`; update `build_multilinear_terms_cache` and `compute_multilinear_terms!` |

### Verification

1. For each monomial index `l`, assert `_replay_all_fem_splits!` matches the per-term
   reference to relative tolerance `1e-12`.
2. Full pipeline: run the Ferrite beam demo; compare W, R to O1–O3 baseline.
3. Zero allocations in `compute_multilinear_terms!` for any degree-≥2 monomial.
4. Benchmark on 50×5×5 or 100×5×5 mesh; `reinit!` cost halved relative to O1–O3.

---

## §5 — Dense-path Schur complement for resonant monomials *(Low–Medium)*

### Motivation

For resonant monomials (`nR > 0`), the dense path factors the full
`(FOM + nR) × (FOM + nR)` bordered system. With `nR ≤ ROM` small (1–4) and `FOM` large,
factoring only the `FOM × FOM` block `L(s)` and condensing to an `nR × nR` Schur system
saves `O(FOM² · nR)` flops relative to the full bordered factor. The sparse path already
implements this; this item brings the dense path to parity.

Resonant monomials are rare for high-order expansions (typically < 1% of total L), so
the gain is problem-dependent — most significant for systems with many internal resonances
or at low polynomial orders (degree ≤ 3).

### Implementation

In `CohomologicalSolver.jl`, split the dense `_solve_monomial!` on `nR`:

```julia
function _solve_monomial!(ctx, s, nR, resonance, lc, ed)
    _assemble_bordered_system!(ctx, s, nR, resonance, lc, ed)
    n_sys = FOM + nR
    if nR == 0
        F = lu!(view(ctx.buffers.system_matrix, 1:FOM, 1:FOM), check = false)
        ldiv!(F, view(ctx.buffers.rhs, 1:FOM))
    else
        # Factor FOM×FOM L block; solve 1+nR right-hand sides
        F = lu!(view(ctx.buffers.system_matrix, 1:FOM, 1:FOM), check = false)
        C_view = view(ctx.buffers.system_matrix, 1:FOM, (FOM+1):n_sys)
        # Solve [W', C'] = L\[rhs, C] in-place using the right half of system_matrix as scratch
        rhs_ext = view(ctx.buffers.system_matrix, 1:FOM, (FOM+1):(FOM+nR+1))
        rhs_ext[:, 1]      .= view(ctx.buffers.rhs, 1:FOM)
        rhs_ext[:, 2:nR+1] .= C_view
        ldiv!(F, rhs_ext)
        W_prime = view(rhs_ext, :, 1)
        C_prime = view(rhs_ext, :, 2:nR+1)
        # nR×nR Schur complement
        Ĵ = view(ctx.buffers.system_matrix, (FOM+1):n_sys, 1:FOM)
        Ĉ = view(ctx.buffers.system_matrix, (FOM+1):n_sys, (FOM+1):n_sys)
        g = view(ctx.buffers.rhs, (FOM+1):n_sys)
        S   = Ĵ * C_prime - Matrix(Ĉ)       # nR×nR; tiny
        r_α = S \ (Ĵ * W_prime .- g)
        view(ctx.buffers.rhs, 1:FOM)         .= W_prime .- C_prime * r_α
        view(ctx.buffers.rhs, (FOM+1):n_sys) .= r_α
    end
    return
end
```

The `system_matrix` buffer right-half columns are used as scratch for the multi-RHS
solve; this is valid since `_assemble_bordered_system!` has already written what it needs
into those columns and they are not read again.

### Files to modify

| File | Change |
|:-----|:-------|
| `CohomologicalSolver.jl` | Replace flat `lu!` / `ldiv!` with nR-branching Schur implementation |

---

## §6 — Multiindex generation: eliminate recursive `vcat()` *(Low)*

### Location

`src/Multiindices.jl` — `_generate_ascending_lex_fixed!` / inner `recurse`.

### Problem

The recursion calls `vcat(prefix, e)` at every level, generating `O(L × order)` small
heap allocations during the one-time construction of `MultiindexSet`. For degree=11,
NVAR=6 (`L ≈ 18 564`) this creates ~100k allocations.

### Fix

Replace with standard DFS-with-backtracking using a single pre-allocated buffer:

```julia
function _generate_ascending_lex_fixed!(exponents, NVAR, degree)
    buf = zeros(Int, NVAR)
    _recurse_dfs!(exponents, buf, 1, NVAR, degree)
end

function _recurse_dfs!(exponents, buf, depth, NVAR, remaining)
    if depth > NVAR
        push!(exponents, SVector{NVAR, Int}(buf))
        return
    end
    for e in 0:remaining
        buf[depth] = e
        _recurse_dfs!(exponents, buf, depth + 1, NVAR, remaining - e)
    end
    buf[depth] = 0   # restore for backtracking
end
```

One `SVector` allocation per completed multiindex; zero intermediate `Vector` allocations.

### Files to modify

| File | Change |
|:-----|:-------|
| `src/Multiindices.jl` | Replace `vcat`-based recursion with DFS-with-backtracking |

---

## §7 — Accept `SVector` in `indices_in_box_with_bounded_degree` *(Low)*

### Location

`src/Multiindices.jl` — `indices_in_box_with_bounded_degree`. Called in
`LowerOrderResources` constructor with `mset[i]` (an `SVector{NVAR,Int}`).

### Problem

The function signature requires a `Vector`, so the call site does `collect(mset[i])`,
allocating one `Vector{Int}` per monomial during setup.

### Fix

Widen the argument type to `AbstractVector{Int}` (or add an `SVector` method). Inspect
the function body to confirm `multi` is only read, never mutated — if so, no body changes
are needed.

### Files to modify

| File | Change |
|:-----|:-------|
| `src/Multiindices.jl` | Widen `multi` argument type |
| `SolverResources.jl` | Remove `collect()` at call site |

---

## §8 — Pre-allocate Horner scratch in `PropagateEigenmodes` *(Negligible)*

### Location

`PropagateEigenmodes.jl` — `propagate_right_jordan_vector` and
`propagate_left_jordan_vector`.

### Problem

Each call does `tmp_mat = copy(linear_terms[1])`, allocating a fresh `FOM × FOM` matrix.
This is called once per Jordan vector during eigenproblem setup — not a hot path.

### Fix

Add a `scratch::AbstractMatrix` parameter; the driver allocates it once before the Jordan
vector loop.

### Files to modify

| File | Change |
|:-----|:-------|
| `PropagateEigenmodes.jl` | Add `scratch` parameter; replace `copy(...)` with `copyto!(scratch, ...)` |
| Call site in `CohomologicalDriver.jl` | Allocate `scratch = similar(linear_terms[1])` once |

---

## §9 — Remove `monomial_map` from `ConjugateSymmetryData` *(Cleanup)*

### Context

After §2 (`solve_jobs`), `monomial_map` is never read from the struct in any hot-path
or downstream code — confirmed by full-source grep (zero hits outside
`ConjugateSymmetry.jl`). It is only used locally during `_build_conjugate_symmetry` to
construct `skip_bits` and `solve_jobs`.

### Fix

Make `monomial_map` a local variable inside `_build_conjugate_symmetry`; remove it from
the struct. Target struct after §2 and this cleanup:

```julia
struct ConjugateSymmetryData{CP}
    permutation      ::CP
    skip_bits        ::BitVector
    solve_jobs       ::Vector{NTuple{2, Int}}
    degree_boundaries::Vector{Int}
end
```

### Files to modify

| File | Change |
|:-----|:-------|
| `ConjugateSymmetry.jl` | Remove `monomial_map` field; demote to local variable |

---

## Implementation order

```
§3  detect_conjugate_permutation   — isolated new function, no struct changes
        ↓
§2  solve_jobs                     — replaces primary_pairs; adds degree_boundaries scaffold
        ↓
§9  Remove monomial_map            — cleanup after §2 removes the last hot-path use
        ↓
§1  Threading                      — builds on degree_boundaries from §2
        ↓
§4  FEM O4                         — independent of §1–§3; can proceed in parallel with §1
        ↓
§5  Dense Schur                    — independent; small targeted change
        ↓
§6–§8  Cleanup                     — independent, low-risk; can be batched
```

---

## Verification protocol (all items)

After each merged change:

1. **Test suite**: `GROUP=tests julia --project test/runtests.jl` — all tests must pass.
2. **Demo regression**: at least one Ferrite or Gridap demo must produce W, R matching
   the stored reference to `1e-10 * norm(W)`.
3. **Allocation check** (hot-path items): `@allocated solve_single_monomial!(...)` must
   remain below 800 bytes (only the LU metadata struct).
4. **Timing**: record `@benchmark solve_cohomological_problem(...)` before and after;
   document the speedup ratio.
