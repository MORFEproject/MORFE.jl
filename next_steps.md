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
| `_replay_term!` multiple dispatch — no `isa` in `compute_multilinear_terms!` | ✅ |
| FEM O4 combined element loop (`FEMGlobalEntry`, `FEMGlobalSplit`, `_replay_all_fem_splits!`) | ✅ |
| Allocation-free multiindex generation (pre-allocated matrix + in-place buffer, no `vcat`) | ✅ |
| `indices_in_box_with_bounded_degree` accepts `AbstractVector{Int}` — no `collect()` at call site | ✅ |
| `detect_conjugate_permutation` standalone utility, exported | ✅ |
| Horner scratch kwarg in `propagate_left/right_jordan_vector` | ✅ |
| Sparse-path Schur complement for resonant block | ✅ |

---

## Priority table

| # | Item | Impact | Effort | Status | Section |
|:--|:-----|:-------|:-------|:-------|:--------|
| 1 | Thread-parallel solve by degree group | **Critical** | Large | ✅ | §1 |
| 2 | `solve_jobs` flat work list | Medium | Small | ✅ | §2 |
| 3 | `detect_conjugate_permutation` standalone utility | Medium (UX) | Trivial | ✅ | §3 |
| 4a | `_replay_term!` multiple dispatch (prerequisite to §4) | Cleanup | Trivial | ✅ | §4a |
| 4 | FEM O4 — combined element loop | **Critical** (FEM) | Large | ✅ | §4 |
| 5 | Dense-path Schur complement for resonant block | Low–Medium | Small | ✅ | §5 |
| 6 | Multiindex generation: eliminate `vcat()` | Low | Small | ✅ | §6 |
| 7 | Accept `SVector` in `indices_in_box_with_bounded_degree` | Low | Trivial | ✅ | §7 |
| 8 | Pre-allocate Horner scratch in `PropagateEigenmodes` | Negligible | Trivial | ✅ | §8 |
| 9 | Remove `monomial_map` from `ConjugateSymmetryData` | Cleanup | Trivial | ✅ | §9 |

**Not pursued**: Real arithmetic for self-conjugate monomials. Self-conjugate monomials
are a small fraction of L (which is itself small), so their 4× LU speedup is swamped by
the cost of the non-self-conjugate primaries. The implementation complexity is not
justified.

---

## §1 — Thread-parallel solve by degree group *(Critical)* ✅

`solve_cohomological_equations_threaded!` added to `CohomologicalEquations.jl`.
`ThreadedMonoSolveResources{T}` (per-thread buffers + lo_buffer) added to `SolverResources.jl`.
`alloc_thread_resources` allocates one bundle per thread.
`solve_cohomological_problem` gains `n_threads::Int = 1` kwarg; sparse models fall back to
serial with `@warn`. The degree-barrier pattern ensures intra-degree parallelism is correct:
all degree-`d` primary solves complete before conjugate fills, which complete before
degree-`d+1` begins. `compute_multilinear_terms!` is serialized via `ReentrantLock` (O(FOM)
vs O(FOM³) for LU — negligible overhead for large FOM).

---

## §2 — `solve_jobs` flat work list *(Medium)* ✅

`primary_pairs` replaced by `solve_jobs::Vector{NTuple{2,Int}}` in `ConjugateSymmetryData`.
`degree_boundaries::Vector{Int}` added as a field and populated by `_build_degree_boundaries`.
Both `_build_conjugate_symmetry` overloads (inactive and active) build `solve_jobs` and
call `_build_degree_boundaries`. Solve loops in `CohomologicalEquations.jl` iterate
`sym.solve_jobs` directly instead of checking `skip_bits` per iteration.

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

## §4 — FEM O4: combined element loop across all FEM terms *(Critical for FEM)* ✅

`FEMGlobalEntry{DEG}` and `FEMGlobalSplit{ENTRIES_TUPLE}` are defined in `MultilinearTerms/Structs.jl`.
`_replay_all_fem_splits!` and `_accumulate_global_entries!` (recursive tuple-peeling) are in `CachedEval.jl`.
`MultilinearTermsCache` carries `global_fem_splits`, `global_∇W_qp`, and `global_Fe_buffers`.
`compute_multilinear_terms!` dispatches the combined loop first, then closure terms.

---

## §5 — Dense-path Schur complement for resonant monomials *(Low–Medium)* ✅

`_dense_solve_bordered!(buffers, FOM, nR)` extracted in `CohomologicalSolver.jl`.
For `nR == 0`: identical to the previous path (`lu!` + `ldiv!` on the FOM×FOM block).
For `nR > 0`: factors only the FOM×FOM block, then two separate `ldiv!` calls (RHS
in-place, then C_r columns in-place reusing their existing slots as scratch), followed by
the tiny nR×nR Schur complement. The `_solve_monomial!` dense overload now calls this helper.

---

## §6 — Multiindex generation: eliminate recursive `vcat()` *(Low)* ✅

`_generate_ascending_lex_fixed!` already uses a pre-allocated `Matrix{Int}` written
column-by-column and a single `buf = Vector{Int}(undef, n)` mutated in-place during
recursion. No `vcat` calls; one `SVector` allocation per completed multiindex.

---

## §7 — Accept `SVector` in `indices_in_box_with_bounded_degree` *(Low)* ✅

Function signature is `(set, box_upper::AbstractVector{Int}, ...)`. Call site in
`SolverResources.jl` passes `mset[i]` (an `SVector`) directly — no `collect()`.

---

## §8 — Pre-allocate Horner scratch in `PropagateEigenmodes` *(Negligible)* ✅

`propagate_left_jordan_vector` and `propagate_right_jordan_vector` both accept
`scratch::Union{Nothing, AbstractMatrix} = nothing`; when provided, use `copyto!`
instead of `copy`. These are user-facing functions; the driver does not call them.

---

## §9 — Remove `monomial_map` from `ConjugateSymmetryData` *(Cleanup)* ✅

`monomial_map` demoted to a local variable in `_build_conjugate_symmetry` (active overload).
Final struct has four fields: `permutation`, `skip_bits`, `solve_jobs`, `degree_boundaries`.

---

## Implementation order

All items complete.

```
✅ §3  detect_conjugate_permutation   — done
✅ §4a _replay_term! dispatch         — done
✅ §4  FEM O4                         — done
✅ §6–§8  Cleanup                     — done
✅ §2  solve_jobs + degree_boundaries — done
✅ §9  Remove monomial_map            — done
✅ §1  Threading                      — done (n_threads kwarg; dense path only)
✅ §5  Dense Schur (dense path)       — done
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
