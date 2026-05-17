# Multi-threaded cohomological solve — analysis

## 1. What the change does

`solve_cohomological_problem` iterates monomials in GrLex order (ascending total
degree) and solves one bordered linear system per monomial.  Each monomial at
degree `d` reads only `W[β]` and `R[β]` with `|β| < d`, so **all monomials at
the same degree are mutually independent**.  The proposed change introduces
`Threads.@threads` over each degree group, with a serial barrier after each
group before proceeding to the next.  The conjugate-fill step (which writes
the secondary monomial from the solved primary) remains serial and runs inside
the barrier.

---

## 2. Parallelism structure

### 2.1 Monomial distribution by degree (NVAR = 4, degree ≤ 9, conjugate symmetry active)

| Degree | Total monomials | Primaries (≈ half) | Ideal threads used at 4T | Ideal threads used at 8T |
|-------:|----------------:|-------------------:|-------------------------:|-------------------------:|
| 2      | 10              | 5                  | 4 (1 idle)               | 5 (3 idle)               |
| 3      | 20              | 10                 | 4                        | 8 (2 idle)               |
| 4      | 35              | 18                 | 4                        | 8 (2 idle)               |
| 5      | 56              | 28                 | 4                        | 8                        |
| 6      | 84              | 42                 | 4                        | 8 (6 idle at final step) |
| 7      | 120             | 60                 | 4                        | 8                        |
| 8      | 165             | 83                 | 4                        | 8 (3 idle at final step) |
| 9      | 220             | 110                | 4                        | 8 (6 idle at final step) |
| **Total** | **714**      | **356**            |                          |                          |

Starting from degree 5, thread utilisation is high at 4T; at 8T there is
residual imbalance (up to 3 idle threads at the last step of each group) but
most threads stay busy.

### 2.2 Effective serial solves under `Threads.@threads`

`Threads.@threads` uses a static-range scheduler.  The effective serial-
equivalent solve count per degree group is `ceil(primaries / n_threads)`.

| n_threads | Effective serial steps | Theoretical speedup on solve loop |
|----------:|----------------------:|----------------------------------:|
| 1         | 356                   | 1.00×                             |
| 2         | 2+5+9+14+21+30+42+55 = 178 | 2.00×                        |
| 4         | 2+3+5+7+11+15+21+28 = 92 | 3.87×                          |
| 8         | 1+2+3+4+6+8+11+14 = 49 | 7.27×                           |
| 16        | 1+1+2+2+3+4+6+7 = 26 | 13.7× (capped by group sizes)   |

These are **ideal** numbers; real overhead reduces them (Section 4).

### 2.3 NVAR = 2 (Ferrite beam demo, single conjugate pair)

| Degree | Primaries (with conj sym) | Effective at 4T | Effective at 8T |
|-------:|-------------------------:|----------------:|----------------:|
| 2      | 2                        | 1               | 1               |
| 3      | 2                        | 1               | 1               |
| 4      | 3                        | 1               | 1               |
| 5      | 3                        | 1               | 1               |
| 6      | 4                        | 1               | 1               |
| 7      | 4                        | 1               | 1               |
| 8      | 5                        | 2               | 1               |
| 9      | 5                        | 2               | 1               |
| **Total** | **28**               | **10**          | **8**           |

Ideal speedup on solve loop: **2.8× at 4T**, **3.5× at 8T** — but most groups
have only 2–5 primaries so thread-scheduling overhead (see Section 4.3) is
significant relative to the per-monomial work.

---

## 3. Profiling context (from existing data)

From profiling the Ferrite beam demo (FOM ≈ 4 977 free DOFs, NVAR = 2,
degree = 9, dense path):

| Component | Fraction of total wall time |
|:----------|:--------------------------|
| `lu!` / `getrf!` | 62.9 % |
| RHS assembly (multilinear terms + lower-order couplings) | ~30 % |
| Overhead (eigenproblems, context build, fills) | ~7 % |

All of the 62.9 % + ~30 % = ~93 % sits inside the per-monomial solve loop
and is fully parallelisable within each degree group.  Only the ~7 %
(eigenproblems, context build, conjugate fills, degree barriers) is serial.

---

## 4. Performance analysis

### 4.1 Ideal parallel speedup (Amdahl's Law)

Let $P = 0.93$ (parallelisable fraction of total time; conservative, using
the profiling breakdown above).

$$S_n^{\text{Amdahl}} = \frac{1}{(1 - P) + P/n}$$

| n_threads | Amdahl limit | Degree-group limit (NVAR=4) | Min of the two |
|----------:|-------------:|----------------------------:|---------------:|
| 2         | 1.98×        | 2.00×                       | **1.98×**      |
| 4         | 3.59×        | 3.87×                       | **3.59×**      |
| 8         | 5.89×        | 7.27×                       | **5.89×**      |
| 16        | 8.25×        | 13.7×                       | **8.25×**      |

For NVAR = 4, Amdahl's 7 % serial floor is the binding constraint from 8T
upwards.

For NVAR = 2, the degree-group size is the binding constraint:

| n_threads | Amdahl limit | Degree-group limit (NVAR=2) | Min of the two |
|----------:|-------------:|----------------------------:|---------------:|
| 2         | 1.98×        | 2.80×                       | **1.98×**      |
| 4         | 3.59×        | 2.80×                       | **2.80×**      |
| 8         | 5.89×        | 3.50×                       | **3.50×**      |

### 4.2 Conservative estimates

Three sources of efficiency loss are applied multiplicatively:

| Factor | Conservative value | Rationale |
|:-------|:-------------------|:---------|
| Memory bandwidth saturation | 0.75 | Dense LU (`dgetrf!`) is memory-bound at large FOM; multiple threads competing for L3/DRAM bandwidth reduce per-core throughput by 25 % |
| `Threads.@threads` scheduling + barrier | 0.90 | Per-group barrier cost ≈ 1 × T_solve equivalent spread across 8 groups; ~10 % overhead |
| Julia GC pressure from per-thread allocations | 0.95 | Minor residual allocs in resonant path (small `S` matrix); GC pauses rare but possible |
| **Combined** | **0.64** | |

**Conservative wall-time speedup estimates for the cohomological solve:**

| Scenario | n_threads | Ideal | Conservative (×0.64) |
|:---------|----------:|------:|---------------------:|
| NVAR=4, degree=9 (SSM study) | 4  | 3.59× | **2.3×** |
| NVAR=4, degree=9 (SSM study) | 8  | 5.89× | **3.8×** |
| NVAR=2, degree=9 (Ferrite beam) | 4 | 2.80× | **1.8×** |
| NVAR=2, degree=9 (Ferrite beam) | 8 | 3.50× | **2.2×** |

These are estimates for the **solve loop only** (which is 93 % of total time).
End-to-end `solve_cohomological_problem` speedup is bounded by Amdahl's 7 %
serial overhead: theoretical maximum ≈ 14× regardless of thread count.

### 4.3 Regime where threading hurts

For small FOM (< 200), each monomial solve runs in microseconds.  The
`Threads.@threads` barrier overhead (thread wake-up, scheduler interaction)
is on the order of 1–10 µs per group and 8 groups means 8–80 µs of pure
overhead.  For FOM = 100 with a ~50 µs solve, the overhead alone is 16–160 %
of the work.  Threading is a **net regression** for FOM < ~500 on dense
problems and should not be used in that regime.

---

## 5. Memory overhead

### 5.1 Per-thread resource costs

| Resource | Formula | FOM = 1 000, T=8 | FOM = 5 000, T=8 |
|:---------|:--------|:-----------------|:-----------------|
| `system_matrix` (dense) | `(FOM+ROM)² × 16 B` | **128 MB** | **3.2 GB** |
| `system_matrix` (sparse) | same allocation (wasteful) | 128 MB | 3.2 GB |
| `lower_order_buffer` | `ORD × FOM × 16 B` | 0.25 MB | 1.3 MB |
| `rhs_extended` | `FOM × (ROM+1) × 16 B` | 0.08 MB | 0.4 MB |
| `ml_cache scratch` | `3 × FOM × 16 B` | 0.24 MB | 1.2 MB |
| `L_template deepcopy` | `nnz × 16 B` (nnz ≈ 10×FOM) | 0.16 MB | 0.8 MB |

For the **dense path** at FOM = 5 000 with 8 threads: `system_matrix` alone
requires **3.2 GB** extra.  This is prohibitive; dense solves at FOM = 5 000
are already unusual (typical dense regime: FOM ≤ 2 000).  At FOM = 1 000
the cost is 128 MB — acceptable on a workstation.

For the **sparse path**, `system_matrix` is currently over-allocated at
`(FOM+ROM)²` even though the sparse Schur complement only touches
`FOM × nR` entries (nR ≤ ROM = 4 in practice).  This means the sparse
per-thread cost is identical to the dense path, which is unnecessary.
A targeted fix (allocate only `FOM × nR + nR × (FOM+nR)` scratch for the
sparse path) would reduce per-thread overhead from 3.2 GB to **< 1 MB** at
FOM = 5 000.  This fix is **independent** of the threading plan and worth
tracking.

### 5.2 One-time construction cost

`deepcopy(L_template)` is called once per thread at setup time.  For a
typical sparse FEM matrix (FOM = 5 000, nnz ≈ 50 000):

- Copy cost ≈ 50 000 × 16 B = 0.8 MB per thread
- Wall-clock time: < 1 ms per thread (memcpy-bound at ~10 GB/s)

Negligible in context of a multi-second solve.  For FOM = 100 000 (nnz ≈ 10⁶):
16 MB per thread, 8 threads = 128 MB allocated at startup; still fast (~10 ms).

---

## 6. Summary: pros and cons

### Pros

**P1 — Large speedup for NVAR ≥ 4 studies.**  Conservative estimate: **2–4×**
wall-time reduction for the standard NVAR=4, degree=9 case on a 4–8 core
workstation.  The largest single optimisation available; no algorithmic change
is required, only parallelism.

**P2 — Minimal serial-path overhead.**  With `n_threads=1`, the only changes
are the Phase A struct refactor (loop rewrite from `primary_pairs` to
`solve_jobs`) and the driver dispatch.  There are zero new allocations in the
hot path.

**P3 — Conceptually clean parallelism.**  The degree-group independence is a
formal property of the GrLex ordering, not an approximation.  Results are
bit-reproducible relative to serial for `n_threads=1`, and within floating-
point round-off for `n_threads > 1` (only reassociation differences).

**P4 — Pardiso users get a clear path.**  When Pardiso is desired, users set
`n_threads=1` (exploiting Pardiso's own MKL threading).  When they want Julia-
level parallelism (KLU), they set `n_threads > 1`.  Both paths co-exist.

**P5 — Enables the highest-degree expansions.**  At degree = 11, NVAR = 4 with
conjugate symmetry: ~940 primaries, degree-11 group alone has ~300.  Threading
efficiency improves with order; this feature pays for itself most at the
extreme orders typically needed for practical SSM computation.

### Cons and quantified risks

**C1 — Dense path at large FOM: memory blowup.**
`CohomologicalBuffers.system_matrix` is allocated at `(FOM+ROM)²` per thread,
even on the sparse path.  For FOM = 5 000, 8 threads: **3.2 GB** extra memory.
This is untenable.  The sparse path does not need a dense `(FOM+ROM)²` matrix;
it only needs `FOM × nR` and `nR × (FOM+nR)` scratch.  The fix is simple
(allocate smaller scratch on the sparse path) but is a separate change.
**Until fixed, threading should only be enabled on the sparse path** or for
FOM ≤ ~1 500 on the dense path.

**C2 — Pardiso downgrade is silent on the hot path.**
When `n_threads > 1`, all per-thread sparse solvers are constructed with
`pardiso = nothing`, falling back to KLU.  If KLU is 2–5× slower than Pardiso
for a particular matrix (common for large bandwidth FEM matrices), using 4
Julia threads + KLU could be **slower** than 1 Julia thread + Pardiso +
MKL's 16 threads.  A `@warn` at call time is necessary but not sufficient;
the documentation must explicitly state the trade-off and give the user a
way to benchmark both configurations.

**C3 — No benefit for NVAR = 2 below FOM ≈ 500.**
For the standard single-mode Ferrite beam (NVAR = 2, degree = 9, conjugate
symmetry): max 5 primaries per degree group.  Conservative speedup at 4T: **1.8×**.
For FOM < 500, each solve runs in < 0.5 ms and `Threads.@threads` barrier
overhead (~1–5 µs per group) erodes the gain; net speedup could be **< 1.05×**
or even negative.  The code should check `n_threads > 1 && FOM < threshold`
and emit a warning.

**C4 — Code duplication: `_solve_single_threaded!` vs `solve_single_monomial!`.**
Two nearly-identical ~50-line functions exist after the change.  Any future
fix (e.g. a new resonant path, a new derivative term) must be applied to both.
The compiler cannot detect divergence.  Conservative estimate: within 12 months
a one-line bug fix is applied to `solve_single_monomial!` and not to
`_solve_single_threaded!`.  Mitigation: encapsulate the mutable resource
access behind a thin abstraction (a resource-provider type) and have both
functions call a shared inner implementation.  This adds one level of
indirection but eliminates the duplication.

**C5 — Progress reporting degrades from per-monomial to per-degree.**
The current implementation reports progress on every solved monomial (~350
ticks for NVAR=4 degree=9).  The threaded loop ticks once per degree group
(8 ticks total).  Time-to-completion estimates based on the progress bar
become an order of magnitude coarser — effectively useless for monitoring
long runs.  A per-monomial atomic counter (`Threads.Atomic{Int}`) in the
threaded loop would restore granularity at negligible cost.

**C6 — Inactive-path `_build_conjugate_symmetry` signature break.**
The third argument changes from `L::Int` to `mset::MultiindexSet`.  There
are at least two call sites in `CohomologicalEquations.jl` (line 325) and
`CohomologicalDriver.jl` that must be updated.  If any test or external
script passes an `Int`, it will dispatch to the wrong method (or throw
`MethodError` at runtime) with no compile-time warning.

**C7 — Thread-imbalance at low-degree groups.**
Degree 2 with conjugate symmetry has 5 primaries; degree 3 has 10.  At
`n_threads = 8`, degrees 2 and 3 use only 5 and 8 threads respectively,
leaving 3 and 0 idle.  These groups contribute less than 5 % of total solve
time (their FOM-scale solves are the cheapest), so the absolute wasted time
is small — but it means real speedup is always less than the theoretical
maximum tabulated in Section 4.

**C8 — `me > 0` FEM terms: latent data race if not fixed.**
`_replay_fem_split!` uses `fem_qp_buffer(t)`, a scratch buffer owned by the
`FEMMultilinearMap` term object.  If two threads process different monomials
with the same FEM term (which they will), this is a data race.  The fix
(add `∇W_qp` keyword to `_replay_fem_split!`) is small but affects an
internal FEM backend API.  Missing this fix in the initial PR would produce
silently wrong results **only for models with external-forcing FEM terms**
(`me > 0`); most structural problems without external forcing are unaffected.

---

## 7. Recommended approach

| Item | Action |
|:-----|:-------|
| Phase A: `solve_jobs` + `degree_boundaries` | Proceed; also fixes serial-path loop clarity |
| C1 fix (sparse path `system_matrix` over-allocation) | **Fix before enabling threading** on sparse path; otherwise prohibitive memory |
| C5 fix (progress counter) | Add `Threads.Atomic{Int}` counter in threaded loop; trivial |
| C8 fix (`me > 0` FEM race) | Required before threading models with external FEM forcing |
| C4 (code duplication) | Accept for now; schedule refactor to shared inner function in follow-up |
| C2 (Pardiso warning) | Add explicit `@warn` at `n_threads > 1 && pardiso !== nothing` in the driver |
| C3 (small-FOM regression) | Add guard: `n_threads > 1 && FOM < 500 && !(MT <: SparseMatrixCSC)` → `@warn` |

Threading provides meaningful returns only for FOM > ~500 and NVAR ≥ 4, or
for FOM > ~2 000 and NVAR = 2.  For the primary target case (large sparse FOM,
NVAR = 4, degree ≥ 7), after fixing C1 the conservative expected speedup is
**2–4× on a 4–8 core workstation**, with the absolute gain growing with both
NVAR and polynomial order.
