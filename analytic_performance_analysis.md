# Analytic Performance Analysis — MORFE.jl Cohomological Solver

## Notation

| Symbol | Meaning |
|--------|---------|
| N | FOM — full-order DOF count |
| n | NVAR = ROM + N\_EXT — reduced variable count |
| p | ORD — ODE order (1 for first-order, 2 for second-order) |
| γ | parametrisation order (maximum total monomial degree) |
| k | nonlinear term degree (k=2 quadratic, k=3 cubic, …) |
| NNZ | nonzeros in each linear operator matrix Bₖ |
| NNZ\_LU | nonzeros in KLU/Pardiso L+U factors (fill): O(N log N) [2-D], O(N^{4/3}) [3-D] |
| N\_fact | numeric factorisation flop count: O(N^{3/2}) [2-D], O(N²) [3-D] |
| M(γ,n) | total monomials = C(γ+n, n) ≈ γⁿ/n! |

---

## Monomial count

The multiindex set over n variables up to total degree γ contains

```
M(γ, n) = C(γ+n, n) ≈ γⁿ / n!
```

monomials (including the zero monomial; linear monomials are initialised from
eigenvectors and skipped). When conjugate symmetry is active the number of
monomials that must be solved is approximately M(γ,n)/2.

The order-d layer alone contains C(d+n−1, n−1) monomials.

---

## Linear system structure per monomial

For monomial α the cohomological system is the bordered linear system

```
┌              ┐ ┌       ┐   ┌         ┐
│  L(s)  C(s)  │ │  W[α] │ = │ RHS_inv │   N rows (invariance)
│  L̂(s)  Ĉ(s)  │ │  R_res│   │ RHS_ort │   nR rows (orthogonality)
└              ┘ └       ┘   └         ┘
```

where s = ⟨λ, α⟩ is the superharmonic, nR ≤ ROM is the number of resonant master
modes, L(s) = Σ Bₖ s^{k−1} is the parametrisation operator (N×N), and C(s)
(N×nR), L̂(s) (nR×N), Ĉ(s) (nR×nR) are bordered blocks.

---

## Is the factorisation reused?

### Symbolic factorisation — computed once

On the first monomial, `klu(L(s))` performs both the symbolic reordering (fill-
reducing permutation, sparsity of L and U factors) and the numeric factorisation.
The result is stored in `klu_cache::Ref{Any}`. All subsequent calls invoke
`klu!(F, L(s))`, which reuses the symbolic structure and recomputes only the
numeric values.

**Cost: O(NNZ_LU) once, amortised over all M(γ,n) monomials.**

#### Why this matters

The symbolic phase is purely combinatorial: it finds the fill-reducing permutation
(e.g. nested dissection) and determines the sparsity pattern of L and U via graph
analysis — no floating-point work is done. Because all monomials share the same
nonzero pattern (L(s) = Σ Bₖ s^{k−1} has the same structure for every s; only
the values change), this analysis is identical across all M(γ,n) monomials.

Without caching, each monomial would pay the full three-phase cost:

```text
symbolic  ~  O(NNZ_LU)   graph analysis + permutation  (combinatorial, memory-irregular)
numeric   ~  O(N_fact)   fill float values into L, U   (float ops)
solve     ~  O(NNZ_LU)   forward/back substitution
```

With the cache only numeric + solve are paid per monomial. The saving is:

```text
M(γ,n) × O(NNZ_LU)   ← avoided symbolic work
```

Note that O(NNZ_LU) < O(N_fact): the numeric factorisation costs more than the fill
count because eliminating the top-level separator block costs O(|sep|³) flops while
producing only O(|sep|²) fill entries. The symbolic saving is therefore a
sub-dominant fraction of the per-monomial cost for large N (O(N^{4/3}/N²) → 0 in
3-D), but the symbolic phase is memory-irregular and typically comparable in
wall-clock time to the numeric phase at the moderate FOM sizes (N ~ 10²–10⁴)
common in MORFE applications, so caching it yields a meaningful constant-factor
saving in practice.

### Numeric factorisation — one per monomial, not reused

Every monomial α has a distinct superharmonic s = ⟨λ, α⟩. Because
L(s) = Σ Bₖ s^{k−1} depends on s, a fresh numeric factorisation is required
for every monomial. The code does not detect or exploit collisions in s values.

**Cost: O(N_fact) per monomial × M(γ,n) monomials.**  
(O(N^{3/2}) per monomial for 2-D FEM; O(N²) per monomial for 3-D FEM.)

For Pardiso the same split applies: phase 11 (reordering) is done once; phases
22 (numeric) and 33 (solve) are executed per monomial.

### Resonant path: one factorisation, multiple right-hand sides

When nR > 0 the sparse path assembles an extended right-hand-side matrix of
size N × (1 + nR) — the rhs vector plus the nR columns of C(s) — and passes it
to `_sparse_solve` in a single call. This yields 1 + nR solutions from one
numeric factorisation, after which a dense (nR × nR) Schur complement eliminates
the reduced-dynamics unknowns.

**Cost: 1 numeric factorisation + (1+nR) triangular solves + O(nR² × N) for the
Schur complement, per resonant monomial.**

---

## Per-monomial cost breakdown (sparse path)

### Non-resonant case (nR = 0)

| Step | Cost | Location |
|------|------|----------|
| Superharmonic s = ⟨λ,α⟩ | O(n) | `CohomologicalEquations.jl` |
| Lower-order ξ — degree-1 part | O(n² · p · N) | `_sum_degree_one_terms!` |
| Lower-order ξ — higher-degree part | O(#candidates · n · p · N) | `_sum_higher_degree_terms!` |
| Nonlinear RHS replay | O(#factorisations · N) | `compute_multilinear_terms!` |
| Sparse L(s) assembly (Horner) | O(p · NNZ) | `build_sparse_L_and_rhs!` |
| KLU numeric refactorisation | O(N_fact) — O(N^{3/2}) [2-D] / O(N²) [3-D] | `_sparse_solve` |
| Triangular solve (1 RHS) | O(NNZ_LU) — O(N log N) [2-D] / O(N^{4/3}) [3-D] | `_sparse_solve` |
| Write W[:,1,α], R[:,α] | O(N) | `solve_single_monomial!` |
| Higher-derivative coefficients (p>1) | O(p · N) | `compute_higher_derivative_coefficients!` |

### Additional cost for the resonant case (nR > 0)

| Step | Extra cost |
|------|-----------|
| Evaluate nR columns of C(s) | O(nR · p · N) |
| Triangular solves for nR extra RHS | O(nR · NNZ_LU) |
| Orthogonality assembly L̂(s), Ĉ(s) | O(p · nR · N) |
| Schur complement J\_r · C\_prime | O(nR² · N) |
| Dense nR × nR solve | O(nR³) |

### Dense path

Replaces the sparse steps with a dense `lu!` on the bordered (N+nR)×(N+nR)
system. Dominant cost is **O(N³)** per monomial — only feasible for N ≲ a few
hundred.

---

## Total cost up to order γ

### Linear system solves

Numeric factorisation and triangular solve have different costs.
Mesh-dependent values with nested-dissection fill-reducing ordering:

| Mesh type | NNZ  | Fill (NNZ_LU)  | Numeric factorisation (N_fact) | Triangular solve |
|-----------|------|----------------|--------------------------------|------------------|
| 2-D FEM   | O(N) | O(N log N)     | O(N^{3/2})                     | O(N log N)       |
| 3-D FEM   | O(N) | O(N^{4/3})     | O(N^2)                         | O(N^{4/3})       |

Complete cost formula (all additive terms):

**2-D FEM:**

```text
O(N log N)                       [symbolic factorisation, once]
+ M(γ,n) × O(N^{3/2})           [numeric refactorisations]
+ M(γ,n) × O(N log N)           [triangular solves]
```

Dominated by: O(M(γ,n) × N^{3/2})

**3-D FEM:**

```text
O(N^{4/3})                       [symbolic factorisation, once]
+ M(γ,n) × O(N^2)                [numeric refactorisations]
+ M(γ,n) × O(N^{4/3})           [triangular solves]
```

Dominated by: O(M(γ,n) × N^2)  ≈  O(γⁿ/n! × N^2)

### Sparse L(s) assembly

```
M(γ,n) × O(p · NNZ)
```

Linear in NNZ per monomial; fast relative to the solve.

### Lower-order coupling ξ

At degree d, the candidate set of a monomial α consists of all β with |β| ∈ [2,
d−1] and β ≤ α componentwise. Summing over all monomials and degrees:

```
Total ξ cost ≈ O(γ^{2n} · n · p · N / (n!)²)
```

This grows as γ^{2n}, faster than the monomial count γⁿ, and can become the
dominant cost for large n or γ when N is moderate.

### Nonlinear RHS — how the factorisation count changes with order

For a degree-k nonlinear term, the number of ways to write a degree-d monomial
α as an ordered k-tuple (β₁, …, βₖ) with each |βᵢ| ≥ 1 is proportional to
C(d−1+n−1, n−1)^{k−1} ~ d^{(k−1)(n−1)}.

Summing over all monomials up to order γ:

| k | Per-monomial at degree d | Total to order γ |
|---|--------------------------|------------------|
| 2 (quadratic) | ~ C(d−1+n−1, n−1) ~ d^{n−1}/(n−1)! | O(γ^{2n}) |
| 3 (cubic) | ~ d^{2(n−1)} | O(γ^{3n}) |
| k (general) | ~ d^{(k−1)(n−1)} | O(γ^{kn}) |

The `build_multilinear_terms_cache` call precomputes all factorisation
bookkeeping (index lists) once before the solve loop. Per-monomial runtime cost
is then purely replaying stored entries:

```
Per monomial (degree d, term degree k):  O(#factorisations(d,k) × N)
Total to order γ (term degree k):        O(γ^{kn} × N / (n!)^k)
```

The cache build itself has the same O(γ^{kn}) index cost but no FOM-level work.

### Summary of total scaling

```text
Time ≈  O(NNZ_LU)                              (symbolic factorisation, once)
      + M(γ,n) × O(N_fact)                     (numeric refactorisations)
      + M(γ,n) × O(NNZ_LU)                     (triangular solves)
      + M(γ,n) × O(p · NNZ)                    (L(s) assembly)
      + O(γ^{2n} · n · p · N)                  (lower-order couplings)
      + O(γ^{kn} · N)                           (nonlinear RHS, per degree-k term)

with  M(γ,n) = C(γ+n,n) ≈ γⁿ/n!
      N_fact  = O(N^{3/2}) [2-D FEM],   O(N²)      [3-D FEM]
      NNZ_LU  = O(N log N) [2-D FEM],   O(N^{4/3}) [3-D FEM]
      NNZ     = O(N)
```

---

## Memory cost up to order γ

| Object | Size | Notes |
|--------|------|-------|
| Parametrisation W | O(N · p · M(γ,n)) complex floats | Dominant for large N and γ |
| Reduced dynamics R | O(n · M(γ,n)) complex floats | Small |
| MultilinearTermsCache | O(total factorisations) integers | Scales as O(γ^{kn}) |
| LowerOrderResources.candidate\_indices | O(Σ #candidates per monomial) integers | Scales as O(γ^{2n}) |
| KLU factorisation (L, U, permutations) | O(NNZ_LU) complex floats | Fixed once sparsity is set |
| SparseLinearSolverState.rhs\_extended | O(N · (ROM+1)) | Fixed |
| CohomologicalBuffers | O(N · ROM) | Fixed |

W dominates storage for large N; the index caches (MultilinearTermsCache,
candidate\_indices) dominate for large γ or n.

---

## Dominant bottlenecks

1. **Monomial count M(γ,n) ~ γⁿ/n!** — exponential in n. The curse of
   dimensionality in ROM size. Conjugate symmetry halves the effective count.

2. **One numeric refactorisation per monomial** — symbolic is cached but numeric
   must be repeated because s changes. For 3-D FEM, each refactorisation is
   O(N²) (nested dissection); total O(M(γ,n) · N²). The triangular solve per
   monomial is cheaper at O(N^{4/3}) but sub-dominant.

3. **Lower-order couplings** — total FOM-level work scales as O(γ^{2n}), growing
   faster than the monomial count. Hot path is the FOM-length `axpy!` inside
   `_sum_higher_degree_terms!`.

4. **Multilinear RHS factorisations** — total replay work O(γ^{kn} · N) per
   degree-k term; the k=3 cubic case grows especially fast with γ and n.

5. **Dense path** is O(M(γ,n) · N³) — completely infeasible for N ≳ 500.
