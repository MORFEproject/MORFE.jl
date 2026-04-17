# Cohomological Equations — Design & Implementation Documentation

## Table of Contents

1. [Purpose and context](#1-purpose-and-context)
2. [Mathematical background](#2-mathematical-background)
   - 2.1 [The invariance equation](#21-the-invariance-equation)
   - 2.2 [The cohomological system](#22-the-cohomological-system)
   - 2.3 [Superharmonic frequency](#23-superharmonic-frequency)
   - 2.4 [Resonance and reduced dynamics](#24-resonance-and-reduced-dynamics)
3. [Data layout](#3-data-layout)
   - 3.1 [Parametrisation `W`](#31-parametrisation-w)
   - 3.2 [Reduced dynamics `R`](#32-reduced-dynamics-r)
   - 3.3 [External dynamics embedding](#33-external-dynamics-embedding)
4. [Precomputed operators](#4-precomputed-operators)
   - 4.1 [Invariance-equation operators](#41-invariance-equation-operators)
   - 4.2 [Orthogonality-condition operators](#42-orthogonality-condition-operators)
   - 4.3 [CohomologicalContext](#43-cohomologicalcontext)
5. [Solve pipeline](#5-solve-pipeline)
   - 5.1 [Initialisation of linear monomials](#51-initialisation-of-linear-monomials)
   - 5.2 [External forcing directions](#52-external-forcing-directions)
   - 5.3 [Main solve loop](#53-main-solve-loop)
   - 5.4 [Causal (GrLex) ordering](#54-causal-grlex-ordering)
6. [Per-monomial solve](#6-per-monomial-solve)
   - 6.1 [Lower-order couplings](#61-lower-order-couplings)
   - 6.2 [Stacked linear system](#62-stacked-linear-system)
   - 6.3 [Higher time-derivative coefficients](#63-higher-time-derivative-coefficients)
7. [Design decisions](#7-design-decisions)

---

## 1. Purpose and context

`CohomologicalEquations` is the top-level solver in the parametrisation method pipeline.  It orchestrates:

- one-time precomputation of polynomial operator coefficients,
- a causal monomial-by-monomial loop that updates `W` and `R` in-place, and
- extraction of the external forcing directions from the linear cohomological equations.

The public entry point is `solve_cohomological_problem`, which takes raw spectral data (eigenvalues, eigenvectors, resonance set) and a full-order model, and returns the solved pair `(W, R)`.

---

## 2. Mathematical background

### 2.1 The invariance equation

Consider an $N$-th order ODE in $\mathbb{R}^n$:

$$\sum_{k=0}^{N} B_k\, x^{(k)} = F(x, \dot{x}, \ldots) + f_{\text{ext}}(r, t)$$

where $r \in \mathbb{C}^{N_\text{ext}}$ satisfies the autonomous linear external system $\dot{r} = \Lambda_\text{ext}\, r$.  The parametrisation method seeks a polynomial map

$$x(t) = W(\mathbf{z}(t),\, r(t))$$

from $\text{NVAR} = \text{ROM} + N_\text{ext}$ reduced variables to the physical state, such that the image of $W$ is an invariant manifold.  The reduced dynamics are

$$\dot{z}_i = \sum_{\alpha} R_{i,\alpha}\, z^\alpha, \qquad i = 1,\ldots,\text{ROM}$$

Substituting the ansatz and matching monomial coefficients yields one linear system per multi-index $\alpha$ — the **cohomological equations**.

### 2.2 The cohomological system

For each multi-index $\alpha$ with superharmonic $s = \langle\lambda, \alpha\rangle$, the cohomological system reads:

```
┌              ┐ ┌         ┐   ┌           ┐
│  L(s)  C(s)  │ │  W[α]   │ = │  RHS_inv  │   FOM rows  (invariance)
│  L̂(s)  Ĉ(s)  │ │  R_res  │   │  RHS_ort  │   nR  rows  (orthogonality)
└              ┘ └         ┘   └           ┘
```

where:

| Block | Size | Role |
|-------|------|------|
| $L(s)$ | $n \times n$ | Parametrisation operator; encodes the linear part of the ODE evaluated at frequency $s$ |
| $C(s)$ | $n \times n_R$ | Couples the resonant reduced-dynamics unknowns into the invariance equation |
| $\hat{L}(s)$ | $n_R \times n$ | Left orthogonality rows; projects the invariance equation onto left eigenmodes |
| $\hat{C}(s)$ | $n_R \times n_R$ | Orthogonality block for the resonant unknowns |
| $W[\alpha]$ | $n$ | Unknown parametrisation coefficient (zeroth time-derivative) |
| $R_\text{res}$ | $n_R$ | Unknown resonant reduced-dynamics coefficients |

Non-resonant master modes have $R_{r,\alpha} = 0$ by construction and are excluded from the system.  External dynamics entries $R_{\text{ROM}+e,\, \alpha}$ are pre-filled from the model and never solved for.

### 2.3 Superharmonic frequency

Given eigenvalues $\lambda_1,\ldots,\lambda_\text{NVAR}$ (master + external), the **superharmonic frequency** for multi-index $\alpha$ is

$$s = \langle\lambda,\, \alpha\rangle = \sum_{i=1}^{\text{NVAR}} \lambda_i\, \alpha_i.$$

It appears as the evaluation point of all polynomial operators $L(s)$, $C(s)$, $\hat{L}(s)$, $\hat{C}(s)$.

The diagonal $(\lambda_1,\ldots,\lambda_\text{NVAR})$ is read directly from the reduced-dynamics polynomial $R$: the coefficient of the $i$-th unit-vector monomial $e_i$ gives the $i$-th column of the Jordan matrix $\Lambda$, whose diagonal entries are the $\lambda_i$.

### 2.4 Resonance and reduced dynamics

Master mode $r$ is **resonant** at $\alpha$ when $|\lambda_r - s|$ is small relative to the spectral gap.  This is encoded in the [`ResonanceSet`](@ref MORFE.Resonance.ResonanceSet), which maps each monomial index to a bitmask over the $\text{ROM}$ master modes.

For a resonant mode the reduced-dynamics coefficient $R_{r,\alpha}$ is an *unknown* solved simultaneously with $W[\alpha]$.  For a non-resonant mode it is set to zero.

---

## 3. Data layout

### 3.1 Parametrisation `W`

`W.poly.coefficients` has shape $(n,\, N,\, L)$ where $L = |\mathcal{A}|$ is the number of monomials:

```
W.poly.coefficients[:, k, l]  →  W^(k-1)[α_l]  (zeroth to (N-1)-th time derivative)
```

The zeroth-derivative slice `W.poly.coefficients[:, 1, l]` is the physical-state coefficient; higher slices `k > 1` give velocity and higher derivatives and are recovered from lower-order data via the recurrence in [`compute_higher_derivative_coefficients!`](@ref MORFE.ParametrisationMethod.compute_higher_derivative_coefficients!).

### 3.2 Reduced dynamics `R`

`R.poly.coefficients` has shape $(\text{NVAR},\, L)$:

```
R.poly.coefficients[1:ROM,    l]  →  master-mode dynamics at monomial l
R.poly.coefficients[ROM+1:NVAR, l]  →  external dynamics at monomial l  (pre-filled, read-only)
```

The first $\text{ROM}$ rows are solved by the cohomological equations.  The last $N_\text{ext}$ rows are copied from `model.external_system.first_order_dynamics` at initialisation and are never modified by the solver.

The **linear part of the reduced dynamics** is encoded in the first $\text{NVAR}$ columns: column $r$ of `R.poly.coefficients` holds the $r$-th column of the Jordan matrix $\Lambda$.  In particular, `R.poly.coefficients[r, r] = λ_r` (up to Jordan off-diagonals).

### 3.3 External dynamics embedding

The external system's polynomial $\dot{r} = g(r)$ is defined over $N_\text{ext}$ variables with multiindex set $\mathcal{A}_\text{ext}$.  To place these coefficients in the $\text{NVAR}$-variable space used by $W$ and $R$, each external monomial $\alpha_\text{ext} \in \mathbb{N}^{N_\text{ext}}$ is embedded as:

$$\alpha_\text{full} = (\underbrace{0,\ldots,0}_{\text{ROM}},\; \alpha_\text{ext}) \in \mathbb{N}^{\text{NVAR}}.$$

The function `_embed_external_dynamics!` performs this mapping once at initialisation.  The augmented multiindex set must contain every such $\alpha_\text{full}$; this is guaranteed if the monomial set was constructed to cover all monomials of the external system up to the desired expansion order.

---

## 4. Precomputed operators

### 4.1 Invariance-equation operators

The invariance-equation block $[L(s)\; C(s)]$ is assembled by [`InvarianceEquation.assemble_cohomological_matrix_and_rhs`](@ref MORFE.InvarianceEquation.assemble_cohomological_matrix_and_rhs) using two families of precomputed polynomial coefficients:

- **`invariance_C_coeffs[r]`** (shape $n \times N$): degree-$j$ coefficient of the master-mode column operator $C_r(s)$; one matrix per master mode $r = 1,\ldots,\text{ROM}$.
- **`invariance_E_coeffs[e]`** (shape $n \times N$): degree-$j$ coefficient of the external column operator $E_e(s)$; one matrix per external variable $e = 1,\ldots,N_\text{ext}$.

Both families are polynomial in $s$ and are evaluated at runtime via a Horner pass.  The master-mode coefficients depend only on the master modes and the Jordan matrix; the external coefficients depend additionally on the external forcing directions $\Phi_\text{ext}$ (the columns of $W$ at the external linear monomials).

### 4.2 Orthogonality-condition operators

The orthogonality block $[\hat{L}(s)\; \hat{C}(s)]$ is assembled by [`MasterModeOrthogonality.assemble_orthogonality_matrix_and_rhs`](@ref MORFE.MasterModeOrthogonality.assemble_orthogonality_matrix_and_rhs) using:

- **`orthogonality_J_coeffs[r]`** (shape $N \times n$): row operator coefficients for left eigenmode $r$.
- **`orthogonality_C_coeffs[r]`** (shape $(N{-}1) \times \text{ROM}$): column operator coupling master-mode unknowns.
- **`orthogonality_E_coeffs[r]`** (shape $(N{-}1) \times N_\text{ext}$): column operator for the external variables.

### 4.3 CohomologicalContext

All precomputed arrays, the resonance set, the pre-allocated solve buffers, and the lower-order coupling resources are bundled into the flat struct [`CohomologicalContext`](@ref MORFE.CohomologicalEquations.CohomologicalContext).  Using one flat struct avoids naming ambiguities between the invariance and orthogonality operator families and makes data provenance explicit.

---

## 5. Solve pipeline

### 5.1 Initialisation of linear monomials

Before the main loop, the linear monomials $e_r$ ($r = 1,\ldots,\text{ROM}$) are initialised from the spectral data:

```
W.poly.coefficients[:, 1, idx_{e_r}]  ←  φ_r       (master-mode eigenvector)
W.poly.coefficients[:, k, idx_{e_r}]  ←  W^(k)[e_r]  (higher derivatives, k = 2…N)
R.poly.coefficients[r, idx_{e_r}]     ←  λ_r       (master-mode eigenvalue)
```

In GrLex order the zero vector (if present) occupies index 1, so $e_r$ is at index $r$ (no zero vector) or $r + 1$. This allows O(NVAR) indexing instead of a full scan.

### 5.2 External forcing directions

The external forcing directions $\Phi_\text{ext}[:,e] = W[e_{\text{ROM}+e}]$ are *not* known from the spectral decomposition alone — they are the particular solutions of the invariance equation at the external forcing frequencies.  They are computed by a preliminary call to `solve_single_monomial!` for each external linear monomial $e_{\text{ROM}+e}$, using a partial context in which $\Phi_\text{ext} = 0$.

After these preliminary solves the full operator columns `invariance_E_coeffs` and the orthogonality coefficients are recomputed with the correct $\Phi_\text{ext}$, and the main context is assembled.

The key efficiency gain is that the master-mode operator columns `invariance_C_coeffs` depend only on master eigenvectors and the Jordan matrix, so they are computed **once** and reused in both passes.  The precomputed intermediate Horner steps (`D_master_steps`) are passed to the external-column computation to avoid repeating the master-column work.

### 5.3 Main solve loop

`solve_cohomological_equations!` iterates over all monomials in the multiindex set in index order (which equals GrLex order), skipping the pre-initialised linear monomials.  For each remaining monomial `idx` it calls `solve_single_monomial!`.

### 5.4 Causal (GrLex) ordering

The cohomological equation for monomial $\alpha$ depends only on lower-order coefficients $W[\beta]$ with $|\beta| < |\alpha|$.  Because the multiindex set is stored in GrLex order (ascending total degree), iterating `1:L` is already causal — no sort is required.

---

## 6. Per-monomial solve

`solve_single_monomial!` performs seven steps for each monomial:

1. **Superharmonic** $s = \langle\lambda, \alpha\rangle$ from `ctx.lambda_diag`.
2. **Resonance bitmask** from `ctx.resonance_set`.
3. **Lower-order coupling vectors** $\xi_j$ (length $n$) via [`LowerOrderCouplings.compute_lower_order_couplings`](@ref MORFE.LowerOrderCouplings.compute_lower_order_couplings); these capture the contribution of all previously solved monomials to the current RHS.
4. **Nonlinear RHS** via [`MultilinearTerms.compute_multilinear_terms`](@ref MORFE.MultilinearTerms.compute_multilinear_terms); evaluates all multilinear terms of the full-order model using the cached factorisation of the multiindex set.
5. **External dynamics** at the current monomial, read from `view(R.poly.coefficients, ROM+1:NVAR, idx)`.
6. **Stacked solve**: assemble the $(n + n_R) \times (n + n_R)$ system in a pre-allocated buffer and call `lu!` + `ldiv!` in-place.
7. **Higher derivatives**: recover `W^(k)[α]` for $k = 2,\ldots,N$ from the zeroth-derivative result and the lower-order couplings via [`compute_higher_derivative_coefficients!`](@ref MORFE.ParametrisationMethod.compute_higher_derivative_coefficients!).

### 6.1 Lower-order couplings

The lower-order coupling vector $\xi_j$ at derivative order $j$ collects contributions from all monomials $\beta$ that appear as factors of $\alpha$ in the time-differentiation recurrence.  It is assembled by iterating over all previously-solved monomials whose exponent is dominated by $\alpha$, using a pre-built candidate list `ctx.candidate_indices_by_monomial[idx]` to avoid a global scan.

### 6.2 Stacked linear system

The invariance equation and orthogonality conditions are assembled into a single square system:

```
[ M_inv  ] [ W[α]  ]   [ rhs_inv ]
[ M_ort  ] [ R_res ] = [ rhs_ort ]
```

of size $(n + n_R)$.  The system is written into a pre-allocated `(n + \text{ROM}) \times (n + \text{ROM})$ buffer, factorised in-place with `lu!`, and solved with `ldiv!` — no allocation occurs in the steady state.

### 6.3 Higher time-derivative coefficients

For an $N$-th order ODE the parametrisation carries $N$ derivative slices.  Given the zeroth-derivative solution $W^{(1)}[\alpha]$ and the lower-order couplings, the recurrence

$$W^{(k+1)}[\alpha] = s\, W^{(k)}[\alpha] + \Phi_\text{master}\, R_\text{res}[\alpha] + \Phi_\text{ext}\, e_\text{dyn} + \xi_k$$

is applied for $k = 1,\ldots,N-1$ by [`compute_higher_derivative_coefficients!`](@ref MORFE.ParametrisationMethod.compute_higher_derivative_coefficients!).

---

## 7. Design decisions

### Type parameters extracted at compile time

`solve_cohomological_problem` is typed as `NDOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT}`, making `ORD`, `ORDP1`, `N_EXT`, and the matrix element type `LT` available as compile-time constants.  This avoids `length(model.linear_terms)` calls at runtime and lets Julia specialise the entire call stack on these values.

### Reduced dynamics as the source of Λ

Rather than storing a separate $\text{NVAR} \times \text{NVAR}$ Jordan matrix, $\Lambda$ is read from the first `NVAR` columns of `R.poly.coefficients`.  Only the diagonal `lambda_diag` (a `Vector{T}` of length `NVAR`) is stored in the context; this is sufficient for computing superharmonics.  The full $\Lambda$ is reconstructed locally in `solve_cohomological_problem` for the one-time precompute calls that need the off-diagonal blocks.

### External dynamics rows pre-filled in R

The last $N_\text{ext}$ rows of `R.poly.coefficients` are filled once from the external system polynomial via `_embed_external_dynamics!` and never touched by the solver.  This eliminates the need for a separate $N_\text{ext} \times L$ external-dynamics matrix in the context.

### Two-pass operator precomputation

The master-mode operator columns `C_coeffs` depend only on `master_modes` and $\Lambda[1:\text{ROM}, 1:\text{ROM}]$ and are independent of the external directions $\Phi_\text{ext}$.  Only the external operator columns `E_coeffs` depend on $\Phi_\text{ext}$, which is not known until after the preliminary external monomial solve.

The precomputation is therefore split into two phases:

1. Compute `C_coeffs` and save the intermediate Horner steps `D_master_steps` (cost: one $O(N \cdot n \cdot \text{ROM})$ pass).
2. After the preliminary solve, compute `E_coeffs` by extending the Horner recurrence with $\Phi_\text{ext}$, reusing `D_master_steps` (cost: one $O(N \cdot n \cdot N_\text{ext})$ pass).

This avoids repeating the master-column work.

### Pre-allocated buffers

All per-monomial allocations are eliminated via buffers stored in `CohomologicalContext`:

| Buffer | Purpose |
|--------|---------|
| `lower_order_buffer` | Accumulate $\xi_j$ vectors (zeroed at each monomial) |
| `candidate_indices_by_monomial` | Pre-filtered candidate index lists for lower-order couplings |
| `system_matrix_buffer` | Stacked $(n + \text{ROM})^2$ system matrix |
| `rhs_buffer` | Stacked RHS and solution vector |
| `external_rhs_buffer` | Scratch for external forcing contributions to the RHS |
