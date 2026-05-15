# Plan: `InvarianceError` Module

## 1. Purpose

Provide a black-box validation tool for the output of `solve_cohomological_problem`.
Given a `NDOrderModel` (or `FirstOrderModel`) and a computed pair `(W, R)`,
measure how well the invariance equation is satisfied over a point cloud in reduced
coordinates.

---

## 2. Mathematical Background

### Native-order full-order model

```
B_ORD x^{(ORD)} + B_{ORD-1} x^{(ORD-1)} + … + B_1 ẋ + B_0 x = F(x, ẋ, …, x^{(ORD-1)})
```

`linear_terms = (B_0, B_1, …, B_ORD)` — Julia tuple, 1-indexed.  
`nonlinear_terms` — tuple of `AbstractMultilinearMap{ORD}`.

### First-order companion

`linear_first_order_matrices(model)` returns `(A_FO, B_FO)` such that

```
B_FO Ẏ = A_FO Y + F_FO(Y)
```

with state `Y = [x; ẋ; …; x^{(ORD-1)}] ∈ R^{ORD·FOM}`.

Block structure (ORD = 2 example):

```
B_FO = [I      0   ]    A_FO = [0    I     ]
       [0    B_ORD ]            [-B_0  -B_1 ]
```

`F_FO(Y)` is nonzero only in the last FOM rows — it equals `F(x, ẋ, …, x^{(ORD-1)})`.

### Invariance equation

Given `y = W(z)` (first-order parametrisation) and `ż = R(z)`:

```
E_FO(z) = B_FO · J_{W_FO}(z) · R(z) − A_FO · W_FO(z) − F_FO(W_FO(z))
```

The first `(ORD-1)·FOM` rows are **always zero** by construction (they encode
`d/dt[x^{(k)}] = x^{(k+1)}`, which is satisfied by the chain rule on the
stored derivative blocks).  Only the **last FOM rows** carry physical content:

```
E(z) = B_ORD · x^{(ORD)}(z)
       + B_{ORD-1}·x^{(ORD-1)}(z) + … + B_0·x(z)
       − F(x, ẋ, …, x^{(ORD-1)})          ∈ R^{FOM}
```

This is the **native-order residual**.  For ORD = 1 all rows are the last rows;
the formula degenerates correctly.

### Where x^{(ORD)} comes from

`Parametrisation.poly.coefficients` has shape `(FOM, ORD, L)`:

| Column index (Julia) | Meaning |
|----------------------|---------|
| `k = 1` | x = x^{(0)} |
| `k = 2` | ẋ = x^{(1)} |
| … | … |
| `k = ORD` | x^{(ORD-1)} |

`x^{(ORD)}` is **not stored**.  It is computed as the Jacobian-vector product
of the *last stored block* `W[:, ORD, :]` with `R(z)`:

```
x^{(ORD)}(z) = J_{W[:,ORD,:]}(z) · R(z)
```

---

## 3. Key Quantities and Their Sources

| Quantity | Source |
|----------|--------|
| `x^{(k)}(z)` for k = 0…ORD-1 | `evaluate(DensePolynomial from W.poly, z)` — 3-D evaluate returns `(FOM, ORD)` array |
| `R(z)` | `evaluate(R.poly, z)` — 2-D evaluate returns `NVAR`-vector |
| `x^{(ORD)}(z)` | JVP of `W.poly.coefficients[:, ORD, :]` with `R(z)` |
| Linear residual | `mul!` with `model.linear_terms` tuple |
| Nonlinear residual | `evaluate_nonlinear_terms!(buf, model, deg, state_vectors)` for each degree |

---

## 4. Jacobian-Vector Product (JVP) — Analytic Implementation

No new dependencies needed.  The multiindex set already provides all
exponent information.

### Formula

For a polynomial `P(z) = ∑_l C_l · z^{α_l}` with `C_l ∈ R^K` and a tangent `v ∈ C^NVAR`:

```
(J_P(z) · v)  =  ∑_l C_l · ∑_{j : α_{l,j} > 0}  α_{l,j} · v_j · ∂_j m_l(z)
```

where the **safe** derivative monomial (no division by z_j) is:

```
∂_j m_l(z)  =  α_{l,j} · z_j^{α_{l,j}-1} · ∏_{k≠j} z_k^{α_{l,k}}
             =  α_{l,j} · pw[j][α_{l,j}-1] · (m_l / pw[j][α_{l,j}])  when pw[j][α_{l,j}] ≠ 0
             =  α_{l,j} · pw[j][α_{l,j}-1] · ∏_{k≠j} pw[k][α_{l,k}]  always (safe form)
```

### Algorithm: `_jvp_dense_poly!(result, coeffs, mset, z, v)`

```
Input:
  coeffs : Matrix{T}  (K × L)   — reshaped coefficient slice
  mset   : MultiindexSet{NVAR}
  z      : AbstractVector{T}    — NVAR evaluation point
  v      : AbstractVector{T}    — NVAR tangent (= R(z))

Output:
  result : Vector{T}  (K,)      — in-place accumulation

Steps:
  1. Precompute powers:
       max_exp[j] = maximum over l of mset[l][j]
       pw[j][e] = z[j]^e  for e = 0 … max_exp[j]   (length max_exp[j]+1 vector)
  2. Loop over monomials l = 1 … L:
       m_l = ∏_j pw[j][mset[l][j]]    (= z^{α_l})
       Loop over variables j = 1 … NVAR  where mset[l][j] > 0:
           α_j = mset[l][j]
           dm = α_j · pw[j][α_j - 1] · (m_l / pw[j][α_j])  if pw[j][α_j] ≠ 0
              = α_j · pw[j][α_j - 1] · ∏_{k≠j} pw[k][mset[l][k]]  otherwise (fall-through)
           result .+= v[j] · dm · view(coeffs, :, l)
```

**Complexity**: O(L · NVAR · K) per point — acceptable for a validation tool
(L ≈ few hundred for degree-5 polynomial, NVAR ≈ 4–10, K = FOM).

**Note**: `_jvp_dense_poly!` operates on a `(K, L)` matrix (one derivative block),
not the full 3-D array, so it can be called independently per block.

---

## 5. Point-Cloud Sampling

### Coordinates

`z ∈ C^NVAR` where `NVAR = ROM + external_system_size`.

- **Master-mode coordinates** `z[1:ROM]`: draw from complex Gaussian,
  i.e. `Re(z_i), Im(z_i) ~ N(0, amplitude²/2)` independently.
  This is isotropic in the complex plane and respects conjugate symmetry
  (the sample set should include conjugate pairs if the model is real).
- **External-system coordinates** `z[ROM+1:end]`: default to zero
  (unforced validation); can be supplied by the user.

### Conjugate pairing

When the parametrisation is real (computed under conjugate symmetry), each
sample `z_k` and its conjugate `conj(z_k)` should both be included so that
the physical state `y = W(z_k) + W(conj(z_k))` is real.  The error should be
computed on individual complex points (the polynomial is evaluated at each z
separately); the norm is taken in C^FOM.

### Alternative: spherical shell

Optionally sample uniformly on a sphere `‖z[1:ROM]‖ = amplitude` in C^ROM ≃ R^{2·ROM}
to probe the manifold at a fixed amplitude level.

---

## 6. Algorithm: `invariance_error_at(model, W, R, z)`

Single-point evaluation.  Called per-sample inside the loop.

```
Input:  model :: NDOrderModel{ORD, ORDP1, …}
        W     :: Parametrisation{ORD, NVAR, T}
        R     :: ReducedDynamics{ROM, NVAR, T}
        z     :: AbstractVector{T}   (length NVAR)

Output: E :: Vector{T}  (length FOM)

Steps:
  1. Evaluate x^{(k)} for k = 0 … ORD-1:
       X = evaluate(W.poly, z)        # returns (FOM, ORD) matrix
       x_k = view(X, :, k+1)         # 1-indexed: col k+1 = k-th derivative

  2. Evaluate reduced dynamics:
       rz = evaluate(R.poly, z)       # NVAR-vector

  3. Compute x^{(ORD)} via JVP:
       last_coeffs = reshape(view(W.poly.coefficients, :, ORD, :), FOM, L)
       E = zeros(T, FOM)
       _jvp_dense_poly!(E, last_coeffs, W.poly.multiindex_set, z, rz)
       # E now holds x^{(ORD)}

  4. Multiply by B_ORD:
       mul!(buf_fom, model.linear_terms[end], E)   # B_ORD · x^{(ORD)}
       copyto!(E, buf_fom)

  5. Add linear terms B_k · x^{(k)} for k = 0 … ORD-1:
       for k in 0 : ORD-1
           mul!(E, model.linear_terms[k+1], view(X, :, k+1), one(T), one(T))
       end
       # E = ∑_k B_k · x^{(k)}   (linear LHS)

  6. Subtract nonlinear RHS F:
       state_vectors = ntuple(k -> view(X, :, k), ORD)   # (x^{(0)}, …, x^{(ORD-1)})
       max_deg = maximum(t.deg for t in model.nonlinear_terms; init=0)
       for deg in 2 : max_deg
           evaluate_nonlinear_terms!(E, model, deg, state_vectors, nothing)
           # accumulates with − sign → redefine: subtract separately
       end
       # Carefully: evaluate_nonlinear_terms! adds to E, but we want E − F.
       # Solution: compute F in a separate buffer `buf_nl`, then E .-= buf_nl

  Return E
```

### FirstOrderModel dispatch

`linear_first_order_matrices(m::FirstOrderModel)` returns `(A, B) = (-B0, B1)`.

The invariance error reduces to:
```
E = B1 · J_W(z) · R(z) + B0 · W(z) − F(W(z))
```

i.e., steps 1–6 with ORD = 1 and only one x-block.

---

## 7. State Error Estimate from the Invariance Error

### From residual force to state error

The invariance error `E(z)` is a residual in the FOM equation of motion —
it has units of force, not of displacement.  To estimate how far the ROM
trajectory `x_ROM(z) = W(z)` deviates from the true FOM solution `x_FOM`,
linearise the FOM residual around the ROM point (Newton–Raphson first step):

```
G(x) = B_ORD x^{(ORD)} + … + B_0 x − F(x, ẋ, …)

G(x_ROM) = E(z)             (by definition of the invariance error)
G(x_FOM) = 0               (FOM is satisfied exactly)

⟹   J_G(x_ROM) · (x_FOM − x_ROM) ≈ −E(z)
⟹   δx  :=  x_FOM − x_ROM  ≈  −J_G(x_ROM)⁻¹ E(z)
```

where `J_G` is the tangent of `G` with respect to `x`.

### Why a per-sample factorisation is infeasible

`J_G(x_ROM(z))` depends on the nonlinear part and on frequency:

```
J_G(x, s) = ∑_k sᵏ Bₖ  −  ∂F/∂x|_x  −  s ∂F/∂ẋ|_x  −  …
```

For each sample `z_k` both `x_ROM(z_k)` and the dominant frequency `s(z_k)`
change.  For a FEM model with `FOM = O(10⁴)` DOFs, refactoring a sparse N×N
matrix per sample repeats the cost of a full FEM Newton step thousands of
times — completely infeasible as a validation utility.

### Why B_0 alone is inadequate for resonance and multiphysics

**DPIM targets resonance by construction.**  The dynamic stiffness at the
master eigenfrequency `s₁ = λ₁` is:

```
L(s₁)  =  ∑_k s₁ᵏ Bₖ  =  B_0 + s₁ B_1 + s₁² B_2 + …
```

For a lightly-damped second-order system near resonance (`s₁ = iω₀`):

```
L(iω₀)  =  (K − ω₀² M) + iω₀ C  ≈  iω₀ C     (near resonance)
```

The static stiffness `B_0 = K` satisfies `‖K‖ ≫ ‖L(iω₀)‖` by the
quality factor `Q = ω₀ M / C ≈ 1/(2ζ)`, typically 10²–10³ for structural
systems.  Using `B_0⁻¹` therefore **underestimates the state error by the
Q-factor** — a systematic error of one to three orders of magnitude.

Problem-class breakdown:

| Problem class | What B_0 misses | Severity |
| ------------- | --------------- | -------- |
| Structural resonance (geometric NL) | Inertia + damping in `L(s₁)` | Factor Q — always significant |
| Fluid-structure interaction | `s ∂F/∂ẋ` (velocity-dependent fluid forces) | Problem-dependent; large for aeroelastic problems |
| Piezoelectric coupling | Coupling is already in Bₖ if linear; nonlinear `∂F/∂x` if not | Minor if coupling is linear |
| Thermoelastic / multiphysics | Temperature-dependent `B_0(T)` evaluated at wrong temperature | Grows with deformation amplitude |

### Local superharmonic from the reduced dynamics

At a sample point `z`, the reduced velocity is `ż = R(z)` — already evaluated
in the invariance error loop at zero extra cost.  The **local superharmonic**
is the Rayleigh quotient of the reduced-dynamics Jacobian `J_R(z)` at `z`:

```
s_eff(z)  =  ⟨z, R(z)⟩ / ⟨z, z⟩        (one dot product, no Jacobian needed)
```

The denominator is `⟨z, z⟩ = ∑ᵢ |zᵢ|²` — the squared norm, no square root.
In Julia: `dot(z, R_z) / dot(z, z)` (both real denominator, complex numerator).

Derivation: since `ż = R(z)` and `J_R(z) z ≈ R(z)` at leading order,
`⟨z, J_R(z) z⟩ / ⟨z, z⟩ = s_eff(z)` is the Rayleigh quotient of `J_R(z)`.
As `‖z‖ → 0`, `J_R(0)` is diagonal with entries `λ_i` (the master
eigenvalues), so `s_eff(z) → ∑ᵢ λᵢ|zᵢ|² / ∑ᵢ|zᵢ|²` — a weighted average of
eigenvalues, approaching `λ₁` for a point dominated by the first mode.  At
larger amplitudes `s_eff(z)` accumulates nonlinear corrections automatically,
with no external eigenvalue input required.

This scalar is available from the already-computed `R(z_k)` at essentially
zero additional cost: one `dot` call per sample.

### Recommended approximation: dynamic stiffness at the median superharmonic

Per-sample factorisation of `L(s_k) = ∑_k s_k^{j} B_j` would repeat the cost
of a full FEM Newton step for every sample — still infeasible.  Instead:

1. For each `z_k` compute `s_k = ⟨z_k, R(z_k)⟩ / ‖z_k‖²` (free, `R(z_k)`
   is already available).
2. Take a representative scalar `s̄ = median(Re(s_k)) + i·median(Im(s_k))`.
3. Factor `L(s̄)` **once** and reuse across all samples.

`s̄` is determined entirely from the sample cloud — no eigenvalue argument
needs to be passed by the caller.  At small amplitude `s̄ ≈ λ₁` (fundamental
frequency); at larger amplitudes it reflects amplitude-stiffening or softening.

This estimate remains a **linearisation at the origin** (nonlinear correction
`∂F/∂x|_{x_ROM}` is still dropped).  For polynomial nonlinearities vanishing
at zero, `∂F/∂x|_{x=0} = 0`, so this is exact at linear order and accurate
for small amplitudes.  For FSI problems where `F` depends on `ẋ`, the
velocity-dependent Jacobian `s ∂F/∂ẋ` is also dropped; the estimate is then
an underestimate by the ratio of fluid-force stiffness to dynamic stiffness.

### Implementation: one factorisation, cheap solves

```julia
# Pass 1: evaluate R(z_k) for all samples (already done for E_k)
s_vals = [dot(z_k, R_k) / dot(z_k, z_k) for (z_k, R_k) in samples]  # ⟨z,R⟩/⟨z,z⟩
s_bar  = complex(median(real(s_vals)), median(imag(s_vals)))

# Assemble and factor dynamic stiffness at s̄
L_bar  = sum(s_bar^(k-1) .* model.linear_terms[k]
             for k in eachindex(model.linear_terms))
lu_Lbar = lu(L_bar)          # once, O(N^p) for sparse, p ≈ 1–1.5

# Pass 2: triangular solve per sample
for each (z_k, E_k):
    δx_k       = lu_Lbar \ E_k    # O(N) triangular solve
    state_err[k] = norm(real(δx_k))
```

`lu_Lbar` reuses the same KLU/Pardiso infrastructure already present in the
codebase.  The additional cost per sample is a single sparse triangular solve
— negligible above the invariance error computation itself.

### Expected convergence

Since `‖E(z)‖ = O(‖z‖^{max_order+1})` and `L(s̄)⁻¹` is a fixed linear map:

```
‖δx(z)‖ = O(‖z‖^{max_order+1})
```

Both curves have the same log-log slope.  The vertical gap encodes the dynamic
flexibility `1/σ_min(L(s̄))`; for a lightly-damped structure near resonance
this is `~Q` times larger than the static flexibility `1/σ_min(B_0)`, making
the state error visible at amplitudes `~Q` times smaller than the force
residual alone would suggest.

---

## 8. Convergence Analysis vs Amplitude

### Theory

The parametrisation `W` and reduced dynamics `R` are polynomials truncated at
degree `max_order = sum(mset.exponents[end])` (the Grlex-last multiindex gives the
maximum total degree, directly readable from `MultiindexSet`).

The DPIM cohomological equations are solved exactly at every monomial order up
to `max_order`.  The invariance error therefore only receives contributions from
omitted monomials of degree `max_order + 1` and higher:

```
‖E(z)‖₂  =  O(‖z‖^{max_order + 1})   as ‖z‖ → 0
```

In log-log space this is a line of **slope = max_order + 1**:

```
log ‖E(z)‖  ≈  (max_order + 1) · log ‖z‖  +  C
```

Verifying this slope numerically gives strong evidence that:

- the parametrisation is correctly computed (wrong coefficients would break
  the slope at lower amplitude),
- the truncation order has been identified correctly,
- the implementation of the error formula itself is correct (slope < max_order+1
  signals a bug, slope > max_order+1 would signal accidental cancellation or
  over-accurate coefficients, which is benign but surprising).

### Sampling strategy for the convergence plot

Draw `n_samples` points with master-mode coordinates from a standard normal
distribution: each component of `z[1:ROM]` is drawn i.i.d. from `N(0, 1)`
(real and imaginary parts independently).  External coordinates are set to zero.
The distance `‖z_k‖` then varies naturally across samples, populating the
amplitude axis without any grid construction.

### Algorithm: `invariance_error_convergence(model, W, R; kwargs...)`

Draw `n_samples` complex vectors `z_k ∈ C^ROM` with i.i.d. `N(0,1)` entries,
pad with zeros for the external coordinates, and for each point record the pair
`(‖z_k‖, ‖E(z_k)‖₂)`.  The truncation order is read directly from the
multiindex set as `max_order = sum(last(mset.exponents))` (the Grlex-last
entry carries the maximum total degree).  Return both vectors for plotting.

### Plot specification

Function: `plot_invariance_convergence(result; kwargs...)`

Generates a **log-log scatter** with `Plots.jl` (already used in demos and
notebooks; not a new dependency at the package level — call it only from
demo/validation scripts).

```julia
using Plots

function plot_invariance_convergence(radii, force_errors, state_errors, max_order;
                                     title = "Invariance error convergence")
    p = scatter(radii, force_errors;
                xscale = :log10, yscale = :log10,
                label  = "‖E(z)‖₂  (force residual)",
                xlabel = "‖z‖",
                ylabel = "error norm",
                title  = title,
                markersize = 2, markerstrokewidth = 0, alpha = 0.5)

    scatter!(p, radii, state_errors;
             label     = "‖B₀⁻¹E(z)‖₂  (state error est.)",
             markersize = 2, markerstrokewidth = 0, alpha = 0.5)

    # Single reference line anchored at the median of the force errors
    idx     = sortperm(radii)
    i_ref   = length(radii) ÷ 2
    r_ref   = radii[idx[i_ref]]
    e_ref   = force_errors[idx[i_ref]]
    C_ref   = e_ref / r_ref^(max_order + 1)
    r_range = exp.(LinRange(log(minimum(radii)), log(maximum(radii)), 100))
    plot!(p, r_range, C_ref .* r_range .^ (max_order + 1);
          label = "O(‖z‖^{$(max_order+1)})",
          lw = 2, ls = :dash, color = :black)

    return p
end
```

**Visual diagnosis guide:**

| Observed behaviour | Likely cause |
| ------------------ | ------------ |
| Slope ≈ max_order+1 over full range | Correct |
| Slope < max_order+1 at small ‖z‖ | Bug in error computation or coefficient storage |
| Plateau or upturn at large ‖z‖ | Expected — polynomial approximation breaks down |
| Slope much larger than max_order+1 | Accidental near-cancellation; benign |

---

## 9. Top-Level API

```julia
"""
    invariance_error_norms(model, W, R;
                           n_samples = 1000,
                           amplitude = 1.0,
                           rng       = Random.default_rng())
    → NamedTuple{(:max, :mean, :rms, :pointwise)}

Compute the invariance-equation residual ‖E(z)‖₂ over a Gaussian point cloud
in reduced coordinates.

`amplitude` sets the standard deviation of the complex-Gaussian sampling in
each master-mode direction.  External coordinates are fixed to zero.
"""
function invariance_error_norms(model, W::Parametrisation, R::ReducedDynamics; kwargs...)

"""
    invariance_error_convergence(model, W, R;
                                 n_samples = 1000,
                                 rng       = Random.default_rng())
    → NamedTuple{(:radii, :force_errors, :state_errors, :s_bar, :max_order)}

Draw n_samples points from a standard normal distribution in reduced coordinates.
For each point record ‖z‖, the force residual ‖E(z)‖₂, and the dynamic state
error estimate ‖L(s̄)⁻¹E(z)‖₂.

The representative superharmonic s̄ is computed automatically as the median of
s_eff(z_k) = ⟨z_k, R(z_k)⟩/‖z_k‖² over all samples — no eigenvalue needs to be
passed by the caller.  L(s̄) = ∑_k s̄ᵏ Bₖ is factored once and reused.
Both error series have expected log-log slope max_order+1.
"""
function invariance_error_convergence(model, W::Parametrisation, R::ReducedDynamics; kwargs...)
```

Return type of `invariance_error_norms`:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `max` | `Float64` | `max_k ‖E(z_k)‖₂` |
| `mean` | `Float64` | `mean_k ‖E(z_k)‖₂` |
| `rms` | `Float64` | `sqrt(mean_k ‖E(z_k)‖₂²)` |
| `pointwise` | `Vector{Float64}` | all individual norms (length `n_samples`) |

Return type of `invariance_error_convergence`:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `radii` | `Vector{Float64}` | `‖z_k‖` for each sample (length `n_samples`) |
| `force_errors` | `Vector{Float64}` | `‖E(z_k)‖₂` — force residual |
| `state_errors` | `Vector{Float64}` | `‖L(s̄)⁻¹E(z_k)‖₂` — state error estimate |
| `s_bar` | `ComplexF64` | representative superharmonic used for `L(s̄)` |
| `max_order` | `Int` | truncation order (expected log-log slope - 1) |

---

## 8. Module Structure

```
src/Validation/InvarianceError.jl
────────────────────────────────
module InvarianceError

using LinearAlgebra: mul!
using Random
using Statistics: mean

using ..Polynomials: DensePolynomial, evaluate
using ..Multiindices: MultiindexSet
using ..ParametrisationMethod: Parametrisation, ReducedDynamics, coefficients, multiindex_set
using ..FullOrderModel: NDOrderModel, FirstOrderModel, evaluate_nonlinear_terms!

export invariance_error_norms

# Internal helpers (not exported):
#   _jvp_dense_poly!(result, coeffs, mset, z, v)
#   _invariance_error_at!(E, buf_nl, buf_fom, X, model, W, R, z)
#   _sample_reduced_coords(NVAR, ROM, n_samples, amplitude, rng)

end # module
```

### Integration into `src/MORFE.jl`

Add after the `ParametrisationMethod` include group (it depends on both FOM and PM modules):

```julia
include("Validation/InvarianceError.jl")
using .InvarianceError
```

---

## 9. Buffer Pre-Allocation

To avoid allocations inside the sample loop, `invariance_error_norms` should
pre-allocate:

| Buffer | Size | Purpose |
|--------|------|---------|
| `z_buf` | `NVAR` | current sample point |
| `X_buf` | `FOM × ORD` | state derivative blocks |
| `rz_buf` | `NVAR` | R(z) |
| `E_buf` | `FOM` | running invariance error |
| `nl_buf` | `FOM` | nonlinear term accumulator |
| `fom_buf` | `FOM` | scratch for mul! |
| `pw_buf` | ragged, max_exp+1 per variable | powers for JVP |

All allocated once before the loop and reused.

---

## 10. Open Questions / Design Decisions

1. **`evaluate(W.poly, z)` returns `(FOM, ORD)`?**  
   Confirm that the existing `DensePolynomial` 3-D evaluator (via `gemv` on
   reshaped array) returns a `(FOM, ORD)` array when called with the
   `(FOM, ORD, L)` coefficient tensor.  If not, adjust the reshape step.

2. **External system handling.**  
   When `model.external_system !== nothing`, `evaluate_nonlinear_terms!` needs
   `r` (the external state vector).  For the default unforced case pass
   `zeros(T, N_EXT)`.  A keyword argument `r_external` can allow the user to
   fix a non-trivial external state.

3. **`max_deg` determination.**  
   Use `maximum(term.deg for term in model.nonlinear_terms; init=0)`.
   If `N_NL = 0` skip the loop entirely.

4. **Sign convention in `evaluate_nonlinear_terms!`.**  
   The function *adds* to `res` (it does not zero it first).  Use a separate
   `nl_buf` zeroed at each sample point, then `E .-= nl_buf`.

5. **Normalisation denominator.**  
   The denominator `‖B_FO · Ẏ‖` requires the full `ORD·FOM` first-order JVP.
   This is more expensive (ORD JVP calls).  Consider making `normalise` a
   separate convenience function `invariance_error_normalised` to keep the
   hot path lean.

6. **Complex vs real arithmetic.**  
   `z` and the polynomial coefficients are `ComplexF64` in general.
   `‖E(z)‖₂ = sqrt(real(E' * E))`.  The returned norms are always real.

7. **`FirstOrderModel` specialisation.**  
   `FirstOrderModel` stores `B0`, `B1` directly and uses `MultilinearMap{1}`.
   Provide a dedicated method to avoid unnecessary 3-D reshaping overhead.
