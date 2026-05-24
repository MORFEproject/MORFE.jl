# Parametric beam demo — uniform axial stretch

Companion to `demo/BenchmarkFerrite/benchmark_ferrite.jl`.  A scalar
parameter `θ ∈ ℝ` controls a *uniform axial stretch* of the beam:

* every cross-section is left untouched
* every length-wise slice `dx₁` is scaled by `(1 + θ)`
* the beam length goes `L → (1 + θ) · L`

The reference mesh `beam_h27.msh` (the same file used by the original
benchmark) is **never re-meshed**.  All `θ`-dependence enters through
the pull-back of the weak form to the reference configuration; the
reduced model is a *parametric* ROM in `(z₁, z₂, θ)`.

## Files

| File | Purpose |
| --- | --- |
| `parametric_beam_demo.jl`   | Main script: load mesh, build parametric geometry, assemble, solve. |
| `parametric_assembly.jl`    | `assemble_K_M_polynomial!` and `ParametricGeometricNonlinearity{2|3}` with `evaluate_kth_quadratic!` / `evaluate_kth_cubic!`. |
| `parametric_geometry.jl`    | Closed-form `det_series`, `adj_series`, `inv_series` for an affine `J = J₀ + θ J₁` in 3D. |
| `theta_polynomials.jl`      | Generic truncated power-series algebra: `poly_mul`, `poly_contract`, `reciprocal_series`. |

The demo depends only on the same packages as `benchmark_ferrite.jl`
plus `StaticArrays` (already a transitive dependency).

## How to run

From the repository root:

```julia
using Pkg
Pkg.activate("demo/BenchmarkFerrite")           # reuse the benchmark env
include("demo/BenchmarkFerrite/generate_beam_mesh.jl")  # once, to produce beam_h27.msh
include("demo/ParametricBeam/parametric_beam_demo.jl")
```

The `N_θ` constant near the top of the main script controls the
truncation order of every `θ`-expansion.  It is the only knob needed
to push the parametric ROM to higher order.

## Mathematical structure

### Parametric reference map

```
x(θ, x₀) = x₀ + θ φ(x₀)                  ⇒   J(θ) = J₀ + θ J₁
```

For uniform stretch, `φ(x₀) = (x₀₁, 0, 0)`, so `J₁ = e₁ ⊗ e₁` and `J₀ = I`.

### Why `adj(J) / det(J)`

The pulled-back weak form needs `J⁻¹(θ)`.  In 3D:

* `det J(θ)` is **exact** polynomial of degree ≤ 3
* `adj J(θ)` is **exact** polynomial of degree ≤ 2
* `J⁻¹(θ) = adj J(θ) / det J(θ)` is *rational*

Only the scalar series `1 / det J(θ)` requires a Taylor expansion.
We obtain its coefficients from the **reciprocal-series recurrence**:

Given `p(θ) = Σ_k p_k θ^k` with `p₀ ≠ 0`, the series `q(θ) = 1 / p(θ)`
satisfies, by matching coefficients of `θ^k` in `p · q ≡ 1`:

```
q₀  = 1 / p₀
q_k = − (1 / p₀) · Σ_{j=1}^{k} p_j q_{k-j}      for k ≥ 1
```

implemented in `reciprocal_series(p, N)` in `theta_polynomials.jl`.

### Truncation order

The truncation order `N_θ` is an arbitrary positive integer.  The cost
of building `invJ_coeffs` is `O(N_θ²)` for the reciprocal step plus one
truncated convolution; for an affine `J` this is negligible compared
to a single FE assembly pass.

The K-assembly produces `K(θ) = Σ_k θ^k K_k` truncated at any user-chosen
`N_K_used` (default = `N_θ`).  The M-assembly produces only
`length(detJ_coeffs)` non-zero coefficients (since the mass integrand
has no `J⁻¹` dependence).

### Sanity check baked into the script

```
inv_resid = check_inv_series(J₀, J₁, invJ_coeffs, N_θ)
@assert all(<(1e-12), inv_resid)
```

verifies `J(θ) · J⁻¹(θ) ≡ I (mod θ^{N_θ+1})` to machine precision.

### Quadratic and cubic FEM terms

Exactly the same St-Venant–Kirchhoff multilinear forms as the original
benchmark, with all gradients replaced by their parametric counterparts:

```
∇_θ u = ∇₀ u · J⁻¹(θ) ,    ε_θ u = sym(∇_θ u) ,    σ(ε) = λ tr(ε) I + 2μ ε
```

and the integration measure replaced by `det J(θ) · dV₀`.  Both forms
truncate naturally at `θ^{N_θ}`.

### θ as a DPIM external variable

`θ` enters the augmented dynamical system as one *real* external state
with trivial dynamics

```
θ̇ = 0   ⇒   external eigenvalue λ_θ = 0,     N_EXT = 1.
```

The multiindex set is built on `(z₁, z₂, θ)` with `max_degree = 5`;
`conjugate_permutation = [2, 1, 3]` swaps `z₁ ↔ z₂` and leaves `θ`
self-conjugate (as a real parameter).

## Analytical reference

For a clamped-clamped beam in Euler–Bernoulli theory:

```
ω_b(θ) / ω_b(0) ≈ (1 + θ)⁻²        (bending — lowest modes)
ω_a(θ) / ω_a(0) ≈ (1 + θ)⁻¹        (axial)
```

so the slope of the diagonal `z_i` entry of the reduced dynamics w.r.t.
`θ` should match `−2 ω_i(0)` for bending masters and `−ω_i(0)` for
axial masters.  The script prints both for visual comparison.

## Caveats / known limitations

* **MORFE arity contract.**  Each `MultilinearMap` closure must take
  *exactly* `1 + a_pos + a_vel + a_ext` positional arguments.  We
  metaprogram one closure factory per external multiplicity
  `k = 0 … _MAX_EXT_ARITY` (currently 8) at the bottom of
  `parametric_assembly.jl`.  Raise `_MAX_EXT_ARITY` if you need higher
  truncation orders.

* **Sign convention.**  Internal-force terms are added with a *minus*
  sign because MORFE writes every `MultilinearMap` on the right-hand
  side of `M ẍ + C ẋ + K x = Σ multilinear`.  This matches the
  benchmark's `term_cubic = (…) -> res .+= -β · x₁·x₂·x₃` pattern.

* **Parametric mass is approximated.**  `MultilinearMap`'s modal arity
  `(a_pos, a_vel)` has no acceleration slot, so a term `θ^k M_k ẍ`
  cannot be expressed directly.  We *omit* the inertial part of the
  parametric mass and only capture its damping contribution through
  Rayleigh `C(θ) = α M(θ) + β K(θ)`.  For our problem β = 0 and
  det J(θ) is degree 1, so only `C₁ = α M₁` is non-zero — the
  approximation is `O(θ)` accurate at small stretch.  For large `θ`
  or applications sensitive to inertial detuning, the missing
  `θ^k M_k ẍ` terms must be folded back into the rest of the model
  by hand (e.g. by left-multiplying the entire ODE by `M(θ)⁻¹` and
  re-deriving the `K(θ)`, `C(θ)`, internal-force expansions).

* `multilinear_maps` does not cache the QP series across θ-powers for
  the same input tuple — work scales as `N_θ²` rather than `N_θ` if
  every map is called for the same `(u₁, …)`.  Acceptable for moderate
  `N_θ`; profile-driven caching is left to follow-up.

* `parametric_geometry.jl` is written for a *constant* `(J₀, J₁)` on
  the whole domain.  For a spatially varying parametrisation (e.g. a
  curved-beam map), the same routines are called per quadrature point
  inside the assembly instead of once at the top of the demo.

* The cubic-term prefactor `1/3` reflects the symmetric trilinear-form
  convention; double-check against the convention used by your local
  `ferrite_assembly.jl` before comparing ROM coefficients term-by-term.

## Extending to other parametrisations

To change the parametrisation (e.g. bending into an arch, twisting,
non-uniform stretch) you only edit the geometry block in
`parametric_beam_demo.jl`:

```julia
const J₀ = ...                         # ∇x at θ = 0  (usually I)
const J₁ = ...                         # ∇φ at θ = 0
```

Everything downstream (`det_series`, `adj_series`, `inv_series`,
the assembly, the ROM) is parametrisation-agnostic.