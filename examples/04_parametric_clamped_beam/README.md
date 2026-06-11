# 04 — Parametric clamped beam (two-parameter ROM: axial stretch θ₁ + arch pre-deformation θ₂)

## Model

A 3D continuum clamped-clamped beam with two external parameters:

- **θ₁** — uniform axial stretch: every cross-section scaled by `(1+θ₁)` via constant Jacobian
  contribution `J₁ = e₁⊗e₁`.
- **θ₂** — arch pre-deformation: displacement along the first bending eigenmode `φ₁(x₀)`, with
  per-quadrature-point Jacobian `J₂(x₀) = ∇₀φ₁(x₀)`.

Reference map: `x(θ₁,θ₂,x₀) = x₀ + θ₁ J₁ x₀ + θ₂ φ₁(x₀)`  
Full Jacobian: `J(θ₁,θ₂,x₀) = I + θ₁ J₁ + θ₂ J₂(x₀)`

Since `J₂` varies per quadrature point, the `det J`, `adj J`, and `1/det J` series are bivariate
polynomials in `(θ₁,θ₂)` computed per QP. The reference mesh `beam_h27.msh` is **never re-meshed**;
all parameter dependence enters through the pull-back of the weak form.

The augmented dynamical system has `N_EXT = 2` external states and `NVAR = 4` reduced coordinates
`(z₁, z₂, θ₁, θ₂)`.

## Files

| File | Purpose |
| ---- | ------- |
| `main.jl` | Top-level driver (six steps: mesh → eigenproblem → bivariate assembly → nonlinear maps → multiindex set → DPIM solve) |
| `fem/theta_polynomials.jl` | Truncated scalar power-series algebra for a single parameter θ: `poly_mul`, `poly_contract`, `reciprocal_series` |
| `fem/parametric_geometry.jl` | Closed-form `det_series`, `adj_series` for an affine `J = J₀ + θ J₁` in 3D; `check_adj_det_identity` |
| `fem/bivariate_polynomials.jl` | Truncated bivariate series `p(θ₁,θ₂)` stored as `(N+1)×(N+1)` matrices; `bpoly_mul`, `bpoly_dot`, `bpoly_contract`, `breciprocal_series`, `bpoly_power` |
| `fem/bivariate_geometry.jl` | Per-QP bivariate `det J(θ₁,θ₂,x₀)` and `adj J(θ₁,θ₂,x₀)` from closed-form trace/matrix-product identities |
| `fem/parametric_assembly.jl` | `assemble_K_M_bivariate!`, `ParametricGeometricNonlinearity2D{2\|3}`, bivariate K/M/C correction builders |
| `backbone/compute_backbone.jl` | θ₁-sweep backbone curves: `backbone_curves.png`, `backbone_shift.png`, `omega0_slope_minus2.png` |
| `backbone/compute_backbone_theta2.jl` | θ₂-sweep backbone curves: `backbone_theta2_curves.png`, `backbone_theta2_shift.png`, `omega0_vs_theta2.png` |
| `backbone/backbone_derivation.md` | Mathematical derivation of the backbone from the normal-form ROM |
| `plotting/backbone_plots.jl` | (loaded by validation) backbone overlay figures |
| `validation/run_validation.jl` | Non-parametric validation: deform mesh to exact geometry, run full DPIM, compare |
| `validation/check_eigenvalue_scaling.jl` | Direct FEM eigenvalue check of `dω₀/dθ` |
| `validation/exact_geometry_assembly.jl` | Assembly on the exactly deformed mesh |
| `validation/exact_nonlinear_maps.jl` | Exact (non-parametric) nonlinear maps for validation |
| `validate.jl` | Lightweight smoke-test runner (used by CI) |

## How to run

The example has its own `Project.toml`. Activate it from the repository root:

```bash
julia --project=examples/04_parametric_clamped_beam -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/04_parametric_clamped_beam/main.jl")'
```

**Prerequisite:** `main.jl` loads the mesh from
`examples/04_parametric_clamped_beam/../BenchmarkFerrite/beam_h27.msh`
(i.e. `benchmark/ferrite/beam_h27.msh` relative to the repo root).
Run `benchmark/ferrite/generate_beam_mesh.jl` once if the file does not exist; it is also
restored from the archive into `benchmark/ferrite/` by Phase 2 of the examples-fixes plan.

The `N_θ` constant near the top of `main.jl` (default: 4) controls the truncation order for both
`θ₁`- and `θ₂`-expansions. `max_degree = 5` is the DPIM parametrisation order.

Post-processing (run after `main.jl` has produced `results/data/R.jls`):

```julia
include("examples/04_parametric_clamped_beam/backbone/compute_backbone.jl")
include("examples/04_parametric_clamped_beam/backbone/compute_backbone_theta2.jl")
include("examples/04_parametric_clamped_beam/validation/run_validation.jl")
```

## Expected outputs

```text
results/
  data/
    W.jls                  — parametrisation W(z₁,z₂,θ₁,θ₂) (serialised)
    R.jls                  — reduced dynamics R(z₁,z₂,θ₁,θ₂) (serialised)
    arch_mode.jls          — first bending eigenmode φ₁ (used for θ₂ map)
    R_coefficients.csv     — reduced dynamics coefficients, one row per non-zero monomial
    summary.txt            — model description, eigenfrequencies, timing, Julia version
  figures/
    backbone_curves.png           — Ω vs |z₁| for various (θ₁,θ₂)  [compute_backbone.jl]
    backbone_shift.png            — (Ω − ω₀) vs |z₁|  [compute_backbone.jl]
    omega0_slope_minus2.png       — ω₀(θ₁,0) vs (1+θ₁) log-log, slope -2 reference  [compute_backbone.jl]
    backbone_theta2_curves.png    — Ω vs |z₁| for θ₂ sweep  [compute_backbone_theta2.jl]
    backbone_theta2_shift.png     — (Ω − ω₀) vs |z₁|, θ₂ sweep  [compute_backbone_theta2.jl]
    omega0_vs_theta2.png          — ω₀(0,θ₂) vs θ₂  [compute_backbone_theta2.jl]
    validation_backbone_t1=*.png  — per test-point ROM vs exact overlay  [run_validation.jl]
    validation_omega0.png         — ω₀ comparison bar chart  [run_validation.jl]
```

## Reference results

Curated reference outputs live in `results/reference/` (tracked in git).
Regenerate only in a reviewed commit by copying fresh outputs:

```bash
cp results/data/R_coefficients.csv results/reference/
cp results/data/summary.txt results/reference/
```

## Historical results

A full archived run is restored untracked from the archive repo into:

```text
results/archived/
  data/     — W.jls, R.jls, R_coefficients.csv, summary.txt, validation_metrics.csv
  figures/  — all backbone and validation figures
```

These are not version-controlled here; the durable copy is `MORFE_results_archive`.

## Measured runtime

not yet measured

## Mathematical structure

### Bivariate parametric reference map

The two-parameter reference map is:

```
x(θ₁, θ₂, x₀) = x₀ + θ₁ J₁ x₀ + θ₂ φ₁(x₀)
J(θ₁, θ₂, x₀) = I + θ₁ J₁ + θ₂ J₂(x₀)
```

where `J₁ = e₁⊗e₁` (constant) and `J₂(x₀) = ∇₀φ₁(x₀)` (varies per QP).

### Why `adj(J) / det(J)` — never materialised as `J⁻¹`

In 3D for an affine-in-θ Jacobian:

- `det J(θ₁,θ₂)` is an exact bivariate polynomial of total degree ≤ 3
- `adj J(θ₁,θ₂)` is an exact bivariate polynomial of total degree ≤ 2
- `J⁻¹ = adj(J) / det(J)` is rational

Rather than assembling a series for `J⁻¹`, we substitute `adj(J)/det(J)` and let one
`1/det(J)` factor cancel the `det(J)` from `dV = det(J) dV₀`. Each form retains only a
scalar reciprocal factor:

| Form        | `N_input` | residual reciprocal |
| ----------- | --------- | ------------------- |
| Linear K    | 1         | `(1/det)¹`          |
| Linear M    | 0         | none (has `det¹`)   |
| Quadratic G | 2         | `(1/det)²`          |
| Cubic H     | 3         | `(1/det)³`          |

The scalar bivariate series `1/det J(θ₁,θ₂)` is computed via `breciprocal_series` in
`fem/bivariate_polynomials.jl`.

### Truncation order

`N_θ` (default 4) truncates both `θ₁`- and `θ₂`-power series. The bivariate coefficient
matrices `K_{k₁,k₂}`, `M_{k₁,k₂}` are assembled for all `k₁+k₂ ≤ N_θ`.
The DPIM parametrisation order is `max_degree = 5`.

### DPIM external variables

Both `θ₁` and `θ₂` enter the augmented system as frozen real external states:

```
θ̇₁ = 0, θ̇₂ = 0   ⇒   λ_θ₁ = λ_θ₂ = 0,   N_EXT = 2
```

The multiindex set is built on `(z₁, z₂, θ₁, θ₂)` with `max_degree = 5`;
`conjugate_permutation = [2, 1, 3, 4]` swaps `z₁ ↔ z₂` and leaves `θ₁, θ₂` self-conjugate.

### Analytical reference

For a clamped-clamped Euler-Bernoulli beam under uniform axial stretch:

```
ω_b(θ₁) / ω_b(0) ≈ (1 + θ₁)⁻²   (bending — lowest modes)
```

This slope (-2) is verified by `omega0_slope_minus2.png` from `compute_backbone.jl`.

<!-- claims:
  θ₁ axial stretch + J₁=e₁⊗e₁ → main.jl:4-7, main.jl:117
  θ₂ arch pre-deformation + J₂(x₀) → main.jl:4-7
  reference map formula → main.jl:8
  N_EXT=2 → main.jl:130
  NVAR=4 → main.jl:131
  max_degree=5 → main.jl:132
  N_θ default=4 → main.jl:119
  conjugate_permutation=[2,1,3,4] → main.jl:323
  bivariate adj/det computation → fem/bivariate_geometry.jl:1-22
  breciprocal_series → fem/bivariate_polynomials.jl (header)
  mesh path → main.jl:73
  output paths → main.jl:383-422
  backbone figures → backbone/compute_backbone.jl:14-16, backbone/compute_backbone_theta2.jl:12-15
  validation figures → validation/run_validation.jl:13-15, plotting/backbone_plots.jl:13-14
-->
