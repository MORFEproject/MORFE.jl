# Example 07 — Parametric Clamped-Clamped Arch Beam

Analysis of a 3D sinusoidal arch beam whose geometry is controlled by a single
scalar parameter θ. A single DPIM run produces a ROM that is valid for a
continuous family of arch heights, rather than just one fixed configuration.

---

## Physical model

### Geometry

A clamped-clamped beam with rectangular cross-section:

| Quantity | Value |
|----------|-------|
| Span L | 1 000 mm |
| Height (y) | 10 mm |
| Width (z) | 24 mm |
| Base arch rise h₀ | h₀_L_ratio · L (default 0.005 · 1000 = 5 mm) |

The undeformed centreline follows a sinusoidal arch:

```
y(x) = h₀ sin(π x / L),   x ∈ [0, L]
```

All other nodes are shifted by the same sinusoidal offset applied to their
x₁-coordinate, so the mesh is a smooth body-fitted mapping of a straight-beam
mesh.

### Material

St. Venant-Kirchhoff (SVK) hyperelastic solid, isotropic:

| Parameter | Value |
|-----------|-------|
| Young modulus E | 160 000 N/mm² |
| Poisson ratio ν | 0.22 |
| Density ρ | 2.32 × 10⁻³ g/mm³ |
| Rayleigh damping α, β | 0 (undamped) |

### Finite element discretisation

| Item | Setting |
|------|---------|
| Mesh | 10 × 2 × 2 hexahedral cells |
| Element type | 27-node serendipity (Lagrange order 2) |
| Quadrature | 3 × 3 × 3 Gauss–Legendre |
| Boundary conditions | All DOFs clamped at both ends (`Dirichlet` facetset) |

---

## Parametric formulation

### Parameter θ

θ controls the arch rise as a **scalar multiplier relative to the base**:

```
h(θ) = h₀ (1 + θ)
```

| θ | Arch height | Meaning |
|---|-------------|---------|
| −1 | 0 | Flat (straight) beam |
| 0 | h₀ | Base arch (reference configuration) |
| +1 | 2 h₀ | Doubled arch rise |

The `config.jl` file sets `h0_L_ratio` and `N_INCREMENTS`, which together define
the discrete set of arch heights used for validation:

```
H_RATIOS = range(0, 2·h0_L_ratio; length = N_INCREMENTS + 1)
θ(h_ratio) = h_ratio / h0_L_ratio − 1
```

### Jacobian expansion

The physical-to-reference deformation map is affine in θ:

```
x(θ, x₀) = x₀ + (1 + θ) w(x₀)
J(θ, x₀) = I + (1 + θ) J_arch(x₀)
J_arch    = (π h₀ / L) cos(π x₁ / L) · (e₂ ⊗ e₁)
```

Because `J_arch` is strictly lower-triangular its determinant is always 1, so the
element volume is preserved for all θ and the mass matrix expansion reduces to:

```
M(θ) = M₀ + θ M₁ + θ² M₂ + …
```

(M₀ is the flat-beam mass matrix; `assemble_K_M_arch!` assembles all coefficients
up to order `N_θ`.)

### θ-polynomial stiffness and mass

The linear stiffness and mass matrices are expanded as power series in θ:

```
K(θ) = K₀ + θ K₁ + θ² K₂ + … + θ^{N_θ} K_{N_θ}
M(θ) = M₀ + θ M₁ + θ² M₂ + … + θ^{N_θ} M_{N_θ}
```

The quadratic and cubic elastic forms also expand in θ at each quadrature point
via the per-QP adjugate/determinant series (see `arch_assembly.jl`).

---

## ROM construction — DPIM

### Master modes

The eigenproblem on the base-arch stiffness K₀ and mass M₀ gives a conjugate
pair of eigenvalues ±iω₀ for the first bending mode.  These two modes are chosen
as the *master modes* for the spectral submanifold (SSM).

### Augmented state

θ enters as a *frozen* external state with eigenvalue 0.  The full reduced-order
variable vector is:

```
(z₁, z₂, θ),   NVAR = 3
z₂ = z̄₁  (conjugate symmetry enforced)
```

### Multiindex set (anisotropic)

```
α = (a, b, c) ∈ ℤ³≥0
    a + b ≤ max_degree_z  (= 9 by default)
    c     ≤ max_degree_θ  (= 9 by default)
    1 ≤ a + b + c ≤ max_degree_total (= 9)
```

This allows the parametrisation to be a degree-9 polynomial in the oscillation
amplitudes (z₁, z₂) and simultaneously up to degree-9 in θ.

### Cohomological equations

MORFE solves the invariance equation order-by-order in the GrLex monomial order
to produce:

| Object | Size | Meaning |
|--------|------|---------|
| W(z₁, z₂, θ) | FOM × L | Map from reduced coords to full physical state |
| R(z₁, z₂, θ) | 2 × L | Reduced ODE: ż = R(z, θ) |

Both are serialised to `results/data/arch_h<h0_L_ratio>/W.jls` and `R.jls`.

---

## Workflow

```
config.jl                       ← edit here to change h0_L_ratio, N_INCREMENTS
    │
    ├── main.jl                 → results/data/arch_h<base>/W.jls, R.jls
    │       Parametric DPIM.  Reads config.jl.  Produces a single ROM valid
    │       for all θ ∈ [−1, +1].
    │
    ├── reference/
    │   └── compute_references.jl → results/reference/arch_h_<ratio>/data/
    │           Non-parametric DPIM for each arch height in H_RATIOS.
    │           Mode selected by mass-weighted projection onto base mode.
    │
    └── validation/
        ├── backbone.jl          → results/backbone/backbones.csv, metrics.csv
        │       Extracts conservative backbone curves from both ROMs.
        │       Generates one parametric curve per θ-truncation order.
        │
        └── plot_backbone.py     → results/backbone/*.png
                Produces the four figures described below.
```

---

## Conservative backbone extraction

For zero damping the oscillation on the SSM is conservative.  Along the backbone
the oscillation frequency Ω varies with amplitude r = |z₁|.

### From the realified reduced dynamics

After realification R₁(x₁, x₂, θ) with x₁ = Re(z₁), x₂ = Im(z₁):

```
Linear eigenfrequency:  ω₀(θ)   = Im( ∂R₁/∂x₁ |_{x=0, θ} )
Backbone frequency:     Ω(r, θ)  = Im( R₁(r, 0, θ) ) / r
```

The `r = 0` limit recovers ω₀.

### Physical amplitude

At each r, z₁ sweeps a full orbit z₁(φ) = r · e^{iφ} for φ ∈ [0, 2π).
The peak transverse displacement at the beam midpoint is:

```
a(r, θ) = max_φ | u_mid( W(z₁(φ), z₂(φ), θ) ) − u_static_mid(θ) |
```

where `u_mid` selects the y-component DOF closest to x₁ = L/2, and
`u_static_mid(θ)` is the arch camber at that DOF (W evaluated at r = 0).

### θ-truncation convergence study

The backbone is computed **four times per arch height**, with the θ-polynomial
truncated at orders:

| Truncation | Monomials retained |
|------------|-------------------|
| max (= 9) | all θ^k, k ≤ 9 |
| max − 1 (= 8) | θ^k, k ≤ 8 |
| max − 2 (= 7) | θ^k, k ≤ 7 |
| max − 3 (= 6) | θ^k, k ≤ 6 |

Convergence of the four curves to the reference backbone validates that the θ
expansion has sufficient order.

---

## Figures

All frequency axes are normalised by **ω₀_base** — the linear eigenfrequency of
the base arch (θ = 0) from the reference FEM eigensolver.

### backbone_absolute.png

| Axis | Quantity |
|------|----------|
| x | Ω / ω₀_base  — dimensionless backbone frequency |
| y | Peak transverse displacement at beam midpoint (mm) |

**Colors** (viridis, dark → light): one per arch height from h = 0 (flat) to
h = 2 h₀.

**Line styles** per model:

| Style | Model |
|-------|-------|
| Solid | Reference non-parametric ROM |
| Dashed | Parametric ROM, max θ-order |
| Dash-dot | Parametric ROM, max − 1 |
| Dotted | Parametric ROM, max − 2 |
| Loosely dashed | Parametric ROM, max − 3 |

**What to look for:** At h = 0 (flat beam) the backbone bends rightward —
hardening nonlinearity. As h increases toward h₀ and beyond, the nonlinear
frequency shift decreases and eventually reverses — softening nonlinearity.
The flat beam and doubled-arch configurations are symmetric only in the linear
limit; the nonlinear correction is not symmetric in θ.

---

### backbone_shift.png

| Axis | Quantity |
|------|----------|
| x | (Ω − ω₀) / ω₀_base  — normalised nonlinear frequency shift |
| y | Peak transverse displacement at beam midpoint (mm) |

Same color and line-style conventions as `backbone_absolute.png`.

**What to look for:** The vertical dashed line at x = 0 is the linear limit
(r → 0). Curves bending to the right = hardening; curves bending to the left =
softening. This plot reveals the nonlinear stiffness character more clearly than
the absolute-frequency plot because the linear offset is removed.

---

### backbone_h\<ratio\>.png  (one per arch height)

| Axis | Quantity |
|------|----------|
| x | (Ω − ω₀) / ω₀_base |
| y | Peak transverse displacement at beam midpoint (mm) |

**Colors:** black = reference; red = parametric ROM (all four θ-orders on the
same axes in different line styles).

**What to look for:** The four parametric curves should converge to the
reference as the θ-truncation order increases.  Large discrepancy between
adjacent orders indicates that the θ polynomial has not yet converged for that
arch height; excellent convergence (overlapping curves) indicates that the ROM
accurately represents the geometry.

---

### eigenfreq_vs_h.png

| Axis | Quantity |
|------|----------|
| x | h₀/L  — arch height ratio |
| y | ω₀ / ω₀_base  — normalised linear eigenfrequency |

**Black circles + solid line:** reference FEM eigenfrequency for each physically
assembled arch in H_RATIOS.

**Red squares + line styles (4):** parametric ROM prediction of ω₀(θ) for the
same arch heights, one line per θ-truncation order.

**What to look for:** The eigenfrequency of a sinusoidal arch increases with
rising height (arch effect stiffens the structure in the transverse direction).
All four parametric lines should match the reference closely near h = h₀
(θ = 0) and may show small divergence at the extremes h = 0 and h = 2 h₀
(θ = ±1) — these extremes are the hardest for the polynomial approximation.
Lower θ-truncation orders deviate earlier from the reference.

---

## Validation metrics

`metrics.csv` (one row per arch height per model/θ-order):

| Column | Meaning |
|--------|---------|
| `h_ratio` | Arch rise / span |
| `theta` | θ value = h_ratio / h0_L_ratio − 1 |
| `model` | `"reference"` or `"parametric"` |
| `theta_order` | θ-truncation order (−1 for reference) |
| `omega0` | Linear eigenfrequency (rad/ms) |
| `delta_omega0_rel` | \|ω₀_ref − ω₀_param\| / ω₀_ref |
| `modal_proj` | φ_ref^H M₀ φ_param(θ); ≈ ±1 → same mode, ≈ 0 → orthogonal |

The `modal_proj` check guards against the reference eigensolver returning a
different physical mode at extreme arch heights.

---

## Key source files

| File | Role |
|------|------|
| `config.jl` | Single source of truth: `h0_L_ratio`, `N_INCREMENTS` |
| `main.jl` | Parametric DPIM — produces W, R |
| `fem/arch_assembly.jl` | K/M assembly and nonlinear closure factories |
| `fem/arch_geometry.jl` | Analytical J₀, J₁ and sinusoidal arch map |
| `fem/parametric_geometry.jl` | Generic det/adj/reciprocal polynomial series |
| `reference/compute_references.jl` | Non-parametric reference ROMs |
| `validation/backbone.jl` | Backbone extraction and CSV output |
| `validation/plot_backbone.py` | Figure generation |
