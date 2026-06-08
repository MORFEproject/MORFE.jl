# Kármán Vortex Street — DPIM Demo Plan

## Context

Replicate arXiv:2510.26542v1: DPIM applied to the Hopf bifurcation in cylinder flow
(Turek–Schäfer 2D benchmark, Re_c ≈ 49.03). The ROM is a 3-DOF normal-form model
(2 Hopf modes + 1 Re-parameter mode) computed at several expansion points Re₀;
it predicts limit-cycle amplitudes, Strouhal number, and TKE across a ±10% Re range
at 283× speedup over FOM.

This is the first ORD=1 (first-order PDE) use of MORFE. The unknown is velocity u,
so time appears only to first order: B₁ u̇ + B₀ u = f(u, r).

Paper results to replicate (in order of complexity):

- **Fig 4**: Eigenvalue branch λ(Re) from FOM vs ROM at Re₀ ∈ {20, Re_c, 70}
- **Fig 5**: Bifurcation diagram — TKE vs Re, FOM vs ROMs
- **Fig 6**: Order convergence (ORD = 3, 5, 7, 9) at Re₀ = Re_c, graph style
- **Fig 7**: Vorticity fields: FOM vs ROM mean, shift-mode, snapshots at Re = 52, 54
- **Fig 8–9**: a priori / a posteriori NRMSE error

---

## Mathematical Formulation (ORD = 1)

### Full-order system

Incompressible NSE linearized at steady state (u₀, p₀) and Re₀, with perturbation
η′ = 1/Re − 1/Re₀:

```text
B₁ ṡ = −B₀ s + f₂(s, s) + g₁(s, η′) + h₀(η′)
∂t η′ = 0
```

where s = [u′; p′] ∈ ℝ^FOM (velocity + pressure perturbations), and:

| Symbol | Matrix | Structure |
| ------ | ------ | --------- |
| B₁ | [M, 0; 0, 0] | velocity mass + zero for pressure (singular) |
| B₀ | −[η₀L − L_conv(u₀), −Gᵀ; G, 0] | linearised NSE operator (η₀ = 1/Re₀) |
| f₂(s,s) | −(u₁′·∇)u₂′ − (u₂′·∇)u₁′ | quadratic convection |
| g₁(s,η′) | η′ · L u′ | parametric viscous coupling |
| h₀(η′) | η′ · L u₀ | base-flow parametric forcing |

### MORFE model encoding

```julia
# ORD = 1  →  linear_terms = (B₀, B₁)  (ORDP1 = 2 elements)
# N_EXT = 1 (η′ mode via ExternalSystem, eigenvalue = 0)
# NVAR = ROM + N_EXT = 2 + 1 = 3
# master_modes_derivatives = nothing  (ORD = 1, no velocity slice needed)

ext_sys = ExternalSystem((0.0 + 0.0im,))
model = NDOrderModel((B₀, B₁), (convection, param_coupling, base_forcing), ext_sys)
```

Multilinear term encodings:

| Term | multiindex | multiplicity_external |
| ---- | ---------- | --------------------- |
| f₂ | (2,) | 0 |
| g₁ | (1,) | 1 |
| h₀ | (0,) | 1 |

### Reduced dynamics (normal form)

With master coordinates z = (z₁, z₂, η′), conjugate pair (z₁, z₂), parameter η′:

```text
ż₁ = λz₁ + c₃ z₁²z₂ + c₅ z₁³z₂² + …  (+ η′ terms at each order)
ż₂ = λ̄z₂ + …
∂t η′ = 0
```

Limit cycles: ρ̇ = Re(c₁)ρ + Re(c₃)ρ³ + …, θ̇ = Im(c₁) + Im(c₃)ρ² + …
where c_{2k+1} are polynomial functions of η′ (i.e., of Re).

---

## File Structure

```text
demo/KármánVortexStreet/
├── plan.md                   ← this document
├── mesh.jl                   ← Gmsh Turek–Schäfer mesh generation
├── fem_setup.jl              ← Ferrite DofHandler, BCs, element values
├── steady_state.jl           ← Newton solver for (u₀, p₀) at Re₀
├── linear_operators.jl       ← Assemble B₀, B₁ from FEM matrices
├── fluid_maps.jl             ← FEMMultilinearMap for f₂, g₁, h₀
├── eigensolver.jl            ← AbstractEigensolver for descriptor NSE
├── main.jl                   ← Top-level driver (mesh → DPIM → save)
└── postprocess.jl            ← Bifurcation diagram, TKE, error, plots
```

---

## Implementation Steps

### Step 1 — Mesh generation (`mesh.jl`)

Generate the Turek–Schäfer benchmark geometry with Gmsh.jl:

- Domain: [0, 2.2] × [0, 0.41]
- Cylinder: center (0.2, 0.2), diameter D = 0.1
- Physical groups: `"inlet"`, `"outlet"`, `"walls"`, `"cylinder"`, `"Ω"`
- Target DOF count ≈ 17 973 (match paper); tune via `MeshSizeMax`
- Export as `.msh` (Gmsh format 2.2 for Ferrite compatibility)

Reference: `demo/Gridap/clamped_clamped_beam.msh` generation pattern.

---

### Step 2 — FEM setup (`fem_setup.jl`)

Mixed Taylor-Hood elements in Ferrite.jl:

- Velocity: `Lagrange{RefQuadrilateral, 2}^2` (Q2, vector-valued)
- Pressure: `Lagrange{RefQuadrilateral, 1}` (Q1, scalar)
- `DofHandler` with two fields: `:u` (Q2), `:p` (Q1)
- `CellValues` for both velocity and pressure (shared quadrature rule)
- `ConstraintHandler` for Dirichlet BCs:
  - Inlet (`"inlet"`): Poiseuille profile u_inlet(y) = 4Ū y(H−y)/H²
  - Walls + cylinder (`"walls"`, `"cylinder"`): no-slip u = 0
  - Pressure: pin a pressure DOF or use natural BC at outlet
- Store `free_to_local` mapping (global DOF → index in free-DOF vector)
- Assemble global DOF partition: `[u_free; p_free]` → FOM = n_u_free + n_p_free

---

### Step 3 — Steady-state computation (`steady_state.jl`)

Newton iteration for the nonlinear steady NSE at Re₀:

```text
R(u, p) = K_visc(Re₀) u + C(u, u) + Gᵀ p − f_inlet = 0
G u = 0
```

Algorithm:

1. Start from Stokes solution (linearize convection = 0) or previous Re₀ solution
2. Assemble residual R and tangent J at each Newton step (full assembly via Ferrite)
3. Solve J Δs = −R with `KLU.klu(J)` (sparse direct solver, available in deps)
4. Update s ← s + Δs; repeat until ‖R‖ < tol = 1e-10
5. Validate: check drag/lift coefficients against Turek–Schäfer benchmark values

For continuation in Re₀: solve at Re₀ = 20 first, then arc-length-continue to
Re₀ = Re_c ≈ 49.03, 70, 80.

---

### Step 4 — Linear operator assembly (`linear_operators.jl`)

Assemble sparse matrices from FEM integrals:

```julia
M          # velocity mass matrix (n_u_free × n_u_free)
K          # viscosity stiffness (n_u_free × n_u_free): ∫ ∇v:∇u dΩ
L_conv_u0  # linearised convection at u₀ (n_u_free × n_u_free):
           # ∫ v·((u₀·∇)u + (u·∇)u₀) dΩ
G          # discrete divergence (n_p_free × n_u_free): ∫ q (∇·u) dΩ
```

Form the FOM block matrices:

```julia
A_lin = [-(1/Re₀)*K + L_conv_u0, Gᵀ; -G, 0]  # linearised NSE operator
B₁ = blockdiag(M, spzeros(n_p_free, n_p_free))
B₀ = -A_lin
# Check: eigenvalues of (B₀, B₁) should include Hopf pair near ±iω_c at Re₀=Re_c
```

---

### Step 5 — FEMMultilinearMap implementations (`fluid_maps.jl`)

Implement three `FEMMultilinearMap{1}` subtypes (ORD=1 → first-order system):

#### 5a. `FluidConvection <: FEMMultilinearMap{1}`

Assembles f₂(s₁, s₂) = −[(u₁·∇)u₂ + (u₂·∇)u₁] (symmetrised quadratic convection):

- `multiindex = (2,)`, `multiplicity_external = 0`
- `scatter_qp!`: extract velocity values and gradients at QPs from global DOF vector
- `accumulate_qp!`: compute (u₁·∇)u₂ + (u₂·∇)u₁ at each QP, add to element residual
- `assemble_element!`: scatter to velocity DOFs of global accumulator (pressure rows = 0)
- Follow `FerriteGeometricNonlinearity{2}` pattern in `demo/Ferrite/ferrite_assembly.jl`

#### 5b. `FluidParamCoupling <: FEMMultilinearMap{1}`

Assembles g₁(s, η′) = η′ · L u′ (parametric viscous coupling):

- `multiindex = (1,)`, `multiplicity_external = 1`
- Effectively returns `K_visc · u′` (scaled by η′ at evaluation time)
- Can be assembled as a sparse matrix and applied matrix-vector style,
  or implemented as a proper FEMMultilinearMap for consistency

#### 5c. `FluidBaseForcing <: FEMMultilinearMap{1}`

Assembles h₀(η′) = η′ · L u₀ (base flow parametric forcing):

- `multiindex = (0,)`, `multiplicity_external = 1`
- Returns a fixed vector (precomputed as `K_visc * u₀_free`) scaled by η′
- Provides the "driving direction" when Re departs from Re₀

Note: g₁ and h₀ might alternatively be handled via the external-system eigenvector
mechanism in MORFE (the `Φ_ext` column of the generalised eigenmodes matrix).
Verify against paper Section 2.3 and `CohomologicalDriver.jl` for external modes.

---

### Step 6 — Eigensolver (`eigensolver.jl`)

Implement `FluidEigensolver <: AbstractEigensolver` (abstract type in
`src/Eigenproblems/Eigensolvers.jl`).

**Challenge**: B₁ is singular (descriptor system). Standard ARPACK `eigs(B₀, B₁)`
fails. Use shift-invert on the linearised operator:

```julia
σ = 0.0 + iω₀_estimate * im  # target shift near Hopf frequency
# Factorize (B₀ + σ B₁) once (KLU or SuiteSparse.lu)
# Then eigs via LinearMap: v ↦ (B₀ + σ B₁)⁻¹ (B₁ v)
# This gives eigenvalues μ = 1/(λ − σ); recover λ = σ + 1/μ
```

ω₀ estimate: start with ω₀ ≈ 1.0 rad/s (dimensionless Strouhal ≈ 0.2 for Re ≈ 50).

Return right eigenvectors Y (full FOM including pressure), eigenvalues λ.

**Left eigenvectors** (adjoint problem): solve `(B₀ + σ B₁)ᴴ w = μ B₁ᴴ w`
using the same factorisation infrastructure. Required for bordered system in DPIM.

**Normalisation**: enforce biorthogonality ⟨ψᵢ, B₁ φⱼ⟩ = δᵢⱼ.

Select master modes: the complex conjugate pair with Re(λ) closest to 0 (Hopf pair).

---

### Step 7 — NDOrderModel and DPIM solve (`main.jl`)

```julia
ext_sys = ExternalSystem((0.0 + 0.0im,))  # η′ mode
model = NDOrderModel((B₀, B₁), (convection, param_coupling, base_forcing), ext_sys)

NVAR    = 3      # 2 Hopf + 1 parameter
ROM     = 2      # master modes from Hopf pair
max_ord = 5      # paper uses 3, 5, 7, 9; start with 5

mset = all_multiindices_up_to(NVAR, max_ord; min_degree = 1)
resonance_set = resonance_set_from_complex_normal_form_style(
    mset, master_eigenvalues, 0.05)

master_modes    = Y[1:FOM, 1:ROM]   # right eigenvectors (position only, ORD=1)
left_eigenmodes = X[1:FOM, 1:ROM]   # left eigenvectors
# master_modes_derivatives = nothing  ← ORD=1: not needed

W, R = solve_cohomological_problem(
    model, mset,
    master_eigenvalues,
    master_modes, left_eigenmodes,
    resonance_set;
    conjugate_permutation = SVector(2, 1, 3),  # z₁ ↔ z₂ conjugate, η′ self-conjugate
)
```

Repeat for each expansion point Re₀ ∈ {20, Re_c, 70, 80}.

---

### Step 8 — Post-processing (`postprocess.jl`)

```julia
conj_map = [2, 1, 3]
Rr = ReducedDynamics(realify(R.poly, conj_map), R.external_system_size)
```

**Bifurcation diagram** (Fig 5):

- Extract Stuart–Landau coefficients c₁(η′), c₃(η′), … from Rr
- Find limit-cycle amplitude ρ*(Re) from ρ̇ = 0: Re(c₁)ρ + Re(c₃)ρ³ + … = 0
- Compute TKE = ½ρ² ‖W[:, 1, :]‖² (L² norm over velocity DOFs)

**Eigenvalue branch** (Fig 4):

- Evaluate λ(Re) = c₁(η′(Re)) from reduced dynamics

**Flow fields** (Fig 7):

- Evaluate W(z*(Re)) to reconstruct mean flow, shift mode, instantaneous snapshots
- Use Ferrite WriteVTK extension to export; vorticity ω_z = ∂_x u_y − ∂_y u_x

**Error analysis** (Fig 8–9):

- NRMSE = ‖s_FOM − W(z*(Re))‖ / max‖s_FOM‖
- a priori from MORFE; a posteriori from FOM time integration

**FOM validation** (`fom_timeint.jl`):

- Crank–Nicholson time integrator; Δt ≈ 0.01–0.05 (dimensionless)
- Integrate ≥ 5 shedding periods per Re value

---

## Key Design Decisions

| Decision | Choice | Reason |
| -------- | ------ | ------ |
| FEM library | Ferrite.jl | In MORFE.jl deps; FEMMultilinearMap batch path available |
| Elements | Q2/Q1 Taylor-Hood | LBB-stable, standard for incompressible flow |
| Eigensolver | Shift-invert ARPACK via LinearMap.jl | Handles singular B₁; KLU factorisation already in deps |
| Parametrisation style | Normal form (primary) + graph (conv.) | Matches paper; cleaner reduced dynamics |
| Expansion orders | 5 (primary), 3/7/9 (convergence) | Paper Figs 5–6 |
| Conjugate symmetry | `conjugate_permutation = [2, 1, 3]` | η′ is real → self-conjugate; halves solve cost |

---

## Potential Issues to Verify Before Full DPIM Run

1. **ORD=1 in MORFE**: Codebase tested only for ORD=2. Check that
   `CohomologicalEquations.jl` and `compute_higher_derivative_coefficients!`
   do not hard-code ORD=2 assumptions.

2. **Singular B₁ / descriptor**: Confirm `(B₀ + σB₁)` is non-singular for σ = iω₀ ≠ 0.
   Verify the bordered system in MORFE does not break with a singular mass matrix.

3. **ExternalSystem eigenvalue = 0**: Check that the bordered system does not divide by
   λ_ext = 0 anywhere in `CohomologicalDriver.jl`.

4. **h₀ direction encoding**: Verify η′ L u₀ is captured as multiindex=(0,) term OR
   through the `Φ_ext` external eigenvector mechanism (paper Section 2.3).

5. **Inlet BC lifting**: Poiseuille profile must be homogenised before linearisation;
   residual absorbed into the steady-state RHS.

---

## Verification Sequence

1. **Mesh**: visualise, confirm cylinder boundary, DOF count ≈ 17 973.
2. **Steady state** at Re₀ = 20: verify drag Cd ≈ 5.57 (Turek benchmark).
3. **Eigenvalue at Re₀ = Re_c**: Re(λ) ≈ 0, Im(λ) ≈ ω_c (Strouhal St ≈ 0.2).
4. **DPIM order 3 at Re₀ = Re_c**: bifurcation point recovered exactly.
5. **Bifurcation diagram**: TKE increases past Re ≈ 49.03, matches Fig 5 qualitatively.
6. **Order convergence**: orders 3, 5, 7, 9 progressively improve, matching Fig 6.
