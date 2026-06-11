# Implementation plan — DEA soft beam as a third-order `NDOrderModel`

Goal: implement the dielectric-elastomer-actuated cantilever (see `motivation.md`) as a genuine
`ORD = 3` `NDOrderModel` with harmonic voltage forcing via an `ExternalSystem`, and reduce it
with `solve_cohomological_problem`. Discretisation: self-contained 1D cubic-Hermite
Euler–Bernoulli FE (no Ferrite backend needed).

---

## 1. Physical model and the route to an exactly-polynomial third-order system

Continuous equations (state `w(x,t)`, scalar charge `Q(t)`):

```
ρA ẅ + EI w'''' + ηI ẇ''''  =  (Q²/2εb) · p(w)          (mech)
R Q̇ + c₀ Q (1 − α ⟨w_x⟩)    =  V(t),   c₀ = h₀/(εbL)     (elec)
```

`p(w)` is the electrostatic load shape (expanded `p(w) = p₀ + P₁w + P₂(w,w)`), `⟨w_x⟩ = gᵀw`
a strain-weighting functional. The naive elimination of `Q` in `motivation.md` is not
polynomial; the clean route is:

**Step A — bias expansion.** Apply `V(t) = V₀ + v(t)`. Solve the static problem
(Newton) for `(w₀, Q₀)`. Shift `u = w − w₀`, `q = Q − Q₀` and Taylor-expand both equations
to **cubic** order. This yields, discretely:

```
M ü + D u̇ + K̃ u = β₁ q + N₂(u,q) + N₃(u,q)              (mech′)
R q̇ + ĉ q       = v + ĉ_w gᵀu + c₀α q gᵀu               (elec′)
```

with `K̃ = K − (Q₀²/2εb)P₁` (electrostatic softening), `β₁ = (Q₀/εb)p(w₀)`,
`ĉ = c₀(1 − α gᵀw₀)`, `ĉ_w = c₀αQ₀`. All nonlinear terms are polynomial in `(u, q)`.

**Step B — exact linear closure.** Form `R·d/dt(mech′) + ĉ·(mech′)`. The linear charge
contribution appears only as `β₁(R q̇ + ĉ q)`, which (elec′) replaces by `v` plus polynomial
terms. Result:

```
B₃ u⁗⁽³⁾ + B₂ ü + B₁ u̇ + B₀ u = F(u, u̇, ü, r)
B₃ = R M
B₂ = R D + ĉ M
B₁ = R K̃ + ĉ D
B₀ = ĉ K̃ − ĉ_w β₁ gᵀ        (rank-one electromechanical update)
```

**Step C — residual-charge substitution.** `q` still appears inside quadratic/cubic terms.
Two rules, applied in this order:

1. `q̇` is always replaced via (elec′) **before** any substitution (so `u⁽³⁾` never enters F).
2. Remaining `q` factors are replaced by the leading-order functional
   `q ≈ ℓ(u, u̇, ü) := (β₁ᵀ(M ü + D u̇ + K̃ u)) / (β₁ᵀβ₁)`,
   i.e. `q = ℓ₀ᵀu + ℓ₁ᵀu̇ + ℓ₂ᵀü`.

Since these `q` factors only occur inside terms of degree ≥ 2, the substitution error is
O(4) — beyond the cubic truncation. The third-order system is therefore **exact to cubic
order**, with purely polynomial nonlinearities in `(u, u̇, ü, r)`. Document this derivation in
`derivation.md` with the full term bookkeeping (do it symbolically once, by hand or with
Symbolics.jl, and freeze the coefficient formulas in the assembly code).

## 2. Term catalogue → `MultilinearMap`s

`multiindex = (i₀, i₁, i₂)` = multiplicities of `(u, u̇, ü)`; `me` = `multiplicity_external`.
Each term's `f!` is a closure over precomputed sparse operators (vectors `β₁, g, ℓₖ`,
matrices `M, D, K̃, P₁`, and the FE 3-tensor for `P₂`). Schematic catalogue:

| Origin | Resulting multiindices | deg | me |
|---|---|---|---|
| `β₁ v` (direct forcing) | `(0,0,0)` | 1 | 1 |
| `(ℓᵀX)·v` from `q·v` products | `(1,0,0), (0,1,0), (0,0,1)` | 2 | 1 |
| `q²`, `q·gᵀu`, bilinear coupling | all pairs from `{u,u̇,ü}`: `(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)` | 2 | 0 |
| `ĉN₃ + R dN₃/dt`, `q·N₂`-type | triples, e.g. `(3,0,0), (2,1,0), (2,0,1), (1,1,1), …` | 3 | 0 |
| geometric beam nonlinearity (optional, von-Kármán axial coupling) | `(2,0,0), (3,0,0)` | 2–3 | 0 |

Symmetry flags: terms with one multiindex entry > 1 (e.g. `(2,0,0)`) must have `f!`
symmetric in those slots — implement them symmetrised and pass `fully_asymmetric = false`
explicitly to silence the construction `@info`. Mixed terms like `(1,0,1)` are
`FullyAsymmetric` automatically.

Implementation detail: rank-one structure everywhere — e.g. the `(1,0,1)` term from
`c₀α β₁ (gᵀu)(ℓ₂ᵀü)` is `f!(res, a, b) = res .+= c·(gᵀa)·(ℓ₂ᵀb)·β₁`, O(n) per call. No FEM
element loops are needed at solve time, so plain `MultilinearMap` suffices (no
`FEMMultilinearMap` subtype).

## 3. External system

`v(t) = v_a cos(Ωt)` → `r = (r₁, r₂)`, `ṙ = diag(iΩ, −iΩ) r`, `v = v_a(r₁ + r₂)/…` folded
into coefficients. Use the eigenvalue-tuple constructor:
`ext = ExternalSystem((im*Ω, -im*Ω))`. No `v̇` term arises (the closure in Step B never
differentiates (elec′)), so the `R V̇` term from `motivation.md` is avoided by construction.

## 4. Model assembly

```julia
model = NDOrderModel(
    (B0, B1, B2, B3),          # ORDP1 = 4 → ORD = 3
    (terms_deg2..., terms_deg3..., terms_forcing...),
    ext,
)
```

All `Bᵢ` sparse (`SparseMatrixCSC`) so the sparse Pardiso→KLU→SuiteSparse bordered-solve
path is exercised. `B₃ = R M` is SPD — companion form is regular.

## 5. Eigenproblem

Cubic pencil `(λ³B₃ + λ²B₂ + λB₁ + B₀)φ = 0`, 3n eigenvalues: n underdamped
mechanical-like pairs + n (mostly real) RC-relaxation branch. Plan: custom
`AbstractEigensolver` (mirroring `Mechanical_Problem_Solver` in example 01) that

1. builds `(A, B) = linear_first_order_matrices(model)` (3n × 3n sparse),
2. runs Arpack shift-invert at `σ ≈ i ω₁` (first bending frequency of `K̃, M`) for right
   eigenpairs, and on the transposed pencil for left eigenvectors (`solve_left`),
3. returns eigenvectors reshaped `(FOM, 3, nev)` — blocks `(φ, λφ, λ²φ)`.

**Parameter choice matters:** pick `R` so the electrical time constant `τ = R/ĉ` satisfies
`ω₁τ = O(1)` (the regime motivating third order). Verify a spectral gap between the master
pair and both the second bending pair and the nearest RC eigenvalue before trusting the ROM.

## 6. Reduction setup

ROM = 2 (first bending pair), N_EXT = 2, NVAR = 4.

1. `mset = all_multiindices_up_to(4, max_degree; min_degree = 1)`, filtered to external
   degree ≤ 1 (ε-order forcing) — keeps L small and avoids `r`-powers.
2. `master_modes_derivatives::Array{ComplexF64}(FOM, 2, 2)` filled with the `λφ, λ²φ`
   blocks (ORD − 1 = 2 derivative blocks — note this differs from the second-order examples).
3. `resonance_set_from_complex_normal_form_style(mset, master_eigenvalues, tol)`; check that
   near-resonant forcing `Ω ≈ ω₁` is marked resonant in the external monomials. Watch the
   RC eigenvalue: if it falls near `ω₁` combinations, widen `tol` deliberately or move `R`.
4. `solve_cohomological_problem(model, mset, master_eigenvalues, master_modes,
   left_eigenmodes, resonance_set; master_modes_derivatives, conjugate_permutation = [2,1])`.
5. Realify as in example 01 (`realify` + `extract_component`, conj-map pairing `z, z̄` and
   the external pair).

## 7. Validation

1. **Invariance error** (`src/Validation`): evaluate the residual of the full third-order
   system on the manifold over a grid of `(z, r)` amplitudes.
2. **FOM reference**: integrate the *original coupled* `(u, q)` system (second order + first
   order, no elimination) with OrdinaryDiffEq; compare tip displacement against the ROM and
   against direct integration of the eliminated third-order system. This independently
   validates Step B/C, not just the reduction.
3. **FRF**: sweep `Ω` near `ω₁`, compare ROM frequency response (continuation on reduced
   dynamics) vs FOM harmonic-balance or long-time integration at a few points.
4. Convergence in `max_degree` (5 → 7 → 9) and in mesh size.

## 8. File layout

```
examples/06_dielectric_elastomer_actuator/
├── Project.toml                  # MORFE (dev path) + Arpack, LinearMaps, SparseArrays,
│                                 # StaticArrays, OrdinaryDiffEq (validation only)
├── README.md                     # summary + how to run
├── derivation.md                 # Steps A–C algebra, full term/coefficient catalogue
├── fem/
│   └── hermite_beam.jl           # K, M, D=ηI·K-pattern, g, p₀, P₁, P₂ assembly; bias Newton
├── model/
│   ├── parameters.jl             # geometry, silicone material, ε, R, V₀, nondimensionalisation
│   └── coupling_terms.jl         # builds (B0..B3) and all MultilinearMaps from Step C
├── dea_demo.jl                   # main: assemble → eigensolve → reduce → realify → FRF
└── validation/
    ├── coupled_fom_integration.jl
    └── invariance_check.jl
```

## 9. Milestones

1. Hermite FE assembly + bias equilibrium; check `ω₁` against analytic cantilever value.
2. Autonomous model (`v = 0`, no external system): build `NDOrderModel` ORD=3, eigensolve,
   backbone of the biased beam. First exercise of the ORD=3 pipeline end-to-end.
3. Add `ExternalSystem` ±iΩ and forcing terms; forced response near `Ω ≈ ω₁`.
4. Validation suite (Section 7); README.

## 10. Risks / open points

- `q`-substitution functional `ℓ` is one defensible choice of leading-order inverse
  (projection along `β₁`); the invariance-error check is the arbiter. If error is too large,
  carry the next-order correction of `ℓ` (still polynomial).
- Nondimensionalise (`x/L`, `t·ω₁`, `u/h₀`, `q/Q₀`): raw SI coefficients span ~12 orders of
  magnitude and will hurt `lu!`/KLU conditioning in the bordered solves.
- Hermite DOFs mix `w` and `w′` units — include `w′·L` scaling in the nondimensionalisation.
- Arpack on the 3n companion pencil with complex shift: use the real-shift trick or
  `LinearMaps` complex shift-invert as in the existing MORFEArpackExt path; verify left/right
  biorthogonality `Yᵀ B X = I` numerically before the solve.
- If degree-3 term count from Step C explodes, group same-multiindex contributions into one
  `MultilinearMap` each (sum of rank-one ops inside a single `f!`) — keeps `N_NL` small.
