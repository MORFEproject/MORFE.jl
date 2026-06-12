# 06 — Dielectric elastomer actuated cantilever (third-order DPIM)

A soft cantilever driven by a dielectric elastomer actuator with **linearly tapered
electrodes** and a non-negligible RC charging time. Eliminating the electrode charge
yields a genuinely **third-order** system of ODEs,

```
B₃ u⁽³⁾ + B₂ ü + B₁ u̇ + B₀ u = F(u, u̇, ü, v),
B₃ = RM,  B₂ = RD + ĉM,  B₁ = RK + ĉD,  B₀ = ĉK − 2c₀Q₀² b gᵀ,
F  = 2Q₀ b v + 2b σv + 4c₀Q₀ b σγ − ĉ b σ² + (ĉ/Q₀) b σ³ − (1/Q₀) b σ²v,
```

with charge proxy `σ = ℓ₀ᵀu + ℓ₁ᵀu̇ + ℓ₂ᵀü`, strain `γ = gᵀu`, and harmonic drive
`v = v_a cos Ωt` represented as a 2-D linear `ExternalSystem` (±iΩ). The closure is exact
to cubic order in the fluctuation amplitude. See `physics_description.md` (model) and
`implementation_plan_detailed.md` (derivation §1 + full term catalogue §6).

This is the first MORFE example exercising the `ORD = 3` pipeline end-to-end: cubic
eigenvalue pencil via dense companion form, `master_modes_derivatives` with two derivative
blocks, and nonlinear terms involving the *acceleration* (`multiindex` entries in the third
slot).

## Run

```bash
julia --project=. main.jl                           # checks + backbone + forced reduction
julia --project=. validation/compare_rom_fom.jl     # closure / ROM / mini-FRF vs FOM
```

First run installs the demo-local environment (MORFE via `Pkg.develop`).

## Files

| file | content |
|---|---|
| `main.jl` | top-level driver: FE → bias → model → eigen → reductions → realification → ROM response |
| `config.jl` | nondimensional parameters (knobs: `ωτ`, `ξ1`, `V0`, `v_a`, `n_elem`) |
| `fem/hermite_beam.jl` | cubic-Hermite FE assembly: `M, K, D`, tapered load `b`, strain functional `g` |
| `model/bias.jl` | static bias point `(x₀, Q₀)` (scalar cubic Newton) + derived constants `ĉ, ℓ₀, ℓ₁, ℓ₂` |
| `model/coupling_terms.jl` | `B₀…B₃`, all 26 `MultilinearMap`s, literal RHS `F_thirdorder` |
| `solver/eigensolver.jl` | cubic-pencil eigenanalysis, master-pair extraction (avoids global left/right matching on the degenerate RC cluster) |
| `solver/rom_utils.jl` | polynomial evaluation, RK4, IMEX-CN integrator, amplitude extraction |
| `validation/` | coupled `(u,q)` ground truth and the three-way comparison |

Acceptance checks run inline in `main.jl`; disable with `DEA_RUN_CHECKS=0`.

## What to expect

- Phase checks print and assert (FE: `ω₁ ≈ 3.516`; bias residuals ≈ 1e-14; map-vs-literal
  consistency 1e-12; closure error ratio ≈ 8 under ε-halving).
- Spectrum: `n` underdamped bending pairs + `n` real RC-relaxation eigenvalues near
  `−ĉ/R ≈ −ω₁` (regime `ωτ = 1`).
- Milestone 1 prints the realified autonomous reduced dynamics and the backbone
  coefficient of `z₁²z̄₁`; milestone 2 the forced reduced dynamics in `(z, z̄, r₁, r₂)` and
  the ROM steady tip amplitude at `Ω = Im λ₁`.
- Validation: closure mismatch shrinking ≈8× when `v_a` halves; ROM steady amplitude
  within 5% of the coupled FOM; ROM FRF over `Ω/ω₁ ∈ [0.97, 1.03]`.

## Notes

- Dense `Matrix{Float64}` throughout (`DefaultEigensolver`, dense bordered solves) —
  intentional for v1; the sparse path is a separate optimisation.
- Time integration is dependency-free: RK4 for the 4-D ROM, IMEX Crank–Nicolson/AB2 for
  the stiff full-order systems (Kelvin–Voigt makes high FE modes very stiff — do not swap
  in an explicit integrator).
- Each forcing frequency requires its own reduction (the external eigenvalues `±iΩ` are
  baked into the model); the mini-FRF loop in the validation script does exactly that.
- `numeric_mirror.py` replays the Phase 3–6 acceptance numerics in NumPy/SciPy
  (FE assembly, bias, proxy identity, closure order, companion spectrum) — a
  language-independent cross-check of the formulas; it does not touch MORFE.
