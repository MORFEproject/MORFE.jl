# 05 — Kármán vortex street

## Model

Incompressible Navier-Stokes flow past a cylinder (2D, Ferrite P2/P1 Taylor-Hood,
Turek–Schäfer channel geometry). The steady base flow is computed first; then DPIM
parametrises the unstable spectral submanifold associated with the Hopf bifurcation
(Re_c ≈ 49) that gives rise to the Kármán vortex street, with the Reynolds-number
offset `η′ = 1/Re − 1/Re₀` as an extra parametric coordinate. Parameters are set in
`config.jl` (default: Re₀ = 49.03, expansion order 9).

## Three-step workflow

```bash
cd examples/05_karman_vortex_street

# 1. One DPIM run at order MAX_ORD
julia --project=. main.jl

# 2. ROM limit-cycle branch vs Re, one rom_branch_ord<N>.csv per truncation order
julia --project=. solve_rom.jl

# 3. Order comparison: max|lift| and period-averaged TKE vs Re per truncation
python3 compare_orders.py     # needs numpy + matplotlib
```

A single run at `MAX_ORD` suffices for the whole order-convergence study: the
cohomological solve is graded (degree-N coefficients never depend on degrees > N), so
the order-N ROM is the exact truncation of the order-9 one (verified bit-exact).
Lower orders are obtained by truncating (1) the reduced dynamics `R` when tracing the
bifurcation diagram and (2) the observables — lift polynomial and TKE Gram — when
evaluating the physical quantities, per `TRUNC_ORDERS` in config.jl.

Measured runtime (Apple Silicon, ~58k free DOFs): ≈ 7 min — shared stages ≈ 35 s +
order-9 cohomological solve ≈ 344 s. Steps 2 and 3 take seconds.

## Outputs

```text
results/
  Re49.03_ord9/                — the single DPIM run directory
    summary.log                — tee'd run log
    summary.txt                — structured key:value summary
    data/
      W.jls, R.jls             — parametrisation + reduced dynamics (serialised)
      reduced_dynamics.txt     — realified Stuart-Landau coefficients
      R_coefficients.csv       — complex reduced dynamics, one monomial per row
      L_coefficients.csv       — pressure-lift polynomial L(z) (+ base-flow constant)
      lift_polynomial.jls      — same, serialised
      tke_gram_{re,im}.csv     — kinetic-energy Gram matrix G = WᵀM_velW/|Ω|
      tke_avector.csv          — monomial exponents for G
      rom_branch_ord{3,5,7,9}.csv — (from solve_rom.jl)  eta, Re, rho, omega, T
      vtk_data.jls             — mesh + mode bundle for ParaView export
  comparison/                  — (from compare_orders.py)
    comparison.csv             — order, eta, Re, rho, omega, T, avg_TKE, max_abs_lift
    lift_vs_Re.png, tke_vs_Re.png
```

## Files

| File | Purpose |
| ---- | ------- |
| `main.jl` | Step 1 — single DPIM run at MAX_ORD |
| `solve_rom.jl` | Step 2 — ROM limit-cycle branch (PALC) per truncation order |
| `compare_orders.py` | Step 3 — truncation-order lift / avg-TKE comparison |
| `config.jl` | All parameters (Re₀, MAX_ORD, TRUNC_ORDERS, mesh, eigensolver, branch) |
| `fem/mesh.jl` | Gmsh channel-with-cylinder mesh generation |
| `fem/fem_setup.jl` | FEM spaces, boundary conditions, DOF sets |
| `fem/linear_operators.jl` | Linearised NSE operators B₀, B₁ + lift weights |
| `fem/fluid_maps.jl` | Convection / parametric multilinear maps + K_visc |
| `fem/energy_gram.jl` | Kinetic-energy Gram matrix for the TKE observable |
| `solver/steady_state.jl` | Newton solve for the base flow |
| `solver/eigensolver.jl` | Shift-invert ARPACK Hopf eigensolver |
| `solver/rom_palc.jl` | Pseudo-arclength continuation toolkit for the ROM branch |
| `validation/average_tke.py` | TKE evaluation library (used by compare_orders.py) |
| `validation/run_tke.py` | Single-orbit TKE runner (cross-check) |
| `validation/validate_tke.jl` | Independent FOM-space TKE check |
| `validation/generate_matlab.py` | Optional matcont/COCO export (`EXPORT_MATLAB = true`) |

## Validation

- `julia --project=. validation/validate_tke.jl results/Re49.03_ord9` recomputes the
  period-averaged TKE by direct integration in the full DOF space (independent of the
  Gram-matrix path).
- `python3 validation/run_tke.py --data-dir results/Re49.03_ord9/data --orbit <csv> --eta <η′>`
  evaluates a single orbit's TKE.
- Setting `EXPORT_MATLAB = true` in `config.jl` additionally emits
  `vec_fields_karman.m` / `lift_karman.m` per run for matcont/COCO cross-checks.
