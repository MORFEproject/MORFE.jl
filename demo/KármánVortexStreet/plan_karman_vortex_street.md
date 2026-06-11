# Plan: Fix and Complete the Kármán Vortex Street DPIM Demo

## Problem Statement

The DPIM ROM produces `Re(c₂₁₀) = +9.93×10¹⁷` (subcritical) when the cylinder-flow
Hopf bifurcation is known to be supercritical — `Re(c₂₁₀)` must be **negative**.
Every downstream result (bifurcation diagram, 3D Hopf tube, Strouhal frequency) is
wrong as a consequence. The η-dependent coefficients are equally broken
(`c₁₀₂ = 2.28×10¹¹`, `c₁₀₄ = 1.77×10³³`), confirming the entire ROM is globally wrong.

---

## What Has Been Ruled Out

| Hypothesis | Status | Reason |
|---|---|---|
| K_visc sign error | Already fixed | Affects c₁₀₁ only (η coupling), not c₂₁₀ |
| h₀ forcing sign error | Already fixed | Same — c₂₁₀ is autonomous (no η) |
| Bilinear vs sesquilinear left eigenvector | Not the cause | For real A, B: ψ_MORFE = conj(ψ_L) → ψ_MORFE^H B = ψ_L^T B (identical projection) |
| Phase rotation of φ₁ | Not the cause | Re(c₂₁₀) is invariant under e^{iθ} rotation of z₁ |
| Pure scaling of φ₁ | Not the cause | c₂₁₀ → c₂₁₀ × |α|² (positive real factor, preserves sign) |
| DOF set contamination (free_dpim) | **Untested** | 94 inlet DOFs with Neumann BC could contaminate φ₁ |
| FluidConvection sign vs MORFE convention | **Primary suspect** | c₂₁₀ is purely from FluidConvection; MORFE may expect opposite sign |

`c₂₁₀` is determined solely by `FluidConvection` (the quadratic convective nonlinearity
has no η involvement). The sign error must originate either there or in how MORFE
ingests multilinear map output inside the cohomological equations.

---

## Reference Comparison

| Quantity | Reference (DPIM2D_NS, Re=49) | MORFE (Re₀=49.03) |
|---|---|---|
| σ₀ = Re(λ₁) | −0.00288 | −0.071477 |
| ω₀ = Im(λ₁) | 16.851 | 16.939 |
| Re(c₁₀₁) | −21.19 | −77.06 ✓ (same sign) |
| **Re(c₂₁₀)** | **−112.51 (supercritical ✓)** | **+9.93×10¹⁷ (subcritical ✗)** |
| Backbone method | matcont numerical continuation | analytic F(ρ,η)=0 (fails with wrong sign) |
| Max DPIM order | 3 | 5 |

The reference computes the backbone via **matcont numerical continuation on the
realified equations**, not analytically. The realified form is for BifurcationKit /
matcont — not for the amplitude equation.

---

## Phase 1 — Read MORFE's Internal Sign Convention (no re-run required)

**Goal**: Determine whether MORFE's cohomological framework expects the multilinear
map to contribute as `+N` or `−N` to the cohomological RHS.

### Step 1.1 — Read `CohomologicalEquations.jl`

File: `src/ParametrisationMethod/CohomologicalEquations.jl`

Find where the bordered system is assembled and locate how the nonlinear RHS enters:

```
[A − σB  |  C ] [W]   [  ??? · nonlinear_rhs  ]
[  Ĉ     |  0 ] [r] = [          0            ]
```

Determine: is the RHS `+nonlinear_rhs` or `−nonlinear_rhs`?

### Step 1.2 — Read `MultilinearTerms.jl`

File: `src/ParametrisationMethod/MultilinearTerms.jl`

Check:
- Does it negate the map's output before adding to the RHS vector?
- Does it treat `multiindex=(2,)` (quadratic, no external parameter) differently
  from `multiindex=(1,)` (linear + one external)?

### Step 1.3 — Compare with a known-working demo

Read the `FEMMultilinearMap` implementation in the Ferrite geometric nonlinearity demo
(`demo/Ferrite/` or `demo/ArchComsolWedge/`). That demo produces correct Landau
coefficients. Compare the sign in its `accumulate_qp!` with `FluidConvection`.

If the working demo uses `+conv` and `FluidConvection` uses `−conv` (or vice versa),
the sign convention mismatch is confirmed.

**Decision gate**: If MORFE expects `−N(s,s)`, flip the sign in
`FluidConvection.accumulate_qp!`. One-line fix.

---

## Phase 2 — Unit-Test FluidConvection (no re-run required)

**Goal**: Verify that `FluidConvection` computes the correct nonlinear convection by
checking it against the finite-difference second derivative of the full NSE residual.

### Step 2.1 — Write `debug_conv.jl`

The quadratic remainder of the NSE satisfies:

```
N₂(φ₁, φ₁) ≈ [R(s₀ + ε·φ₁) − R(s₀) − ε·A_lin·φ₁] / ε²
```

where `R(s)` is the full nonlinear NSE residual. Procedure:

1. Load `W.jls` and extract `φ₁ = W.poly.coefficients[:, 1, 1]` (monomial [1,0,0]).
2. Re-assemble the Newton residual at `s₀ + ε·φ₁` and at `s₀` (both fast).
3. Compute the FD estimate above.
4. Compute `FluidConvection(φ₁, φ₁)` directly using `scatter_qp!` + `accumulate_qp!`
   + `assemble_element!`.
5. Compare: if they agree to within FD error, the assembly is correct. If they differ
   by sign, the convention is wrong.

### Step 2.2 — Check `N₂(φ₁, φ̄₁)` is physically reasonable

`N₂(φ₁, φ̄₁)` is the mean-flow modification (Reynolds stress from the Hopf mode).
It should be:
- Real-valued (imaginary part ≈ 0)
- Localized near the cylinder wake
- Magnitude O(‖φ₁‖²) × geometric factor

If it is purely imaginary or has wrong spatial structure, the assembly is wrong.

---

## Phase 3 — Check Inlet DOF Contamination (30 min, no re-run)

**Goal**: Quantify whether the 94 inlet DOFs in `free_dpim \ free` carry significant
amplitude in φ₁ and could corrupt c₂₁₀.

### Step 3.1 — Measure φ₁ amplitude at inlet DOFs

```julia
# load fom (re-run setup_fem) or load from vtk_data.jls
inlet_local = [i for (i,d) in enumerate(fom.free_dpim)
               if d ∉ Set(fom.free)]
φ₁_inlet    = φ₁[inlet_local]
@printf "Inlet: max=%.3g  rms=%.3g  (‖φ₁‖=%.3g)\n" \
    maximum(abs, φ₁_inlet) norm(φ₁_inlet) norm(φ₁)
```

**Interpretation**:
- `‖φ₁_inlet‖ / ‖φ₁‖ < 1e-4` → inlet DOFs negligible; DOF set is not the issue.
- `‖φ₁_inlet‖ / ‖φ₁‖ ~ O(1)` → significant contamination; switch eigenproblem to `free`.

### Step 3.2 (conditional) — Re-run eigenproblem on `free` DOFs

If Step 3.1 shows significant inlet amplitude: modify `main.jl` to assemble `B₀`, `B₁`
on `free` (57860 DOFs) instead of `free_dpim` (57954 DOFs) for the eigensolver step
only. Re-run just the eigenproblem (fast) and compare the new eigenvalue and eigenvector.

---

## Phase 4 — Apply the Fix and Re-Run (runtime: ~60 s, ~80 GB)

Once Phases 1–3 identify the root cause, apply the corresponding fix and re-run `main.jl`.

### Scenario A (most likely): FluidConvection sign is wrong for MORFE's convention

In `demo/KármánVortexStreet/fluid_maps.jl`, `accumulate_qp!`:

```julia
# Current (may be wrong):
conv = -0.5 * (qp2.grad ⋅ qp1.val + qp1.grad ⋅ qp2.val)

# Fix (if MORFE expects the map to return the integrand with opposite sign):
conv = +0.5 * (qp2.grad ⋅ qp1.val + qp1.grad ⋅ qp2.val)
```

### Scenario B: Eigenproblem should use `free` (inlet fixed)

In `linear_operators.jl`, change the restriction from `free_dpim` to `free` for the
matrices passed to `solve_hopf_eigenproblem`. Adjust `FluidConvection` and
`assemble_K_visc` accordingly so all DPIM operators use the same DOF set.

### Scenario C: Combination of A and B.

### Verification after re-run

Check `summary.log` immediately after:

1. `Re(c₂₁₀) < 0` — supercritical ✓
2. `|c₂₁₀| / ‖φ₁‖²` is O(10²–10³) — physically reasonable magnitude ✓
3. `ω₀ ≈ 16.9 rad/s` (unchanged, since linear problem is not affected) ✓
4. `|c₁₀₂|` and `|c₁₀₄|` reduced to physically reasonable values ✓

---

## Phase 5 — Rewrite `compute_backbone.jl` (2–3 hours, after Phase 4)

### Step 5.1 — Replace realify with direct complex amplitude equation

The limit-cycle amplitude satisfies:

```
F(ρ, η) = Σ_{α₁−α₂=1} c_α · ρ^{α₁+α₂−1} · η^{α₃} = 0
```

This function is already proven correct in `debug_norm.jl`. The realify-based approach
(`R1_full = extract_component(realify(R.poly, conj_map), 1)`) is only for BifurcationKit
and must NOT be used for the amplitude equation. Replace `find_limit_cycle`:

```julia
function F_complex(ρ::Float64, η::Float64, exps, c1)
    val = zero(ComplexF64)
    for (k, α) in enumerate(exps)
        α[1] - α[2] == 1 || continue
        val += c1[k] * ρ^(α[1]+α[2]-1) * η^α[3]
    end
    return val
end

function find_limit_cycle(η, exps, c1)
    f(ρ) = real(F_complex(ρ, η, exps, c1))
    rhos = [10.0^x for x in range(-15, 4, 80)]
    vals = f.(rhos)
    for i in 1:length(rhos)-1
        vals[i]*vals[i+1] < 0 || continue
        return find_zero(f, (rhos[i], rhos[i+1]), Bisection())
    end
    return NaN
end
```

### Step 5.2 — Find Re_c stably from the linear coefficients

The σ(Re) function must use ONLY the η-linear terms to avoid divergence from higher-order
η coefficients:

```julia
# Extract c₁₀₀ and c₁₀₁ directly
c100 = c1[findfirst(α -> α == SA[1,0,0], exps)]
c101 = c1[findfirst(α -> α == SA[1,0,1], exps)]

# Linear growth rate (stable, no higher-order pollution):
σ_linear(Re) = real(c100) + real(c101) * (1.0/Re - 1.0/Re₀)

Re_c = find_zero(σ_linear, (44.0, 62.0), Bisection())
```

### Step 5.3 — Scale to physical amplitude

```julia
φ₁_norm = norm(W.poly.coefficients[:, 1, 1])   # ‖φ₁‖₂ from W monomial [1,0,0]
ρ_phys  = ρ_star * φ₁_norm                      # physical velocity amplitude [m/s]
```

Plot `ρ_phys` (not raw `ρ_star`) on the y-axis. Expected range: O(0.01–0.3) m/s.

### Step 5.4 — Compute Strouhal number correctly

```julia
Ω_lc = imag(F_complex(ρ_star, η, exps, c1))   # angular frequency [rad/s]
St   = Ω_lc * _CYL_D / (2π * U_MEAN)          # Strouhal = f·D/U
```

Expected: Ω ≈ 16–19 rad/s, St ≈ 0.18–0.20.

### Step 5.5 — Fix the three plots

**`bifurcation_diagram.png`** (ρ_phys vs Re):
- Solid horizontal line at ρ=0 for Re ≤ Re_c (stable fixed point)
- Dashed horizontal line at ρ=0 for Re > Re_c (unstable fixed point)
- Bifurcation marker at (Re_c, 0)
- Limit-cycle branch emerging perpendicularly at Re_c, growing as √(Re − Re_c)
- y-axis: `ρ_phys` in physical units

**`hopf_3d.png`** (Re, a₁, a₂ where aᵢ = ρ_phys × cos/sin(θ)):
- Fixed-point line along Re-axis (circles of radius 0 for Re ≤ Re_c)
- Circles of radius `ρ_phys(Re)` for Re > Re_c, stacked along Re-axis
- Circles must be drawn in the (a₁, a₂) plane, not (ρ, 0) — use full angle range

**`strouhal_vs_Re.png`** (Ω vs Re):
- Single smooth curve; should be nearly flat near ω₀ with slight variation

---

## Phase 6 (Optional) — BifurcationKit Cross-Check

The reference uses MATLAB matcont for numerical continuation. To reproduce it in Julia:

1. Use the already-computed realified ODE `Rr` (variable: a₁, a₂, η).
2. Integrate at Re = Re_c + δ (small δ > 0) from a small initial condition to find
   a limit cycle.
3. Feed into `BifurcationKit.continuation` with ν (or Re) as the parameter.
4. Compare the BifurcationKit FRC with the analytic backbone from Phase 5.

This cross-check validates that the analytic amplitude equation gives the same branch
as full numerical continuation.

---

## Expected Final Results

| Quantity | Expected value |
|---|---|
| Re_c | ≈ 51.4 (from σ₀ = −0.0715, Re(c₁₀₁) = −77.06) |
| Re(c₂₁₀) after fix | Negative |
| ρ_phys at Re = 55 | O(0.05–0.2) m/s |
| Ω at Re = 55 | ≈ 16–19 rad/s |
| Strouhal number | ≈ 0.18–0.20 |
| Bifurcation diagram shape | Perpendicular branches at Re_c; ρ_phys ∝ √(Re − Re_c) |
| 3D Hopf tube | Visible circles growing from Re_c |

---

## Execution Order

```
Phase 1  (read MORFE source — determines root cause)
    ↓
Phase 2  (unit test FluidConvection — confirms Phase 1 conclusion)
    ↓
Phase 4  (apply fix, re-run main.jl — ~60 s, ~80 GB)
    ↓
Phase 5  (rewrite compute_backbone.jl — produces correct plots)
    ↓
Phase 3  (only if Phase 4 still shows wrong sign — check inlet DOFs)
    ↓
Phase 6  (optional BifurcationKit cross-check)
```

Phase 3 is deprioritised because the Hopf mode is localized in the cylinder wake and
inlet DOF amplitudes are expected to be near zero. Phase 1 and 2 address the more
probable FluidConvection sign convention error.
