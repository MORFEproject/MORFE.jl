# DEA soft beam — detailed implementation specification

This document is a **complete, step-by-step specification** for implementing
`examples/06_dielectric_elastomer_actuator` as a third-order (`ORD = 3`) `NDOrderModel`
reduced with `solve_cohomological_problem`. It is written so that an implementer can follow
it phase by phase **without re-deriving any physics or guessing any MORFE API**. All
formulas below are final; all API calls below were verified against the current source.

Read `implementation_plan.md` first for the high-level picture; this file supersedes it on
every detail where they differ.

---

## 0. Ground rules for the implementer

1. **Do not invent API.** Every MORFE call you need is shown in this document with its
   exact signature. If something is unclear, the canonical reference example for the forced
   case is `test/ParametrisationMethod/test_noconj_debug.jl` (a small forced second-order
   system — our setup is the same pattern at ORD = 3). The canonical mechanical example is
   `examples/01_clamped_beam_ferrite/main.jl`.
2. **Use dense `Matrix{Float64}` everywhere.** FOM ≈ 100–200, so the dense path
   (`DefaultEigensolver`, dense `lu!` bordered solves) is fast and avoids all
   Arpack/shift-invert complexity. Do not use sparse matrices in v1.
3. **Work nondimensionally.** All parameters in Phase 2 are already nondimensional.
   Never introduce SI values.
4. Implement the phases **in order**; each phase has an *Acceptance* block — do not proceed
   until it passes. Put each acceptance check into a small runnable test at the bottom of
   the corresponding file (guarded by `if abspath(PROGRAM_FILE) == @__FILE__` or a
   `run_checks()` function called from the demo).
5. Run everything with the example-local environment:
   `julia --project=examples/06_dielectric_elastomer_actuator <script>`.

### File layout to create

```
examples/06_dielectric_elastomer_actuator/
├── Project.toml                # Phase 1
├── parameters.jl               # Phase 2
├── hermite_beam.jl             # Phase 3   (FE assembly: M, K, D, b, g)
├── bias.jl                     # Phase 4   (static equilibrium Q₀, x₀; derived constants)
├── coupling_terms.jl           # Phase 5   (B₀…B₃ and all MultilinearMaps)
├── dea_demo.jl                 # Phases 6–8 (main driver: eigen → reduce → realify)
├── validation/
│   ├── fom_reference.jl        # Phase 9   (coupled (u,q) FOM integration)
│   └── compare_rom_fom.jl      # Phase 9
└── README.md                   # Phase 10
```

---

## 1. Mathematical model (fixed — do not modify)

### 1.1 Discrete starting equations

After FE discretisation (Phase 3), with `x ∈ ℝⁿ` the Hermite DOF vector (transverse
displacement + rotations), `Q(t) ∈ ℝ` the electrode charge, `V(t)` the voltage:

```
M ẍ + D ẋ + K x = b Q²                                  (mech)
R Q̇ + c₀ Q (1 − gᵀx) = V(t)                             (elec)
```

- `b ∈ ℝⁿ`: constant electrostatic load vector (uniform actuation moment, Phase 3.4).
- `g ∈ ℝⁿ`: strain-average functional; for a cantilever with mean-strain capacitance,
  `gᵀx = α_c · w_tip / L` exactly (Phase 3.5).
- All electrostatic constants are absorbed into `b`, `c₀`, `α_c`.

### 1.2 Bias point

Apply `V(t) = V₀ + v(t)`, `v(t) = v_a cos(Ω t)`. The static equilibrium `(x₀, Q₀)` solves

```
K x₀ = b Q₀²            ⇒  x₀ = Q₀² · K⁻¹ b =: Q₀² x_b
c₀ Q₀ (1 − gᵀx₀) = V₀   ⇒  scalar cubic:  c₀ Q₀ − c₀ (gᵀx_b) Q₀³ = V₀
```

So the bias reduces to **one scalar cubic equation for Q₀** (Phase 4). Define:

```
ĉ  := c₀ (1 − gᵀx₀)          # effective electrical stiffness  (must be > 0)
β  := 2 Q₀ b                  # linear electromechanical coupling vector
ℓ₀ := K b / (2 Q₀ bᵀb)        # charge-proxy functionals (see 1.4)
ℓ₁ := D b / (2 Q₀ bᵀb)
ℓ₂ := M b / (2 Q₀ bᵀb)
```

### 1.3 Shifted equations (exact)

With `x = x₀ + u`, `Q = Q₀ + q`:

```
M ü + D u̇ + K u = β q + b q²                            (mech′)
R q̇ + ĉ q = v + c₀Q₀ (gᵀu) + c₀ q (gᵀu)                 (elec′)
```

Both are exact (no truncation): (mech) is exactly quadratic in `Q`, (elec) exactly bilinear.

### 1.4 Exact charge-proxy identity

Project (mech′) onto `β/(βᵀβ)` and define the scalar **charge proxy**

```
σ(u, u̇, ü) := ℓ₀ᵀu + ℓ₁ᵀu̇ + ℓ₂ᵀü
```

Then, **exactly on trajectories** (using `βᵀb/βᵀβ = 1/(2Q₀)`):

```
σ = q + q²/(2Q₀)
```

Inverting the scalar series: `q = σ − σ²/(2Q₀) + σ³/(2Q₀²) + O(σ⁴)`,
hence `q² = σ² − σ³/Q₀ + O(σ⁴)`.

### 1.5 Third-order closure (exact to cubic order)

Form `R·d/dt(mech′) + ĉ·(mech′)`, replace `β(R q̇ + ĉ q)` using (elec′), replace
`R q̇` inside `d/dt(b q²) = 2 b q q̇` using (elec′), then substitute the series of 1.4 and
truncate at total degree 3 (counting `v` as degree 1). After cancellation
(the `σ²(gᵀu)` cubic contributions cancel identically), the result is:

```
B₃ u⁽³⁾ + B₂ ü + B₁ u̇ + B₀ u = F(u, u̇, ü, v)

B₃ = R M
B₂ = R D + ĉ M
B₁ = R K + ĉ D
B₀ = ĉ K − 2 c₀ Q₀² (b gᵀ)        # nonsymmetric rank-one softening update

F  =  2Q₀ b v                      # (f0) linear forcing
   +  2 b σ v                      # (f1) parametric forcing, degree 2
   +  4 c₀ Q₀ b σ γ                # (n2a) degree 2
   −  ĉ b σ²                       # (n2b) degree 2
   +  (ĉ/Q₀) b σ³                  # (n3)  degree 3
   −  (1/Q₀) b σ² v                # (f2)  degree 3 (with forcing)
```

where `γ := gᵀu` and `σ` as in 1.4. **Every nonlinear term is `b` times a scalar** — the
implementation is rank-one products only, no tensors. The truncation error is O(amplitude⁴).

### 1.6 Harmonic forcing representation

`v(t) = v_a cos(Ωt) = (v_a/2)(r₁ + r₂)` with external state
`ṙ₁ = iΩ r₁`, `ṙ₂ = −iΩ r₂`, initial condition `r(0) = (1, 1)`.
**Convention: fold `v_a/2` into every term coefficient and use `sum(r)` in `f!`** (this is
exactly what `test_noconj_debug.jl` does). So in all `f!` below, `v ↦ (v_a/2)·sum(r)`.

---

## 2. Phase 1 — Project setup

Create `Project.toml` with dependencies (add via `Pkg.develop` for MORFE like the other
examples do in their bootstrap block):

```
MORFE (develop, path = "../..")
LinearAlgebra, StaticArrays, Printf
OrdinaryDiffEq        # validation only
```

Copy the bootstrap block from `examples/01_clamped_beam_ferrite/main.jl`
(lines 15–21), replacing the package list with `["StaticArrays", "OrdinaryDiffEq"]`.

*Acceptance:* `julia --project=. -e 'using MORFE, OrdinaryDiffEq'` succeeds in the folder.

---

## 3. Phase 2 — `parameters.jl`

Nondimensional defaults. Define a single `NamedTuple` constructor:

```julia
function dea_parameters(;
    n_elem   = 50,        # FE elements → FOM = 2*n_elem free DOFs
    L        = 1.0,       # beam length            (nondimensional)
    EI       = 1.0,       # bending stiffness
    ρA       = 1.0,       # mass per length        → ω₁ ≈ 3.516
    ξ1       = 0.005,     # target damping ratio of mode 1 (sets η)
    ω1_target = 3.516015, # analytic cantilever ω₁ = (1.8751)²·√(EI/ρA L⁴)
    R        = 1.0,       # electrical resistance
    ωτ       = 1.0,       # ω₁ · (R/ĉ) ≈ ωτ  → sets c₀ ≈ ω1_target * R / 1
    α_c      = 1.0,       # capacitance strain sensitivity (gᵀx = α_c w_tip/L)
    m_b      = 1.0,       # actuation moment per unit Q² (scales b)
    V0       = 0.15,      # bias voltage  (keep ≪ pull-in, see Phase 4)
    v_a      = 0.02,      # harmonic amplitude
)
    η_over_E = 2ξ1 / ω1_target          # D = η_over_E * K  (Kelvin–Voigt)
    c0 = ω1_target * R / ωτ
    return (; n_elem, L, EI, ρA, η_over_E, R, c0, α_c, m_b, V0, v_a)
end
```

Notes: `ωτ = 1` puts the RC pole at the mechanical frequency — the regime that motivates a
third-order model. `Ω` (forcing frequency) is NOT a parameter here; it is chosen in
Phase 7 relative to the computed master eigenvalue.

*Acceptance:* file loads; `dea_parameters()` returns the tuple.

---

## 4. Phase 3 — `hermite_beam.jl` (FE assembly)

Cantilever Euler–Bernoulli beam, `n_elem` equal elements of length `h = L/n_elem`,
nodes `0 … n_elem`, 2 DOFs per node `(w, θ=w′)`. Global DOF numbering:
node `i` (0-based) → DOFs `2i+1` (w), `2i+2` (θ). Clamp node 0: delete DOFs 1, 2.
**Free DOF count `n = 2*n_elem`.** Keep an index map `free = 3:(2*n_elem+2)` (simply all
DOFs minus the first two — contiguous, so slicing suffices).

### 4.1 Element matrices (standard, copy verbatim)

```julia
Ke(EI, h) = EI/h^3 * [ 12    6h   -12    6h ;
                       6h   4h^2  -6h   2h^2;
                      -12   -6h    12   -6h ;
                       6h   2h^2  -6h   4h^2]

Me(ρA, h) = ρA*h/420 * [156   22h    54   -13h ;
                        22h  4h^2   13h  -3h^2;
                         54   13h   156   -22h ;
                       -13h -3h^2  -22h   4h^2]
```

Assemble `K_full`, `M_full` (size `(2n_elem+2)²`, dense), then restrict:
`K = K_full[free, free]`, `M = M_full[free, free]`. Damping: `D = η_over_E * K`.

### 4.2 Electrostatic load vector `b`

Uniform distributed actuation moment `m_b` per unit `Q²`. Consistent load:
`fe = m_b * ∫ N′ dx = m_b * [-1, 0, 1, 0]` per element (this telescopes — only boundary
contributions survive; that is physically correct for uniform unimorph actuation).
Assemble over all elements into `b_full`, restrict to `b = b_full[free]`.
(Result: `b` is zero except `−m_b` at the clamped-node w-DOF (removed) and `+m_b` at the
tip w-DOF. Keep the assembly loop anyway — it generalises to non-uniform actuation.)

> Check the sign/physics note: `∫ N₁′ = N₁(h)−N₁(0) = −1`, `∫ N₃′ = +1`, `∫ N₂′ = ∫ N₄′ = 0`.

### 4.3 Strain functional `g`

Mean bending strain ∝ mean slope: `⟨w′⟩ = (w(L) − w(0))/L = w_tip/L`. So:

```julia
g = zeros(n);  g[end-1] = α_c / L     # w-DOF of the tip node is the second-to-last free DOF
```

**Careful with DOF ordering:** with the numbering above, the last two free DOFs are
`(w_tip, θ_tip)`; `w_tip` is index `n-1`, `θ_tip` is index `n`. Verify with a static test
(4.A3) rather than trusting this sentence.

### 4.4 Exports

`assemble_beam(p) -> (; M, K, D, b, g, n, idx_wtip)` where `idx_wtip = n-1`.

### Acceptance (Phase 3)

- A1: `K`, `M` symmetric (`norm(K-K') < 1e-12`), positive definite (`isposdef(Symmetric(K))`).
- A2: first undamped frequency: `ω₁ = sqrt(eigmin-generalised)`; compute via
  `sqrt(minimum(real(eigvals(K, M))))` and require `|ω₁ − 3.516015| / 3.516015 < 1e-3`
  (Hermite beams converge fast; at `n_elem = 50` error ≈ 1e-7).
- A3: static tip-load test: `u = K \ f` with unit transverse tip force
  (`f[idx_wtip] = 1`) must give `u[idx_wtip] ≈ L³/(3EI)` to 1e-6 relative.
- A4: `gᵀ(K \ (b * 1.0)) > 0` (positive bias deflection convention; if negative, flip the
  sign of `m_b` — the bias analysis in Phase 4 assumes `gᵀx_b > 0`).

---

## 5. Phase 4 — `bias.jl` (static equilibrium and derived constants)

```julia
function bias_point(p, fe)            # fe = output of assemble_beam
    xb = fe.K \ fe.b                  # x_b
    a  = p.c0 * dot(fe.g, xb)         # cubic: c0*Q0 - a*Q0^3 = V0
    # Newton on  φ(Q0) = c0*Q0 - a*Q0^3 - V0,  start Q0 = V0/c0
    Q0 = p.V0 / p.c0
    for _ in 1:50
        φ  = p.c0*Q0 - a*Q0^3 - p.V0
        dφ = p.c0 - 3a*Q0^2
        Q0 -= φ/dφ
        abs(φ) < 1e-14 && break
    end
    x0   = Q0^2 .* xb
    ĉ    = p.c0 * (1 - dot(fe.g, x0))
    bᵀb  = dot(fe.b, fe.b)
    ℓ0   = (fe.K * fe.b) ./ (2Q0 * bᵀb)
    ℓ1   = (fe.D * fe.b) ./ (2Q0 * bᵀb)
    ℓ2   = (fe.M * fe.b) ./ (2Q0 * bᵀb)
    return (; Q0, x0, ĉ, ℓ0, ℓ1, ℓ2)
end
```

**Pull-in guard:** require `ĉ > 0.5 * p.c0` (i.e. bias strain `gᵀx₀ < 0.5`); if violated,
error with a message to lower `V0`. Also require `dφ > 0` at the converged `Q0` (stable
branch of the cubic).

### Acceptance (Phase 4)

- A1: residuals of the unshifted equilibrium `< 1e-12`:
  `norm(K*x0 - b*Q0^2)` and `abs(c0*Q0*(1 - g'x0) - V0)`.
- A2: exactness of the proxy identity on a random consistent point: pick random
  `(u, u̇)` small, pick random `q`, set `ü := M \ (β*q + b*q^2 - D*u̇ - K*u)`; then check
  `σ(u,u̇,ü) ≈ q + q²/(2Q0)` to 1e-12. (`β = 2Q0*b`.) **This test pins down most possible
  sign/ordering bugs in ℓ₀, ℓ₁, ℓ₂ — do not skip it.**
- A3: `gᵀx0 / 1.0` (bias strain) is in `[0.005, 0.2]`; tune `V0`/`m_b` defaults otherwise.

---

## 6. Phase 5 — `coupling_terms.jl` (linear matrices + all MultilinearMaps)

### 6.1 Linear matrices

```julia
B3 = p.R .* fe.M
B2 = p.R .* fe.D .+ bp.ĉ .* fe.M
B1 = p.R .* fe.K .+ bp.ĉ .* fe.D
B0 = bp.ĉ .* fe.K .- (2 * p.c0 * bp.Q0^2) .* (fe.b * fe.g')   # note: outer product, dense
```

All four must be `Matrix{Float64}` of size `n × n`.

### 6.2 MultilinearMap conventions (read carefully)

- `MultilinearMap(f!, multiindex::NTuple{3,Int})` for autonomous terms;
  `MultilinearMap(f!, multiindex, multiplicity_external::Int)` for forcing terms.
- `multiindex = (i₀, i₁, i₂)` = number of times `(u, u̇, ü)` appear as arguments.
- `f!` receives `(res, u-slots..., u̇-slots..., ü-slots..., r-slots...)` in that order and
  must **accumulate** (`res .+= …`), never overwrite.
- `f!` must have **exactly one method** with **exactly** `1 + deg` arguments after `res`
  (the constructor asserts this). Use one dedicated closure per term.
- The polynomial term represented is the *diagonal evaluation* `f!(x…x, ẋ…ẋ, ẍ…ẍ, r…)`.
  So multinomial factors from expanding `σ³ = (σ₀+σ₁+σ₂)³` **must be included inside `f!`**
  (they are included in the tables below — copy coefficients verbatim).
- When a multiindex entry is `> 1`, `f!` must be symmetric in those slots; the tables below
  are already symmetrised. Pass `fully_asymmetric = false` for those terms to silence the
  construction `@info`. Terms with all entries ≤ 1 need no keyword.
- Every term computes `res .+= (scalar) .* b`. Implement a helper:

```julia
addb!(res, s) = (res .+= s .* bvec; nothing)   # bvec = fe.b captured in closure scope
```

Below, `c2 := 4*c0*Q0`, `cs := ĉ`, `c3 := ĉ/Q0`, `cf1 := v_a` (from `2b·σ·(v_a/2)`),
`cf0 := Q0*v_a` (from `2Q0·b·(v_a/2)`), `cf2 := -v_a/(2Q0)`, and
`ℓ0, ℓ1, ℓ2, g` are the vectors from Phases 3–4. `S(r) = sum(r)` (r is the length-2
external state). `⟨a,b⟩ = dot(a,b)`.

### 6.3 Degree-2 autonomous terms (6 maps) — from `+ c2·b·σγ − cs·b·σ²`

| # | multiindex | `f!(res, args...)` accumulates |
|---|---|---|
| q1 | (2,0,0) | `( (c2/2)*(⟨ℓ0,a⟩⟨g,b⟩ + ⟨g,a⟩⟨ℓ0,b⟩) − cs*⟨ℓ0,a⟩⟨ℓ0,b⟩ ) .* bvec`, args `(a,b)`; `fully_asymmetric=false` |
| q2 | (1,1,0) | `( c2*⟨g,a⟩⟨ℓ1,y⟩ − 2cs*⟨ℓ0,a⟩⟨ℓ1,y⟩ ) .* bvec`, args `(a,y)` |
| q3 | (1,0,1) | `( c2*⟨g,a⟩⟨ℓ2,z⟩ − 2cs*⟨ℓ0,a⟩⟨ℓ2,z⟩ ) .* bvec`, args `(a,z)` |
| q4 | (0,2,0) | `( −cs*⟨ℓ1,y1⟩⟨ℓ1,y2⟩ ) .* bvec`; `fully_asymmetric=false` |
| q5 | (0,1,1) | `( −2cs*⟨ℓ1,y⟩⟨ℓ2,z⟩ ) .* bvec` |
| q6 | (0,0,2) | `( −cs*⟨ℓ2,z1⟩⟨ℓ2,z2⟩ ) .* bvec`; `fully_asymmetric=false` |

(`a` = u-slot, `y` = u̇-slot, `z` = ü-slot.)

### 6.4 Degree-3 autonomous terms (10 maps) — from `+ c3·b·σ³`

`σ³` expansion; coefficient = `c3 ×` multinomial:

| # | multiindex | scalar inside `f!` | sym keyword |
|---|---|---|---|
| c1 | (3,0,0) | `c3*⟨ℓ0,a⟩⟨ℓ0,b⟩⟨ℓ0,c⟩` | `fully_asymmetric=false` |
| c2t | (2,1,0) | `3c3*⟨ℓ0,a⟩⟨ℓ0,b⟩⟨ℓ1,y⟩` | `fully_asymmetric=false` |
| c3t | (2,0,1) | `3c3*⟨ℓ0,a⟩⟨ℓ0,b⟩⟨ℓ2,z⟩` | `fully_asymmetric=false` |
| c4 | (1,2,0) | `3c3*⟨ℓ0,a⟩⟨ℓ1,y1⟩⟨ℓ1,y2⟩` | `fully_asymmetric=false` |
| c5 | (1,1,1) | `6c3*⟨ℓ0,a⟩⟨ℓ1,y⟩⟨ℓ2,z⟩` | — |
| c6 | (1,0,2) | `3c3*⟨ℓ0,a⟩⟨ℓ2,z1⟩⟨ℓ2,z2⟩` | `fully_asymmetric=false` |
| c7 | (0,3,0) | `c3*⟨ℓ1,y1⟩⟨ℓ1,y2⟩⟨ℓ1,y3⟩` | `fully_asymmetric=false` |
| c8 | (0,2,1) | `3c3*⟨ℓ1,y1⟩⟨ℓ1,y2⟩⟨ℓ2,z⟩` | `fully_asymmetric=false` |
| c9 | (0,1,2) | `3c3*⟨ℓ1,y⟩⟨ℓ2,z1⟩⟨ℓ2,z2⟩` | `fully_asymmetric=false` |
| c10 | (0,0,3) | `c3*⟨ℓ2,z1⟩⟨ℓ2,z2⟩⟨ℓ2,z3⟩` | `fully_asymmetric=false` |

### 6.5 Forcing terms (10 maps) — only when the external system is enabled

Linear forcing (degree 1, me = 1):

| # | multiindex | me | `f!` |
|---|---|---|---|
| f0 | (0,0,0) | 1 | `(res, r) -> res .+= cf0*sum(r) .* bvec` |

Parametric forcing from `+2b·σ·v` (degree 2, me = 1):

| # | multiindex | me | scalar |
|---|---|---|---|
| f1a | (1,0,0) | 1 | `cf1*⟨ℓ0,a⟩*sum(r)` |
| f1b | (0,1,0) | 1 | `cf1*⟨ℓ1,y⟩*sum(r)` |
| f1c | (0,0,1) | 1 | `cf1*⟨ℓ2,z⟩*sum(r)` |

Cubic forcing correction from `−(1/Q₀)b·σ²·v` (degree 3, me = 1) — these six are **optional
exactness terms**; implement them, but put them behind a flag `include_cubic_forcing=true`:

| # | multiindex | me | scalar | sym |
|---|---|---|---|---|
| f2a | (2,0,0) | 1 | `cf2*⟨ℓ0,a⟩⟨ℓ0,b⟩*sum(r)` | `fully_asymmetric=false` |
| f2b | (1,1,0) | 1 | `2cf2*⟨ℓ0,a⟩⟨ℓ1,y⟩*sum(r)` | — |
| f2c | (1,0,1) | 1 | `2cf2*⟨ℓ0,a⟩⟨ℓ2,z⟩*sum(r)` | — |
| f2d | (0,2,0) | 1 | `cf2*⟨ℓ1,y1⟩⟨ℓ1,y2⟩*sum(r)` | `fully_asymmetric=false` |
| f2e | (0,1,1) | 1 | `2cf2*⟨ℓ1,y⟩⟨ℓ2,z⟩*sum(r)` | — |
| f2f | (0,0,2) | 1 | `cf2*⟨ℓ2,z1⟩⟨ℓ2,z2⟩*sum(r)` | `fully_asymmetric=false` |

### 6.6 Builder function

```julia
build_model(p, fe, bp; forced::Bool, Ω = 0.0) -> NDOrderModel
```

- `forced = false`: `NDOrderModel((B0,B1,B2,B3), (16 autonomous maps...))` — no external
  system. (NVAR = 2 path, Phase 7 milestone 1.)
- `forced = true`: `ext = ExternalSystem((im*Ω, -im*Ω))` and
  `NDOrderModel((B0,B1,B2,B3), (all 26 maps...), ext)`.
  External eigenvalue order **must** be `(+iΩ, −iΩ)` (matches `conjugate_permutation`
  in Phase 7).

### 6.7 Direct RHS evaluator (for validation)

Also export a plain function (no MORFE types) used by Phase 9:

```julia
F_thirdorder(u, u̇, ü, t) = …   # sum of all terms 6.3–6.5 evaluated literally,
                                # with v = v_a*cos(Ω t)
```

Build it by summing the same scalar formulas — do NOT reuse the MultilinearMaps (the point
is an independent re-implementation to catch transcription errors via Phase 6 A2).

### Acceptance (Phase 5)

- A1: `NDOrderModel` constructs without error in both modes (the `@info` about implicit
  symmetry must NOT appear — if it does, you forgot a `fully_asymmetric=false`).
- A2: **consistency of maps vs. literal formulas**: for 5 random states `(u,u̇,ü)` and
  `r=(1.0+0im, 1.0+0im)`, compare `evaluate_nonlinear_terms!(res, model, deg, (u,u̇,ü), r)`
  summed over `deg ∈ 1:3` against `F_thirdorder(u,u̇,ü,0)`; relative error `< 1e-12`.
- A3: **closure check against the coupled system** (catches derivation transcription
  errors): take small random `(u, u̇, q)` with magnitude `ε = 1e-4`; compute `ü` from
  (mech′) and `q̇` from (elec′) with `v = v_a`; compute `u⁽³⁾` two ways:
  (i) differentiate (mech′): `u⁽³⁾ = M \ (β*q̇ + 2b*q*q̇ − D*ü − K*u̇)`;
  (ii) from the third-order model: `u⁽³⁾ = B3 \ (F_thirdorder(u,u̇,ü,0) − B2*ü − B1*u̇ − B0*u)`.
  Require `norm(diff)/norm(u⁽³⁾) < C·ε³` — verify the **order**: halving ε must reduce the
  error by ≈ 8×.

---

## 7. Phase 6 — eigenanalysis (inside `dea_demo.jl`)

Use the built-in dense solver — **no custom eigensolver needed**:

```julia
ep = solve_eigenproblem(model)           # DefaultEigensolver: dense eigen(A,B), 3n pairs,
                                         # sorted |λ| ascending, left modes matched & normalized
(λs, Y, X) = get_eigenpairs(ep)          # Y, X :: (FOM, 3, 3n);  blocks of Y[:,:,k] = (φ, λφ, λ²φ)
```

Spectrum structure to expect: `n` underdamped mechanical-like complex pairs (near
`±i·ω_i` of the biased beam) + `n` mostly-real RC-relaxation eigenvalues near `−ĉ/R`.
Print the 10 smallest-|λ| eigenvalues.

**Master pair selection** (first bending pair):

```julia
i1 = findfirst(k -> imag(λs[k]) > 0.1, 1:length(λs))    # first oscillatory mode, +Im member
λ1 = λs[i1]
```

Find its conjugate partner index `i2` (`argmin(abs.(λs .- conj(λ1)))`). Then **enforce exact
conjugate symmetry by construction** (do not trust LAPACK pairing):

```julia
φ  = Y[:, 1, i1];  φd1 = Y[:, 2, i1];  φd2 = Y[:, 3, i1]
master_eigenvalues = SVector{2,ComplexF64}(λ1, conj(λ1))
master_modes       = hcat(φ, conj.(φ))                           # FOM × 2
master_modes_derivatives = cat(hcat(φd1, conj.(φd1)),            # FOM × 2 × 2:
                               hcat(φd2, conj.(φd2)); dims=3)    # WRONG SHAPE — see below
```

⚠ Shape: `master_modes_derivatives` must be `FOM × (ORD−1) × ROM = FOM × 2 × 2`, with
`[:, k, r] = k`-th derivative block of mode `r`. Build it explicitly:

```julia
mmd = zeros(ComplexF64, n, 2, 2)
mmd[:, 1, 1] .= φd1;  mmd[:, 2, 1] .= φd2
mmd[:, 1, 2] .= conj.(φd1);  mmd[:, 2, 2] .= conj.(φd2)
```

**Left eigenmodes**: the driver wants the `FOM × ROM` matrix of *pencil* left vectors
`ℓ_r` satisfying `ℓᵀ(λ³B₃+λ²B₂+λB₁+B₀) = 0`. That is the **last (third) block** of the
companion left eigenvector:

```julia
ℓ = X[:, 3, i1]                          # match i1 by eigenvalue, see note
left_eigenmodes = hcat(ℓ, conj.(ℓ))
```

(`X` is already sorted to match `λs` by `solve_eigenproblem`; overall scaling of `ℓ` is
irrelevant — the resonant projections are scale-invariant ratios.)

### Acceptance (Phase 6)

- A1: right pencil residual: `norm((λ1^3*B3 + λ1^2*B2 + λ1*B1 + B0)*φ) / norm(φ) < 1e-8`.
- A2: left pencil residual: `norm((λ1^3*B3' + λ1^2*B2' + λ1*B1' + B0')*ℓ) / norm(ℓ) < 1e-8`.
  *(If this fails, the left eigen convention differs by conjugation — try `conj.(X[:,3,i1])`
  and/or match against `conj(λ1)`; pick whichever satisfies A2 and document it.)*
- A3: derivative-block identity: `φd1 ≈ λ1*φ` and `φd2 ≈ λ1^2*φ` to 1e-8 relative.
- A4: stability: all 3n eigenvalues satisfy `real(λ) < 0`.
- A5: spectral gap report: print `|Re λ_RC,min|`, `|λ_2nd bending|` vs `|λ1|`; require the
  master pair to be the slowest oscillatory pair.

---

## 8. Phase 7 — reduction (inside `dea_demo.jl`)

### Milestone 1 — autonomous backbone (run this first)

```julia
model = build_model(p, fe, bp; forced = false)
ROM, N_EXT, NVAR = 2, 0, 2
max_degree = 7
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
res_set = resonance_set_from_complex_normal_form_style(
    mset, Vector{ComplexF64}(master_eigenvalues), 0.05)
W, R = solve_cohomological_problem(
    model, mset, master_eigenvalues, master_modes, left_eigenmodes, res_set;
    master_modes_derivatives = mmd, conjugate_permutation = [2, 1])
```

### Milestone 2 — forced

Pick `Ω = imag(λ1)` (exact primary resonance) for the nominal run:

```julia
model = build_model(p, fe, bp; forced = true, Ω = imag(λ1))
ROM, N_EXT, NVAR = 2, 2, 4
max_degree = 5                              # NVAR=4: L = binom(9,4)=126 monomials
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
ext_eigs = ComplexF64[im*Ω, -im*Ω]
res_set = resonance_set_from_complex_normal_form_style(
    mset, Vector{ComplexF64}(master_eigenvalues), 0.05;
    external_eigenvalues = ext_eigs)
W, R = solve_cohomological_problem(
    model, mset, master_eigenvalues, master_modes, left_eigenmodes, res_set;
    master_modes_derivatives = mmd, conjugate_permutation = [2, 1, 4, 3])
```

Use the **full** graded multiindex set — do not filter external degrees (unverified
territory; the cost is negligible at this size).

### Realification

```julia
conj_map = [2, 1]            # milestone 1;  [2,1,4,3] for milestone 2
Rr = ReducedDynamics(realify(extract_component(R.poly, 1), conj_map),
                     R.external_system_size)
```

Print nonzero reduced-dynamics coefficients exactly like example 01 (lines 247–253).

### Acceptance (Phase 7)

- A1: solve completes with no NaN/Inf in `W.poly.coefficients`, `R.poly.coefficients`.
- A2: linear part of `R` ≈ `diag(λ1, conj(λ1))` (plus `±iΩ` rows for the external block).
- A3: realified coefficients are real to 1e-10 (imag parts below tolerance).
- A4: backbone sanity (milestone 1): the coefficient of the resonant cubic monomial
  `z₁²z̄₁` in `R` is finite and changes by < 5% between `max_degree = 5` and `7`.

---

## 9. Phase 8 — ROM evaluation utilities (inside `dea_demo.jl` or a small `rom_utils.jl`)

Hand-roll polynomial evaluation (do not hunt for an API):

```julia
function eval_poly(coeffs::AbstractMatrix, mset, zvals)         # coeffs: (ncomp, L)
    acc = zeros(ComplexF64, size(coeffs, 1))
    for (l, α) in enumerate(mset.exponents)
        acc .+= coeffs[:, l] .* prod(zvals .^ α)
    end
    return acc
end
```

- Reduced ODE: `ż = R(z)` where `z = (z₁, z₂, r₁, r₂)`; integrate with OrdinaryDiffEq
  (complex state, `Tsit5()`), holding the external components: either include them in the
  state (their dynamics are exactly `±iΩ r`) or substitute `r(t) = (e^{iΩt}, e^{−iΩt})`
  analytically. Initial condition: `z₁ = z₂ = 0`, `r = (1, 1)`.
- Physical reconstruction: `u(t) = real(eval_poly(W.poly.coefficients[:, 1, :], mset, z(t)))`
  (block 1 of the `(FOM, ORD, L)` coefficient tensor = displacement map).
  Tip displacement = `u[idx_wtip]`; add back `x0[idx_wtip]` when comparing with the
  unshifted FOM.

---

## 10. Phase 9 — validation (`validation/`)

### 10.1 `fom_reference.jl` — coupled reference model

First-order state `s = (u, u̇, q) ∈ ℝ^{2n+1}`, integrating **(mech′) + (elec′)** literally
(this never went through the third-order closure — it is the ground truth):

```
u̇  = u̇
M ü = 2Q0*b*q + b*q^2 − D*u̇ − K*u
R q̇ = v_a*cos(Ωt) + c0*Q0*(gᵀu) + c0*q*(gᵀu) − ĉ*q
```

Use `Rodas5P()` or `Tsit5()` with `abstol=reltol=1e-10`. Provide
`integrate_fom(p, fe, bp; Ω, T_end, u0=zeros, ...) -> (t, u_tip(t))`.

### 10.2 `compare_rom_fom.jl` — three-way comparison

1. **Closure validation**: integrate the eliminated third-order system directly
   (state `(u, u̇, ü)`, using `B0…B3` and `F_thirdorder`) from consistent initial conditions
   (`ü(0)` from (mech′), small amplitude) and compare `u_tip(t)` against 10.1 over ~20
   forcing periods. Required: relative L2 mismatch `< 1e-3` at `v_a = 0.02`, decreasing
   ~8× when `v_a` is halved (cubic-order closure error).
2. **ROM validation**: integrate the reduced ODE (Phase 8), reconstruct `u_tip(t)`, compare
   the steady-state amplitude against 10.1 (discard the transient; compare the last 5
   periods). Required: amplitude error `< 2%` at nominal `v_a`, improving with
   `max_degree`.
3. **Mini-FRF**: repeat 2 for `Ω ∈ imag(λ1) .* (0.97:0.005:1.03)` — note each `Ω` needs a
   **new** `build_model`/reduction (the external eigenvalues change). Plot/print
   amplitude-vs-Ω for FOM (long-time integration at 5 of those points) vs ROM (all points).
   Required: ROM curve passes within 5% of the FOM points and bends (softening/hardening)
   consistently.

### 10.3 Convergence report

Table: steady tip amplitude at `Ω = imag(λ1)` for `max_degree ∈ (3, 5, 7)` and
`n_elem ∈ (25, 50, 100)`.

---

## 11. Phase 10 — `README.md`

Contents: one-paragraph physics summary; the exact third-order system of §1.5 (copy it);
how to run (`julia --project dea_demo.jl`, then `validation/compare_rom_fom.jl`); expected
output (eigenvalue table, reduced-dynamics coefficients, validation table); knobs
(`ωτ`, `V0`, `v_a`, `n_elem`, `max_degree`). Also move §1 of this document into
`derivation.md` essentially verbatim — it is the permanent record of the model.

---

## 12. Known pitfalls (read before debugging)

1. **`f!` arity assertion.** The `MultilinearMap` constructor counts the arguments of `f!`'s
   single method. Anonymous closures defined in a loop can accidentally share methods —
   define each `f!` on its own line.
2. **Accumulate, don't assign.** `res .+= …`. Assigning loses previously accumulated terms
   of the same degree.
3. **Argument order** is `(u-slots, u̇-slots, ü-slots, r-slots)` per `multiindex` then `me` —
   never alphabetical or "as in the formula".
4. **Multinomial factors live inside `f!`** (diagonal-evaluation convention). The tables in
   §6.3–6.5 already include them; do not add framework factors on top.
5. **`conjugate_permutation` validity** requires eigenvectors chosen exactly conjugately —
   that is why Phase 6 builds mode 2 as `conj.(mode 1)` instead of using the second LAPACK
   column.
6. **Left eigenvector block**: it is the *third* block `X[:, 3, k]` (pencil left vector),
   not the first. Wrong block ⇒ garbage resonant coefficients while everything else looks
   plausible. Phase 6 A2 catches this.
7. **B₀ is nonsymmetric** (rank-one `b gᵀ` update). Don't "fix" it with `Symmetric()`.
8. **Sign of `m_b` / direction conventions**: enforced by Phase 3 A4 and Phase 4 A2; if
   the bias Newton diverges, these are the first suspects.
9. **Per-Ω reduction**: the forced ROM is valid for the single `Ω` baked into the
   `ExternalSystem`. An FRF sweep loops the whole Phase 7 milestone-2 block.
10. **Don't introduce sparse matrices** until everything passes dense; the sparse path has
    different solver branches (KLU/Pardiso) and is a separate optimization task.

## 13. Definition of done

All phase acceptance checks pass at defaults (`n_elem=50`, `max_degree` 7/5,
`ξ1=0.005`, `ωτ=1`, `V0=0.15`, `v_a=0.02`); `dea_demo.jl` runs end-to-end in < 5 min;
`validation/compare_rom_fom.jl` prints the three-way comparison meeting the §10.2
tolerances; README and derivation.md written; `JuliaFormatter.format` applied with the
repo's `.JuliaFormatter.toml`.
