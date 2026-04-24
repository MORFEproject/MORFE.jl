# MORFE.jl

![MORFE.jl logo](assets/logo.svg)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Pre-Alpha](https://img.shields.io/badge/Project_Status-Pre--Alpha-FF9900)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-9558B2.svg)](https://julialang.org/downloads/)

MORFE.jl implements the **Direct Parametrisation of Invariant Manifolds (DPIM)** — a
high-order, spectral-submanifold-based method for nonlinear model order reduction of large finite
element models.

---

## Why MORFE.jl?

Large finite-element models of nonlinear structures routinely reach $n \sim 10^5$ degrees of
freedom.  Direct time-integration or frequency-response continuation at this scale is prohibitively
expensive, while classical linear modal truncation misses essential nonlinear phenomena (hardening,
softening, internal resonances).

MORFE.jl resolves this by computing a **polynomial parametrisation** of a low-dimensional
*Spectral Submanifold* (SSM) that:

- **captures exact nonlinear dynamics** on the manifold — not just a linear approximation;
- reduces the system to $m \sim 2$–$10$ coordinates regardless of FE model size;
- handles **harmonic and quasi-periodic external forcing** via autonomous external variables;
- is **FEM-agnostic** — works with any backend that provides $K$, $C$, $M$ and nonlinear force vectors.

!!! tip "Read the theory"
    See [DPIM Theory](theory/dpim.md) for a self-contained derivation of the invariance equation,
    cohomological equations, and the DPIM algorithm.

!!! tip "Run a demo interactively"
    Launch the Duffing oscillator demo as a live Pluto notebook — no local installation needed:

    [![Open in Pluto on Binder](https://img.shields.io/badge/launch-Pluto%20notebook-4063D8?logo=julia&logoColor=white)](https://binder.plutojl.org/v2/gh/MORFEproject/MORFE.jl/main?path=notebooks%2Fduffing_demo.jl)

---

## Installation

MORFE.jl is not yet registered in the Julia General Registry.  Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/MORFEproject/MORFE.jl.git")
```

---

## Quick Start

```julia
using MORFE
using LinearAlgebra, StaticArrays

# 1. Define system matrices  M ẍ + C ẋ + K x = F_nl(x, ẋ)
B0 = [2.0 -1.0; -1.0 2.0]   # stiffness
B1 = [0.01  0.0;  0.0 0.01] # damping
B2 = [1.0   0.0;  0.0 1.0]  # mass

# 2. Nonlinear term: cubic stiffness  −β x³
using MORFE.FullOrderModel: MultilinearMap, NDOrderModel
term_cubic = MultilinearMap(
    (res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3),
    (3, 0),
)
model = NDOrderModel((B0, B1, B2), (term_cubic,))

# 3. Solve the generalised eigenproblem
using MORFE.FullOrderModel: linear_first_order_matrices
A, B = linear_first_order_matrices(model)
result = eigen(A, B)

# 4. Select master modes and build multiindex set
using MORFE.Multiindices: all_multiindices_up_to
ROM = 2;  NVAR = ROM;  p = 5
mset = all_multiindices_up_to(NVAR, p; min_degree = 1)

# 5. Build resonance set and solve the parametrisation
using MORFE.Resonance: resonance_set_from_graph_style
using MORFE.CohomologicalEquations: solve_cohomological_problem
# (see demo/ or notebooks/ for the complete pipeline)
```

---

## How It Works

```text
         Full-order FEM model (n ~ 10⁵ DOF)
                     │
             Spectral decomposition
          (generalised eigenproblem)
                     │
            Select m master modes
        (least-damped conjugate pairs)
                     │
         Build polynomial multiindex set
            𝒜 ⊂ ℕᵐ⁺ᴺᵉˣᵗ, degree ≤ p
                     │
         ┌───────────────────────────┐
         │  Cohomological solve loop │
         │  for each α ∈ 𝒜 (GrLex)  │
         │    s = ⟨λ, α⟩            │
         │    [A−sB]W[α] = N[α]     │  ← invariance equation
         │    (augmented at resonance)│
         └───────────────────────────┘
                     │
        Parametrisation W : ℂᵐ → ℂⁿ  (SSM embedding)
        Reduced dynamics R : ℂᵐ → ℂᵐ (flow on SSM)
                     │
         Post-process: FRC, backbone,
         time histories, realification
```

---

## Features

- **DPIM implementation** — Direct Parametrisation of Invariant Manifolds (Cabré, de la Llave &
  Jorba 2003; Opreni, Vizzaccaro, Touzé & Frangi 2021)
- **N-th order ODEs** — native support for second-order (and higher) structural ODE systems
- **External forcing** — polynomial external forcing handled autonomously in the invariance equation
- **Resonance handling** — graph-style, complex/real normal form, and condition-number strategies
- **Polynomial framework** — multiindex sets, dense polynomials, and realification utilities
- **FEM-agnostic** — interfaces with Gridap.jl, Ferrite.jl, or any custom FEM backend
- **Julia-native** — multiple dispatch and static type parameters for performance

---

## Module Overview

| Module | Description |
|--------|-------------|
| `Multiindices` | Multiindex sets with graded lex ordering and factorisation utilities |
| `Polynomials` | Dense multivariate polynomials aligned to multiindex sets |
| `Realification` | Change of variables from complex $(z, \bar z)$ to real $(a, b)$ coordinates |
| `FullOrderModel` | `NDOrderModel` and `FirstOrderModel` with multilinear nonlinear terms |
| `Eigensolvers` | ARPACK-based generalised eigensolver with shift-and-invert |
| `PropagateEigenmodes` | Left/right eigenvector and Jordan vector propagation |
| `Resonance` | Resonance set construction (graph, normal form, condition-number) |
| `InvarianceEquation` | Cohomological system assembly via fused Horner passes |
| `MasterModeOrthogonality` | Orthogonality condition assembly for resonant master modes |
| `ParametrisationMethod` | Core `Parametrisation` and `ReducedDynamics` types |
| `MultilinearTerms` | Nonlinear right-hand side contributions with symmetry-aware caching |
| `LowerOrderCouplings` | Lower-order coupling vectors for the cohomological equation |

---

## Status

> **Pre-Alpha**: The API may change significantly between versions.

See [Project Overview](project-overview.md) for design decisions, roadmap, and planned features.
