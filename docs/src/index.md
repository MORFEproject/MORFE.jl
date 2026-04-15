# MORFE.jl

![MORFE.jl logo](logo.svg)

**Model Order Reduction for Finite Elements in Julia**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Pre-Alpha](https://img.shields.io/badge/Project_Status-Pre--Alpha-FF9900)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-9558B2.svg)](https://julialang.org/downloads/)

MORFE.jl implements the **DPIM (Direct Parametrisation of Invariant Manifolds)** algorithm in Julia —
a spectral submanifold reduction technique for high-dimensional nonlinear dynamical systems arising
from finite element models.

## Features

- **DPIM Implementation**: Direct Parametrisation of Invariant Manifolds for nonlinear model order reduction
- **N-th order ODEs**: Native support for second-order (and higher-order) mechanical systems
- **External forcing**: Polynomial external forcing handled at the level of the invariance equation
- **Resonance handling**: Graph-style, complex/real normal form, and condition-number–based strategies
- **Polynomial framework**: Multiindex sets, dense polynomials, and realification utilities
- **FEM-agnostic design**: Works with Gridap.jl, Ferrite.jl, or any custom FEM backend
- **Julia native**: Multiple dispatch and static type parameters for performance

## Installation

MORFE.jl is not yet registered in the Julia General Registry. Install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/MORFEproject/MORFE.jl.git")
```

## Quick Start

```julia
using MORFE

# Define a second-order full-order model: M ẍ + C ẋ + K x = f(x, ẋ)
model = NDOrderModel((K, C, M), nonlinear_terms)

# Extract the equivalent first-order matrices for the eigensolver
A, B = linear_first_order_matrices(model)
result = generalised_eigenpairs(A, B; nev = 4, sigma = 0.0)

# Build the resonance set and solve the invariance equation
# (see demo/ in the repository for complete worked examples)
```

## Contents

```@contents
Pages = [
    "project-overview.md",
    "multiindices.md",
    "polynomials.md",
    "realification.md",
    "multilinear_terms.md",
    "api.md",
]
Depth = 2
```

## Status

> **Pre-Alpha**: The API may change significantly between versions.

### Implemented Modules

| Module | Description |
| --- | --- |
| `Multiindices` | Multiindex sets with graded lex ordering and factorisation utilities |
| `Polynomials` | Dense multivariate polynomials aligned to multiindex sets |
| `Realification` | Change of variables from complex (z, z̄) to real (x, y) coordinates |
| `FullOrderModel` | `NDOrderModel` and `FirstOrderModel` with multilinear nonlinear terms and external forcing |
| `Eigensolvers` | ARPACK-based generalised eigensolver with shift-and-invert |
| `EigenModesPropagation` | Left/right eigenvector and Jordan vector propagation for N-th order systems |
| `Resonance` | Resonance set construction (graph, normal form, condition-number strategies) |
| `InvarianceEquation` | Cohomological system assembly via fused Horner passes |
| `MasterModeOrthogonality` | Orthogonality condition assembly for resonant master modes |
| `ParametrisationMethod` | Core `Parametrisation` and `ReducedDynamics` types |
| `MultilinearTerms` | Nonlinear right-hand side contributions with caching |
| `LowerOrderCouplings` | Lower-order coupling vectors for the cohomological equation |

### Roadmap

See [Project Overview](project-overview.md) for detailed plans and design decisions.
