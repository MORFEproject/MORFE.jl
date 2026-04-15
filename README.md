# MORFE.jl

![MORFE.jl logo](docs/logo.svg)

`Model Order Reduction for Finite Elements in Julia`

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Pre-Alpha](https://img.shields.io/badge/Project_Status-Pre--Alpha-FF9900)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-9558B2.svg)](https://julialang.org/downloads/)
[![CI Status](https://img.shields.io/badge/CI-Pending-orange)](https://github.com/MORFEproject/MORFE.jl/actions)

MORFE.jl implements the **DPIM (Direct Parametrisation of Invariant Manifolds)** algorithm in Julia — a spectral submanifold reduction technique for high-dimensional nonlinear dynamical systems arising from finite element models.

---

## Features

- **DPIM Implementation**: Direct Parametrisation of Invariant Manifolds for nonlinear model order reduction
- **N-th order ODEs**: Native support for second-order (and higher-order) mechanical systems — no need to manually convert to first-order form
- **External forcing**: Polynomial external forcing systems are handled at the level of the invariance equation
- **Resonance handling**: Multiple strategies — graph-style, complex/real normal form, and condition-number–based resonance detection
- **Polynomial framework**: Built-in multiindex sets, dense polynomials, and realification utilities
- **FEM-agnostic design**: Works with any FEM backend (Gridap.jl, Ferrite.jl, or custom solvers)
- **Julia native**: Written in Julia with multiple dispatch and static type parameters for performance

---

## Installation

MORFE.jl is not yet registered in the Julia General Registry. To install:

```julia
using Pkg
Pkg.add(url="https://github.com/MORFEproject/MORFE.jl.git")
```

Or in Pkg mode:

```julia
] add https://github.com/MORFEproject/MORFE.jl.git
```

---

## Quick Start

```julia
using MORFE

# Define a second-order full-order model: M ẍ + C ẋ + K x = f(x, ẋ)
model = NDOrderModel((K, C, M), nonlinear_terms)

# Compute the linear eigenpairs
A, B = linear_first_order_matrices(model)
result = generalised_eigenpairs(A, B; nev=4, sigma=0.0)

# Build the resonance set and solve the parametrisation
# (see demo/ for complete worked examples)
```

For detailed examples, see the [`demo/`](demo/) directory.

---

## Project Structure

```text
MORFE.jl/
├── src/
│   ├── MORFE.jl                        # Main package module
│   ├── Multiindices.jl                 # Multiindex set utilities
│   ├── Polynomials.jl                  # Dense polynomial representation
│   ├── Realification.jl                # Complex-to-real transformation
│   ├── FullOrderModel/                 # FOM types and nonlinear maps
│   ├── SpectralDecomposition/          # Eigensolvers and mode propagation
│   └── ParametrisationMethod/          # DPIM core: resonance, invariance equation, ROM
├── demo/                               # Worked examples
├── test/                               # Test suite
├── docs/                               # Documentation
└── Project.toml                        # Julia project manifest
```

See [folder_structure_and_dependencies.md](folder_structure_and_dependencies.md) for a full module-level breakdown with dependency graph.

---

## Implemented Modules

| Module | Description |
| --- | --- |
| `Multiindices` | Multiindex sets with graded lex ordering and factorisation utilities |
| `Polynomials` | Dense multivariate polynomials aligned to multiindex sets |
| `Realification` | Change of variables from complex (z, z̄) to real (x, y) coordinates |
| `FullOrderModel` | `NDOrderModel` and `FirstOrderModel` with multilinear nonlinear terms and external forcing |
| `Eigensolvers` | ARPACK-based generalised eigensolver with shift-and-invert |
| `EigenModesPropagation` | Left/right eigenvector and Jordan vector propagation for N-th order systems |
| `JordanChain` | Jordan chain computation for defective eigenvalue problems |
| `Resonance` | Resonance set construction (graph, normal form, condition-number strategies) |
| `InvarianceEquation` | Cohomological system assembly via fused Horner passes |
| `MasterModeOrthogonality` | Orthogonality condition assembly for resonant master modes |
| `ParametrisationMethod` | Core `Parametrisation` and `ReducedDynamics` types |
| `MultilinearTerms` | Nonlinear right-hand side contributions with caching |
| `LowerOrderCouplings` | Lower-order coupling vectors for the cohomological equation |

---

## Status

> **Pre-Alpha**: The API may change significantly between versions.

### Roadmap

See [Project Overview & Requirements](docs/src/project-overview.md) for detailed plans.

---

## Documentation

- [Project Overview & Requirements](docs/src/project-overview.md) — design decisions and roadmap
- [Module Structure & Dependencies](folder_structure_and_dependencies.md) — full dependency graph

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/MORFEproject/MORFE.jl).

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## References

- Haller, G. & Ponsioen, S. (2016). *Nonlinear normal modes and spectral submanifolds*. Nonlinear Dynamics.
- Opreni, A. et al. (2023). *High-order direct parametrisation of invariant manifolds for model order reduction of finite element structures.* Nonlinear Dynamics.
- MORFE2.0 — previous version of the framework (Julia-based)
