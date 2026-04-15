# MORFE.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Pre-Alpha](https://img.shields.io/badge/Project_Status-Pre--Alpha-FF9900)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-9558B2.svg)](https://julialang.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-stable-blue)](https://morfeproject.github.io/MORFE.jl)

MORFE.jl implements the **Direct Parametrisation of Invariant Manifolds (DPIM)** algorithm — a spectral submanifold reduction technique for high-dimensional nonlinear dynamical systems arising from finite element models.

---

## Features

- **DPIM implementation** — Direct Parametrisation of Invariant Manifolds for nonlinear model order reduction
- **N-th order ODEs** — native support for second-order (and higher-order) mechanical systems, no manual conversion to first-order form required
- **External forcing** — polynomial external forcing systems handled at the level of the invariance equation
- **Resonance handling** — graph-style, complex/real normal form, and condition-number–based resonance detection
- **Polynomial framework** — built-in multiindex sets, dense polynomials, and realification utilities
- **FEM-agnostic** — works with Gridap.jl, Ferrite.jl, or any custom FEM backend
- **Julia-native** — multiple dispatch and static type parameters for performance

---

## Installation

MORFE.jl is not yet registered in the Julia General Registry. Install directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/MORFEproject/MORFE.jl.git")
```

Or in Pkg REPL mode (`]`):

```julia-repl
add https://github.com/MORFEproject/MORFE.jl.git
```

---

## Quick Start

```julia
using MORFE

# Define a second-order full-order model: M ẍ + C ẋ + K x = f(x, ẋ)
model = NDOrderModel((K, C, M), nonlinear_terms)

# Extract first-order matrices and compute eigenpairs
A, B = linear_first_order_matrices(model)
result = generalised_eigenpairs(A, B; nev = 4, sigma = 0.0)

# Build the resonance set and solve the parametrisation
# See demo/ for complete worked examples
```

For detailed examples, see the [`demo/`](demo/) directory.

---

## Project Structure

```text
MORFE.jl/
├── src/
│   ├── MORFE.jl                      # Main package module
│   ├── Multiindices.jl               # Multiindex set utilities
│   ├── Polynomials.jl                # Dense polynomial representation
│   ├── Realification.jl              # Complex-to-real transformation
│   ├── FullOrderModel/               # FOM types and nonlinear maps
│   ├── SpectralDecomposition/        # Eigensolvers and mode propagation
│   └── ParametrisationMethod/        # DPIM core: resonance, invariance equation, ROM
├── demo/                             # Worked examples
├── test/                             # Test suite
└── docs/                             # Documentation source
```

---

## Modules

| Module | Description |
|--------|-------------|
| `Multiindices` | Multiindex sets with graded lex ordering and factorisation utilities |
| `Polynomials` | Dense multivariate polynomials aligned to multiindex sets |
| `Realification` | Change of variables from complex (z, z̄) to real (x, y) coordinates |
| `FullOrderModel` | `NDOrderModel` and `FirstOrderModel` with multilinear nonlinear terms |
| `Eigensolvers` | ARPACK-based generalised eigensolver with shift-and-invert |
| `EigenModesPropagation` | Left/right eigenvector and Jordan vector propagation for N-th order systems |
| `Resonance` | Resonance set construction (graph, normal form, condition-number strategies) |
| `InvarianceEquation` | Cohomological system assembly via fused Horner passes |
| `MasterModeOrthogonality` | Orthogonality condition assembly for resonant master modes |
| `ParametrisationMethod` | Core `Parametrisation` and `ReducedDynamics` types |
| `MultilinearTerms` | Nonlinear right-hand side contributions with caching |
| `LowerOrderCouplings` | Lower-order coupling vectors for the cohomological equation |

---

## Documentation

Full documentation is available at **[morfeproject.github.io/MORFE.jl](https://morfeproject.github.io/MORFE.jl)**.

- [Project Overview & Roadmap](docs/src/project-overview.md)
- [Module Structure & Dependencies](folder_structure_and_dependencies.md)

---

## Status

> **Pre-Alpha**: The API may change significantly between versions.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/MORFEproject/MORFE.jl).

---

## References

- Haller, G. & Ponsioen, S. (2016). *Nonlinear normal modes and spectral submanifolds*. Nonlinear Dynamics.
- Opreni, A. et al. (2023). *High-order direct parametrisation of invariant manifolds for model order reduction of finite element structures.* Nonlinear Dynamics.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
