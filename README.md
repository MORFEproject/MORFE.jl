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

MORFE is FEM-backend-agnostic: it owns the DPIM solver and the abstract
`FEMMultilinearMap` interface. The Ferrite.jl backends — the St. Venant-Kirchhoff
"mesh → ROM" UI (`StructuralSVK`), the general parametric-structural engine
(`ParametricStructural`), and the incompressible-fluid backend
(`FluidNavierStokes`) — live in the companion package
[**MORFEFerrite.jl**](https://github.com/MORFEproject/MORFEFerrite.jl):

```julia
using Pkg
Pkg.add(url="https://github.com/MORFEproject/MORFEFerrite.jl.git")
```

The shortest path from mesh to ROM:

```julia
using MORFE, MORFEFerrite
SVK = MORFEFerrite.StructuralSVK

beam = SVK.mechanical_model("beam.msh";
    material  = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
    damping   = SVK.RayleighDamping(α = 5.4e-3, β = 1.9e-2),
    dirichlet = "Dirichlet")              # clamped facetset name

rom = SVK.parametrise(beam; master = [1], order = 7)   # autonomous (backbone)

# Near-resonant harmonic forcing, shaped like mode 1 at mode 1's frequency:
rom = SVK.parametrise(beam; master = [1], order = 7,
    forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.02))

SVK.print_equations(rom)                  # realified reduced dynamics
SVK.save_rom(rom, "results")
```

The low-level API (explicit `NDOrderModel`, eigensolvers, resonance sets,
`solve_cohomological_problem`) remains fully available — see
[`examples/01_clamped_beam_ferrite/low_level.jl`](https://github.com/MORFEproject/MORFEFerrite.jl/blob/main/examples/01_clamped_beam_ferrite/low_level.jl)
in MORFEFerrite for the same computation written out in full.

For detailed examples, see [`examples/`](examples/) here (Gridap beam,
dielectric actuator, internals) and
[MORFEFerrite's `examples/`](https://github.com/MORFEproject/MORFEFerrite.jl/tree/main/examples)
(Ferrite beam, COMSOL arch, parametric beam/arch, Kármán vortex street, MEMS
micromirror).

---

## Project Structure

```text
MORFE.jl/
├── src/
│   ├── MORFE.jl                      # Main package module
│   ├── Multiindices.jl               # Multiindex set utilities
│   ├── Polynomials.jl                # Dense polynomial representation
│   ├── Realification.jl              # Complex-to-real transformation
│   ├── FullOrderModel/               # FOM types and nonlinear maps (FEMMultilinearMap interface)
│   ├── SpectralDecomposition/        # Eigensolvers and mode propagation
│   └── ParametrisationMethod/        # DPIM core: resonance, invariance equation, ROM
├── examples/                         # Gridap beam, dielectric actuator, internals/, mesh_import/
├── benchmark/                        # Benchmark scripts
├── test/                             # Test suite
└── website/                          # Project website (morfeproject.github.io/MORFE.jl)
```

Ferrite.jl backends, the SVK/parametric/fluid UIs, and all Ferrite examples live in
[MORFEFerrite.jl](https://github.com/MORFEproject/MORFEFerrite.jl).

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

Full documentation is available at **[morfeproject.github.io/MORFE.jl](https://morfeproject.github.io/MORFE.jl)** —
tutorials, DPIM theory, and the API reference.

---

## Status

> **Pre-Alpha**: The API may change significantly between versions.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/MORFEproject/MORFE.jl).

---

## References

- Cabré, X., Fontich, E. & de la Llave, R. (2003). *The parameterization method for invariant manifolds I: Manifolds associated to non-resonant subspaces.* Indiana University Mathematics Journal 52(2), 283–328.
- Opreni, A. et al. (2023). *High-order direct parametrisation of invariant manifolds for model order reduction of finite element structures.* Nonlinear Dynamics.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
