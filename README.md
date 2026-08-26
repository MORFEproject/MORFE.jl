# MORFE.jl — Model-Order Reduction for Finite Elements

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Pre-Alpha](https://img.shields.io/badge/Project_Status-Pre--Alpha-FF9900)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-9558B2.svg)](https://julialang.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-morfeproject.github.io-blue)](https://morfeproject.github.io/MORFE.jl/documentation.html)
[![Tests](https://github.com/MORFEproject/MORFE.jl/actions/workflows/tests.yml/badge.svg)](https://github.com/MORFEproject/MORFE.jl/actions/workflows/tests.yml)
[![Format](https://github.com/MORFEproject/MORFE.jl/actions/workflows/format.yml/badge.svg)](https://github.com/MORFEproject/MORFE.jl/actions/workflows/format.yml)

MORFE.jl implements the **Direct Parametrisation of Invariant Manifolds (DPIM)** algorithm — a
spectral submanifold reduction technique that computes invariant manifolds of large finite-element
models in a single pass, collapsing million-DOF nonlinear oscillators into ROMs of usually **two to
four variables** that run in microseconds while preserving the true backbone, internal resonances and
bifurcations.

$$\mathbf{B}_0\,\mathbf{u} + \mathbf{B}_1\,\dot{\mathbf{u}} + \mathbf{B}_2\,\ddot{\mathbf{u}} + \cdots = \mathbf{F}(\mathbf{u}, \dot{\mathbf{u}}, \ldots, \mathbf{r}), \qquad \dot{\mathbf{r}} = \mathbf{E}(\mathbf{r})$$

$$\Downarrow \quad \text{DPIM, order } k \quad \Downarrow$$

$$\dot{\mathbf{z}} = \mathbf{R}(\mathbf{z}, \mathbf{r}), \qquad \mathbf{u} = \mathbf{W}(\mathbf{z}, \mathbf{r}), \qquad n = 2 \sim 4 \ll N$$

---

## Documentation & Website

The full project website lives at **[morfeproject.github.io/MORFE.jl](https://morfeproject.github.io/MORFE.jl)**:

| Page | Contents |
|------|----------|
| [Tutorials](https://morfeproject.github.io/MORFE.jl/tutorials/) | Step-by-step guides: full-order model building, multiindex sets, SVK mesh → ROM, Kármán vortex street, parametric models, MEMS micromirror |
| [Code documentation](https://morfeproject.github.io/MORFE.jl/documentation.html) | API reference and docstrings for every module |
| [Features](https://morfeproject.github.io/MORFE.jl/features.html) | How DPIM works, and why it differs from classical reduction |
| [Gallery](https://morfeproject.github.io/MORFE.jl/gallery.html) | Application showcase — MEMS, beams, arches, shells |
| [Publications](https://morfeproject.github.io/MORFE.jl/publications.html) | Method papers and citation info |
| [Team](https://morfeproject.github.io/MORFE.jl/team.html) | Developers, contributors and institutions |

---

## Features

| Capability | What it means |
|------------|---------------|
| **DPIM implementation** | Direct Parametrisation of Invariant Manifolds for nonlinear model order reduction |
| **N-th order ODEs** | Native support for second-order (and higher-order) mechanical systems — no manual conversion to first-order form |
| **External forcing** | Polynomial external forcing systems handled at the level of the invariance equation |
| **Resonance handling** | Graph-style, complex/real normal form, and condition-number-based resonance detection |
| **Polynomial framework** | Built-in multiindex sets, dense polynomials, and realification utilities |
| **FEM-agnostic** | Works with Gridap.jl, Ferrite.jl, or any custom FEM backend via the `FEMMultilinearMap` interface |
| **Julia-native** | Multiple dispatch and static type parameters (`SVector`, compile-time dimensions) for performance |

---

## How it works

A DPIM computation is a single pipeline from FE model to reduced dynamics:

```text
NthOrderModel
    │
    ▼
linear_first_order_matrices ──► generalised_eigenpairs
                                         │
                                         ▼
                                  select master modes
                                         │
                                         ▼
                                  ResonanceSet construction
                                         │
                                         ▼
                                  solve_cohomological_problem
                                         │
                                         ▼
                     (Parametrisation W, ReducedDynamics R)
```

The cohomological equations are solved order-by-order on a GrLex-ordered multiindex set, producing a
symbolic, executable parametrisation `W` (full-state map) and reduced dynamics `R` on a
low-dimensional invariant manifold tangent to the chosen eigenmodes.

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

The low-level API (explicit `NthOrderModel`, eigensolvers, resonance sets,
`solve_cohomological_problem`) remains fully available — see
[`examples/01_clamped_beam_ferrite/low_level.jl`](https://github.com/MORFEproject/MORFEFerrite.jl/blob/main/examples/01_clamped_beam_ferrite/low_level.jl)
in MORFEFerrite for the same computation written out in full.

---

## Examples

Self-contained, runnable examples demonstrate the full pipeline. Each manages its own Julia
environment and writes outputs under a git-ignored `results/` folder — run them from the repository
root with:

```bash
julia --project=examples/internals -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/internals/multiindex_sets/main.jl")'
```

See [`examples/README.md`](examples/README.md) for the example contract and validation workflow.

### In this repository

| Folder | Model | Demonstrates |
|--------|-------|--------------|
| [`mesh_import/`](examples/mesh_import/) | Test meshes | Abaqus/COMSOL → GMSH format conversion |
| [`internals/`](examples/internals/) | Synthetic models | Low-level API: polynomials, multiindices, parametrisation method |

### Developed outside this repository

Every example with its own FEM stack lives outside the package tree, so that a heavy backend
(and its meshes and result archives) never becomes a dependency of the library:

- **Ferrite-backed** — the companion package
  [MORFEFerrite.jl/examples](https://github.com/MORFEproject/MORFEFerrite.jl/tree/main/examples).
- **Gridap-backed clamped beam** and the **dielectric elastomer actuator** — maintained
  alongside this repository rather than in it.

| Folder | Model |
|--------|-------|
| `01_clamped_beam_ferrite/` | Clamped-clamped SVK beam — high-level `StructuralSVK` UI + low-level pipeline |
| `03_arch_comsol_wedge/` | Arch wedge, COMSOL `.mphtxt` P18 mesh |
| `04_parametric_clamped_beam/` | Two-parameter ROM (axial stretch θ₁ + bending-mode arch θ₂), general `ParametricStructural` engine |
| `05_karman_vortex_street/` | Cylinder wake flow (Kármán), `FluidNavierStokes` backend |
| `07_parametric_arch/` | Single-parameter sinusoidal arch, `ParametricStructural` |
| `08_mems_micromirror/` | MEMS scanning micromirror, `StructuralSVK` |

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
| `FullOrderModel` | `NthOrderModel` and `FirstOrderModel` with multilinear nonlinear terms |
| `Eigensolvers` | ARPACK-based generalised eigensolver with shift-and-invert |
| `EigenModesPropagation` | Left/right eigenvector and Jordan vector propagation for N-th order systems |
| `Resonance` | Resonance set construction (graph, normal form, condition-number strategies) |
| `InvarianceEquation` | Cohomological system assembly via fused Horner passes |
| `MasterModeOrthogonality` | Orthogonality condition assembly for resonant master modes |
| `ParametrisationMethod` | Core `Parametrisation` and `ReducedDynamics` types |
| `MultilinearTerms` | Nonlinear right-hand side contributions with caching |
| `LowerOrderCouplings` | Lower-order coupling vectors for the cohomological equation |

---

## Status

> **Pre-Alpha**: The API may change significantly between versions. The cohomological solver,
> eigenproblem pipeline, and FEM backend interface are fully functional today.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on
[GitHub](https://github.com/MORFEproject/MORFE.jl). See the
[team page](https://morfeproject.github.io/MORFE.jl/team.html) for the contribution guide and
current contributors.

---

## References

- Cabré, X., Fontich, E. & de la Llave, R. (2003). *The parameterization method for invariant manifolds I: Manifolds associated to non-resonant subspaces.* Indiana University Mathematics Journal 52(2), 283–328.
- Opreni, A. et al. (2023). *High-order direct parametrisation of invariant manifolds for model order reduction of finite element structures.* Nonlinear Dynamics.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
