# MORFE.jl — Model-Order Reduction for Finite Elements

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Alpha](https://img.shields.io/badge/Project_Status-Alpha-FF9900)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-9558B2.svg)](https://julialang.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-morfeproject.github.io-blue)](https://morfeproject.github.io/MORFE.jl/documentation.html)
[![Tests](https://github.com/MORFEproject/MORFE.jl/actions/workflows/tests.yml/badge.svg)](https://github.com/MORFEproject/MORFE.jl/actions/workflows/tests.yml)
[![Format](https://github.com/MORFEproject/MORFE.jl/actions/workflows/format.yml/badge.svg)](https://github.com/MORFEproject/MORFE.jl/actions/workflows/format.yml)
[![codecov](https://codecov.io/gh/MORFEproject/MORFE.jl/graph/badge.svg)](https://codecov.io/gh/MORFEproject/MORFE.jl)

MORFE.jl implements the **Direct Parametrisation of Invariant Manifolds (DPIM)** algorithm — a
spectral submanifold reduction technique that computes invariant manifolds of large finite-element
models in a single pass, collapsing million-DOF nonlinear oscillators into ROMs of usually **two to
four variables** that run in seconds while preserving the true backbone, internal resonances and
bifurcations.

$$\mathbf{B}_0 \mathbf{u} + \mathbf{B}_1 \dot{\mathbf{u}} + \mathbf{B}_2 \ddot{\mathbf{u}} + \cdots = \mathbf{F}(\mathbf{u}, \dot{\mathbf{u}}, \ldots, \mathbf{r}), \qquad \dot{\mathbf{r}} = \mathbf{E}(\mathbf{r})$$

$$\Downarrow \quad \text{DPIM, order } k \quad \Downarrow$$

$$\dot{\mathbf{z}} = \mathbf{R}(\mathbf{z}, \mathbf{r}), \qquad \mathbf{u} = \mathbf{W}(\mathbf{z}, \mathbf{r}), \qquad n = 2 \sim 4 \ll N$$

> **Alpha**: the API may still change between versions. The cohomological solver, eigenproblem
> pipeline and FEM backend interface are fully functional today.

---

## Installation

Install [Julia 1.10 or later](https://julialang.org/downloads/), then add MORFE from the Julia registry:

```julia
using Pkg
Pkg.add("MORFE")
using MORFE
```

To use the latest development version instead, install directly from GitHub:

```julia
Pkg.add(url = "https://github.com/MORFEproject/MORFE.jl.git")
```

---

## Usage

A complete reduction, using nothing but this package — two coupled Duffing oscillators reduced to a
fifth-order normal form on the first mode pair:

```julia
using MORFE

# M ü + C u̇ + K u + u³ = 0
K = [2.0 -1.0; -1.0 2.0]
M = [1.0 0.0; 0.0 1.0]
C = 0.001 * M
cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0);
    fully_asymmetric = false)                        # symmetric in its three arguments

model = NthOrderModel((K, C, M), (cubic,))           # linear terms as (B₀, B₁, B₂)

spec = spectrum(model)                               # generalised eigenproblem
sd = SpectralData(model, spec; master = master_by_sorting(2))
W, R = parametrise(model, sd, 5;                     # 5th-order expansion
    resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05))

R.poly.coefficients    # reduced dynamics  ż = R(z)
W.poly.coefficients    # parametrisation   u = W(z)
```

The same three calls drive a million-DOF finite-element model; only the construction of `model`
changes. The [tutorials](https://morfeproject.github.io/MORFE.jl/tutorials/) walk through external
forcing, internal resonances, parametric models and FEM backends.

---

## Finite-element backends

MORFE owns the DPIM solver and the abstract `FEMMultilinearMap` interface, so any FEM library can
supply the physics. The Ferrite.jl backends — the St. Venant-Kirchhoff "mesh → ROM" interface
(`StructuralSVK`), the parametric-structural engine (`ParametricStructural`) and the incompressible
fluid backend (`FluidNavierStokes`) — live in the optional companion package
[MORFEFerrite.jl](https://github.com/MORFEproject/MORFEFerrite.jl), installed directly from GitHub:

```julia
Pkg.add(url = "https://github.com/MORFEproject/MORFEFerrite.jl.git")
```

Its [clamped-beam notebook](https://github.com/MORFEproject/MORFEFerrite.jl/blob/main/examples/01_clamped_beam_ferrite/clamped_beam.ipynb)
is the shortest path from a mesh to a ROM.

---

## Documentation

| Page | Contents |
|------|----------|
| [Tutorials](https://morfeproject.github.io/MORFE.jl/tutorials/) | Full-order model building, multiindex sets, SVK mesh → ROM, Kármán vortex street, parametric models, MEMS micromirror |
| [Code documentation](https://morfeproject.github.io/MORFE.jl/documentation.html) | API reference and docstrings for every module |
| [Features](https://morfeproject.github.io/MORFE.jl/features.html) | How DPIM works, and why it differs from classical reduction |
| [Publications](https://morfeproject.github.io/MORFE.jl/publications.html) | Method papers and citation info |
| [Team](https://morfeproject.github.io/MORFE.jl/team.html) | Developers, contributors and institutions |

Runnable low-level demos live in [`examples/`](examples/README.md); the FEM-backed examples, which
carry their own meshes and environments, live in
[MORFEFerrite.jl/examples](https://github.com/MORFEproject/MORFEFerrite.jl/tree/main/examples).

---

## Contributing

Contributions are welcome — see the [contribution guide](CONTRIBUTING.md) for setup, quality checks,
and pull-request guidance. For questions, feature proposals, or research collaborations, please
[open an issue](https://github.com/MORFEproject/MORFE.jl/issues).

---

## References

- Cabré, X., Fontich, E. & de la Llave, R. (2003). *The parameterization method for invariant manifolds I: Manifolds associated to non-resonant subspaces.* Indiana University Mathematics Journal 52(2), 283–328.
- Opreni, A. et al. (2023). *High-order direct parametrisation of invariant manifolds for model order reduction of finite element structures.* Nonlinear Dynamics.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
