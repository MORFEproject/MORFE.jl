# MORFE.jl

![MORFE.jl logo](logo.svg)

**Model Order Reduction for Finite Elements in Julia**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Pre-Alpha](https://img.shields.io/badge/Project_Status-Pre--Alpha-FF9900)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-9558B2.svg)](https://julialang.org/downloads/)

MORFE.jl implements the **DPIM (Direct Parametrisation of Invariant Manifolds)** algorithm in Julia, offering a modular, FEM-agnostic approach to reduced-order modeling.

## Features

- **DPIM Implementation**: Direct Parametrisation of Invariant Manifolds for model order reduction
- **FEM-Agnostic Design**: Works with multiple FEM backends (Gridap.jl, Ferrite.jl, or custom solvers)
- **Polynomial Framework**: Built-in support for multiindices, polynomials, and realification
- **Julia Native**: Written in Julia with modern best practices (multiple dispatch, type stability)
- **Modular Architecture**: Easily extensible with custom solvers and visualization tools

## Installation

MORFE.jl is not yet registered in the Julia General Registry. To install:

```julia
using Pkg
Pkg.add(url="https://github.com/MORFEproject/MORFE.jl.git")
```

## Quick Start

```julia
using MORFE

# Load your Full Order Model (FOM)
fom = FullOrderModel()

# Use the reduced model for fast computations
# (more examples coming soon)
```

## Contents

```@contents
Pages = [
    "index.md",
    "project-overview.md",
    "multiindices.md",
    "polynomials.md",
    "realification.md",
    "multilinear_terms.md",
]
Depth = 2
```

## Project Structure

```
MORFE.jl/
├── src/
│   └── MORFE.jl          # Main package module
├── demo/                  # Example scripts
├── test/                  # Test suite
├── docs/                  # Documentation
└── Project.toml          # Julia project manifest
```

## Status

**Pre-Alpha**: This project is in early development. The API may change significantly.

### Implemented

- `FullOrderModel` module for representing FEM problems
- Polynomial framework (multiindices, basis functions)
- Realification utilities
- Lower-order coupling support

### Roadmap

See [Project Overview & Requirements](project-overview.md) for detailed plans.
