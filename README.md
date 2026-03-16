# MORFE.jl

```
░▒▓██████████████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓████████▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓██████▓▒░ ░▒▓██████▓▒░   
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓████████▓▒░ 
```
 Model Order Reduction for Finite Elements in Julia

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Status: Pre-Alpha](https://img.shields.io/badge/Project_Status-Pre--Alpha-FF9900)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-9558B2.svg)](https://julialang.org/downloads/)
[![CI Status](https://img.shields.io/badge/CI-Pending-orange)](https://github.com/MORFEproject/MORFE3.0/actions)

MORFE.jl is the next generation of the **Model Order Reduction for Finite Elements** framework. It implements the **DPIM (Direct Parametrisation of Invariant Manifolds)** algorithm in Julia, offering a modular, FEM-agnostic approach to reduced-order modeling.

---

## Features

- **DPIM Implementation**: Direct Parametrisation of Invariant Manifolds for model order reduction
- **FEM-Agnostic Design**: Works with multiple FEM backends (Gridap.jl, Ferrite.jl, or custom solvers)
- **Polynomial Framework**: Built-in support for multiindices, polynomials, and realification
- **Julia Native**: Written in Julia with modern best practices (multiple dispatch, type stability)
- **Modular Architecture**: Easily extensible with custom solvers and visualization tools

---

## Installation

MORFE.jl is not yet registered in the Julia General Registry. To install:

```julia
using Pkg
Pkg.add(url="https://github.com/MORFEproject/MORFE3.0.git")
```

Or inPkg mode:

```julia
] add https://github.com/MORFEproject/MORFE3.0.git
```

---

## Quick Start

```julia
using MORFE

# Load your Full Order Model (FOM)
fom = FullOrderModel()

# Use the reduced model for fast computations
# (more examples coming soon)
```

For more detailed examples, see the `demo/` directory.

---

## Project Structure

```
MORFE3.0/
├── src/
│   └── MORFE.jl          # Main package module
├── demo/                  # Example scripts
├── test/                  # Test suite
├── docs/                  # Documentation
└── Project.toml          # Julia project manifest
```

---

## Status

⚠️ **Pre-Alpha**: This project is in early development. The API may change significantly.

### Implemented

- `FullOrderModel` module for representing FEM problems
- Polynomial framework (multiindices, basis functions)
- Realification utilities
- Lower-order coupling support

### Roadmap

See [Project Overview & Requirements](docs/project-overview.md) for detailed plans.

---

## Documentation

- [Project Overview & Requirements](docs/project-overview.md) - Detailed requirements and design decisions

---

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

- DPIM (Direct Parametrisation of Invariant Manifolds) method for nonlinear model order reduction
- MORFE2.0 - Previous version of the framework (Julia-based)
