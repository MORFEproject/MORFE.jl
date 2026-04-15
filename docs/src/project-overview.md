# Project Overview

This document summarises the key decisions, requirements, and future directions for **MORFE.jl**. It serves as a living reference for contributors and stakeholders.

---

## Overview

MORFE.jl is the next generation of the **Model Order Reduction for Finite Elements** framework. It aims to overcome the limitations of its predecessor (MORFE2.0) by adopting modern software practices, improving modularity, and leveraging the Julia ecosystem for performance and flexibility.

---

## Background

- **MORFE2.0** is based on an older Fortran FEM implementation that is difficult to maintain and extend.
- **Goal**: Retain the proven algorithms of MORFE2.0 while rebuilding the codebase from scratch in **Julia**, following modern software engineering guidelines.

---

## Development Philosophy

- **Selective reuse**: Only the sound concepts of MORFE2.0 are carried over.
- **Julia best practices**:
  - Write **small, composable functions**.
  - Organise code into **modules** with clear responsibilities.
  - Use **type stability** and **multiple dispatch** where appropriate.
- **Design agnosticism**: Core algorithms (e.g., DPIM) are not tied to any specific FEM package or solver.

---

## Language Choice: Julia vs Python

| Aspect      | Python                          | Julia                                |
|-------------|---------------------------------|--------------------------------------|
| Popularity  | Wider community, more libraries | Niche but growing                    |
| Performance | Slower, needs C extensions      | Fast, JIT-compiled, "like C"         |
| Code reuse  | Partial                         | Can reuse MORFE2.0 logic directly    |

**Decision**: **Julia** is the primary language. Python bindings may be offered later for accessibility.

---

## FEM Backend

MORFE.jl must interface with a FEM engine to extract mass, stiffness, and force arrays. Three options are under consideration:

| Option              | Notes                                                                 |
|---------------------|-----------------------------------------------------------------------|
| **Gridap.jl**       | Modern, native Julia — [github.com/gridap/Gridap.jl](https://github.com/gridap/Gridap.jl) |
| **Ferrite.jl**      | Lightweight, Julia-only — [ferrite-fem.github.io](https://ferrite-fem.github.io/Ferrite.jl/stable/) |
| **MORFE2.0 FEM**    | Existing Julia implementation — must be decoupled from MORFE2.0      |

**Requirement**: The **DPIM** algorithm must be **FEM-agnostic** — it must work with any backend that provides the necessary operators.

---

## Post-Processing and Solving

### Solving

- Current solving procedure is a "nightmare" — must become **plug-and-play**.
- Acceptable solvers (to be kept in Julia if possible):
  - **HBM** (Harmonic Balance Method)
  - **COCO** / **MatCont** (continuation)
  - Others, as long as they are interfaced cleanly.
- The solver interface must be **design-agnostic** (no hidden dependencies).

### Visualisation

- Export to **ParaView** (VTK format) is mandatory.
- Julia packages such as `WriteVTK.jl` or `MeshIO.jl` are candidates.

---

## Outputs of MORFE.jl

1. **Reduced dynamics**
   - Realified reduced dynamics `f(z)`
   - Parametrisation `W(z)`
   - Stored as **coefficient matrices**.

2. **Post-processing utilities**
   - Coefficient matrices → `FILE_FUNCTIONS`
   - The `FILE_FUNCTIONS` must be in **Julia**, **Python**, and **MATLAB** to maximise usability.

---

## Validation Strategy

### Internal (inside MORFE.jl)

- Compute the **residue (force)** of the invariance equation in the **time domain**, over each orbit.

### External (outside MORFE.jl)

- Compare against Full Order Model (FOM) displacement solutions:
  - **HBFEM** (existing, keep alive)
  - Time integration or other solvers.

---

## Desirable Future Features

These are not mandatory for the first release but are targeted for future iterations.

### Design Optimisation

Enable optimisation workflows that reuse the reduced model.

### Parametric FEM

Support parametric studies without rebuilding the FEM model from scratch.

### Automatic Differentiation

Use AD to compute derivatives needed for reduced-order modelling and sensitivity analysis.

| Package | Mode |
|---------|------|
| [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) | Forward-mode AD |
| [Zygote.jl](https://github.com/FluxML/Zygote.jl) | Reverse-mode AD |

### Multi-Physics

The FE formulation allows different physics on different mesh domains.

---

## Next Steps

- [ ] Evaluate FEM backends (Gridap vs Ferrite vs wrapped MORFE2.0)
- [ ] Prototype DPIM — FEM-agnostic API
- [ ] Design output file format for coefficient matrices
- [ ] Define solver interface
- [ ] Add ParaView export

---

> This document will evolve as the project progresses. Update it whenever major decisions are made or requirements change.
