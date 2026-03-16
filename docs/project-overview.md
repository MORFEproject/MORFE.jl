# MORFE.jl – Project Overview & Preliminary Requirements

This document summarises the key decisions, requirements, and future directions discussed for the development of **MORFE.jl**. It serves as a living reference for contributors and stakeholders.

---

## 🎯 Overview

MORFE.jl is the next generation of the **Model Order Reduction for Finite Elements** framework. It aims to overcome the limitations of its predecessor (MORFE2.0) by adopting modern software practices, improving modularity, and leveraging the Julia ecosystem for performance and flexibility.

---

## 🧱 Background

- **MORFE2.0** is based on an old Fortran FEM implementation, which is difficult to maintain and extend.
- **Goal**: Retain the proven concepts of MORFE2.0 while rebuilding the codebase from scratch in **Julia**, following modern software engineering guidelines.

---

## 🧭 Development Philosophy

- **Selective reuse**: Only the good parts of MORFE2.0 will be carried over.
- **Julia best practices**:
  - Write **small, composable functions**.
  - Organise code into **modules** with clear responsibilities.
  - Use **type stability** and **multiple dispatch** where appropriate.
- **Design agnosticism**: Core algorithms (e.g., DPIM) shall not be tied to a specific FEM package or solver.

---

## 💬 Language Choice: Julia vs Python

| Aspect         | Python                          | Julia                          |
|----------------|---------------------------------|--------------------------------|
| Popularity     | Wider community, more libraries | Niche, but growing             |
| Performance    | Slower, needs C extensions      | Fast, JIT compiled, “like C”   |
| Code reuse     | Partial                         | **Can reuse some MORFE2.0 logic** ✅ |

**Decision**: **Julia** is the primary language. Python bindings may be offered later for accessibility.

---

## 🔧 FEM Backend

MORFE.jl must interface with a FEM engine to extract mass, stiffness, and force arrays. Three options are under consideration:

| Option                     | Notes                                                                 |
|----------------------------|-----------------------------------------------------------------------|
| **Gridap.jl**              | [Gridap](https://github.com/gridap/Gridap.jl) – modern, native Julia  |
| **Ferrite.jl**             | [Ferrite](https://ferrite-fem.github.io/Ferrite.jl/stable/) – lightweight, Julia-only |
| **MORFE2.0 FEM** | Existing in Julia – we must decouple it from MORFE2.0  |

**Requirement**: The **DPIM (Direct Parametrisation of Invariant Manifolds)** algorithm **must be FEM‑agnostic**. It should work with any sufficiently capable FEM backend that provides the necessary operators.

---

## 📊 Post‑Processing & Solving

### Solving
- Current solving procedure is a “nightmare” – **must become plug‑and-play**.
- Acceptable solvers (to be kept in Julia if possible):
  - **HBM** (Harmonic Balance Method)
  - **COCO** / **MatCont** (continuation)
  - Others … as long as they are interfaced cleanly.
- The solver interface must be **design agnostic** (no hidden dependencies).

### Visualisation
- Export to **ParaView** (VTK format) is mandatory.
- Explore Julia packages similar to **PyVista** (e.g., `MeshIO.jl`, `WriteVTK.jl`, or dedicated visualisation tools).

---

## 📦 Outputs of MORFE.jl

1. **Reduced dynamics**  
   - Realified reduced dynamics `f(z)`  
   - Parametrisation `W(z)`  
   - Stored as **coefficient matrices**.

2. **Post‑processing of outputs**  
   - Coefficient matrices → **`FILE_FUNCTIONS`**  
   - The `FILE_FUNCTIONS` must be in **Julia**, **Python**, and **MATLAB** to maximise usability.

---

## ✅ Validation Strategy

### Internal validation (inside MORFE.jl)
- Compute the **residue (force)** of the invariance equation in the **time domain**, **over each orbit**.

### External validation (outside MORFE.jl)
- Compare against **Full Order Model (FOM)** displacement solutions:
  - **HBFEM** (existing, keep alive)
  - Time integration or other solvers.

---

## 🚀 “Would like to have” Features

These are not mandatory for the first release but are highly desirable for future iterations.

### 1. Design optimisation
- Enable optimisation workflows that reuse the reduced model.

### 2. Parametric FEM
- Support parametric studies without rebuilding the FEM model from scratch.

### 3. Automatic Differentiation (AD)
- Use AD to compute derivatives needed for reduced‑order modelling and sensitivity analysis.
- Candidate packages:
  - [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) – for forward‑mode AD.
  - [Zygote.jl](https://github.com/FluxML/Zygote.jl) – for reverse‑mode AD.

### 4. Multi-Physics capabilities
- The FE allows easy definition of different physics foer different domains of the mesh.

---

## 📌 Next Steps / Task Assignment

*(To be filled with specific tasks and owners)*

- [ ] Set up repository structure (Julia project, modules)
- [ ] Evaluate FEM backends (Gridap vs Ferrite vs wrapped Fortran)
- [ ] Prototype DPIM – FEM‑agnostic API
- [ ] Design output file format for coefficient matrices
- [ ] Define solver interface
- [ ] …

---

**Maintainer notes**:  
This document will evolve as the project progresses. Please update it whenever major decisions are made or requirements change.
