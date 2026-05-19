# Plan: Decouple MORFE.jl Dependencies

## Goal

Make MORFE.jl a proper minimal package. FEM backends (Ferrite, Gridap) become
separate companion packages. Optional solver/visualization features become Julia
package extensions (weakdeps). Dev and demo-only packages are removed from
`[deps]` entirely.

---

## Dependency audit

| Package | Currently | Action |
|---------|-----------|--------|
| `KLU` | `[deps]` | Keep (core sparse solver, hot path) |
| `StaticArrays` | `[deps]` | Keep (pervasive SVector use) |
| `LinearAlgebra` `SparseArrays` `Printf` `Random` `Mmap` `Statistics` | `[deps]` | Keep (stdlib) |
| `Arpack` | `[deps]` | → `[weakdeps]` + `ext/MORFEArpackExt.jl` |
| `LinearMaps` | `[deps]` | → `[weakdeps]` + `ext/MORFEArpackExt.jl` |
| `Pardiso` | `[deps]` | → `[weakdeps]` + `ext/MORFEPardisoExt.jl` |
| `Plots` | `[deps]` | → `[weakdeps]` + `ext/MORFEPlotsExt.jl` |
| `Gmsh` | `[deps]` | → `[weakdeps]` + `ext/MORFEGmshExt.jl` |
| `Ferrite` `FerriteGmsh` | `[deps]` | → remove; move to separate `MORFEFerrite.jl` repo |
| `Gridap` `GridapGmsh` | `[deps]` | → remove; kept in demo script only |
| `BenchmarkTools` `ProfileCanvas` `ProfileView` `JuliaFormatter` | `[deps]` | → remove (dev tools) |
| `HDF5` `WriteVTK` `ExtendableSparse` `FEMQuad` `KrylovKit` | `[deps]` | → remove (demo only) |
| `Profile` | `[deps]` | → remove (stdlib, dev only) |

---

## Step 1 — Restructure `Project.toml`

Replace the entire file with:

```toml
name = "MORFE"
uuid = "5d3630f1-b769-440d-b6a6-faf39c53b66e"
version = "3.1.0"
authors = ["MORFEproject <https://github.com/MORFE.jl>"]

[deps]
KLU          = "ef3ab10e-7fda-4108-b977-705223b18434"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Mmap         = "a63ad114-7e13-5084-954f-fe012c677804"
Printf       = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random       = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Statistics   = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[weakdeps]
Arpack     = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
Gmsh       = "705231aa-382f-11e9-3f0c-b7cb4346fdeb"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
Pardiso    = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
Plots      = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[extensions]
MORFEArpackExt  = ["Arpack", "LinearMaps"]
MORFEGmshExt    = "Gmsh"
MORFEPardisoExt = "Pardiso"
MORFEPlotsExt   = "Plots"

[extras]
Arpack         = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HDF5           = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
Test           = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Arpack", "BenchmarkTools", "HDF5", "Test"]

[compat]
Arpack       = "0.5"
Gmsh         = "0.3.1"
KLU          = "0.6.0"
LinearMaps   = "3"
Pardiso      = "1.1.2"
Plots        = "1.41.6"
StaticArrays = "1"
julia        = "1.10"
```

**Why `[extras]`/`[targets]`?** Julia's test runner installs packages listed under
`[targets] test` automatically. `Arpack` is needed by the eigensolver demo tests
and `BenchmarkTools` by the benchmark demos. `HDF5` is needed by some
parametrisation demos. These are test-time dependencies, not runtime.

---

## Step 2 — Arpack extension: modify two `src/` files

### 2a. `src/SpectralDecomposition/Eigenproblems.jl`

- **Remove** line 26: `using Arpack`
- **Replace** the two `solve` / `solve_left` method bodies for `ArpackEigensolver`
  (lines 87–105) with stubs:

```julia
function solve(model::NDOrderModel, solver::ArpackEigensolver)
    error(
        "ArpackEigensolver requires Arpack.jl and LinearMaps.jl.\n" *
        "Load them before loading MORFE: `using Arpack, LinearMaps; using MORFE`\n" *
        "or after: they are declared as weakdeps and the extension activates automatically."
    )
end

function solve_left(model::NDOrderModel, solver::ArpackEigensolver)
    error(
        "ArpackEigensolver requires Arpack.jl and LinearMaps.jl.\n" *
        "Load them before loading MORFE: `using Arpack, LinearMaps; using MORFE`\n" *
        "or after: they are declared as weakdeps and the extension activates automatically."
    )
end
```

Keep everything else unchanged: the struct definition, `DefaultEigensolver`,
`Eigenproblem`, `compute_eigenproblem`, selection helpers, etc.

### 2b. `src/SpectralDecomposition/Eigensolvers.jl`

- **Remove** lines 13–15: `using Arpack`, `using LinearMaps`
- **Replace** the body of `generalised_eigenpairs` with a stub:

```julia
function generalised_eigenpairs(A::AbstractMatrix, B::AbstractMatrix; nev, kwargs...)
    error(
        "generalised_eigenpairs requires Arpack.jl and LinearMaps.jl.\n" *
        "Load them with `using Arpack, LinearMaps` to activate the MORFE extension."
    )
end
```

- Keep `_sort_largest_real` (it uses only `LinearAlgebra`, which is always available).

### 2c. Create `ext/MORFEArpackExt.jl`

This extension is activated automatically when both `Arpack` and `LinearMaps` are
loaded in the same session. It overrides the stubs defined above.

```julia
module MORFEArpackExt

using MORFE
using MORFE: Eigenproblems, Eigensolvers
using Arpack
using LinearAlgebra
using LinearMaps
using SparseArrays

# ── ArpackEigensolver ─────────────────────────────────────────────────────────

function Eigenproblems.solve(
        model::MORFE.FullOrderModel.NDOrderModel, solver::Eigenproblems.ArpackEigensolver)
    A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)
    eig_result = if solver.nev === nothing
        eigs(A, B; which = :SM)
    else
        eigs(A, B; nev = solver.nev, which = :SM)
    end
    return eig_result[1], eig_result[2]
end

function Eigenproblems.solve_left(
        model::MORFE.FullOrderModel.NDOrderModel, solver::Eigenproblems.ArpackEigensolver)
    A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)
    eig_result = if solver.nev === nothing
        eigs(A', B'; which = :SM)
    else
        eigs(A', B'; nev = solver.nev, which = :SM)
    end
    return conj(eig_result[1]), eig_result[2]
end

# ── generalised_eigenpairs (full shift-invert implementation) ─────────────────
# Move the full body verbatim from the current src/SpectralDecomposition/Eigensolvers.jl

function Eigensolvers.generalised_eigenpairs(
        A::AbstractMatrix, B::AbstractMatrix;
        nev::Integer, shift = nothing, which::Symbol = :LR,
        tol::Real = 0.0, maxiter::Integer = 3000,
        ncv::Union{Nothing, Integer} = nothing, v0 = nothing,
        ritzvec::Bool = true, sort_largest_real::Bool = false)
    # ... full implementation moved here verbatim from current Eigensolvers.jl ...
end

end # module MORFEArpackExt
```

**Note on extension dispatch:** Julia extensions can define methods for functions
already defined in the parent package. The signatures here must exactly match the
dispatch target — the stubs in core define the function; the extension defines the
method.

---

## Step 3 — Pardiso extension: modify three `src/` files

The Pardiso coupling has three touch-points:
1. `CohomologicalEquations.jl:93` — the `using Pardiso` import
2. `SolverResources.jl:87` — `pardiso::Union{Nothing, AbstractPardisoSolver}` field type
3. `CohomologicalSolver.jl:16` — `_sparse_solve(ps::AbstractPardisoSolver, ...)` dispatch

### 3a. `src/ParametrisationMethod/CohomologicalEquations/CohomologicalEquations.jl`

- **Remove** line 93:
  ```julia
  using Pardiso: AbstractPardisoSolver, MKLPardisoSolver, solve as pardiso_solve
  ```
- **Add** two extension-hook functions (near the top of the module, before the includes
  that reference `SolverResources.jl`):

```julia
# Extension hooks: overridden by ext/MORFEPardisoExt.jl when Pardiso is loaded.
_try_build_pardiso_solver() = nothing
_pardiso_solve(ps, A, B) =
    error("Pardiso solver object present but MORFEPardisoExt not active — internal error.")
```

Export these from the `CohomologicalEquations` module so the extension can qualify them:
```julia
export _try_build_pardiso_solver, _pardiso_solve
```

### 3b. `src/ParametrisationMethod/CohomologicalEquations/SolverResources.jl`

- **Line 87**: Change field type from
  ```julia
  pardiso::Union{Nothing, AbstractPardisoSolver}
  ```
  to
  ```julia
  pardiso::Any   # Nothing, or an AbstractPardisoSolver when Pardiso ext is loaded
  ```

- **Lines 104–112**: Replace the direct `MKLPardisoSolver()` / `Pardiso.PardisoSolver()`
  constructor calls:
  ```julia
  # Before:
  ps = nothing
  try ps = MKLPardisoSolver() catch end
  if ps === nothing
      try ps = Pardiso.PardisoSolver() catch end
  end
  if ps === nothing
      @warn "Neither MKL Pardiso nor open-source Pardiso is available. " *
            "Falling back to KLU (SuiteSparse) for the sparse cohomological solve."
  end
  ```
  With:
  ```julia
  ps = _try_build_pardiso_solver()   # returns nothing unless MORFEPardisoExt is loaded
  ```

### 3c. `src/ParametrisationMethod/CohomologicalEquations/CohomologicalSolver.jl`

- **Line 16**: Replace the type-dispatched overload
  ```julia
  _sparse_solve(ps::AbstractPardisoSolver, ::Any, A, B) = pardiso_solve(ps, A, B)
  ```
  with a duck-typed fallback that routes through the extension hook:
  ```julia
  _sparse_solve(ps, ::Any, A, B) = _pardiso_solve(ps, A, B)
  ```
  The existing `_sparse_solve(::Nothing, ...)` methods are unchanged.

  **Dispatch order is preserved:**
  - `ps === nothing` → `_sparse_solve(::Nothing, klu_cache, ...)` (KLU path)
  - `ps !== nothing` → `_sparse_solve(ps, ...)` (duck-typed; calls `_pardiso_solve`)

### 3d. Create `ext/MORFEPardisoExt.jl`

```julia
module MORFEPardisoExt

using MORFE
using MORFE.CohomologicalEquations: _try_build_pardiso_solver, _pardiso_solve
using Pardiso

function MORFE.CohomologicalEquations._try_build_pardiso_solver()
    ps = nothing
    try ps = MKLPardisoSolver() catch end
    if ps === nothing
        try ps = Pardiso.PardisoSolver() catch end
    end
    if ps === nothing
        @warn "Neither MKL Pardiso nor open-source Pardiso is available. " *
              "Falling back to KLU (SuiteSparse) for the sparse cohomological solve."
    end
    return ps
end

function MORFE.CohomologicalEquations._pardiso_solve(ps, A, B)
    return Pardiso.solve(ps, A, B)
end

end # module MORFEPardisoExt
```

---

## Step 4 — Plots extension: modify one `src/` file

### 4a. `src/Validation/InvarianceError.jl`

- **Remove** line 16: `using Plots`
- **Replace** the full body of `plot_invariance_convergence` (and any private helpers
  it calls — `_plot_convergence`, `_reference_line_params`, etc.) with a stub:

```julia
function plot_invariance_convergence(results; kwargs...)
    error(
        "plot_invariance_convergence requires Plots.jl.\n" *
        "Load it with `using Plots` to activate the MORFE extension."
    )
end
```

Keep `invariance_error_norms` and `invariance_error_convergence` unchanged (they
use only `LinearAlgebra`, `Random`, `Statistics` — all in core `[deps]`).

### 4b. Create `ext/MORFEPlotsExt.jl`

```julia
module MORFEPlotsExt

using MORFE
using MORFE.InvarianceError
using Plots
using Statistics: median

# Move the full implementation of plot_invariance_convergence,
# _plot_convergence, _reference_line_params here verbatim from
# src/Validation/InvarianceError.jl.
function MORFE.InvarianceError.plot_invariance_convergence(results; kwargs...)
    # ... full implementation ...
end

end # module MORFEPlotsExt
```

---

## Step 5 — Gmsh extension

The three FEMUtility modules (`AbaqusToGmsh`, `ComsolToGmsh`, `GmshToComsol`) are
already in `src/FEMUtility/` but are **not currently included** by `src/MORFE.jl`
(confirmed: no `include("FEMUtility/...")` lines exist there). They need stubs so the
names are exported from MORFE before the extension loads.

### 5a. Create `src/FEMUtility/FEMUtility.jl`

```julia
module FEMUtility

export abaqus_to_gmsh, abaqus_to_gmsh_linear,
       comsol_to_gmsh, comsol_to_gmsh_linear,
       gmsh_to_comsol

_ext_err(f) = error("$f requires Gmsh.jl. Load it with `using Gmsh` to activate the extension.")

abaqus_to_gmsh(args...; kw...)        = _ext_err("abaqus_to_gmsh")
abaqus_to_gmsh_linear(args...; kw...) = _ext_err("abaqus_to_gmsh_linear")
comsol_to_gmsh(args...; kw...)        = _ext_err("comsol_to_gmsh")
comsol_to_gmsh_linear(args...; kw...) = _ext_err("comsol_to_gmsh_linear")
gmsh_to_comsol(args...; kw...)        = _ext_err("gmsh_to_comsol")

end # module
```

### 5b. `src/MORFE.jl` — add the new submodule

Add after the existing `include`s (before the `using .` block):

```julia
include("FEMUtility/FEMUtility.jl")
```

Add to the `using .` block:

```julia
using .FEMUtility
```

Export:

```julia
# FEMUtility (Gmsh extension must be loaded for these to work)
export abaqus_to_gmsh, abaqus_to_gmsh_linear,
       comsol_to_gmsh, comsol_to_gmsh_linear,
       gmsh_to_comsol
```

### 5c. Create `ext/MORFEGmshExt.jl`

The FEMUtility source files use Gmsh directly; they can be included inside the
extension module with `pkgdir` to resolve the absolute path:

```julia
module MORFEGmshExt

using MORFE
using MORFE.FEMUtility
using Gmsh
using Printf

# The existing implementation files stay in src/FEMUtility/ unchanged.
# We include them here so they are only compiled when Gmsh is available.
_src(f) = joinpath(pkgdir(MORFE), "src", "FEMUtility", f)
include(_src("AbaqusToGmsh.jl"))
include(_src("ComsolToGmsh.jl"))
include(_src("GmshToComsol.jl"))

# Override the stubs
MORFE.FEMUtility.abaqus_to_gmsh(args...; kw...)        = AbaqusToGmsh.abaqus_to_gmsh(args...; kw...)
MORFE.FEMUtility.abaqus_to_gmsh_linear(args...; kw...) = AbaqusToGmsh.abaqus_to_gmsh_linear(args...; kw...)
MORFE.FEMUtility.comsol_to_gmsh(args...; kw...)        = ComsolToGmsh.comsol_to_gmsh(args...; kw...)
MORFE.FEMUtility.comsol_to_gmsh_linear(args...; kw...) = ComsolToGmsh.comsol_to_gmsh_linear(args...; kw...)
MORFE.FEMUtility.gmsh_to_comsol(args...; kw...)        = GmshToComsol.gmsh_to_comsol(args...; kw...)

end # module MORFEGmshExt
```

---

## Step 6 — Create `MORFEFerrite.jl` as a separate GitHub repository

This exactly mirrors the GridapGmsh → Gridap relationship. MORFE knows nothing
about MORFEFerrite; MORFEFerrite depends on MORFE.

### New repo structure: `github.com/<org>/MORFEFerrite.jl`

```
MORFEFerrite.jl/          ← new standalone git repository
  Project.toml
  src/
    MORFEFerrite.jl
    GeometricNonlinearity.jl   ← content from demo/Ferrite/ferrite_assembly.jl
  test/
    runtests.jl
  README.md
```

### `Project.toml`

```toml
name = "MORFEFerrite"
uuid = "<generate with using UUIDs; UUIDs.uuid4()>"
version = "0.1.0"

[deps]
Ferrite       = "c061ca5d-56c9-439f-9c0e-210fe06d3992"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MORFE         = "5d3630f1-b769-440d-b6a6-faf39c53b66e"
SparseArrays  = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
Ferrite = "1.4"
MORFE   = "3"
julia   = "1.10"
```

### `src/MORFEFerrite.jl`

```julia
module MORFEFerrite

using MORFE
using Ferrite
using LinearAlgebra
using SparseArrays

include("GeometricNonlinearity.jl")

export FerriteGeometricNonlinearity, assemble_KM!

end # module
```

### `src/GeometricNonlinearity.jl`

Copy verbatim from `demo/Ferrite/ferrite_assembly.jl` (in the MORFE repo), with:
1. Remove the top-level `using` statements (handled by `MORFEFerrite.jl`)
2. Remove `import MORFE` (already available via `using MORFE`)
3. Keep all `MORFE.fem_elements(...)` etc. qualified, or drop the `MORFE.` prefix —
   both work since `using MORFE` imports the interface functions.

### `test/runtests.jl`

```julia
using Test, MORFE, MORFEFerrite, Ferrite

@testset "MORFEFerrite" begin
    @test FerriteGeometricNonlinearity{2} <: FEMMultilinearMap{2}
    @test FerriteGeometricNonlinearity{3} <: FEMMultilinearMap{2}
end
```

### Installation workflow (for users)

```julia
# Before registration in General Registry:
Pkg.add(url = "https://github.com/<org>/MORFEFerrite.jl")

# After registration:
Pkg.add("MORFEFerrite")

# In scripts:
using MORFE
using MORFEFerrite      # provides FerriteGeometricNonlinearity
using FerriteGmsh       # provides togrid() for mesh loading
```

### Development workflow (while iterating on both repos locally)

```julia
Pkg.develop(path = "/path/to/MORFE_jl")
Pkg.develop(path = "/path/to/MORFEFerrite.jl")
```

### What stays in the MORFE repo

`demo/Ferrite/ferrite_assembly.jl` is deleted from the MORFE repo (its content moves
to MORFEFerrite.jl). The demo script `demo/Ferrite/demo_mechanical_problem.jl` is
updated to `using MORFEFerrite` instead of the `include` (see Step 7).

---

## Step 7 — Update demos

### `demo/Ferrite/demo_mechanical_problem.jl`

Replace:
```julia
using Ferrite
# ...
include(joinpath(@__DIR__, "ferrite_assembly.jl"))
```
with:
```julia
using MORFEFerrite    # provides FerriteGeometricNonlinearity, assemble_KM!
using FerriteGmsh     # provides togrid()
using Arpack, LinearMaps   # trigger MORFEArpackExt (needed for ArpackEigensolver)
```

Remove `demo/Ferrite/ferrite_assembly.jl` (its content is now in the companion
package).

### `demo/Gridap/demo_mechanical_problem.jl`

The Gridap demo uses a closure-based `MultilinearMap` — no `FEMMultilinearMap`
interface needed. The custom `Mechanical_Problem_Solver` calls `eigs` directly,
which is fine after `using Arpack`. No structural change needed except:
- Keep `using Arpack` (now a weakdep; loading it triggers `MORFEArpackExt`)
- The `using Gmsh` line stays as-is (triggers `MORFEGmshExt`, harmless if FEMUtility
  not used by the demo)

### `demo/FEMUtility/` demos

Any demo that calls `abaqus_to_gmsh` etc. must `using Gmsh` before `using MORFE`
(or after — the extension activates either way). No other change needed.

---

## Step 8 — `Manifest.toml`

After editing `Project.toml`, regenerate the manifest:

```julia
using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()
```

Commit the new `Manifest.toml`. The old one will have many more packages pinned
(due to the removed deps); the new one will be much leaner.

---

## File change summary

| File | Action |
|------|--------|
| `Project.toml` | Full rewrite (Step 1) |
| `src/SpectralDecomposition/Eigenproblems.jl` | Remove `using Arpack`; stub `solve`/`solve_left` for ArpackEigensolver |
| `src/SpectralDecomposition/Eigensolvers.jl` | Remove `using Arpack`, `using LinearMaps`; stub `generalised_eigenpairs` |
| `src/ParametrisationMethod/CohomologicalEquations/CohomologicalEquations.jl` | Remove `using Pardiso`; add `_try_build_pardiso_solver` and `_pardiso_solve` hooks |
| `src/ParametrisationMethod/CohomologicalEquations/SolverResources.jl` | `pardiso::Any`; call hook instead of constructors |
| `src/ParametrisationMethod/CohomologicalEquations/CohomologicalSolver.jl` | Duck-typed `_sparse_solve` dispatch |
| `src/Validation/InvarianceError.jl` | Remove `using Plots`; stub `plot_invariance_convergence`; move implementation to ext |
| `src/MORFE.jl` | Add `include("FEMUtility/FEMUtility.jl")`, `using .FEMUtility`, exports |
| `src/FEMUtility/FEMUtility.jl` | **New** stub module |
| `ext/MORFEArpackExt.jl` | **New** extension |
| `ext/MORFEPardisoExt.jl` | **New** extension |
| `ext/MORFEPlotsExt.jl` | **New** extension |
| `ext/MORFEGmshExt.jl` | **New** extension |
| `demo/Ferrite/ferrite_assembly.jl` | **Delete** (content → `MORFEFerrite.jl` repo) |
| `demo/Ferrite/demo_mechanical_problem.jl` | Update imports (Step 7) |

---

## Verification

### 1. Core loads without optional deps

```julia
# In a clean environment with only core [deps] installed:
using MORFE
# Must load without error. Arpack, Ferrite, Pardiso etc. must NOT be required.
```

### 2. Stubs give helpful errors

```julia
using MORFE
m = NDOrderModel(...)
MORFE.Eigensolvers.generalised_eigenpairs(A, B; nev=5)
# Expected: ErrorException "requires Arpack.jl and LinearMaps.jl..."

using MORFE
plot_invariance_convergence([])
# Expected: ErrorException "requires Plots.jl..."
```

### 3. Extensions activate automatically

```julia
# Arpack extension:
using Arpack, LinearMaps  # or: these come first; MORFE second
using MORFE
generalised_eigenpairs(A, B; nev=5)   # must work without error

# Pardiso extension (on machines with Pardiso):
using Pardiso, MORFE
# SparseLinearSolverState constructor must pick up the Pardiso handle

# Plots extension:
using Plots, MORFE
plot_invariance_convergence(results)   # must produce a plot
```

### 4. MORFEFerrite companion package

```julia
using Pkg
# Before registration — install directly from the new repo:
Pkg.add(url = "https://github.com/<org>/MORFEFerrite.jl")
using MORFEFerrite
@assert FerriteGeometricNonlinearity{2} <: FEMMultilinearMap{2}
```

### 5. Ferrite demo runs end-to-end

```julia
include("demo/Ferrite/demo_mechanical_problem.jl")
# Must complete without error
```

### 6. Test suite

```bash
GROUP=tests julia --project test/runtests.jl    # core tests, no optional deps
GROUP=demos julia --project test/runtests.jl    # demo tests; needs Arpack in [extras]
GROUP=all   julia --project test/runtests.jl    # everything
```

---

## Notes and tradeoffs

- **`KLU` stays hard**: it is on the critical path of `CohomologicalSolver.jl` with
  symbolic-factor caching. The install cost is a single artifact binary via
  BinaryBuilder — negligible.

- **No `MORFEGridap` package**: the Gridap demo uses closure-based `MultilinearMap`
  (not the `FEMMultilinearMap` interface), which requires no companion package.
  It is self-contained and correct as a demo script.

- **Fallback without Arpack**: `DefaultEigensolver()` (dense `LinearAlgebra.eigen`)
  is always available and correct for small FOM (< ~5000 DOF). For large systems,
  users must `using Arpack, LinearMaps`.

- **Extension loading order**: Julia activates an extension when ALL trigger packages
  are loaded. For `MORFEArpackExt`, both `Arpack` and `LinearMaps` must be loaded.
  Order relative to `MORFE` does not matter.
