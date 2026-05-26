# Plan: Decouple MORFE.jl Dependencies
## (Adapted to current codebase — 2026-05-26)

## Goal

Make MORFE.jl a proper minimal package. FEM backends (Ferrite, Gridap) become
separate companion packages. Optional solver/visualization/continuation features
become Julia package extensions (weakdeps). Dev and demo-only packages are
removed from `[deps]` entirely.

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
| `BifurcationKit` | absent | → `[weakdeps]` + `ext/MORFEBifurcationKitExt.jl` |
| `Ferrite` `FerriteGmsh` | `[deps]` | → remove; move to separate `MORFEFerrite.jl` repo |
| `Gridap` `GridapGmsh` | `[deps]` | → remove; kept in demo script only |
| `BenchmarkTools` `ProfileCanvas` `ProfileView` `JuliaFormatter` | `[deps]` | → remove (dev tools) |
| `HDF5` `WriteVTK` `ExtendableSparse` `FEMQuad` `KrylovKit` `Tensors` | `[deps]` | → remove (demo only) |
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
Arpack          = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
BifurcationKit  = "0f109fa4-8a5d-4b75-95aa-f515264e7665"
Gmsh            = "705231aa-382f-11e9-3f0c-b7cb4346fdeb"
LinearMaps      = "7a12625a-238d-50fd-b39a-03d52299707e"
Pardiso         = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
Plots           = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[extensions]
MORFEArpackExt         = ["Arpack", "LinearMaps"]
MORFEBifurcationKitExt = "BifurcationKit"
MORFEGmshExt           = "Gmsh"
MORFEPardisoExt        = "Pardiso"
MORFEPlotsExt          = "Plots"

[extras]
Arpack         = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
HDF5           = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
Test           = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Arpack", "BenchmarkTools", "HDF5", "Test"]

[compat]
Arpack         = "0.5"
BifurcationKit = "0.4"
Gmsh           = "0.3.1"
KLU            = "0.6.0"
LinearMaps     = "3"
Pardiso        = "1.1.2"
Plots          = "1.41.6"
StaticArrays   = "1"
julia          = "1.10"
```

**`[extras]`/`[targets]`**: Julia's test runner installs these automatically.
`Arpack` is needed by the eigensolver tests; `BenchmarkTools` by benchmark demos;
`HDF5` by some parametrisation demos.

---

## Step 2 — Arpack extension

### Architecture note

The entire `src/SpectralDecomposition/` directory moves out of `src/`. These are
eigenanalysis utilities, not part of the core DPIM algorithm.

Julia's type system requires that **struct definitions** reside in the module they
belong to — they cannot be injected into a module from an extension. The practical
resolution:

- `src/SpectralDecomposition/Eigenproblems.jl` and `Eigensolvers.jl` become
  **type-and-stub files** in `src/SpectralDecomposition/`: all struct/type
  definitions stay (required for dispatch), Arpack-dependent method bodies are
  replaced with stubs. The `DefaultEigensolver` full implementation stays here
  (it uses only `LinearAlgebra`).
- The Arpack-dependent **method implementations** physically live in
  `ext/MORFEArpackExt.jl`.
- `JordanChain.jl` and `PropagateEigenmodes.jl` are **not DPIM** and move
  entirely to `ext/SpectralDecomposition/`. They are removed from `src/MORFE.jl`.
  `PropagateEigenmodes.jl` uses `..FullOrderModel` and `..ParametrisationMethod`
  relative imports that must be updated to `MORFE.FullOrderModel` /
  `MORFE.ParametrisationMethod` because when compiled inside `MORFEArpackExt`
  the `..` parent is the extension module, not `MORFE`.

**API consequence:** after this change, `compute_jordan_chain`,
`propagate_right_eigenvector_from_first`, `propagate_left_eigenvector_from_last`,
`propagate_right_jordan_vector`, `propagate_left_jordan_vector` are only available
after `using Arpack, LinearMaps`. They are accessible as
`MORFEArpackExt.JordanChain.*` / `MORFEArpackExt.PropagateEigenmodes.*`, or
re-exported at the extension top level (see §2c).

---

### 2a. `src/SpectralDecomposition/Eigensolvers.jl`

**Remove** lines 13–15:
```julia
using Arpack
using LinearMaps
```
(keep `using LinearAlgebra`, `using SparseArrays`)

**Replace** the entire body of `generalised_eigenpairs` with a stub:

```julia
function generalised_eigenpairs(
    A::AbstractMatrix,
    B::AbstractMatrix;
    nev::Integer,
    shift = nothing,
    which::Symbol = :LR,
    tol::Real = 0.0,
    maxiter::Integer = 3000,
    ncv::Union{Nothing, Integer} = nothing,
    v0 = nothing,
    ritzvec::Bool = true,
    sort_largest_real::Bool = false,
)
    error(
        "generalised_eigenpairs requires Arpack.jl and LinearMaps.jl.\n" *
        "Load them with `using Arpack, LinearMaps` to activate the MORFE extension."
    )
end
```

**Keep** `_sort_largest_real` unchanged (uses only `LinearAlgebra`).

---

### 2b. `src/SpectralDecomposition/Eigenproblems.jl`

**Remove** line 51: `using Arpack`

**Keep unchanged**:
- `abstract type AbstractEigensolver end` and the generic fallback `solve`/`solve_left`
- `struct DefaultEigensolver` + its full `solve` and `solve_left` implementations
  (they use only `LinearAlgebra.eigen` — no change needed)
- `struct ArpackEigensolver` and both inner constructors (type definition only)
- `struct MorfeEigensolver` and inner constructors (type definition only)
  > **Pre-existing bug**: `MorfeEigensolver()` calls `new(nothing, nothing)` but the struct has 3 fields (`nev`, `shift`, `eigenvalues`). This is a pre-existing defect; do not fix it in this PR as it is out of scope.
- `struct StructureModalDampingEigensolver` and inner constructor (type definition only)
- `struct Eigenproblem{T}` and both inner constructors
- All `solve_eigenproblem` overloads
- `sort_by_magnitude!`, `sort_left_eigenmodes`, `normalize_biorthogonal!`,
  `get_eigenpairs`, `select_master_modes_*` functions
- The thin wrapper `solve(model::NDOrderModel, solver::StructureModalDampingEigensolver)`
  (it delegates to `solve(mass, stiffness, solver)` whose body moves to ext)
- Both `solve_eigenproblem(model, solver::StructureModalDampingEigensolver; ...)` specialisations
  (they call `solve(model, solver)` which triggers the stub → extension chain)

**Replace** the bodies of the three Arpack-dependent `solve`/`solve_left` methods
with stubs:

```julia
# ArpackEigensolver — right eigenproblem
function solve(model::NDOrderModel, solver::ArpackEigensolver)
    error(
        "ArpackEigensolver requires Arpack.jl and LinearMaps.jl.\n" *
        "Load them with `using Arpack, LinearMaps` to activate the MORFE extension."
    )
end

# ArpackEigensolver — left eigenproblem
function solve_left(model::NDOrderModel, solver::ArpackEigensolver)
    error(
        "ArpackEigensolver requires Arpack.jl and LinearMaps.jl.\n" *
        "Load them with `using Arpack, LinearMaps` to activate the MORFE extension."
    )
end

# MorfeEigensolver — right eigenproblem
# NOTE: MorfeEigensolver.solve calls generalised_eigenpairs (already stubbed in Eigensolvers.jl).
# The body can be kept as-is; it will fail via the Eigensolvers stub automatically.
# No separate stub needed here.

# MorfeEigensolver — left eigenproblem
# Same logic as above. Keep body as-is.

# StructureModalDampingEigensolver — calls eigs() directly; needs its own stub.
function solve(
    mass::AbstractMatrix{T},
    stiffness::AbstractMatrix{T},
    solver::StructureModalDampingEigensolver,
) where {T}
    error(
        "StructureModalDampingEigensolver requires Arpack.jl and LinearMaps.jl.\n" *
        "Load them with `using Arpack, LinearMaps` to activate the MORFE extension."
    )
end
```

---

### 2c. Move `JordanChain.jl` and `PropagateEigenmodes.jl` to `ext/SpectralDecomposition/`

Physically move both files:
```
src/SpectralDecomposition/JordanChain.jl          → ext/SpectralDecomposition/JordanChain.jl
src/SpectralDecomposition/PropagateEigenmodes.jl  → ext/SpectralDecomposition/PropagateEigenmodes.jl
```

**Edit `ext/SpectralDecomposition/PropagateEigenmodes.jl`** — update the two relative imports:
```julia
# Before:
using ..FullOrderModel
using ..ParametrisationMethod: Parametrisation

# After:
using MORFE.FullOrderModel
using MORFE.ParametrisationMethod: Parametrisation
```

`JordanChain.jl` uses only stdlib (`LinearAlgebra`, `SparseArrays`, `Printf`) —
no import changes needed.

Remove the following four lines from **`src/MORFE.jl`**:
```julia
include("SpectralDecomposition/JordanChain.jl")
include("SpectralDecomposition/PropagateEigenmodes.jl")
using .JordanChain
using .PropagateEigenmodes
```
Also remove any top-level `export` lines for `compute_jordan_chain`,
`propagate_right_eigenvector_from_first`, `propagate_left_eigenvector_from_last`,
`propagate_right_jordan_vector`, `propagate_left_jordan_vector`.

---

### 2d. Create `ext/MORFEArpackExt.jl`

This extension is activated when **both** `Arpack` and `LinearMaps` are loaded.
It contains the **exact current implementations** from `src/` — do not simplify.

```julia
module MORFEArpackExt

using MORFE
using MORFE.Eigenproblems
using MORFE.Eigensolvers
using Arpack
using LinearAlgebra
using LinearMaps
using SparseArrays
using Printf   # needed by JordanChain

# ── JordanChain and PropagateEigenmodes ────────────────────────────────────────
# These modules are not DPIM; they live in ext/ and are compiled only here.
_ext(f) = joinpath(pkgdir(MORFE), "ext", "SpectralDecomposition", f)
include(_ext("JordanChain.jl"))          # defines MORFEArpackExt.JordanChain
include(_ext("PropagateEigenmodes.jl"))  # defines MORFEArpackExt.PropagateEigenmodes

# Re-export at extension top level so `using Arpack, LinearMaps, MORFE` followed by
# `using MORFEArpackExt` (or direct module qualification) provides access.
using .JordanChain: compute_jordan_chain
using .PropagateEigenmodes: propagate_right_eigenvector_from_first,
                             propagate_left_eigenvector_from_last,
                             propagate_right_jordan_vector,
                             propagate_left_jordan_vector

# ── generalised_eigenpairs ─────────────────────────────────────────────────────
# Full implementation — move verbatim from current src/SpectralDecomposition/Eigensolvers.jl

function MORFE.Eigensolvers.generalised_eigenpairs(
    A::AbstractMatrix,
    B::AbstractMatrix;
    nev::Integer,
    shift = nothing,
    which::Symbol = :LR,
    tol::Real = 0.0,
    maxiter::Integer = 3000,
    ncv::Union{Nothing, Integer} = nothing,
    v0 = nothing,
    ritzvec::Bool = true,
    sort_largest_real::Bool = false,
)
    n = size(A, 1)
    @assert size(A, 2) == n "A must be square"
    @assert size(B, 1) == n && size(B, 2) == n "B must be square and match A size"
    @assert 0 <= nev <= n "nev must satisfy 0 < nev < size(A,1)"

    Tval = isnothing(shift) ? promote_type(eltype(A), eltype(B)) :
           promote_type(eltype(A), eltype(B), typeof(shift))
    Ac = sparse(Tval.(A))
    Bc = sparse(Tval.(B))

    ncv_eff = min(isnothing(ncv) ? max(Int(nev) + 30, 120) : Int(ncv), n - 1)
    v0_eff = isnothing(v0) ? nothing : Tval.(v0)

    base_eigs_kwargs = (
        nev = Int(nev),
        which = which,
        tol = Float64(tol),
        maxiter = Int(maxiter),
        ncv = ncv_eff,
        ritzvec = ritzvec,
    )
    eigs_kwargs = isnothing(v0_eff) ? base_eigs_kwargs :
                  merge(base_eigs_kwargs, (v0 = v0_eff,))

    vals = Vector{Tval}()
    vecs = Matrix{Tval}(undef, n, 0)
    nconv = 0
    niter = 0
    nmult = 0
    resid = Tval[]

    if isnothing(shift)
        vals, vecs, nconv, niter, nmult, resid = eigs(Ac, Bc; eigs_kwargs...)
    else
        sigc = convert(Tval, shift)
        F = lu(Ac - sigc * Bc)
        T_lm = LinearMap{Tval}(n, n; ismutating = false) do x
            F \ (Bc * x)
        end
        mu, vecs, nconv, niter, nmult,
        resid = eigs(T_lm; merge(eigs_kwargs, (which = :LM,))...)

        tiny = eps(real(float(one(Tval))))
        mu_safe = similar(mu)
        for i in eachindex(mu)
            mu_safe[i] = abs(mu[i]) < tiny ? convert(Tval, tiny) : mu[i]
        end
        vals = sigc .+ inv.(mu_safe)
    end

    if sort_largest_real
        vals, vecs = MORFE.Eigensolvers._sort_largest_real(vals, vecs)
    end

    return (
        values = vals,
        vectors = vecs,
        nconv = nconv,
        niter = niter,
        nmult = nmult,
        resid = resid,
    )
end

# ── ArpackEigensolver.solve ────────────────────────────────────────────────────
# Move verbatim from Eigenproblems.jl:155-171. Preserves reshape and nev/eigenvalues mutation.

function MORFE.Eigenproblems.solve(
    model::MORFE.FullOrderModel.NDOrderModel,
    solver::MORFE.Eigenproblems.ArpackEigensolver,
)
    A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    if solver.nev === nothing
        solver.nev = FOM * ORD
        (values, vectors) = eigs(A, B, which = :SM)
    else
        (values, vectors) = eigs(A, B, nev = solver.nev, which = :SM)
    end
    num_eigenvals = length(values)
    reshaped_eigenvectors = reshape(vectors, FOM, ORD, num_eigenvals)
    solver.eigenvalues = values
    return values, reshaped_eigenvectors
end

# ── ArpackEigensolver.solve_left ───────────────────────────────────────────────
# Move verbatim from Eigenproblems.jl:178-196.
# Uses per-eigenvalue sigma-shifts — NOT a simple eigs(A', B'; which=:SM) call.

function MORFE.Eigenproblems.solve_left(
    model::MORFE.FullOrderModel.NDOrderModel,
    solver::MORFE.Eigenproblems.ArpackEigensolver,
)
    A, B = MORFE.FullOrderModel.linear_first_order_matrices(model)
    A_c = complex.(A)
    B_c = complex.(B)
    A_adjoint = A_c'
    B_adjoint = B_c'
    @assert solver.nev == length(solver.eigenvalues)
    FOM = size(model.linear_terms[1], 1)
    ORD = length(model.linear_terms) - 1
    left_eigenvectors = Array{ComplexF64}(undef, FOM, ORD, solver.nev)
    eigenvalues = Vector{ComplexF64}(undef, solver.nev)
    for i in 1:(solver.nev)
        values, vectors = eigs(
            A_adjoint, B_adjoint,
            sigma = conj(solver.eigenvalues[i]),
            which = :LM, nev = 1, ncv = 30,
        )
        left_eigenvectors[:, :, i] = reshape(vectors[:, 1], FOM, ORD)
        eigenvalues[i] = conj(values[1])
    end
    return eigenvalues, left_eigenvectors
end

# ── StructureModalDampingEigensolver.solve(mass, stiffness, solver) ────────────
# Move verbatim from Eigenproblems.jl:308-348.
# The thin wrapper solve(model, solver) stays in core and delegates here.

function MORFE.Eigenproblems.solve(
    mass::AbstractMatrix{T},
    stiffness::AbstractMatrix{T},
    solver::MORFE.Eigenproblems.StructureModalDampingEigensolver,
) where {T}
    ω2, ϕ = eigs(stiffness, mass; nev = solver.nev, which = :SM, check = 1)
    any(x -> abs(imag(x)) > 1e-12 * abs(real(x)), ω2) && error("Eigenvalues not real.")
    ω2 = real.(ω2)
    ϕ = real.(ϕ)

    FOM = size(ϕ, 1)
    CT = Complex{T}
    λ = zeros(CT, solver.nev * 2)
    eigenvectors = Array{CT}(undef, FOM, 2, solver.nev * 2)

    for i in 1:(solver.nev)
        ω2i = ω2[i]
        real_part = -0.5 * (solver.α + solver.β * ω2i)
        discriminant = real_part^2 - ω2i
        if discriminant < 0
            imag_part = sqrt(-discriminant)
            λ[2*i-1] = complex(real_part, imag_part)
            λ[2*i]   = complex(real_part, -imag_part)
        else
            delta = sqrt(discriminant)
            λ[2*i-1] = complex(real_part + delta, 0.0)
            λ[2*i]   = complex(real_part - delta, 0.0)
        end

        ϕ_i = view(ϕ, :, i)
        norm_sq = dot(ϕ_i, mass, ϕ_i)
        ϕ_i ./= sqrt(norm_sq)

        eigenvectors[:, 1, 2i-1] .= ϕ_i
        eigenvectors[:, 1, 2i]   .= ϕ_i
        eigenvectors[:, 2, 2i-1] .= λ[2i-1] * ϕ_i
        eigenvectors[:, 2, 2i]   .= λ[2i] * ϕ_i
    end
    return λ, eigenvectors
end

end # module MORFEArpackExt
```

**Critical correctness notes for the extension:**

1. `ArpackEigensolver.solve` reshapes `(ORD*FOM, nev) → (FOM, ORD, nev)` — do not drop this.
2. `ArpackEigensolver.solve` mutates `solver.nev` and `solver.eigenvalues` — required by `solve_left`.
3. `ArpackEigensolver.solve_left` uses **per-eigenvalue sigma-shift** `eigs(A', B'; sigma=conj(λᵢ), which=:LM, nev=1)` in a loop — it is NOT a simple adjoint solve `eigs(A', B'; which=:SM)`.
4. `StructureModalDampingEigensolver` operates on the **second-order** system (mass, stiffness directly), not the first-order companion form.
5. `MorfeEigensolver.solve/solve_left` **remain in `src/`** unchanged: they call `generalised_eigenpairs`, so they will fail via the `Eigensolvers` stub when Arpack is not loaded — no additional stub needed.

**Dispatch correctness:** Julia resolves method ambiguity by specificity. When the extension adds `MORFE.Eigenproblems.solve(model, solver::ArpackEigensolver)`, this method has the same signature as the stub in core and **overwrites it by loading order** — the extension method is found first by the Julia method table once the extension is active.

---

## Step 3 — Pardiso extension: modify three `src/` files

The Pardiso coupling has three touch-points in the current code:
1. `CohomologicalEquations.jl:93` — `using Pardiso: AbstractPardisoSolver, MKLPardisoSolver, solve as pardiso_solve`
2. `SolverResources.jl:87` — `pardiso::Union{Nothing, AbstractPardisoSolver}` field type
3. `CohomologicalSolver.jl:16` — `_sparse_solve(ps::AbstractPardisoSolver, ::Any, A, B)` dispatch

### 3a. `src/ParametrisationMethod/CohomologicalEquations/CohomologicalEquations.jl`

**Remove** line 93:
```julia
using Pardiso: AbstractPardisoSolver, MKLPardisoSolver, solve as pardiso_solve
```

**Add** two extension-hook functions before the `include("SolverResources.jl")` line:

```julia
# Extension hooks: overridden by ext/MORFEPardisoExt.jl when Pardiso is loaded.
_try_build_pardiso_solver() = nothing
_pardiso_solve(ps, A, B) =
    error("Pardiso solver object present but MORFEPardisoExt not active — internal error.")

export _try_build_pardiso_solver, _pardiso_solve
```

### 3b. `src/ParametrisationMethod/CohomologicalEquations/SolverResources.jl`

**Line 87**: change field type:
```julia
# Before:
pardiso::Union{Nothing, AbstractPardisoSolver}

# After:
pardiso::Any   # Nothing, or an AbstractPardisoSolver when Pardiso ext is loaded
```

**Lines 104–122**: Replace the three-part Pardiso constructor block:
```julia
# Before (lines 104–118):
ps = nothing
try
    ps = MKLPardisoSolver()
catch
end
if ps === nothing
    try
        ps = Pardiso.PardisoSolver()
    catch
    end
end
if ps === nothing
    @warn "Neither MKL Pardiso nor open-source Pardiso is available. " *
          "Falling back to KLU (SuiteSparse) for the sparse cohomological solve."
end

# After (single line):
ps = _try_build_pardiso_solver()
```

### 3c. `src/ParametrisationMethod/CohomologicalEquations/CohomologicalSolver.jl`

**Line 16**: Replace typed Pardiso dispatch:
```julia
# Before:
_sparse_solve(ps::AbstractPardisoSolver, ::Any, A, B) = pardiso_solve(ps, A, B)

# After (duck-typed — routes through the hook):
_sparse_solve(ps, ::Any, A, B) = _pardiso_solve(ps, A, B)
```

**Dispatch order is preserved** because Julia picks the most specific method:
- `ps === nothing` → `_sparse_solve(::Nothing, klu_cache::Ref{Any}, ...)` (KLU path — more specific)
- `ps !== nothing` → `_sparse_solve(ps, ::Any, ...)` (duck-typed Pardiso hook)
- `ps === nothing, klu_cache === nothing` → `_sparse_solve(::Nothing, ::Nothing, ...)` (dense fallback)

The two `Nothing`-typed overloads remain more specific than `_sparse_solve(ps, ...)` for `Nothing`
inputs, so dispatch is correct.

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

**Remove** line 16: `using Plots`

**Replace** the bodies of `plot_invariance_convergence`, `_plot_convergence`, and
`_reference_line_params` with a single stub (keep the function signatures):

```julia
function plot_invariance_convergence(results; kwargs...)
    error(
        "plot_invariance_convergence requires Plots.jl.\n" *
        "Load it with `using Plots` to activate the MORFE extension."
    )
end

# _plot_convergence and _reference_line_params are private helpers called only by
# plot_invariance_convergence. They move to ext/MORFEPlotsExt.jl; no stubs needed here.
```

**Keep unchanged**: `invariance_error_norms`, `invariance_error_convergence`,
all `_invariance_error_at!`, `_sample_z!`, `_log_log_regression`, `_jvp_last_block!`
(they use only `LinearAlgebra`, `Random`, `Statistics` — all core deps).

### 4b. Create `ext/MORFEPlotsExt.jl`

```julia
module MORFEPlotsExt

using MORFE
using MORFE.InvarianceError
using Plots
using Statistics: median

# Move _reference_line_params and _plot_convergence verbatim from
# src/Validation/InvarianceError.jl, then override the stub:

function _reference_line_params(results, get_radii, max_order)
    # ... verbatim copy ...
end

function _plot_convergence(results, x_axis, show_state_errors, show_regression, title)
    # ... verbatim copy ...
end

function MORFE.InvarianceError.plot_invariance_convergence(
    results;
    x_axis::Symbol = :both,
    show_state_errors::Bool = true,
    show_regression::Bool = false,
    title::AbstractString = "Invariance error convergence",
)
    isempty(results) && error("results is empty")
    x_axis in (:both, :full, :master) ||
        error("x_axis must be :both, :full, or :master")
    if x_axis == :both
        return (
            full   = _plot_convergence(results, :full, show_state_errors, show_regression, title),
            master = _plot_convergence(results, :master, show_state_errors, show_regression, title),
        )
    else
        return _plot_convergence(results, x_axis, show_state_errors, show_regression, title)
    end
end

end # module MORFEPlotsExt
```

---

## Step 5 — Gmsh extension

The three FEMUtility modules (`AbaqusToGmsh`, `ComsolToGmsh`, `GmshToComsol`) are
in `src/FEMUtility/` but are **not currently included** by `src/MORFE.jl`. They need
stubs so the names are exported from MORFE before the extension loads.

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

Add exports:
```julia
# FEMUtility (Gmsh extension must be loaded for these to work)
export abaqus_to_gmsh, abaqus_to_gmsh_linear,
       comsol_to_gmsh, comsol_to_gmsh_linear,
       gmsh_to_comsol
```

### 5c. Create `ext/MORFEGmshExt.jl`

```julia
module MORFEGmshExt

using MORFE
using MORFE.FEMUtility
using Gmsh
using Printf

# The implementation files stay in src/FEMUtility/ — compiled only when Gmsh is available.
_src(f) = joinpath(pkgdir(MORFE), "src", "FEMUtility", f)
include(_src("AbaqusToGmsh.jl"))
include(_src("ComsolToGmsh.jl"))
include(_src("GmshToComsol.jl"))

# Override stubs
MORFE.FEMUtility.abaqus_to_gmsh(args...; kw...)        = AbaqusToGmsh.abaqus_to_gmsh(args...; kw...)
MORFE.FEMUtility.abaqus_to_gmsh_linear(args...; kw...) = AbaqusToGmsh.abaqus_to_gmsh_linear(args...; kw...)
MORFE.FEMUtility.comsol_to_gmsh(args...; kw...)        = ComsolToGmsh.comsol_to_gmsh(args...; kw...)
MORFE.FEMUtility.comsol_to_gmsh_linear(args...; kw...) = ComsolToGmsh.comsol_to_gmsh_linear(args...; kw...)
MORFE.FEMUtility.gmsh_to_comsol(args...; kw...)        = GmshToComsol.gmsh_to_comsol(args...; kw...)

end # module MORFEGmshExt
```

---

## Step 6 — BifurcationKit extension (new)

BifurcationKit.jl is not yet in the codebase. This extension exposes the reduced
dynamics `R` produced by `solve_cohomological_problem` as a BifurcationKit
`BifFunction`, enabling continuation analysis (FRC, bifurcation diagrams) directly
from the ROM.

### 6a. Add stub to `src/MORFE.jl` (or a new `src/Validation/ReducedDynamicsPostprocess.jl`)

The preferred location is a new source file `src/Validation/BifurcationKitInterface.jl`
included from `src/MORFE.jl`.

**`src/Validation/BifurcationKitInterface.jl`**:
```julia
module BifurcationKitInterface

using ..ParametrisationMethod: ReducedDynamics
using ..Polynomials: DensePolynomial, evaluate

export make_bk_problem

"""
    make_bk_problem(R::ReducedDynamics; bifparam_index, kwargs...)

Wrap the reduced dynamics `R` as a BifurcationKit-compatible problem.
Requires BifurcationKit.jl to be loaded.
"""
function make_bk_problem(R::ReducedDynamics; kwargs...)
    error(
        "make_bk_problem requires BifurcationKit.jl.\n" *
        "Load it with `using BifurcationKit` to activate the MORFE extension."
    )
end

end # module
```

**`src/MORFE.jl`** additions:
```julia
include("Validation/BifurcationKitInterface.jl")
using .BifurcationKitInterface
export make_bk_problem
```

### 6b. Create `ext/MORFEBifurcationKitExt.jl`

```julia
module MORFEBifurcationKitExt

using MORFE
using MORFE.BifurcationKitInterface
using MORFE.ParametrisationMethod: ReducedDynamics
using MORFE.Polynomials: evaluate
using BifurcationKit

"""
    make_bk_problem(R::ReducedDynamics; bifparam_index, u0=nothing, kwargs...)

Build a `BifurcationKit.BifFunction` wrapping `ż = R(z)`.

- `bifparam_index` (Int, required): which component of `z` is the bifurcation
  parameter (continuation variable). This component is held fixed by BifurcationKit.
- `u0`: initial point in reduced coordinates; defaults to zero vector.
- `kwargs`: forwarded to `BifurcationKit.BifFunction`.

The returned object is ready for `BifurcationKit.continuation`.
"""
function MORFE.BifurcationKitInterface.make_bk_problem(
    R::ReducedDynamics;
    bifparam_index::Int,
    u0 = nothing,
    kwargs...,
)
    ROM = Base.size(R)           # number of master modes (NVAR includes ext vars)
    NVAR = size(R.poly.coefficients, 1)

    # f(z, p) = R(z) where p is the scalar bifurcation parameter embedded at
    # component bifparam_index of z.
    f = (z, p) -> begin
        z_full = copy(z)
        z_full[bifparam_index] = p
        return evaluate(R.poly, z_full)
    end

    # Dense Jacobian via finite differences (BifurcationKit default); analytic
    # Jacobian can be added later by differentiating the polynomial.
    z0 = isnothing(u0) ? zeros(ComplexF64, NVAR) : u0

    return BifurcationKit.BifFunction(f, nothing; kwargs...)
end

end # module MORFEBifurcationKitExt
```

> **Design note**: the `make_bk_problem` API is intentionally minimal. BifurcationKit
> has many continuation options (codim-1, shooting, etc.). Future iterations can add
> `make_bk_shooting_problem`, `make_bk_collocation_problem`, etc. as the interface matures.
> Keeping the extension thin and the stub simple means the API can evolve without
> breaking core MORFE.

---

## Step 7 — Create `MORFEFerrite.jl` as a separate GitHub repository

This mirrors the GridapGmsh → Gridap relationship. MORFE knows nothing
about MORFEFerrite; MORFEFerrite depends on MORFE.

### New repo structure: `github.com/<org>/MORFEFerrite.jl`

```
MORFEFerrite.jl/
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
uuid = "<generate with: using UUIDs; UUIDs.uuid4()>"
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

Copy verbatim from `demo/Ferrite/ferrite_assembly.jl`, with:
1. Remove the top-level `using` statements (handled by `MORFEFerrite.jl`)
2. Remove `import MORFE` (already in scope via `using MORFE`)
3. All `MORFE.fem_elements(...)` qualifications can be kept or dropped since
   `using MORFE` imports the interface functions.

### `test/runtests.jl`

```julia
using Test, MORFE, MORFEFerrite, Ferrite

@testset "MORFEFerrite" begin
    @test FerriteGeometricNonlinearity{2} <: FEMMultilinearMap{2}
    @test FerriteGeometricNonlinearity{3} <: FEMMultilinearMap{2}
end
```

---

## Step 8 — Update demos

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

Remove `demo/Ferrite/ferrite_assembly.jl` (content moved to companion package).

### `demo/Gridap/demo_mechanical_problem.jl`

No structural change needed:
- `using Arpack` now triggers `MORFEArpackExt` — correct
- `using Gmsh` triggers `MORFEGmshExt` (harmless if FEMUtility not used by this demo)

### `demo/FEMUtility/` demos

Any demo calling `abaqus_to_gmsh` etc. must `using Gmsh` (before or after `using MORFE` —
the extension activates either way). No other change needed.

### `demo/ParametricCantileverBeam/parametric_beam_demo.jl`

This demo uses `FerriteGeometricNonlinearity` from `demo/Ferrite/ferrite_assembly.jl`
(or the parametric version from `demo/ParametricCantileverBeam/parametric_assembly.jl`).
After the refactoring, update the `include` or `using` to point to `MORFEFerrite`.
Add `using Arpack, LinearMaps` to trigger the eigensolver extension.

---

## Step 9 — `Manifest.toml`

After editing `Project.toml`, regenerate:

```julia
using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()
```

Commit the new `Manifest.toml`. It will be much leaner (many fewer pinned packages).

---

## File change summary

| File | Action |
|------|--------|
| `Project.toml` | Full rewrite (Step 1) |
| `src/SpectralDecomposition/Eigensolvers.jl` | Remove `using Arpack/LinearMaps`; stub `generalised_eigenpairs` |
| `src/SpectralDecomposition/Eigenproblems.jl` | Remove `using Arpack`; stub `ArpackEigensolver.solve/solve_left` and `StructureModalDampingEigensolver.solve(mass,stiffness,solver)` |
| `src/SpectralDecomposition/JordanChain.jl` | **Unchanged** (no Arpack) |
| `src/SpectralDecomposition/PropagateEigenmodes.jl` | **Unchanged** (no Arpack) |
| `src/ParametrisationMethod/CohomologicalEquations/CohomologicalEquations.jl` | Remove `using Pardiso`; add `_try_build_pardiso_solver` and `_pardiso_solve` hooks |
| `src/ParametrisationMethod/CohomologicalEquations/SolverResources.jl` | `pardiso::Any`; call hook instead of constructors |
| `src/ParametrisationMethod/CohomologicalEquations/CohomologicalSolver.jl` | Duck-typed `_sparse_solve` dispatch |
| `src/Validation/InvarianceError.jl` | Remove `using Plots`; stub `plot_invariance_convergence`; private helpers move to ext |
| `src/Validation/BifurcationKitInterface.jl` | **New** stub module |
| `src/FEMUtility/FEMUtility.jl` | **New** stub module |
| `src/MORFE.jl` | Add includes/usings/exports for `FEMUtility` and `BifurcationKitInterface` |
| `ext/MORFEArpackExt.jl` | **New** extension (Arpack + LinearMaps trigger) |
| `ext/MORFEPardisoExt.jl` | **New** extension |
| `ext/MORFEPlotsExt.jl` | **New** extension |
| `ext/MORFEGmshExt.jl` | **New** extension |
| `ext/MORFEBifurcationKitExt.jl` | **New** extension |
| `demo/Ferrite/ferrite_assembly.jl` | **Delete** (content → `MORFEFerrite.jl` repo) |
| `demo/Ferrite/demo_mechanical_problem.jl` | Update imports (Step 8) |

---

## Verification

### 1. Core loads without optional deps

```julia
# In a clean environment with only core [deps] installed:
using MORFE
# Must load without error. Arpack, Ferrite, Pardiso, Plots, Gmsh, BifurcationKit NOT required.
```

### 2. Stubs give helpful errors

```julia
using MORFE

# Arpack stubs
MORFE.Eigensolvers.generalised_eigenpairs(A, B; nev=5)
# Expected: ErrorException "requires Arpack.jl and LinearMaps.jl..."

ep = solve_eigenproblem(model; solver=ArpackEigensolver(4))
# Expected: ErrorException "requires Arpack.jl and LinearMaps.jl..."

ep = solve_eigenproblem(stiffness, mass; solver=StructureModalDampingEigensolver(4, 0.0, 0.0))
# Expected: ErrorException "requires Arpack.jl and LinearMaps.jl..."

# Plots stub
plot_invariance_convergence([])
# Expected: ErrorException "requires Plots.jl..."

# BifurcationKit stub
make_bk_problem(R; bifparam_index=1)
# Expected: ErrorException "requires BifurcationKit.jl..."
```

### 3. Extensions activate automatically

```julia
# Arpack extension:
using Arpack, LinearMaps, MORFE
ep = solve_eigenproblem(model; solver=ArpackEigensolver(4))   # must work
W, R = solve_cohomological_problem(...)

# Pardiso extension (on machines with Pardiso):
using Pardiso, MORFE
# SparseLinearSolverState constructor must pick up the Pardiso handle

# Plots extension:
using Plots, MORFE
plots = plot_invariance_convergence(results)   # must return a Plots object

# Gmsh extension:
using Gmsh, MORFE
abaqus_to_gmsh("mesh.inp", "mesh.msh")        # must work

# BifurcationKit extension:
using BifurcationKit, Arpack, LinearMaps, MORFE
prob = make_bk_problem(R; bifparam_index=1)
```

### 4. DefaultEigensolver works without extensions

```julia
using MORFE   # no Arpack loaded
ep = solve_eigenproblem(model; solver=DefaultEigensolver())
# Must work — DefaultEigensolver uses only LinearAlgebra.eigen
```

### 5. MORFEFerrite companion package

```julia
# Before registration:
Pkg.add(url = "https://github.com/<org>/MORFEFerrite.jl")
using MORFEFerrite
@assert FerriteGeometricNonlinearity{2} <: FEMMultilinearMap{2}
```

### 6. Test suite

```bash
GROUP=tests julia --project test/runtests.jl    # core tests, no optional deps
GROUP=demos julia --project test/runtests.jl    # demo tests; needs Arpack in [extras]
GROUP=all   julia --project test/runtests.jl    # everything
```

**Note on `test/SpectralDecomposition/test_eigenproblems.jl`**: This test currently
calls `ArpackEigensolver` and `StructureModalDampingEigensolver`. After the refactoring,
it must `using Arpack, LinearMaps` at the top to trigger the extension. The test runner
(`runtests.jl`) must ensure Arpack is available (it is, via `[extras]/[targets]`).

---

## Notes and trade-offs

- **`KLU` stays hard**: it is on the critical path of `CohomologicalSolver.jl` with
  symbolic-factor caching. The install cost is a single artifact binary via
  BinaryBuilder — negligible.

- **`DefaultEigensolver` always available**: the dense `LinearAlgebra.eigen` path
  is correct for small FOM (< ~5000 DOF) and requires no extension. Users with
  large sparse systems must `using Arpack, LinearMaps`.

- **`MorfeEigensolver` pre-existing bug**: the no-argument constructor
  `MorfeEigensolver()` calls `new(nothing, nothing)` but the struct has 3 fields
  (`nev`, `shift`, `eigenvalues`). This fails at runtime. It is pre-existing and
  out of scope for this PR — note it in the commit message.

- **Extension loading order**: Julia activates `MORFEArpackExt` when **both**
  `Arpack` and `LinearMaps` are loaded. Order relative to `MORFE` does not matter.

- **BifurcationKit API maturity**: the `make_bk_problem` stub is intentionally
  thin — BifurcationKit has many problem types and continuation options. The extension
  provides a first-pass wrapper; the API will evolve as use cases solidify.

- **No `MORFEGridap` package**: the Gridap demo uses closure-based `MultilinearMap`
  (not the `FEMMultilinearMap` interface), requiring no companion package.

- **`test/FEMUtility/test_gmsh_to_comsol.jl`** is NOT in `runtests.jl`. It uses
  `Gmsh` directly (not MORFE functions) and remains a standalone script; no changes needed.
