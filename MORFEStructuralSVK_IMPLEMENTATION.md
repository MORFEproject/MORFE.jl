# MORFEStructuralSVK — Implementation Specification

Companion to `HIGH_LEVEL_API_PLAN.md`. That document gives the rationale; this
one gives the literal content of every file. Execute steps in order; each step
ends with a verification and a commit. **`src/` is frozen** — `git diff --stat
src/` must be empty at every commit. All code below was derived from the
verified pipelines in `examples/01_clamped_beam_ferrite/main.jl` ("ex01") and
`examples/03_arch_comsol_wedge/main.jl` ("ex03"); where this spec and those
files disagree, the example files win — report the discrepancy instead of
guessing.

Branch: `git checkout -b structural-svk` after committing any WIP.

---

## Step 1 — Registration and skeleton

### 1.1 Root `Project.toml`

- `[weakdeps]`: add FerriteGmsh. Get the UUID by running
  `julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("FerriteGmsh"); println(Pkg.project().dependencies)'`
  — wait, that prints the temp project's deps; instead read the UUID from the
  output of `Pkg.add` or from `~/.julia/packages/FerriteGmsh/*/Project.toml`.
  Do not invent it.
- `[compat]`: `FerriteGmsh = "<resolved major>.<minor>"` (the version the temp env installs).
- `[extensions]`: add `MORFEStructuralSVK = ["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps"]`.

### 1.2 `ext/MORFEStructuralSVK.jl` (complete file)

```julia
"""
MORFEStructuralSVK — high-level UI for St. Venant-Kirchhoff structural models
with the Ferrite backend: mesh → `mechanical_model` → `parametrise` → ROM,
autonomous or with near-resonant harmonic forcing.

Access:

    using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps
    SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)

(Direct `MORFE.parametrise` access would require stubs in `src/`, which is
intentionally avoided; see HIGH_LEVEL_API_PLAN.md.)
"""
module MORFEStructuralSVK

using MORFE: MORFE
using MORFE
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using LinearAlgebra, SparseArrays, Serialization, Printf
using StaticArrays

_svk(file) = joinpath(@__DIR__, "StructuralSVK", file)
include(_svk("types.jl"))
include(_svk("rayleigh_solver.jl"))
include(_svk("mechanical_model.jl"))
include(_svk("parametrise.jl"))
include(_svk("postprocess.jl"))

end
```

KNOWN RISK: `using StaticArrays` — an extension may only depend on its parent's
deps and its triggers. StaticArrays IS a hard dep of MORFE, so this resolves;
if precompilation complains, replace with `using MORFE.StaticArrays`.

### 1.3 Create `ext/StructuralSVK/{types,rayleigh_solver,mechanical_model,parametrise,postprocess}.jl`

Empty placeholder comments are fine for this step.

### 1.4 Verify + commit

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); using MORFE'
julia --project -e '
  using Pkg; Pkg.activate(temp=true); Pkg.develop(path=".");
  Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps"])
  using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps
  @assert Base.get_extension(MORFE, :MORFEStructuralSVK) !== nothing
  println("EXTENSION OK")'
git diff --stat src/    # empty
git add -A && git commit -m "MORFEStructuralSVK: extension skeleton"
```

---

## Step 2 — `ext/StructuralSVK/types.jl` (complete file)

```julia
"""
    SVKMaterial(; E, ν, ρ)

St. Venant-Kirchhoff material: Young's modulus `E`, Poisson ratio `ν`,
density `ρ`. Lamé constants `λ`, `μ` are derived. The SVK model generates
quadratic and cubic nonlinearities in the displacement — nothing higher.
"""
struct SVKMaterial{T}
    E::T
    ν::T
    ρ::T
    λ::T
    μ::T
end
function SVKMaterial(; E, ν, ρ)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    return SVKMaterial(promote(E, ν, ρ, λ, μ)...)
end

"""
    RayleighDamping(; α, β)

Rayleigh damping `C = α M + β K`.
"""
struct RayleighDamping{T}
    α::T
    β::T
end
RayleighDamping(; α, β) = RayleighDamping(promote(α, β)...)

"""
    HarmonicForcing(; mode, amplitude, Ω = nothing)

Harmonic load `f(t) = amplitude · M·ϕ_mode · cos(Ω t)`: shaped like mode
`mode`, oscillating at that same mode's natural frequency unless `Ω` is given.
`mode` must be one of the `master` mode pairs passed to `parametrise`.
"""
struct HarmonicForcing{T}
    mode::Int
    amplitude::T
    Ω::Union{Nothing, T}
end
HarmonicForcing(; mode, amplitude, Ω = nothing) =
    HarmonicForcing(mode, amplitude, Ω === nothing ? nothing : convert(typeof(amplitude), Ω))

"""
Assembled second-order mechanical model `M ü + C u̇ + K u = f_nl(u) (+ forcing)`,
restricted to free DOFs, with a lazy factory for the FEM nonlinear terms:
`term_factory(degree, max_cols)` → `FEMMultilinearMap`.
"""
struct AssembledMechanicalModel{TK, TM, TC, F, MAT, DMP}
    K::TK
    M::TM
    C::TC
    term_factory::F
    nonlinear_degrees::Tuple{Vararg{Int}}
    material::MAT
    damping::DMP
    info::NamedTuple
end

function Base.show(io::IO, ::MIME"text/plain", m::AssembledMechanicalModel)
    println(io, "AssembledMechanicalModel ($(m.info.backend))")
    println(io, "  free DOFs : $(m.info.n_dofs) (of $(m.info.n_dofs_total))")
    println(io, "  material  : SVK  E=$(m.material.E)  ν=$(m.material.ν)  ρ=$(m.material.ρ)")
    print(io,   "  damping   : Rayleigh  α=$(m.damping.α)  β=$(m.damping.β)")
end

"""
Invariant-manifold ROM returned by `parametrise`. Fields: `W` (parametrisation),
`R` (reduced dynamics), `eigenvalues`, `master` (mode pairs), `order`,
`forcing` (`nothing` or `HarmonicForcing`), `info` (NamedTuple).
"""
struct InvariantManifoldROM{TW, TR, T, FRC}
    W::TW
    R::TR
    eigenvalues::Vector{Complex{T}}
    master::Vector{Int}
    order::Int
    forcing::FRC
    info::NamedTuple
end

function Base.show(io::IO, ::MIME"text/plain", rom::InvariantManifoldROM)
    ROM = 2 * length(rom.master)
    println(io, "InvariantManifoldROM")
    println(io, "  master pairs : $(rom.master)  (ROM = $ROM, N_EXT = $(rom.info.N_EXT))")
    for p in rom.master
        λ = rom.eigenvalues[2p - 1]
        println(io, "    pair $p: λ = $λ   (f = $(abs(λ) / 2π))")
    end
    if rom.forcing === nothing
        println(io, "  forcing      : none (autonomous)")
    else
        println(io, "  forcing      : mode $(rom.forcing.mode), amplitude $(rom.forcing.amplitude), Ω = $(rom.info.Ω)")
    end
    println(io, "  order        : $(rom.order)   ($(rom.info.n_monomials) monomials)")
    print(io,   "  solve time   : $(round(rom.info.solve_time_s; digits = 2)) s")
end
```

Verify (temp env as Step 1.4): construct `SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3)`
and check `λ ≈ 160e3*0.22/((1.22)*(0.56))`; construct `SVK.HarmonicForcing(mode = 1, amplitude = 0.03)`.
Commit: `"MORFEStructuralSVK: types"`.

---

## Step 3 — `ext/StructuralSVK/rayleigh_solver.jl` (complete file)

This is ex01 §5 (lines 118–175) with ONE change: the struct is renamed
`Mechanical_Problem_Solver` → `RayleighEigenSolver`. Copy the function bodies
from the example file character-for-character (they are the verified reference;
do NOT retype from this spec if the example differs):

```julia
"""
    RayleighEigenSolver <: AbstractEigensolver

Solves the undamped eigenproblem K ϕ = ω² M ϕ and recovers the damped
eigenvalues λ = ω(-ξ ± i√(1-ξ²)) using Rayleigh damping ξ = ½(α/ω + βω).
(Promoted from examples/01_clamped_beam_ferrite — see that file's history.)
"""
mutable struct RayleighEigenSolver <: AbstractEigensolver
    right_eig_result::Union{Nothing, Matrix}
    eigenvalues::Union{Nothing, Vector}
    nev::Int
    α::Float64
    β::Float64
end

function MORFE.Eigenproblems.solve(model::NDOrderModel, solver::RayleighEigenSolver)
    # ← body of ex01's solve(model, ::Mechanical_Problem_Solver), lines ~132–159, verbatim
end

function MORFE.Eigenproblems.solve_left(model::NDOrderModel, solver::RayleighEigenSolver)
    # ← body of ex01's solve_left, lines ~161–175, verbatim
end
```

NOTE: after Step 7 renames the example file, the source of this copy is
`examples/01_clamped_beam_ferrite/low_level.jl`. Do the copy NOW from `main.jl`
(it still has that name at this step).

Verify: temp env; `SVK.RayleighEigenSolver(nothing, nothing, 10, 0.1, 0.1)` constructs.
Commit: `"MORFEStructuralSVK: Rayleigh eigensolver (from example 01)"`.

---

## Step 4 — `ext/StructuralSVK/mechanical_model.jl` (complete file)

ex01 §§1–4 (lines 38–102) wrapped as a function. The only new logic is
`_refshape`.

```julia
# Ferrite cells are parametrised by their reference shape:
# `AbstractCell{refshape}`. Confirm against the installed Ferrite source
# (grep "abstract type AbstractCell" in the Ferrite package directory);
# if the parameter position differs, adapt and note it in the commit message.
_refshape(::Ferrite.AbstractCell{RS}) where {RS} = RS

"""
    mechanical_model(grid::Ferrite.Grid; material, damping, dirichlet,
                     fe_order = 2, quad_order = fe_order + 1)
    mechanical_model(mesh_path::AbstractString; kwargs...)

Build an `AssembledMechanicalModel` (K, M, C on free DOFs + lazy SVK
nonlinearity factory) from a Ferrite grid or a GMSH mesh file. `dirichlet`
names the facetset that is fully clamped (all three displacement components).
"""
function mechanical_model(grid::Ferrite.Grid;
        material::SVKMaterial,
        damping::RayleighDamping,
        dirichlet::String,
        fe_order::Int = 2,
        quad_order::Int = fe_order + 1)
    RefShape = _refshape(Ferrite.getcells(grid, 1))
    ip = Lagrange{RefShape, fe_order}()^3
    qr = QuadratureRule{RefShape}(quad_order)
    cv = CellValues(qr, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, dirichlet), (x, t) -> zeros(3), [1, 2, 3]))
    close!(ch)
    update!(ch, 0.0)

    K_full = allocate_matrix(dh)
    M_full = allocate_matrix(dh)
    MORFE.ferrite_assemble_KM!(K_full, M_full, dh, cv, material.λ, material.μ, material.ρ)

    free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
    free_to_local = Dict(d => i for (i, d) in enumerate(free))
    n_free = length(free)

    K = K_full[free, free]
    M = M_full[free, free]
    C = damping.α * M + damping.β * K

    factory(deg::Int, max_cols::Int) = MORFE.ferrite_nonlinearity(deg, dh, cv,
        free_to_local, n_free, material.λ, material.μ; max_unique_cols = max_cols)

    return AssembledMechanicalModel(K, M, C, factory, (2, 3), material, damping,
        (n_dofs = n_free, n_dofs_total = ndofs(dh), backend = "Ferrite",
            fe_order = fe_order, quad_order = quad_order, dirichlet = dirichlet))
end

mechanical_model(mesh_path::AbstractString; kwargs...) =
    mechanical_model(FerriteGmsh.togrid(mesh_path); kwargs...)
```

(`MORFE.ferrite_nonlinearity` and `MORFE.ferrite_assemble_KM!` are the stubs in
`src/FullOrderModel/MultilinearMaps.jl` lines ~341–357, implemented by
`MORFEFerriteExt`, which is active whenever Ferrite is loaded — verified.)

Verify (temp env):

```julia
beam = SVK.mechanical_model(joinpath("examples", "02_clamped_beam_gridap", "clamped_clamped_beam.msh");
    material = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
    damping = SVK.RayleighDamping(α = 5.370828278264171e-3, β = 1 / 53.70828278264171),
    dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
@assert beam.info.n_dofs == 4977     # FOM from ex01's docstring
```

Commit: `"MORFEStructuralSVK: mechanical_model (Ferrite + FerriteGmsh)"`.

---

## Step 5 — `ext/StructuralSVK/parametrise.jl` (complete file)

Merges ex01 §§3, 6–8 with ex03's forcing block (ex03 lines ~112–124 and the
resonance call at ~133). Conventions are ex01's (its eigensolver is the one we
ship): master quantities at columns `1:ROM`, `left_eigenmodes = Y[:, 2, 1:ROM]`
— this looks surprising but is the VERIFIED ex01 behaviour protected by Gate A;
copy it, do not "fix" it.

```julia
"""
    parametrise(m::AssembledMechanicalModel; master = [1], order,
                forcing = nothing, resonance_tol = 0.05, nev = ..., eigensolver = nothing)

Compute the DPIM invariant-manifold ROM. `master` lists physical mode PAIRS
(`[1]` → first conjugate pair, ROM = 2). With `forcing::HarmonicForcing`,
two external states with eigenvalues ±iΩ are appended (N_EXT = 2).
"""
function parametrise(m::AssembledMechanicalModel;
        master::Vector{Int} = [1],
        order::Int,
        forcing::Union{Nothing, HarmonicForcing} = nothing,
        resonance_tol::Real = 0.05,
        nev::Int = max(10, 2 * max(maximum(master), forcing === nothing ? 0 : forcing.mode) + 4),
        eigensolver = nothing)

    @assert master == collect(1:length(master)) "only contiguous leading mode pairs are supported (master = [1], [1, 2], …); got master = $master"
    if forcing !== nothing
        @assert forcing.mode in master "forcing.mode = $(forcing.mode) must be a master mode pair (master = $master): the near-resonant reduction requires the forced mode on the manifold"
    end

    ROM = 2 * length(master)
    N_EXT = forcing === nothing ? 0 : 2
    NVAR = ROM + N_EXT
    mset = all_multiindices_up_to(NVAR, order; min_degree = 1)

    terms = Tuple(m.term_factory(d, length(mset)) for d in m.nonlinear_degrees)

    # ── Eigenproblem (autonomous operator; forcing does not enter) — ex01 §5/§6
    eig_model = NDOrderModel((m.K, m.C, m.M), terms)
    solver = eigensolver === nothing ?
        RayleighEigenSolver(nothing, nothing, nev, Float64(m.damping.α), Float64(m.damping.β)) :
        eigensolver
    t_eig = @elapsed eigenproblem = solve_eigenproblem(eig_model;
        solver = solver, sorter! = (args...) -> nothing)
    (eigenvalues, Y, X) = get_eigenpairs(eigenproblem)

    select_master_modes_by_sorting(eigenproblem, ROM)
    master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
    master_modes = Y[:, 1, 1:ROM]
    left_eigenmodes = Y[:, 2, 1:ROM]

    ORD_model = length(eig_model.linear_terms) - 1   # = 2
    FOM = m.info.n_dofs
    master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
    for r in 1:ROM, k in 1:(ORD_model - 1)
        master_modes_derivatives[:, k, r] .= Y[:, k + 1, r]
    end

    # ── Model with forcing (ex03 lines ~112–124, shape_mode == frequency_mode)
    local Ω = nothing
    if forcing === nothing
        model = eig_model
        ext_eigs = ComplexF64[]
    else
        Ω = forcing.Ω === nothing ? abs(eigenvalues[2 * forcing.mode - 1]) : forcing.Ω
        fv = real((forcing.amplitude / 2) .* (m.M * Y[:, 1, 2 * forcing.mode - 1]))
        # Degree-1 term: multiindex (0,0) in the state derivatives, multiplicity 1
        # in the external state ⇒ f! signature is exactly (res, r). Verified against
        # the 3-arg MultilinearMap constructor (deg = 0 + 0 + 1 = 1, nargs = deg + 2).
        force_term = MultilinearMap((res, r) -> (res .+= fv * sum(r)), (0, 0), 1)
        ext_eigs = ComplexF64[im * Ω, -im * Ω]
        model = NDOrderModel((m.K, m.C, m.M), (terms..., force_term),
            ExternalSystem(Tuple(ext_eigs)))
    end

    # ── Resonance set (scalar tol; signature verified in src/.../Resonance.jl:493)
    resonance_set = resonance_set_from_complex_normal_form_style(
        mset, Vector{ComplexF64}(master_eigenvalues), Float64(resonance_tol);
        external_eigenvalues = ext_eigs)

    conjugate_permutation = reduce(vcat, [[2p, 2p - 1] for p in 1:length(master)])

    t_solve = @elapsed W, R = solve_cohomological_problem(
        model, mset, master_eigenvalues, master_modes, left_eigenmodes, resonance_set;
        master_modes_derivatives = master_modes_derivatives,
        conjugate_permutation = conjugate_permutation)

    return InvariantManifoldROM(W, R, collect(eigenvalues), master, order, forcing,
        (; m.info..., n_monomials = length(mset), N_EXT = N_EXT, Ω = Ω,
            eig_time_s = t_eig, solve_time_s = t_solve))
end
```

Verify (temp env, FAST settings): `parametrise(beam; master = [1], order = 3)`
runs; `parametrise(beam; master = [1], order = 3, forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.01))`
runs and its `rom.info.N_EXT == 2`, `rom.info.Ω ≈ abs(rom.eigenvalues[1])`.
Also assert the error paths: `master = [2]` throws; forced `mode = 2` with
`master = [1]` throws. Commit: `"MORFEStructuralSVK: parametrise (autonomous + harmonic forcing)"`.

---

## Step 6 — `ext/StructuralSVK/postprocess.jl` (complete file)

`real_dynamics`/`print_equations` are ex01 §9 (lines 243–261); `save_rom`
mirrors `examples/common/results_io.jl` BYTE-COMPATIBLY (the gates diff the CSV).
Open `examples/common/results_io.jl` and copy its `save_rom` CSV block and
`write_summary` exactly; do not retype from memory.

```julia
"""
    real_dynamics(rom) -> DensePolynomial

Realified master equation ż₁ in the real pair z₁ = x₁ + i y₁:
Re(c) is the ẋ₁-equation, Im(c) the ẏ₁-equation. External (forcing) states
also come in conjugate ±iΩ pairs, so the same pairwise conjugation map applies.
"""
function real_dynamics(rom::InvariantManifoldROM)
    NVAR = 2 * length(rom.master) + rom.info.N_EXT
    conj_map = [isodd(i) ? i + 1 : i - 1 for i in 1:NVAR]
    return realify(extract_component(rom.R.poly, 1), conj_map)
end

"""
    print_equations(rom; tol = 1e-12, io = stdout)

Print the realified ż₁ monomial table (ex01 §9 format). In the forced case the
trailing exponents belong to the external states (e^{+iΩt}, e^{-iΩt}).
"""
function print_equations(rom::InvariantManifoldROM; tol = 1e-12, io = stdout)
    Rr = real_dynamics(rom)
    header = rom.info.N_EXT == 0 ?
        "(x₁,y₁) exponents : ẋ₁-coeff, ẏ₁-coeff" :
        "(x₁,y₁,ext₊,ext₋) exponents : ẋ₁-coeff, ẏ₁-coeff"
    println(io, "Reduced dynamics ż₁ = ẋ₁ + i·ẏ₁ in real variables:")
    println(io, "  " * header)
    for (m, mi) in enumerate(Rr.multiindex_set.exponents)
        c = Rr.coefficients[m]
        abs(c) > tol && println(io, "  $(Tuple(mi)) : " *
            "$(round(real(c); sigdigits = 6)), $(round(imag(c); sigdigits = 6))")
    end
end

"""
    save_rom(rom, dir)

Write `dir/data/{W.jls, R.jls, R_coefficients.csv}` and `dir/summary.txt`
(format identical to examples/common/results_io.jl).
"""
function save_rom(rom::InvariantManifoldROM, dir::AbstractString)
    data = joinpath(dir, "data"); mkpath(data); mkpath(joinpath(dir, "figures"))
    serialize(joinpath(data, "W.jls"), rom.W)
    serialize(joinpath(data, "R.jls"), rom.R)
    open(joinpath(data, "R_coefficients.csv"), "w") do io
        # ← copy the CSV-writing block of examples/common/results_io.jl save_rom
        #   verbatim, with `R` replaced by `rom.R` (header exp_i, R{i}_re/R{i}_im,
        #   rows skipped when all |c| ≤ 1e-14)
    end
    open(joinpath(dir, "summary.txt"), "w") do io
        println(io, "model: SVK + Ferrite (MORFEStructuralSVK)")
        println(io, "n_dofs: $(rom.info.n_dofs)")
        println(io, "master_pairs: $(rom.master)")
        println(io, "master_eigenvalues: $(rom.eigenvalues[1:2length(rom.master)])")
        println(io, "parametrisation_order: $(rom.order)")
        println(io, "n_monomials: $(rom.info.n_monomials)")
        if rom.forcing === nothing
            println(io, "forcing: none")
        else
            println(io, "forcing: mode=$(rom.forcing.mode) amplitude=$(rom.forcing.amplitude) Omega=$(rom.info.Ω)")
        end
        println(io, "eigenproblem_time_s: $(rom.info.eig_time_s)")
        println(io, "cohomological_solve_time_s: $(rom.info.solve_time_s)")
        println(io, "julia_version: $(VERSION)")
        commit = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end
        println(io, "morfe_commit: $commit")
        println(io, "timestamp: $(time())")
    end
    return nothing
end
```

Verify: order-3 run, `print_equations(rom)` produces ≥1 line, `save_rom`
writes the four files. Commit: `"MORFEStructuralSVK: postprocessing"`.

---

## Step 7 — Showcase `examples/01_clamped_beam_ferrite/main.jl`

```bash
mv examples/01_clamped_beam_ferrite/main.jl examples/01_clamped_beam_ferrite/low_level.jl
```

New `main.jl` (complete file):

```julia
"""
Clamped-clamped beam, St. Venant-Kirchhoff, Ferrite backend — high-level UI.
The fully explicit construction of the same ROM lives in `low_level.jl`;
both produce identical reduced dynamics (enforced by test group structural_svk).
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "MORFE")
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
    Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays"])
end
Pkg.instantiate()

using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps
SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)

beam = SVK.mechanical_model(
    joinpath(@__DIR__, "..", "02_clamped_beam_gridap", "clamped_clamped_beam.msh");
    material  = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
    damping   = SVK.RayleighDamping(α = 5.370828278264171e-3, β = 1 / 53.70828278264171),
    dirichlet = "Dirichlet",
    fe_order  = 2, quad_order = 3)

rom = SVK.parametrise(beam; master = [1], order = 9)

# Near-resonant harmonic forcing (shape = frequency mode = 1) — uncomment to use:
# rom = SVK.parametrise(beam; master = [1], order = 9,
#     forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.03))

println(rom)
SVK.print_equations(rom)
SVK.save_rom(rom, joinpath(@__DIR__, "results"))
println("Results written to $(joinpath(@__DIR__, "results"))")
```

The Rayleigh constants MUST equal `low_level.jl` lines 67–68 (α = 0.5370828278264171/100,
β = 1/(0.5370828278264171*100)) — check, don't trust this spec. Update the
example README (headline = this script; "Harmonic forcing" section: what
`HarmonicForcing` does physically, Ω defaults to the mode's frequency;
"Under the hood" links `low_level.jl`). Update any references to the old
entry-point name: `grep -rn "01_clamped_beam_ferrite/main\|demo_mechanical" examples test docs README.md CLAUDE.md`.

Commit: `"Example 01: high-level showcase via MORFEStructuralSVK; low-level path preserved"`.

---

## Step 8 — Gates A and B

### Gate A — high-level ≡ low-level (autonomous, full order 9)

```bash
julia --project=examples/01_clamped_beam_ferrite -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/01_clamped_beam_ferrite/low_level.jl")'
cp examples/01_clamped_beam_ferrite/results/data/R_coefficients.csv /tmp/R_low.csv
julia --project=examples/01_clamped_beam_ferrite -e '
  include("examples/01_clamped_beam_ferrite/main.jl")'
julia -e 'using DelimitedFiles
  a = readdlm("/tmp/R_low.csv", ',', skipstart = 1)
  b = readdlm("examples/01_clamped_beam_ferrite/results/data/R_coefficients.csv", ',', skipstart = 1)
  size(a) == size(b) || error("row/col mismatch: $(size(a)) vs $(size(b))")
  d = maximum(abs.(a .- b) ./ max.(abs.(a), 1e-12))
  println("Gate A max rel dev: ", d)
  d < 1e-10 || error("GATE A FAILED - STOP")'
```

Failure → bug in Step 5's eigen bookkeeping (`1:ROM` indices, `conjugate_permutation`)
or Step 6's CSV block. Fix there only.

### Gate B — forced(amplitude = 0) ≡ autonomous (order 5, fast)

Run in the temp env (script, full code):

```julia
rom0 = SVK.parametrise(beam; master = [1], order = 5)
romf = SVK.parametrise(beam; master = [1], order = 5,
    forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.0))
exps0 = rom0.R.poly.multiindex_set.exponents     # SVectors, NVAR = 2
expsf = romf.R.poly.multiindex_set.exponents     # SVectors, NVAR = 4
lookup = Dict(e => i for (i, e) in enumerate(expsf))
maxdev = 0.0
for (i, e) in enumerate(exps0)
    ef = [e[1], e[2], 0, 0]                       # same z-exponents, zero ext
    j = get(lookup, typeof(expsf[1])(ef), nothing)
    j === nothing && error("monomial $e not found in forced mset")
    c0 = rom0.R.poly.coefficients[:, i]
    cf = romf.R.poly.coefficients[1:2, j]         # master components only
    maxdev = max(maxdev, maximum(abs.(c0 .- cf) ./ max.(abs.(c0), 1e-12)))
end
println("Gate B max rel dev: ", maxdev)
maxdev < 1e-10 || error("GATE B FAILED - STOP")
```

NOTE: the field access `R.poly.coefficients` (NVAR × L) is the layout used by
`examples/common/results_io.jl` — confirmed there. If `extract_component`
errors guide you otherwise, read `src/Polynomials.jl` (read-only) and adjust
indices, not the tolerance. Failure → Step 5's forcing block (fv, ext_eigs,
resonance `external_eigenvalues`). Fix there only.

Commit (gate scripts may live in `test/StructuralSVK/` already — see Step 9):
`"MORFEStructuralSVK: equivalence gates pass (A: 1e-10 vs low-level; B: zero-amplitude consistency)"`.

---

## Step 9 — Tests

`test/StructuralSVK/test_structural_svk.jl`: Gates A and B logic on a TINY
in-memory grid so the group runs in seconds — no mesh file needed:

```julia
using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps, Test
SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)

grid = generate_grid(Hexahedron, (6, 1, 1), Vec(0.0, 0.0, 0.0), Vec(100.0, 5.0, 5.0))
addfacetset!(grid, "Dirichlet", x -> x[1] ≈ 0.0 || x[1] ≈ 100.0)

beam = SVK.mechanical_model(grid;
    material = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
    damping = SVK.RayleighDamping(α = 1e-3, β = 1e-4),
    dirichlet = "Dirichlet", fe_order = 1, quad_order = 2)

@testset "autonomous vs hand-built" begin
    rom = SVK.parametrise(beam; master = [1], order = 3)
    # Hand-built reference: replicate ex01 §§3–8 inline with the SAME grid and
    # constants (copy the code, using beam.K/.M/.C and beam.term_factory),
    # then compare R coefficients to 1e-10.
end

@testset "zero-amplitude forcing consistency" begin
    # Gate B code from Step 8, order 3, @test maxdev < 1e-10
end

@testset "error paths" begin
    @test_throws AssertionError SVK.parametrise(beam; master = [2], order = 3)
    @test_throws AssertionError SVK.parametrise(beam; master = [1], order = 3,
        forcing = SVK.HarmonicForcing(mode = 2, amplitude = 0.1))
end
```

CAUTION: `fe_order = 1` exercises a code path ex01 never ran (it used order 2).
If `ferrite_nonlinearity` fails on linear hexes, switch the test grid to
`fe_order = 2` with `(3, 1, 1)` cells — adjust, don't fight it.

Wire into `test/runtests.jl` (allowed — only `src/` is frozen) as group
`structural_svk` following the existing `should_run` pattern, and add
`Ferrite`, `FerriteGmsh`, `LinearMaps` to `[extras]` + the `test` target in the
root `Project.toml` (Arpack is already there).

Verify: `GROUP=structural_svk julia --project test/runtests.jl` passes AND
`julia --project -e 'using Pkg; Pkg.test()'` (default groups) still passes.
Commit: `"Add structural_svk test group"`.

---

## Step 10 — Docs and final checklist

1. README.md quickstart: the Step-7 script (shortened: drop the Pkg bootstrap).
2. CLAUDE.md: paragraph on `MORFEStructuralSVK` — location (`ext/StructuralSVK/`),
   trigger packages, `Base.get_extension` access, `HarmonicForcing` semantics,
   and the rule that it must stay behaviour-identical to `low_level.jl` (gates).
3. Website teaser (`website/index.html` code-teaser block): badge →
   `implemented for SVK + Ferrite 🚧 FRF continuation`; snippet function names
   aligned with reality (`mechanical_model`, `parametrise(beam; master = [1],
   order = 7, forcing = HarmonicForcing(mode = 1, amplitude = 0.02))`, the
   `get_extension` line). Nothing else on the page.
4. `using JuliaFormatter; format("ext/"); format("test/")`.
5. Final:
   ```bash
   julia --project -e 'using Pkg; Pkg.test()'
   GROUP=structural_svk julia --project test/runtests.jl
   GROUP=examples julia --project test/runtests.jl
   git diff --stat main -- src/        # MUST be empty for the whole branch
   git log --oneline main..HEAD
   ```
   Report for human approval; do not merge or push.

## Verified-against-source facts this spec relies on

| Fact | Where verified |
|---|---|
| `MultilinearMap(f!, (0, 0), 1)`: deg = 1, `f!` takes `(res, r)` | `src/FullOrderModel/MultilinearMaps.jl:231` 3-arg constructor |
| `resonance_set_from_complex_normal_form_style(mset, eigs, tol::Float64; external_eigenvalues)` | `src/ParametrisationMethod/Resonance.jl:493` |
| `solve_eigenproblem(model::NDOrderModel; solver, sorter!, normalizer!)` | `src/SpectralDecomposition/Eigenproblems.jl:388` |
| `get_eigenpairs(ep) → (eigenvalues, eigenmodes, left_eigenmodes)` | `Eigenproblems.jl:500` |
| `select_master_modes_by_sorting(ep, nev)` marks the first `nev` modes | `Eigenproblems.jl:521` |
| `ExternalSystem(eigenvalues::NTuple)` exists | `src/FullOrderModel/ExternalSystems.jl:102` |
| `ferrite_nonlinearity` / `ferrite_assemble_KM!` stubs exported; implemented in `MORFEFerriteExt` | `MultilinearMaps.jl:341–357`, `ext/MORFEFerriteExt.jl` |
| CSV layout (`exp_i`, `R{i}_re/_im`, skip ≤1e-14) | `examples/common/results_io.jl` |
| Forcing recipe (Ω = abs(λ), fv = re(a/2·M·ϕ), ext_eigs = ±iΩ) | `examples/03_arch_comsol_wedge/main.jl:112–124` |
| `conjugate_permutation` covers master pairs only | ex03 `main.jl:44` |
