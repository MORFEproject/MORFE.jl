# High-Level API Plan — `MORFEStructuralSVK` (SVK material, Ferrite backend)

Goal: collapse `examples/01_clamped_beam_ferrite/main.jl` (≈280 lines) into the
"easiest possible UI", in the spirit of the website code teaser
(`website/index.html`, the `target API 🚧` block). Scope: **St. Venant-Kirchhoff
material, Ferrite backend, Rayleigh damping**, autonomous **or** harmonically
forced — where the forcing is shaped like the mode whose frequency drives it
(the standard near-resonance setting of example 03 with `shape_mode ==
frequency_mode`).

**Hard constraint: `src/` is NOT modified in any way.** The entire layer lives
under `ext/`. The only files outside `ext/`, `examples/`, `test/`, and docs that
may change are the root `Project.toml` (extension registration is not `src/`).

## Consequence of the constraint (acknowledge, then comply)

Package extensions cannot export names into the `MORFE` namespace without a stub
in `src/`. Therefore the user obtains the module with one extra line:

```julia
SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)
```

This is the documented access pattern. (If one-line stubs in `src/` are ever
allowed, `MORFE.parametrise` etc. become direct — note in the module docstring,
do not implement.)

## Target user script (the contract for this plan)

```julia
using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps
SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)

beam = SVK.mechanical_model("clamped_clamped_beam.msh";
    material  = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
    damping   = SVK.RayleighDamping(α = 5.370828278264171e-3, β = 1/53.70828278264171),
    dirichlet = "Dirichlet",        # physical-group name in the mesh
    fe_order  = 2, quad_order = 3)

# Autonomous ROM (backbone):
rom = SVK.parametrise(beam; master = [1], order = 9)

# Forced ROM: harmonic load shaped like mode 1, at mode 1's natural frequency:
rom_f = SVK.parametrise(beam; master = [1], order = 9,
                        forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.03))

println(rom_f)                   # eigenfrequencies, order, #monomials, forcing, timings
SVK.print_equations(rom_f)       # realified ż₁ monomial table
SVK.save_rom(rom_f, joinpath(@__DIR__, "results"))
```

Anything in examples 01/03 not visible in this script becomes extension code.

## Architecture decisions (fixed — do not re-litigate during execution)

| Decision | Choice |
|---|---|
| Name and location | Extension `MORFEStructuralSVK = ["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps"]`, entry file `ext/MORFEStructuralSVK.jl`, implementation in `ext/StructuralSVK/*.jl` include files (mirroring `ext/FEMUtility/` + `MORFEGmshExt`). |
| Why one extension? | Extensions cannot depend on each other; `parametrise` needs the Arpack eigensolver AND Ferrite assembly, so one extension triggers on all four weakdeps. |
| Reuse | With Ferrite loaded, `MORFEFerriteExt` is also active → call the EXISTING public `MORFE.ferrite_nonlinearity` / `MORFE.ferrite_assemble_KM!`. The mechanical eigensolver from example 01 has no public hook → copied into `ext/StructuralSVK/` (do not touch `ext/MORFEArpackExt.jl`). |
| `master` semantics | List of physical mode PAIRS: `master = [1]` → first conjugate pair, `ROM = 2length(master)`. |
| Forcing semantics | `HarmonicForcing(mode, amplitude; Ω = nothing)`: load vector `f(t) = amplitude · M·ϕ_mode · cos(Ωt)` with `Ω` defaulting to the natural frequency of `mode` (`abs(λ_mode)`), exactly the `shape_mode == frequency_mode` case of example 03. Adds `N_EXT = 2` external states with eigenvalues `±iΩ`. One forcing only in this scope (a `Vector{HarmonicForcing}` generalisation is Deferred). |
| Lazy nonlinearity | `ferrite_nonlinearity` needs `max_unique_cols = length(mset)`, known only in `parametrise` (mset depends on `order` AND on `N_EXT`!) → the model stores a factory closure `term_factory(degree, max_cols)`. |
| Teaser's raw-matrix `parametrise(M, C, K, G, H; ...)`, `continuation`, `plot` | OUT OF SCOPE — Deferred. |

## Ground rules

1. Run from repo root; one phase = one commit; stop on failed verification.
2. SciML formatting (`format("ext/")` before each commit); in-place functions end in `!`.
3. Copy working code from examples 01 and 03 instead of rewriting. They are the
   verified references; the refactor must be behaviour-preserving (Phase 5 gates).
4. `git diff --stat src/` must be empty at EVERY commit of this plan — add this
   check to each phase's verification.

---

## Phase 1 — Extension skeleton and registration

1. Root `Project.toml`:
   - `[weakdeps]`: add `FerriteGmsh` (get the UUID by adding FerriteGmsh in a
     temp env and reading its Project.toml — do not guess it).
   - `[compat]`: add FerriteGmsh (the major version the temp env resolves).
   - `[extensions]`: `MORFEStructuralSVK = ["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps"]`.
2. `ext/MORFEStructuralSVK.jl`:
   ```julia
   """
   MORFEStructuralSVK — high-level UI for St. Venant-Kirchhoff structural models
   with the Ferrite backend: mesh → mechanical_model → parametrise → ROM,
   autonomous or with near-resonant harmonic forcing.
   Access: `SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)`.
   """
   module MORFEStructuralSVK

   using MORFE: MORFE
   using MORFE   # exported names: NDOrderModel, ExternalSystem, MultilinearMap, …
   using Ferrite, FerriteGmsh, Arpack, LinearMaps
   using LinearAlgebra, SparseArrays, Serialization, Printf

   _svk(file) = joinpath(@__DIR__, "StructuralSVK", file)
   include(_svk("types.jl"))
   include(_svk("rayleigh_solver.jl"))
   include(_svk("mechanical_model.jl"))
   include(_svk("parametrise.jl"))
   include(_svk("postprocess.jl"))

   end
   ```
   `StaticArrays` is needed for `SVector` (master eigenvalues): it is a hard dep
   of MORFE, so `using MORFE.StaticArrays` if plain `using StaticArrays` fails
   inside the extension (verify at precompile).
3. Create the five near-empty files under `ext/StructuralSVK/` so it precompiles.
4. Verify:
   ```bash
   julia --project -e 'using Pkg; Pkg.instantiate(); using MORFE'
   julia --project -e '
     using Pkg; Pkg.activate(temp=true); Pkg.develop(path=".");
     Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps"])
     using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps
     @assert Base.get_extension(MORFE, :MORFEStructuralSVK) !== nothing
     println("EXTENSION OK")'
   git diff --stat src/   # empty
   ```

---

## Phase 2 — `types.jl` and `rayleigh_solver.jl`

### 2.1 `types.jl`

```julia
"St. Venant-Kirchhoff material. Stores E, ν, ρ and derived Lamé constants λ, μ."
struct SVKMaterial{T}
    E::T; ν::T; ρ::T; λ::T; μ::T
end
SVKMaterial(; E, ν, ρ) = (λ = E*ν/((1+ν)*(1-2ν)); μ = E/(2(1+ν));
    SVKMaterial(promote(E, ν, ρ, λ, μ)...))

"Rayleigh damping C = αM + βK."
struct RayleighDamping{T}
    α::T; β::T
end
RayleighDamping(; α, β) = RayleighDamping(promote(α, β)...)

"""
Harmonic forcing  f(t) = amplitude · M·ϕ_mode · cos(Ωt).
The load is shaped like mode `mode` and oscillates at that same mode's natural
frequency (Ω = |λ_mode|) unless `Ω` is given explicitly.
"""
struct HarmonicForcing{T}
    mode::Int
    amplitude::T
    Ω::Union{Nothing, T}
end
HarmonicForcing(; mode, amplitude, Ω = nothing) = HarmonicForcing(mode, amplitude, Ω)

"Assembled second-order mechanical model with lazy nonlinear-term factory."
struct AssembledMechanicalModel{TK, TM, TC, F, MAT, DMP}
    K::TK; M::TM; C::TC
    term_factory::F
    nonlinear_degrees::Tuple{Vararg{Int}}   # (2, 3) for SVK
    material::MAT
    damping::DMP
    info::NamedTuple
end

"Result of `parametrise`."
struct InvariantManifoldROM{TW, TR, T, FRC}
    W::TW; R::TR
    eigenvalues::Vector{Complex{T}}
    master::Vector{Int}
    order::Int
    forcing::FRC               # ::Nothing or ::HarmonicForcing (with resolved Ω in info)
    info::NamedTuple
end
```

`Base.show` for the ROM: master eigenvalues, order, `info.n_monomials`,
forcing (mode, amplitude, resolved `info.Ω` — or "none"), `info.solve_time_s`.

### 2.2 `rayleigh_solver.jl`

Copy example 01 `main.jl` §5 (lines ~118–175) verbatim: the `mutable struct`
(rename `Mechanical_Problem_Solver` → `RayleighEigenSolver`, keep
`<: AbstractEigensolver`), `MORFE.Eigenproblems.solve`, `MORFE.Eigenproblems.solve_left`.
Change nothing inside the bodies.

Verify: temp-env load; construct `SVK.RayleighEigenSolver(nothing, nothing, 10, 0.1, 0.1)`
and `SVK.HarmonicForcing(mode = 1, amplitude = 0.03)`. `src/` diff empty.

---

## Phase 3 — `mechanical_model.jl`

Transplant example 01 §§1–4 (lines ~43–102). Two methods:

```julia
function mechanical_model(grid::Ferrite.Grid;
        material::SVKMaterial, damping::RayleighDamping,
        dirichlet::String, fe_order::Int = 2, quad_order::Int = fe_order + 1)
    RefShape = Ferrite.getrefshape(getcells(grid, 1))   # verify accessor against installed Ferrite
    ip = Lagrange{RefShape, fe_order}()^3
    qr = QuadratureRule{RefShape}(quad_order)
    cv = CellValues(qr, ip)
    dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, dirichlet), (x, t) -> zeros(3), [1, 2, 3]))
    close!(ch); update!(ch, 0.0)
    # K, M assembly + free-DOF restriction + C = αM + βK: copy example 01 lines 70–82,
    # using MORFE.ferrite_assemble_KM! and material.λ/.μ/.ρ, damping.α/.β
    ...
    factory(deg, max_cols) = MORFE.ferrite_nonlinearity(deg, dh, cv, free_to_local,
        n_free, material.λ, material.μ; max_unique_cols = max_cols)
    return AssembledMechanicalModel(K, M, C, factory, (2, 3), material, damping,
        (n_dofs = n_free, n_dofs_total = ndofs(dh), backend = "Ferrite",
         fe_order = fe_order, quad_order = quad_order, dirichlet = dirichlet))
end

mechanical_model(mesh_path::AbstractString; kwargs...) =
    mechanical_model(FerriteGmsh.togrid(mesh_path); kwargs...)
```

The `RefShape` line is the only new code — confirm the accessor exists in the
installed Ferrite version (grep its source); if not, dispatch on
`eltype(getcells(grid))` with a clear error for unsupported cell types.

Verify: build the model from `examples/02_clamped_beam_gridap/clamped_clamped_beam.msh`,
check `info.n_dofs == 4977` (FOM stated in example 01's docstring). `src/` diff empty.

---

## Phase 4 — `parametrise.jl` and `postprocess.jl`

### 4.1 `parametrise` — autonomous AND forced

References: example 01 §§3, 6–8 (autonomous skeleton) and example 03
`main.jl` lines ~44, ~100–151 (forced pieces). Signature:

```julia
function parametrise(m::AssembledMechanicalModel;
        master::Vector{Int} = [1], order::Int,
        forcing::Union{Nothing, HarmonicForcing} = nothing,
        resonance_tol::Real = 0.05,
        nev::Int = max(10, 2max(maximum(master), forcing === nothing ? 0 : forcing.mode) + 4),
        eigensolver = nothing)
```

Body, in order:
1. `@assert master == collect(1:length(master))` with error
   `"only contiguous leading mode pairs are supported (master = [1], [1,2], …)"`.
   If forcing is given: `@assert forcing.mode in master` — the forced mode must
   be a master mode (near-resonant reduction); error message must say so.
2. Sizes — note the forced difference:
   `ROM = 2length(master)`; `N_EXT = forcing === nothing ? 0 : 2`;
   `NVAR = ROM + N_EXT`;
   `mset = all_multiindices_up_to(NVAR, order; min_degree = 1)`.
3. Terms and eigen solve FIRST (the forcing vector needs eigenvectors):
   `terms = Tuple(m.term_factory(d, length(mset)) for d in m.nonlinear_degrees)`.
   Eigen solve exactly as example 01 lines 177–199 (build a plain
   `NDOrderModel((m.K, m.C, m.M), terms)` just for the eigenproblem — it does
   not need the forcing), `solver = eigensolver === nothing ?
   RayleighEigenSolver(nothing, nothing, nev, m.damping.α, m.damping.β) : eigensolver`,
   then `select_master_modes_by_sorting(eigenproblem, ROM)`, eigenvalues/modes
   at `1:ROM`, `master_modes_derivatives` (lines 201–207).
4. Forced model assembly — copy example 03 lines ~112–124, simplified to one
   force with `shape_mode = frequency_mode = forcing.mode`:
   ```julia
   if forcing === nothing
       model = NDOrderModel((m.K, m.C, m.M), terms)
       ext_eigs = ComplexF64[]
   else
       Ω  = forcing.Ω === nothing ? abs(eigenvalues[2forcing.mode - 1]) : forcing.Ω
       fv = real((forcing.amplitude / 2) .* (m.M * Y[:, 1, 2forcing.mode - 1]))
       force_term = MultilinearMap((res, r) -> (res .+= fv * sum(r)), (0, 0), 1)
       ext_eigs = ComplexF64[im * Ω, -im * Ω]
       model = NDOrderModel((m.K, m.C, m.M), (terms..., force_term),
                            ExternalSystem(Tuple(ext_eigs)))
   end
   ```
   IMPORTANT: keep the `(0, 0)` multiindex and the `1` multiplicity argument of
   `MultilinearMap` exactly as in example 03 — read the `MultilinearMap`
   docstring (read-only) to document what they mean, but do not "improve" them.
5. Resonance set — the forced case passes the external eigenvalues (example 03
   line ~133):
   `resonance_set_from_complex_normal_form_style(mset, master_eigs, resonance_tol; external_eigenvalues = ext_eigs)`.
   Check the positional/keyword form: example 03 passes a per-monomial `tol_vec`
   where example 01 passes a scalar — read the method signatures in
   `src/ParametrisationMethod/Resonance.jl` (read-only) and use the scalar form
   if it accepts `external_eigenvalues`, otherwise build the `tol_vec` exactly
   as example 03 lines ~130–132.
6. `conjugate_permutation = reduce(vcat, [[2p, 2p-1] for p in 1:length(master)])`
   — identical in both cases (example 03 line 44 confirms it covers master pairs
   only, not the external states).
7. Timed `solve_cohomological_problem(model, mset, master_eigenvalues,
   master_modes, left_eigenmodes, resonance_set; master_modes_derivatives,
   conjugate_permutation)` — same call in both cases (example 03 lines ~148–151).
8. Return `InvariantManifoldROM(W, R, collect(eigenvalues), master, order, forcing,
   (; m.info..., n_monomials = length(mset), N_EXT, Ω = (forcing === nothing ?
   nothing : Ω), solve_time_s = t))`.

### 4.2 `postprocess.jl`

- `real_dynamics(rom)`: example 01 §9 (lines ~243–253). The `conj_map` loop is
  written over `NVAR` and the external states also come in conjugate `±iΩ`
  pairs, so the same `isodd(i) ? i+1 : i-1` rule applies in the forced case —
  state this in a comment, and set `NVAR = 2length(rom.master) + rom.info.N_EXT`.
- `print_equations(rom; tol = 1e-12)`: printing loop, lines ~255–261; in the
  forced case label the last two exponents as the external variables.
- `save_rom(rom, dir)`: W.jls, R.jls, R_coefficients.csv, summary.txt from
  `rom.info` (+ forcing line). CSV format byte-compatible with
  `examples/common/results_io.jl` — Phase 5 depends on it. Leave
  `examples/common/results_io.jl` unchanged.

Verify: temp env, BOTH pipelines on the example mesh at `order = 3` (fast):
autonomous, and forced with `HarmonicForcing(mode = 1, amplitude = 0.01)`.
`print_equations` runs on both; `save_rom` writes four files. `src/` diff empty.

---

## Phase 5 — Showcase example + two equivalence gates

1. `mv examples/01_clamped_beam_ferrite/main.jl examples/01_clamped_beam_ferrite/low_level.jl`
   (the explicit construction stays runnable). New `main.jl` = the target script
   (autonomous `rom` only by default; the forced call present but commented,
   with one line saying what it adds), plus the env-bootstrap block (copy
   `low_level.jl` lines 15–23, adding `"FerriteGmsh"` to the `Pkg.add` list) and
   the same Rayleigh constants.
2. **Gate A — autonomous equivalence** (must pass before Gate B):
   run `low_level.jl`, save its `R_coefficients.csv` aside, run `main.jl`,
   compare: max relative deviation `< 1e-10`, else STOP. (Commands as in the
   previous revision of this plan: readdlm both, `@assert size`, tolerance check.)
   If it fails, the bug is in Phase 4 steps 3 or 6 — fix there only.
3. **Gate B — forcing consistency**: a forced ROM with `amplitude = 0.0` must
   reproduce the autonomous ROM on the shared monomials:
   ```julia
   # pseudo-outline for a script in /tmp:
   rom0 = SVK.parametrise(beam; master = [1], order = 5)
   romf = SVK.parametrise(beam; master = [1], order = 5,
                          forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.0))
   # For every monomial of rom0's mset (NVAR = 2), find the monomial of romf's
   # mset (NVAR = 4) with the same z-exponents and zero external exponents;
   # assert the R coefficients match to 1e-10.
   ```
   Use `mset.exponents` to build the index map. If Gate B fails, the bug is in
   Phase 4 steps 4–5 (forcing assembly / resonance set) — fix there only.
4. README: high-level script as the headline, a short "Harmonic forcing" section
   (what `HarmonicForcing` does physically, that Ω defaults to the mode's
   natural frequency), "Under the hood" linking `low_level.jl`.
5. Website teaser: minimal alignment only — badge
   `implemented for SVK + Ferrite 🚧 FRF continuation`, names
   (`mechanical_model`, `parametrise(beam; master = [1], order = 7, forcing = HarmonicForcing(mode = 1, amplitude = 0.02))`,
   the `get_extension` line). Do not redesign the page.

---

## Phase 6 — Tests and docs

1. `test/StructuralSVK/test_structural_svk.jl`: small-mesh versions of Gate A
   and Gate B (tiny hex mesh, low order — must run in seconds). Wire into
   `test/runtests.jl` as a gated group `structural_svk` (same `should_run`
   pattern; editing `test/` is allowed — only `src/` is frozen). Add `Ferrite`,
   `FerriteGmsh`, `LinearMaps` to `[extras]`/`[targets]` (Arpack already there).
2. README.md quickstart: the target script. CLAUDE.md: one paragraph on
   `MORFEStructuralSVK` (location, trigger packages, `get_extension` access,
   forcing semantics).
3. Final:
   ```bash
   julia --project -e 'using Pkg; Pkg.test()'
   GROUP=structural_svk julia --project test/runtests.jl
   git diff --stat main -- src/   # MUST print nothing for the whole branch
   ```
   Report for human approval; do not merge.

## Deferred (record, do not implement)

- `src/` stubs to expose `MORFE.parametrise` etc. directly (forbidden by the src-freeze).
- Multiple simultaneous forcings (`Vector{HarmonicForcing}`) and `shape_mode ≠ frequency_mode`
  — example 03 supports both; the high-level API restricts to the common case by design.
- FRF continuation over Ω (`continuation(rom; ω_range)`) via `MORFEBifurcationKitExt`; `plot(frf)`.
- Raw-matrix `parametrise(M, C, K, G, H; ...)` entry point (teaser form).
- Gridap parity; non-SVK materials.
