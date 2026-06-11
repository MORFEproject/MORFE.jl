# Demo Folder Reorganisation Plan

Goal: replace the current `demo/` folder with a clean structure:

- `ext/` — Ferrite FEM backend becomes a proper package extension (plug-and-play via `using MORFE, Ferrite`).
- `examples/` — self-contained, runnable, user-facing examples (numbered) + `examples/internals/` low-level API demos for advanced users.
- `benchmark/` — benchmark scripts only, no result data.
- Everything generated, vendored, or research-private is moved to an **archive folder outside the repo**. Nothing is ever deleted.

---

## Ground rules for the executing agent

1. **Run every command from the repository root** (the folder containing `Project.toml` with `name = "MORFE"`). Verify with `test -f Project.toml && grep -q 'name = "MORFE"' Project.toml`.
2. **Never use `rm`/`rm -rf` on anything.** Files leave the repo only via `mv` into the archive folder `../MORFE_demo_archive/`.
3. Use plain `mv` (not `git mv`) for all moves. Git rename detection plus a final `git add -A` handles tracking.
4. **Stop and report** if any verification step fails. Do not improvise fixes beyond what a phase explicitly allows.
5. Do not edit any file under `src/` except the two small edits listed in Phase 3 and Phase 6.
6. The folder `demo/KármánVortexStreet` has non-ASCII characters. Always reference it with the glob `demo/K*rm*nVortexStreet` to avoid encoding mistakes.
7. Each phase ends with a commit, so progress is checkpointed and reversible.

---

## Phase 0 — Preflight

```bash
# 0.0 Pre-existing work in progress: demo/ParametricClampedClampedBeam was already
# restructured by the user (staged renames into fem/, backbone/, validation/,
# plotting/, results/{data,figures}). Commit it as-is before anything else:
git add demo/ParametricClampedClampedBeam
git commit -m "Restructure ParametricClampedClampedBeam demo into subfolders"

# 0.1 Confirm clean working tree (untracked files are OK; staged/modified are not)
git status --porcelain | grep -v '^??' && echo "DIRTY - STOP" || echo "OK"

# 0.2 Create work branch
git checkout -b reorganise-demos

# 0.3 Create archive folder OUTSIDE the repo
mkdir -p ../MORFE_demo_archive

# 0.4 Baseline: tests must pass before touching anything
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

If 0.4 fails, STOP — the reorganisation must not start from a broken state.

---

## Phase 1 — Archive research projects, vendored code, and generated data

These are moved to `../MORFE_demo_archive/`, preserving their folder names. Most are already git-ignored (IFX, Teo_COLSON, `from_reference_paper`, `arXiv-2510.26542v1`, Manifests), so this mostly affects untracked files; the rest will show up as deletions, which is intended.

```bash
A=../MORFE_demo_archive
mkdir -p "$A/KarmanVortexStreet" "$A/BenchmarkFerrite" "$A/ParametrisationMethod" \
         "$A/Eigensolver" "$A/BenchmarkMorfe20" "$A/ArchComsolWedge" \
         "$A/ParametricClampedClampedBeam" "$A/Teo_COLSON_extras"

# 1.1 Whole private/research projects
mv demo/IFX                 "$A/"
mv demo/Teo_COLSON          "$A/"

# 1.2 Vendored third-party / reference material (note: contains a nested .git — do not inspect, just move)
mv demo/K*rm*nVortexStreet/from_reference_paper   "$A/KarmanVortexStreet/"
mv demo/K*rm*nVortexStreet/arXiv-2510.26542v1     "$A/KarmanVortexStreet/"
mv demo/BenchmarkMorfe20/Morfe_2_0                "$A/BenchmarkMorfe20/"

# 1.3 Generated results / outputs (tracked — git will record deletions)
mv demo/K*rm*nVortexStreet/results                "$A/KarmanVortexStreet/"
mv demo/ArchComsolWedge/results                   "$A/ArchComsolWedge/"
# Rescue documentation that lives inside results/ before archiving it:
mv demo/ParametricClampedClampedBeam/results/data/backbone_derivation.md demo/ParametricClampedClampedBeam/backbone/
mv demo/ParametricClampedClampedBeam/results      "$A/ParametricClampedClampedBeam/"
mv demo/BenchmarkFerrite/benchmark_results        "$A/BenchmarkFerrite/"
mv demo/BenchmarkFerrite/benchmark_results_backup_20h00_20_May "$A/BenchmarkFerrite/"

# 1.4 Serialized data, profiles, generated images, env-only folders
mv demo/ParametrisationMethod/output.h5           "$A/ParametrisationMethod/"
mv demo/ParametrisationMethod/*.png               "$A/ParametrisationMethod/"
mv demo/Eigensolver/*.png                         "$A/Eigensolver/"
mv demo/Eigensolver/.plotenv                      "$A/Eigensolver/"
mv demo/BenchmarkMorfe20/profile_after_opt.html   "$A/BenchmarkMorfe20/"
mv demo/BenchmarkMorfe20/profile_sparse.html      "$A/BenchmarkMorfe20/"
mv demo/BenchmarkFerrite/invariance_convergence_master.png "$A/BenchmarkFerrite/"
mv demo/backbone_env                              "$A/"

# 1.5 Regenerable meshes (BenchmarkFerrite has generator scripts; keep those, archive the .msh files)
mkdir -p "$A/BenchmarkFerrite/meshes"
mv demo/BenchmarkFerrite/*.msh                    "$A/BenchmarkFerrite/meshes/"

# 1.6 Stray working notes
mv demo/K*rm*nVortexStreet/plan.md                       "$A/KarmanVortexStreet/"
mv demo/K*rm*nVortexStreet/plan_karman_vortex_street.md  "$A/KarmanVortexStreet/"
```

Notes:
- If a `mv` fails with "No such file or directory", check whether the file was already moved/renamed; if it simply doesn't exist anymore, log it and continue. Any other error: STOP.
- `*.zip` files inside moved folders travel with them automatically.
- Do NOT archive: `demo/Eigensolver/K.csv` and `M.csv` (inputs, not outputs), `demo/BenchmarkFerrite/reduced_dynamics_*.txt` (small reference data), `demo/BenchmarkMorfe20/beam.mphtxt` (input mesh), `demo/Gridap/clamped_clamped_beam.msh`, `demo/K*rm*nVortexStreet/cylinder_flow.msh`, `demo/ArchComsolWedge/arch_2_force.mphtxt` (input meshes), all files under `demo/FEMUtility/` (importer test fixtures).

Commit:

```bash
git add -A
git commit -m "Archive generated results, vendored code, and research projects out of demo/"
```

---

## Phase 2 — Update .gitignore

Append to `.gitignore` (keep existing content):

```
# Generated example/benchmark outputs
results/
benchmark_results*/
*.h5
*.jls
*.vtu
*.pvd
profile*.html
__pycache__/
*.pyc

# Never track IFX content, even if the folder reappears anywhere in the repo
IFX/
```

Then remove the now-obsolete lines `Teo_COLSON`, `arXiv-2510.26542v1`, `from_reference_paper` from `.gitignore` (those folders no longer live in the repo). **Do NOT remove the existing `IFX` line** — IFX contents must never be tracked, as a permanent guard in case the folder is copied back.

```bash
git add .gitignore && git commit -m "Ignore generated outputs"
```

---

## Phase 3 — Ferrite backend as a package extension

The repo already uses extensions (`ext/MORFEGmshExt.jl` etc.) and `Ferrite` is already a `weakdep`. Follow the existing pattern exactly.

### 3.1 Move the backend file

```bash
mkdir -p ext/FerriteBackend
mv demo/Ferrite/ferrite_assembly.jl ext/FerriteBackend/ferrite_assembly.jl
```

### 3.2 Add API stubs in `src`

Open `src/FullOrderModel/MultilinearMaps.jl`. Near the docstring that currently says
`See demo/Gridap/ and demo/Ferrite/ for reference FEM backend implementations.` (around line 13):

1. Change that sentence to: `See ext/FerriteBackend/ and examples/02_clamped_beam_gridap/ for reference FEM backend implementations.`
2. At the end of the file (top level, inside the module), add:

```julia
"""
    ferrite_nonlinearity(degree::Integer, args...; kwargs...)

Construct a Ferrite-backed geometric nonlinearity term of the given polynomial
`degree` (2 or 3). Requires `using Ferrite` (loads the `MORFEFerriteExt`
extension). See `ext/FerriteBackend/ferrite_assembly.jl` for the argument list.
"""
function ferrite_nonlinearity end

"""
    ferrite_assemble_KM!(K, M, dh, cv, λ, μ, ρ)

Assemble linear stiffness and mass matrices with the Ferrite backend.
Requires `using Ferrite`.
"""
function ferrite_assemble_KM! end

export ferrite_nonlinearity, ferrite_assemble_KM!
```

(If exports in this module are declared in `src/MORFE.jl` instead — check where existing exports live — put the `export` line in the same place the module's other exports are declared. Verify with `grep -rn "export" src/FullOrderModel/MultilinearMaps.jl src/MORFE.jl`.)

### 3.3 Create `ext/MORFEFerriteExt.jl`

```julia
module MORFEFerriteExt

using MORFE: MORFE
using Ferrite
using LinearAlgebra
using SparseArrays

_ext(file) = joinpath(@__DIR__, "FerriteBackend", file)
include(_ext("ferrite_assembly.jl"))

MORFE.ferrite_nonlinearity(degree::Integer, args...; kwargs...) =
    FerriteGeometricNonlinearity{Int(degree)}(args...; kwargs...)

MORFE.ferrite_assemble_KM!(args...; kwargs...) = assemble_KM!(args...; kwargs...)

end
```

Check the top of `ext/FerriteBackend/ferrite_assembly.jl`: it already contains `using Ferrite`, `using LinearAlgebra`, `using SparseArrays`, `using MORFE: MORFE`. Since the file is now `include`d inside a module that already has these, **delete those four `using` lines from `ferrite_assembly.jl`** (and only those lines).

### 3.4 Register the extension in `Project.toml`

In the `[extensions]` section add (alphabetical position is fine):

```toml
MORFEFerriteExt = "Ferrite"
```

### 3.5 Verify

```bash
julia --project -e 'using Pkg; Pkg.instantiate(); using MORFE'   # must precompile cleanly
julia --project -e '
  using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add("Ferrite")
  using MORFE, Ferrite
  @assert Base.get_extension(MORFE, :MORFEFerriteExt) !== nothing
  @assert isdefined(Base.get_extension(MORFE, :MORFEFerriteExt), :FerriteGeometricNonlinearity)
  println("EXTENSION OK")'
julia --project -e 'using Pkg; Pkg.test()'
```

All three must succeed. Commit: `git add -A && git commit -m "Promote Ferrite FEM backend to MORFEFerriteExt package extension"`.

---

## Phase 4 — Build the `examples/` tree

### 4.1 Moves

```bash
mkdir -p examples/internals
mv demo/Ferrite                          examples/01_clamped_beam_ferrite
mv demo/Gridap                           examples/02_clamped_beam_gridap
mv demo/ArchComsolWedge                  examples/03_arch_comsol_wedge
mv demo/ParametricClampedClampedBeam     examples/04_parametric_clamped_beam
mv demo/K*rm*nVortexStreet               examples/05_karman_vortex_street
mv demo/FEMUtility                       examples/mesh_import

mv demo/demo_polynomials.jl                  examples/internals/
mv demo/demo_multiindices_factorisations.jl  examples/internals/
mv demo/FullOrderModel                       examples/internals/full_order_model
mv demo/ParametrisationMethod                examples/internals/parametrisation_method
mv demo/Eigensolver                          examples/internals/eigensolver
```

### 4.2 Fix the Ferrite example after the extension move

Edit `examples/01_clamped_beam_ferrite/demo_mechanical_problem.jl`:

1. Delete the line `include(joinpath(@__DIR__, "ferrite_assembly.jl"))`.
2. Ensure the top of the script has `using MORFE` and `using Ferrite` (add if missing).
3. Replace `assemble_KM!(` → `ferrite_assemble_KM!(` (all occurrences).
4. Replace `FerriteGeometricNonlinearity{2}(` → `ferrite_nonlinearity(2, ` and `FerriteGeometricNonlinearity{3}(` → `ferrite_nonlinearity(3, ` (note: the `{N}` becomes the first positional argument).
5. The example's `Project.toml` must list `MORFE` and `Ferrite` under `[deps]` (check; add UUIDs from the root `Project.toml` weakdeps section if missing).

### 4.2b ParametricClampedClampedBeam is already internally organised

`examples/04_parametric_clamped_beam/` already has the desired internal layout
(`fem/`, `backbone/`, `validation/`, `plotting/`, top-level entry script
`parametric_beam_demo.jl`, `README.md`, `Project.toml`). Move it as-is — do not
restructure its contents. Use this folder as the structural template when
writing READMEs for the other examples (Phase 6).

### 4.3 Per-example environments

For each of `examples/01_*` … `examples/05_*`:
- Keep `Project.toml` if present; create one if absent (deps: `MORFE` plus whatever the script `using`s).
- `Manifest.toml` files are git-ignored already; leave them on disk, do not add them.

### 4.4 Verify

```bash
# Smoke-run the cheap internals demos (each must exit 0)
for f in examples/internals/demo_polynomials.jl examples/internals/demo_multiindices_factorisations.jl; do
  julia --project -e "include(\"$f\")" || { echo "FAIL $f"; exit 1; }
done

# Smoke-run the Ferrite example end-to-end (this validates Phase 3 + 4.2)
julia --project=examples/01_clamped_beam_ferrite -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/01_clamped_beam_ferrite/demo_mechanical_problem.jl")'
```

If the Ferrite example fails, fix only the symbol renames from 4.2 — do not modify `ext/`. Commit: `git add -A && git commit -m "Restructure demos into examples/"`.

---

## Phase 5 — Build the `benchmark/` tree

```bash
mkdir -p benchmark
mv demo/BenchmarkFerrite   benchmark/ferrite
mv demo/BenchmarkO4        benchmark/order4
mv demo/BenchmarkMorfe20   benchmark/morfe20
# demo/ should now be empty except possibly .DS_Store:
ls -A demo/
mv demo/.DS_Store ../MORFE_demo_archive/ 2>/dev/null
rmdir demo
```

`rmdir` (not `rm -rf`) is deliberate: it fails if anything is left, which is the safety check. If it fails, list the leftovers and move each to the correct destination per the Appendix table before retrying.

Commit: `git add -A && git commit -m "Move benchmarks to benchmark/"`.

---

## Phase 6 — Documentation updates

### 6.1 `examples/README.md` (create)

Contents: a table listing each example (folder, one-line description, what it demonstrates, approximate runtime), how to run one (`julia --project=examples/01_clamped_beam_ferrite`, `Pkg.develop(path="../..")`, `Pkg.instantiate()`, `include("demo_mechanical_problem.jl")`), and one sentence pointing advanced users to `examples/internals/` and `ext/FerriteBackend/`.

### 6.2 Per-example `README.md`

For each `examples/0X_*` without a README, create a short one: what the model is, the entry script name, expected outputs, where the mesh comes from. `04_parametric_clamped_beam` already has one — keep it, just fix any paths that mention `demo/`.

### 6.3 Repo-wide path references

```bash
grep -rn "demo/" --include="*.jl" --include="*.md" --include="*.toml" src ext test docs examples benchmark README.md CLAUDE.md 2>/dev/null
```

Update every hit to the new path (`examples/...` or `benchmark/...`). In `CLAUDE.md`, also update the GROUP documentation (see Phase 7) and the FEM backend sentence to reference `ext/FerriteBackend/`.

Commit: `git add -A && git commit -m "Update docs and path references"`.

---

## Phase 7 — Wire examples into the test groups

`test/runtests.jl` currently has groups `full_order_model`, `parametrisation_method`, `spectral_decomposition`, `utils`, and an empty `end_to_end`. Add an `examples` group that smoke-runs only the cheap internals scripts:

```julia
if should_run("examples")
    @testset "Examples smoke tests" begin
        @testset "internals" begin
            include(joinpath(@__DIR__, "..", "examples", "internals", "demo_polynomials.jl"))
            include(joinpath(@__DIR__, "..", "examples", "internals", "demo_multiindices_factorisations.jl"))
            @test true
        end
    end
end
```

IMPORTANT: `should_run(group)` returns true when `GROUP == "all"`, which would make `Pkg.test()` run examples by default. To keep default test time unchanged, gate it explicitly instead:

```julia
if GROUP == "examples"
    ...
end
```

Verify: `GROUP=examples julia --project test/runtests.jl` passes, and plain `julia --project -e 'using Pkg; Pkg.test()'` still passes with unchanged scope.

Commit: `git add -A && git commit -m "Add GROUP=examples smoke tests"`.

---

## Phase 8 — Final verification checklist

Run all of these; every one must pass before merging:

```bash
# 1. Package loads and tests pass
julia --project -e 'using Pkg; Pkg.test()'

# 2. Extension loads
julia --project -e '
  using Pkg; Pkg.activate(temp=true); Pkg.develop(path="."); Pkg.add("Ferrite")
  using MORFE, Ferrite
  @assert Base.get_extension(MORFE, :MORFEFerriteExt) !== nothing; println("OK")'

# 3. No stale references
grep -rn "demo/" --include="*.jl" --include="*.md" src ext test examples benchmark | grep -v "MORFE_demo_archive" && echo "STALE REFS - FIX" || echo "OK"

# 4. demo/ is gone, archive is populated
test ! -d demo && echo "OK"
ls ../MORFE_demo_archive

# 5. No large/generated files tracked (everything listed should be source, meshes, or small reference data)
git ls-files examples benchmark | xargs -I{} du -k "{}" | sort -rn | head -20

# 6. Repo size sanity: tracked content under examples+benchmark should be a few MB, not GB
du -sh examples benchmark

# 7. IFX guard: must print the matching ignore rule and nothing may be tracked
git check-ignore -v IFX/dummy.txt
git ls-files | grep -i "IFX" && echo "IFX TRACKED - FIX" || echo "OK"
```

Then: `git log --oneline main..HEAD` to review the commit sequence, and report back for human review before merging. **Do not merge or push without explicit approval.**

---

## Appendix — Full source → destination map

| Old path | New location |
|---|---|
| `demo/Ferrite/ferrite_assembly.jl` | `ext/FerriteBackend/ferrite_assembly.jl` |
| `demo/Ferrite/` (rest) | `examples/01_clamped_beam_ferrite/` |
| `demo/Gridap/` | `examples/02_clamped_beam_gridap/` |
| `demo/ArchComsolWedge/` (minus `results/`) | `examples/03_arch_comsol_wedge/` |
| `demo/ParametricClampedClampedBeam/` (minus `results/`; keep its existing `fem/`, `backbone/`, `validation/`, `plotting/` layout; `results/data/backbone_derivation.md` moves to `backbone/`) | `examples/04_parametric_clamped_beam/` |
| `demo/KármánVortexStreet/` (minus `results/`, `from_reference_paper/`, `arXiv-*`, `plan*.md`) | `examples/05_karman_vortex_street/` |
| `demo/FEMUtility/` | `examples/mesh_import/` |
| `demo/demo_polynomials.jl`, `demo/demo_multiindices_factorisations.jl` | `examples/internals/` |
| `demo/FullOrderModel/` | `examples/internals/full_order_model/` |
| `demo/ParametrisationMethod/` (minus `output.h5`, `*.png`) | `examples/internals/parametrisation_method/` |
| `demo/Eigensolver/` (minus `*.png`, `.plotenv`; keep `K.csv`, `M.csv`) | `examples/internals/eigensolver/` |
| `demo/BenchmarkFerrite/` (scripts, generators, `reduced_dynamics_*.txt`) | `benchmark/ferrite/` |
| `demo/BenchmarkO4/` | `benchmark/order4/` |
| `demo/BenchmarkMorfe20/` (minus `Morfe_2_0/`, `profile*.html`) | `benchmark/morfe20/` |
| `demo/IFX/`, `demo/Teo_COLSON/` | `../MORFE_demo_archive/` |
| All `results/`, `benchmark_results*/`, `*.msh` in BenchmarkFerrite, `output.h5`, `*.png` outputs, `profile*.html`, `Morfe_2_0/`, `from_reference_paper/`, `arXiv-*`, `backbone_env` (top-level), `.plotenv` | `../MORFE_demo_archive/<original parent>/` |

## Deferred (explicitly out of scope for this migration)

- Gridap extension: `examples/02_clamped_beam_gridap/demo_mechanical_problem.jl` is self-contained; extract a `MORFEGridapExt` only after the Ferrite extension pattern has been validated in use.
- Turning examples into Literate.jl tutorials under `docs/`.
- Registering separate backend packages (`MORFEFerrite.jl`); only worth it once the extension API is stable.
