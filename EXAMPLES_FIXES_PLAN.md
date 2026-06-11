# Examples Fixes Plan

Four problems to fix, in this order:

1. `05_karman_vortex_street` is disorganised (7 loose `.jl` files at its root) and inconsistent with the `04` layout.
2. The archived results were never restored — `W.jls`, `R.jls`, ParaView files, `output.h5`, benchmark data etc. must ALL come back into the repo tree, properly organised. Nothing gets left behind except the explicit stays-in-archive list.
3. `04_parametric_clamped_beam/README.md` describes the wrong model (single scalar θ / uniform stretch; the code is bivariate θ₁+θ₂) — everything downstream of that is wrong too.
4. All "Approximate runtime" values in READMEs are invented. Only measured numbers are allowed.

---

## Ground rules for the executing agent

1. Run everything from the repository root. Never use `rm`/`rm -rf` (exception: none in this plan).
2. **Phase 2 uses `cp -R`, never `mv`** — the archive repo (`../MORFE_results_archive`) is the durable backup and must remain byte-for-byte intact.
3. One phase = one commit. Stop and report on any failed verification.
4. **Never write a fact into a README that you did not extract from code or measure yourself.** No estimates, no "typical" values, no rounding up "to be safe". If you don't have a measurement, write `not yet measured`.
5. Nothing from `IFX/` or `Teo_COLSON/` enters the repo.

## Phase 0 — Preflight

```bash
git status --porcelain | grep -v '^??' && { git add -A; git commit -m "WIP before examples fixes"; }
git checkout -b examples-fixes
ls ../MORFE_results_archive || ls ../MORFE_demo_archive || { echo "ARCHIVE MISSING - STOP"; exit 1; }
# If only MORFE_demo_archive exists, use that path as $A throughout and report
# that the archive was never converted to a git repo (Phase 1 of the previous plan).
```

---

## Phase 1 — Reorganise `05_karman_vortex_street` to match `04`

### 1.1 Target layout

```
05_karman_vortex_street/
├── main.jl              (stays)
├── config.jl            (stays — user-facing knobs)
├── README.md, Project.toml
├── cylinder_flow.msh    (stays — input mesh)
├── fem/                 ← mesh.jl, fem_setup.jl, fluid_maps.jl, linear_operators.jl
├── solver/              ← steady_state.jl, eigensolver.jl
├── backbone/            ← backbone_env/  (postprocessing environment)
└── results/             (unchanged)
```

### 1.2 Moves

```bash
cd examples/05_karman_vortex_street
mkdir -p fem solver backbone
mv mesh.jl fem_setup.jl fluid_maps.jl linear_operators.jl fem/
mv steady_state.jl eigensolver.jl solver/
mv backbone_env backbone/backbone_env
cd -
```

### 1.3 Fix every path reference

`main.jl` lines ~43–49 contain `include("mesh.jl")` etc. Update each to the new
relative path (`include(joinpath(@__DIR__, "fem", "mesh.jl"))` style — use
`@__DIR__`-based paths, matching the contract). Then search exhaustively; the
moved files may also include each other or reference `backbone_env`:

```bash
grep -rn "include(\|backbone_env\|mesh.jl\|fem_setup\|fluid_maps\|linear_operators\|steady_state\|eigensolver.jl" \
  examples/05_karman_vortex_street --include="*.jl" --include="*.md"
```

Fix every hit that points to an old location. Also update the *Files* /
*Expected outputs* tables in its `README.md` to the new layout.

### 1.4 Verify

```bash
# Parse-check every moved file (catches broken includes without a 30-min run):
for f in $(find examples/05_karman_vortex_street -name "*.jl" -not -path "*backbone_env*"); do
  julia -e "Meta.parseall(read(\"$f\", String)); println(\"parse OK: $f\")" || exit 1
done
# Stale-path check must return nothing:
grep -rn 'include("mesh\|include("fem_setup\|include("fluid_maps\|include("linear_operators\|include("steady_state\|include("eigensolver' examples/05_karman_vortex_street && echo "STALE - FIX" || echo "OK"
```

Optional full verification: if `config.jl` exposes the parametrisation order,
run once at the lowest order to confirm the pipeline executes; restore the
default afterwards. If not feasible, state so in the commit message.

Commit: `git add -A && git commit -m "Reorganise 05_karman_vortex_street into fem/solver/backbone layout"`.

---

## Phase 2 — Restore ALL archived results into the repo tree

`A=../MORFE_results_archive` (adjust per Phase 0). First run `ls -R "$A" | head -100`
and read `"$A/INDEX.md"` (if present) to confirm the actual layout; the
mappings below assume the structure created by the original migration. Adapt
source paths if they differ, but the destination rules are fixed.

### 2.1 Restoration map — copy with `cp -R`, preserve internal structure

| Archive source | Destination in repo | Note |
|---|---|---|
| `$A/ArchComsolWedge/results/mode_1_order_5_cnf/` | `examples/03_arch_comsol_wedge/results/mode_1_order_5_cnf/` | Matches 03's `results/<run_name>/` contract; includes `W.jls`, `R.jls`, `postprocess/` |
| `$A/ArchComsolWedge/results/visualise_geometry_and_modes/` | `examples/03_arch_comsol_wedge/results/visualise_geometry_and_modes/` | Includes the `paraview/` folder |
| `$A/KarmanVortexStreet/results/Re49.03_ord5/` | `examples/05_karman_vortex_street/results/Re49.03_ord5/` | Includes `paraview/` and `postprocess/` |
| `$A/KarmanVortexStreet/plan.md`, `plan_karman_vortex_street.md` | `examples/05_karman_vortex_street/notes/` | Working notes — small, keep with the example |
| `$A/ParametricClampedClampedBeam/results/` | `examples/04_parametric_clamped_beam/results/archived/` | Goes under `archived/` (NOT `data/`) so a fresh run never overwrites or mixes with the historical `W.jls`/`R.jls`/figures |
| `$A/ParametrisationMethod/output.h5` and `*.png` | `examples/internals/parametrisation_method/results/archived/` | |
| `$A/Eigensolver/*.png` | `examples/internals/eigensolver/results/archived/` | |
| `$A/Eigensolver/.plotenv/` | `examples/internals/eigensolver/.plotenv/` | Plot environment, back where scripts expect it |
| `$A/BenchmarkFerrite/benchmark_results/` | `benchmark/ferrite/benchmark_results/` | Already covered by the `benchmark_results*/` gitignore rule |
| `$A/BenchmarkFerrite/benchmark_results_backup_20h00_20_May/` | `benchmark/ferrite/benchmark_results_backup_20h00_20_May/` | |
| `$A/BenchmarkFerrite/meshes/*.msh` | `benchmark/ferrite/` (folder root) | The benchmark scripts reference meshes at the folder root — verify with `grep -n '\.msh' benchmark/ferrite/*.jl` and place exactly where the scripts expect |
| `$A/BenchmarkFerrite/invariance_convergence_master.png` | `benchmark/ferrite/results/archived/` | |
| `$A/BenchmarkMorfe20/profile_after_opt.html`, `profile_sparse.html` | `benchmark/morfe20/results/archived/` | |
| `$A/BenchmarkMorfe20/Morfe_2_0/` | `benchmark/morfe20/Morfe_2_0/` ONLY IF `grep -rn "Morfe_2_0" benchmark/morfe20/*.jl` shows the scripts need it; otherwise it stays archived (it is vendored code, not results) | |

### 2.2 Stays in the archive (record this list in the commit message verbatim)

- `$A/IFX/` — private; never tracked, never restored.
- `$A/Teo_COLSON/` — private research project.
- `$A/KarmanVortexStreet/from_reference_paper/` — vendored third-party code with a nested `.git`.
- `$A/KarmanVortexStreet/arXiv-2510.26542v1/` — third-party paper source.
- `$A/backbone_env/` — orphaned Manifest, superseded by per-example envs.
- `$A/BenchmarkMorfe20/Morfe_2_0/` — unless the grep in 2.1 proves it is needed.

Everything in the archive must appear in exactly one of: the 2.1 map or the 2.2 list.
Run `find "$A" -maxdepth 2 -type d` and confirm every entry is covered; anything
unaccounted for: STOP and report it.

### 2.3 Git hygiene

The restored files must stay UNTRACKED (the archive git repo is the durable
copy; the working tree gets the convenient copy). Check, then patch `.gitignore`:

```bash
git status --porcelain | grep '^??' | grep -v 'results/\|benchmark_results' 
```

For every path that shows up (expected: `benchmark/ferrite/*.msh`, possibly
`Morfe_2_0/`, `notes/`, `.plotenv/`), add a matching `.gitignore` line:

```
benchmark/ferrite/*.msh
benchmark/morfe20/Morfe_2_0/
examples/internals/eigensolver/.plotenv/
examples/05_karman_vortex_street/notes/
```

Rationale to record in the commit message: meshes are regenerable
(`generate_beam_meshes.jl`), binaries live in the archive repo; if the user
later wants them version-controlled, the right tool is git-lfs — flag this as
an open question for the user, do not enable LFS yourself.

### 2.4 Completeness verification

```bash
# Each restored mapping must be identical to its archive source:
diff -rq "$A/ArchComsolWedge/results" examples/03_arch_comsol_wedge/results --exclude=reference --exclude=mode_1_order_5_cnf_new 2>/dev/null
diff -rq "$A/KarmanVortexStreet/results" <(true) ... # repeat per mapping:
#   diff -rq <archive-src> <repo-dest> must print nothing for each row of 2.1
#   (for 04: diff -rq "$A/ParametricClampedClampedBeam/results" examples/04_parametric_clamped_beam/results/archived)
# Global count check — files in archive not in the stays-archived list:
find "$A" -type f \
  -not -path "*/IFX/*" -not -path "*/Teo_COLSON/*" \
  -not -path "*/from_reference_paper/*" -not -path "*/arXiv-2510.26542v1/*" \
  -not -path "*/.git/*" -not -path "*backbone_env*" -not -name "INDEX.md" \
  -not -name ".DS_Store" | wc -l
# Compare with the count of restored files; investigate any shortfall file-by-file.
```

Update each affected README's *Expected outputs* / add a short *Historical
results* line: where the restored run folders are and what they contain.

Commit (this commits `.gitignore` + READMEs; the data itself is untracked):
`git add -A && git commit -m "Restore archived results into examples and benchmarks (untracked; archive repo remains the backup)"`.

---

## Phase 3 — Rewrite `04_parametric_clamped_beam/README.md` from the code

The current README describes a SINGLE parameter θ (uniform axial stretch). The
code implements TWO external parameters: read the header comment of
`examples/04_parametric_clamped_beam/main.jl` (lines 1–30) — it is the source
of truth:

- θ₁ — uniform axial stretch, constant Jacobian contribution `J₁`
- θ₂ — arch pre-deformation along the first bending eigenmode φ₁, per-QP Jacobian `J₂(x₀)`
- Reference map: `x(θ₁,θ₂,x₀) = x₀ + θ₁ J₁ x₀ + θ₂ φ₁(x₀)`; `J = I + θ₁J₁ + θ₂J₂(x₀)`
- `N_EXT = 2`, `NVAR = 4`, reduced coordinates `(z₁, z₂, θ₁, θ₂)`
- `N_θ` truncates BOTH θ-expansions; the algebra is bivariate (`fem/bivariate_polynomials.jl`)

### Procedure

1. Extract, do not invent. For every section rewrite, the content must come from:
   `main.jl` (header + section comments), `fem/bivariate_geometry.jl`,
   `fem/bivariate_polynomials.jl`, `backbone/backbone_derivation.md`, and the
   actual folder listing (`find examples/04_parametric_clamped_beam -name "*.jl"`).
2. Sections to fix:
   - **Title/Model**: bivariate parametric ROM (θ₁ stretch + θ₂ arch). Keep the
     "never re-meshed / pull-back to reference configuration" explanation but
     state it for the two-parameter map.
   - **Files table**: regenerate from the real tree (`fem/`, `backbone/`,
     `validation/`, `plotting/` with correct per-file purposes — read each
     file's docstring/header; currently the table lists pre-reorganisation
     filenames without folders and omits the `bivariate_*` files entirely).
   - **How to run**: the example has its own `Project.toml` — replace the stale
     instruction to activate `benchmark/ferrite`. Verify whether
     `generate_beam_mesh.jl` is still a prerequisite by grepping `main.jl` for
     the mesh filename it loads, and document exactly that.
   - **Mathematical structure**: replace the univariate `J(θ)=J₀+θJ₁` story with
     the bivariate map; mention `adj(J)/det(J)` handling only as implemented in
     `fem/bivariate_geometry.jl` (read it first).
   - **Expected outputs**: keep, but add `results/archived/` (Phase 2) and check
     the figure list against `plotting/backbone_plots.jl` and
     `backbone/compute_backbone_theta2.jl` (which figures does the code actually
     `savefig`?).
3. Verification checklist — for each factual claim in the new README, write the
   code location that supports it as an HTML comment at the end of the README
   (`<!-- claims: θ₂ arch mode → main.jl:182-193; NVAR=4 → main.jl:131; ... -->`).
   A claim with no code location must be deleted.
4. Also fix the one-line description of 04 in `examples/README.md`'s table
   ("axial-stretch parameter θ" → "two-parameter ROM: axial stretch θ₁ + arch
   pre-deformation θ₂").

Commit: `git add -A && git commit -m "Rewrite 04 README: bivariate (θ1, θ2) parametric model"`.

---

## Phase 4 — Replace invented runtimes with measurements

### 4.1 Rules

- A runtime may appear in a README ONLY as the wall-clock of a completed run on
  this machine, with the machine recorded. Otherwise write `not yet measured`.
- While measuring, also verify the other quantitative claims currently sitting
  next to the runtimes ("order-9 parametrisation", "~5k-DOF", "~25k-DOF
  non-symmetric system"): extract the true order from each example's
  config/constants and the true DOF count from its `summary.txt` after the run.
  Fix or delete every number you cannot confirm.

### 4.2 Measure

```bash
# Record machine info once:
julia -e 'println(VERSION); println(Sys.CPU_NAME); println(Sys.total_memory() ÷ 2^30, " GiB")'

# For each runnable example (01, 02, 03, 04 — and 05 only with user approval, it is the long one):
/usr/bin/time -p julia --project=examples/01_clamped_beam_ferrite -e '
  using Pkg; Pkg.develop(path="."); Pkg.instantiate();
  include("examples/01_clamped_beam_ferrite/main.jl")' 2> timing_01.txt
tail -3 timing_01.txt   # "real" line = wall clock, includes compilation
```

Notes: report the `real` value, rounded to the nearest minute (or "< 1 min");
label it "including compilation, first run". For `mesh_import` and `internals`,
time the scripts the same way (they should be seconds). For 05: ask the user
before launching; if not approved, write `not yet measured`.

### 4.3 Update

1. Each example README's "Approximate runtime" section → "Measured runtime":
   `X min (Julia <ver>, <CPU>, <RAM>; first run incl. compilation)` or `not yet measured`.
2. The runtime column in `examples/README.md`'s table → same values.
3. While editing, fix any DOF/order numbers per 4.1.

Commit: `git add -A && git commit -m "Replace estimated runtimes with measured values"`.

---

## Phase 5 — Final verification

```bash
# 1. Restored data present where promised (spot-check the heavy hitters):
ls examples/03_arch_comsol_wedge/results/mode_1_order_5_cnf
ls examples/05_karman_vortex_street/results/Re49.03_ord5
ls examples/04_parametric_clamped_beam/results/archived/data   # W.jls, R.jls expected
# 2. Archive untouched:
cd ../MORFE_results_archive && git status --porcelain && cd -   # must be empty
# 3. No stale includes in 05 (repeat Phase 1.4 grep) and tests still pass:
julia --project -e 'using Pkg; Pkg.test()'
GROUP=examples julia --project test/runtests.jl
# 4. No unmeasured number survives:
grep -rn "Approximate runtime" examples && echo "FIX: sections must be 'Measured runtime'" || echo "OK"
grep -rn "depending on hardware\|modern workstation" examples/*/README.md examples/README.md && echo "FIX" || echo "OK"
# 5. Nothing big became tracked:
git ls-files examples benchmark | xargs du -k 2>/dev/null | sort -rn | head -5
```

Report the commit list (`git log --oneline main..HEAD`) for human review.
Do not merge or push without explicit approval.

---

## Open question to surface to the user (do not decide alone)

The restored results are on disk but untracked; the durable copy is the archive
git repo. If the user wants the historical `W.jls`/`R.jls`/ParaView data
version-controlled inside MORFE.jl itself, that requires git-lfs (multi-GB
binaries). Ask before changing anything in that direction.
