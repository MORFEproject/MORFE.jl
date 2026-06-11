# Examples Consistency Plan

Goal: every example in `examples/` follows the same contract — same entry point,
same output layout, every run **generates results** into a predictable place —
and the archived results from the demo reorganisation get a permanent,
documented home (they are important and must never be lost).

---

## Ground rules for the executing agent

1. Run every command from the repository root (`Project.toml` with `name = "MORFE"`).
2. Never use `rm`/`rm -rf`. Files leave the repo only via `mv` into the archive.
3. One phase = one commit. Stop and report if a verification step fails.
4. Do not modify `src/` or `ext/` in this plan. Only `examples/`, `.gitignore`,
   `test/runtests.jl`, and the external archive folder are touched.
5. Do not re-import anything from `IFX` into the repo — its contents are never tracked.
6. When editing example scripts, change only what each step prescribes
   (entry-point renames, output paths, helper usage). Do not refactor the science code.

---

## The example contract (target state)

Every runnable example `examples/0X_name/` must satisfy:

1. **Entry point** is `main.jl` at the example root. Running it end-to-end must
   produce files on disk — a run that only prints to the terminal is non-compliant.
2. **Outputs** go under `examples/0X_name/results/` (never the CWD, never `@__DIR__`
   root), created with `mkpath`, using paths built from `joinpath(@__DIR__, ...)`:
   - *Single-run examples* (01, 02, 04): `results/data/` (ROM: `W.jls`, `R.jls`,
     `R_coefficients.csv`, `summary.txt`), `results/figures/` (PNGs, if any).
   - *Config-driven examples* (03, 05): `results/<run_name>/` containing the same
     `data/` + `figures/` + `summary.txt` structure, where `<run_name>` is derived
     from the config (e.g. `mode_1_order_5_cnf`, `Re49.03_ord5`).
3. **`summary.txt` is mandatory** and contains at minimum: model description and
   mesh/size, master modes and eigenfrequencies, parametrisation order, wall-clock
   time, Julia version, and the MORFE git commit
   (`readchomp(`git -C $(pkgdir(MORFE)) rev-parse --short HEAD`)` guarded by try/catch).
4. **Reference results** live in `results/reference/` — small, curated, tracked
   in git (see Phase 4). Everything else under `results/` is git-ignored.
5. **Determinism**: call `Random.seed!(0)` before any stochastic step.
6. **README.md** with exactly these sections: *Model*, *How to run*,
   *Expected outputs* (explicit file list), *Reference results*, *Approximate runtime*.
7. Own `Project.toml`; `Manifest.toml` never tracked.

`examples/internals/` and `examples/mesh_import/` are exempt from the `main.jl`
rule (they are collections of small scripts) but must still write any file output
under a git-ignored `results/` or `output/` subfolder, never next to the scripts.

---

## Phase 0 — Preflight

```bash
# 0.1 The working tree currently has uncommitted modifications. Commit them first
#     (they are the user's work — do not review or alter them):
git add -A && git commit -m "WIP before examples consistency pass"

# 0.2 Branch
git checkout -b examples-consistency

# 0.3 Confirm the archive from the previous migration exists
ls ../MORFE_demo_archive || { echo "ARCHIVE MISSING - STOP"; exit 1; }

# 0.4 Baseline
julia --project -e 'using Pkg; Pkg.test()'
```

---

## Phase 1 — Give the archived results a permanent home

The archive is currently a loose folder. Make it a self-describing, version-
controlled repository so the results can never be silently lost.

```bash
mv ../MORFE_demo_archive ../MORFE_results_archive
cd ../MORFE_results_archive
git init
```

Create `../MORFE_results_archive/INDEX.md` with one section per result set,
each recording: original repo path, producing example/benchmark, the MORFE
commit that produced it (everything in this archive predates commit
`40c893d "Archive generated results..."` — record that), date, and a one-line
description of what the data is. Walk the folder tree and list every
subfolder; do not skip any.

Then:

```bash
cd ../MORFE_results_archive
git add -A && git commit -m "Initial archive of MORFE demo results with provenance index"
cd - 
```

Recommend to the user (do not execute): push this repo to a private remote
(GitHub + git-lfs for the large `.jls`/`.h5`/`.vtu` files, or institutional
storage/Zenodo for published result sets) so the archive survives disk failure.

Note: `IFX/` and `Teo_COLSON/` inside the archive are private research projects —
include them in `INDEX.md` but flag them as "private — do not publish".

---

## Phase 2 — Shared infrastructure

### 2.1 `.gitignore` fix (required before reference results can be tracked)

Git cannot re-include files inside an ignored directory, so the existing
`results/` rule must be rewritten. In `.gitignore`, replace the line `results/` with:

```
results/*
!results/reference/
!results/reference/**
```

Verify:

```bash
mkdir -p examples/01_clamped_beam_ferrite/results/reference examples/01_clamped_beam_ferrite/results/data
touch examples/01_clamped_beam_ferrite/results/reference/probe.txt examples/01_clamped_beam_ferrite/results/data/probe.txt
git check-ignore examples/01_clamped_beam_ferrite/results/data/probe.txt   # must print the path
git check-ignore examples/01_clamped_beam_ferrite/results/reference/probe.txt && echo "FAIL - STOP" || echo "OK"
rm examples/01_clamped_beam_ferrite/results/reference/probe.txt examples/01_clamped_beam_ferrite/results/data/probe.txt   # the only rm allowed: removing probes this step created
```

### 2.2 Shared results helper

Create `examples/common/results_io.jl` (plain include-file, not a package):

```julia
# Shared result-writing helpers for MORFE examples. Include with:
#   include(joinpath(@__DIR__, "..", "common", "results_io.jl"))
using Serialization, Printf, LinearAlgebra

"Create and return results dir. `run_name=nothing` → flat results/{data,figures}."
function results_dirs(example_dir; run_name = nothing)
    base = run_name === nothing ? joinpath(example_dir, "results") :
                                  joinpath(example_dir, "results", run_name)
    data, figs = joinpath(base, "data"), joinpath(base, "figures")
    mkpath(data); mkpath(figs)
    return (; base, data, figs)
end

"Serialize W and R, write R coefficients as CSV."
function save_rom(dirs, W, R)
    serialize(joinpath(dirs.data, "W.jls"), W)
    serialize(joinpath(dirs.data, "R.jls"), R)
    open(joinpath(dirs.data, "R_coefficients.csv"), "w") do io
        coeffs = R.coefficients   # (NVAR, L) matrix — adjust if field name differs
        println(io, "monomial_index," * join(("R$i" for i in 1:size(coeffs, 1)), ","))
        for j in axes(coeffs, 2)
            println(io, string(j) * "," * join((string(coeffs[i, j]) for i in axes(coeffs, 1)), ","))
        end
    end
end

"Write summary.txt from a vector of `\"key: value\"` strings plus environment info."
function write_summary(dirs, lines::Vector{String})
    open(joinpath(dirs.base, "summary.txt"), "w") do io
        for l in lines; println(io, l); end
        println(io, "julia_version: $(VERSION)")
        commit = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end
        println(io, "morfe_commit: $commit")
        println(io, "timestamp: $(time())")
    end
end
```

Before committing: open `src/ParametrisationMethod/ParametrisationMethod.jl`,
check the actual field name of the coefficient matrix in `ReducedDynamics`
(grep `struct ReducedDynamics`), and fix `R.coefficients` in the helper if it
differs. Also compare against the existing CSV-writing code in
`examples/04_parametric_clamped_beam/parametric_beam_demo.jl` (section 10) —
the helper must reproduce that format.

Commit: `git add -A && git commit -m "Examples: shared results helper and gitignore reference exception"`.

---

## Phase 3 — Bring each example up to contract (one commit per example)

### 3.1 `01_clamped_beam_ferrite` — currently writes NOTHING (worst offender)

1. `mv examples/01_clamped_beam_ferrite/demo_mechanical_problem.jl examples/01_clamped_beam_ferrite/main.jl`
2. At the top of `main.jl` add the include of `../common/results_io.jl`.
3. At the end of the script (after the DPIM solve produces `W` and `R` — find the
   variable names actually used), append:
   ```julia
   dirs = results_dirs(@__DIR__)
   save_rom(dirs, W, R)
   write_summary(dirs, [
       "example: 01_clamped_beam_ferrite",
       "model: clamped-clamped beam, St. Venant-Kirchhoff, Ferrite backend",
       "n_dofs: $(...)", "master_modes: $(...)", "eigenfrequencies: $(...)",
       "order: $(...)",
   ])   # fill the ... from variables present in the script
   ```
4. Update README (contract sections), update any reference to the old script
   name: `grep -rn "demo_mechanical_problem" examples/ test/ docs/ CLAUDE.md README.md`.
5. Verify (must create `results/data/{W.jls,R.jls,R_coefficients.csv}` and `results/summary.txt`):
   ```bash
   julia --project=examples/01_clamped_beam_ferrite -e '
     using Pkg; Pkg.develop(path="."); Pkg.instantiate();
     include("examples/01_clamped_beam_ferrite/main.jl")'
   ls examples/01_clamped_beam_ferrite/results/data
   ```

### 3.2 `02_clamped_beam_gridap` — writes `./equations.txt` into the CWD

1. Rename entry script to `main.jl` (as in 3.1).
2. Replace the `open("./equations.txt", "w")` block target with
   `joinpath(dirs.data, "equations.txt")` using the helper, and add
   `save_rom` + `write_summary` as in 3.1.
3. README, grep for old name, verify run as in 3.1 (expect `equations.txt`
   additionally in `results/data/`).

### 3.3 `03_arch_comsol_wedge` — BROKEN: default config was archived

The script's default config is `results/mode_1_order_5_cnf/config.jl`, which no
longer exists (the `results/` folder went to the archive). Configs are *inputs*
and belong in the repo:

1. ```bash
   mkdir -p examples/03_arch_comsol_wedge/configs
   cp ../MORFE_results_archive/ArchComsolWedge/results/mode_1_order_5_cnf/config.jl \
      examples/03_arch_comsol_wedge/configs/mode_1_order_5_cnf.jl
   ```
   (If other archived run folders under `ArchComsolWedge/results/` contain
   distinct `config.jl` files, copy each as `configs/<run_name>.jl`.)
2. `mv examples/03_arch_comsol_wedge/arch_2_force.jl examples/03_arch_comsol_wedge/main.jl`
3. Edit `main.jl`: the config is now read from `configs/<name>.jl` and results are
   written to `results/<name>/` (run name = config file basename). Replace the
   current logic that derives `results_dir` from the config file's location:
   ```julia
   config_path = get(ARGS, 1, joinpath(@__DIR__, "configs", "mode_1_order_5_cnf.jl"))
   run_name    = splitext(basename(config_path))[1]
   results_dir = joinpath(@__DIR__, "results", run_name)
   mkpath(results_dir)
   ```
   Keep its existing logging/serialization (it already writes `W.jls`, `R.jls`,
   `summary.log`) but ensure they all target `results_dir`. Add a `summary.txt`
   via the shared helper (the existing `summary.log` stream may stay).
4. Update the usage docstring at the top of the file, README, grep for `arch_2_force.jl`.
5. Verify: run with no arguments; expect `results/mode_1_order_5_cnf/` populated.

### 3.4 `04_parametric_clamped_beam` — already the gold standard

1. `mv examples/04_parametric_clamped_beam/parametric_beam_demo.jl examples/04_parametric_clamped_beam/main.jl`
2. `grep -rn "parametric_beam_demo" examples/04_parametric_clamped_beam/ CLAUDE.md README.md`
   and update every hit (the `backbone/`, `validation/`, `plotting/` scripts may
   reference it).
3. Its results layout already matches the contract; switch its hand-rolled
   summary/CSV code to the shared helper ONLY if the output is byte-equivalent —
   otherwise leave the existing code untouched and just confirm `summary.txt`
   includes the Phase-2 environment fields (add them if missing).
4. Verify by running `main.jl`; expect `results/data/` + `results/figures/` populated.

### 3.5 `05_karman_vortex_street` — closest to compliant, longest runtime

1. Entry is already `main.jl`. Edit it so the run folder follows the contract:
   inside `results/Re%.2f_ord%d/` write into `data/` and `figures/` subfolders and
   a `summary.txt` (keep the existing `summary.log` tee). Keep all existing
   serialization, only re-point paths.
2. README to contract sections; note the ~30 min runtime and that `vtk_data.jls`
   enables ParaView export.
3. Verification: this run is expensive. Add (or use, if present) a config constant
   to lower the order, run once with the cheapest settings, confirm the file
   layout, then restore the default settings. If a cheap run is not possible,
   verify by code inspection and `julia --project=... -e 'include(...)'` is NOT
   required — state this in the commit message.

### 3.6 `06_dielectric_elastomer_actuator` — design notes only, no code

Do not delete and do not implement. Make its status explicit:

1. Create `examples/06_dielectric_elastomer_actuator/README.md` starting with:
   `> STATUS: DRAFT — design notes only (motivation.md, bibliography.md). Not yet runnable; exempt from the examples contract until an implementation lands.`
2. List it in the top-level `examples/README.md` table with status "draft".

### 3.7 `mesh_import` and `internals`

1. `mesh_import`: both `demo_*_to_gmsh.jl` scripts must write generated/converted
   meshes into `examples/mesh_import/<Comsol|Abaqus>/output/` (create with
   `mkpath`), not the CWD or next to the inputs. Add `output/` to `.gitignore`.
   Verify by running both scripts and checking `git status --porcelain` shows
   no new tracked-area files.
2. `internals/parametrisation_method/demo_parametrisation_method.jl`: re-point
   `output.h5` and the two `savefig` calls from `@__DIR__` into
   `joinpath(@__DIR__, "results")` (mkpath first). Print-only internals demos
   are compliant as-is.
3. Verify: `GROUP=examples julia --project test/runtests.jl` still passes.

Commit after each numbered sub-step (3.1 … 3.7).

---

## Phase 4 — Curate reference results back into the repo

The archived results are the *blessed reference outputs*. Small files return to
the repo, tracked, so every example ships with the expected answer; bulky
binaries stay in the archive repo.

**Selection rule:** copy a file into `results/reference/` only if it is ≤ 1 MB
and human-comparable (`summary*.txt`, `*.csv`, `reduced_dynamics*.txt`,
`equations.txt`, `*.png`, `*.md`). NEVER copy: `*.jls`, `*.h5`, `*.vtu`, `*.pvd`,
meshes, or anything from `IFX`/`Teo_COLSON`. Hard cap: 5 MB per example.

```bash
A=../MORFE_results_archive
# 03 — from the archived run(s)
mkdir -p examples/03_arch_comsol_wedge/results/reference/mode_1_order_5_cnf
cp "$A"/ArchComsolWedge/results/mode_1_order_5_cnf/summary.log \
   examples/03_arch_comsol_wedge/results/reference/mode_1_order_5_cnf/ 2>/dev/null
# 04 — figures and data CSVs from the archived results
mkdir -p examples/04_parametric_clamped_beam/results/reference
cp "$A"/ParametricClampedClampedBeam/results/data/summary.txt \
   "$A"/ParametricClampedClampedBeam/results/data/R_coefficients.csv \
   "$A"/ParametricClampedClampedBeam/results/data/validation_metrics.csv \
   examples/04_parametric_clamped_beam/results/reference/ 2>/dev/null
mkdir -p examples/04_parametric_clamped_beam/results/reference/figures
cp "$A"/ParametricClampedClampedBeam/results/figures/*.png \
   examples/04_parametric_clamped_beam/results/reference/figures/
# 05 — whatever small text/csv/png exists under the archived Re49.03_ord5 run
# (inspect first: find "$A/KarmanVortexStreet/results" -size -1M -type f)
```

For each copy: check the size cap, then inspect any remaining archived files for
that example and apply the selection rule — the lists above are the known
candidates, not necessarily exhaustive.

**01 and 02 have no archived results** (01 never wrote any; 02 only wrote
`equations.txt`). Bless fresh runs instead: after Phase 3 verification runs,
copy the newly generated `summary.txt`, `R_coefficients.csv` (and 02's
`equations.txt`) into the respective `results/reference/`.

For the internals: copy the archived
`ParametrisationMethod/invariance_convergence_*.png` into
`examples/internals/parametrisation_method/results/reference/`.

Verify nothing oversized got tracked, then commit:

```bash
git add examples/*/results/reference examples/internals/*/results/reference 2>/dev/null
git ls-files --cached -s examples | grep reference | awk '{print $4}' | xargs du -k | sort -rn | head
git commit -m "Add curated reference results to examples"
```

---

## Phase 5 — Validation scripts and CI wiring

1. For 01, 02, 04: create `validate.jl` in the example folder:
   ```julia
   # Compare fresh R_coefficients.csv against results/reference/. Exit nonzero on mismatch.
   using DelimitedFiles
   fresh = readdlm(joinpath(@__DIR__, "results", "data", "R_coefficients.csv"), ',', skipstart = 1)
   ref   = readdlm(joinpath(@__DIR__, "results", "reference", "R_coefficients.csv"), ',', skipstart = 1)
   @assert size(fresh) == size(ref) "coefficient table size changed"
   maxrel = maximum(abs.(fresh .- ref) ./ max.(abs.(ref), 1e-12))
   println("max relative deviation vs reference: $maxrel")
   maxrel < 1e-6 || error("Results deviate from reference beyond tolerance")
   ```
   Adjust the tolerance to `1e-6` initially; if a verified-correct run fails only
   due to floating-point noise, relax to `1e-4` — never beyond.
2. In `test/runtests.jl`, extend the `GROUP == "examples"` block: after the
   internals includes, run example 01 end-to-end followed by its `validate.jl`
   (01 only — keep the group under ~10 min; 02–05 are validated manually).
3. Document in `examples/README.md`: how to run `validate.jl`, and that
   reference results are regenerated by deliberately re-blessing
   (`cp results/data/... results/reference/`) in a reviewed commit only.

Verify: `GROUP=examples julia --project test/runtests.jl`. Commit.

---

## Phase 6 — Top-level README and final verification

1. Rewrite `examples/README.md`: the contract (entry `main.jl`, `results/`
   layout, `summary.txt`, reference results, validation), the updated example
   table including `06_*` marked "draft", and a pointer:
   *"Historical result sets and large binaries live in the `MORFE_results_archive`
   repository (sibling folder, see its INDEX.md)."*
2. Final checks (all must pass):
   ```bash
   julia --project -e 'using Pkg; Pkg.test()'
   GROUP=examples julia --project test/runtests.jl
   grep -rn "demo_mechanical_problem\|arch_2_force\|parametric_beam_demo" \
        examples test docs README.md CLAUDE.md && echo "STALE REFS - FIX" || echo "OK"
   # every runnable example has main.jl + README + Project.toml:
   for d in examples/01_* examples/02_* examples/03_* examples/04_* examples/05_*; do
     for f in main.jl README.md Project.toml; do test -f "$d/$f" || echo "MISSING $d/$f"; done
   done
   # no large files tracked:
   git ls-files examples | xargs du -k 2>/dev/null | sort -rn | head -10
   ```
3. `git log --oneline main..HEAD`, report for human review. Do not merge or push
   without explicit approval.

---

## Summary of decisions

| Question | Decision |
|---|---|
| Entry point | `main.jl` everywhere (03/05 accept a config argument) |
| Output location | `results/` inside the example; `data/` + `figures/` + `summary.txt`; config-driven examples nest under `results/<run_name>/` |
| "All must generate results" | 01 gains output code; 02 redirected; 03 repaired (configs restored from archive as tracked inputs); 06 explicitly exempt as draft |
| Archived results home | `../MORFE_results_archive` — its own git repo with `INDEX.md` provenance; user advised to push to private remote/LFS |
| Important small results | Curated back into the repo as tracked `results/reference/` (≤1 MB/file, ≤5 MB/example), enabling `validate.jl` regression checks |
| Large binaries (`.jls`, `.h5`, `.vtu`, meshes, IFX) | Stay in the archive repo, never re-tracked |
