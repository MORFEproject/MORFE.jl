# Handoff — `parametrise` / spectral-layer refactor

Read this top to bottom before touching anything. It is written to be followed literally.

**Current state: green.** `main` is clean. 2762 tests pass (the count varies — see §7).
All coefficient gates are bit-identical. MORFEFerrite precompiles, its suite passes, and both
SVK gates pass.

---

## 0. What this refactor is for, in one paragraph

`parametrise` used to take eleven arguments, five of which were separate spectral arrays
(`master_eigenvalues`, `master_modes`, `master_modes_derivatives`, `left_eigenmodes`,
`left_modes_derivatives`) that the caller had to keep mutually consistent by hand. The goal is
**one way to do each thing**: `parametrise(model, spectral_data, expansion_order)`, where
`SpectralData` is a single object holding all the spectral input, and each pipeline step is a
separate function so a new policy is a new method rather than a new branch.

**Hard rules for all work below.** These are acceptance criteria, not aspirations:

1. **Accuracy: bit-identical.** `(W, R)` coefficients must compare `==`, not `≈`. This refactor
   changes plumbing, never mathematics. A `≈` comparison hides exactly the index-swap and
   sign-flip bugs this work is most likely to cause.
2. **Performance: no increased cost.** Not in time, not in allocations. The user has been
   explicit about this more than once. If a change adds cost, either remove the cost or do not
   make the change.

---

## 1. What has already landed

Eleven commits on `main`, `d8aef4d` … `48b1c99` (your own `ae1f749` is interleaved). Each
commit message explains its reasoning — **read them rather than re-deriving**.

| Commit | What |
|---|---|
| `d8aef4d` | `ParametrisationObjects` split out; `parametrise` moved into its own module; `parametrise_entry.jl` deleted; `build_multiindex_set` dispatch seam added |
| `bca437e` | The two left-eigenvector block builders unified behind one recurrence (`apply = adjoint \| identity`) |
| `01d44a3` | Conjugate-permutation detection moved to the spectral layer |
| `a655692` | `SpectralData` + `ModeBundle` — five loose spectral arrays become one object |
| `0875b24` | `ResonanceConfig`; separating inner/outer tolerances fixed a per-target bounds error |
| `a4bbcb3` | `Eigensolvers` + `Eigenproblems` merged into `SpectralDecomposition`; API renamed |
| `84cd712` | Flattened to one module; `Spectrum.jl` folded into `Eigensolvers.jl` |
| `f13b8e8` | **`parametrise(model, spectral_data, expansion_order)`** — the unified entry point |
| `59a7335` | One conjugate involution, detected once over the whole spectrum |
| `d2d2ee7` | Restrictions stored rather than re-derived; per-solve overhead 832 → 608 B |
| `48b1c99` | Eigenvalue-spacing guard made allocation-free (6832 → 64 B) |

### The API as it now stands

```julia
sp   = spectrum(model; solver = DefaultEigensolver())   # was solve_eigenproblem
sd   = SpectralData(model, sp; master = [1, 2])         # pure selection, no mutation
W, R = parametrise(model, sd, 5)                        # 5, or a MultiindexSet
```

Renames already done: `Eigenproblem`→`Spectrum`, `solve`/`solve_left`→`eigensolve`/
`eigensolve_left`, `solve_eigenproblem`→`spectrum`, `get_eigenpairs` deleted (read the fields).

---

## 2. How to verify — run these, do not skip them

Harnesses live in **`benchmark/refactor_gates/`**. They are untracked; commit or gitignore as
you prefer. They use `DefaultEigensolver` (LAPACK) deliberately so they are deterministic
across sessions — Arpack's eigenvector gauge is not reproducible
(see the `project_karman_demo_validation` memory).

```bash
cd /Users/tiago/Desktop/AM-TUM/Code/MORFE_jl

# GATE 1 — accuracy + allocations. MUST print "GATE PASSED".
julia --project=test benchmark/refactor_gates/baseline.jl /tmp/cand
julia --project=test benchmark/refactor_gates/compare.jl \
      benchmark/refactor_gates/baseline_HEAD /tmp/cand

# GATE 2 — full test suite. NOTE: --project=test, NOT --project (see §7).
GROUP=all julia --project=test test/runtests.jl

# GATE 3 — old path vs SpectralData path, same session (no cross-run noise)
julia --project=test benchmark/refactor_gates/differential.jl

# GATE 4 — unified entry point ≡ old entry point
julia --project=test benchmark/refactor_gates/unified.jl

# GATE 5 — is a cost O(1), or does it scale with problem size?
julia --project=test benchmark/refactor_gates/allocscale.jl
```

The five baseline models cover: dense, sparse/KLU, conjugate-symmetry, external-system, and
ORD-mismatch (an ORD-3 model fed by an ORD-2 eigenproblem).

MORFEFerrite, from `/Users/tiago/Desktop/AM-TUM/Code/MORFEFerrite/MORFEFerrite.jl`:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
# gates via the EXAMPLE env, NOT Pkg.test — see the project_svk_gate_env_failure memory
julia --project=examples/01_clamped_beam_ferrite test/StructuralSVK/run_gates.jl
```

Gate A and Gate B have reported **1.0149555905989023e-9** and **6.501646250144121e-10**,
unchanged through every commit. They are the sharpest check available. If those numbers move,
something real changed — investigate, do not adjust the threshold.

**Formatting, before every commit:**

```bash
julia --project=format format/format.jl
git checkout -- examples/internals/full_order_model/main.jl   # see §7
```

---

## 3. Outstanding work

Do them in this order. Changes 2 and 3 are additive and keep both repos green. Change 4 is
breaking and must land in one pass.

---

### CHANGE 2 — make the off-manifold warning generic *and cheap*

**File:** `src/ParametrisationMethod/Resonance.jl`, function `_warn_outer_resonances`.

**Two problems, fix both together.**

#### 2a. It warns once per eigenvalue; it should warn once per conjugate pair

Everything needed already exists (added in `59a7335`):

| Helper | Gives you |
|---|---|
| `outer_conjugate_permutation(sd)` | σ restricted to the outer bundle, re-indexed to `1:n_outer` |
| `physical_mode(sd, i)` | the physical mode number of spectrum entry `i` |
| `spectrum_entries(sd, p)` | the (possibly non-consecutive) entries making up mode `p` |
| `indices(bundle)` | the source spectrum positions of a bundle's modes |

Group the flagged outer targets by that involution and emit one `@warn` per pair. Target text:

```
Monomials are near-resonant with outer physical mode pair 2 (spectrum entries 3, 7;
λ = -1.688e-04 ± 3.376e+00i). That mode is not on the manifold, so its direction is solved
through a near-singular operator and the ROM will lose accuracy there regardless of how the
load is shaped. Offending monomial exponents: (1, 0), (0, 1). Add mode 2 to the master set,
detune the forcing, or add damping.
```

Say **"outer"**, not "non-master" — "outer" is the term used everywhere else in the codebase
(`outer_eigenvalues`, `outer_resonances`, `OuterResonanceCondition`, `sd.outer`).

Report **both** the physical mode number and the spectrum entries. They answer different
questions: the mode number is what a user adds to `master`; the entries are what they index in
their own spectrum. They coincide only under an adjacency assumption that need not hold.

#### 2b. It costs 95× and is on by default — make it cheap, do NOT switch it off

Measured (`n = 30`, `|mset| = 35`, 58 outer modes):

| `build_resonance_set` | bytes |
|---|---|
| `warn_outer = false` | 2 608 |
| `warn_outer = true` (**the default**) | 249 008 |

It builds an **entire second `ResonanceSet`** as a probe over every outer eigenvalue, so the
cost is `O(NMON × n_outer)` — it *scales*. MORFE's `parametrise` never warned before this
refactor, so this is a new cost.

> ⚠️ **`warn_outer = true` must remain the default.** The user has said explicitly that this
> warning is important. Do not "fix" the cost by disabling it. Fix it by not building a second
> set: either reuse the set `build_resonance_set` has already constructed (when
> `config.outer_targets` is on, the outer block is already there), or compute only the outer
> block instead of a full `ResonanceSet`.

#### 2c. Then delete MORFEFerrite's duplicate

`MORFEFerrite/src/StructuralSVK/parametrise.jl`: delete `_warn_outer_resonances` and its
`resonances` helper — both re-implement detection MORFE now does — and stop passing
`warn_outer = false` there.

> ⚠️ **MORFEFerrite's outer-resonance testset is the specification.**
> `test/StructuralSVK/test_structural_svk.jl:238-268`. It must pass **completely unchanged**:
> - the eight `occursin("mode pair 2", …)` checks match a substring of `"physical mode pair 2"`;
> - the `length(warns) == 1` counts are exactly what per-pair grouping produces;
> - the monomial-exponent checks depend only on exponent formatting, which is unchanged.
>
> A failure there is a regression in your implementation, **not** a test to update.

> ⚠️ SVK is being rewritten separately by the user ("SVK will be redone"). Coordinate 2c with
> that work before deleting anything in `StructuralSVK/`.

**Done when:** MORFE tests green; MORFEFerrite tests green with the outer-resonance testset
untouched; `build_resonance_set` with `warn_outer = true` costs approximately what
`warn_outer = false` costs; GATE 1 still bit-identical.

---

### CHANGE 3a — add the missing `show` for `NthOrderModel`

**File:** `src/FullOrderModel/FullOrderModel.jl`.

There is no `Base.show` for `NthOrderModel`, so the REPL dumps every field including the entire
Ferrite `DofHandler`. Add:

```julia
Base.show(io::IO, ::MIME"text/plain", m::NthOrderModel)
```

Copy the established house style from `MORFEFerrite/src/StructuralSVK/types.jl:71` — a header
line, then 2-space-indented rows with keys padded to a fixed width.

Everything needed is already on the type: `n_fom` (= FOM), `ORD`, `N_EXT`, `max_nl_degree`, and
the type parameter `MT` (which is where sparse-vs-dense reads off, e.g.
`SparseMatrixCSC{Float64,Int64}` vs `Matrix{Float64}`).

**Reuse, do not rewrite:** `FullOrderModel._term_label(t)` (line ~31) gives a one-line
`"MultilinearMap @ file.jl:123"` label that works for both `MultilinearMap` and
`FEMMultilinearMap` and never throws. `MultilinearMaps._symmetry_label` and `_call_signature`
complete the set.

**Done when:** `show(stdout, MIME"text/plain"(), model)` prints a readable summary; no test
changes needed; GATE 1 unaffected.

---

### CHANGE 3b — `print_setup`, called by `parametrise`

**Files:** `src/ParametrisationMethod/ParametrisationMethod.jl` (the function and the kwarg).

```julia
print_setup(io::IO, model, spectral, mset, resonance)   # a normal, overloadable function
parametrise(model, sd, expansion_order; verbose = true, setup_io = stderr, …)
```

`parametrise` receives an `NthOrderModel`, and **every backend's model is that same type**, so
there is nothing backend-specific to dispatch on inside it. Print only what it can genuinely
see. Backends wanting richer output print their own summary before calling — they already have
`show` methods for that.

Target output:

```
======================================================================
MORFE parametrisation
----------------------------------------------------------------------
  model       : FOM = 1734,  ORD = 2,  sparse (SparseMatrixCSC{Float64})
  nonlinear   : FerriteGeometricNonlinearity @ ferrite_assembly.jl:88  (deg 2)
                FerriteGeometricNonlinearity @ ferrite_assembly.jl:88  (deg 3)
  external    : N_EXT = 2   (λ = ±3.376e+00i)
  reduced     : ROM = 2,  NVAR = 4
  masters     : -1.688e-04 ± 3.376e+00i
  expansion   : total degree ≤ 5   →   125 monomials
  resonance   : complex_normal_form,  tol = 0.05
  conjugate   : master [2, 1]  (+ external, derived)
======================================================================
```

Style reference: `MORFEFerrite/examples/10_turbine_blade/Blade/setup/logging.jl`. ASCII `=`/`-`
rules (never unicode box-drawing in *printed* output), title line, padded-key rows, `@printf`
for numeric columns.

> ⚠️ **Gate the output exactly like `_make_progress`**
> (`src/ParametrisationMethod/CohomologicalEquations/CohomologicalEquations.jl:205`): decide
> once, at the top, and write to `stderr` only when `stderr isa Base.TTY`. This keeps test logs
> and redirected output clean, matches how the progress reporter already behaves, and means
> adding this changes **no** existing script's captured output.

> ⚠️ Building the strings must cost nothing when output is off. Compute the message **inside**
> the `if enabled` branch. See §5 — `@info`/string interpolation in a loop boxes the variables
> it captures even when nothing is printed.

**Done when:** a TTY run prints the banner; a redirected run prints nothing; GATE 1 and GATE 2
unchanged.

---

### CHANGE 4 — delete the superseded API (BREAKING — one pass, no partial states)

| Delete | Replacement |
|---|---|
| `parametrise(model, order, spectrum)` | `parametrise(model, sd, expansion_order)` |
| `solve_cohomological_problem(model, mset, λ, Ψ, ℓ, rset; …)` | `solve_cohomological_problem(model, mset, sd, rset; …)` |
| `select_master_modes_by_hand` / `_by_sorting` / `_by_target_frequency` | `SpectralData(model, sp; master = …)` |

Also in this change:

- `Spectrum` becomes **immutable** and loses `master_modes` (that field existed only so the
  mutators had somewhere to write) and `external_modes` (dead — never written anywhere in
  `src/`, `ext/`, `test/` or `examples/`).
- Replace the mutators with **pure** helpers that *return* indices, e.g. `master_by_sorting(n)`
  and `master_by_target_frequency(sp, freqs, tol)`.
- Delete the unconditional `println("Chosen mastermodes: …")` in
  `src/SpectralDecomposition/Eigensolvers.jl` — the only stray `stdout` print in library code.

**Why the old API is worth deleting** (in case you are asked): they are not merely equivalent —
for the solve, one is *implemented in terms of* the other. The `SpectralData` method's body
unpacks the bundle into the five arrays and calls the positional method. And
`parametrise(model, x, y)` currently means different things depending on whether `x` is an
`Integer` or a `SpectralData` — the 2nd and 3rd positional arguments are **swapped** between
the two methods.

**Scope: 35 `solve_cohomological_problem` call sites across 17 files** — `test/` (6),
`examples/` (4), `benchmark/` (6), `notebooks/` (1) — plus MORFEFerrite. Most of them hand-roll
exactly the slicing `SpectralData` now does, so they get *shorter*.

> ⚠️ **This was attempted in an earlier session and deliberately reverted. Read this before
> retrying, it will save you the same dead end:**
>
> - Deleting the positional method and making `SpectralData` the primary implementation takes
>   the per-solve overhead from **+608 B to +112 B**. Measured, constant across problem sizes.
> - Keeping the old signature as an **adapter** onto the new primary does **not** work:
>   reassembling the order-blocks costs legacy callers **+18 KB** on the sparse model. An
>   adapter *relocates* the cost onto 35 call sites rather than removing it — and that violates
>   the no-increased-cost rule from the other direction.
> - **Therefore the deletion and the migration must land together.** There is no safe
>   intermediate state. Do not commit a half-migrated tree.

**Completion check.** Across **both** repos this must return only docstring/CHANGELOG hits.
Exclude `graphify-out/`, which floods repo-wide greps (`feedback_grep_excludes_graphify`):

```bash
grep -rn --include=*.jl \
  -e select_master_modes -e master_modes_derivatives \
  -e left_modes_derivatives -e get_eigenpairs . | grep -v graphify-out
```

**Done when:** that grep is clean in both repos; GATE 1–5 pass; MORFEFerrite tests and both SVK
gates pass at the same numbers as §2; `allocscale.jl` shows the overhead gone.

---

## 4. Regressions introduced so far — do not add to this list

### Accuracy: none

Every stage was gated bit-identical on `(W, R)` across all five models, and the SVK gates have
not moved a digit. The one change that could have moved numbers — routing the structural
left-block builder through the general recurrence — is bit-identical by construction because
that path passes `apply = identity` (`bca437e`).

*Latent, and not currently triggered:* enabling `conjugate_permutation = :detect` where symmetry
was previously off changes which monomials are solved versus filled by conjugation. Results then
agree to round-off, **not** bit-for-bit. It defaults to `nothing`, so nothing has changed — but
any migration that switches models to `:detect` must expect this and re-baseline deliberately.

### Cost: two open, one fixed

| # | Issue | Status |
|---|---|---|
| a | Off-manifold warning builds a second full `ResonanceSet`: 2 608 → 249 008 bytes, `O(NMON × n_outer)`, **on by default** | **OPEN** — fix in Change 2b, keeping `warn_outer = true` |
| b | `SpectralData` path costs **+608 B per solve** vs passing arrays directly. Constant, O(1), outside the graded loop | **OPEN** — drops to +112 B when Change 4 deletes the positional method |
| c | Eigenvalue-spacing guard materialised `ROM(ROM-1)/2` distances to take a minimum: 6832 B at ROM = 20 | **FIXED** in `48b1c99` (now 64 B) |

Deliberate and documented, not regressions to chase:

- `ModeBundle` caches `right_physical`/`left_physical` — `2·FOM·ROM` complex numbers duplicated
  from the blocks. Keeps BLAS paths and avoids per-call slicing; small beside the
  `FOM × ORD × L` parametrisation. Documented in the type's docstring.
- `:detect` verifies eigenvectors — `O(FOM · ORD · ROM)` extra work versus checking eigenvalues
  alone. Eigenvalue pairing is necessary but not sufficient, and a wrong permutation *silently*
  corrupts `W` and `R`. Opt-in.

### UI/UX: three open, all removed by Change 4

- **`parametrise(model, x, y)` means two different things** depending on argument types, with the
  2nd and 3rd positionals swapped between methods. Worst of these. Change 4 removes it.
- **`SpectralData.conjugate_permutation` silently changed meaning** in `59a7335`: it was the
  ROM-length master block, it is now the involution over the whole spectrum. Read it through
  `master_conjugate_permutation(sd)`. In-repo callers were updated; external code was not.
- **`ModeBundle` gained type parameters** (`RD`, `LD` for the cached derivative views), so
  `ModeBundle{ORD, EV}` no longer names the type.
- Renames have **no deprecation shims** — the old names simply do not exist.

---

## 5. Performance lessons learned the hard way

Do not re-derive these; they cost real time to establish.

- **`@info`/`@warn` with string interpolation inside a loop boxes the loop variables it
  captures, on every iteration, even when the branch is never taken and nothing is logged.**
  This made a supposedly cheaper early-exit loop (3104 B) *worse* than the comprehension it
  replaced. Split the search into its own function and log outside the loop — that took it to
  64 B. See `_first_close_pair` in `Resonance.jl`.
- **A comprehension `[f(i,j) for i in …, j in …]` reduced immediately by `minimum` allocates the
  whole array.** Use a generator `(f(i,j) for …)`, or better, short-circuit if you only need a
  yes/no answer.
- **The +608 B on the `SpectralData` path is the forwarding itself**, not the accessors. All of
  these were tried and changed nothing: caching the views, parametrising the `Union{Nothing,…}`
  fields concrete, marking the forwarder `@inline` (the callee is too large to inline). Only
  deleting the positional method helps.
- **Always check whether a cost is O(1) or scales** with `allocscale.jl` before deciding it
  matters. +608 B constant on a 546 KB solve is 0.1%; the warning's `O(NMON × n_outer)` is not.

---

## 6. Design decisions worth not reversing

- **Right and left blocks index oppositely.** Right physical is `[:, 1, :]` with derivatives
  `2:ORD`; left physical is `[:, ORD, :]` with orthogonality blocks `1:(ORD-1)`. Both arrays
  have the same shape, so a swap is type-correct and compiles silently. The convention is
  encoded **only** in the `ModeBundle` constructor and four accessors — keep it that way.
  `check_biorthogonality(sd, model)` is the numerical guard: `φᴴBψ = δ` fails loudly under any
  swap. It is diagnostic-only and must not be called from the solve.
- **ORD reconciliation: slice when the orders match, extend only when they don't.** When the
  model has the higher order (augmented `(K,C,M,0)` fed by a 2nd-order eigenproblem), missing
  right blocks are `λ ·` the **last available** block — *not* a fresh `λ^{k-1}ψ`, because the
  eigensolver's own block carries the numerical content. Left blocks are rebuilt against
  `model.linear_terms`. Tests assert both reproduce the hand-rolled example code exactly.
- **Physical mode numbers come from σ's orbits, never `⌈i/2⌉`.** Conjugate partners need not be
  adjacent — a shift-invert or filtered solve can return `{1, 5}` as a pair. Numbering by first
  appearance agrees with the adjacent case and stays correct otherwise.
- **`apply = identity` on the structural left-block path.** For exactly symmetric `B`, `B*x` and
  `B'*x` *are* bit-identical (measured, dense and sparse). The risk is that assembled `M`, `C`
  need not be exactly symmetric — one ulp of asymmetry gives ~4.4e-16 — so `identity` keeps that
  path bit-for-bit unconditionally instead of relying on the assembler.
- **`tol` defaults to `nothing`, not `1e-2`.** The `@info` guards fire on explicitly-set values;
  with a numeric default a plain `ResonanceConfig()` would emit "tolerance unused" on every run,
  and guards that cry wolf get ignored.
- **`multiindex_set` / `coefficients` are distinct generic functions** in `Polynomials` and in
  `ParametrisationObjects`. Same names, different functions, deliberately not merged.

---

## 7. Environment gotchas that will cost you time

- **`julia --project test/runtests.jl` does not work.** The test dependencies (Symbolics, HDF5,
  …) live in `test/Project.toml`. Use **`--project=test`**. The command in `CLAUDE.md` is wrong.
- **The test count varies run to run** (2745–2790). `test/Utils/test_multiindices.jl` randomises
  how many assertions it makes. A differing total is **not** a skipped testset.
- **The formatter re-indents `examples/internals/full_order_model/main.jl` every run** — that
  file is tab-indented on `main` and the pinned formatter wants spaces. Revert it after
  formatting; it is unrelated collateral, not your change.
- **`CLAUDE.md` is gitignored** in this repo, so edits to it stay local and never commit.
- **`docs/` does not exist.** The documentation generator is
  `website/generate_documentation.jl`, and **a new submodule must be added to its `MODS`
  constant or its docstrings are silently dropped from the site.**
- **A `#` comment between a docstring and its definition silently voids the docstring**
  (`project_docstring_traps`). Two docstrings in a row is a hard error
  ("cannot document the following expression").
- **`NthOrderModel` cannot infer `ORD` from an empty nonlinear-terms tuple** — give it at least
  one term. And a term's multiindex must have `ORD` entries: `(3, 0)` for ORD=2, `(3, 0, 0)` for
  ORD=3.
- **MORFEFerrite's working tree contains uncommitted work of the user's** — multi-harmonic
  forcing in `types.jl`, `parametrise.jl`, `fluid_maps.jl`, and several `examples/*/main.jl` —
  interleaved with the rename tracking from this refactor. **Never run `git add -A` there.**
  Stage named files only, and confirm with the user before committing anything in that repo.
- **`git add -A` in MORFEFerrite also times out** — the `examples/` tree contains large meshes
  and result archives.
- **Run `graphify query "<question>"` before repo-wide greps** (a hook enforces this), and
  always exclude `graphify-out/` from grep results.

---

## 8. Quick orientation — where things live

| What | Where |
|---|---|
| Module load order | `src/MORFE.jl` |
| Spectral layer (one module) | `src/SpectralDecomposition/` — `SpectralDecomposition.jl`, `Eigensolvers.jl`, `SpectralData.jl`, `ConjugatePermutation.jl` |
| `parametrise` + pipeline steps | `src/ParametrisationMethod/ParametrisationMethod.jl` |
| Coefficient containers + `mset` contract | `src/ParametrisationMethod/ParametrisationObjects.jl` |
| The solve | `src/ParametrisationMethod/CohomologicalEquations/CohomologicalDriver.jl` |
| Resonance + `ResonanceConfig` | `src/ParametrisationMethod/Resonance.jl` |
| MORFEFerrite `build_model` contract | `../MORFEFerrite/MORFEFerrite.jl/src/common/assembled_model.jl` |

`ParametrisationObjects` must load **before** `CohomologicalEquations` (which imports the
containers); `ParametrisationMethod` must load **after** it (because `parametrise` calls the
solver). That split is what removed the old `parametrise_entry.jl` hack — do not undo it.
