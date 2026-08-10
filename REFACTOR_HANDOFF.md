# Handoff — `parametrise` / spectral-layer refactor

**The refactor is complete.** Changes 2, 3a, 3b and 4 all landed; the only deferred item is
2c, which lives in MORFE's sibling repo and is written up there
(`../MORFEFerrite/MORFEFerrite.jl/MORFE_API_MIGRATION.md`).

**Current state: green.** 2786 tests pass (the count varies — see §7). Coefficients are
bit-identical to the pre-refactor archive on all five gate models. MORFEFerrite's suite
passes and both SVK gates read the numbers they have read all along:
**1.0149555905989023e-9** and **6.501646250144121e-10**.

---

## 0. What this refactor was for, in one paragraph

`parametrise` used to take eleven arguments, five of which were separate spectral arrays
(`master_eigenvalues`, `master_modes`, `master_modes_derivatives`, `left_eigenmodes`,
`left_modes_derivatives`) that the caller had to keep mutually consistent by hand. The goal was
**one way to do each thing**: `parametrise(model, spectral_data, expansion_order)`, where
`SpectralData` is a single object holding all the spectral input, and each pipeline step is a
separate function so a new policy is a new method rather than a new branch.

**The two acceptance criteria, which every commit was gated against:**

1. **Accuracy: bit-identical.** `(W, R)` coefficients compare `==`, not `≈`. This refactor
   changed plumbing, never mathematics. A `≈` comparison hides exactly the index-swap and
   sign-flip bugs this work was most likely to cause.
2. **Performance: no increased cost.** Not in time, not in allocations.

---

## 1. The API

```julia
sp   = spectrum(model; solver = DefaultEigensolver())
sd   = SpectralData(model, sp; master = master_by_sorting(2))   # or master = [1, 2]
W, R = parametrise(model, sd, 5)                                # 5, or a MultiindexSet
```

`SpectralData` can also be built from raw arrays — physical slices plus their companion
blocks, which is the shape every legacy caller already had:

```julia
sd = SpectralData(; eigenvalues = λ,
    right_modes = Ψ, right_derivatives = mmd,
    left_modes  = ℓ, left_blocks       = lmd,
    conjugate_permutation = [2, 1])     # ROM-length master block, optional
```

An `NVAR`-length permutation (one spanning external variables) is **not** a constructor
argument — the constructor validates a ROM-length master block. Pass it to `parametrise` /
`solve_cohomological_problem`, where it is used verbatim.

### Commits

| Commit | What |
|---|---|
| `d8aef4d` | `ParametrisationObjects` split out; `parametrise` moved into its own module; `parametrise_entry.jl` deleted; `build_multiindex_set` dispatch seam |
| `bca437e` | The two left-eigenvector block builders unified behind one recurrence |
| `01d44a3` | Conjugate-permutation detection moved to the spectral layer |
| `a655692` | `SpectralData` + `ModeBundle` — five loose spectral arrays become one object |
| `0875b24` | `ResonanceConfig`; separating inner/outer tolerances fixed a per-target bounds error |
| `a4bbcb3` | `Eigensolvers` + `Eigenproblems` merged into `SpectralDecomposition` |
| `84cd712` | Flattened to one module; `Spectrum.jl` folded into `Eigensolvers.jl` |
| `f13b8e8` | **`parametrise(model, spectral_data, expansion_order)`** — the unified entry point |
| `59a7335` | One conjugate involution, detected once over the whole spectrum |
| `d2d2ee7` | Restrictions stored rather than re-derived |
| `48b1c99` | Eigenvalue-spacing guard made allocation-free |
| `e0247d3` | **Change 2** — off-manifold warning: per conjugate pair, and 246 kB → 64 B |
| `a881ec2` | **Change 3a** — `show` for `NthOrderModel` |
| `c0d0135` | **Change 3b** — `print_setup`, called by `parametrise` |
| `819ef8b` | `NDOrderModel` → `NthOrderModel` (also carries the src half of Change 4) |
| `fac3fd1` | **Change 4** — the superseded API deleted, every call site migrated |

---

## 2. How to verify

Harnesses live in **`benchmark/refactor_gates/`**. They use `DefaultEigensolver` (LAPACK)
deliberately so they are deterministic across sessions — Arpack's eigenvector gauge is not
reproducible (see the `project_karman_demo_validation` memory).

```bash
cd /Users/tiago/Desktop/AM-TUM/Code/MORFE_jl

# GATE 1 — accuracy + allocations. MUST print "GATE PASSED".
julia --project=test benchmark/refactor_gates/baseline.jl /tmp/cand
julia --project=test benchmark/refactor_gates/compare.jl \
      benchmark/refactor_gates/baseline_change4 /tmp/cand

# The coefficient oracle, unchanged since before the refactor. Allocations differ (a
# different API is being measured) but every W and R must still read "identical".
julia --project=test benchmark/refactor_gates/compare.jl \
      benchmark/refactor_gates/baseline_HEAD /tmp/cand

# GATE 2 — full test suite. NOTE: --project=test, NOT --project (see §7).
GROUP=all julia --project=test test/runtests.jl

# GATE 3 — is a cost O(1), or does it scale with problem size?
julia --project=test benchmark/refactor_gates/allocscale.jl
```

`differential.jl` and `unified.jl` are gone: both compared the old path against the new one,
and the old path no longer exists.

The five baseline models cover: dense, sparse/KLU, conjugate-symmetry, external-system, and
ORD-mismatch (an ORD-3 model fed by an ORD-2 eigenproblem).

MORFEFerrite, from `/Users/tiago/Desktop/AM-TUM/Code/MORFEFerrite/MORFEFerrite.jl`:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
# gates via the EXAMPLE env, NOT Pkg.test — see the project_svk_gate_env_failure memory
julia --project=examples/01_clamped_beam_ferrite test/StructuralSVK/run_gates.jl
```

Gate A and Gate B report **1.0149555905989023e-9** and **6.501646250144121e-10**, unchanged
through every commit including Change 4. They are the sharpest check available. If those
numbers move, something real changed — investigate, do not adjust the threshold.

**Formatting, before every commit:**

```bash
julia --project=format format/format.jl
git checkout -- examples/internals/full_order_model/main.jl   # see §7
```

---

## 3. What each change did

### Change 2 — off-manifold warning (`e0247d3`)

It built an entire second `ResonanceSet` as a probe: `O(NMON × n_outer)` with a large
constant (a discarded inner block, `_superharmonics` allocating a temporary per monomial
twice, and `_local_index`'s `findfirst` making the outer build `O(NMON × n_outer²)`).
2 608 → 249 008 bytes, on by default.

It is now a direct scan — one distance test per (monomial, outer target), hits collected only
on the first flag. A quiet run costs **64 B, constant** in `|mset|` and `n_outer`. The
criterion is unchanged. Reusing the already-built outer block was rejected deliberately:
under `:real_normal_form` that block ORs each target with its conjugate, which would silently
change what is reported.

Warnings are now grouped **per conjugate pair** and name both the physical mode number and
the spectrum entries. A per-target `tol` vector used to make the probe throw — a live crash
whenever a vector tolerance met the default `warn_outer = true` — and now skips the scan
with a notice.

### Change 3a / 3b — display (`a881ec2`, `c0d0135`)

`show` for `NthOrderModel` (there was none, so a FEM model dumped its whole `DofHandler`),
and `print_setup`, an exported overloadable function that `parametrise` calls. Both print
only what the type already carries.

`print_setup` is gated exactly like `_make_progress`: decided once, and on the default
`stderr` only when `stderr isa Base.TTY`, so redirected output and test logs are unchanged.
An explicitly passed `setup_io` is always written to, which also makes it testable.

### Change 4 — the deletion (`819ef8b` src half, `fac3fd1` migration)

Deleted: `parametrise(model, order, spectrum)`, the positional
`solve_cohomological_problem`, the three `select_master_modes_*` mutators, `get_eigenpairs`,
the `Spectrum`-based `build_resonance_set`, and the stray `println` of the chosen master
modes. `Spectrum` is immutable and has lost `master_modes` and `external_modes`.

The `SpectralData` method **became** the implementation rather than staying a forwarder.
That is the whole point: an adapter in either direction relocates cost rather than removing
it (measured at +18 kB per legacy call on the sparse model).

Measured on the same model and monomial set, against the pre-promotion code:

| | before | after |
|---|---|---|
| no conjugate symmetry | 253 632 B | **253 168 B** |
| conjugate symmetry | 167 152 B | **154 304 B** |

The conjugate-path saving is the per-solve `@info` advisory, now `maxlog = 1`: static text
about the model's assumptions, so re-rendering it on every solve of a sweep cost ~12 kB and
said nothing new. The residual +160 B versus the *deleted* positional path is the bundle
unpacking a positional caller did at its own call site instead — in line with the +112 B this
document predicted.

---

## 4. Regressions — the list is now empty

### Accuracy: none

Every stage was gated bit-identical on `(W, R)` across all five models, and the SVK gates
have not moved a digit.

*Latent, and not currently triggered:* enabling `conjugate_permutation = :detect` where
symmetry was previously off changes which monomials are solved versus filled by conjugation.
Results then agree to round-off, **not** bit-for-bit. It defaults to `nothing`, so nothing
has changed — but any migration that switches models to `:detect` must re-baseline
deliberately.

### Cost: all three closed

| # | Issue | Status |
|---|---|---|
| a | Off-manifold warning built a second full `ResonanceSet` | **FIXED** in `e0247d3` (64 B, constant) |
| b | `SpectralData` path cost +608 B per solve | **FIXED** in `fac3fd1` (now below the deleted positional path except for +160 B on the autonomous case) |
| c | Eigenvalue-spacing guard materialised `ROM(ROM-1)/2` distances | **FIXED** in `48b1c99` |

Deliberate and documented, not regressions to chase:

- `ModeBundle` caches `right_physical`/`left_physical` — `2·FOM·ROM` complex numbers
  duplicated from the blocks. Keeps BLAS paths and avoids per-call slicing; small beside the
  `FOM × ORD × L` parametrisation.
- `:detect` verifies eigenvectors — `O(FOM · ORD · ROM)` extra work versus checking
  eigenvalues alone. Eigenvalue pairing is necessary but not sufficient, and a wrong
  permutation *silently* corrupts `W` and `R`. Opt-in.

### UI/UX

The old dual meaning of `parametrise(model, x, y)` is gone with the positional method.
Renames have **no deprecation shims** — the old names simply do not exist. External code
must be migrated; `MORFE_API_MIGRATION.md` in MORFEFerrite is the migration table.

---

## 5. Performance lessons learned the hard way

Do not re-derive these; they cost real time to establish.

- **`@info`/`@warn` with string interpolation inside a loop boxes the loop variables it
  captures, on every iteration, even when the branch is never taken.** See `_first_close_pair`
  in `Resonance.jl`.
- **A repeated `@info` is not free.** The conjugate-permutation advisory cost ~12 kB per
  solve just to render — more than the whole `SpectralData` overhead everyone was chasing.
  `maxlog = 1` for anything static.
- **A comprehension reduced immediately by `minimum` allocates the whole array.** Use a
  generator, or short-circuit if you only need a yes/no answer.
- **Always check whether a cost is O(1) or scales** with `allocscale.jl` before deciding it
  matters. +160 B constant on a 253 kB solve is 0.06%; the warning's `O(NMON × n_outer)` was
  not constant, which is what made it worth fixing.
- **A reformatted string literal changes what it costs to log.** An editor re-indenting a
  triple-quoted `@info` body lengthened the message and showed up as ~500 B per solve in the
  gate. The repo formatter does not touch string bodies, so it will not undo that for you.

---

## 6. Design decisions worth not reversing

- **Right and left blocks index oppositely.** Right physical is `[:, 1, :]` with derivatives
  `2:ORD`; left physical is `[:, ORD, :]` with orthogonality blocks `1:(ORD-1)`. Both arrays
  have the same shape, so a swap is type-correct and compiles silently. The convention is
  encoded **only** in the `ModeBundle` constructor, `_stack_blocks`, and four accessors — keep
  it that way. `check_biorthogonality(sd, model)` is the numerical guard; it is
  diagnostic-only and must not be called from the solve.
- **ORD reconciliation: slice when the orders match, extend only when they don't.** Missing
  right blocks are `λ ·` the **last available** block — *not* a fresh `λ^{k-1}ψ`. Left blocks
  are rebuilt against `model.linear_terms`.
- **Physical mode numbers come from σ's orbits, never `⌈i/2⌉`.** Conjugate partners need not
  be adjacent.
- **`tol` defaults to `nothing`, not `1e-2`.** Guards that cry wolf get ignored.
- **`multiindex_set` / `coefficients` are distinct generic functions** in `Polynomials` and in
  `ParametrisationObjects`. Same names, different functions, deliberately not merged.
- **A `Spectrum` is what the eigensolver computed, nothing more.** Which modes are master is
  a property of the `SpectralData` built from it. That is why `Spectrum` is immutable and the
  selectors return indices instead of writing a mask back.

---

## 7. Environment gotchas that will cost you time

- **`julia --project test/runtests.jl` does not work.** Test dependencies live in
  `test/Project.toml`. Use **`--project=test`**. The command in `CLAUDE.md` is wrong.
- **The test count varies run to run.** `test/Utils/test_multiindices.jl` randomises how many
  assertions it makes. A differing total is **not** a skipped testset.
- **The formatter re-indents `examples/internals/full_order_model/main.jl` every run** — that
  file is tab-indented on `main` and the pinned formatter wants spaces. Revert it after
  formatting.
- **`grep` here is `ugrep`.** `-Z` means *fuzzy match*, not NUL-separated, so
  `grep -rlZ … | xargs -0` silently does nothing useful. And zsh does not word-split an
  unquoted `$var` in a `for` loop. Both cost a wasted edit pass; use Python for bulk rewrites.
- **`CLAUDE.md` is gitignored** in this repo, so edits to it stay local.
- **`docs/` does not exist.** The documentation generator is
  `website/generate_documentation.jl`, and **a new submodule must be added to its `MODS`
  constant or its docstrings are silently dropped from the site.**
- **A `#` comment between a docstring and its definition silently voids the docstring**
  (`project_docstring_traps`).
- **`NthOrderModel` cannot infer `ORD` from an empty nonlinear-terms tuple** — give it at least
  one term. A term's multiindex must have `ORD` entries.
- **MORFEFerrite's working tree contains uncommitted work of the user's.** **Never run
  `git add -A` there** — and `git add -A` also times out, because `examples/` holds large
  meshes and result archives.
- **Run `graphify query "<question>"` before repo-wide greps** (a hook enforces this), and
  always exclude `graphify-out/` from grep results.

---

## 8. Quick orientation — where things live

| What | Where |
|---|---|
| Module load order | `src/MORFE.jl` |
| Spectral layer (one module) | `src/SpectralDecomposition/` |
| `parametrise`, `print_setup`, pipeline steps | `src/ParametrisationMethod/ParametrisationMethod.jl` |
| Coefficient containers + `mset` contract | `src/ParametrisationMethod/ParametrisationObjects.jl` |
| The solve | `src/ParametrisationMethod/CohomologicalEquations/CohomologicalDriver.jl` |
| Resonance + `ResonanceConfig` + the off-manifold scan | `src/ParametrisationMethod/Resonance.jl` |
| MORFEFerrite migration table + the deferred Change 2c | `../MORFEFerrite/MORFEFerrite.jl/MORFE_API_MIGRATION.md` |

`ParametrisationObjects` must load **before** `CohomologicalEquations` (which imports the
containers); `ParametrisationMethod` must load **after** it (because `parametrise` calls the
solver). That split is what removed the old `parametrise_entry.jl` hack — do not undo it.
