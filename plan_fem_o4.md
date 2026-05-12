# O4 Plan: Combined Element Loop Across All FEM Terms

## Context and Motivation

After O1–O3, the `_replay_fem_split!` path in `MultilinearTerms.jl` executes one element loop
per `(term, split)` pair. For a model with a quadratic (`term_quad`, DEG=2) and a cubic
(`term_cubic`, DEG=3) SVK term, a degree-3 monomial triggers two separate element loops:

```
for element in fem_elements(term_quad)
    fem_reinit!(element, term_quad)   # reinit #1
    scatter_qp!(...)                   # scatter unique cols for quad split
    accumulate_qp!(...)
    assemble_element!(...)
end

for element in fem_elements(term_cubic)
    fem_reinit!(element, term_cubic)  # reinit #2  ← redundant
    scatter_qp!(...)
    accumulate_qp!(...)
    assemble_element!(...)
end
```

Both terms share the same mesh and the same `cv` object, so `fem_reinit!` (which calls
`reinit!(cv, element)` and recomputes all shape-function values/gradients) is executed twice
per element per monomial. Additionally, if both splits happen to reference the same W-column
(same `(order, col)` pair), `scatter_qp!` will call `function_gradient` for that column twice.

O4 eliminates this redundancy by merging all FEM term loops for a given monomial into a
**single element traversal**: `reinit!` once, scatter each globally-unique W-column once,
then dispatch `accumulate_qp!` for every entry from every term at each quadrature point.

**Expected gain:** For SVK with 2 FEM terms, the dominant `reinit!` cost is halved per
monomial. For higher-degree material models with `n_fem_terms > 2` the saving scales linearly.

---

## Scope

**In scope:**
- All FEM terms sharing the same mesh (same `fem_elements` iterator and `cv`).
- Terms with `multiplicity_external = 0` (no external forcing). This covers all SVK terms.
- Mixed DEG (quadratic + cubic simultaneously).

**Out of scope / deferred:**
- Terms with `multiplicity_external > 0` (external splits). The current per-term path
  handles these correctly; they can be excluded from the combined loop and processed as
  before (see §External Splits below).
- Terms using different meshes or different `cv` objects (can still be handled with a
  per-cv-group approach; see §Multiple CellValues Groups below).

---

## Current State (after O1–O3)

Relevant types and functions:

| Symbol | Location | Description |
|--------|----------|-------------|
| `FEMCachedSplit{DEG}` | `MultilinearTerms.jl` | One `(monomial, term, ext-split)` triple; holds `unique_cols` and `fem_entries` |
| `FEMFactorisationEntry{DEG}` | `MultilinearTerms.jl` | One factorisation entry; holds `multiplier` and `local_factor_indices` into `unique_cols` |
| `MultilinearTermsCache` | `MultilinearTerms.jl` | Holds `fem_splits[l][t_idx]` and `fem_Fe` |
| `_replay_fem_split!` | `MultilinearTerms.jl` | One element loop per `(term, split)` |
| `fem_reinit!` | `MultilinearMaps.jl` + `ferrite_assembly.jl` | Calls `reinit!(cv, element)` once per call |

---

## New Data Structures

### `FEMGlobalEntry{DEG}`

One accumulated entry in the combined loop: a single factorisation from a single term,
with factor indices remapped into the global (across-all-terms) unique-column table.

```julia
"""
    FEMGlobalEntry{DEG}

One factorisation entry in the combined element loop for a given monomial.

- `term_idx`             — index into `model.nonlinear_terms`
- `multiplier`           — symmetry count (from `FEMFactorisationEntry`)
- `local_factor_indices` — NTuple{DEG,Int}: indices into `FEMGlobalSplit.global_unique_cols`
"""
struct FEMGlobalEntry{DEG}
    term_idx::Int
    multiplier::Int
    local_factor_indices::NTuple{DEG, Int}
end
```

`NTuple{DEG,Int}` ensures that `ntuple(k -> ∇W_qp[entry.local_factor_indices[k], q], Val(DEG))`
is unrolled at compile time when DEG is a type parameter.

### `FEMGlobalSplit`

All pre-computed data for one monomial, covering all FEM terms simultaneously.

```julia
"""
    FEMGlobalSplit

Precomputed combined-loop bookkeeping for one monomial.

- `global_unique_cols` — deduplicated (derivative_order, W_col_idx) pairs across ALL FEM
                         terms and their splits for this monomial. Scattered once per element.
- `entries_by_deg`     — for each degree present, a typed vector of FEMGlobalEntry{DEG}.
                         Stored as a Tuple (or NamedTuple) so that the inner loop can be
                         dispatched type-stably over each degree group.
                         Concretely: `(entries2::Vector{FEMGlobalEntry{2}},
                                       entries3::Vector{FEMGlobalEntry{3}})` for SVK.
                         For a general model, use a Dict{Int,Vector{Any}} at build time
                         and materialise as a Tuple at finalisation.
- `driver_term_idx`    — index of the FEM term that drives the element iterator.
                         All FEM terms must share the same mesh; this is validated at
                         cache-build time.
"""
struct FEMGlobalSplit{ENTRIES_TUPLE}
    global_unique_cols::Vector{Tuple{Int, Int}}
    entries_by_deg::ENTRIES_TUPLE   # e.g. Tuple{Vector{FEMGlobalEntry{2}}, Vector{FEMGlobalEntry{3}}}
    driver_term_idx::Int
end
```

`ENTRIES_TUPLE` is inferred by the constructor and frozen as a type parameter so that
`_replay_all_fem_splits!` can iterate `entries_by_deg` with a generated (unrolled) loop.

### Extended `MultilinearTermsCache`

Add one field:

```julia
struct MultilinearTermsCache{T}
    splits::Vector{Vector{Vector{CachedSplit}}}
    fem_splits::Vector{Vector{Vector{Any}}}         # existing — kept for fallback / me>0 terms
    global_fem_splits::Vector{FEMGlobalSplit}       # NEW — one per monomial; empty if no FEM terms
    result_buffer::Vector{T}
    scratch_buffer::Vector{T}
    temp_buffer::Vector{T}
    unit_vectors::Vector
    fem_Fe::Vector{T}
    global_∇W_qp::Matrix                           # NEW — (max_global_unique, max_n_qp); type from driver term
    global_Fe_buffers::Vector{Vector{T}}            # NEW — one Vector per FEM term, sized to ndofs_per_cell
end
```

`global_∇W_qp` replaces the per-term `∇W_qp` owned by each `FerriteGeometricNonlinearity`
for the combined path. Its element type must match what `scatter_qp!` writes, i.e.
`Tensor{2,3,ComplexF64}` for the Ferrite SVK backend. Since element type varies per backend,
it is typed as `Matrix` (abstractly) at the cache level and sized from the driver term's
`fem_qp_buffer` element type:

```julia
qp_elem_type = eltype(fem_qp_buffer(driver_term))   # e.g. Tensor{2,3,ComplexF64}
global_∇W_qp = Matrix{qp_elem_type}(undef, max_global_unique, max_n_qp)
```

`global_Fe_buffers[t_idx]` is a `Vector{T}` of length `fem_ndofs_per_cell(t)` for each
FEM term `t`. Elements of different terms are assembled separately because `assemble_element!`
maps element-local DOF indices to global free-DOF positions in a term-specific way.

---

## Cache Build: `_build_global_fem_split`

Called once per monomial `l` during `build_multilinear_terms_cache`, after the per-term
`FEMCachedSplit` objects have been built.

```julia
function _build_global_fem_split(model, fem_term_splits_l, fem_term_indices)
    # fem_term_indices: indices into model.nonlinear_terms that are FEMMultilinearMap
    # fem_term_splits_l[i]: Vector{FEMCachedSplit} for fem_term_indices[i]

    # --- Step 1: Build global unique-column table across all terms and splits ---
    global_unique_cols = Tuple{Int, Int}[]
    col_to_global      = Dict{Tuple{Int, Int}, Int}()

    for (i, t_idx) in enumerate(fem_term_indices)
        t = model.nonlinear_terms[t_idx]
        t.multiplicity_external == 0 || continue   # skip me>0 terms (handled separately)
        for fem_split in fem_term_splits_l[i]
            for oc in fem_split.unique_cols
                if !haskey(col_to_global, oc)
                    push!(global_unique_cols, oc)
                    col_to_global[oc] = length(global_unique_cols)
                end
            end
        end
    end

    # --- Step 2: Build FEMGlobalEntry lists, grouped by DEG ---
    entries_dict = Dict{Int, Vector{Any}}()   # DEG => Vector{FEMGlobalEntry{DEG}}

    for (i, t_idx) in enumerate(fem_term_indices)
        t = model.nonlinear_terms[t_idx]
        t.multiplicity_external == 0 || continue
        DEG = t.deg   # me=0 so deg_internal = t.deg

        for fem_split in fem_term_splits_l[i]
            for fem_entry in fem_split.fem_entries
                # Remap: local index k in fem_split.unique_cols → global index
                global_inds = ntuple(Val(DEG)) do k
                    local_oc = fem_split.unique_cols[fem_entry.local_factor_indices[k]]
                    col_to_global[local_oc]
                end
                gentry = FEMGlobalEntry{DEG}(t_idx, fem_entry.multiplier, global_inds)
                if !haskey(entries_dict, DEG)
                    entries_dict[DEG] = Any[]
                end
                push!(entries_dict[DEG], gentry)
            end
        end
    end

    # --- Step 3: Convert entries_dict to typed vectors ---
    # Build a Tuple of typed vectors sorted by DEG for type-stable dispatch.
    degs = sort(collect(keys(entries_dict)))
    typed_vecs = Tuple(
        [FEMGlobalEntry{d}(e.term_idx, e.multiplier, e.local_factor_indices)
         for e in entries_dict[d]]
        for d in degs
    )
    entries_by_deg = Tuple(typed_vecs)

    driver_term_idx = fem_term_indices[1]   # all share the same mesh; first is the driver

    return FEMGlobalSplit(global_unique_cols, entries_by_deg, driver_term_idx)
end
```

**Note on `entries_by_deg` materialisation.** The `Tuple(...)` construction at step 3 uses
runtime values (`degs`) so the resulting tuple type is inferred dynamically. To achieve a
fully concrete `ENTRIES_TUPLE` type parameter on `FEMGlobalSplit`, the build function must
be specialised on the set of degrees, or the tuple must be constructed via a helper that
dispatches on a `Val{(DEG1, DEG2, ...)}` known at model-construction time. For SVK (degrees
2 and 3 only), a fixed `Tuple{Vector{FEMGlobalEntry{2}}, Vector{FEMGlobalEntry{3}}}` can be
hardcoded. For the general case, use a generated function or accept the dynamic type and
annotate `entries_by_deg::Tuple` to avoid instability propagating further.

**Validation at build time:**
```julia
# Verify all FEM terms share the same mesh (same element count and type).
if length(fem_term_indices) > 1
    n_el = sum(1 for _ in fem_elements(model.nonlinear_terms[fem_term_indices[1]]))
    for i in 2:length(fem_term_indices)
        t = model.nonlinear_terms[fem_term_indices[i]]
        @assert sum(1 for _ in fem_elements(t)) == n_el "FEM terms must share the same mesh"
    end
end
```

---

## Combined Element Loop: `_replay_all_fem_splits!`

```julia
"""
    _replay_all_fem_splits!(result, model, W, global_split, global_∇W_qp, global_Fe_buffers)

Execute a single element loop accumulating contributions from ALL FEM terms for the given
monomial. `reinit!` and `scatter_qp!` are called at most once per unique (element, W-column)
pair; `accumulate_qp!` dispatches per entry.
"""
function _replay_all_fem_splits!(
        result, model, W,
        global_split::FEMGlobalSplit,
        global_∇W_qp,
        global_Fe_buffers)

    driver = model.nonlinear_terms[global_split.driver_term_idx]
    n_qp   = fem_n_qp(driver)
    n_uniq = length(global_split.global_unique_cols)

    for element in fem_elements(driver)

        # 1. reinit! once for all terms (they share cv).
        fem_reinit!(element, driver)

        # 2. Scatter each globally-unique W-column to qp-level field quantities.
        for i in 1:n_uniq
            (order, col) = global_split.global_unique_cols[i]
            scatter_qp!(
                @view(global_∇W_qp[i, 1:n_qp]),
                @view(W[:, order, col]),
                element, driver)
        end

        # 3. Zero all element residual buffers.
        for Fe in global_Fe_buffers
            fill!(Fe, zero(eltype(Fe)))
        end

        # 4. Accumulate at each quadrature point — one dispatch per degree group.
        for q in 1:n_qp
            dΩ = fem_getdetJdV(element, q, driver)
            _accumulate_global_entries!(global_Fe_buffers, global_∇W_qp, global_split.entries_by_deg,
                                        model, element, q, dΩ)
        end

        # 5. Assemble each term's element residual into the global accumulator.
        for t_idx in _fem_term_indices(global_split)
            t      = model.nonlinear_terms[t_idx]
            n_dofs = fem_ndofs_per_cell(t)
            assemble_element!(result, @view(global_Fe_buffers[t_idx][1:n_dofs]), element, t)
        end
    end
end
```

### Type-Stable Inner Dispatch: `_accumulate_global_entries!`

The inner dispatch over degree groups is the critical type-stability point. The signature:

```julia
# Generated/unrolled over the Tuple of entry groups.
@inline function _accumulate_global_entries!(Fe_buffers, ∇W_qp, entries_by_deg::Tuple,
                                              model, element, q, dΩ)
    _accumulate_global_entries!(Fe_buffers, ∇W_qp, entries_by_deg, model, element, q, dΩ)
end

# Base case (empty tuple).
@inline _accumulate_global_entries!(_, _, ::Tuple{}, _, _, _, _) = nothing

# Recursive case: process the head group, then the tail.
@inline function _accumulate_global_entries!(Fe_buffers, ∇W_qp,
        entries_by_deg::Tuple{Vector{FEMGlobalEntry{DEG}}, Vararg},
        model, element, q, dΩ) where {DEG}
    head = first(entries_by_deg)
    for gentry in head
        t      = model.nonlinear_terms[gentry.term_idx]
        Fe     = Fe_buffers[gentry.term_idx]
        n_dofs = fem_ndofs_per_cell(t)
        ∇W_args = ntuple(k -> ∇W_qp[gentry.local_factor_indices[k], q], Val(DEG))
        accumulate_qp!(@view(Fe[1:n_dofs]), ∇W_args, gentry.multiplier, element, q, dΩ, t)
    end
    _accumulate_global_entries!(Fe_buffers, ∇W_qp, Base.tail(entries_by_deg),
                                 model, element, q, dΩ)
end
```

The tuple-peeling pattern makes `Val(DEG)` concrete in each recursive call so that Julia can
specialise `ntuple(..., Val(DEG))` and `accumulate_qp!` on the exact degree. This is
equivalent to `@generated` but uses standard recursive dispatch on the tuple type.

For SVK where `entries_by_deg::Tuple{Vector{FEMGlobalEntry{2}}, Vector{FEMGlobalEntry{3}}}`,
Julia will specialise two methods: one for `DEG=2` and one for `DEG=3`. The recursion depth
is 2, with no dynamic dispatch in either body.

### Helper: `_fem_term_indices`

```julia
# Collect the set of distinct term_idx values referenced in global_split.
# Called only during assemble (step 5 above), not in the qp loop.
function _fem_term_indices(global_split::FEMGlobalSplit)
    seen = Set{Int}()
    for entries in global_split.entries_by_deg
        for e in entries
            push!(seen, e.term_idx)
        end
    end
    return sort(collect(seen))
end
```

Alternatively, store `driver_term_idx` and a `participating_term_indices::Vector{Int}` field
directly in `FEMGlobalSplit` to avoid recomputation.

---

## Integration into `compute_multilinear_terms!`

Replace the existing FEM branch:

```julia
# BEFORE (O1–O3 state):
if t isa FEMMultilinearMap
    for fem_split in cache.fem_splits[exp_index][t_idx]
        _replay_fem_split!(result, t, W, fem_split, cache.fem_Fe)
    end
end

# AFTER (O4):
# The combined loop is called once per monomial, outside the per-term loop.
```

The updated `compute_multilinear_terms!`:

```julia
function compute_multilinear_terms!(
        result::AbstractVector,
        model::NDOrderModel{ORD}, exp_index::Int,
        parametrisation::Parametrisation{ORD, NVAR},
        cache::MultilinearTermsCache) where {ORD, NVAR}
    fill!(result, zero(eltype(result)))
    W = parametrisation.poly.coefficients
    deg_max = sum(parametrisation.poly.multiindex_set.exponents[exp_index])
    scratch = cache.scratch_buffer
    temp    = cache.temp_buffer
    unit_vectors = cache.unit_vectors

    # --- O4: Combined FEM loop (me=0 terms only) ---
    global_split = cache.global_fem_splits[exp_index]
    if !isempty(global_split.global_unique_cols)
        _replay_all_fem_splits!(result, model, W, global_split,
                                cache.global_∇W_qp, cache.global_Fe_buffers)
    end

    # --- Closure (non-FEM) terms and me>0 FEM terms ---
    for (t_idx, t) in enumerate(model.nonlinear_terms)
        t.deg > deg_max && continue
        deg = t.deg - t.multiplicity_external
        if t isa FEMMultilinearMap
            # Only process me>0 splits here (me=0 handled by combined loop above).
            for fem_split in cache.fem_splits[exp_index][t_idx]
                isempty(fem_split.args_ext_indices) && continue
                _replay_fem_split!(result, t, W, fem_split, cache.fem_Fe)
            end
        else
            for split in cache.splits[exp_index][t_idx]
                _replay_split!(result, scratch, temp, t, W, split, deg, unit_vectors)
            end
        end
    end
    return nothing
end
```

---

## Multiple CellValues Groups (generalisation)

If a model contains FEM terms with different `cv` objects (e.g. different quadrature rules),
sharing a single `fem_reinit!` call is unsafe. The generalisation:

1. At cache-build time, partition `fem_term_indices` into groups where all terms within a
   group share the same `cv` (compared by `===`).
2. Build one `FEMGlobalSplit` per group.
3. Execute one element loop per group.

For SVK all terms share one `cv`, so there is exactly one group and the current design covers
this without modification.

---

## External Splits (`multiplicity_external > 0`)

Terms with `me > 0` require per-split external-variable scaling. In the current per-term
path, `_replay_fem_split!` handles this by routing accumulation into a temporary buffer and
then `axpy!`-ing with `ext_count`. Including such terms in the combined loop complicates the
global accumulation significantly (each split has a different scaling factor and different
W-column subsets).

**Recommended approach:** Keep `me > 0` FEM terms on the existing `_replay_fem_split!` path
(as shown in the integration code above). Only `me = 0` terms participate in the combined
loop. For SVK this covers 100% of the FEM terms.

---

## Sizing of `global_∇W_qp` and `global_Fe_buffers`

In `build_multilinear_terms_cache`:

```julia
# Max globally-unique columns across all monomials and all FEM terms.
max_global_unique = maximum(
    length(cache.global_fem_splits[l].global_unique_cols)
    for l in 1:L if !isempty(cache.global_fem_splits[l].global_unique_cols);
    init = 0)

# Max quadrature points across all FEM terms.
max_n_qp = maximum(fem_n_qp(t) for t in model.nonlinear_terms if t isa FEMMultilinearMap;
                   init = 0)

# qp value element type from the driver term.
driver_term = first(t for t in model.nonlinear_terms if t isa FEMMultilinearMap)
qp_elem_type = eltype(fem_qp_buffer(driver_term))

global_∇W_qp = Matrix{qp_elem_type}(undef, max_global_unique, max_n_qp)

# One residual buffer per FEM term.
global_Fe_buffers = [
    (t isa FEMMultilinearMap
     ? zeros(T, fem_ndofs_per_cell(t))
     : T[])
    for t in model.nonlinear_terms
]
```

---

## Files to Modify

### `src/ParametrisationMethod/RightHandSide/MultilinearTerms.jl`

1. **New structs** (after `FEMCachedSplit`):
   - `FEMGlobalEntry{DEG}`
   - `FEMGlobalSplit{ENTRIES_TUPLE}`

2. **Extend `MultilinearTermsCache`**: add `global_fem_splits`, `global_∇W_qp`,
   `global_Fe_buffers` fields.

3. **New build helper**: `_build_global_fem_split(model, fem_term_splits_l, fem_term_indices)`.

4. **Extend `build_multilinear_terms_cache`**: call `_build_global_fem_split` per monomial
   after the existing per-term split construction. Size and allocate `global_∇W_qp` and
   `global_Fe_buffers` after the per-monomial loop.

5. **New replay function**: `_replay_all_fem_splits!` + `_accumulate_global_entries!`
   (recursive tuple dispatch).

6. **Update `compute_multilinear_terms!`**: call `_replay_all_fem_splits!` for the combined
   path; keep `_replay_fem_split!` for `me > 0` splits and as a fallback.

### No changes required in:
- `src/FullOrderModel/MultilinearMaps.jl`
- `src/FullOrderModel/FullOrderModel.jl`
- `src/MORFE.jl`
- `demo/Ferrite/ferrite_assembly.jl`

---

## Validation Strategy

### Step 1 — Unit test against O1–O3 output

For a small beam (e.g. 4×2×2 elements, degree 3, ROM=2), compute:

```julia
# Reference: per-term path (disable O4 by using `_replay_fem_split!` directly)
ref = zeros(ComplexF64, n_free)
for (t_idx, t) in enumerate(model.nonlinear_terms)
    t isa MORFE.FEMMultilinearMap || continue
    for fem_split in cache.fem_splits[exp_index][t_idx]
        MORFE.MultilinearTerms._replay_fem_split!(ref, t, W, fem_split, cache.fem_Fe)
    end
end

# O4 output
out = zeros(ComplexF64, n_free)
MORFE.MultilinearTerms._replay_all_fem_splits!(out, model, W, cache.global_fem_splits[exp_index],
                                               cache.global_∇W_qp, cache.global_Fe_buffers)

@test norm(out - ref) / norm(ref) < 1e-12
```

Run for every monomial index `l = 1:L`.

### Step 2 — Full pipeline comparison

Run `solve_cohomological_problem` on the 20×3×3 beam demo with and without O4. Compare
`R.poly.coefficients` and `W.poly.coefficients` to relative tolerance `1e-10`.

### Step 3 — Allocation check

```julia
using BenchmarkTools
exp_index = findfirst(mi -> sum(mi) == 3, mset.exponents)   # pick a degree-3 monomial
allocs = @allocated compute_multilinear_terms!(cache.result_buffer, model, exp_index,
                                               parametrisation, cache)
@test allocs == 0
```

### Step 4 — Performance benchmark

```julia
@benchmark solve_cohomological_problem($model, $mset, $master_eigenvalues,
    $master_modes, $left_eigenmodes, $resonance_set; master_modes_derivatives=$master_modes_derivatives)
```

Compare median time before/after O4 on the 20×3×3 beam. For larger meshes (50×5×5 or
100×5×5) the saving should be proportionally larger because `reinit!` cost is mesh-independent
per element but total element count grows.

---

## Implementation Order

1. Add `FEMGlobalEntry{DEG}` and `FEMGlobalSplit{ENTRIES_TUPLE}` structs.
2. Add `global_fem_splits`, `global_∇W_qp`, `global_Fe_buffers` to `MultilinearTermsCache`
   (update constructor call sites in `build_multilinear_terms_cache` and tests).
3. Implement `_build_global_fem_split`; call it in `build_multilinear_terms_cache`.
4. Implement `_accumulate_global_entries!` (recursive tuple dispatch).
5. Implement `_replay_all_fem_splits!`.
6. Update `compute_multilinear_terms!` to gate on O4 path.
7. Run unit tests (Step 1 and 2 above).
8. Run allocation check (Step 3).
9. Run benchmark (Step 4) and record speedup.
