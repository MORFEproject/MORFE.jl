# Conjugate Symmetry Exploitation — Implementation Plan

## 1. Mathematical Summary

For a real-valued FOM, the parametrisation and reduced dynamics satisfy:

```
W_{P·γ}  = conj(W_γ)           [W-symmetry]
f_{P·β}  = P_ROM · conj(f_β)   [f-symmetry]
```

where **P** is an involutory permutation on mode indices that swaps each conjugate
eigenvalue pair.  The permutation acts on multi-indices by permuting exponents:

```
(P · γ)[k] = γ[P(k)]
```

**Example:** `conjugate_permutation = SVector(2,1,3)` means P swaps modes 1↔2 and
fixes mode 3.  Monomial (a,b,c) maps to (b,a,c).

**Consequence:** for every non-self-conjugate pair (γ, P·γ), only one cohomological
equation needs to be solved; the other is filled by conjugation.  For systems with
fully-paired master modes this halves the number of `solve_single_monomial!` calls.

---

## 2. Encoding Conventions

### 2a. `conjugate_permutation::SVector{NVAR, Int}` — permutation on modes

| Value of `conjugate_permutation[i]` | Meaning |
|---|---|
| `j ≠ i, j ≠ 0` | Mode i is complex-conjugate to mode j (and j maps back to i) |
| `i` | Mode i has a real eigenvalue — self-conjugate, no partner needed |
| `0` | Mode i is an unpaired complex mode (no conjugate partner in the set) |

The permutation is involutory on non-zero entries: `perm[perm[i]] == i` whenever
`perm[i] ≠ 0`.

### 2b. `conjugate_monomial_map::Vector{Int}` (length L) — permutation on monomials

| Value of `conjugate_monomial_map[i]` | Meaning |
|---|---|
| `i` | Self-conjugate monomial: P·γ = γ; solve normally (real arithmetic if applicable) |
| `j > i` | Monomial i is **primary**; its conjugate is at index j — solve i, then fill j |
| `j < i` | Monomial i is **secondary**; it will be filled when primary j is processed — skip |
| `0` | No conjugate available: involves an unpaired mode or conjugate outside mset — solve normally |

---

## 3. Precomputing `conjugate_monomial_map`

```julia
conjugate_monomial_map = Vector{Int}(undef, L)
for i in 1:L
    γ = mset[i]

    # Check if any active component sits on an unpaired complex mode
    has_unpaired = any(j -> γ[j] > 0 && conjugate_permutation[j] == 0, 1:NVAR)
    if has_unpaired
        conjugate_monomial_map[i] = 0
        continue
    end

    # Compute P·γ (safe: no zero-index components are active)
    Pγ = SVector{NVAR, Int}(ntuple(k -> γ[conjugate_permutation[k]], Val(NVAR)))

    if Pγ == γ
        conjugate_monomial_map[i] = i          # self-conjugate
    else
        j = get(multiindex_dict, Pγ, 0)
        conjugate_monomial_map[i] = j          # j=0 if P·γ not in mset; j>i or j<i otherwise
    end
end
```

**Also compute** `is_self_conjugate::BitVector` in the same pass:

```julia
is_self_conjugate[i] = (conjugate_monomial_map[i] == i)
```

This is used for the real-arithmetic solve path (Section 7).

---

## 4. Multiple Dispatch Architecture

The `conjugate_permutation` is **optional**.  When absent, no symmetry is exploited
and the existing code path runs with zero overhead.  This is achieved by
parametrising `CohomologicalContext` on a permutation type, then dispatching
`solve_cohomological_equations!` on that type parameter.

### 4a. Permutation type hierarchy

```julia
# Sentinel: no conjugate symmetry
struct NoConjugatePermutation end

# Active permutation (NVAR known at compile time)
# Alias: SVector{NVAR, Int} is used directly — no wrapper needed
```

`CohomologicalContext` gains a type parameter `CP` for the permutation:

```julia
struct CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT, CP}
    ...
    conjugate_permutation::CP       # NoConjugatePermutation  OR  SVector{NVAR, Int}
    conjugate_monomial_map::Vector{Int}   # length L; empty when CP = NoConjugatePermutation
    is_self_conjugate::BitVector          # length L; empty when CP = NoConjugatePermutation
    conjugate_skip_set::Set{Int}          # linear_monomial_skip_set ∪ secondary indices
    ...
end
```

When `CP = NoConjugatePermutation`:
- `conjugate_monomial_map` and `is_self_conjugate` are empty (length-0) vectors
- `conjugate_skip_set = linear_monomial_skip_set` (identical to the existing skip set)
- All conjugate-related fields cost one allocation each (empty vectors — negligible)

### 4b. `solve_cohomological_equations!` — two methods, no runtime conditionals

**Without symmetry** (`CP = NoConjugatePermutation`) — identical to current code:

```julia
function solve_cohomological_equations!(
    W, R,
    ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT, NoConjugatePermutation},
    model, ml_cache,
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT}
    nterms = length(multiindex_set(W))
    for idx in 1:nterms
        idx in ctx.conjugate_skip_set && continue   # same as linear_monomial_skip_set
        solve_single_monomial!(W, R, idx, ctx, model, ml_cache)
    end
    return nothing
end
```

**With symmetry** (`CP = SVector{NVAR, Int}`):

Secondary monomials are pre-inserted into `conjugate_skip_set`, so the loop body
needs only one extra integer lookup and one comparison to distinguish primary from
self-conjugate:

```julia
function solve_cohomological_equations!(
    W, R,
    ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT, SVector{NVAR, Int}},
    model, ml_cache,
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT}
    nterms = length(multiindex_set(W))
    cmap = ctx.conjugate_monomial_map

    for idx in 1:nterms
        idx in ctx.conjugate_skip_set && continue   # skips linears AND secondaries

        solve_single_monomial!(W, R, idx, ctx, model, ml_cache)

        conj_idx = cmap[idx]
        conj_idx != idx && fill_conjugate_monomial!(W, R, conj_idx, idx, ctx)
        # conj_idx == idx  →  self-conjugate: nothing to fill
        # conj_idx == 0    →  no conjugate: nothing to fill
        # conj_idx >  idx  →  primary: fill secondary at conj_idx (secondaries are in skip set)
    end
    return nothing
end
```

The `conjugate_skip_set` is precomputed to contain all secondaries, so `conj_idx <
idx` never occurs in the loop body.

### 4c. Building `conjugate_skip_set`

```julia
conjugate_skip_set = copy(linear_monomial_skip_set)
for i in 1:L
    j = conjugate_monomial_map[i]
    j > i && push!(conjugate_skip_set, j)   # j is secondary
end
```

---

## 5. `fill_conjugate_monomial!`

```julia
function fill_conjugate_monomial!(
    W::Parametrisation{ORD, NVAR, T},
    R::ReducedDynamics{ROM, NVAR, T},
    conj_idx::Int,
    source_idx::Int,
    ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT, SVector{NVAR, Int}},
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    Wc = W.poly.coefficients   # FOM × ORD × L
    Rc = R.poly.coefficients   # NVAR × L

    # W_{P·γ} = conj(W_γ)  for all ORD time-derivative orders
    for j in 1:ORD
        @views Wc[:, j, conj_idx] .= conj.(Wc[:, j, source_idx])
    end

    # f_{P·β}[r] = conj(f_β[conjugate_permutation[r]])  for ALL r = 1..NVAR
    # This covers both master-mode rows (1:ROM) and external-dynamics rows (ROM+1:NVAR).
    # conjugate_permutation is NVAR-dimensional and applies uniformly to all rows.
    perm = ctx.conjugate_permutation
    for r in 1:NVAR
        pr = perm[r]
        # pr = r   → real/self-conjugate mode: R[r, conj] = conj(R[r, src])
        # pr = j≠r → paired mode:              R[r, conj] = conj(R[j, src])
        # pr = 0   → unpaired complex mode: cannot arise here — monomials with active
        #            unpaired-mode components are excluded from the map at build time
        Rc[r, conj_idx] = conj(Rc[pr, source_idx])
    end
    return nothing
end
```

**Precondition for master-mode rows (1:ROM)**: all ORD orders of W and master-mode
rows of R at `source_idx` have been written by `solve_single_monomial!` before this
is called.  Guaranteed since `fill_conjugate_monomial!` is called immediately after
`solve_single_monomial!` returns.

**Precondition for external-dynamics rows (ROM+1:NVAR)**: these rows are pre-set by
`_embed_external_dynamics!` for all monomials (including both primary and secondary)
before the solve loop starts.  The `fill_conjugate_monomial!` call overwrites the
secondary's external rows with `conj(R[perm[r], source_idx])`.  For a real-valued
system the external dynamics polynomial satisfies conjugate symmetry, so this
overwrite is consistent with the pre-set values — but writing it explicitly via fill
is the authoritative path and removes any dependence on `_embed_external_dynamics!`
being conjugate-consistent.

---

## 6. `detect_conjugate_permutation`

New function in `src/ParametrisationMethod/ConjugateSymmetry.jl`:

```julia
"""
    detect_conjugate_permutation(lambda; atol=1e-8) -> SVector{NVAR, Int}

Build the conjugate permutation from eigenvalues `lambda` (length NVAR):

- `perm[i] = j` if `lambda[j] ≈ conj(lambda[i])` and `j ≠ i`
- `perm[i] = i` if `lambda[i]` is real (no conjugate partner found with same magnitude)
- `perm[i] = 0` if `lambda[i]` is complex but no conjugate partner is found in the list

Raises ArgumentError if a detected pairing is not consistent (same index claimed twice).
"""
function detect_conjugate_permutation(lambda::AbstractVector{<:Number}; atol=1e-8)
    NVAR = length(lambda)
    perm = zeros(Int, NVAR)
    assigned = falses(NVAR)

    for i in 1:NVAR
        assigned[i] && continue
        if isreal(lambda[i])           # real eigenvalue
            perm[i] = i
            assigned[i] = true
            continue
        end
        found = false
        for j in (i+1):NVAR
            assigned[j] && continue
            isreal(lambda[j]) && continue
            if abs(lambda[j] - conj(lambda[i])) ≤ atol
                perm[i] = j
                perm[j] = i
                assigned[i] = true
                assigned[j] = true
                found = true
                break
            end
        end
        if !found
            perm[i] = 0               # unpaired complex mode
            assigned[i] = true
        end
    end
    return SVector{NVAR, Int}(perm)
end
```

**`isreal` check**: use `abs(imag(lambda[i])) ≤ atol` to handle numerical noise.

---

## 7. Real-Arithmetic Solve for Self-Conjugate Monomials

### Mathematical basis

For self-conjugate γ (P·γ = γ):

1. **s_γ ∈ ℝ** — because s_{P·γ} = conj(s_γ) and P·γ = γ force s_γ = conj(s_γ).
2. **W_γ ∈ ℝ^FOM** — by W-symmetry with P·γ = γ.
3. **A(s_γ) ∈ ℝ^{FOM×FOM}** when all FOM matrices are real (`LT <: Real`) and
   s_γ ∈ ℝ.
4. **RHS is real** by induction on degree: lower-degree W satisfy conjugate
   symmetry; their linear combinations under real FOM operators are real.

Therefore, for real FOM the entire bordered system at a self-conjugate monomial is
real-valued.  Solving in Float64 instead of ComplexF64 reduces the LU cost by ≈4×.

### Additional context fields (dense path only)

```
real_system_buffer::Union{Nothing, Matrix{Float64}}  # (FOM+ROM)×(FOM+ROM); nothing for complex FOM
real_rhs_buffer   ::Union{Nothing, Vector{Float64}}  # length FOM+ROM;       nothing for complex FOM
```

Allocated only when `LT <: Real`; `nothing` otherwise (zero overhead for complex
FOM models).

### Dispatch via `is_self_conjugate` bitvector

`is_self_conjugate[idx]` is a scalar Bool lookup — unavoidably runtime, since the
set of self-conjugate monomials is data-dependent.  The per-monomial overhead is
one `BitVector` index (< 1 ns) plus one branch.

Inside `solve_single_monomial!`:

```julia
if ctx.real_system_buffer !== nothing && ctx.is_self_conjugate[idx]
    _solve_monomial_real!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)
else
    _solve_monomial!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)
end
```

(The `real_system_buffer !== nothing` check is a field-existence test resolved at
compile time when `LT` is a concrete type.)

### `_solve_monomial_real!` (dense path)

```julia
function _solve_monomial_real!(ctx, s, nR, resonance, lower_order_couplings,
                                external_dynamics)
    n_sys = FOM + nR
    # Same assembly as the complex path
    assemble_cohomological_matrix_and_rhs!(
        view(ctx.system_matrix_buffer, 1:FOM, 1:n_sys), view(ctx.rhs_buffer, 1:FOM),
        s, ctx.linear_terms, ctx.invariance_C_coeffs, ctx.invariance_E_coeffs,
        resonance, lower_order_couplings, external_dynamics, ctx.external_rhs_buffer,
    )
    view(ctx.rhs_buffer, 1:FOM) .+= ctx.ml_result_buffer
    assemble_orthogonality_matrix_and_rhs!(
        view(ctx.system_matrix_buffer, (FOM+1):n_sys, 1:n_sys),
        view(ctx.rhs_buffer, (FOM+1):n_sys),
        s, ctx.orthogonality_J_coeffs, ctx.orthogonality_C_coeffs,
        ctx.orthogonality_E_coeffs, resonance, lower_order_couplings, external_dynamics,
    )
    # Cast to Float64 (imaginary parts are zero by construction)
    M_r   = view(ctx.real_system_buffer, 1:n_sys, 1:n_sys)
    rhs_r = view(ctx.real_rhs_buffer,    1:n_sys)
    M_r   .= real.(view(ctx.system_matrix_buffer, 1:n_sys, 1:n_sys))
    rhs_r .= real.(view(ctx.rhs_buffer,           1:n_sys))

    F = lu!(M_r, check = false)   # Float64 LU — ~4× cheaper than ComplexF64
    ldiv!(F, rhs_r)

    view(ctx.rhs_buffer, 1:n_sys) .= rhs_r   # write back as real-valued complex
    return
end
```

O(FOM²) cast overhead is negligible vs. O(FOM³) LU saving.

### Self-conjugate monomials and resonance

For purely oscillatory systems (λ_r = iω_r ≠ 0), s_γ ∈ ℝ and all λ_r are purely
imaginary → resonance (s_γ = λ_r) requires ℝ = iℝ, which is impossible → nR = 0
for all self-conjugate monomials.  The bordered system is always FOM×FOM when the
real-path fires on such systems.

### External dynamics rows at self-conjugate monomials

For self-conjugate γ (P·γ = γ) with external-mode rows: since s_γ ∈ ℝ and the
external dynamics satisfy conjugate symmetry, the external dynamics view
`R[(ROM+1):NVAR, idx]` at a self-conjugate monomial is also real-valued.  It is
used (read-only) in `_solve_monomial_real!` via `external_dynamics` — no change
needed to how it is accessed.

---

## 8. API Change to `solve_cohomological_problem`

Add one keyword argument:

```julia
function solve_cohomological_problem(
    model, mset, master_eigenvalues, master_modes, left_eigenmodes, resonance_set;
    initial_W = nothing,
    initial_R = nothing,
    master_modes_derivatives = nothing,
    conjugate_permutation::Union{Nothing, AbstractVector{Int}} = nothing,  # NEW
)
```

Resolution in the function body:

```julia
# Resolve conjugate permutation → always concrete, never Union at this point
_conj_perm = if conjugate_permutation !== nothing
    SVector{NVAR, Int}(conjugate_permutation)
else
    auto = detect_conjugate_permutation(lambda_diag)
    all(auto[i] == i || auto[i] == 0 for i in 1:NVAR) ?
        NoConjugatePermutation() :    # all modes real or unpaired → no symmetry to exploit
        auto
end
```

This logic:
- Uses caller-supplied permutation if provided
- Auto-detects from `lambda_diag` if not
- Falls back to `NoConjugatePermutation()` if auto-detection finds nothing to exploit
  (all eigenvalues are real or all complex modes are unpaired)

Both partial and full `CohomologicalContext` constructor calls receive `_conj_perm`,
triggering the appropriate type-parametrised path.

---

## 9. `CohomologicalContext` Changes

### New fields (appended after `unit_vectors`):

```julia
conjugate_permutation::CP               # CP = NoConjugatePermutation  or  SVector{NVAR, Int}
conjugate_monomial_map::Vector{Int}     # length L; empty when CP = NoConjugatePermutation
is_self_conjugate::BitVector            # length L; empty when CP = NoConjugatePermutation
conjugate_skip_set::Set{Int}            # linear_monomial_skip_set ∪ secondaries

# Real-arithmetic buffers (dense path, LT <: Real only)
real_system_buffer::Union{Nothing, Matrix{Float64}}
real_rhs_buffer   ::Union{Nothing, Vector{Float64}}
```

**When `CP = NoConjugatePermutation`**: `conjugate_monomial_map`, `is_self_conjugate`
are empty vectors; `conjugate_skip_set = linear_monomial_skip_set`; real buffers
follow the `LT <: Real` rule independently.

### Type parameter change:

```julia
struct CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT <: AbstractMatrix{LT}, CP}
```

Both constructor call sites in `solve_cohomological_problem` must pass the 6 new
fields.

---

## 10. File-by-File Change Summary

| File | Change |
|---|---|
| `CohomologicalEquations.jl` | Add type param `CP`; add 6 new fields; update both constructor calls; add `fill_conjugate_monomial!`; add two dispatch methods for `solve_cohomological_equations!`; precompute `conjugate_monomial_map`, `is_self_conjugate`, `conjugate_skip_set`; add real-buffer precomputation; add `conjugate_permutation` kwarg to `solve_cohomological_problem`; dispatch `_solve_monomial_real!` in `solve_single_monomial!` |
| New: `ConjugateSymmetry.jl` | `NoConjugatePermutation`, `detect_conjugate_permutation` |
| Module root | Export `detect_conjugate_permutation`, `NoConjugatePermutation` |

No changes required to `LowerOrderCouplings.jl`, `MultilinearTerms.jl`,
`InvarianceEquation.jl`, `MasterModeOrthogonality.jl`, or `PropagateEigenmodes.jl`.

---

## 11. Edge Cases

### Partially-paired spectra
Auto-detection returns `perm[i] = i` (real) or `perm[i] = j` (paired) or `perm[i]
= 0` (unpaired) per mode.  Monomials with any active `perm[k] = 0` component get
`conjugate_monomial_map[i] = 0` and are solved normally.  No special-casing needed.

### `P·γ` not in the multiindex set
At truncated polynomial degree, some conjugates fall outside `mset`.
`get(multiindex_dict, Pγ, 0)` returns 0 → `conjugate_monomial_map[i] = 0` → solved
normally.  Conservative but correct.

### Consistency assertion (precomputation)
```julia
for i in 1:L
    j = conjugate_monomial_map[i]
    if j ∉ (0, i)
        @assert conjugate_monomial_map[j] == i  "conjugate map must be symmetric"
        @assert !(j in linear_monomial_skip_set) "conjugate of non-skip must be non-skip"
    end
end
```

### External modes
`conjugate_permutation` covers all NVAR indices including external modes (ROM+1:NVAR).
`fill_conjugate_monomial!` fills **all NVAR rows** of R by the uniform rule
`R[r, conj_idx] = conj(R[perm[r], source_idx])`, whether r is a master-mode or
external-mode index.

For the unit-vector monomials corresponding to external modes (in the skip set):
these are initialised by the partial solve and the skip set prevents the fill loop
from touching them.  Both a mode and its conjugate partner are initialised
independently by the partial solve — no conjugate fill is applied there.

For higher-degree monomials involving external-mode components: if the external
modes are conjugate-paired (`perm[ROM+e] = ROM+e'`), the secondary monomial is
added to `conjugate_skip_set` and its R rows (both master and external) are filled
via `fill_conjugate_monomial!` when the primary is processed.  The pre-set values
from `_embed_external_dynamics!` are overwritten by the fill, which is authoritative
for secondary monomials.

---

## 12. Expected Performance Gain

For a system with `k` fully-paired conjugate mode pairs (ROM = 2k), at polynomial
degree d:

| Metric | Value |
|---|---|
| Secondary monomials (skipped) | ≈ 50% of non-skip monomials |
| `solve_single_monomial!` calls | ≈ halved |
| `fill_conjugate_monomial!` cost | O(ORD × FOM + ROM) per fill — negligible |
| Additional gain from real LU | ≈ 4× speedup on self-conjugate solves (real FOM) |

For NVAR=4 (two conjugate pairs), degree 7: L ≈ 120, ~55 are secondary → ~55
solves instead of ~105 (after skipping linears).

Self-conjugate monomials at degree d, NVAR=2k:
`C(d+k-1, k-1) / C(d+2k-1, 2k-1)` — typically a small fraction at high degree.

---

## 13. Verification

1. Run demo with default (`conjugate_permutation = nothing`, auto-detect enabled)
   and confirm output matches the identity-perm run to machine precision.
2. Run with `conjugate_permutation = collect(1:NVAR)` (explicit identity — no
   symmetry) and verify identical output to step 1.
3. Provide a known conjugate pair (e.g., NVAR=2, perm=(2,1)) and verify that
   `W.poly.coefficients[:, :, conj_idx] ≈ conj.(W.poly.coefficients[:, :, source_idx])`.
4. Profile: secondary monomials must trigger zero allocations in the solve path
   (`--track-allocation=user`).
5. Timing: for ROM=4 at degree 7, wall time of the solve loop should drop by ≈40–50%.
