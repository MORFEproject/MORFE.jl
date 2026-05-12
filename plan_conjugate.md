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

### 2b. `monomial_map::Vector{Int}` (length L) — permutation on monomials

| Value of `monomial_map[i]` | Meaning |
|---|---|
| `i` | Self-conjugate monomial: P·γ = γ; solve normally (real arithmetic if applicable) |
| `j > i` | Monomial i is **primary**; its conjugate is at index j — solve i, then fill j |
| `j < i` | Monomial i is **secondary**; it will be filled when primary j is processed — skip |
| `0` | No conjugate available: involves an unpaired mode or conjugate outside mset — solve normally |

---

## 3. Precomputing `monomial_map`

This precomputation is encapsulated inside `_build_conjugate_symmetry` (Section 4d).
Shown here for reference:

```julia
function _build_monomial_map(mset::MultiindexSet{NVAR},
                             perm::SVector{NVAR, Int},
                             mdict::Dict{SVector{NVAR, Int}, Int}) where {NVAR}
    L = length(mset)
    monomial_map = Vector{Int}(undef, L)
    is_self_conjugate = BitVector(undef, L)

    for i in 1:L
        γ = mset[i]

        # Any active component on an unpaired mode → no conjugate symmetry here
        has_unpaired = any(k -> γ[k] > 0 && perm[k] == 0, 1:NVAR)
        if has_unpaired
            monomial_map[i] = 0
            is_self_conjugate[i] = false
            continue
        end

        Pγ = SVector{NVAR, Int}(ntuple(k -> γ[perm[k]], Val(NVAR)))

        if Pγ == γ
            monomial_map[i] = i
            is_self_conjugate[i] = true
        else
            j = get(mdict, Pγ, 0)
            monomial_map[i] = j          # 0 if P·γ ∉ mset; j>i or j<i otherwise
            is_self_conjugate[i] = false
        end
    end
    return monomial_map, is_self_conjugate
end
```

---

## 4. `ConjugateSymmetryData` — The Optimization Layer

All conjugate-symmetry state is encapsulated in a single struct.
`CohomologicalContext` is **not modified in any way**.

### 4a. Type sentinels

```julia
# Sentinel: conjugate symmetry inactive (no permutation stored)
struct NoConjugatePermutation end
```

`SVector{NVAR, Int}` is used directly for active permutations — no wrapper needed.

### 4b. `RealArithmeticBuffers` — pre-allocated Float64 scratch

```julia
"""
    RealArithmeticBuffers

Pre-allocated Float64 buffers for the real-arithmetic dense solve path activated
on self-conjugate monomials when the FOM matrices are real-valued.
"""
struct RealArithmeticBuffers
    system::Matrix{Float64}   # (FOM+ROM)×(FOM+ROM)
    rhs   ::Vector{Float64}   # length FOM+ROM
end

RealArithmeticBuffers(FOM::Int, ROM::Int) =
    RealArithmeticBuffers(Matrix{Float64}(undef, FOM + ROM, FOM + ROM),
                          Vector{Float64}(undef, FOM + ROM))
```

### 4c. `ConjugateSymmetryData{CP, RB}` — self-contained struct

```julia
"""
    ConjugateSymmetryData{CP, RB}

Self-contained optimization layer for exploiting complex-conjugate symmetry in
the cohomological solve.

# Type parameters

| Parameter | Meaning |
|:----------|:--------|
| `CP` | `NoConjugatePermutation` (inactive) or `SVector{NVAR, Int}` (active) |
| `RB` | `Nothing` (complex FOM or inactive) or `RealArithmeticBuffers` (real FOM + active) |

Compile-time dispatch on `CP` eliminates secondary-monomial bookkeeping when
inactive.  Compile-time dispatch on `RB` eliminates the real-arithmetic branch
when not applicable.  The only remaining runtime check is the `BitVector` index
`is_self_conjugate[idx]` (< 1 ns), active only when `RB = RealArithmeticBuffers`.
"""
struct ConjugateSymmetryData{CP, RB}
    permutation      ::CP              # NoConjugatePermutation  or  SVector{NVAR,Int}
    monomial_map     ::Vector{Int}     # length L; empty when CP = NoConjugatePermutation
    is_self_conjugate::BitVector       # length L; empty when CP = NoConjugatePermutation
    skip_set         ::Set{Int}        # linear_skip_set ∪ secondary indices
    real_buffers     ::RB              # Nothing  or  RealArithmeticBuffers
end
```

### 4d. Factory function `_build_conjugate_symmetry`

Located in `ConjugateSymmetry.jl`.

```julia
# Dispatch helper: allocate real buffers only for real FOM + active conjugate symmetry.
_make_real_buffers(::Type{LT}, FOM::Int, ROM::Int) where {LT}          = nothing
_make_real_buffers(::Type{LT}, FOM::Int, ROM::Int) where {LT <: Real}  =
    RealArithmeticBuffers(FOM, ROM)

# Inactive path: wrap the existing linear skip set, allocate nothing.
function _build_conjugate_symmetry(
        ::NoConjugatePermutation,
        linear_skip_set::Set{Int}, _, _, _, _, _
)
    return ConjugateSymmetryData{NoConjugatePermutation, Nothing}(
        NoConjugatePermutation(), Int[], BitVector(), copy(linear_skip_set), nothing
    )
end

# Active path: build monomial map, augment skip set, conditionally allocate real buffers.
function _build_conjugate_symmetry(
        perm::SVector{NVAR, Int},
        linear_skip_set::Set{Int},
        mset::MultiindexSet{NVAR},
        mdict::Dict{SVector{NVAR, Int}, Int},
        FOM::Int, ROM::Int, ::Type{LT}
) where {NVAR, LT}
    monomial_map, is_self_conjugate = _build_monomial_map(mset, perm, mdict)

    skip_set = copy(linear_skip_set)
    for i in eachindex(monomial_map)
        j = monomial_map[i]
        j > i && push!(skip_set, j)   # j is secondary → skip it in the outer loop
    end

    # Consistency assertion
    for i in eachindex(monomial_map)
        j = monomial_map[i]
        if j ∉ (0, i)
            @assert monomial_map[j] == i   "conjugate map must be symmetric at i=$i"
            @assert !(j in linear_skip_set) "conjugate of a non-linear must not be linear"
        end
    end

    real_buffers = _make_real_buffers(LT, FOM, ROM)
    RB = typeof(real_buffers)
    return ConjugateSymmetryData{SVector{NVAR, Int}, RB}(
        perm, monomial_map, is_self_conjugate, skip_set, real_buffers
    )
end
```

---

## 5. `fill_conjugate_monomial!`

Located in `ConjugateSymmetry.jl`.  Takes `sym` (not `ctx`) — only the permutation
is needed; `CohomologicalContext` is not involved.

```julia
function fill_conjugate_monomial!(
    W::Parametrisation{ORD, NVAR, T},
    R::ReducedDynamics{ROM, NVAR, T},
    conj_idx::Int,
    source_idx::Int,
    sym::ConjugateSymmetryData{SVector{NVAR, Int}, RB},
) where {ORD, NVAR, T, ROM, RB}
    Wc = W.poly.coefficients   # FOM × ORD × L
    Rc = R.poly.coefficients   # NVAR × L

    # W_{P·γ} = conj(W_γ)  for all ORD time-derivative orders
    for j in 1:ORD
        @views Wc[:, j, conj_idx] .= conj.(Wc[:, j, source_idx])
    end

    # f_{P·β}[r] = conj(f_β[perm[r]])  for master-mode rows only (1:ROM).
    # External rows (ROM+1:NVAR) need no explicit fill — see precondition note below.
    perm = sym.permutation
    for r in 1:ROM
        pr = perm[r]
        # pr = r   → real mode: R[r, conj] = conj(R[r, src])
        # pr = j≠r → paired:   R[r, conj] = conj(R[j, src])
        # pr = 0   → unpaired: cannot arise (excluded by _build_monomial_map)
        Rc[r, conj_idx] = conj(Rc[pr, source_idx])
    end
    return nothing
end
```

**Preconditions**: all ORD orders of W and master-mode rows (1:ROM) of R at
`source_idx` have been finalised by `solve_single_monomial!` before this is
called.

External rows (ROM+1:NVAR) of R at `conj_idx` are left untouched and are
already correct for two distinct reasons:

- **Mixed monomials** (source has at least one non-zero internal component):
  `_embed_external_dynamics!` only writes pure-external monomials
  `(0,...,0,α_ext)`, so external rows at any mixed monomial are zero from
  initialisation; `conj(0) = 0` is trivially satisfied.
- **Pure-external monomials** `(0,...,0,α_ext)` (all internal components zero):
  `_embed_external_dynamics!` has set both the primary's and the secondary's
  external rows from the external dynamics polynomial; because that polynomial
  satisfies conjugate symmetry by construction, the secondary's values are
  already `conj` of the primary's.

---

## 6. `detect_conjugate_permutation`

Located in `ConjugateSymmetry.jl`.

```julia
"""
    detect_conjugate_permutation(lambda; atol=1e-8) -> SVector{NVAR, Int}

Build the conjugate permutation from eigenvalues `lambda` (length NVAR):

- `perm[i] = j` if `lambda[j] ≈ conj(lambda[i])` and `j ≠ i`
- `perm[i] = i` if `lambda[i]` is numerically real (`|imag| ≤ atol`)
- `perm[i] = 0` if `lambda[i]` is complex but no conjugate partner is found
"""
function detect_conjugate_permutation(lambda::AbstractVector{<:Number}; atol=1e-8)
    NVAR = length(lambda)
    perm = zeros(Int, NVAR)
    assigned = falses(NVAR)

    for i in 1:NVAR
        assigned[i] && continue
        if abs(imag(lambda[i])) ≤ atol
            perm[i] = i
            assigned[i] = true
            continue
        end
        found = false
        for j in (i+1):NVAR
            assigned[j] && continue
            abs(imag(lambda[j])) ≤ atol && continue
            if abs(lambda[j] - conj(lambda[i])) ≤ atol
                perm[i] = j;  perm[j] = i
                assigned[i] = assigned[j] = true
                found = true
                break
            end
        end
        found || (perm[i] = 0; assigned[i] = true)
    end
    return SVector{NVAR, Int}(perm)
end
```

---

## 7. Real-Arithmetic Solve for Self-Conjugate Monomials

### Mathematical basis

For self-conjugate γ (P·γ = γ):

1. **s_γ ∈ ℝ** — because s_{P·γ} = conj(s_γ) and P·γ = γ force s_γ = conj(s_γ).
2. **W_γ ∈ ℝ^FOM** — by W-symmetry.
3. **A(s_γ) ∈ ℝ^{FOM×FOM}** when `LT <: Real` and s_γ ∈ ℝ.
4. **RHS ∈ ℝ^FOM** — by induction on degree.

Therefore, for real FOM the entire bordered system is real-valued.  Solving in
Float64 instead of ComplexF64 reduces the LU cost by ≈4×.

The buffers for this path live in `RealArithmeticBuffers` inside
`ConjugateSymmetryData`.  `CohomologicalBuffers` and `CohomologicalContext` are
**not modified**.

### Compile-time dispatch on `RB`

`solve_single_monomial!` with a `sym` argument dispatches at compile time on `RB`:

```julia
# RB = Nothing: real arithmetic never available → always complex solve
@inline function _sym_solve_monomial!(
        ctx, sym::ConjugateSymmetryData{CP, Nothing}, idx,
        s, nR, resonance, lower_order_couplings, external_dynamics
) where {CP}
    _solve_monomial!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)
end

# RB = RealArithmeticBuffers: one runtime check per monomial
@inline function _sym_solve_monomial!(
        ctx, sym::ConjugateSymmetryData{CP, RealArithmeticBuffers}, idx,
        s, nR, resonance, lower_order_couplings, external_dynamics
) where {CP}
    if sym.is_self_conjugate[idx]
        _solve_monomial_real!(ctx, sym.real_buffers, s, nR, resonance,
                              lower_order_couplings, external_dynamics)
    else
        _solve_monomial!(ctx, s, nR, resonance, lower_order_couplings, external_dynamics)
    end
end
```

`_sym_solve_monomial!` is a private helper in `CohomologicalEquations.jl`.
The only runtime check is the single `BitVector` index (< 1 ns).

### `_solve_monomial_real!` (dense path, in `CohomologicalSolver.jl`)

Takes `ctx` (for the shared `buffers.system_matrix`/`rhs` and assembly functions)
and `rb::RealArithmeticBuffers` (for the Float64 scratch space):

```julia
function _solve_monomial_real!(
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        rb::RealArithmeticBuffers,
        s, nR, resonance, lower_order_couplings, external_dynamics
) where {T, ORD, ORDP1, NVAR, FOM, LT, MT}
    n_sys = FOM + nR

    # Assemble into the existing complex buffers (same call as dense complex path)
    assemble_cohomological_matrix_and_rhs!(
        view(ctx.buffers.system_matrix, 1:FOM, 1:n_sys),
        view(ctx.buffers.rhs, 1:FOM),
        s, ctx.linear_terms,
        ctx.invariance.C_coeffs, ctx.invariance.E_coeffs,
        resonance, lower_order_couplings, external_dynamics,
        ctx.buffers.external_rhs,
    )
    view(ctx.buffers.rhs, 1:FOM) .+= ctx.buffers.ml_result
    assemble_orthogonality_matrix_and_rhs!(
        view(ctx.buffers.system_matrix, (FOM+1):n_sys, 1:n_sys),
        view(ctx.buffers.rhs, (FOM+1):n_sys),
        s, ctx.orthogonality.J_coeffs,
        ctx.orthogonality.C_coeffs, ctx.orthogonality.E_coeffs,
        resonance, lower_order_couplings, external_dynamics,
    )

    # Cast to Float64 (imaginary parts are zero by construction)
    M_r   = view(rb.system, 1:n_sys, 1:n_sys)
    rhs_r = view(rb.rhs,    1:n_sys)
    M_r   .= real.(view(ctx.buffers.system_matrix, 1:n_sys, 1:n_sys))
    rhs_r .= real.(view(ctx.buffers.rhs,           1:n_sys))

    F = lu!(M_r, check = false)   # Float64 LU — ~4× cheaper than ComplexF64
    ldiv!(F, rhs_r)

    view(ctx.buffers.rhs, 1:n_sys) .= rhs_r   # write back as real-valued ComplexF64
    return
end
```

O(FOM²) cast overhead is negligible vs O(FOM³) LU saving.

### Self-conjugate monomials and resonance

For purely oscillatory systems (λ_r = iω_r), s_γ ∈ ℝ and resonance requires
ℝ = iℝ → impossible → nR = 0 at every self-conjugate monomial.  The bordered
system collapses to FOM×FOM when the real path fires on such systems.

---

## 8. New `solve_cohomological_equations!` Overloads

Located in `CohomologicalEquations.jl`.  The **existing** overload (no `sym`
argument) is completely unchanged.  Two new overloads are added, dispatching on
`CP`:

**Without active symmetry** (`CP = NoConjugatePermutation`):

```julia
function solve_cohomological_equations!(
        W, R,
        ctx::CohomologicalContext,
        sym::ConjugateSymmetryData{NoConjugatePermutation, RB},
        model, ml_cache,
) where {RB}
    nterms = length(multiindex_set(W))
    for idx in 1:nterms
        idx in sym.skip_set && continue   # sym.skip_set == linear_skip_set here
        solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
    end
    return nothing
end
```

**With active symmetry** (`CP = SVector{NVAR, Int}`):

```julia
function solve_cohomological_equations!(
        W, R,
        ctx::CohomologicalContext,
        sym::ConjugateSymmetryData{<:SVector, RB},
        model, ml_cache,
) where {RB}
    nterms = length(multiindex_set(W))
    cmap   = sym.monomial_map

    for idx in 1:nterms
        idx in sym.skip_set && continue   # skips linears AND secondary monomials

        solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)

        j = cmap[idx]
        j != idx && fill_conjugate_monomial!(W, R, j, idx, sym)
        # j == idx → self-conjugate: nothing to fill
        # j == 0   → no conjugate:   nothing to fill
        # j >  idx → primary: fill secondary at j (j is in skip_set)
    end
    return nothing
end
```

The `conjugate_skip_set` is precomputed to contain all secondaries, so `j < idx`
never occurs in the loop body.

---

## 9. New `solve_single_monomial!` Overload

Located in `CohomologicalEquations.jl`.  The **existing** overload is completely
unchanged.

```julia
function solve_single_monomial!(
        W::Parametrisation{ORD, NVAR, T},
        R::ReducedDynamics{ROM, NVAR, T},
        idx::Int,
        ctx::CohomologicalContext{T, ORD, ORDP1, NVAR, FOM, LT, MT},
        sym::ConjugateSymmetryData,
        model::NDOrderModel,
        ml_cache::MultilinearTermsCache
) where {ORD, NVAR, T, ROM, FOM, ORDP1, LT, MT}
    multi = multiindex_set(W)[idx]

    s         = sum(multi[i] * ctx.lambda_diag[i] for i in 1:NVAR)
    resonance = _resonance_vector(ctx.resonance_set, idx, Val(ROM))

    for v in ctx.lower_order.buffer;  fill!(v, zero(T));  end
    lower_order_couplings = compute_lower_order_couplings(
        multi, W, R,
        ctx.lower_order.multiindex_dict, ctx.lower_order.buffer,
        ctx.lower_order.candidate_indices[idx], ctx.lower_order.unit_vectors
    )

    compute_multilinear_terms!(ctx.buffers.ml_result, model, idx, W, ml_cache)

    external_dynamics = view(R.poly.coefficients, (ROM + 1):NVAR, idx)
    nR    = count(resonance)
    n_sys = FOM + nR

    # Compile-time dispatch on RB; one runtime BitVector check when RB ≠ Nothing
    _sym_solve_monomial!(ctx, sym, idx, s, nR, resonance,
                         lower_order_couplings, external_dynamics)

    sol = view(ctx.buffers.rhs, 1:n_sys)
    W.poly.coefficients[:, 1, idx] .= view(sol, 1:FOM)

    rr = 1
    for r in 1:ROM
        if resonance[r]
            R.poly.coefficients[r, idx] = sol[FOM + rr];  rr += 1
        else
            R.poly.coefficients[r, idx] = zero(T)
        end
    end

    compute_higher_derivative_coefficients!(
        W.poly.coefficients,
        view(R.poly.coefficients, 1:ROM, :),
        external_dynamics, s, idx,
        ctx.generalised_eigenmodes, lower_order_couplings
    )
    return nothing
end
```

---

## 10. API Change to `solve_cohomological_problem`

Located in `CohomologicalDriver.jl`.  One keyword argument is added; `_build_context`
and the context building logic are **not modified**.

```julia
function solve_cohomological_problem(
        model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT, LT, MT},
        mset, master_eigenvalues, master_modes, left_eigenmodes, resonance_set;
        initial_W = nothing,
        initial_R = nothing,
        master_modes_derivatives = nothing,
        conjugate_permutation::Union{Nothing, AbstractVector{Int}} = nothing,  # NEW
) where {ORD, ORDP1, N_NL, N_EXT, LT, MT, NVAR, ROM}
```

After the existing `lambda_diag` computation, resolve to a concrete permutation type:

```julia
_conj_perm = if conjugate_permutation !== nothing
    SVector{NVAR, Int}(conjugate_permutation)
else
    auto = detect_conjugate_permutation(lambda_diag)
    all(auto[i] == i || auto[i] == 0 for i in 1:NVAR) ?
        NoConjugatePermutation() :   # all modes real or unpaired → no symmetry to exploit
        auto
end
```

After the existing `lower_order` and `buffers` allocation (which are unchanged),
build the symmetry object and call the new overload:

```julia
sym = _build_conjugate_symmetry(
    _conj_perm,
    linear_skip_set,                          # already computed
    mset,
    lower_order.multiindex_dict,              # already computed inside lower_order
    FOM, ROM, LT
)

# ... existing operator precomputation and partial context solve (unchanged) ...

solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache)   # NEW overload
```

The partial-context solve for external directions uses `_solve_external_directions!`
which calls the existing (sym-less) `solve_single_monomial!` — no change needed.

---

## 11. `CohomologicalContext` Changes

**None.**

`CohomologicalContext`, `CohomologicalBuffers`, `SolverResources.jl`,
`OperatorData.jl`, and `_build_context` are all completely unchanged.

---

## 12. File-by-File Change Summary

| File | Change |
| :--- | :----- |
| New: `CohomologicalEquations/ConjugateSymmetry.jl` | `NoConjugatePermutation`, `RealArithmeticBuffers`, `ConjugateSymmetryData{CP,RB}`, `detect_conjugate_permutation`, `_build_conjugate_symmetry`, `_build_monomial_map`, `_make_real_buffers`, `fill_conjugate_monomial!` |
| `CohomologicalEquations/CohomologicalSolver.jl` | Add `_solve_monomial_real!(ctx, rb, ...)` |
| `CohomologicalEquations/CohomologicalEquations.jl` | `include("ConjugateSymmetry.jl")`; add `_sym_solve_monomial!` (2 dispatch variants); add `solve_single_monomial!` overload (with `sym`); add `solve_cohomological_equations!` overloads (2 dispatch variants on `CP`); export new public symbols |
| `CohomologicalEquations/CohomologicalDriver.jl` | Add `conjugate_permutation` kwarg; resolve `_conj_perm`; build `sym` via `_build_conjugate_symmetry`; call new `solve_cohomological_equations!` overload |
| `src/MORFE.jl` | Add `NoConjugatePermutation`, `ConjugateSymmetryData`, `detect_conjugate_permutation` to `export` block |

**Completely unchanged:**

| File | Reason |
| :--- | :----- |
| `CohomologicalEquations/CohomologicalContext.jl` | Conjugate state lives in `ConjugateSymmetryData` |
| `CohomologicalEquations/SolverResources.jl` | Real buffers live in `RealArithmeticBuffers` |
| `CohomologicalEquations/OperatorData.jl` | No conjugate-related operators |
| `InvarianceEquation/`, `MasterModeOrthogonality/` | No changes |
| `LowerOrderCouplings.jl`, `MultilinearTerms.jl` | No changes |
| `PropagateEigenmodes.jl` | No changes |

---

## 13. Dispatch Chain Summary

```text
solve_cohomological_problem(... ; conjugate_permutation=...)
  └─ _build_conjugate_symmetry(_conj_perm, ...) → ConjugateSymmetryData{CP, RB}
  └─ solve_cohomological_equations!(W, R, ctx, sym, model, ml_cache)
       ↓ compile-time dispatch on CP
       ├─ CP = NoConjugatePermutation : straight loop over sym.skip_set
       └─ CP = SVector{NVAR,Int}      : loop + fill_conjugate_monomial! for primaries
            └─ solve_single_monomial!(W, R, idx, ctx, sym, model, ml_cache)
                 └─ _sym_solve_monomial!(ctx, sym, idx, ...)
                      ↓ compile-time dispatch on RB
                      ├─ RB = Nothing                : _solve_monomial!(ctx, ...)
                      └─ RB = RealArithmeticBuffers  : runtime check is_self_conjugate[idx]
                           ├─ true  → _solve_monomial_real!(ctx, sym.real_buffers, ...)
                           └─ false → _solve_monomial!(ctx, ...)
```

Three compile-time dispatch steps; one runtime check (single `BitVector` index, < 1 ns).

---

## 14. Edge Cases

### Partially-paired spectra
Monomials with any active `perm[k] = 0` component get `monomial_map[i] = 0` and
are solved normally.  No special-casing needed.

### `P·γ` outside the multiindex set
`get(mdict, Pγ, 0)` returns 0 → `monomial_map[i] = 0` → solved normally.
Conservative but correct.

### External modes
External-mode components of a multi-index are **part of the conjugacy check**.
`_build_monomial_map` applies the full NVAR permutation — including external indices
— to determine whether two monomials are conjugate partners:

- `(1,2,3,4)` is conjugate to `(2,1,4,3)` under `perm = (2,1,4,3)` because
  `P·(1,2,3,4) = (2,1,4,3)`.
- `(1,1,2,3)` is **not** conjugate to `(1,1,2,4)` because
  `P·(1,1,2,3) = (1,1,3,2) ≠ (1,1,2,4)`.

For the **fill step**, external rows of R are not touched — they are already
correct before `fill_conjugate_monomial!` is called:

- `_embed_external_dynamics!` constructs `α_full` by zeroing all internal
  components: it exclusively sets monomials of the form `(0,...,0, α_ext)`.
  Mixed monomials with any non-zero internal component are never written, so their
  external rows remain zero; `conj(0) = 0` trivially satisfies the symmetry rule.
- For pure-external secondary monomials `(0,...,0,α_ext)`, both the primary and
  secondary have their external rows set by `_embed_external_dynamics!` from the
  external dynamics polynomial, which is conjugate-symmetric by construction.

---

## 15. Expected Performance Gain

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

---

## 16. Verification

1. Run demo with default (`conjugate_permutation = nothing`, auto-detect enabled)
   and confirm output matches the identity-perm run to machine precision.
2. Run with `conjugate_permutation = collect(1:NVAR)` (explicit identity — no
   symmetry) and verify identical output to step 1.
3. Provide a known conjugate pair (e.g., NVAR=2, perm=(2,1)) and verify that
   `W.poly.coefficients[:, :, conj_idx] ≈ conj.(W.poly.coefficients[:, :, source_idx])`.
4. Profile: secondary monomials must trigger zero allocations in the solve path
   (`--track-allocation=user`).
5. Timing: for ROM=4 at degree 7, wall time of the solve loop should drop by ≈40–50%.
