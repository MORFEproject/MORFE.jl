"""
	MasterModeOrthogonality

Assemble the orthogonality conditions that arise in the parametrisation method,
a reduced-order modelling technique for high-dimensional dynamical systems.

---

# Nomenclature

| Symbol  | Meaning |
|:--------|:--------|
| FOM     | Full-order model dimension |
| ROM     | Number of master modes (reduced coordinates) |
| N_EXT   | Number of external forcing modes |
| NVAR    | `ROM + N_EXT` (total reduced variables) |
| R       | Set of resonant master modes (`|R| ≤ ROM`) |

Non-resonant master modes have trivial (zero) reduced dynamics.  They are *not*
dropped from the block: each contributes the trivial row `f_r = 0`, so the block
keeps a constant `ROM × (FOM+ROM)` shape whatever the resonance pattern.

---

# Mathematical origin: sesquilinear B-orthogonality

For each multi-index **γ** with superharmonic `s = Σᵢ γᵢ λᵢ`, the condition for
master mode `r` is the sesquilinear B-orthogonality of the companion left
eigenvector `φ_r = [φ_{r,1}; …; φ_{r,ORD}]` (solving `φ_rᴴ (λ_r B − A) = 0`)
against the first-order state `𝒲` of the monomial:

```
φ_rᴴ B 𝒲 = 0,     𝒲 = [W_1; …; W_ORD],   W_1 = W,   W_{j+1} = s W_j + Y_j f + ξ_j
```

Here `Y_j` are the right eigenmode order-blocks, `f` the reduced-dynamics
coefficient at **γ** (master part unknown, external part known forcing) and
`ξ_j` the known lower-order couplings. Solving the recurrence and collecting
terms reduces the condition to a single row equation

```
Ĵ_r(s) W + C_r(s) f_res = − E_r(s) f_ext − Σ_{k=1}^{ORD-1} G_{r,k}(s) ξ_k
```

with

```
Ĵ_r(s)   = Σ_{j=1}^{ORD} J_r[j] · s^{j-1}          (1 × FOM row on W)
G_{r,k}(s) = Σ_{j=k+1}^{ORD} J_r[j] · s^{j-1-k}     (Horner tails of Ĵ_r)
C_r(s)   = Σ_{k=1}^{ORD-1} G_{r,k}(s) · Y_k^m       (coupling to unknown f_res)
E_r(s)   = Σ_{k=1}^{ORD-1} G_{r,k}(s) · Y_k^e       (known forcing → RHS)
```

Only the resonant master columns of `C_r` carry a value; the rest are written as
zeros, since the matching trivial rows pin those reduced-dynamics coefficients to
zero anyway.

---

# Coefficients from eigenvector order-blocks (no eigenvalue folding)

The row coefficients are read directly off the (conjugated) left eigenvector
order-blocks — no eigenvalue appears:

```
J_r[j]   = conj(φ_{r,j})              j = 1, …, ORD-1
J_r[ORD] = conj(B_ORDᴴ φ_{r,ORD})
```

The conjugation is the sesquilinear `ᴴ` of the condition; it is baked into the
stored coefficients so that every assembled contraction (`row · W`, `row · ξ`,
`row · Y`) is **bilinear**.

`C_r`/`E_r` contract the same tails `G_{r,k}` against the right eigenmode
order-blocks: master blocks come from the eigensolver, external blocks from the
recurrence `Y_{k+1}^e = Y_k^e Λ_e + Y_k^m Λ_me` (the master↔external coupling of
the reduced linear dynamics — the one confined place an eigenvalue-derived
matrix remains).

---

# Precomputation and assembly

`J_r`, `C_r` and `E_r` coefficients are pre-computed once per order. At
assembly time, only a subset is used:

- **LHS matrix** `[Ĵ_r  Ĉ_r]`: the columns of `C_r(s)` for non-resonant master
  modes are zeroed rather than omitted, which keeps the block shape independent of
  the resonance pattern.  Nothing is lost — those coefficients are pinned to zero by
  their own trivial rows.
- **RHS scalar**: only the `N_EXT_active` non-zero external forcing entries of
  `E_r(s)` contribute; their values are multiplied by `external_dynamics` and
  accumulated into `RHS_r`.

---

# Right-hand-side assembly

`RHS_r` is the sum of two scalar contributions:

## Lower-order RHS

During the Horner evaluation of `Ĵ_r(s)`, the intermediate row vectors

```
Ĵ_r[j](s) = Σ_{k=j+1}^{ORD} J_r[k] · s^{k-(j+1)},   j = 1, …, ORD-1
```

are naturally available. Dotting each with the pre-computed coupling vector
`ξ[j]` gives a scalar contribution:

```
RHS_lower_r = -Σ_{j=1}^{ORD-1} Ĵ_r[j](s) · ξ[j]
```

This accumulation is performed **in the same Horner loop** that computes `Ĵ_r(s)`,
avoiding recomputation of the `Ĵ_r[j]` intermediates.

## External forcing RHS

For external forcing modes `e = 1, …, N_EXT`, the scalar-valued polynomial

```
E_r_e(s) = Σ_{j=1}^{ORD-1} E_coeffs[r][j, e] · s^{j-1}
```

gives the contribution of mode `e` to `RHS_r` when multiplied by
`external_dynamics[e]`. The total external contribution is

```
RHS_ext_r = -Σ_{e active} external_dynamics[e] · E_r_e(s)
```

Only active (non-zero) external modes are processed. Their contributions are
combined into a single scalar Horner pass, avoiding per-mode evaluations.

The complete right-hand side is therefore

```
RHS_r = RHS_lower_r + RHS_ext_r
```

---

# Full system assembly

Stacking one row per master mode — resonant or not — yields the global linear system

```
[ P Ĵ   P Ĉ P + τ Q ] · [ W; f ] = P · RHS_R,      P = diag(resonance), Q = I − P
```

where:
- `Ĵ` is `ROM × FOM` (rows `Ĵ_r`), kept only on resonant rows,
- `Ĉ` is `ROM × ROM` (columns of `C_r` from the joint operator), masked on both axes,
- `f` is the **full** `ROM`-vector of reduced-dynamics coefficients,
- `RHS_R` is the assembled `ROM`-vector of scalar right-hand sides,
- `τ = 1` on the non-resonant diagonal, turning row `r` into `f_r = 0`.

The block size is therefore independent of how many modes are resonant, which is
what allows the sparse cohomological solver to hold one sparsity pattern — and one
cached symbolic factorisation — for the whole solve.  Masking loses nothing: the
dropped `C` entries multiply coefficients that the trivial rows pin to zero.

---

# Module contents

| Function | Description |
|:---------|:------------|
| [`precompute_orthogonality_operator_coefficients`](@ref)   | Pre-compute `J_r` coefficient arrays for the orthogonality row operators `Ĵ_r(s)` |
| [`precompute_orthogonality_column_polynomials`](@ref)      | Pre-compute `Q_r` coefficient arrays split into `C_coeffs` and `E_coeffs` |
| [`evaluate_orthogonality_row_and_lower_order_rhs!`](@ref)  | Fused Horner pass for `Ĵ_r(s)` (row) + scalar lower-order RHS |
| [`evaluate_orthogonality_column_row!`](@ref)               | Evaluate `C_r(s)` into one row of the `C` block, masked by the resonance vector |
| [`evaluate_orthogonality_external_rhs`](@ref)              | Compute the scalar external-forcing RHS for mode `r` |
| [`assemble_orthogonality_matrix_and_rhs!`](@ref)           | Constant-size `ROM × (FOM+ROM)` block and RHS assembly (in-place) |
"""
module MasterModeOrthogonality

using LinearAlgebra
using StaticArrays

include("HornerEvaluator.jl")
include("ColumnPolynomials.jl")

export precompute_orthogonality_operator_coefficients,
       precompute_orthogonality_column_polynomials,
       evaluate_orthogonality_row_and_lower_order_rhs!,
       evaluate_orthogonality_column_row!,
       evaluate_orthogonality_external_rhs,
       assemble_orthogonality_matrix_and_rhs!

# =============================================================================
# Full orthogonality matrix and RHS assembly (in-place only)
# =============================================================================

"""
	assemble_orthogonality_matrix_and_rhs!(M, rhs, s, J_coeffs, C_coeffs, E_coeffs,
											resonance, lower_order_couplings,
											external_dynamics) → nothing

In-place variant: writes the orthogonality block and its RHS directly into the
caller-supplied `M` (`ROM × (FOM+ROM)`) and `rhs` (length `ROM`) buffers.  No heap
allocation occurs.

The block has **constant size** — one row per master mode, resonant or not — and the
resonance vector selects each row's *content* rather than the block's dimensions:

- resonant `r`: row `r` is the orthogonality condition
  `Ĵ_r(s) W_α + Σ_m Ĉ_{rm}(s) R_{m,α} = g_{r,α}`, with the corner entries masked to
  the resonant modes by [`evaluate_orthogonality_column_row!`](@ref);
- non-resonant `r`: row `r` becomes the trivial equation `τ R_{r,α} = 0` — everything
  zeroed except `M[r, FOM+r] = τ = 1` — which encodes the style choice that
  non-resonant reduced-dynamics coefficients vanish.

Keeping the width constant is what lets the sparse path reuse one symbolic
factorisation across every monomial.  Note that `τ` is structural only: the solver
never reads the trivial rows back out of the solution vector, it writes the hard
zeros directly during unpacking.
"""
function assemble_orthogonality_matrix_and_rhs!(
        M::AbstractMatrix,
        rhs::AbstractVector,
        s::T,
        J_coeffs::AbstractVector{<:AbstractMatrix{T}},
        C_coeffs::Vector{<:AbstractMatrix{T}},
        E_coeffs::Vector{<:AbstractMatrix{T}},
        resonance::SVector{ROM, Bool},
        lower_order_couplings::AbstractVector{<:AbstractVector{T}},
        external_dynamics::AbstractVector{T}
) where {T, ROM}
    FOM = size(J_coeffs[1], 2)
    @assert size(M) == (ROM, FOM + ROM) "orthogonality block must be ROM×(FOM+ROM) = \
      $((ROM, FOM + ROM)), got $(size(M))"
    @assert length(rhs) == ROM "orthogonality rhs must have length ROM = $ROM, \
      got $(length(rhs))"
    for r in eachindex(resonance)
        if resonance[r]
            rhs[r] = evaluate_orthogonality_row_and_lower_order_rhs!(
                view(M, r, 1:FOM), s, lower_order_couplings, J_coeffs[r]
            )
            evaluate_orthogonality_column_row!(
                view(M, r, (FOM + 1):(FOM + ROM)), s, r, C_coeffs, resonance
            )
            rhs[r] += evaluate_orthogonality_external_rhs(s, r, external_dynamics, E_coeffs)
        else
            fill!(view(M, r, 1:(FOM + ROM)), zero(T))
            M[r, FOM + r] = one(T)   # τ: pins R[r, α] = 0
            rhs[r] = zero(T)
        end
    end
    return nothing
end

end # module MasterModeOrthogonality
