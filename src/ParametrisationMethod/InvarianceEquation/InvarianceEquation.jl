"""
	InvarianceEquation

Assemble the part of the cohomological equations that corresponds to the [`FullOrderDynamics`](@ref),
thereby imposing invariance of the manifold.
[`MasterModeOrthogonality`](@ref) handles the orthogonality conditions induced by the parametrisation styles.

---

# Nomenclature

| Symbol  | Meaning |
|:--------|:--------|
| FOM     | Full-order model dimension |
| ROM     | Number of master modes (reduced coordinates) |
| N_EXT   | Number of external forcing modes |
| NVAR    | `ROM + N_EXT` (total reduced variables) |
| R       | Set of resonant master modes |

Non-resonant master modes have trivial (zero) reduced dynamics and are excluded
from the cohomological equations; their columns in `C(s)` are omitted.

---

# Per-multiindex cohomological equation

For each multi-index **k** with superharmonic `s = Σᵢ kᵢ λᵢ` (λᵢ eigenvalues of
the master modes), the cohomological equation has the block structure

```
[ L(s)  C(s) ] * [ W_k; f_res ] = RHS_k
```

where:
- `L(s)` (`FOM × FOM`) is the parametrisation operator (characteristic-matrix
  polynomial of the full-order model),
- `C(s)` (`FOM × |R|`) acts on the unknown reduced-dynamics coefficients `f_res`
  of the resonant master modes,
- `RHS_k` contains all known lower-order contributions and external forcing.

External forcing modes are **not** unknowns; their contributions are handled
separately and appear on the right-hand side.

---

# Construction of `L(s)` and `C(s)` via Horner

The operator `L(s)` is defined as

```
L(s) = Σ_{k=1}^{ORD+1} B[k] s^{k-1}
```

where `B[k]` are the coefficient matrices of the *linear* part of the full-order
model (size `FOM × FOM`). `L(s)` is evaluated efficiently using Horner's method.

The operator acting on the reduced dynamics is

```
C(s) = Σ_{j=1}^{ORD} D[j] s^{j-1}
```

with pre-computed coefficient matrices `D[j]` (size `FOM × NVAR`) given by

```
D[j] = Σ_{k=j+1}^{ORD+1} B[k] · generalised_right_eigenmodes · reduced_dynamics_linear^{k-(j+1)}
```

Here:
- `generalised_right_eigenmodes`: `NVAR × FOM` matrix collecting the generalised
  eigenvectors of the master modes and the external forcing modes.
- `reduced_dynamics_linear`: Jordan matrix of the linear part of the reduced dynamics.

The `D[j]` matrices are pre-computed once per order using a downward recurrence
(similar to the Horner scheme in `MasterModeOrthogonality`).

---

# Precomputation and assembly

The coefficients `D[j]` are pre-computed for **all** `NVAR = ROM + N_EXT` variables.
However, when assembling the linear system for a given multi-index (with
superharmonic `s`), only a **subset of the columns** of `C(s)` is used:

- For the left-hand-side matrix `[L(s)  C(s)]`, only the columns corresponding
  to the **resonant** master modes (size `|R|`) are extracted from `C(s)`.
  Non-resonant master modes are omitted because their reduced dynamics is
  identically zero.
- The external forcing modes (`N_EXT` columns) are handled separately and do not
  appear as unknowns; their contributions are moved to the right-hand side via
  the operator `-E(s)` (see below).

---

# Right-hand-side assembly

`RHS_k` is the sum of two independent contributions: lower-order terms from the
cohomological equation, and external forcing terms. Both are evaluated using
**fused Horner passes** that reuse intermediate matrices to minimise
computational cost.

## Lower-order RHS (cohomological coupling)

During the Horner evaluation of `L(s)`, the intermediate matrices

```
L[j](s) = Σ_{k=j+1}^{ORD+1} B[k] · s^{k-(j+1)},   j = 1,…,ORD
```

are naturally available. Multiplying each `L[j](s)` by a pre-computed coupling
vector `ξ[j]` (obtained from lower-order solution coefficients) gives the
contribution of lower-order terms to the RHS:

```
RHS_lower = -Σ_{j=1}^{ORD} L[j](s) · ξ[j]
```

The negative sign arises because these terms originate from the left-hand side
of the cohomological equation and are moved to the right-hand side. This
accumulation is performed **in the same Horner loop** that computes `L(s)`,
avoiding recomputation of the `L[j](s)` intermediates.

## External forcing RHS

For external forcing modes `e = 1,…,N_EXT`, the polynomial coefficients
`E_e[L]` (`FOM × 1` column vectors) are pre-computed such that

```
E_e(s) = Σ_{L=1}^{ORD} E_e[L] · s^{L-1}
```

is the contribution of forcing mode `e` to the cohomological equation when
multiplied by its known amplitude `external_dynamics[e]`. The total external
contribution is

```
RHS_ext = Σ_{e=1}^{N_EXT} E_e(s) · external_dynamics[e]
```

To evaluate this efficiently, the coefficients of all active (non-zero) external
modes are first combined into a single vector polynomial:

```
g_L = Σ_{e active} external_dynamics[e] · E_e[L],   L = 1,…,ORD
```

Then `g(s) = Σ_{L=1}^{ORD} g_L · s^{L-1}` is evaluated in a single Horner
pass. The result is **added** to the RHS accumulator. This fused approach
avoids evaluating each `E_e(s)` independently and scales only with the number
of active external modes.

The complete right-hand side is therefore

```
RHS_k = RHS_lower + RHS_ext
```

where both parts are computed using dedicated fused Horner passes that share the
polynomial evaluation structure of the main operator `L(s)`.

---

# Module contents

| Function | Description |
|:---------|:------------|
| [`precompute_column_polynomials`](@ref)                       | Pre-compute `D_{L,j}` coefficient arrays for both the system-matrix columns and the external-forcing RHS |
| [`evaluate_system_matrix_and_lower_order_rhs!`](@ref)         | Fused Horner pass for `L(s)` + lower-order RHS |
| [`evaluate_column!`](@ref)                                    | Evaluate one `C_r(s)` column |
| [`evaluate_external_rhs!`](@ref)                              | Accumulate external-forcing RHS |
| [`assemble_cohomological_matrix_and_rhs!`](@ref)              | Full block-matrix and RHS assembly (in-place) |
"""
module InvarianceEquation

using LinearAlgebra
using SparseArrays
using StaticArrays

include("HornerEvaluator.jl")
include("ColumnPolynomials.jl")

export precompute_column_polynomials,
       precompute_master_column_polynomials,
       precompute_external_column_polynomials,
       evaluate_system_matrix_and_lower_order_rhs!,
       evaluate_column!,
       evaluate_external_rhs!,
       assemble_cohomological_matrix_and_rhs!,
       build_sparse_L_and_rhs!,
       precompute_sparse_L_template,
       precompute_sparse_bordered_template,
       scatter_L_into_bordered!

# =============================================================================
# Full cohomological-matrix and RHS assembly (in-place only)
# =============================================================================

"""
	assemble_cohomological_matrix_and_rhs!(M, rhs, s, linear_terms, C_coeffs, E_coeffs,
											resonance, lower_order_couplings,
											external_dynamics, g_buffer) → nothing

In-place variant: writes the invariance-equation system matrix and RHS directly into
the caller-supplied `M` and `rhs` buffers.  No heap allocation occurs.

`M`   must have size `FOM × (FOM + ROM)` — the border is **constant width** and the
resonance mask selects its content rather than its size: column `FOM + r` holds
`C_r(s)` when mode `r` is resonant and is zeroed otherwise.  The zeroed columns are
harmless because the matching orthogonality row pins `R_{r,α} = 0`
(see [`assemble_orthogonality_matrix_and_rhs!`](@ref)).

`rhs` must have length `FOM`.  Both are overwritten on entry.
`g_buffer` is the pre-allocated `FOM`-length scratch buffer for the external RHS.
"""
function assemble_cohomological_matrix_and_rhs!(
        M::AbstractMatrix,
        rhs::AbstractVector,
        s::Number,
        linear_terms::NTuple{ORDP1, <:AbstractMatrix},
        C_coeffs::Vector{<:AbstractMatrix},
        E_coeffs::Vector{<:AbstractMatrix},
        resonance::SVector{ROM, Bool},
        lower_order_couplings::AbstractVector{<:AbstractVector},
        external_dynamics::AbstractVector,
        g_buffer::AbstractVector
) where {ROM, ORDP1}
    FOM = size(linear_terms[1], 1)
    fill!(rhs, zero(eltype(rhs)))
    evaluate_system_matrix_and_lower_order_rhs!(
        view(M, :, 1:FOM), rhs, s, lower_order_couplings, linear_terms
    )
    for j in eachindex(resonance)
        column = view(M, :, FOM + j)
        if resonance[j]
            evaluate_column!(column, s, j, C_coeffs)
        else
            fill!(column, zero(eltype(M)))
        end
    end
    evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs, g_buffer)
    return nothing
end

end # module InvarianceEquation
