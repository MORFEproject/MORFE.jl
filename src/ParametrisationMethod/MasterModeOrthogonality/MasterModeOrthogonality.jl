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

Non-resonant master modes have trivial (zero) reduced dynamics and are excluded
from the orthogonality equations.

---

# Per-multiindex orthogonality equation

For each multi-index **γ** and superharmonic `s_k = Σᵢ γᵢ λᵢ`, the orthogonality
condition with respect to master mode `r` has the block structure

```
[ L_r(s)  C_r(s) ] * [ W; f_res ] = RHS_r
```

where:
- `L_r(s)` (`1 × FOM`) acts on the parametrisation `W`,
- `C_r(s)` (`1 × |R|`) acts on the unknown reduced dynamics of the resonant modes
  `f_res`,
- `RHS_r` is a scalar containing all known contributions (lower-order terms and
  external forcing).

The external forcing modes are **not** unknowns; their contributions are
incorporated into `RHS_r` via the operator `E_r(s)`.

---

# Joint operator `D_r` and its blocks `C_r` and `E_r`

Define the joint operator acting on the full reduced state as

```
D_r(s) = [ C_r(s)  E_r(s) ],      size: 1 × NVAR
```

where `C_r` acts on the master modes and `E_r` acts on the external forcing.
`D_r` is built via Horner's method:

```
D_r(s) = Σ_{j=1}^{ORD-1} Q_r[j] · s^{j-1}
```

with `Q_r[j]` (`1 × NVAR`) pre-computed by a downward recurrence.

Because the external forcing is known, the term `E_r(s) · (external dynamics)`
is moved to the right-hand side. Consequently, the linear system for the
unknowns `[W; f_res]` involves only `L_r` and `C_r`.

---

# Construction of `L_r` and `Q_r` via Horner

```
L_r(s) = Σ_{j=1}^{ORD} J_r[j] · s^{j-1}
```

with `J_r[j]` (`1 × FOM`) computed by the downward recurrence

```
J_r[ORD]   = Xℓ_r · B[ORD+1]
J_r[j]     = λ_r · J_r[j+1] + Xℓ_r · B[j+1],   j = ORD-1, …, 1
```

where `Xℓ_r` is the left eigenmode of master mode `r`, `λ_r` its eigenvalue, and
`B[k]` the `k`-th coefficient matrix of the linear part of the full-order model.

For the joint operator:

```
Q_r[ORD-1] = J_r[ORD] · Y
Q_r[j]     = J_r[j+1] · Y + Q_r[j+1] · Λ,   j = ORD-2, …, 1
```

where:
- `Y = generalised_right_eigenmodes` (`FOM × NVAR`) collects the generalised
  eigenvectors of all master and external forcing modes,
- `Λ = reduced_dynamics_linear` (`NVAR × NVAR`) is the Jordan matrix of the
  linear part of the reduced dynamics.

The split `Q_r = [ Ĉ_r  Ê_r ]` yields `C_r(s)` from the first `ROM` entries
and `E_r(s)` from the remaining `N_EXT` entries of each `Q_r[j]`.

---

# Precomputation and assembly

`Q_r` coefficients are pre-computed for all `NVAR = ROM + N_EXT` entries. At
assembly time, only a subset is used:

- **LHS matrix** `[L_r  C_r]`: only the `|R|` columns of `C_r(s)` corresponding
  to resonant master modes are included; non-resonant columns are omitted because
  their reduced dynamics is identically zero.
- **RHS scalar**: only the `N_EXT_active` non-zero external forcing entries of
  `E_r(s)` contribute; their values are multiplied by `external_dynamics` and
  accumulated into `RHS_r`.

---

# Right-hand-side assembly

`RHS_r` is the sum of two scalar contributions:

## Lower-order RHS

During the Horner evaluation of `L_r(s)`, the intermediate row vectors

```
L_r[j](s) = Σ_{k=j+1}^{ORD} J_r[k] · s^{k-(j+1)},   j = 1, …, ORD-1
```

are naturally available. Dotting each with the pre-computed coupling vector
`ξ[j]` gives a scalar contribution:

```
RHS_lower_r = -Σ_{j=1}^{ORD-1} L_r[j](s) · ξ[j]
```

This accumulation is performed **in the same Horner loop** that computes `L_r(s)`,
avoiding recomputation of the `L_r[j]` intermediates.

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

Stacking the per-mode conditions for all `r ∈ R` yields the global linear system

```
[ L   C ] · [ W; f_res ] = RHS_R
```

where:
- `L` is `|R| × FOM` (rows `L_r`),
- `C` is `|R| × |R|` (resonant columns of `C_r` from the joint operator),
- `f_res` is the vector of unknown reduced-dynamics coefficients for the resonant
  modes,
- `RHS_R` is the assembled `|R|`-vector of scalar right-hand sides.

---

# Module contents

| Function | Description |
|:---------|:------------|
| [`precompute_orthogonality_operator_coefficients`](@ref)   | Pre-compute `J_r` coefficient arrays for the orthogonality row operators `L_r(s)` |
| [`precompute_orthogonality_column_polynomials`](@ref)      | Pre-compute `Q_r` coefficient arrays split into `C_coeffs` and `E_coeffs` |
| [`evaluate_orthogonality_row_and_lower_order_rhs!`](@ref)  | Fused Horner pass for `L_r(s)` (row) + scalar lower-order RHS |
| [`evaluate_orthogonality_column_row!`](@ref)               | Evaluate the resonant block of `C_r(s)` into one row of the `C` block |
| [`evaluate_orthogonality_external_rhs`](@ref)              | Compute the scalar external-forcing RHS for mode `r` |
| [`assemble_orthogonality_matrix_and_rhs!`](@ref)           | Full block-matrix and RHS assembly for all resonant modes (in-place) |
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
caller-supplied `M` (`nR × n_sys`) and `rhs` (length `nR`) buffers.  Returns
immediately when `nR == 0` (non-resonant monomial).  No heap allocation occurs.
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
	external_dynamics::AbstractVector{T},
) where {T, ROM}
	isempty(rhs) && return nothing
	FOM = size(J_coeffs[1], 2)
	nR = count(resonance)
	row = 1
	for r in eachindex(resonance)
		if resonance[r]
			rhs[row] = evaluate_orthogonality_row_and_lower_order_rhs!(
				view(M, row, 1:FOM), s, lower_order_couplings, J_coeffs[r],
			)
			evaluate_orthogonality_column_row!(
				view(M, row, (FOM+1):(FOM+nR)), s, r, C_coeffs, resonance,
			)
			rhs[row] += evaluate_orthogonality_external_rhs(s, r, external_dynamics, E_coeffs)
			row += 1
		end
	end
	return nothing
end

end # module MasterModeOrthogonality
