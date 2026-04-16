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
| [`assemble_orthogonality_matrix_and_rhs`](@ref)            | Full block-matrix and RHS assembly for all resonant modes |
"""
module MasterModeOrthogonality

using LinearAlgebra
using StaticArrays

export precompute_orthogonality_operator_coefficients,
       precompute_orthogonality_column_polynomials,
       evaluate_orthogonality_row_and_lower_order_rhs!,
       evaluate_orthogonality_column_row!,
       evaluate_orthogonality_external_rhs,
       assemble_orthogonality_matrix_and_rhs

# =============================================================================
# 1.  Pre-compute orthogonality row-operator coefficients J_r
# =============================================================================

"""
	precompute_orthogonality_operator_coefficients(fom_matrices, left_eigenmodes,
												   master_eigenvalues)
	-> Vector{Matrix{T}}

Pre-compute the polynomial coefficient arrays for the orthogonality row operators
`L_r(s)` using a downward Horner recurrence on the left eigenstructure.

## Return value

An `NTuple{ROM, Matrix{T}}` where entry `r` is an `ORD × FOM` matrix
`J_coeffs[r]`.  Row `j` of `J_coeffs[r]` stores the degree-`(j-1)` row-vector
coefficient of `L_r`:

```
L_r(s) = Σ_{j=1}^{ORD} J_coeffs[r][j, :] · s^{j-1}
```

## Arguments

- `fom_matrices       :: NTuple{ORD+1, <:AbstractMatrix{T}}` – linear matrices of
  the full-order model; `fom_matrices[k+1]` corresponds to `B_k` (0-indexed).
- `left_eigenmodes    :: AbstractMatrix{T}` – left eigenvectors of
  the master modes; `left_eigenmodes[:, r]` is the length-`FOM` vector for mode `r`.
- `master_eigenvalues :: SVector{ROM, T}` – eigenvalues `λ_r` of the master modes.

## Recurrence

For each mode `r`, a single downward pass fills `J_coeffs[r]` (`ORD × FOM`):

```
J_r[ORD, :]   ←  B[ORD+1]ᵀ · ℓ_r
J_r[j,   :]   ←  λ_r · J_r[j+1, :] + B[j+1]ᵀ · ℓ_r,   j = ORD-1, …, 1
```

Here `B[k+1]ᵀ · ℓ_r` is a `FOM`-vector equal to the row vector `ℓ_rᵀ · B[k+1]`
stored as a column; this is standard Julia `mul!(dest, B', x)`.

## Complexity

- Time:    `O(ROM · ORD · FOM²)` (one `FOM`-vector update per `(r, j)` pair)
- Storage: `O(ROM · ORD · FOM)`
"""
function precompute_orthogonality_operator_coefficients(
	fom_matrices::NTuple{ORDP1, <:AbstractMatrix{T}},
	left_eigenmodes::AbstractMatrix{T},
	master_eigenvalues::SVector{ROM, T},
) where {ORDP1, ROM, T}
	ORD = ORDP1 - 1
	FOM = size(first(fom_matrices), 1)

	@assert ORD ≥ 1 "ODE order ORD = length(fom_matrices) - 1 must be ≥ 1."
	@assert ROM ≥ 1 "ROM must be ≥ 1."
	@assert size(left_eigenmodes) == (FOM, ROM) "left_eigenmodes must be FOM × ROM ($(FOM) × $(ROM))."

	result = Vector{Matrix{T}}(undef, ROM)
	for r in 1:ROM
		ℓ = view(left_eigenmodes, :, r)    # length-FOM left eigenmode
		λ = master_eigenvalues[r]

		J_r = Matrix{T}(undef, ORD, FOM)

		# Highest degree: J_r[ORD, :] = B[ORDP1]ᵀ · ℓ  (= ℓᵀ · B[ORDP1] as a row)
		mul!(view(J_r, ORD, :), fom_matrices[ORDP1]', ℓ)

		# Downward recurrence: J_r[j, :] = λ · J_r[j+1, :] + B[j+1]ᵀ · ℓ
		for j in (ORD-1):-1:1
			view(J_r, j, :) .= λ .* view(J_r, j+1, :)            # copy and scale
			mul!(view(J_r, j, :), fom_matrices[j+1]', ℓ, one(T), one(T))  # accumulate
		end

		result[r] = J_r
	end
	return result
end

# =============================================================================
# 2.  Pre-compute joint operator coefficients Q_r → (C_coeffs, E_coeffs)
# =============================================================================

"""
	precompute_orthogonality_column_polynomials(J_coeffs,
												generalised_right_eigenmodes,
												reduced_dynamics_linear)
	-> (C_coeffs, E_coeffs)

Pre-compute the polynomial coefficient arrays for the joint operator `D_r(s) =
[C_r(s)  E_r(s)]` that couples the orthogonality equations to the unknown
reduced dynamics and to the external forcing.

## Return values

- `C_coeffs :: Vector{Matrix{T}}` of length `ROM`, where `C_coeffs[r]` is an
  `(ORD-1) × ROM` matrix.  Row `j` of `C_coeffs[r]` is the degree-`(j-1)`
  coefficient of the master-mode block of `D_r`:
  ```
  C_r(s) = Σ_{j=1}^{ORD-1} C_coeffs[r][j, :] · s^{j-1}      (1 × ROM row polynomial)
  ```

- `E_coeffs :: Vector{Matrix{T}}` of length `ROM`, where `E_coeffs[r]` is an
  `(ORD-1) × N_EXT` matrix.  Row `j` of `E_coeffs[r]` is the degree-`(j-1)`
  coefficient of the external-forcing block of `D_r`:
  ```
  E_r(s) = Σ_{j=1}^{ORD-1} E_coeffs[r][j, :] · s^{j-1}      (1 × N_EXT row polynomial)
  ```

  When `ORD = 1` both matrices have zero rows and the polynomials are identically
  zero; the corresponding blocks are absent from the assembled system.

## Arguments

- `J_coeffs                    :: Vector{<:AbstractMatrix{T}}` – output of
  [`precompute_orthogonality_operator_coefficients`](@ref); `J_coeffs[r]` is
  `ORD × FOM`.
- `generalised_right_eigenmodes :: AbstractMatrix{T}` of size `FOM × NVAR` –
  generalised eigenvectors; columns `1:ROM` are the master modes, columns
  `ROM+1:NVAR` are the external forcing modes.
- `reduced_dynamics_linear      :: AbstractMatrix{T}` of size `NVAR × NVAR` –
  Jordan-form matrix of the linear part of the reduced dynamics.

## Recurrence

For each mode `r`, a single downward pass computes the `NVAR`-vector polynomial
`Q_r` using two alternating buffers:

```
q ← Yᵀ · J_r[ORD, :]                                                    (j = ORD-1)
C_coeffs[r][ORD-1, :] ← q[1:ROM];   E_coeffs[r][ORD-1, :] ← q[ROM+1:NVAR]

q ← Yᵀ · J_r[j+1, :] + Λᵀ · q_prev,   j = ORD-2, …, 1
C_coeffs[r][j, :] ← q[1:ROM];   E_coeffs[r][j, :] ← q[ROM+1:NVAR]
```

Here `Yᵀ · v = Y' * v` is the FOM → NVAR projection of a row vector `vᵀ`
onto the eigenmode basis, and `Λᵀ · q` implements the right-multiply
`q · Λ` stored as a column.  The buffer swap avoids copying.

## Complexity

- Time:    `O(ROM · ORD · FOM · NVAR)` (one `NVAR`-vector update per `(r, j)`)
- Storage: `O(ROM · ORD · NVAR)`
"""
function precompute_orthogonality_column_polynomials(
	J_coeffs::AbstractVector{<:AbstractMatrix{T}},
	generalised_right_eigenmodes::AbstractMatrix{T},   # FOM × NVAR
	reduced_dynamics_linear::AbstractMatrix{T},        # NVAR × NVAR
) where {T}
	ROM = length(J_coeffs)
	ORD = size(J_coeffs[1], 1)    # J_coeffs[r] is ORD × FOM
	FOM = size(generalised_right_eigenmodes, 1)
	NVAR = size(generalised_right_eigenmodes, 2)
	N_EXT = NVAR - ROM

	@assert size(J_coeffs[1], 2) == FOM "J_coeffs rows must have length FOM = $(FOM)."
	@assert size(reduced_dynamics_linear) == (NVAR, NVAR) "reduced_dynamics_linear must be NVAR × NVAR."
	@assert ROM ≥ 1 && ROM ≤ NVAR "ROM must satisfy 1 ≤ ROM ≤ NVAR = $(NVAR)."

	# C_coeffs[r] : (ORD-1) × ROM   — row j = degree-(j-1) coeff of C_r(s)
	# E_coeffs[r] : (ORD-1) × N_EXT — row j = degree-(j-1) coeff of E_r(s)
	C_coeffs = [Matrix{T}(undef, ORD - 1, ROM) for _ in 1:ROM]
	E_coeffs = [Matrix{T}(undef, ORD - 1, N_EXT) for _ in 1:ROM]

	# Two alternating NVAR-length buffers for the current and previous Q_r step.
	q     = Vector{T}(undef, NVAR)
	q_tmp = Vector{T}(undef, NVAR)

	for r in 1:ROM
		Jr = J_coeffs[r]   # ORD × FOM

		if ORD == 1
			# No Q_r terms exist; C_coeffs[r] and E_coeffs[r] are 0×… (already allocated).
			continue
		end

		# ── Step j = ORD-1: Q_r[ORD-1] = Yᵀ · J_r[ORD, :] ─────────────────
		mul!(q, generalised_right_eigenmodes', view(Jr, ORD, :))
		C_coeffs[r][ORD-1, :] .= @view q[1:ROM]
		N_EXT > 0 && (E_coeffs[r][ORD-1, :] .= @view q[(ROM+1):NVAR])

		# ── Steps j = ORD-2, …, 1: Q_r[j] = Yᵀ · J_r[j+1,:] + Λᵀ · Q_r[j+1] ──
		for j in (ORD-2):-1:1
			mul!(q_tmp, generalised_right_eigenmodes', view(Jr, j+1, :))  # q_tmp = Yᵀ · J_r[j+1,:]
			mul!(q_tmp, reduced_dynamics_linear', q, one(T), one(T))       # q_tmp += Λᵀ · Q_r[j+1]
			q, q_tmp = q_tmp, q                                            # swap buffers (no copy)
			C_coeffs[r][j, :] .= @view q[1:ROM]
			N_EXT > 0 && (E_coeffs[r][j, :] .= @view q[(ROM+1):NVAR])
		end
	end

	return C_coeffs, E_coeffs
end

# =============================================================================
# 3.  Fused Horner pass: orthogonality row L_r(s) and scalar lower-order RHS
# =============================================================================

"""
	evaluate_orthogonality_row_and_lower_order_rhs!(row, s,
													lower_order_couplings,
													J_coeffs_r)
	-> scalar_rhs :: T

Evaluate the orthogonality row operator `L_r(s)` **and** compute the scalar
lower-order right-hand-side contribution for mode `r` in a **single Horner
pass**, reusing the transient intermediate row vectors.

## Mathematical context

At step `j` of the Horner recurrence (before the scalar multiply by `s`),
the intermediate row vector

```
L_r[j](s) = Σ_{k=j+1}^{ORD} J_r[k, :] · s^{k-(j+1)}
```

is available.  Dotting with the pre-computed coupling vector `ξ[j]` gives the
scalar contribution of lower-order solution terms at step `j`:

```
contribution[j] = -L_r[j](s) · ξ[j]
```

The negative sign arises because these terms originate from the left-hand side
of the cohomological equation.  Summed over `j = 1, …, ORD-1`:

```
RHS_lower_r = -Σ_{j=1}^{ORD-1} L_r[j](s) · ξ[j]
```

The sum runs to `ORD-1` (one fewer than in [`InvarianceEquation`](@ref)) because
the joint operator `Q_r` has one fewer degree.  Sharing the loop with the
`L_r(s)` evaluation avoids recomputing the `L_r[j]` intermediates.

## Arguments

- `row                  :: AbstractVector{T}` – output buffer (length `FOM`),
  overwritten with `L_r(s) = Σ_{j=1}^{ORD} J_r[j, :] · s^{j-1}`.
- `s                    :: T` – evaluation superharmonic.
- `lower_order_couplings :: SVector{ORD_M1, <:AbstractVector{T}}` – coupling
  vectors `ξ[j]` for `j = 1, …, ORD-1`; each is a length-`FOM` vector.
- `J_coeffs_r           :: AbstractMatrix{T}` – `ORD × FOM` matrix; row `j`
  is `J_r[j, :]`, the degree-`(j-1)` coefficient of `L_r`.  Obtained from
  [`precompute_orthogonality_operator_coefficients`](@ref).

## Returns

The scalar lower-order RHS accumulation
`RHS_lower_r = -Σ_{j=1}^{ORD-1} L_r[j](s) · ξ[j]`.

## Complexity

`O(ORD · FOM)`, shared with the `L_r(s)` evaluation.
"""
function evaluate_orthogonality_row_and_lower_order_rhs!(
	row::AbstractVector{T},
	s::T,
	lower_order_couplings::SVector{ORD, <:AbstractVector{T}},
	J_coeffs_r::AbstractMatrix{T},  # ORD × FOM,  ORD = ORD_M1 + 1
) where {T, ORD}

	copyto!(row, view(J_coeffs_r, ORD, :))  # row ← J_r[ORD, :]  (highest degree)

	scalar_rhs = zero(T)
	for j in (ORD-1):-1:1
		# row = L_r[j](s) = Σ_{k=j+1}^{ORD} J_r[k, :] · s^{k-(j+1)}
		# Accumulate scalar dot: scalar_rhs -= row · ξ[j]
		scalar_rhs -= dot(row, lower_order_couplings[j])
		row .*= s
		row .+= view(J_coeffs_r, j, :)   # row ← row · s + J_r[j, :]
		# row = L_r[j-1](s) = Σ_{k=j}^{ORD} J_r[k, :] · s^{k-j}
	end
	# On exit: row = L_r(s) = Σ_{k=1}^{ORD} J_r[k, :] · s^{k-1}

	return scalar_rhs
end

# =============================================================================
# 4.  Resonant-column row evaluation: C_r(s) restricted to resonant modes
# =============================================================================

"""
	evaluate_orthogonality_column_row!(c, s, r, C_coeffs, resonance) -> c

Evaluate the resonant block of the joint operator row `C_r(s)` in-place via
Horner's method, overwriting the pre-allocated `|R|`-vector `c`.

`C_r(s) = Σ_{j=1}^{ORD-1} C_coeffs[r][j, :] · s^{j-1}` is a `1 × ROM` row
polynomial; this function evaluates it at `s` and extracts only the `|R|` entries
corresponding to resonant master modes (those with `resonance[j] == true`), in
increasing-`j` order.

`c` may be a plain `Vector{T}` or a row view `view(M, row, col_range)`.

## Horner recurrence (column-wise, no allocation)

For each resonant column index `j` independently:

```
val  ←  C_coeffs[r][ORD-1, j]
for L = ORD-2, …, 1:
	val ← val · s + C_coeffs[r][L, j]
c[resonant_rank(j)] ← val
```

Column `j` of `C_coeffs[r]` is contiguous in memory (Julia is column-major),
so each per-column Horner pass is cache-friendly.

## Arguments

- `c        :: AbstractVector{T}`           – output buffer (length `|R|`),
  overwritten with the resonant entries of `C_r(s)`.
- `s        :: T`                           – evaluation frequency.
- `r        :: Int`                         – 1-based master-mode index for the
  row equation (`1 ≤ r ≤ ROM`).
- `C_coeffs :: Vector{<:AbstractMatrix{T}}` – pre-computed coefficients from
  [`precompute_orthogonality_column_polynomials`](@ref);
  `C_coeffs[r]` is `(ORD-1) × ROM`.
- `resonance :: SVector{ROM, Bool}`         – `resonance[j]` is `true` iff master
  mode `j` is resonant at the current multi-index.

## Complexity

`O((ORD-1) · |R|)`, with no heap allocation.
"""
function evaluate_orthogonality_column_row!(
	c::AbstractVector{T},
	s::T,
	r::Int,
	C_coeffs::Vector{<:AbstractMatrix{T}},
	resonance::SVector{ROM, Bool},
) where {T, ROM}
	Cr = C_coeffs[r]       # (ORD-1) × ROM
	ORD_M1 = size(Cr, 1)      # ORD - 1

	if ORD_M1 == 0
		fill!(c, zero(T))
		return c
	end

	# Evaluate each resonant column of Cr independently via a scalar Horner pass.
	# Column j of Cr is C_coeffs[r][:, j], which is contiguous in memory.
	col = 1
	for j in eachindex(resonance)
		if resonance[j]
			val = Cr[ORD_M1, j]                  # highest-degree coefficient
			for L in (ORD_M1-1):-1:1
				val = val * s + Cr[L, j]
			end
			c[col] = val
			col += 1
		end
	end
	return c
end

# =============================================================================
# 5.  Scalar external-forcing RHS for one orthogonality equation
# =============================================================================

"""
	evaluate_orthogonality_external_rhs(s, r, external_dynamics, E_coeffs) -> T

Compute the scalar external-forcing contribution to the right-hand side of the
orthogonality equation for master mode `r`:

```
RHS_ext_r = -Σ_{e active} external_dynamics[e] · E_r_e(s)
```

where `E_r_e(s) = Σ_{j=1}^{ORD-1} E_coeffs[r][j, e] · s^{j-1}` is the scalar
polynomial for forcing mode `e` in the row equation for mode `r` (pre-computed by
[`precompute_orthogonality_column_polynomials`](@ref)).

The negative sign reflects that these terms are moved from the left-hand side of
the cohomological equation to the right-hand side.

## Sparse exploitation

Only the non-zero entries of `external_dynamics` are processed. For periodic
forcing of a few harmonics this is typically a small subset of `N_EXT`.

## Combined Horner pass

The non-zero contributions are combined into a single scalar polynomial

```
g(s) = Σ_{e active} external_dynamics[e] · E_r_e(s)
```

and evaluated in one Horner pass (`ORD-2` scalar multiplies and `ORD-2 · N_EXT_active`
scalar additions), instead of `N_EXT_active` separate Horner passes.

## Arguments

- `s                 :: T`                           – evaluation frequency.
- `r                 :: Int`                         – 1-based master-mode index
  (`1 ≤ r ≤ ROM`).
- `external_dynamics :: AbstractVector{T}`           – known amplitudes of the
  `N_EXT` external forcing modes; typically sparse.
- `E_coeffs          :: Vector{<:AbstractMatrix{T}}` – pre-computed coefficients
  from [`precompute_orthogonality_column_polynomials`](@ref);
  `E_coeffs[r]` is `(ORD-1) × N_EXT`.

## Returns

The scalar `RHS_ext_r = -g(s)`.

## Complexity

`O(N_EXT_active · (ORD-1))` for combining coefficients plus `O(ORD-1)` for the
single Horner evaluation.
"""
function evaluate_orthogonality_external_rhs(
	s::T,
	r::Int,
	external_dynamics::AbstractVector{T},
	E_coeffs::Vector{<:AbstractMatrix{T}},
) where {T}
	Er    = E_coeffs[r]
	N_EXT = length(external_dynamics)
	@assert size(Er, 2) == N_EXT "E_coeffs[r] must have N_EXT = $(N_EXT) columns."

	ORD_M1 = size(Er, 1)   # ORD - 1

	ORD_M1 == 0 && return zero(T)

	# Collect active (non-zero) external indices.
	active = findall(!iszero, external_dynamics)
	isempty(active) && return zero(T)

	# Combine active external contributions into a single scalar polynomial g(s),
	# then evaluate via a single Horner pass.
	g = zero(T)
	for e in active
		g += Er[ORD_M1, e] * external_dynamics[e]
	end
	for L in (ORD_M1-1):-1:1
		g *= s
		for e in active
			g += Er[L, e] * external_dynamics[e]
		end
	end

	return -g   # sign flip: term moved from LHS to RHS
end

# =============================================================================
# 6.  Full orthogonality matrix and RHS assembly
# =============================================================================

"""
	assemble_orthogonality_matrix_and_rhs(
			s, J_coeffs, C_coeffs, E_coeffs,
			resonance, lower_order_couplings, external_dynamics)
	-> (M, rhs)

Assemble the full orthogonality system matrix and the complete right-hand side
for one multi-index, stacking one scalar equation per resonant master mode.

## System matrix

```
M = [ L | C ]
```

where each row `r` (corresponding to the `r`-th resonant mode in increasing
order) is

```
M[r, :] = [ L_r(s)  C_r(s)|_resonant ]
```

`M` has size `nR × (FOM + nR)` with `nR = count(resonance)`.

- Columns `1 : FOM`         → `L` block; row `r` is `L_r(s)` (`1 × FOM`).
- Columns `FOM+1 : FOM+nR`  → `C` block; row `r` contains the `nR` resonant
							   entries of `C_r(s)`, in increasing-mode order.

## Right-hand side

```
rhs_r = RHS_lower_r + RHS_ext_r
```

where

```
RHS_lower_r  = -Σ_{j=1}^{ORD-1} L_r[j](s) · ξ[j]        (scalar dot products)
RHS_ext_r    = -Σ_{e active} external_dynamics[e] · E_r_e(s)    (scalar)
```

`RHS_lower_r` and the `L` row are computed in a **single fused Horner pass** by
[`evaluate_orthogonality_row_and_lower_order_rhs!`](@ref).  `RHS_ext_r` is then
computed by [`evaluate_orthogonality_external_rhs`](@ref) and added.

## Arguments

- `s                      :: T` – evaluation frequency `s_k = Σᵢ kᵢ λᵢ`.
- `J_coeffs               :: Vector{<:AbstractMatrix{T}}` – pre-computed
  operator row coefficients from
  [`precompute_orthogonality_operator_coefficients`](@ref); `J_coeffs[r]` is
  `ORD × FOM`.
- `C_coeffs               :: Vector{<:AbstractMatrix{T}}` – pre-computed resonant
  block coefficients from [`precompute_orthogonality_column_polynomials`](@ref);
  `C_coeffs[r]` is `(ORD-1) × ROM`.
- `E_coeffs               :: Vector{<:AbstractMatrix{T}}` – pre-computed external
  block coefficients from [`precompute_orthogonality_column_polynomials`](@ref);
  `E_coeffs[r]` is `(ORD-1) × N_EXT`.
- `resonance              :: SVector{ROM, Bool}` – `resonance[j]` is `true` iff
  master mode `j` is resonant at the current multi-index.
- `lower_order_couplings  :: SVector{ORD_M1, <:AbstractVector{T}}` – coupling
  vectors `ξ[j]` for `j = 1, …, ORD-1`; each is a length-`FOM` vector, obtained
  from `MORFE.LowerOrderCouplings.compute_lower_order_couplings`.
- `external_dynamics      :: AbstractVector{T}` – known amplitudes of the `N_EXT`
  external forcing modes; typically sparse.

## Returns

`(M, rhs)` where `M` is `nR × (FOM + nR)` and `rhs` is a length-`nR` vector.
"""
function assemble_orthogonality_matrix_and_rhs(
	s::T,
	J_coeffs::AbstractVector{<:AbstractMatrix{T}},
	C_coeffs::Vector{<:AbstractMatrix{T}},
	E_coeffs::Vector{<:AbstractMatrix{T}},
	resonance::SVector{ROM, Bool},
	lower_order_couplings::SVector{ORD_M1, <:AbstractVector{T}},
	external_dynamics::AbstractVector{T},
) where {T, ROM, ORD_M1}
	FOM = size(J_coeffs[1], 2)   # J_coeffs[r] is ORD × FOM
	nR  = count(resonance)
	M   = Matrix{T}(undef, nR, FOM + nR)
	rhs = Vector{T}(undef, nR)

	row = 1
	for r in eachindex(resonance)
		if resonance[r]
			# Fill L_r(s) into columns 1:FOM of row `row`, accumulate scalar lower-order RHS.
			rhs[row] = evaluate_orthogonality_row_and_lower_order_rhs!(
				view(M, row, 1:FOM), s, lower_order_couplings, J_coeffs[r],
			)

			# Fill the resonant block of C_r(s) into columns FOM+1:FOM+nR of row `row`.
			evaluate_orthogonality_column_row!(
				view(M, row, (FOM+1):(FOM+nR)), s, r, C_coeffs, resonance,
			)

			# Accumulate scalar external-forcing contribution.
			rhs[row] += evaluate_orthogonality_external_rhs(
				s, r, external_dynamics, E_coeffs,
			)

			row += 1
		end
	end

	return M, rhs
end

end # module MasterModeOrthogonality
