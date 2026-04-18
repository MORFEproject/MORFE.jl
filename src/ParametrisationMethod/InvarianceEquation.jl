"""
	InvarianceEquation

Assemble the cohomological linear systems that arise in the parametrisation method,
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
D[j] = Σ_{k=j+1}^{ORD+1} B[k] · generalised_right_eigenmodes
							 · reduced_dynamics_linear^{k-(j+1)}
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
| [`assemble_cohomological_matrix_and_rhs`](@ref)               | Full block-matrix and RHS assembly |
"""
module InvarianceEquation

using LinearAlgebra
using StaticArrays

export precompute_column_polynomials,
	precompute_master_column_polynomials,
	precompute_external_column_polynomials,
	evaluate_system_matrix_and_lower_order_rhs!,
	evaluate_column!,
	evaluate_external_rhs!,
	assemble_cohomological_matrix_and_rhs

# =============================================================================
# 1.  Coefficient pre-computation
# =============================================================================

"""
	precompute_column_polynomials(fom_matrices, generalised_right_eigenmodes,
								  reduced_dynamics_linear, ROM)
	-> (C_coeffs, E_coeffs)

Pre-compute the polynomial coefficient arrays for the cohomological operator
columns (master modes) and the external-forcing right-hand-side columns.

These arrays are computed once per polynomial order and reused for every
multi-index at that order.

## Return values

- `C_coeffs :: Vector{Matrix{T}}` of length `ROM`, where `C_coeffs[r]` is
  `FOM × ORD`.  Column `j` of `C_coeffs[r]` is the degree-`(j-1)` coefficient
  of the `r`-th reduced-dynamics operator column:
  ```
  C_r(s) = Σ_{j=1}^{ORD} C_coeffs[r][:, j] · s^{j-1}
  ```

- `E_coeffs :: Vector{Matrix{T}}` of length `N_EXT = NVAR - ROM`, where
  `E_coeffs[e]` is `FOM × ORD`.  Column `j` of `E_coeffs[e]` is the
  degree-`(j-1)` coefficient of the `e`-th external-forcing operator column:
  ```
  E_e(s) = Σ_{j=1}^{ORD} E_coeffs[e][:, j] · s^{j-1}
  ```

## Arguments

- `fom_matrices     :: NTuple{ORD+1, <:AbstractMatrix{T}}` – linear matrices of
  the full-order model; `fom_matrices[k+1]` corresponds to `B[k]` (0-indexed in
  the ODE).
- `generalised_right_eigenmodes :: AbstractMatrix{T}` of size `FOM × NVAR` –
  generalised eigenvectors; columns `1:ROM` are the master modes, columns
  `ROM+1:NVAR` are the external forcing modes.
- `reduced_dynamics_linear :: AbstractMatrix{T}` of size `NVAR × NVAR` –
  Jordan-form matrix of the linear part of the reduced dynamics on the SSM.
- `ROM :: Int` – number of master modes (dimension of the reduced-order model).

## Recurrence

The output matrices are filled by a single downward Horner recurrence
(`j` runs from `ORD` down to `1`) using one `FOM × NVAR` working buffer `D`:

```
D ← B[ORD+1] * generalised_right_eigenmodes                               (j = ORD)
C_coeffs[r][:, j] ← D[:, r]       for r = 1…ROM
E_coeffs[e][:, j] ← D[:, ROM+e]   for e = 1…N_EXT

D ← D * reduced_dynamics_linear + B[j+1] * generalised_right_eigenmodes  (j = ORD-1, …, 1)
C_coeffs[r][:, j] ← D[:, r]       for r = 1…ROM
E_coeffs[e][:, j] ← D[:, ROM+e]   for e = 1…N_EXT
```

After step `j`, column `j` of every per-target matrix holds the exact
degree-`(j-1)` coefficient

```
D[:, ·] = Σ_{k=j+1}^{ORD+1} B[k] * generalised_right_eigenmodes * reduced_dynamics_linear^{k-(j+1)}
```

## Complexity

- Time:    `O(ORD · FOM · NVAR)`  (dominated by `ORD` matrix–matrix products)
- Storage: `O(ORD · FOM · NVAR)`
"""
function precompute_column_polynomials(
	fom_matrices::NTuple{ORDP1, <:AbstractMatrix{T}}, # fom_matrices[k+1] = Bₖ,  k = 0,…,ORD
	generalised_right_eigenmodes::AbstractMatrix{T},  # FOM × NVAR
	reduced_dynamics_linear::AbstractMatrix{T},       # NVAR × NVAR  (generally upper triangular, or Jordan form)
	ROM::Int,
) where {ORDP1, T <: Number}

	ORD   = ORDP1 - 1                                # polynomial order; compile-time constant
	FOM   = size(fom_matrices[1], 1)
	NVAR  = size(generalised_right_eigenmodes, 2)
	N_EXT = NVAR - ROM

	@assert ORD ≥ 1 "ODE order ORD = length(fom_matrices) - 1 must be ≥ 1."
	@assert size(generalised_right_eigenmodes, 1) == FOM "generalised_right_eigenmodes must have FOM = $(FOM) rows."
	@assert size(reduced_dynamics_linear) == (NVAR, NVAR) "reduced_dynamics_linear must be NVAR × NVAR."
	@assert 1 ≤ ROM ≤ NVAR "ROM must satisfy 1 ≤ ROM ≤ NVAR = $(NVAR)."

	# Allocate output structures in their final per-target layout.
	# C_coeffs[r][:, j] holds the degree-(j-1) coefficient for master mode r.
	# E_coeffs[e][:, j] holds the degree-(j-1) coefficient for external mode e.
	C_coeffs = [Matrix{T}(undef, FOM, ORD) for _ in 1:ROM]
	E_coeffs = [Matrix{T}(undef, FOM, ORD) for _ in 1:N_EXT]

	# ── Downward Horner recurrence ────────────────────────────────────────────
	# D is the single FOM × NVAR working buffer.
	# Invariant before the write step at index j:
	#   D = Σ_{k=j+1}^{ORDP1} fom_matrices[k] * generalised_right_eigenmodes
	#         * reduced_dynamics_linear^{k-(j+1)}.
	# Each column of D is written directly into column j of the corresponding
	# per-target matrix; no intermediate storage or repacking is needed.
	D     = Matrix{T}(undef, FOM, NVAR)   # current Horner step
	D_tmp = Matrix{T}(undef, FOM, NVAR)   # scratch for the next step

	mul!(D, fom_matrices[ORDP1], generalised_right_eigenmodes) # D ← B[ORD+1] · Y  (step j = ORD)
	for r in 1:ROM
		C_coeffs[r][:, ORD] .= @view D[:, r]
	end
	for e in 1:N_EXT
		E_coeffs[e][:, ORD] .= -@view D[:, ROM+e] # sign flip: terms moved from the LHS to the RHS
	end

	for j in (ORD-1):-1:1
		mul!(D_tmp, D, reduced_dynamics_linear) # D_tmp ← D · Λ
		mul!(D_tmp, fom_matrices[j+1], generalised_right_eigenmodes, one(T), one(T)) # D_tmp += B[j+1] · Y
		D, D_tmp = D_tmp, D # swap buffers (no copy)
		for r in 1:ROM
			C_coeffs[r][:, j] .= @view D[:, r]
		end
		for e in 1:N_EXT
			E_coeffs[e][:, j] .= -@view D[:, ROM+e] # sign flip: terms moved from the LHS to the RHS
		end
	end

	return C_coeffs, E_coeffs
end

"""
	precompute_master_column_polynomials(fom_matrices, master_modes, Λ_master)
	-> (C_coeffs, D_master_steps)

Φ_ext-independent half of [`precompute_column_polynomials`](@ref).  Computes only
the master-mode column polynomials `C_r(s)` and saves the intermediate FOM×ROM
Horner buffer at every step for later reuse by
[`precompute_external_column_polynomials`](@ref).

Because `Λ` is upper-triangular, the master columns of the Horner buffer `D`
form a closed subsystem under the recurrence `D ← D·Λ + B[j+1]·Y`: for column
`r ≤ ROM`, `(D·Λ)[:,r] = Σ_{k≤r} D[:,k]·Λ[k,r]` depends only on master columns.
Therefore `C_coeffs` is independent of the external directions `Φ_ext`.

## Returns
- `C_coeffs :: Vector{Matrix{T}}` — same layout as from `precompute_column_polynomials`.
- `D_master_steps :: Vector{Matrix{T}}` — length `ORD`; `D_master_steps[j]` is the
  FOM×ROM master block of the Horner buffer **at the step that wrote `C_coeffs[:,j]`**.
  Used by `precompute_external_column_polynomials` to avoid recomputing master work.
"""
function precompute_master_column_polynomials(
	fom_matrices::NTuple{ORDP1, <:AbstractMatrix},
	master_modes::AbstractMatrix,  # FOM × ROM
	Λ_master::AbstractMatrix,      # ROM × ROM  (master eigenvalue block)
) where {ORDP1}
	T = promote_type(eltype(fom_matrices[1]), eltype(master_modes), eltype(Λ_master))

	ORD = ORDP1 - 1
	FOM = size(fom_matrices[1], 1)
	ROM = size(master_modes, 2)

	C_coeffs       = [Matrix{T}(undef, FOM, ORD) for _ in 1:ROM]
	D_master_steps = [Matrix{T}(undef, FOM, ROM) for _ in 1:ORD]

	D     = Matrix{T}(undef, FOM, ROM)
	D_tmp = Matrix{T}(undef, FOM, ROM)

	mul!(D, fom_matrices[ORDP1], master_modes)   # D ← B[ORD+1] · master_modes
	for r in 1:ROM
		C_coeffs[r][:, ORD] .= @view D[:, r]
	end
	copyto!(D_master_steps[ORD], D)

	for j in (ORD-1):-1:1
		mul!(D_tmp, D, Λ_master)                                         # D_tmp ← D · Λ_master
		mul!(D_tmp, fom_matrices[j+1], master_modes, one(T), one(T))     # D_tmp += B[j+1] · master_modes
		D, D_tmp = D_tmp, D
		for r in 1:ROM
			C_coeffs[r][:, j] .= @view D[:, r]
		end
		copyto!(D_master_steps[j], D)
	end

	return C_coeffs, D_master_steps
end

"""
	precompute_external_column_polynomials(fom_matrices, external_directions,
											reduced_dynamics_linear, D_master_steps)
	-> E_coeffs

Φ_ext-dependent half of [`precompute_column_polynomials`](@ref).  Computes the
external-mode column polynomials `E_e(s)` reusing the pre-saved master Horner
intermediates `D_master_steps` (from [`precompute_master_column_polynomials`](@ref))
instead of recomputing the master-column work.

Pass `external_directions = zeros(FOM, N_EXT)` to obtain the partial (Φ_ext = 0)
E_coeffs needed for the initial external-forcing solve.

## Arguments
- `D_master_steps` — from `precompute_master_column_polynomials`; length `ORD`,
  each `FOM × ROM`.  `D_master_steps[j]` is the master Horner buffer at step `j`.
"""
function precompute_external_column_polynomials(
	fom_matrices::NTuple{ORDP1, <:AbstractMatrix},
	external_directions::AbstractMatrix,          # FOM × N_EXT
	reduced_dynamics_linear::AbstractMatrix,      # NVAR × NVAR
	D_master_steps::Vector{<:AbstractMatrix},     # length ORD, each FOM × ROM
) where {ORDP1}
	T = promote_type(eltype(fom_matrices[1]), eltype(external_directions),
	                 eltype(reduced_dynamics_linear), eltype(D_master_steps[1]))

	ORD   = ORDP1 - 1
	FOM   = size(fom_matrices[1], 1)
	ROM   = size(D_master_steps[1], 2)
	N_EXT = size(external_directions, 2)
	NVAR  = ROM + N_EXT

	N_EXT == 0 && return Vector{Matrix{T}}()

	Λ_master_ext = view(reduced_dynamics_linear, 1:ROM, (ROM+1):NVAR)  # ROM × N_EXT
	Λ_ext        = view(reduced_dynamics_linear, (ROM+1):NVAR, (ROM+1):NVAR)  # N_EXT × N_EXT

	E_coeffs  = [Matrix{T}(undef, FOM, ORD) for _ in 1:N_EXT]
	D_ext     = Matrix{T}(undef, FOM, N_EXT)
	D_ext_tmp = Matrix{T}(undef, FOM, N_EXT)

	mul!(D_ext, fom_matrices[ORDP1], external_directions)  # D_ext ← B[ORD+1] · Φ_ext
	for e in 1:N_EXT
		E_coeffs[e][:, ORD] .= -@view D_ext[:, e]  # sign flip (LHS → RHS)
	end

	for j in (ORD-1):-1:1
		# D_ext ← D_ext · Λ_ext + D_master_steps[j+1] · Λ_master_ext + B[j+1] · Φ_ext
		# D_master_steps[j+1] is the master buffer at the PREVIOUS step (step j+1),
		# which is the state of D[:,1:ROM] at the start of this loop iteration.
		mul!(D_ext_tmp, D_ext, Λ_ext)
		mul!(D_ext_tmp, D_master_steps[j+1], Λ_master_ext, one(T), one(T))
		mul!(D_ext_tmp, fom_matrices[j+1], external_directions, one(T), one(T))
		D_ext, D_ext_tmp = D_ext_tmp, D_ext
		for e in 1:N_EXT
			E_coeffs[e][:, j] .= -@view D_ext[:, e]  # sign flip
		end
	end

	return E_coeffs
end

# =============================================================================
# 2.  Fused Horner pass: parametrisation operator L(s) and lower-order RHS
# =============================================================================

"""
	evaluate_system_matrix_and_lower_order_rhs!(parametrisation_operator,
												lower_order_rhs,
												s, lower_order_couplings,
												linear_terms)
	-> parametrisation_operator

Evaluate the parametrisation operator `L(s)` **and** accumulate the lower-order
right-hand-side contributions in a **single Horner pass**, reusing the transient
intermediate matrices that are available only during the polynomial evaluation.

## Mathematical context

At step `j` of the Horner recurrence (before the scalar multiply by `s`),
the intermediate matrix

```
L[j](s) = Σ_{k=j+1}^{ORD+1} B[k] · s^{k-(j+1)}
```

is available. Multiplying by the pre-computed coupling vector
`ξ[j] = lower_order_couplings[j]` gives the contribution of lower-order solution
terms at derivative order `j` to the right-hand side:

```
contribution[j] = -L[j](s) · ξ[j]
```

The negative sign arises because these terms originate from the left-hand side of
the cohomological equation and are transposed to the right-hand side.

Summed over `j = 1, …, ORD`, the full lower-order RHS is

```
lower_order_rhs = -Σ_{j=1}^{ORD} L[j](s) · ξ[j]
				= -Σ_{j=1}^{ORD} ( Σ_{k=j+1}^{ORD+1} B[k] · s^{k-(j+1)} ) · ξ[j]
```

This computation **must** share the Horner loop with `L(s)`: the `L[j]`
intermediates are transient, and recomputing them would double the
`O(ORD · FOM²)` work.

The coupling vectors are obtained from
`MORFE.LowerOrderCouplings.compute_lower_order_couplings` applied to the
lower-order multi-indices associated with each Horner step.

## Arguments

- `parametrisation_operator :: AbstractMatrix{T}` – output buffer (`FOM × FOM`),
  overwritten with `L(s) = Σ_{k=1}^{ORD+1} B[k] · s^{k-1}`.
- `lower_order_rhs :: AbstractVector{T}` – accumulator (length `FOM`), updated
  in-place. Must be initialised to zero (or the desired starting value) by the
  caller.
- `s :: T` – evaluation superharmonic.
- `lower_order_couplings :: SVector{ORD, <:AbstractVector{T}}` – coupling vectors
  `ξ[j]` for `j = 1,…,ORD`; each element is an `AbstractVector{T}` of length
  `FOM`.
- `linear_terms :: NTuple{ORD+1, <:AbstractMatrix{T}}` – `linear_terms[k] = B[k]`.

## Complexity

`O(ORD · FOM²)`, shared with the `L(s)` evaluation.
"""
function evaluate_system_matrix_and_lower_order_rhs!(
	parametrisation_operator::AbstractMatrix,
	lower_order_rhs::AbstractVector,
	s::Number,
	lower_order_couplings::AbstractVector{<:AbstractVector},
	linear_terms::NTuple{ORDP1, <:AbstractMatrix},
) where {ORDP1}
	T   = eltype(parametrisation_operator)  # output type set by the caller's buffer
	ORD = ORDP1 - 1
	@assert length(lower_order_couplings) == ORD "length(lower_order_couplings) must equal ORD = length(linear_terms) - 1."

	copyto!(parametrisation_operator, linear_terms[ORDP1]) # L ← B[ORD+1]

	for j in ORD:-1:1
		# Here: parametrisation_operator = L[j](s) = Σ_{k=j+1}^{ORD+1} B[k] · s^{k-(j+1)}.
		# Accumulate: lower_order_rhs -= L[j](s) · ξ[j].
		# mul!(y, M, x, -1, 1) computes y = y - M·x without allocation.
		mul!(lower_order_rhs, parametrisation_operator, lower_order_couplings[j], -one(T), one(T))

		rmul!(parametrisation_operator, s)             # L ← L · s
		parametrisation_operator .+= linear_terms[j]   # L ← L + B[j]
		# Here: parametrisation_operator = L_{j-1}(s) = Σ_{k=j}^{ORD+1} B[k] · s^{k-j}.
	end
	# On exit: parametrisation_operator = L_0(s) = Σ_{k=1}^{ORD+1} B[k] · s^{k-1} = L(s).
	return parametrisation_operator
end

# =============================================================================
# 3.  Single-column polynomial evaluation: C_r(s) = Σ_{L=0}^{ORD-1} D_{L,r} s^L
# =============================================================================

"""
	evaluate_column!(c, s, r, C_coeffs) -> c

Evaluate the `r`-th reduced-dynamics operator column

```
C_r(s) = Σ_{L=1}^{ORD} C_coeffs[r][:, L] · s^{L-1}
```

in-place via Horner's method, overwriting the pre-allocated `FOM`-vector `c`.

`c` may be a plain `Vector{T}` or a column view `view(M, :, col)`.

## Horner recurrence

`L` runs from `ORD-1` down to `1`:

```
c  ←  C_coeffs[r][:, ORD]              (highest-degree coefficient)
for L = ORD-1, …, 1:
	c ← c · s + C_coeffs[r][:, L]
```

Column access `C_coeffs[r][:, L]` reads contiguous memory (Julia is
column-major), so the loop touches sequential cache lines.

## Arguments

- `c        :: AbstractVector{T}`   – output buffer (length `FOM`), overwritten.
- `s        :: T`                   – evaluation frequency.
- `r        :: Int`                 – 1-based master-mode index (`1 ≤ r ≤ ROM`).
- `C_coeffs :: Vector{<:AbstractMatrix{T}}` – pre-computed coefficients from
  [`precompute_column_polynomials`](@ref); `C_coeffs[r]` is `FOM × ORD`.

## Complexity

`O(ORD · FOM)`
"""
function evaluate_column!(
	c::AbstractVector{T},
	s::T,
	r::Int,
	C_coeffs::Vector{<:AbstractMatrix{T}},
) where {T}
	Cr  = C_coeffs[r]              # FOM × ORD;  column L ↔ degree-(L-1) coefficient
	ORD = size(Cr, 2)

	ORD == 0 && (fill!(c, zero(T)); return c)

	copyto!(c, @view Cr[:, ORD])   # c ← highest-degree coefficient
	for L in (ORD-1):-1:1
		c .*= s
		c .+= @view Cr[:, L]       # c ← c · s + degree-(L-1) coefficient
	end
	return c
end

# =============================================================================
# 4.  External-forcing RHS accumulation
# =============================================================================

"""
	evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs) -> rhs

Accumulate the external-forcing contribution to the cohomological right-hand side:

```
rhs += Σ_{e=1}^{N_EXT} external_dynamics[e] · E_e(s)
```

where `E_e(s) = Σ_{L=1}^{ORD} E_coeffs[e][:, L] · s^{L-1}` is the `e`-th
external column polynomial (pre-computed by [`precompute_column_polynomials`](@ref)).

The sign flip was already absorbed into the pre‑computed coefficients E_coeffs, 
since they originate from the left‑hand side of the invariance equation and are 
moved to the right‑hand side, so the contribution is added to the RHS rather than subtracted.

## Sparse exploitation

Only the non-zero entries of `external_dynamics` are processed. For periodic
forcing of a few harmonics this is typically a small subset of `N_EXT`.

## Combined Horner pass

Rather than evaluating each `E_e(s)` independently, the non-zero contributions
are combined into a single degree-`(ORD-1)` vector polynomial

```
g(s) = Σ_{e active} external_dynamics[e] · E_e(s)
```

and evaluated in one Horner pass (`ORD-1` scalar-vector multiplies plus `ORD-1`
`saxpy` operations over `FOM`), instead of `N_EXT_active` separate Horner
passes.

## Arguments

- `rhs               :: AbstractVector{T}` – accumulator (length `FOM`), updated
  in-place.
- `s                 :: T`                 – evaluation frequency.
- `external_dynamics :: AbstractVector{T}` – known amplitudes of the `N_EXT`
  external forcing modes; typically sparse.
- `E_coeffs          :: Vector{<:AbstractMatrix{T}}` – pre-computed external
  coefficients from [`precompute_column_polynomials`](@ref); `E_coeffs[e]` is
  `FOM × ORD`.

## Complexity

- `O(N_EXT_active · FOM · ORD)` for combining coefficients.
- `O(FOM · ORD)` for the single Horner evaluation.
"""
function evaluate_external_rhs!(
	rhs::AbstractVector{T},
	s::T,
	external_dynamics::AbstractVector{T},
	E_coeffs::Vector{<:AbstractMatrix{T}},
	g::AbstractVector{T},   # pre-allocated FOM buffer; zeroed inside
) where {T}
	N_EXT = length(E_coeffs)
	@assert length(external_dynamics) == N_EXT "external_dynamics length must equal N_EXT = $(N_EXT)."
	isempty(E_coeffs) && return rhs

	ORD = size(E_coeffs[1], 2)

	# Check for all-zero external dynamics without allocating (replaces findall).
	all_zero = true
	for e in eachindex(external_dynamics)
		!iszero(external_dynamics[e]) && (all_zero = false; break)
	end
	all_zero && return rhs

	# Form the combined coefficient vector for each polynomial degree:
	#   g[:, L] = Σ_{e active} E_coeffs[e][:, L] · external_dynamics[e],  L = 1…ORD
	# then evaluate g(s) = Σ_{L=1}^{ORD} g[:, L] · s^{L-1} via a single Horner pass.
	fill!(g, zero(T))

	# Initialise with the highest-degree combined coefficient (degree ORD-1).
	for e in eachindex(external_dynamics)
		iszero(external_dynamics[e]) && continue
		@. g += E_coeffs[e][:, ORD] * external_dynamics[e]
	end

	# Descend through remaining degrees.
	for L in (ORD-1):-1:1
		g .*= s
		for e in eachindex(external_dynamics)
			iszero(external_dynamics[e]) && continue
			@. g += E_coeffs[e][:, L] * external_dynamics[e]
		end
	end

	rhs .+= g # addition: the sign flip was already absorbed into the coefficients E_coeffs
	return rhs
end

# Backward-compatible overload that allocates its own buffer.
function evaluate_external_rhs!(
	rhs::AbstractVector{T},
	s::T,
	external_dynamics::AbstractVector{T},
	E_coeffs::Vector{<:AbstractMatrix{T}},
) where {T}
	isempty(E_coeffs) && return rhs
	g = zeros(T, length(rhs))
	return evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs, g)
end

# =============================================================================
# 5.  Full cohomological-matrix and RHS assembly
# =============================================================================

"""
	assemble_cohomological_matrix_and_rhs(
			s, linear_terms, C_coeffs, E_coeffs,
			resonance, lower_order_couplings, external_dynamics)
	-> (M, rhs)

Assemble the full cohomological system matrix and the complete right-hand side
for one multi-index, in a minimal number of Horner passes.

## System matrix

```
M = [ L(s) | C_{j₁}(s)  C_{j₂}(s)  … ]
```

where `j₁ < j₂ < …` are the resonant master modes (`resonance[j] == true`).
`M` has size `FOM × (FOM + nR)` with `nR = count(resonance)`.

- Columns `1 : FOM`         → `L(s)` block, the parametrisation operator.
- Columns `FOM+1 : FOM+nR`  → one column `C[j](s)` per resonant master mode,
							   in increasing-`j` order.

## Right-hand side

```
rhs = lower_order_rhs + rhs_ext
```

where

```
lower_order_rhs = -Σ_{j=1}^{ORD} L[j](s) · ξ[j]
rhs_ext         = -Σ_{e=1}^{N_EXT} external_dynamics[e] · E_e(s)
```

The `L(s)` block and `lower_order_rhs` are computed in a **single fused Horner
pass** by [`evaluate_system_matrix_and_lower_order_rhs!`](@ref).
`rhs_ext` is then accumulated into the same vector by
[`evaluate_external_rhs!`](@ref).

## Arguments

- `s                     :: T` – evaluation frequency `s_k = Σᵢ kᵢ λᵢ`.
- `linear_terms          :: NTuple{ORD+1, <:AbstractMatrix{T}}` –
  `linear_terms[k] = B[k]`.
- `C_coeffs              :: Vector{<:AbstractMatrix{T}}` – pre-computed
  reduced-dynamics coefficients from [`precompute_column_polynomials`](@ref);
  `C_coeffs[j]` is `FOM × ORD`.
- `E_coeffs              :: Vector{<:AbstractMatrix{T}}` – pre-computed
  external-forcing coefficients from [`precompute_column_polynomials`](@ref);
  `E_coeffs[e]` is `FOM × ORD`.
- `resonance             :: SVector{ROM, Bool}` – `resonance[j]` is `true` iff
  master mode `j` is resonant at the current multi-index.
- `lower_order_couplings :: SVector{ORD, <:AbstractVector{T}}` – coupling vectors
  `ξ[j]` for `j = 1,…,ORD`; obtained from
  `MORFE.LowerOrderCouplings.compute_lower_order_couplings`.
- `external_dynamics     :: AbstractVector{T}` – known amplitudes of the `N_EXT`
  external forcing modes; typically sparse.

## Returns

`(M, rhs)` where `M` is `FOM × (FOM + nR)` and `rhs` is a length-`FOM` vector.
"""
function assemble_cohomological_matrix_and_rhs(
	s::Number,
	linear_terms::NTuple{ORDP1, <:AbstractMatrix},
	C_coeffs::Vector{<:AbstractMatrix},
	E_coeffs::Vector{<:AbstractMatrix},
	resonance::SVector{ROM, Bool},
	lower_order_couplings::AbstractVector{<:AbstractVector},
	external_dynamics::AbstractVector,
) where {ROM, ORDP1}
	T   = promote_type(typeof(s), eltype(linear_terms[1]))
	FOM = size(linear_terms[1], 1)
	nR  = count(resonance)
	M   = Matrix{T}(undef, FOM, FOM + nR)

	# Block 1: L(s) in columns 1:FOM, with fused lower-order RHS accumulation.
	rhs = zeros(T, FOM)
	evaluate_system_matrix_and_lower_order_rhs!(
		view(M, :, 1:FOM), rhs, s, lower_order_couplings, linear_terms,
	)

	# Blocks 2…: one column C[j](s) per resonant master mode, in increasing-j order.
	col = FOM + 1
	for j in eachindex(resonance)
		if resonance[j]
			evaluate_column!(view(M, :, col), s, j, C_coeffs)
			col += 1
		end
	end

	# Accumulate external-forcing contribution into rhs.
	evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs)

	return M, rhs
end

# Overload accepting a pre-allocated buffer for evaluate_external_rhs! (no alloc).
function assemble_cohomological_matrix_and_rhs(
	s::Number,
	linear_terms::NTuple{ORDP1, <:AbstractMatrix},
	C_coeffs::Vector{<:AbstractMatrix},
	E_coeffs::Vector{<:AbstractMatrix},
	resonance::SVector{ROM, Bool},
	lower_order_couplings::AbstractVector{<:AbstractVector},
	external_dynamics::AbstractVector,
	g_buffer::AbstractVector,   # pre-allocated FOM buffer for external RHS
) where {ROM, ORDP1}
	T   = promote_type(typeof(s), eltype(linear_terms[1]))
	FOM = size(linear_terms[1], 1)
	nR  = count(resonance)
	M   = Matrix{T}(undef, FOM, FOM + nR)

	rhs = zeros(T, FOM)
	evaluate_system_matrix_and_lower_order_rhs!(
		view(M, :, 1:FOM), rhs, s, lower_order_couplings, linear_terms,
	)

	col = FOM + 1
	for j in eachindex(resonance)
		if resonance[j]
			evaluate_column!(view(M, :, col), s, j, C_coeffs)
			col += 1
		end
	end

	evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs, g_buffer)

	return M, rhs
end

end # module InvarianceEquation
