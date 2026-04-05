module LinearOperator

# =============================================================================
#
#  LinearOperator.jl
#
#  Assembles the cohomological linear systems that arise in the parametrisation method,
#  a reduced order modelling technique for high-dimensional dynamical systems.
#
#  For each multiindex **k** and evaluation frequency  s_k = Σᵢ kᵢ λᵢ,
#  the cohomological equation has the block structure
#
#      [ L(s_k) | C_{j₁}(s_k)  C_{j₂}(s_k)  … ] * [ w_k; r_{j₁}; r_{j₂}; … ] = RHS_k
#
#  where
#
#      L(s) = Σ_{k=1}^{ORD+1} B_k s^{k-1}
#
#  is the parametrisation operator (characteristic-matrix polynomial of the
#  full-order model), and
#
#      C(s) = Σ_{j=1}^{ORD} D_j s^{j-1}
#
#  where we precompute the FOM × ROM coefficient matrices D_j for j = 1,…,ORD as
#
#      D_j = Σ_{k=j+1}^{ORD+1} B_k · generalised_eigenmodes
#                                        · reduced_dynamics_linear^{k-(j+1)} 
#
#  where `generalised_eigenmodes` collects the generalised eigenvectors (master modes
#  followed by external forcing modes) and `reduced_dynamics_linear` is the
#  Jordan-form matrix of the linear part of the reduced dynamics on the manifold.
#
#  Contents
#  --------
#  1.  precompute_rhs_columns                      – precompute D_{L,j} arrays
#  2.  evaluate_system_matrix_and_lower_order_rhs! – fused Horner pass for L(s) + lower-order RHS
#  3.  evaluate_column!                             – evaluate one C_j(s) column
#  4.  evaluate_external_rhs!                       – accumulate external-forcing RHS
#  5.  assemble_cohomological_matrix_and_rhs        – full block-matrix and RHS assembly
#
# =============================================================================

using LinearAlgebra
using StaticArrays

# =============================================================================
# 1.  Coefficient pre-computation
# =============================================================================

"""
	precompute_rhs_columns(B, generalised_eigenmodes, reduced_dynamics_linear, ROM)
	-> (C_coeffs, E_coeffs)
 
Precompute the polynomial coefficient arrays that appear in the cohomological
operator columns (for master modes) and in the external-forcing right-hand side.
 
## Return values
 
- `C_coeffs :: Vector{Matrix{T}}` of length ROM, where `C_coeffs[r]` is
  FOM × ORD.  Column j of `C_coeffs[r]` is the degree-(j-1) coefficient of the
  r-th reduced-dynamics operator column:
	  C_r(s) = Σ_{j=1}^{ORD} C_coeffs[r][:, j] · s^{j-1}.
 
- `E_coeffs :: Vector{Matrix{T}}` of length N_EXT = NVAR - ROM, where
  `E_coeffs[e]` is FOM × ORD.  Column j of `E_coeffs[e]` is the degree-(j-1)
  coefficient of the e-th external-forcing operator column:
	  E_e(s) = Σ_{j=1}^{ORD} E_coeffs[e][:, j] · s^{j-1}.
 
## Arguments
 
- `B::NTuple{ORD+1}` – linear matrices of the full-order
  model; `B[k+1]` corresponds to `B_k` (0-indexed in the ODE).
- `generalised_eigenmodes::AbstractMatrix{T}` of size FOM × NVAR –
  generalised eigenvectors; columns 1:ROM are the master modes, columns
  ROM+1:NVAR are the external forcing modes.
- `reduced_dynamics_linear::AbstractMatrix{T}` of size NVAR × NVAR –
  Jordan-form matrix of the linear part of the reduced dynamics on the SSM.
- `ROM::Int` – number of master modes (dimension of the reduced-order model).
 
## Recurrence
 
The output matrices are filled by a single downward Horner recurrence
(j goes from ORD down to 1) using one FOM × NVAR working buffer `D`:
 
	D ← B[ORDP1] * generalised_eigenmodes                               (j = ORD)
	C_coeffs[r][:, j] ← D[:, r]       for r = 1…ROM
	E_coeffs[e][:, j] ← D[:, ROM+e]   for e = 1…N_EXT
 
	D ← D * reduced_dynamics_linear + B[j+1] * generalised_eigenmodes  (j = ORD-1, …, 1)
	C_coeffs[r][:, j] ← D[:, r]       for r = 1…ROM
	E_coeffs[e][:, j] ← D[:, ROM+e]   for e = 1…N_EXT
 
After step j, column j of every per-target matrix holds the exact degree-(j-1)
coefficient
 
	D[:, ·] = Σ_{k=j+1}^{ORDP1} B[k] * generalised_eigenmodes * reduced_dynamics_linear^{k-(j+1)}.
 
## Complexity
 
- Time:    O(ORD · FOM · NVAR)  (dominated by ORD matrix–matrix products)
- Storage: O(ORD · FOM · NVAR)
"""
function precompute_rhs_columns(
	B::NTuple{ORDP1, <:AbstractMatrix{T}},          # B[k+1] = Bₖ,  k = 0,…,ORD
	generalised_eigenmodes::AbstractMatrix{T},           # FOM × NVAR
	reduced_dynamics_linear::AbstractMatrix{T},      # NVAR × NVAR  (Jordan form)
	ROM::Int,
) where {ORDP1, T <: Number}

	ORD   = ORDP1 - 1                                # polynomial order; compile-time constant
	FOM   = size(B[1], 1)
	NVAR  = size(generalised_eigenmodes, 2)
	N_EXT = NVAR - ROM

	@assert ORD ≥ 1 "ODE order ORD = length(B) - 1 must be ≥ 1."
	@assert size(generalised_eigenmodes, 1) == FOM "generalised_eigenmodes must have FOM = $(FOM) rows."
	@assert size(reduced_dynamics_linear) == (NVAR, NVAR) "reduced_dynamics_linear must be NVAR × NVAR."
	@assert 0 ≤ ROM ≤ NVAR "ROM must satisfy 0 ≤ ROM ≤ NVAR = $(NVAR)."

	# Allocate output structures in their final per-target layout.
	# C_coeffs[r][:, j] will hold the degree-(j-1) coefficient for master mode r.
	# E_coeffs[e][:, j] will hold the degree-(j-1) coefficient for external mode e.
	C_coeffs = [Matrix{T}(undef, FOM, ORD) for _ in 1:ROM]
	E_coeffs = [Matrix{T}(undef, FOM, ORD) for _ in 1:N_EXT]

	# ── Downward Horner recurrence ────────────────────────────────────────────
	# D is the single FOM × NVAR working buffer.
	# Invariant before the write step at index j:
	#   D = Σ_{k=j+1}^{ORDP1} B[k] * generalised_eigenmodes * reduced_dynamics_linear^{k-(j+1)}.
	# Each column of D is written directly into column j of the corresponding
	# per-target matrix; no intermediate storage or repacking is needed.
	D     = Matrix{T}(undef, FOM, NVAR)   # current Horner step
	D_tmp = Matrix{T}(undef, FOM, NVAR)   # scratch for the next step

	mul!(D, B[ORDP1], generalised_eigenmodes) # D ← B_{ORD+1}·Y  (step j = ORD)
	for r in 1:ROM
		C_coeffs[r][:, ORD] .= @view D[:, r];
	end
	for e in 1:N_EXT
		E_coeffs[e][:, ORD] .= @view D[:, ROM+e];
	end

	for j in (ORD-1):-1:1
		mul!(D_tmp, D, reduced_dynamics_linear) # D_tmp ← D · Λ
		mul!(D_tmp, B[j+1], generalised_eigenmodes, one(T), one(T))  # D_tmp ← D_tmp + B_{j+1} · Y
		D, D_tmp = D_tmp, D # swap buffers (no copy)
		for r in 1:ROM
			C_coeffs[r][:, j] .= @view D[:, r];
		end
		for e in 1:N_EXT
			E_coeffs[e][:, j] .= @view D[:, ROM+e];
		end
	end

	return C_coeffs, E_coeffs
end

# =============================================================================
# 2.  Fused Horner pass: parametrisation operator L(s) and lower-order RHS
# =============================================================================

"""
	evaluate_system_matrix_and_lower_order_rhs!(parametrisation_operator,
												lower_order_rhs,
												s, lower_order_couplings,
												linear_terms) -> parametrisation_operator

Evaluate the parametrisation operator `L(s)` **and** accumulate the lower-order 
right-hand-side  contributions in a **single Horner pass**, reusing the transient 
intermediate matrices that are available only during the polynomial evaluation.

## Mathematical context

At step `j` of the Horner recurrence (before the scalar multiply by `s`),
the intermediate matrix

	L_j(s) = Σ_{k=j+1}^{ORD+1} B_k · s^{k-(j+1)}

is available.  Multiplying by the precomputed coupling vector ξ_j =
`lower_order_couplings[j]` gives the contribution of lower-order solution terms
at derivative order j to the right-hand side:

	contribution_j = -L_j(s) · ξ_j

The negative sign arises because these terms originate from the left-hand side
of the cohomological equation and are transposed to the right-hand side.

Summed over j = 1, …, ORD, the full lower-order RHS accumulation is

	lower_order_rhs = -Σ_{j=1}^{ORD} L_j(s) · ξ_j
					= -Σ_{j=1}^{ORD} ( Σ_{k=j+1}^{ORD+1} B_k · s^{k-(j+1)} ) · ξ_j

This computation **must** share the Horner loop with L(s): the L_j intermediates
are transient, and recomputing them would double the O(ORD · FOM²) work.

The coupling vectors are obtained from
`MORFE.LowerOrderCouplings.compute_lower_order_couplings` applied to the
lower-order multiindices associated with each Horner step.

## Arguments

- `parametrisation_operator :: AbstractMatrix{T}` – output buffer (FOM × FOM),
  overwritten with L(s) = Σ_{k=1}^{ORD+1} B_k · s^{k-1}.
- `lower_order_rhs :: AbstractVector{T}` – accumulator (FOM), updated in-place.
  Must be initialised to zero (or the desired starting value) by the caller.
- `s :: T` – evaluation frequency.
- `lower_order_couplings :: SVector{ORD}` – coupling vectors ξ_j for j = 1,…,ORD;
  each element is an AbstractVector{T} of length FOM.
- `linear_terms :: NTuple{ORD+1}` – `linear_terms[k]` = B_k.

## Complexity  O(ORD · FOM²)  (shared with the L(s) evaluation)
"""
function evaluate_system_matrix_and_lower_order_rhs!(
	parametrisation_operator::AbstractMatrix{T},
	lower_order_rhs::AbstractVector{T},
	s::T,
	lower_order_couplings::SVector{ORD, <:AbstractVector{T}},
	linear_terms::NTuple{ORDP1, <:AbstractMatrix{T}},
) where {ORD, ORDP1, T}
	@assert ORDP1 == ORD + 1 "ORDP1 = length(linear_terms) must equal ORD + 1."

	copyto!(parametrisation_operator, linear_terms[ORDP1])   # L ← B_{ORD+1}

	for j in ORD:-1:1
		# Here: parametrisation_operator = L_j(s) = Σ_{k=j+1}^{ORD+1} B_k · s^{k-(j+1)}.
		# Accumulate: lower_order_rhs -= L_j(s) · ξ_j.
		# mul!(y, M, x, -1, 1) computes y = y - M·x without allocation.
		mul!(lower_order_rhs, parametrisation_operator, lower_order_couplings[j], -one(T), one(T))

		rmul!(parametrisation_operator, s)            # L ← L · s
		parametrisation_operator .+= linear_terms[j]  # L ← L + B_j
		# Here: parametrisation_operator = L_{j-1}(s) = Σ_{k=j}^{ORD+1} B_k · s^{k-j}.
	end
	# On exit: parametrisation_operator = L_0(s) = Σ_{k=1}^{ORD+1} B_k · s^{k-1} = L(s).
	return parametrisation_operator
end

# =============================================================================
# 3.  Inner-column polynomial evaluation: C_j(s) = Σ_{L=0}^{ORD-1} D_{L,j} s^L
# =============================================================================

"""
	evaluate_column!(c, s, j, C_coeffs) -> c

Evaluate `C_r(s) = Σ_{L=0}^{ORD-1} C_coeffs[r][:, L+1] · s^L` in-place via
Horner's method, overwriting the pre-allocated FOM-vector `c`.

`c` may be a plain `Vector{T}` or a column view `view(M, :, col)`.

### Horner recurrence  (L runs from ORD-2 down to 0)

	c  ←  C_coeffs[r][:, ORD]                (highest-degree coefficient)
	for L = ORD-1, …, 1:   c ← c · s + C_coeffs[r][:, L]

Column access `C_coeffs[r][:, L+1]` reads contiguous memory (Julia is
column-major), so the loop touches sequential cache lines.

## Arguments

- `c        :: AbstractVector{T}`    – output buffer (FOM), overwritten.
- `s        :: T`                    – evaluation frequency.
- `r        :: Int`                  – 1-based master-mode index (1 ≤ r ≤ ROM).
- `C_coeffs :: Vector{Matrix{T}}`   – precomputed coefficients from
  `precompute_rhs_columns`; `C_coeffs[j]` is FOM × ORD.

## Complexity  O(ORD · FOM)
"""
function evaluate_column!(
	c::AbstractVector{T},
	s::T,
	r::Int,
	C_coeffs::Vector{<:AbstractMatrix{T}},
) where {T}
	Cr  = C_coeffs[r]              # FOM × ORD;  column L ↔ D_{L-1, r}
	ORD = size(Cr, 2)

	ORD == 0 && (fill!(c, zero(T)); return c)

	copyto!(c, @view Cr[:, ORD])   # c ← D_{ORD-1, r}  (highest-degree coefficient)
	for L in (ORD-1):-1:1
		c .*= s
		c .+= @view Cr[:, L]   # c ← c · s + D_{L, r}
	end
	return c
end

# =============================================================================
# 4.  External-forcing RHS accumulation
# =============================================================================

"""
	evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs) -> rhs

Accumulate the external-forcing contribution to the cohomological right-hand side:

	rhs -= Σ_{e=1}^{N_EXT} external_dynamics[e] · E_e(s)

where `E_e(s) = Σ_{L=1}^{ORD} E_coeffs[e][:, L] · s^{L-1}` is the e-th
external column polynomial (precomputed by `precompute_rhs_columns`).

The negative sign reflects that these terms originate from the left-hand side of
the cohomological equation (the known forcing drives the right-hand side).

### Sparse exploitation

Only the nonzero entries of `external_dynamics` are processed.  For periodic
forcing of a few harmonics this is typically a small subset of N_EXT.

### Combined Horner pass

Rather than evaluating each E_e(s) independently, the nonzero contributions
are combined into a single degree-(ORD-1) vector polynomial

	g(s) = Σ_{e active} external_dynamics[e] · E_e(s)

and evaluated in one Horner pass (ORD-1 scalar-vector multiplies plus ORD-1
saxpy operations over FOM), instead of N_EXT_active separate Horner passes.

## Arguments

- `rhs               :: AbstractVector{T}` – accumulator (FOM), updated in-place.
- `s                 :: T`                 – evaluation frequency.
- `external_dynamics :: AbstractVector{T}` – known amplitudes of the N_EXT
  external forcing modes; typically sparse.
- `E_coeffs          :: Vector{Matrix{T}}` – precomputed external coefficients
  from `precompute_rhs_columns`; `E_coeffs[e]` is FOM × ORD.

## Complexity  O(N_EXT_active · FOM · ORD)  for combining coefficients +
			  O(FOM · ORD)                  for the single Horner evaluation.
"""
function evaluate_external_rhs!(
	rhs::AbstractVector{T},
	s::T,
	external_dynamics::AbstractVector{T},
	E_coeffs::Vector{<:AbstractMatrix{T}},
) where {T}
	N_EXT = length(E_coeffs)
	@assert length(external_dynamics) == N_EXT "external_dynamics length must equal N_EXT = $(N_EXT)."
	isempty(E_coeffs) && return rhs

	ORD = size(E_coeffs[1], 2)
	FOM = length(rhs)

	# Collect active (nonzero) external indices to avoid unnecessary work.
	active = findall(!iszero, external_dynamics)
	isempty(active) && return rhs

	# Form the combined coefficient vector for each polynomial degree:
	# g[:, L] = Σ_{e active} E_coeffs[e][:, L] · external_dynamics[e],  L = 1…ORD
	# then evaluate g(s) = Σ_{L=1}^{ORD} g[:, L] · s^{L-1} via a single Horner pass.
	g = zeros(T, FOM)

	# Initialise with the highest-degree combined coefficient (degree ORD-1).
	for e in active
		@. g += E_coeffs[e][:, ORD] * external_dynamics[e]
	end

	# Descend through remaining degrees.
	for L in (ORD-1):-1:1
		g .*= s
		for e in active
			@. g += E_coeffs[e][:, L] * external_dynamics[e]
		end
	end

	rhs .-= g   # sign flip: terms moved from the LHS to the RHS
	return rhs
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
for one multiindex, in a minimal number of Horner passes.

### System matrix

	M = [ L(s) | C_{j₁}(s)  C_{j₂}(s)  … ]

where `j₁ < j₂ < …` are the resonant master modes (`resonance[j] == true`).
`M` has size FOM × (FOM + nR) with `nR = count(resonance)`.

- Columns `1 : FOM`        → `L(s)` block, the parametrisation operator.
- Columns `FOM+1 : FOM+nR` → one column `C_j(s)` per resonant master mode,
							 in increasing-j order.

### Right-hand side

	rhs = lower_order_rhs + rhs_ext

where

	lower_order_rhs = -Σ_{j=1}^{ORD} L_j(s) · ξ_j
	rhs_ext         = -Σ_{e=1}^{N_EXT} external_dynamics[e] · E_e(s)

The `L(s)` block and `lower_order_rhs` are computed in a **single fused Horner
pass** by [`evaluate_system_matrix_and_lower_order_rhs!`](@ref).
`rhs_ext` is then accumulated into the same vector by
[`evaluate_external_rhs!`](@ref).

## Arguments

- `s                     :: T`                   – evaluation frequency s_k = Σᵢ kᵢ λᵢ.
- `linear_terms          :: NTuple{ORD+1}`       – `linear_terms[k]` = B_k.
- `C_coeffs              :: Vector{Matrix{T}}`   – precomputed reduced-dynamics
  coefficients; `C_coeffs[j]` is FOM × ORD.
- `E_coeffs              :: Vector{Matrix{T}}`   – precomputed external-forcing
  coefficients; `E_coeffs[e]` is FOM × ORD.
- `resonance             :: SVector{ROM, Bool}`  – `resonance[j]` is true iff
  master mode j is resonant at the current multiindex.
- `lower_order_couplings :: SVector{ORD}`        – coupling vectors ξ_j for
  j = 1,…,ORD; obtained from
  `MORFE.LowerOrderCouplings.compute_lower_order_couplings`.
- `external_dynamics     :: AbstractVector{T}`   – known amplitudes of the N_EXT
  external forcing modes; typically sparse.

## Returns

`(M, rhs)` where `M` is FOM × (FOM + nR) and `rhs` is a FOM vector.
"""
function assemble_cohomological_matrix_and_rhs(
	s::T,
	linear_terms::NTuple{ORDP1, <:AbstractMatrix{T}},
	C_coeffs::Vector{<:AbstractMatrix{T}},
	E_coeffs::Vector{<:AbstractMatrix{T}},
	resonance::SVector{ROM, Bool},
	lower_order_couplings::SVector{ORD, <:AbstractVector{T}},
	external_dynamics::AbstractVector{T},
) where {T, ROM, ORD, ORDP1}
	FOM = size(linear_terms[1], 1)
	nR  = count(resonance)
	M   = Matrix{T}(undef, FOM, FOM + nR)

	# Block 1: L(s) in columns 1:FOM, with fused lower-order RHS accumulation.
	rhs = zeros(T, FOM)
	evaluate_system_matrix_and_lower_order_rhs!(
		view(M, :, 1:FOM), rhs, s, lower_order_couplings, linear_terms,
	)

	# Blocks 2…: one column C_j(s) per resonant master mode, in increasing-j order.
	col = FOM + 1
	for j in eachindex(resonance)
		if resonance[j]
			evaluate_column!(view(M, :, col), s, j, C_coeffs)
			col += 1
		end
	end

	# Accumulate external-forcing contribution: rhs += rhs_ext.
	evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs)

	return M, rhs
end

end # module LinearOperator