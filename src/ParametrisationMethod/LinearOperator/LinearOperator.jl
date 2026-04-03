module LinearOperator

# =============================================================================
#
#  LinearOperator.jl
#
#  Assembles the cohomological linear systems that arise in Spectral Submanifold
#  (SSM) reduction of high-dimensional dynamical systems.
#
#  For each multiindex **k** and evaluation frequency  s_k = Σᵢ kᵢ λᵢ,
#  the cohomological equation has the block structure
#
#      [ A(s_k) | C_{j₁}(s_k)  C_{j₂}(s_k)  … ] * [ w_k; r_{j₁}; r_{j₂}; … ] = RHS_k
#
#  where
#
#      A(s)   = Σ_{L=0}^{ORD}   B_L  s^L         (FOM × FOM)    system-matrix polynomial
#      C_j(s) = Σ_{L=0}^{ORD-1} D_{L,j} s^L      (FOM-vector)   inner / resonant columns
#      H_j(s) = Σ_{L=0}^{ORD-1} F_{L,j} s^L      (FOM-vector)   external / harmonic columns
#
#  The polynomial degree of A(s) is ORD, the order of the full-order model
#  (type parameter of NDOrderModel).  C_j and H_j have degree ORD-1 because
#  the cohomological structure reduces the order by one.
#
#  Two distinct integers called "order" appear in this module:
#
#    ORD    = order of the ODE = polynomial degree of A(s)
#
#  Because ORD is a compile-time constant encoded in the NDOrderModel type,
#  every function that receives an NDOrderModel dispatches on it.  The Horner
#  loops run over a statically known range, allowing the compiler to specialise
#  and, for small ORD, to unroll completely.
#
#  Contents
#  --------
#  1. CohomologicalCoefficients                 – precomputed D and F coefficient arrays
#  2. precompute_rhs_columns                    – fill D_{L,j} and F_{L,j} via recurrences
#  3. evaluate_system_matrix!                   – Horner evaluation of A(s) only
#  3a. evaluate_system_matrix_and_lower_order_rhs!
#                                               – single-pass: A(s) + Σ_L A_L · ξ_L
#  4. evaluate_inner_column!                    – Horner evaluation of C_j(s) in-place
#  5. evaluate_external_column!                 – Horner evaluation of H_j(s) in-place
#  6. assemble_cohomological_matrix             – build the full block matrix
#  7. compute_frequency                         – s_k = Σᵢ kᵢ λᵢ
#
# =============================================================================

using LinearAlgebra
using StaticArrays

# =============================================================================
# 1.  Data structure
# =============================================================================

"""
	CohomologicalCoefficients{T}

Stores the precomputed polynomial coefficient matrices for the right-hand-side
columns of the cohomological linear systems in SSM reduction.

## Storage layout

Every column polynomial (C_j or H_j) has degree ORD-1 and thus ORD coefficient
vectors per target.  Coefficients are stored **per target** so that Horner
evaluation reads a single matrix contiguously (Julia is column-major):

	C_coeffs[j][:, L+1]  =  D_{L, j}     (L = 0, …, ORD-1;  j = 1, …, ROM)
	H_coeffs[j][:, L+1]  =  F_{L, j}     (L = 0, …, ORD-1;  j = 1, …, N_EXT)

i.e. column 1 holds the degree-0 coefficient and column ORD holds the
highest-degree (degree ORD-1) coefficient.

## Fields

| Field       | Type                  | Size              | Description                        |
|:----------- |:--------------------- |:----------------- |:---------------------------------- |
| `C_coeffs`  | `Vector{Matrix{T}}`   | ROM × [FOM × ORD] | Inner-target column coefficients   |
| `H_coeffs`  | `Vector{Matrix{T}}`   | N_EXT×[FOM × ORD] | External-target column coefficients|
| `FOM`       | `Int`                 | —                 | Full-order model state dimension   |
| `ORD`       | `Int`                 | —                 | ODE order; degree of A(s)          |
| `ROM`       | `Int`                 | —                 | Number of inner (resonant) targets |
| `N_EXT`     | `Int`                 | —                 | Number of external targets         |
"""
struct CohomologicalCoefficients{ORD, T}
	C_coeffs::SVector{ORD, Matrix{T}}   # C_coeffs[j] is FOM × ORD  (inner targets)
	H_coeffs::SVector{ORD, Matrix{T}}   # H_coeffs[j] is FOM × ORD  (external targets)
	FOM::Int
	ROM::Int
	# N_EXT::Int
end

# =============================================================================
# 2.  Precomputation of D_{L,j} and F_{L,j}
# =============================================================================

"""
	precompute_rhs_columns(B, Y_inner, Y_ext, Λ_inner, Λ_clubsuit, Λ_ext)
		-> CohomologicalCoefficients{T}

Precompute the polynomial coefficient matrices  D_{L,j}  and  F_{L,j}  for the
right-hand-side column polynomials  C_j(s)  and  H_j(s)  of the cohomological
equations.

## Arguments

| Argument      | Size          | Description                                              |
|:------------- |:------------- |:-------------------------------------------------------- |
| `B`           | NTuple{ORD+1} | Coefficient matrices of A(s); `B[L+1] = B_L` (1-based). |
| `Y_inner`     | FOM × ROM     | Inner eigenvectors (master modes).                       |
| `Y_ext`       | FOM × N_EXT   | External eigenvectors (harmonic forcing shapes).         |
| `Λ_inner`     | ROM × ROM     | Diagonal inner eigenvalue matrix.                        |
| `Λ_clubsuit`  | N_EXT × N_EXT | Coupling eigenvalue matrix (♣).                          |
| `Λ_ext`       | ROM × N_EXT   | External / harmonic eigenvalue matrix.                   |

`ORD` is inferred as `length(B) - 1` and is a compile-time constant because
`B` is an `NTuple{ORD+1}`.  It equals the ODE order of the corresponding
`NDOrderModel`.

## Recurrences

### Inner coefficient matrices  D[L]  (FOM × ROM),  L = 1, …, ORD

Downward Horner recurrence (L goes from ORD to 1):

	D[ORD] = B[ORD+1] * Y_inner
	D[L]   = D[L+1] * Λ_inner  +  B[L+1] * Y_inner,    L = ORD-1, …, 1

Column j of D[L] gives the degree-(L-1) coefficient of C_j(s).

### External coefficient matrices  F[L]  (FOM × N_EXT),  L = 1, …, ORD

Computed in three sub-steps.

**Step A – base external matrices G[L]:**

	G[ORD] = B[ORD+1] * Y_ext
	G[L]   = G[L+1] * Λ_clubsuit  +  Λ_inner^{ORD-L} * Λ_ext,    L = ORD-1, …, 1

where `Λ_inner^{ORD-L}` is the (ORD-L)-th matrix power of Λ_inner.

**Step B – correction accumulator Ψ[j]  (ROM × N_EXT):**

	Ψ[1] = Λ_ext
	Ψ[j] = Ψ[j-1] * Λ_clubsuit  +  Λ_inner^j * Λ_ext,    j = 2, …, ORD-1

**Step C – final external matrices F[L]:**

	F[ORD] = G[ORD]
	F[L]   = G[L]  +  Σ_{j=L+1}^{ORD-1} B[j+1] * Ψ[j-L],    L = 1, …, ORD-1

## Complexity

- Time:    O(ORD² · FOM · max(ROM, N_EXT))
- Storage: O(ORD · FOM · (ROM + N_EXT))
"""
function precompute_rhs_columns(
	B::NTuple{ORDP1, <:AbstractMatrix{T}},  # B[L+1] = B_L,  L = 0,…,ORD
	Y_inner::AbstractMatrix{T},             # FOM × ROM
	Y_ext::AbstractMatrix{T},               # FOM × N_EXT
	Λ_inner::AbstractMatrix{T},             # ROM × ROM
	Λ_clubsuit::AbstractMatrix{T},          # N_EXT × N_EXT
	Λ_ext::AbstractMatrix{T},               # ROM × N_EXT
) where {ORDP1, T <: Number}

	ORD   = ORDP1 - 1                  # ODE order; compile-time constant
	FOM   = size(Y_inner, 1)
	ROM   = size(Y_inner, 2)
	N_EXT = size(Y_ext, 2)

	@assert ORD ≥ 1 "ODE order ORD = length(B) - 1 must be ≥ 1."
	@assert size(Y_inner, 1) == FOM && size(Y_ext, 1) == FOM "Y_inner and Y_ext must have FOM rows."

	# -----------------------------------------------------------------------
	# Precompute  BY_inner[k] = B[k] * Y_inner  (FOM × ROM)
	#             BY_ext[k]   = B[k] * Y_ext    (FOM × N_EXT)
	# for k = 1, …, ORD+1  (1-based; k ↔ B_{k-1}).
	# These products appear in both the D and G recurrences.
	# -----------------------------------------------------------------------
	BY_inner = [B[k] * Y_inner for k in 1:ORDP1]
	BY_ext   = [B[k] * Y_ext for k in 1:ORDP1]

	# -----------------------------------------------------------------------
	# Step 1 – Inner coefficient matrices  D[L]  (FOM × ROM),  L = 1,…,ORD
	#
	#   D[ORD] = B[ORD+1] * Y_inner
	#   D[L]   = D[L+1] * Λ_inner + B[L+1] * Y_inner,    L = ORD-1, …, 1
	# -----------------------------------------------------------------------
	D = Vector{Matrix{T}}(undef, ORD)

	D[ORD] = copy(BY_inner[ORDP1])
	for L in (ORD-1):-1:1
		D[L] = D[L+1] * Λ_inner .+ BY_inner[L+1]
	end

	# -----------------------------------------------------------------------
	# Step 2A – Base external matrices  G[L]  (FOM × N_EXT),  L = 1,…,ORD
	#
	#   G[ORD] = B[ORD+1] * Y_ext
	#   G[L]   = G[L+1] * Λ_clubsuit + Λ_inner^{ORD-L} * Λ_ext,  L = ORD-1, …, 1
	#
	# Λ_inner_pow tracks Λ_inner^{ORD-L}: starts at I (for L = ORD-1),
	# then gains one factor of Λ_inner per iteration.
	# -----------------------------------------------------------------------
	G = Vector{Matrix{T}}(undef, ORD)

	G[ORD] = copy(BY_ext[ORDP1])
	Λ_inner_pow = Matrix{T}(I, ROM, ROM)       # Λ_inner^0 = I
	for L in (ORD-1):-1:1
		Λ_inner_pow = Λ_inner_pow * Λ_inner   # now holds Λ_inner^{ORD-L}
		G[L] = G[L+1] * Λ_clubsuit .+ Λ_inner_pow * Λ_ext
	end

	# -----------------------------------------------------------------------
	# Step 2B – Correction accumulator  Ψ[j]  (ROM × N_EXT),  j = 1, …, ORD-1
	#
	#   Ψ[1] = Λ_ext
	#   Ψ[j] = Ψ[j-1] * Λ_clubsuit + Λ_inner^j * Λ_ext,  j = 2, …, ORD-1
	# -----------------------------------------------------------------------
	Ψ = Vector{Matrix{T}}(undef, max(ORD-1, 0))

	if ORD ≥ 2
		Ψ[1] = copy(Λ_ext)
		Λ_inner_pow = copy(Λ_inner)                # Λ_inner^1
		for j in 2:(ORD-1)
			Λ_inner_pow = Λ_inner_pow * Λ_inner    # Λ_inner^j
			Ψ[j] = Ψ[j-1] * Λ_clubsuit .+ Λ_inner_pow * Λ_ext
		end
	end

	# -----------------------------------------------------------------------
	# Step 2C – Final external matrices  F[L]  (FOM × N_EXT),  L = 1,…,ORD
	#
	#   F[ORD] = G[ORD]
	#   F[L]   = G[L] + Σ_{j=L+1}^{ORD-1} B[j+1] * Ψ[j-L],    L = 1, …, ORD-1
	# -----------------------------------------------------------------------
	F = Vector{Matrix{T}}(undef, ORD)

	F[ORD] = copy(G[ORD])
	for L in 1:(ORD-1)
		correction = zeros(T, FOM, N_EXT)
		for j in (L+1):(ORD-1)
			correction .+= B[j+1] * Ψ[j-L]
		end
		F[L] = G[L] .+ correction
	end

	# -----------------------------------------------------------------------
	# Step 3 – Repack into per-target layout for O(ORD · FOM) Horner evaluation.
	#
	# D[L][:, j] = D_{L-1, j}  →  C_coeffs[j][:, L] = D[L][:, j]
	# F[L][:, j] = F_{L-1, j}  →  H_coeffs[j][:, L] = F[L][:, j]
	#
	# After repacking, C_coeffs[j] and H_coeffs[j] are FOM × ORD;
	# column L holds the degree-(L-1) coefficient.
	# -----------------------------------------------------------------------
	C_coeffs = [Matrix{T}(undef, FOM, ORD) for _ in 1:ROM]
	H_coeffs = [Matrix{T}(undef, FOM, ORD) for _ in 1:N_EXT]

	for L in 1:ORD
		for j in 1:ROM
			@inbounds C_coeffs[j][:, L] .= D[L][:, j]
		end
		for j in 1:N_EXT
			@inbounds H_coeffs[j][:, L] .= F[L][:, j]
		end
	end

	return CohomologicalCoefficients{T}(C_coeffs, H_coeffs, FOM, ORD, ROM, N_EXT)
end

# =============================================================================
# 3.  System-matrix polynomial evaluation: A(s) = Σ_{L=0}^{ORD} B_L s^L
# =============================================================================

"""
	evaluate_system_matrix!(A, s, model::NDOrderModel{ORD}) -> A

Evaluate `A(s) = Σ_{L=0}^{ORD} B_L sᴸ` in-place via Horner's method,
overwriting the pre-allocated `FOM × FOM` matrix `A`.

`A` may be a plain `Matrix{T}` or a column-slice view into a larger matrix.

### Horner recurrence  (L runs from ORD−1 down to 0)

	A  ←  B_ORD
	for L = ORD−1, …, 0:   A ← A·s + B_L

### Compile-time specialisation

`ORD` is a type parameter of `NDOrderModel`, so this method is specialised at
compile time for each distinct model order.  The loop runs over the statically
known range `(ORD-1):-1:0`; for small `ORD` the compiler typically unrolls it
completely, eliminating loop overhead and enabling further optimisations.

`rmul!(A, s)` performs the in-place scalar multiply without allocating; it
correctly avoids the matrix–matrix form `mul!(A, A, s)` which is undefined
for a general `AbstractMatrix`.

## Arguments
- `A     :: AbstractMatrix{T}`        : output buffer (FOM × FOM), overwritten.
- `s     :: T`                        : evaluation point.
- `model :: NDOrderModel{ORD, ORDP1}` : supplies `model.linear_terms[L+1] = B_L`;
										`ORD` sets the loop bound at compile time.

!!! note
	When the lower-order RHS must also be accumulated, use
	[`evaluate_system_matrix_and_lower_order_rhs!`](@ref) to reuse the Horner
	intermediates at no extra FOM² cost.

## Complexity  O(ORD · FOM²)
"""
function evaluate_system_matrix!(
	A::AbstractMatrix{T},
	s::T,
	model::NDOrderModel{ORD, ORDP1},
) where {ORD, ORDP1, T}
	B = model.linear_terms          # NTuple{ORDP1};  B[L+1] = B_L

	copyto!(A, B[end])              # A ← B_ORD
	for L in (ORD-1):-1:0
		rmul!(A, s)                 # A ← A · s
		A .+= B[L+1]               # A ← A + B_L
	end
	return A
end

"""
	evaluate_system_matrix(s, model::NDOrderModel) -> Matrix{T}

Allocate and return `A(s) = Σ_{L=0}^{ORD} B_L sᴸ`.
"""
function evaluate_system_matrix(s::T, model::NDOrderModel) where {T}
	A = Matrix{T}(undef, model.n_fom, model.n_fom)
	return evaluate_system_matrix!(A, s, model)
end

# =============================================================================
# 3a.  Combined: A(s) evaluation + simultaneous lower-order RHS accumulation
# =============================================================================

"""
	evaluate_system_matrix_and_lower_order_rhs!(A, lower_order_rhs,
												s, lower_order_couplings,
												model::NDOrderModel{ORD}) -> A

Evaluate `A(s)` **and** accumulate the lower-order right-hand-side contributions
in a **single Horner pass**, reusing the intermediate matrices that are only
available during the polynomial evaluation.

## Mathematical context

At step L of the Horner recurrence the intermediate matrix

	A_L  =  Σ_{k=L+1}^{ORD}  B_k · s^{k-1-L}

is available before the L-th scalar multiply.  Multiplying A_L by the
pre-computed coupling vector `ξ_L = lower_order_couplings[L+1]` gives the
contribution of lower-order solution terms at step L to each system-order
component of the RHS:

	lower_order_rhs[i]  +=  A_L · ξ_L[i],    i = 1, …, ORD

Summed over L = 0, …, ORD-1 the full lower-order RHS is

	lower_order_rhs[i]  =  Σ_{L=0}^{ORD-1}  A_L · ξ_L[i]

This computation **must** share the Horner loop with A(s): the A_L matrices
are transient intermediates, and recomputing them would double the
O(ORD · FOM²) work.

The coupling vectors `ξ_L` are obtained from
`MORFE.LowerOrderCouplings.compute_lower_order_couplings` applied to the
lower-order multiindices associated with Horner step L.

## Arguments

- `A :: AbstractMatrix{T}` (FOM × FOM, **output**)
	Overwritten with `A(s) = Σ_{L=0}^{ORD} B_L sᴸ`.

- `lower_order_rhs :: AbstractVector{<:AbstractVector{T}}` (length ORD, **mutated**)
	Accumulated in-place; zero-initialise before calling if a fresh sum is needed.
	`lower_order_rhs[i]` corresponds to system-order component `i`
	(e.g. `i=1` → position, `i=2` → velocity for ORD = 2).

- `s :: T`
	Evaluation frequency.

- `lower_order_couplings :: AbstractVector` (length ORD)
	`lower_order_couplings[L+1]` (1-based) holds `ξ_L` at Horner step L.
	Each element must be indexable by system-order component and return an
	FOM-length vector; concretely the output of
	`compute_lower_order_couplings(multiindex_L, W, R)` satisfies this.

- `model :: NDOrderModel{ORD, ORDP1}`
	Supplies `model.linear_terms[L+1] = B_L`; `ORD` sets the loop bound at
	compile time, enabling full specialisation of the combined loop.

## Complexity  O(ORD · FOM²) — identical to `evaluate_system_matrix!`

The per-step accumulation uses `mul!(y, A, x, α, β)` (BLAS `dgemv`/`zgemv`),
which costs O(FOM²) per (L, i) pair and adds no allocation.  ORD is
typically ≤ 2, so the additional constant factor is negligible.
"""
function evaluate_system_matrix_and_lower_order_rhs!(
	A::AbstractMatrix{T},
	lower_order_rhs::AbstractVector{<:AbstractVector{T}},
	s::T,
	lower_order_couplings::SVector{ORD, <:AbstractVector{T}},
	model::NDOrderModel{ORD, ORDP1},
) where {ORD, ORDP1, T}
	B = model.linear_terms

	copyto!(A, B[end])                    # A ← B_ORD

	for L in (ORD-1):-1:0
		# A currently holds  A_L = Σ_{k=L+1}^{ORD} B_k · s^{k-1-L}.
		# mul!(y, M, x, α, β) computes  y = α·M·x + β·y  without allocation.
		ξ_L = lower_order_couplings[L+1]      # 1-based; coupling at step L
		for i in 1:ORD
			mul!(lower_order_rhs[i], A, ξ_L[i], one(T), one(T))
		end

		rmul!(A, s)                              # A ← A · s
		A .+= B[L+1]                           # A ← A + B_L
	end
	return A
end

# =============================================================================
# 4.  Inner-column polynomial evaluation: C_j(s) = Σ_{L=0}^{ORD-1} D_{L,j} s^L
# =============================================================================

"""
	evaluate_inner_column!(c, s, j, coeffs) -> c

Evaluate `C_j(s) = Σ_{L=0}^{ORD-1} D_{L,j} sᴸ` in-place via Horner's method,
overwriting the pre-allocated FOM-vector `c`.

`c` may be a plain `Vector{T}` or a column view `view(M, :, col)`.

### Horner recurrence  (L runs from ORD-2 down to 0)

	c  ←  D_{ORD-1, j}              (column ORD of  coeffs.C_coeffs[j])
	for L = ORD-2, …, 0:   c ← c·s + D_{L,j}

Column access `C_coeffs[j][:, L+1]` reads contiguous memory (Julia is
column-major), so the loop touches sequential cache lines.

## Arguments
- `c      :: AbstractVector{T}`         : output buffer (FOM), overwritten.
- `s      :: T`                         : evaluation frequency.
- `j      :: Int`                       : 1-based target index (1 ≤ j ≤ ROM).
- `coeffs :: CohomologicalCoefficients` : holds `C_coeffs[j]` (FOM × ORD).

## Complexity  O(ORD · FOM)
"""
function evaluate_inner_column!(
	c::AbstractVector{T},
	s::T,
	j::Int,
	coeffs::CohomologicalCoefficients{T},
) where {T}
	Cj  = coeffs.C_coeffs[j]       # FOM × ORD;  column L ↔ D_{L-1, j}
	ORD = coeffs.ORD

	ORD == 0 && (fill!(c, zero(T)); return c)

	copyto!(c, @view Cj[:, ORD])   # c ← D_{ORD-1, j}  (highest-degree coeff)
	for L in (ORD-2):-1:0
		c .*= s
		c .+= @view Cj[:, L+1]    # c ← c·s + D_{L, j}
	end
	return c
end

"""
	evaluate_inner_column(s, j, coeffs) -> Vector{T}

Allocate and return `C_j(s)`.
"""
function evaluate_inner_column(s::T, j::Int, coeffs::CohomologicalCoefficients{T}) where {T}
	c = Vector{T}(undef, coeffs.FOM)
	return evaluate_inner_column!(c, s, j, coeffs)
end

# =============================================================================
# 5.  External-column polynomial evaluation: H_j(s) = Σ_{L=0}^{ORD-1} F_{L,j} s^L
# =============================================================================

"""
	evaluate_external_column!(h, s, j, coeffs) -> h

Evaluate `H_j(s) = Σ_{L=0}^{ORD-1} F_{L,j} sᴸ` in-place via Horner's method,
overwriting the pre-allocated FOM-vector `h`.

Identical algorithm to [`evaluate_inner_column!`](@ref) but reads from
`coeffs.H_coeffs[j]` (external targets) rather than `coeffs.C_coeffs[j]`.

## Arguments
- `h      :: AbstractVector{T}`         : output buffer (FOM), overwritten.
- `s      :: T`                         : evaluation frequency.
- `j      :: Int`                       : 1-based external-target index (1 ≤ j ≤ N_EXT).
- `coeffs :: CohomologicalCoefficients` : holds `H_coeffs[j]` (FOM × ORD).

## Complexity  O(ORD · FOM)
"""
function evaluate_external_column!(
	h::AbstractVector{T},
	s::T,
	j::Int,
	coeffs::CohomologicalCoefficients{T},
) where {T}
	Hj  = coeffs.H_coeffs[j]
	ORD = coeffs.ORD

	ORD == 0 && (fill!(h, zero(T)); return h)

	copyto!(h, @view Hj[:, ORD])
	for L in (ORD-2):-1:0
		h .*= s
		h .+= @view Hj[:, L+1]
	end
	return h
end

"""
	evaluate_external_column(s, j, coeffs) -> Vector{T}

Allocate and return `H_j(s)`.
"""
function evaluate_external_column(s::T, j::Int, coeffs::CohomologicalCoefficients{T}) where {T}
	h = Vector{T}(undef, coeffs.FOM)
	return evaluate_external_column!(h, s, j, coeffs)
end

# =============================================================================
# 6.  Full cohomological-matrix assembly
# =============================================================================

"""
	assemble_cohomological_matrix(s, model, coeffs, resonance
								  [, lower_order_couplings]) -> M  or  (M, rhs)

Assemble the full cohomological system matrix for one multiindex:

	M = [ A(s) | C_{j₁}(s)  C_{j₂}(s)  … ]

where `j₁ < j₂ < …` are the resonant inner targets (`resonance[j] == true`).
`M` has size `FOM × (FOM + nR)` with `nR = count(resonance)`.

- Columns `1 : FOM`        → `A(s)` block (Horner evaluation specialised on ORD).
- Columns `FOM+1 : FOM+nR` → one column `C_j(s)` per resonant target,
							 in increasing-j order.

## Arguments
- `s         :: T`
	Evaluation frequency `= Σᵢ kᵢ λᵢ`.
- `model     :: NDOrderModel{ORD}`
	Supplies `B_L` matrices; `ORD` is extracted from the type at compile time.
- `coeffs    :: CohomologicalCoefficients{T}`
	Precomputed `D_{L,j}` (and `F_{L,j}`) data.
- `resonance :: SVector{ROM, Bool}`
	`resonance[j]` is true iff target `j` is resonant at the current multiindex.
- `lower_order_couplings` *(optional, length ORD)*
	Pre-computed couplings `ξ_L` for L = 0, …, ORD-1; see
	[`evaluate_system_matrix_and_lower_order_rhs!`](@ref).
	When supplied the lower-order RHS accumulation is fused with the A(s) pass.

## Returns
- Without `lower_order_couplings`: `M`.
- With    `lower_order_couplings`: `(M, lower_order_rhs)` where
  `lower_order_rhs :: Vector{Vector{T}}` has length ORD.
"""
function assemble_cohomological_matrix(
	s::T,
	model::NDOrderModel,
	coeffs::CohomologicalCoefficients{T},
	resonance::SVector{ROM, Bool},
) where {T, ROM}
	FOM = coeffs.FOM
	nR  = count(resonance)
	M   = Matrix{T}(undef, FOM, FOM + nR)

	evaluate_system_matrix!(view(M, :, 1:FOM), s, model)

	col = FOM + 1
	for j in eachindex(resonance)
		if resonance[j]
			evaluate_inner_column!(view(M, :, col), s, j, coeffs)
			col += 1
		end
	end
	return M
end

"""
	assemble_cohomological_matrix(s, model, coeffs, resonance, lower_order_couplings)

Overload that simultaneously builds `M` and accumulates the lower-order RHS in
the same Horner pass.  Returns `(M, lower_order_rhs)`.
"""
function assemble_cohomological_matrix(
	s::T,
	model::NDOrderModel,
	coeffs::CohomologicalCoefficients{T},
	resonance::SVector{ROM, Bool},
	lower_order_couplings::AbstractVector,
) where {T, ROM}
	FOM = coeffs.FOM
	nR  = count(resonance)
	ORD = length(lower_order_couplings[1])
	M   = Matrix{T}(undef, FOM, FOM + nR)

	lower_order_rhs = [zeros(T, FOM) for _ in 1:ORD]

	evaluate_system_matrix_and_lower_order_rhs!(
		view(M, :, 1:FOM), lower_order_rhs, s, lower_order_couplings, model,
	)

	col = FOM + 1
	for j in eachindex(resonance)
		if resonance[j]
			evaluate_inner_column!(view(M, :, col), s, j, coeffs)
			col += 1
		end
	end
	return M, lower_order_rhs
end

"""
	assemble_cohomological_matrix(s, model, coeffs, resonance_set, idx
								  [, lower_order_couplings]) -> M  or  (M, rhs)

Convenience overload: look up the resonance `SVector` at position `idx` in
`resonance_set.resonances` and delegate to the primary method.

	for (idx, midx) in enumerate(resonance_set.multiindices)
		s = compute_frequency(midx, Λ_master)
		M, rhs = assemble_cohomological_matrix(s, model, coeffs,
											   resonance_set, idx, couplings[idx])
		# solve  M * [w_k; r_resonant] = f_k - rhs  …
	end
"""
function assemble_cohomological_matrix(
	s::T,
	model::NDOrderModel,
	coeffs::CohomologicalCoefficients{T},
	resonance_set::ResonanceSet,
	idx::Int,
) where {T}
	return assemble_cohomological_matrix(s, model, coeffs, resonance_set.resonances[idx])
end

function assemble_cohomological_matrix(
	s::T,
	model::NDOrderModel,
	coeffs::CohomologicalCoefficients{T},
	resonance_set::ResonanceSet,
	idx::Int,
	lower_order_couplings::AbstractVector,
) where {T}
	return assemble_cohomological_matrix(
		s, model, coeffs, resonance_set.resonances[idx], lower_order_couplings,
	)
end

# =============================================================================
# 7.  Utility: evaluation frequency  s_k = Σᵢ kᵢ λᵢ
# =============================================================================

"""
	compute_frequency(multiindex, Λ_master) -> T

Compute the evaluation frequency  `s_k = Σᵢ kᵢ λᵢ`  for the cohomological
equation associated with multiindex **k** and master eigenvalues **Λ**.

`multiindex` can be any iterable of integers (`SVector`, `NTuple`, `Vector`, …).
`Λ_master` is the vector of master (inner) eigenvalues.
"""
function compute_frequency(multiindex, Λ_master::AbstractVector{T}) where {T}
	return sum(ki * λi for (ki, λi) in zip(multiindex, Λ_master); init = zero(T))
end

end # module
