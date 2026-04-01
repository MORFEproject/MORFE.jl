module LinearOperator

using LinearAlgebra
using StaticArrays

# ---------------------------------------------------------------------------
# PrecomputedData
# ---------------------------------------------------------------------------

"""
	PrecomputedData{T}

Precomputed coefficient vectors for the `C_j` columns of the cohomological matrix.

For each target `j = 1,…,ROM`, the vector polynomial

	C_j(s) = ∑_{k=0}^{ORD-1} D_{k,j} s^k,    D_{k,j} is FOM × 1

is stored column-wise in `D[j]` which is `FOM × ORD`: D[j][:, k+1] = D_{k,j}.

The coefficients satisfy the divided-difference identity

	D_{k,j} = ∑_{L=k+1}^{ORD} B_L Y_j λ_j^{L-1-k},   k = 0,…,ORD-1,

which equals the original expression

	C_j(s) = ∑_{L=1}^{ORD} ∑_{k=0}^{L-1} B_L Y_j λ_j^{L-1-k} s^k

but is numerically stable near the resonance `s ≈ λ_j` (avoids the
ill-conditioned divided difference `(s^L − λ_j^L)/(s − λ_j)`).

# Fields
- `D::Vector{Matrix{T}}` : `D[j]` has size `(FOM, ORD)`; column `k+1` = `D_{k,j}`
- `FOM::Int`             : full-order state dimension
- `ORD::Int`             : polynomial degree of `A(s)`
- `ROM::Int`       : total number of targets
"""
struct PrecomputedData{T}
	D::Vector{Matrix{T}}
	FOM::Int
	ORD::Int
	ROM::Int
end

# ---------------------------------------------------------------------------
# precompute
# ---------------------------------------------------------------------------

"""
	precompute(model::NDOrderModel, Y, λ) -> PrecomputedData{T}

Precompute the coefficient matrices `D[j]` for all targets `j = 1,…,ROM`.

Extracts the `B_L` matrices from `model.linear_terms` (`NTuple{ORDP1, MT}`,
1-based: `model.linear_terms[L+1] = B_L`) and the state dimension from
`model.n_fom`.

# Arguments
- `model`                                   : `NDOrderModel` (any `ORD`, `ORDP1`, `MT`)
- `Y::AbstractVector{<:AbstractVector{T}}`  : `Y[j]` = right eigenvector for target `j`
- `λ::AbstractVector{T}`                    : `λ[j]` = eigenvalue for target `j`

# Mathematical definition
For each `j` and `k = 0,…,ORD-1`:

	D_{k,j} = ∑_{L=k+1}^{ORD} (B_L Y_j) λ_j^{L-1-k}
			= D_{k+1,j} * λ_j + B_{k+1} Y_j # Horner's

# Complexity
`O(ROM · ORD² · FOM)` work,  `O(ROM · ORD · FOM)` storage.
"""
function precompute(
	model,
	Y::NTuple{ROM, <:AbstractVector{T}},
	λ::AbstractVector{T},
) where {T <: Number, ROM}

	B   = model.linear_terms          # NTuple{ORDP1, MT}: B[L+1] = B_L (1-based)
	ORD = length(B) - 1
	FOM = model.n_fom

	@assert ORD ≥ 1 "Model must have ORD ≥ 1"

	# D[j] = zeros(FOM, ORD): column k+1 accumulates D_{k,j}
	D = ntuple(_ -> zeros(T, FOM, ORD), ROM)
	V = Vector{T}(undef, FOM)         # reusable buffer for B_L * Y_j

	for j in 1:ROM
		yj = Y[j]
		λj = λ[j]
		Dj = D[j]

		for L in 1:ORD
			mul!(V, B[L+1], yj)       # V = B_L * Y_j (in-place, no allocation)

			# λpow is reset to 1 at the start of each L:
			#   k = L-1  →  exponent L-1-k = 0  →  λpow = 1
			#   k = L-2  →  exponent 1          →  λpow = λ_j
			#   k = 0    →  exponent L-1         →  λpow = λ_j^{L-1}
			λpow = one(T)
			for k in (L-1):-1:0
				@views Dj[:, k+1] .+= V .* λpow
				λpow *= λj
			end
		end
	end

	return PrecomputedData{T}(D, FOM, ORD, ROM)
end

function precompute(
	linear_terms::NTuple{ORD, <:AbstractMatrix{T}},
	FOM::Int,
	eigenmodes::NTuple{NVAR, <:AbstractVector{T}},
	eigenvalues::NTuple{ROM, T},
) where {ORD, NVAR, ROM, T <: Number}

	B = linear_terms          # NTuple{ORDP1, MT}: B[L+1] = B_L (1-based)

	@assert ORD ≥ 1 "Model must have ORD ≥ 1"

	D = ntuple(_ -> zeros(T, FOM, ROM), ORD)
	F = ntuple(_ -> zeros(T, FOM, ROM), ORD)

	D[ORD] = BYtilde[ORD]
	F[ORD] = BYhat[ORD]
	for L in (ORD-1):-1:1
		@inbounds D[L] = D[L+1] * eigenvalues_tilde_matrix + BYtilde[L]
		@inbounds F[L] = F[L+1] * eigenvalues_clubsuit_matrix + BYhat[L]
	end

	spadessuit[1] = eigenvalues_hat_matrix
	eigenvalues_tilde_matrix_pow = identity
	for L in 2:(ORD-1)
		eigenvalues_tilde_matrix_pow = eigenvalues_tilde_matrix_pow * eigenvalues_tilde_matrix
		@inbounds spadessuit[L] = spadessuit[L-1] * eigenvalues_clubsuit_matrix + eigenvalues_tilde_matrix_pow * eigenvalues_hat_matrix
	end

	for L in 1:ORD
		@inbounds G[L] += for k in L:(ORD-1)
			BYtilde[k+2] * spadessuit[k]
		end
	end

	return PrecomputedData{T}(D, FOM, ORD, ROM)
end

# ---------------------------------------------------------------------------
# Matrix-polynomial evaluation: A(s)
# ---------------------------------------------------------------------------

"""
	compute_A!(A, s, model)

Evaluate `A(s) = ∑_{L=0}^{ORD} B_L s^L` in-place using Horner's method,
writing the result into the pre-allocated matrix `A`.

Matrices `B_L` are read from `model.linear_terms[L+1]`.
`A` may be a `Matrix{T}` or a column-slice view of a larger matrix.

Horner recurrence:

	A ← B_ORD
	for L = ORD-1, …, 0:   A ← A·s + B_L

`rmul!(A, s)` is used for the scalar multiply (correct for `AbstractMatrix`;
avoids the matrix–matrix `mul!(A, A, s)` mistake).

# Complexity  `O(ORD · FOM²)`
"""
function compute_A!(A::AbstractMatrix{T}, s::T, model) where {T}
	B   = model.linear_terms
	ORD = length(B) - 1
	copyto!(A, B[end])        # A = B_ORD
	for L in (ORD-1):-1:0
		low_order_term .+= A * low_order_couplings[L]
		rmul!(A, s)           # A *= s  (in-place scalar multiply)
		A .+= B[L+1]          # A += B_L
	end
	return A
end

"""
	compute_A(s, model) -> Matrix{T}

Allocate and return `A(s) = ∑_{L=0}^{ORD} B_L s^L`.
"""
function compute_A(s::T, model) where {T}
	A = Matrix{T}(undef, model.n_fom, model.n_fom)
	return compute_A!(A, s, model)
end

# ---------------------------------------------------------------------------
# Vector-polynomial evaluation: C_j(s)
# ---------------------------------------------------------------------------

"""
	compute_C!(c, s, j, data)

Evaluate `C_j(s) = ∑_{k=0}^{ORD-1} D_{k,j} s^k` in-place using Horner's method,
writing the result into the pre-allocated vector `c`.

`c` may be a `Vector{T}` or a column view of a larger matrix
(e.g. `view(M, :, col)`).

Horner recurrence:

	c ← D_{ORD-1,j}
	for k = ORD-2, …, 0:   c ← c·s + D_{k,j}

Column accesses `Dj[:, k+1]` are O(FOM) reads of contiguous memory
(Julia is column-major).

# Complexity  `O(ORD · FOM)`
"""
function compute_C!(c::AbstractVector{T}, s::T, j::Int, data::PrecomputedData{T}) where {T}
	Dj  = data.D[j]
	ORD = data.ORD
	if ORD == 0
		fill!(c, zero(T))
		return c
	end
	# Initialise with the highest-degree coefficient D_{ORD-1,j} (column ORD)
	copyto!(c, @view Dj[:, ORD])
	for k in (ORD-2):-1:0 # Horner's method
		c .*= s
		c .+= @view Dj[:, k+1]    # add D_{k,j}
	end
	return c
end

"""
	compute_C(s, j, data) -> Vector{T}

Allocate and return `C_j(s) = ∑_{k=0}^{ORD-1} D_{k,j} s^k`.
"""
function compute_C(s::T, j::Int, data::PrecomputedData{T}) where {T}
	c = Vector{T}(undef, data.FOM)
	return compute_C!(c, s, j, data)
end

# ---------------------------------------------------------------------------
# Full cohomological-matrix assembly
# ---------------------------------------------------------------------------

"""
	build_matrix(s, model, data, resonance) -> Matrix{T}

Assemble the cohomological linear system matrix for a single multiindex:

	M = [ A(s) | C_{j₁}(s)  C_{j₂}(s)  … ]

where `j₁, j₂, …` are the resonant targets (`resonance[j] == true`).
`M` has size `FOM × (FOM + nR)`, where `nR = count(resonance)`.

The `A(s)` block occupies columns `1:FOM`; each resonant `C_j(s)` column
follows in order of increasing `j`.

# Arguments
- `s::T`                    : evaluation frequency (`= ∑_i k_i Λ_i` for the
							  current multiindex `k` and master eigenvalues `Λ`;
							  see `compute_s`)
- `model`                   : `NDOrderModel` supplying the `B_L` matrices
- `data::PrecomputedData{T}`: precomputed `D_{k,j}` coefficient data
- `resonance`               : `SVector{ROM, Bool}`
							  `resonance[j] == true` labels target `j` as resonant at the current multiindex
"""
function build_matrix(
	s::T,
	model,
	data::PrecomputedData{T},
	resonance::SVector{ROM, Bool},
) where {T}
	FOM = data.FOM
	nR  = count(resonance)
	M   = Matrix{T}(undef, FOM, FOM + nR)

	# Left block: A(s) in columns 1:FOM
	compute_A!(view(M, :, 1:FOM), s, model)

	# Right block: one column C_j(s) per resonant target, in index order
	col = FOM + 1
	for j in eachindex(resonance)
		if resonance[j]
			compute_C!(view(M, :, col), s, j, data)   # 1-D view — not col:col
			col += 1
		end
	end
	return M
end

"""
	build_matrix(s, model, data, resonance_set, idx) -> Matrix{T}

Convenience overload: look up the resonance `SVector` at position `idx`
(1-based) in `resonance_set.resonances` and delegate to the primary method.

Use this when iterating over all multiindices in a `ResonanceSet`:

	for (idx, midx) in enumerate(resonance_set.multiindices)
		s = compute_s(midx, Λ)
		M = build_matrix(s, model, data, resonance_set, idx)
		# … solve cohomological equation …
	end

# Arguments
- `resonance_set::ResonanceSet` : stores `.resonances[idx]::SVector{ROM,Bool}`
- `idx::Int`                    : 1-based position of the current multiindex
"""
function build_matrix(
	s::T,
	model,
	data::PrecomputedData{T},
	resonance_set::ResonanceSet,
	idx::Int,
) where {T}
	return build_matrix(s, model, data, resonance_set.resonances[idx])
end

# ---------------------------------------------------------------------------
# Utility: evaluation frequency
# ---------------------------------------------------------------------------

"""
	compute_s(multiindex, Λ) -> T

Compute the evaluation frequency `s = ∑_i k_i Λ_i` for the cohomological
equation associated with multiindex `k` and master eigenvalues `Λ`.

This is the scalar at which `A(s)` and `C_j(s)` are evaluated.
`multiindex` can be any iterable of integers (e.g. `SVector`, `NTuple`, `Vector`).
"""
function compute_s(multiindex, Λ::AbstractVector{T}) where {T}
	return sum(ki * λi for (ki, λi) in zip(multiindex, Λ); init = zero(T))
end

end # module
