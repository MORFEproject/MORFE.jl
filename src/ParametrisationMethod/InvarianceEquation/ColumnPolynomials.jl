# =============================================================================
# Coefficient pre-computation
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
	ORD = ORDP1 - 1                                # polynomial order; compile-time constant
	FOM = size(fom_matrices[1], 1)
	NVAR = size(generalised_right_eigenmodes, 2)
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
	D = Matrix{T}(undef, FOM, NVAR)   # current Horner step
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

	C_coeffs = [Matrix{T}(undef, FOM, ORD) for _ in 1:ROM]
	D_master_steps = [Matrix{T}(undef, FOM, ROM) for _ in 1:ORD]

	D = Matrix{T}(undef, FOM, ROM)
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

	ORD = ORDP1 - 1
	FOM = size(fom_matrices[1], 1)
	ROM = size(D_master_steps[1], 2)
	N_EXT = size(external_directions, 2)
	NVAR = ROM + N_EXT

	N_EXT == 0 && return Vector{Matrix{T}}()

	Λ_master_ext = view(reduced_dynamics_linear, 1:ROM, (ROM+1):NVAR)  # ROM × N_EXT
	Λ_ext = view(reduced_dynamics_linear, (ROM+1):NVAR, (ROM+1):NVAR)  # N_EXT × N_EXT

	E_coeffs = [Matrix{T}(undef, FOM, ORD) for _ in 1:N_EXT]
	D_ext = Matrix{T}(undef, FOM, N_EXT)
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
# Single-column polynomial evaluation: C_r(s) = Σ_{L=0}^{ORD-1} D_{L,r} s^L
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
	Cr = C_coeffs[r]              # FOM × ORD;  column L ↔ degree-(L-1) coefficient
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
# External-forcing RHS accumulation
# =============================================================================

"""
	evaluate_external_rhs!(rhs, s, external_dynamics, E_coeffs, g) -> rhs

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
- `g                 :: AbstractVector{T}` – pre-allocated `FOM`-length scratch
  buffer; zeroed on entry.

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
