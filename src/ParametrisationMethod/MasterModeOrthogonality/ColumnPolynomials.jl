# =============================================================================
# Pre-compute orthogonality row-operator coefficients J_r
# =============================================================================

"""
	precompute_orthogonality_operator_coefficients(fom_matrices, left_eigenmodes,
												   left_modes_derivatives = nothing)
	-> Vector{Matrix{T}}

Pre-compute the polynomial coefficient arrays for the orthogonality row operators
`J_r(s)` directly from the left-eigenvector order-blocks. No eigenvalue is used.

## Mathematical origin

The orthogonality condition for master mode `r` at a monomial with superharmonic
`s` is the sesquilinear B-orthogonality of the companion left eigenvector
`φ_r = [φ_{r,1}; …; φ_{r,ORD}]` (defined by `φ_rᴴ (λ_r B − A) = 0`) against the
first-order state `𝒲`:

```
φ_rᴴ B 𝒲 = 0,     B = blockdiag(I, …, I, B_ORD)
```

Reducing block-by-block, the row acting on the physical unknown `W` is

```
J_r(s) = Σ_{j=1}^{ORD} J_r[j, :] · s^(j-1)
```

with coefficients read straight off the (conjugated) eigenvector blocks:

```
J_r[j, :]   = conj(φ_{r,j})          j = 1, …, ORD-1
J_r[ORD, :] = conj(B_ORDᴴ φ_{r,ORD}) = conj(B_ORDᴴ ℓ_r)
```

The conjugation is the sesquilinear `ᴴ` of the condition, stored so that the
assembled matrix row acts **bilinearly** on `W` (`row · W = Σᵢ rowᵢ Wᵢ`).

## Arguments

- `fom_matrices :: NTuple{ORD+1, <:AbstractMatrix{T}}` – linear matrices of the
  full-order model; `fom_matrices[k+1]` corresponds to `B_k` (0-indexed).
- `left_eigenmodes :: AbstractMatrix{T}` – physical-space (highest-order) left
  eigenvector slice; `left_eigenmodes[:, r]` is `ℓ_r = φ_{r,ORD}` (length FOM).
- `left_modes_derivatives :: Union{Nothing, AbstractArray{T,3}}` – lower-order
  left eigenvector blocks, size `FOM × (ORD-1) × ROM`;
  `left_modes_derivatives[:, j, r] = φ_{r,j}`. Required when `ORD > 1`
  (the eigensolver returns them; see `solve_left`). May be `nothing` for `ORD == 1`.

## Return value

A `Vector{Matrix{T}}` of length `ROM`; entry `r` is the `ORD × FOM` matrix
`J_coeffs[r]` whose row `j` stores the degree-`(j-1)` coefficient of `J_r(s)`.

## Complexity

- Time:    one `B_ORDᴴ · ℓ_r` product per mode — `O(ROM · FOM²)` dense
  (`O(ROM · nnz)` sparse); the remaining blocks are copies.
- Storage: `O(ROM · ORD · FOM)`
"""
function precompute_orthogonality_operator_coefficients(
        fom_matrices::NTuple{ORDP1, <:AbstractMatrix},
        left_eigenmodes::AbstractMatrix,
        left_modes_derivatives::Union{Nothing, AbstractArray{<:Number, 3}} = nothing
) where {ORDP1}
    T = promote_type(eltype(fom_matrices[1]), eltype(left_eigenmodes))
    if left_modes_derivatives !== nothing
        T = promote_type(T, eltype(left_modes_derivatives))
    end
    ORD = ORDP1 - 1
    FOM = size(first(fom_matrices), 1)
    ROM = size(left_eigenmodes, 2)

    @assert ORD ≥ 1 "ODE order ORD = length(fom_matrices) - 1 must be ≥ 1."
    @assert ROM ≥ 1 "ROM must be ≥ 1."
    @assert size(left_eigenmodes, 1) == FOM "left_eigenmodes must be FOM × ROM ($(FOM) × $(ROM))."
    @assert ORD == 1 || left_modes_derivatives !== nothing """
     left_modes_derivatives must be provided for ORD > 1 systems.
     Supply a FOM × (ORD-1) × ROM array with the lower-order left eigenvector blocks
     left_modes_derivatives[:, j, r] = φ_{r,j} (as returned by solve_left).
     """
    if left_modes_derivatives !== nothing && ORD > 1
        @assert size(left_modes_derivatives) == (FOM, ORD - 1, ROM) """
          left_modes_derivatives must be FOM × (ORD-1) × ROM ($(FOM) × $(ORD-1) × $(ROM)).
          """
    end

    result = Vector{Matrix{T}}(undef, ROM)
    tmp = Vector{T}(undef, FOM)
    for r in 1:ROM
        ℓ = view(left_eigenmodes, :, r)
        J_r = Matrix{T}(undef, ORD, FOM)

        # Highest degree: J_r[ORD, :] = conj(B_ORDᴴ · ℓ_r)  (= ℓ_rᴴ B_ORD as a bilinear row)
        mul!(tmp, fom_matrices[ORDP1]', ℓ)
        @views J_r[ORD, :] .= conj.(tmp)

        # Lower degrees: J_r[j, :] = conj(φ_{r,j}) — the eigenvector order-blocks
        for j in 1:(ORD - 1)
            @views J_r[j, :] .= conj.(left_modes_derivatives[:, j, r])
        end

        result[r] = J_r
    end
    return result
end

# =============================================================================
# Pre-compute joint operator coefficients → (C_coeffs, E_coeffs)
# =============================================================================

"""
	precompute_orthogonality_column_polynomials(J_coeffs, right_master_blocks,
												external_directions,
												reduced_dynamics_linear)
	-> (C_coeffs, E_coeffs)

Pre-compute the coefficient arrays of the operators `C_r(s)` (coupling to the
unknown reduced dynamics) and `E_r(s)` (known external forcing) that appear in
the orthogonality equation for master mode `r`:

```
J_r(s) W + C_r(s) f_m = − E_r(s) f_e − Σ_k G_{r,k}(s) ξ_k
```

## Mathematical origin

With `G_{r,k}(s) = Σ_{j=k+1}^{ORD} J_r[j, :] s^(j-1-k)` the Horner tails of the
row operator, the couplings are bilinear contractions against the **right**
eigenmode order-blocks `Y_k` (`Y_1` = physical mode, `Y_{k+1}` = next derivative
block):

```
C_r(s) = Σ_{k=1}^{ORD-1} G_{r,k}(s) · Y_k^m        (master blocks, from the eigensolver)
E_r(s) = Σ_{k=1}^{ORD-1} G_{r,k}(s) · Y_k^e        (external blocks)
```

The master blocks are supplied directly (`right_master_blocks`). The external
blocks are generalised — the reduced linear dynamics couples them back to the
master modes — and are generated by the block recurrence

```
Y_1^e = Φ_ext,       Y_{k+1}^e = Y_k^e Λ_e + Y_k^m Λ_me
```

where `Λ_me = Λ[1:ROM, ROM+1:NVAR]` and `Λ_e = Λ[ROM+1:NVAR, ROM+1:NVAR]` are
the master↔external and external blocks of `reduced_dynamics_linear`. This is
the one place an eigenvalue-derived matrix remains, confined to the per-order
precompute.

All contractions are **bilinear** (`Σᵢ J_r[·,i] Y[i,·]`): the sesquilinear
conjugation of the orthogonality condition is already baked into `J_coeffs`.

## Arguments

- `J_coeffs :: Vector{<:AbstractMatrix{T}}` – output of
  [`precompute_orthogonality_operator_coefficients`](@ref); `J_coeffs[r]` is `ORD × FOM`.
- `right_master_blocks :: AbstractArray{T,3}` – right master-mode order-blocks,
  size `FOM × ORD × ROM`; `right_master_blocks[:, k, m] = Y_k^m[:, m]`
  (equal to the linear master monomials of the parametrisation `W`).
  Only blocks `k ≤ ORD-1` are used.
- `external_directions :: AbstractMatrix{T}` – physical external directions
  `Φ_ext`, size `FOM × N_EXT`.
- `reduced_dynamics_linear :: AbstractMatrix{T}` – `NVAR × NVAR` linear reduced
  dynamics; only the `Λ_me` and `Λ_e` blocks are read.

## Return values

- `C_coeffs :: Vector{Matrix{T}}` of length `ROM`; `C_coeffs[r]` is
  `(ORD-1) × ROM`, row `p` = degree-`(p-1)` coefficient of `C_r(s)`.
- `E_coeffs :: Vector{Matrix{T}}` of length `ROM`; `E_coeffs[r]` is
  `(ORD-1) × N_EXT`, row `p` = degree-`(p-1)` coefficient of `E_r(s)`.

When `ORD == 1` both matrices have zero rows and the operators are identically
zero (the corresponding blocks are absent from the assembled system).

## Complexity

- Time:    `O(ROM² · ORD² · FOM)` for the contractions plus
  `O(ORD · FOM · NVAR · N_EXT)` for the external block recurrence.
- Storage: `O(ROM · ORD · NVAR)`
"""
function precompute_orthogonality_column_polynomials(
        J_coeffs::AbstractVector{<:AbstractMatrix},
        right_master_blocks::AbstractArray{<:Number, 3},   # FOM × ORD × ROM
        external_directions::AbstractMatrix,               # FOM × N_EXT
        reduced_dynamics_linear::AbstractMatrix           # NVAR × NVAR
)
    T = promote_type(eltype(J_coeffs[1]), eltype(right_master_blocks),
        eltype(external_directions), eltype(reduced_dynamics_linear))
    ROM = length(J_coeffs)
    ORD = size(J_coeffs[1], 1)    # J_coeffs[r] is ORD × FOM
    FOM = size(J_coeffs[1], 2)
    N_EXT = size(external_directions, 2)
    NVAR = ROM + N_EXT

    @assert size(right_master_blocks, 1) == FOM &&
            size(right_master_blocks, 2) == ORD &&
            size(right_master_blocks, 3) == ROM """
             right_master_blocks must be FOM × ORD × ROM ($(FOM) × $(ORD) × $(ROM)).
             """
    @assert size(external_directions, 1) == FOM "external_directions must have FOM = $(FOM) rows."
    @assert size(reduced_dynamics_linear) == (NVAR, NVAR) "reduced_dynamics_linear must be NVAR × NVAR."

    # C_coeffs[r] : (ORD-1) × ROM   — row p = degree-(p-1) coeff of C_r(s)
    # E_coeffs[r] : (ORD-1) × N_EXT — row p = degree-(p-1) coeff of E_r(s)
    C_coeffs = [Matrix{T}(undef, ORD - 1, ROM) for _ in 1:ROM]
    E_coeffs = [Matrix{T}(undef, ORD - 1, N_EXT) for _ in 1:ROM]

    ORD == 1 && return C_coeffs, E_coeffs

    # External order-blocks: Y_1^e = Φ_ext, Y_{k+1}^e = Y_k^e Λ_e + Y_k^m Λ_me.
    Λ_me = view(reduced_dynamics_linear, 1:ROM, (ROM + 1):NVAR)
    Λ_e = view(reduced_dynamics_linear, (ROM + 1):NVAR, (ROM + 1):NVAR)
    Ye = Vector{Matrix{T}}(undef, ORD - 1)
    if N_EXT > 0
        Ye[1] = Matrix{T}(external_directions)
        for k in 1:(ORD - 2)
            Ye[k + 1] = Ye[k] * Λ_e + view(right_master_blocks, :, k, :) * Λ_me
        end
    end

    # Bilinear contractions of the row coefficients against the right blocks:
    #   C_coeffs[r][p, m] = Σ_{k=1}^{ORD-p} J_r[p+k, :] · Y_k^m[:, m]
    #   E_coeffs[r][p, e] = Σ_{k=1}^{ORD-p} J_r[p+k, :] · Y_k^e[:, e]
    for r in 1:ROM
        Jr = J_coeffs[r]
        for p in 1:(ORD - 1)
            for m in 1:ROM
                acc = zero(T)
                for k in 1:(ORD - p)
                    @inbounds for i in 1:FOM
                        acc += Jr[p + k, i] * right_master_blocks[i, k, m]
                    end
                end
                C_coeffs[r][p, m] = acc
            end
            for e in 1:N_EXT
                acc = zero(T)
                for k in 1:(ORD - p)
                    Yk = Ye[k]
                    @inbounds for i in 1:FOM
                        acc += Jr[p + k, i] * Yk[i, e]
                    end
                end
                E_coeffs[r][p, e] = acc
            end
        end
    end

    return C_coeffs, E_coeffs
end

# =============================================================================
# Corner-row evaluation: C_r(s) masked by the resonance vector
# =============================================================================

"""
	evaluate_orthogonality_column_row!(c, s, r, C_coeffs, resonance) -> c

Evaluate the joint operator row `C_r(s)` in-place via Horner's method, overwriting
the pre-allocated **length-`ROM`** vector `c`.

`C_r(s) = Σ_{j=1}^{ORD-1} C_coeffs[r][j, :] · s^{j-1}` is a `1 × ROM` row
polynomial.  Entry `c[j]` holds `C_r(s)[j]` when master mode `j` is resonant and
`zero(T)` otherwise — i.e. the layout is **expanded and masked**, indexed by the
mode `j` itself rather than compacted into resonant rank order.

Masking rather than compacting is what keeps the bordered cohomological system at
the constant size `FOM + ROM` for every monomial (one sparsity pattern, one cached
symbolic factorisation).  Dropping the non-resonant entries is lossless: the
matching orthogonality row pins `R_{j,α} = 0`, so those coefficients multiply zero.

`c` may be a plain `Vector{T}` or a row view `view(M, row, col_range)`.

## Horner recurrence (column-wise, no allocation)

For each resonant column index `j` independently:

```
val  ←  C_coeffs[r][ORD-1, j]
for L = ORD-2, …, 1:
	val ← val · s + C_coeffs[r][L, j]
c[j] ← val
```

Column `j` of `C_coeffs[r]` is contiguous in memory (Julia is column-major),
so each per-column Horner pass is cache-friendly.

## Arguments

- `c        :: AbstractVector{T}`           – output buffer (length `ROM`),
  fully overwritten: resonant entries with `C_r(s)`, the rest with zero.
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
        resonance::SVector{ROM, Bool}
) where {T, ROM}
    Cr = C_coeffs[r]       # (ORD-1) × ROM
    ORD_M1 = size(Cr, 1)      # ORD - 1

    if ORD_M1 == 0
        fill!(c, zero(T))
        return c
    end

    # Evaluate each resonant column of Cr independently via a scalar Horner pass.
    # Column j of Cr is C_coeffs[r][:, j], which is contiguous in memory.
    # Non-resonant modes are written as hard zeros, keeping the corner block masked
    # rather than compacted.
    for j in eachindex(resonance)
        if resonance[j]
            val = Cr[ORD_M1, j]                  # highest-degree coefficient
            for L in (ORD_M1 - 1):-1:1
                val = val * s + Cr[L, j]
            end
            c[j] = val
        else
            c[j] = zero(T)
        end
    end
    return c
end

# =============================================================================
# Scalar external-forcing RHS for one orthogonality equation
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
        E_coeffs::Vector{<:AbstractMatrix{T}}
) where {T}
    Er = E_coeffs[r]
    N_EXT = length(external_dynamics)
    @assert size(Er, 2) == N_EXT "E_coeffs[r] must have N_EXT = $(N_EXT) columns."

    ORD_M1 = size(Er, 1)   # ORD - 1

    ORD_M1 == 0 && return zero(T)

    # Check for all-zero external dynamics without allocating (replaces findall).
    all_zero = true
    for e in eachindex(external_dynamics)
        !iszero(external_dynamics[e]) && (all_zero = false; break)
    end
    all_zero && return zero(T)

    # Combine active external contributions into a single scalar polynomial g(s),
    # then evaluate via a single Horner pass.
    g = zero(T)
    for e in eachindex(external_dynamics)
        iszero(external_dynamics[e]) && continue
        g += Er[ORD_M1, e] * external_dynamics[e]
    end
    for L in (ORD_M1 - 1):-1:1
        g *= s
        for e in eachindex(external_dynamics)
            iszero(external_dynamics[e]) && continue
            g += Er[L, e] * external_dynamics[e]
        end
    end

    return -g   # sign flip: term moved from LHS to RHS
end
