# =============================================================================
# Pre-compute orthogonality row-operator coefficients J_r
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
        fom_matrices::NTuple{ORDP1, <:AbstractMatrix},
        left_eigenmodes::AbstractMatrix,
        master_eigenvalues::SVector{ROM}
) where {ORDP1, ROM}
    T = promote_type(eltype(fom_matrices[1]), eltype(left_eigenmodes), eltype(master_eigenvalues))
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
        for j in (ORD - 1):-1:1
            view(J_r, j, :) .= λ .* view(J_r, j+1, :)                       # copy and scale
            mul!(view(J_r, j, :), fom_matrices[j + 1]', ℓ, one(T), one(T))     # accumulate
        end

        result[r] = J_r
    end
    return result
end

# =============================================================================
# Pre-compute joint operator coefficients Q_r → (C_coeffs, E_coeffs)
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
        J_coeffs::AbstractVector{<:AbstractMatrix},
        generalised_right_eigenmodes::AbstractMatrix,   # FOM × NVAR
        reduced_dynamics_linear::AbstractMatrix        # NVAR × NVAR
)
    T = promote_type(eltype(J_coeffs[1]), eltype(generalised_right_eigenmodes),
        eltype(reduced_dynamics_linear))
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
    q = Vector{T}(undef, NVAR)
    q_tmp = Vector{T}(undef, NVAR)

    for r in 1:ROM
        Jr = J_coeffs[r]   # ORD × FOM

        if ORD == 1
            # No Q_r terms exist; C_coeffs[r] and E_coeffs[r] are 0×… (already allocated).
            continue
        end

        # ── Step j = ORD-1: Q_r[ORD-1] = Yᵀ · J_r[ORD, :] ─────────────────
        mul!(q, generalised_right_eigenmodes', view(Jr, ORD, :))
        C_coeffs[r][ORD - 1, :] .= @view q[1:ROM]
        N_EXT > 0 && (E_coeffs[r][ORD - 1, :] .= @view q[(ROM + 1):NVAR])

        # ── Steps j = ORD-2, …, 1: Q_r[j] = Yᵀ · J_r[j+1,:] + Λᵀ · Q_r[j+1] ──
        for j in (ORD - 2):-1:1
            mul!(q_tmp, generalised_right_eigenmodes', view(Jr, j+1, :))  # q_tmp = Yᵀ · J_r[j+1,:]
            mul!(q_tmp, reduced_dynamics_linear', q, one(T), one(T))       # q_tmp += Λᵀ · Q_r[j+1]
            q, q_tmp = q_tmp, q                                            # swap buffers (no copy)
            C_coeffs[r][j, :] .= @view q[1:ROM]
            N_EXT > 0 && (E_coeffs[r][j, :] .= @view q[(ROM + 1):NVAR])
        end
    end

    return C_coeffs, E_coeffs
end

# =============================================================================
# Resonant-column row evaluation: C_r(s) restricted to resonant modes
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
    col = 1
    for j in eachindex(resonance)
        if resonance[j]
            val = Cr[ORD_M1, j]                  # highest-degree coefficient
            for L in (ORD_M1 - 1):-1:1
                val = val * s + Cr[L, j]
            end
            c[col] = val
            col += 1
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
