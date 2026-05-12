# =============================================================================
# Fused Horner pass: orthogonality row L_r(s) and scalar lower-order RHS
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
        lower_order_couplings::AbstractVector{<:AbstractVector{T}},
        J_coeffs_r::AbstractMatrix{T}  # ORD × FOM,  ORD = ORD_M1 + 1
) where {T}
    ORD = length(lower_order_couplings)

    copyto!(row, view(J_coeffs_r, ORD, :))  # row ← J_r[ORD, :]  (highest degree)

    scalar_rhs = zero(T)
    for j in (ORD - 1):-1:1
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
