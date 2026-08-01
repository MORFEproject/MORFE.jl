# =============================================================================
# Fused Horner pass: orthogonality row Ĵ_r(s) and scalar lower-order RHS
# =============================================================================

"""
	evaluate_orthogonality_row_and_lower_order_rhs!(row, s,
													lower_order_couplings,
													J_coeffs_r)
	-> scalar_rhs :: T

Evaluate the orthogonality row operator `Ĵ_r(s)` **and** compute the scalar
lower-order right-hand-side contribution for mode `r` in a **single Horner
pass**, reusing the transient intermediate row vectors.

## Mathematical context

At step `j` of the Horner recurrence (before the scalar multiply by `s`),
the intermediate row vector

```
Ĵ_r[j](s) = Σ_{k=j+1}^{ORD} J_r[k, :] · s^{k-(j+1)}
```

is available.  Contracting **bilinearly** with the pre-computed coupling vector
`ξ[j]` gives the scalar contribution of lower-order solution terms at step `j`:

```
contribution[j] = -Ĵ_r[j](s) · ξ[j] = -Σᵢ Ĵ_r[j](s)ᵢ · ξ[j]ᵢ
```

The contraction must not conjugate: the sesquilinear conjugation of the
orthogonality condition is already baked into the row coefficients `J_r`
(see [`precompute_orthogonality_operator_coefficients`](@ref)), so `row`
holds the Horner tail `G_{r,j}(s)` with the `ᴴ` applied. The negative sign
arises because these terms originate from the left-hand side of the
cohomological equation.  Summed over `j = 1, …, ORD-1`:

```
RHS_lower_r = -Σ_{j=1}^{ORD-1} Ĵ_r[j](s) · ξ[j]
```

The sum runs to `ORD-1` (one fewer than in [`InvarianceEquation`](@ref)) because
the joint operator `Q_r` has one fewer degree.  Sharing the loop with the
`Ĵ_r(s)` evaluation avoids recomputing the `Ĵ_r[j]` intermediates.

## Arguments

- `row                  :: AbstractVector{T}` – output buffer (length `FOM`),
  overwritten with `Ĵ_r(s) = Σ_{j=1}^{ORD} J_r[j, :] · s^{j-1}`.
- `s                    :: T` – evaluation superharmonic.
- `lower_order_couplings :: SVector{ORD_M1, <:AbstractVector{T}}` – coupling
  vectors `ξ[j]` for `j = 1, …, ORD-1`; each is a length-`FOM` vector.
- `J_coeffs_r           :: AbstractMatrix{T}` – `ORD × FOM` matrix; row `j`
  is `J_r[j, :]`, the degree-`(j-1)` coefficient of `Ĵ_r`.  Obtained from
  [`precompute_orthogonality_operator_coefficients`](@ref).

## Returns

The scalar lower-order RHS accumulation
`RHS_lower_r = -Σ_{j=1}^{ORD-1} Ĵ_r[j](s) · ξ[j]`.

## Complexity

`O(ORD · FOM)`, shared with the `Ĵ_r(s)` evaluation.
"""
function evaluate_orthogonality_row_and_lower_order_rhs!(
	row::AbstractVector{T},
	s::T,
	lower_order_couplings::AbstractVector{<:AbstractVector{T}},
	J_coeffs_r::AbstractMatrix{T},  # ORD × FOM,  ORD = ORD_M1 + 1
) where {T}
	ORD = length(lower_order_couplings)

	copyto!(row, view(J_coeffs_r, ORD, :))  # row ← J_r[ORD, :]  (highest degree)

	scalar_rhs = zero(T)
	for j in (ORD-1):-1:1
		# row = Ĵ_r[j](s) = Σ_{k=j+1}^{ORD} J_r[k, :] · s^{k-(j+1)}
		# Accumulate bilinear contraction: scalar_rhs -= Σᵢ rowᵢ · ξ[j]ᵢ
		# (no conjugation — the sesquilinear ᴴ is already baked into J_r; a
		# conjugating dot() here would double-conjugate the row)
		ξ = lower_order_couplings[j]
		acc = zero(T)
		@inbounds for i in eachindex(row)
			acc += row[i] * ξ[i]
		end
		scalar_rhs -= acc
		row .*= s
		row .+= view(J_coeffs_r, j, :)   # row ← row · s + J_r[j, :]
		# row = Ĵ_r[j-1](s) = Σ_{k=j}^{ORD} J_r[k, :] · s^{k-j}
	end
	# On exit: row = Ĵ_r(s) = Σ_{k=1}^{ORD} J_r[k, :] · s^{k-1}

	return scalar_rhs
end
