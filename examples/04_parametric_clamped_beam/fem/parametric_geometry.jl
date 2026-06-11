"""
	parametric_geometry.jl

Build the θ-power-series coefficients of `det J(θ)` and `adj J(θ)`
for an **affine** reference map

	x(θ, x₀) = x₀ + θ φ(x₀)                  ⇒   J(θ) = J₀ + θ J₁

in 3D.  We deliberately do *not* assemble a series for `J⁻¹(θ)`:
wherever the weak form would call for `inv(J)`, we substitute the
exact identity

	J⁻¹  =  adj(J) / det(J)

and apply `adj(J)` (a degree-2 exact polynomial in θ) and
`1/det(J)` (a scalar reciprocal series) **separately** in the
assembly.  One factor of `1/det(J)` then cancels the `det(J)` from
the volume differential `dV = det(J) · dV₀`, leaving fewer reciprocal
factors and avoiding any tensor-valued reciprocal-series accumulation.

The routines are written for a general pair `(J₀, J₁)` and may be
called per quadrature point if `J₁` varies on the domain (e.g. a
bending or curving map).  For the uniform-stretch demo `J₁` is
constant.

──────────────────────────────────────────────────────────────────────
JOINT DET / ADJ COMPUTATION  (closed form, no inverse)
──────────────────────────────────────────────────────────────────────
For a 3D affine `J(θ) = J₀ + θ J₁`, `det J(θ)` is an *exact* polynomial
of degree ≤ 3 in θ, `adj J(θ)` of degree ≤ 2.  Their coefficients can
be computed directly from `J₀, J₁` *without* forming `inv(J₀)` and
without any Cayley–Hamilton-style cancellation: define the shared
intermediates

	A² = J₀ J₀     B² = J₁ J₁     AB = J₀ J₁     BA = J₁ J₀
	term_A = A² − J₀ · tr J₀      term_B = B² − J₁ · tr J₁
	tr_term_A = ½ tr(term_A)      tr_term_B = ½ tr(term_B)

and read off

	det J(θ):   c₀ = det J₀
				c₁ = tr(A² J₁) − tr_term_A · tr J₁ − tr J₀ · tr(AB)
				c₂ = tr(B² J₀) − tr_term_B · tr J₀ − tr J₁ · tr(AB)
				c₃ = det J₁

	adj J(θ):   C₀ = term_A − tr_term_A · I
				C₁ = (tr J₀ · tr J₁ − tr(AB)) · I
					   − (J₀ · tr J₁ + J₁ · tr J₀) + AB + BA
				C₂ = term_B − tr_term_B · I.

Both expansions are produced in one pass by `det_and_adj_series`; the
backward-compatible wrappers `det_series` and `adj_series` simply
take the first and second component of the returned tuple.

──────────────────────────────────────────────────────────────────────
INVERSE DETERMINANT via reciprocal recurrence
──────────────────────────────────────────────────────────────────────
The scalar series `1/det(J(θ))` is the *only* rational quantity in
the construction.  We obtain its coefficients from
`reciprocal_series(detJ_coeffs, N_θ)` in `theta_polynomials.jl`, a
generic O(N_θ²) recurrence valid for any polynomial p(θ) with
p(0) ≠ 0.  We don't specialise the reciprocal to the degree-3 case
(via the 3D multinomial closed-form) because the recurrence has the
same arithmetic cost and stays general.

──────────────────────────────────────────────────────────────────────
SANITY CHECK
──────────────────────────────────────────────────────────────────────
`check_adj_det_identity(J₀, J₁, adjJ_coeffs, detJ_coeffs)` returns
the norms of `[J(θ) · adj(J(θ)) − det(J(θ)) · I]_k` at each θ-power.
All should be ≤ 1e−12 (machine ε scaled by ‖J₁‖).  This verifies the
*polynomial* identity that underpins the whole construction, without
involving the reciprocal series.

Requires `theta_polynomials.jl` to be already loaded (`poly_mul`,
`poly_dot`).
"""

using Tensors
using LinearAlgebra

# We use `Tensor{2, 3, Float64, 9}` from Tensors.jl (NOT `SMatrix` from
# StaticArrays) so that all tensor operations stay inside the Tensors.jl
# algebra: in particular `⋅` between two `Tensor{2,3}` is the single
# contraction (matrix product), whereas `⋅` between a `Tensor` and an
# `SMatrix` falls back to `LinearAlgebra.dot` and silently returns a
# scalar — that mismatch was the cause of the `symmetric(::Float64)`
# error in earlier revisions.

const Tens3 = Tensor{2, 3, Float64, 9}

# Reuse the tolerance from theta_polynomials.jl (assumed already included).

"""
	det_and_adj_series(J₀, J₁) -> (detJ_coeffs, adjJ_coeffs)

Joint computation of `det(J₀ + θ J₁)` and `adj(J₀ + θ J₁)` as exact
polynomials in θ for 3D, sharing the intermediate matrix products
`A² = J₀ J₀`, `B² = J₁ J₁`, `AB = J₀ J₁`, `BA = J₁ J₀` and the scalar
traces `tr A`, `tr B`, `tr(AB)`, etc.

Returns
-------
- `detJ_coeffs :: Vector{Float64}`  of length 4:
		`det(J₀ + θ J₁) = c₀ + c₁ θ + c₂ θ² + c₃ θ³`
- `adjJ_coeffs :: Vector{Tens3}`    of length 3:
		`adj(J₀ + θ J₁) = C₀ + θ C₁ + θ² C₂`

Formulas
--------
	c₀  =  det J₀
	c₁  =  tr(J₀² J₁) − tr_term_J₀ · tr J₁ − tr J₀ · tr(J₀ J₁)
	c₂  =  tr(J₁² J₀) − tr_term_J₁ · tr J₀ − tr J₁ · tr(J₀ J₁)
	c₃  =  det J₁

	C₀  =  term_J₀ − tr_term_J₀ · I
	C₁  =  (tr J₀ · tr J₁ − tr(J₀ J₁)) · I − (J₀ · tr J₁ + J₁ · tr J₀)
											  + J₀ J₁ + J₁ J₀
	C₂  =  term_J₁ − tr_term_J₁ · I

where  `term_X    := X² − X · tr X`
	   `tr_term_X := ½ · tr(term_X) = ½ (tr(X²) − (tr X)²)`.

Notes
-----
These formulas are **direct**: they do not involve `inv(J₀)`, which
the older `det_series` did via `A = J₀⁻¹ J₁`.  For a general invertible
`J₀` this avoids one explicit inverse and one `inv(J₀)`-driven matrix
multiplication; for the demo's `J₀ = I` the saving is structural
rather than numerical, but the routine is now correct even for
singular *or* near-singular `J₀` (only `det J₀ = 0` is excluded, and
even then only because the constant term `c₀` vanishes).

Likewise the adjugate formula is the **closed-form 3D cofactor
expansion**, with no Cayley–Hamilton-style θ³ cancellation that the
previous `adj_series` had to verify post-hoc.

Source
------
Adapted from the Python reference
`determinant_and_adjugate_expansions_of_3x3_1parameter_degree1_matrix`
(see code review thread); cross-checked numerically against direct
`det(J₀ + xJ₁)` / `det·inv(J₀ + xJ₁)` evaluations at multiple `x`.
"""
function det_and_adj_series(J₀::Tens3, J₁::Tens3)
	I₃ = one(Tens3)

	# ---- traces and matrix products (each computed once) ----
	trA = tr(J₀)
	trB = tr(J₁)

	A_sq = J₀ ⋅ J₀
	B_sq = J₁ ⋅ J₁
	AB   = J₀ ⋅ J₁
	BA   = J₁ ⋅ J₀

	trAB = tr(AB)

	# term_X := X² − X · tr X        (used in both det and adj)
	term_A = A_sq - J₀ * trA
	term_B = B_sq - J₁ * trB

	# tr_term_X := ½ tr(term_X) = ½ (tr(X²) − (tr X)²)
	tr_term_A = 0.5 * tr(term_A)
	tr_term_B = 0.5 * tr(term_B)

	# ---- determinant coefficients ----
	c₀ = det(J₀)
	c₁ = tr(A_sq ⋅ J₁) - tr_term_A * trB - trA * trAB
	c₂ = tr(B_sq ⋅ J₀) - tr_term_B * trA - trB * trAB
	c₃ = det(J₁)
	detJ_coeffs = [c₀, c₁, c₂, c₃]

	# ---- adjugate coefficients ----
	C₀ = term_A - tr_term_A * I₃
	C₁ = (trA * trB - trAB) * I₃ - (J₀ * trB + J₁ * trA) + AB + BA
	C₂ = term_B - tr_term_B * I₃
	adjJ_coeffs = [C₀, C₁, C₂]

	return detJ_coeffs, adjJ_coeffs
end

"""
	det_series(J₀, J₁) -> Vector{Float64}   (length 4)

Coefficients of `det(J₀ + θ J₁)`.  Thin wrapper around
`det_and_adj_series` for callers that only need the determinant.
"""
det_series(J₀::Tens3, J₁::Tens3) = det_and_adj_series(J₀, J₁)[1]

"""
	adj_series(J₀, J₁) -> Vector{Tens3}   (length 3)

Coefficients of `adj(J₀ + θ J₁)`.  Thin wrapper around
`det_and_adj_series` for callers that only need the adjugate.
"""
adj_series(J₀::Tens3, J₁::Tens3) = det_and_adj_series(J₀, J₁)[2]

"""
	check_adj_det_identity(J₀, J₁, adjJ_coeffs, detJ_coeffs) -> Vector{Float64}

Residual norms `‖[J(θ) · adj(J(θ)) − det(J(θ)) · I]_k‖` for `k = 0, 1,
2, 3`.  All four should be at machine precision; the identity
`J · adj(J) = det(J) · I` is the *defining* property of the adjugate
and is what makes `J⁻¹ = adj(J) / det(J)` correct.  This replaces the
older `check_inv_series` (which composed adj and 1/det and verified
`J · J⁻¹ = I`) — by checking the underlying identity directly we
sidestep the reciprocal-series truncation and only test the exact
polynomial algebra in `adj_series` and `det_series`.
"""
function check_adj_det_identity(J₀::Tens3, J₁::Tens3,
	adjJ_coeffs::Vector{Tens3},
	detJ_coeffs::Vector{Float64})
	I₃ = one(Tens3)
	J_ser = [J₀, J₁]                          # length 2
	# J(θ)·adj(J(θ)) has degree 1 + 2 = 3 in θ — same as det(J(θ)).
	J_adj = poly_dot(J_ser, adjJ_coeffs, 3)   # length 4
	return [norm(J_adj[k+1] - detJ_coeffs[k+1] * I₃) for k in 0:3]
end
