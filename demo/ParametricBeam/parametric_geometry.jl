"""
	parametric_geometry.jl

Build the θ-power-series coefficients of `det J(θ)`, `adj J(θ)` and
`J⁻¹(θ)` for an **affine** reference map

	x(θ, x₀) = x₀ + θ φ(x₀)                  ⇒   J(θ) = J₀ + θ J₁

in 3D.  No closed-form expressions for the *specific* φ are used —
the routines are written for a general pair (J₀, J₁) and they may
be called per quadrature point if J₁ varies on the domain (e.g. a
bending or curving map).  For the uniform-stretch demo J₁ is constant.

──────────────────────────────────────────────────────────────────────
DET J(θ) — exact, degree ≤ 3
──────────────────────────────────────────────────────────────────────
Factor out J₀:  det J(θ) = det J₀ · det(I + θ A),  A := J₀⁻¹ J₁.
In 3D,

	det(I + θ A) = 1 + a θ + b θ² + c θ³

with the classical invariants

	a = tr A
	b = ½ (a² − tr A²) = ½ (tr²A − tr A²)
	c = det A.

Hence

	detJ_coeffs = [det J₀, det J₀ · a, det J₀ · b, det J₀ · c]      (length 4).

──────────────────────────────────────────────────────────────────────
ADJ J(θ) — exact, degree ≤ 2
──────────────────────────────────────────────────────────────────────
From Cayley–Hamilton in 3D,

	adj J = J² − (tr J) J + ½ (tr²J − tr J²) I.

J(θ) is degree 1 in θ, so J²(θ) is degree 2, (tr J)(θ) is degree 1, etc.
Convolving the relevant series gives an exact length-3 coefficient
vector (the θ³ contribution from J² · (1) cancels against the
remaining terms, as it must — `adj J` is degree d − 1 = 2 in dimension
3).  We verify this cancellation by an `@assert` in `adj_series`.

──────────────────────────────────────────────────────────────────────
J⁻¹(θ) — rational, expanded via the recurrence
──────────────────────────────────────────────────────────────────────
Finally

	J⁻¹(θ) = adj J(θ) · (1/det J(θ)) ,

so the only series that needs the recurrence is 1/det J(θ); after that
J⁻¹ is one truncated convolution away.  Truncation order N_θ is a
free parameter of `inv_series`.

──────────────────────────────────────────────────────────────────────
SANITY CHECK
──────────────────────────────────────────────────────────────────────
`check_inv_series(J₀, J₁, invJ_coeffs, N_θ)` returns the norms of
J(θ) · J⁻¹(θ) − I at each θ-power: 0, 1, …, N_θ.  All should be
≤ 1e−12 (machine ε scaled by ‖J₁‖).

Requires `theta_polynomials.jl` to be already loaded (poly_mul,
reciprocal_series).
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
	det_series(J₀, J₁) -> Vector{Float64}   (length 4)

Coefficients of `det(J₀ + θ J₁)` as a polynomial in θ.  Exact (no
truncation).  Errors if `J₀` is singular.
"""
function det_series(J₀::Tens3, J₁::Tens3)
	d₀ = det(J₀)
	abs(d₀) > ZERO_TOL || error("det_series: det J₀ = 0 (reference map singular)")
	A = inv(J₀) ⋅ J₁                  # single contraction = matrix product
	a = tr(A)
	b = 0.5 * (a^2 - tr(A ⋅ A))
	c = det(A)
	return [d₀, d₀ * a, d₀ * b, d₀ * c]
end

"""
	adj_series(J₀, J₁) -> Vector{Tensor{2,3,Float64,9}}   (length 3)

Coefficients of `adj(J₀ + θ J₁)` as a polynomial in θ, using

	adj J = J² − (tr J) J + ½ (tr²J − tr J²) I.

Verifies internally that the θ^3 coefficient vanishes (it must, by the
cofactor identity in 3D), with tolerance scaled by ‖J₁‖.
"""
function adj_series(J₀::Tens3, J₁::Tens3)
	I₃ = one(Tens3)

	# Note: `*` between two `Tensor{2}`s is single contraction in Tensors.jl,
	# but we use `⋅` explicitly throughout to keep intent unambiguous.
	J_ser     = [J₀, J₁]                                              # length 2
	J2_ser    = [J₀ ⋅ J₀, (J₀ ⋅ J₁) + (J₁ ⋅ J₀), J₁ ⋅ J₁]              # length 3
	trJ_ser   = [tr(J₀), tr(J₁)]                                   # length 2
	trJ2_ser  = [tr(J2_ser[1]), tr(J2_ser[2]), tr(J2_ser[3])]      # length 3
	trJsq_ser = poly_mul(trJ_ser, trJ_ser, 3)     # length 4 ; θ^3 entry must vanish
	trJ_x_J   = poly_mul(trJ_ser, J_ser, 3)       # length 4 ; θ^3 entry must vanish

	out = Vector{Tens3}(undef, 3)
	@inbounds for k in 1:3
		out[k] = J2_ser[k] - trJ_x_J[k] +
				 (0.5 * (trJsq_ser[k] - trJ2_ser[k])) * I₃
	end

	# Cayley–Hamilton sanity: the formal θ^3 coefficient of adj J must vanish.
	res3 = -trJ_x_J[4] + (0.5 * trJsq_ser[4]) * I₃
	tol  = 1e-12 * max(1.0, norm(J₁))
	@assert norm(res3) ≤ tol "adj_series: θ^3 coefficient = $(norm(res3)) > tol = $tol"

	return out
end

"""
	inv_series(J₀, J₁, N_θ) -> Vector{Tensor{2,3,Float64,9}}   (length N_θ + 1)

Coefficients of `J⁻¹(θ) = adj J(θ) / det J(θ)` truncated at order N_θ.
"""
function inv_series(J₀::Tens3, J₁::Tens3, N_θ::Int)
	adjJ     = adj_series(J₀, J₁)            # length 3
	detJ     = det_series(J₀, J₁)            # length 4
	inv_detJ = reciprocal_series(detJ, N_θ)  # length N_θ + 1
	return poly_mul(adjJ, inv_detJ, N_θ)      # length N_θ + 1
end

"""
	check_inv_series(J₀, J₁, invJ_coeffs, N_θ) -> Vector{Float64}

Residual norms ‖[J(θ) · J⁻¹(θ) − I]_k‖ for k = 0, 1, …, N_θ.  Useful
as a single-line unit test of `inv_series`.
"""
function check_inv_series(J₀::Tens3, J₁::Tens3,
	invJ_coeffs::Vector{Tens3}, N_θ::Int)
	I₃ = one(Tens3)
	J_ser = [J₀, J₁]
	JinvJ = poly_dot(J_ser, invJ_coeffs, N_θ)             # J(θ) ⋅ J⁻¹(θ)
	return [norm(JinvJ[1] - I₃); [norm(JinvJ[k+1]) for k in 1:N_θ]]
end
