"""
	theta_polynomials.jl

Generic *truncated* power-series in a scalar parameter θ.

A series

	p(θ) = Σ_{k=0}^{N} p_k θ^k                          (N = truncation order)

is stored as a `Vector` of coefficients `p[k+1] = p_k`.  The element
type is whatever Julia (and, where relevant, Tensors.jl) supports under
`*`, `+`, `⊡`, `⋅`, etc.  Typical types in this demo:

	Float64                         — scalars (det J, traces, 1/det J)
	Tensor{2, 3, Float64, 9}        — geometry coefficients (J₀, J₁, adj J, J⁻¹)
									   and Ferrite shape-function gradients ∇N
	SymmetricTensor{2, 3, Float64}  — strains ε, stresses σ

The user-facing entry points are

	truncated(c, N)                 — pad / clip a coefficient vector to length N+1
	poly_add!(c_dst, c_src)         — coefficient-wise in-place addition
	poly_mul(a, b, N)               — truncated convolution using `*`
	poly_contract(a, b, N)          — truncated convolution using `⊡`
	reciprocal_series(p, N)         — coefficients of 1/p(θ) via recurrence

The reciprocal recurrence (`reciprocal_series`) is the engine that
turns the rational quantity 1/det J(θ) into a polynomial in θ; see the
function docstring for the derivation.
"""

# We intentionally avoid `module …` so this file can simply be
# `include`d from the demo without import gymnastics.

const ZERO_TOL = 1e-15

# ---------------------------------------------------------------------------
# Padding / truncation
# ---------------------------------------------------------------------------
"""
	truncated(c, N) -> Vector{eltype(c)}

Return a vector of length `N + 1`:
- copies the first `N + 1` coefficients of `c` (clipping if longer),
- pads with `zero(eltype(c))` if `c` is shorter.
"""
function truncated(c::AbstractVector{T}, N::Int) where {T}
	out = Vector{T}(undef, N + 1)
	@inbounds for k in 0:N
		out[k+1] = k + 1 ≤ length(c) ? c[k+1] : zero(T)
	end
	return out
end

# ---------------------------------------------------------------------------
# In-place addition
# ---------------------------------------------------------------------------
"""
	poly_add!(dst, src) -> dst

In-place `dst += src` over the common length `min(length(dst), length(src))`.
"""
function poly_add!(dst::AbstractVector, src::AbstractVector)
	@inbounds for k in 1:min(length(dst), length(src))
		dst[k] = dst[k] + src[k]
	end
	return dst
end

# ---------------------------------------------------------------------------
# Truncated convolution — internal helper
# ---------------------------------------------------------------------------
#
# All three polynomial-product variants below share the same loop;
# only the bilinear operation differs:
#
#     poly_mul       → uses  `*`   (scalar × scalar, scalar × tensor)
#     poly_dot       → uses  `⋅`   (single contraction of tensors)
#     poly_contract  → uses  `⊡`   (double contraction of tensors)
#
# Tensors.jl deliberately disallows `*` between two `Tensor{2}`s so the
# user is forced to choose between single and double contraction.  Hence
# the three named entry points.
#
# `op` is a concrete callable: Julia will specialise `_poly_convolve` per
# call site, so passing the operator carries no runtime overhead.

@inline function _poly_convolve(op, a::AbstractVector, b::AbstractVector, N::Int)
	R = typeof(op(a[1], b[1]))
	out = Vector{R}(undef, N + 1)
	@inbounds for k in 0:N
		s = zero(R)
		jmin = max(0, k - (length(b) - 1))
		jmax = min(k, length(a) - 1)
		for j in jmin:jmax
			s = s + op(a[j+1], b[k-j+1])
		end
		out[k+1] = s
	end
	return out
end

# ---------------------------------------------------------------------------
# Truncated multiplication (uses `*`)
# ---------------------------------------------------------------------------
"""
	poly_mul(a, b, N) -> Vector

Truncated product `(a * b) mod θ^{N+1}` using the scalar `*` operator.
Use when at least one operand is a scalar series (the other may be a
tensor series — `Float64 * Tensor` is the usual scalar multiplication).
For tensor-tensor convolutions, use `poly_dot` (single contraction) or
`poly_contract` (double contraction).
"""
poly_mul(a::AbstractVector, b::AbstractVector, N::Int) = _poly_convolve(*, a, b, N)

# ---------------------------------------------------------------------------
# Truncated single contraction (uses `⋅`)
# ---------------------------------------------------------------------------
"""
	poly_dot(a, b, N) -> Vector

Truncated convolution using the single-contraction operator `⋅`.  Use
for products of two tensor series (e.g. `J(θ) ⋅ J⁻¹(θ)` or
`(∇uᵀ)(θ) ⋅ ∇v(θ)`), where each term `a_i ⋅ b_j` is the Tensors.jl
single contraction.
"""
poly_dot(a::AbstractVector, b::AbstractVector, N::Int) = _poly_convolve(⋅, a, b, N)

# ---------------------------------------------------------------------------
# Truncated double-contraction (uses `⊡`)
# ---------------------------------------------------------------------------
"""
	poly_contract(a, b, N) -> Vector

Truncated convolution using the double-contraction operator `⊡`.  Used
for `ε ⊡ σ`-type accumulations where the elementwise product of two
symmetric second-order tensors yields a scalar.
"""
poly_contract(a::AbstractVector, b::AbstractVector, N::Int) = _poly_convolve(⊡, a, b, N)

# ---------------------------------------------------------------------------
# Reciprocal of a scalar series
# ---------------------------------------------------------------------------
"""
	reciprocal_series(p, N) -> Vector{Float64}

Coefficients of `q(θ) = 1 / p(θ)` truncated at order `N`, i.e.

	p(θ) · q(θ) ≡ 1   (mod θ^{N+1}).

Derivation.  Matching the coefficient of θ^k on both sides of
p · q ≡ 1 (k ≥ 0) gives

	k = 0 :   p₀ q₀ = 1                 ⇒  q₀ = 1 / p₀
	k ≥ 1 :   Σ_{j=0}^{k} p_j q_{k-j} = 0
			  ⇒  p₀ q_k = − Σ_{j=1}^{k} p_j q_{k-j}
			  ⇒  q_k = −(1/p₀) · Σ_{j=1}^{k} p_j q_{k-j}.

Cost: O(N²) multiplies.  Throws if `p₀ ≈ 0`.
"""
function reciprocal_series(p::AbstractVector{<:Real}, N::Int)
	abs(p[1]) > ZERO_TOL || error("reciprocal_series: p(0) = 0, series undefined")
	inv_p0 = 1.0 / p[1]
	q = Vector{Float64}(undef, N + 1)
	q[1] = inv_p0
	@inbounds for k in 1:N
		s = 0.0
		jmax = min(k, length(p) - 1)
		for j in 1:jmax
			s += p[j+1] * q[k-j+1]
		end
		q[k+1] = -inv_p0 * s
	end
	return q
end
