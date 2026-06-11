"""
    bivariate_polynomials.jl

Truncated power series in two scalar parameters (θ₁, θ₂):

    p(θ₁, θ₂) = Σ_{k₁,k₂} p_{k₁,k₂} θ₁^k₁ θ₂^k₂

stored as a `(N+1)×(N+1)` matrix where `A[k₁+1, k₂+1] = p_{k₁,k₂}`.
Only entries with `k₁ + k₂ ≤ N` are meaningful; the rest are zero.

The element type is whatever Julia supports under `*`, `⋅`, `⊡`:
    Float64                        — scalars (det J, 1/det J, …)
    Tensor{2, 3, Float64, 9}       — geometry (adj J, …)
    SymmetricTensor{2, 3, Float64} — strains, stresses

Entry points:
    bpoly_mul(a, b, N)      — truncated bivariate product using `*`
    bpoly_dot(a, b, N)      — truncated bivariate product using `⋅`
    bpoly_contract(a, b, N) — truncated bivariate product using `⊡`
    breciprocal_series(p, N)— coefficients of 1/p(θ₁,θ₂)
    bpoly_power(p, n, N)    — n-th power of a scalar bivariate series
"""

# -----------------------------------------------------------------------
# Internal: bivariate truncated convolution
# -----------------------------------------------------------------------
@inline function _bpoly_convolve(op, a::AbstractMatrix, b::AbstractMatrix, N::Int)
    R = typeof(op(a[1, 1], b[1, 1]))
    out = Matrix{R}(undef, N + 1, N + 1)
    fill!(out, zero(R))
    for k1 in 0:N, k2 in 0:N-k1
        s = zero(R)
        for j1 in 0:k1
            jmin2 = max(0, k2 - (size(b, 2) - 1 - (k1 - j1)))
            jmax2 = min(k2, size(b, 2) - 1)
            for j2 in jmin2:jmax2
                # a contributes j1, j2; b contributes k1-j1, k2-j2
                aj1j2_in_range = (j1 + 1 ≤ size(a, 1)) && (j2 + 1 ≤ size(a, 2))
                bk_in_range    = ((k1 - j1) + 1 ≤ size(b, 1)) && ((k2 - j2) + 1 ≤ size(b, 2))
                aj1j2_in_range && bk_in_range || continue
                s = s + op(a[j1+1, j2+1], b[k1-j1+1, k2-j2+1])
            end
        end
        out[k1+1, k2+1] = s
    end
    return out
end

"""
    bpoly_mul(a, b, N) -> Matrix

Truncated bivariate product `(a * b) mod (θ₁^N θ₂^N)` keeping only
total degree `k₁+k₂ ≤ N`. Uses scalar `*` (and `Float64 * Tensor`).
"""
bpoly_mul(a::AbstractMatrix, b::AbstractMatrix, N::Int) =
    _bpoly_convolve(*, a, b, N)

"""
    bpoly_dot(a, b, N) -> Matrix

Truncated bivariate convolution using single contraction `⋅`.
"""
bpoly_dot(a::AbstractMatrix, b::AbstractMatrix, N::Int) =
    _bpoly_convolve(⋅, a, b, N)

"""
    bpoly_contract(a, b, N) -> Matrix

Truncated bivariate convolution using double contraction `⊡`.
"""
bpoly_contract(a::AbstractMatrix, b::AbstractMatrix, N::Int) =
    _bpoly_convolve(⊡, a, b, N)

# -----------------------------------------------------------------------
# Reciprocal of a bivariate scalar series
# -----------------------------------------------------------------------
"""
    breciprocal_series(p, N) -> Matrix{Float64}

Coefficients of `q(θ₁,θ₂) = 1/p(θ₁,θ₂)` truncated so that `k₁+k₂ ≤ N`.

The recurrence follows from `p·q ≡ 1`:

    p₀₀ q_{k₁,k₂} = δ_{k₁,0}δ_{k₂,0}
                     - Σ_{(j₁,j₂)≠(0,0), j₁≤k₁, j₂≤k₂} p_{j₁,j₂} q_{k₁-j₁,k₂-j₂}

Computed in order of increasing total degree so lower-degree `q` values
are always available.
"""
function breciprocal_series(p::AbstractMatrix{<:Real}, N::Int)
    abs(p[1, 1]) > ZERO_TOL ||
        error("breciprocal_series: p(0,0) = 0, series undefined")
    inv_p00 = 1.0 / p[1, 1]
    q = zeros(Float64, N + 1, N + 1)
    q[1, 1] = inv_p00
    # iterate by total degree d = k₁+k₂, then by k₁
    for d in 1:N
        for k1 in 0:d
            k2 = d - k1
            s = 0.0
            for j1 in 0:k1, j2 in 0:k2
                (j1 == 0 && j2 == 0) && continue
                j1 + 1 ≤ size(p, 1) && j2 + 1 ≤ size(p, 2) || continue
                s += p[j1+1, j2+1] * q[k1-j1+1, k2-j2+1]
            end
            q[k1+1, k2+1] = -inv_p00 * s
        end
    end
    return q
end

# -----------------------------------------------------------------------
# Integer power of a bivariate scalar series
# -----------------------------------------------------------------------
"""
    bpoly_power(p, n, N) -> Matrix{Float64}

Compute `p(θ₁,θ₂)^n` truncated at total degree `N` via repeated `bpoly_mul`.
`n = 0` returns the constant series 1; `n` must be non-negative.
"""
function bpoly_power(p::AbstractMatrix{Float64}, n::Int, N::Int)
    @assert n ≥ 0 "n must be ≥ 0"
    out = zeros(Float64, N + 1, N + 1)
    out[1, 1] = 1.0          # constant series = 1
    for _ in 1:n
        out = bpoly_mul(out, p, N)
    end
    return out
end

# -----------------------------------------------------------------------
# Utility: extract a (k₁, k₂) coefficient as a univariate slice
# -----------------------------------------------------------------------
"""
    bpoly_to_vec(A, N) -> Vector

Flatten a bivariate polynomial matrix `A` to a length-`(N+1)(N+2)/2` vector
indexed by `(k₁+k₂, k₁)` in graded-lex order.  Useful for debug printing.
"""
function bpoly_to_vec(A::AbstractMatrix{T}, N::Int) where {T}
    out = T[]
    for d in 0:N, k1 in 0:d
        push!(out, A[k1+1, d-k1+1])
    end
    return out
end
