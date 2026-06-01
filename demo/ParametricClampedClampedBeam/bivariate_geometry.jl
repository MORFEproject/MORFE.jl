"""
    bivariate_geometry.jl

Per-quadrature-point computation of the bivariate polynomial series

    det J(θ₁, θ₂, x₀)   and   adj J(θ₁, θ₂, x₀)

for the two-parameter reference map

    J(θ₁, θ₂, x₀) = J₀ + θ₁ J₁ + θ₂ J₂(x₀)

where J₁ is the constant axial-stretch Jacobian and J₂(x₀) = ∇₀φ₁(x₀)
is the gradient of the arch (bending) eigenmode — it varies per QP.

Since each entry J_{ij}(θ₁,θ₂) is linear, the adjugate is exactly degree ≤ 2
and the determinant is exactly degree ≤ 3.  All non-zero coefficients are
computed directly from closed-form trace/matrix-product identities (see
parametric_geometry.jl for the univariate analogues), without forming any
intermediate bivariate polynomial arrays and without computing inv(J₀).

Requires `bivariate_polynomials.jl` (and transitively `theta_polynomials.jl`)
to be included first.
"""

using Tensors
using LinearAlgebra

# Tens3 is defined in parametric_geometry.jl (included first).

# -----------------------------------------------------------------------
# det and adj bivariate series
# -----------------------------------------------------------------------
"""
    det_and_adj_bseries(J₀, J₁, J₂, N) -> (det_b, adj_b)

Compute bivariate polynomial coefficients of `det(J₀ + θ₁ J₁ + θ₂ J₂)`
and `adj(J₀ + θ₁ J₁ + θ₂ J₂)` for a 3×3 system, truncated at total
degree `N` in (θ₁, θ₂).

Returns
-------
- `det_b :: Matrix{Float64}` of size `(N+1)×(N+1)`:
        `det_b[k₁+1, k₂+1]` = coefficient of θ₁^k₁ θ₂^k₂
- `adj_b :: Matrix{Tens3}` of size `(N+1)×(N+1)`:
        `adj_b[k₁+1, k₂+1]` = coefficient of θ₁^k₁ θ₂^k₂  (a 3×3 tensor)

Algorithm
---------
Uses closed-form trace/matrix-product identities (extended A+xB formulas).
For the determinant (exact degree ≤ 3) only 10 entries are non-zero; for
the adjugate (exact degree ≤ 2) only 6 are non-zero.  All coefficients are
computed directly from shared scalar and matrix intermediates — no bpoly_mul,
no intermediate bivariate arrays, no inv(J₀).

The mixed θ₁θ₂ coefficient of det uses the identity
    d[2,2] = tr(A)tr(B)tr(C) - tr(AC)tr(B) - tr(C)tr(AB) - tr(A)tr(CB)
             + tr(ACB) + tr(ABC)   (A=J₀, B=J₁, C=J₂)
derived from ∂²det/(∂θ₁∂θ₂)|₀ via Jacobi's formula.  The mixed θ₁²θ₂
coefficient uses d[3,2] = tr(adj(J₁)·J₂), and d[2,3] = tr(adj(J₂)·J₁).
The adj θ₁θ₂ coefficient is assembled entry-by-entry from 2×2 minor
cross-terms (independent of J₀ since total minor degree = 2 = 1+1).
"""
function det_and_adj_bseries(J₀::Tens3, J₁::Tens3, J₂::Tens3, N::Int)
    A = J₀;  B = J₁;  C = J₂
    I₃ = one(Tens3)

    # ---- Shared scalar quantities ----
    tA = tr(A);  tB = tr(B);  tC = tr(C)

    # ---- Shared matrix products ----
    A2 = A ⋅ A;  B2 = B ⋅ B;  C2 = C ⋅ C
    AB = A ⋅ B;  BA = B ⋅ A
    AC = A ⋅ C;  CA = C ⋅ A
    BC = B ⋅ C

    trAB = tr(AB);  trAC = tr(AC);  trBC = tr(BC)

    trA2B = tr(A2 ⋅ B);  trA2C = tr(A2 ⋅ C)
    trB2A = tr(B2 ⋅ A);  trC2A = tr(C2 ⋅ A)
    trB2C = tr(B2 ⋅ C);  trC2B = tr(C2 ⋅ B)

    # term_X := X² − X·tr(X),   tr_term_X := ½ tr(term_X)  [= ½(tr(X²)−tr(X)²)]
    term_A = A2 - A * tA;  tr_term_A = 0.5 * tr(term_A)
    term_B = B2 - B * tB;  tr_term_B = 0.5 * tr(term_B)
    term_C = C2 - C * tC;  tr_term_C = 0.5 * tr(term_C)

    # triple traces for mixed θ₁θ₂ det coefficient
    trABC = tr(AB ⋅ C)   # tr(A·B·C)
    trACB = tr(AC ⋅ B)   # tr(A·C·B) — differs from trABC when B,C don't commute

    # ---- Determinant ----
    det_b = zeros(Float64, N + 1, N + 1)

    det_b[1, 1] = det(A)

    N ≥ 1 && (det_b[2, 1] = trA2B - tr_term_A * tB - tA * trAB)   # θ₁
    N ≥ 1 && (det_b[1, 2] = trA2C - tr_term_A * tC - tA * trAC)   # θ₂

    N ≥ 2 && (det_b[3, 1] = trB2A - tr_term_B * tA - tB * trAB)   # θ₁²
    N ≥ 2 && (det_b[1, 3] = trC2A - tr_term_C * tA - tC * trAC)   # θ₂²
    # mixed θ₁θ₂: tr(CB)=tr(BC) by cyclic, so trCB→trBC; formula from ∂²det/(∂θ₁∂θ₂)|₀
    N ≥ 2 && (det_b[2, 2] = tA * tB * tC - trAC * tB - tC * trAB -
                             tA * trBC + trACB + trABC)

    N ≥ 3 && (det_b[4, 1] = det(B))                                # θ₁³
    N ≥ 3 && (det_b[1, 4] = det(C))                                # θ₂³
    # θ₁²θ₂: tr(adj(B)·C) = tr((term_B − tr_term_B·I)·C)
    N ≥ 3 && (det_b[3, 2] = trB2C - tB * trBC - tr_term_B * tC)
    # θ₁θ₂²: tr(adj(C)·B) — by symmetry swap B↔C
    N ≥ 3 && (det_b[2, 3] = trC2B - tC * trBC - tr_term_C * tB)

    # ---- Adjugate ----
    adj_b = fill(zero(Tens3), N + 1, N + 1)

    adj_b[1, 1] = term_A - tr_term_A * I₃                          # adj(A)

    if N ≥ 1
        adj_b[2, 1] = (tA * tB - trAB) * I₃ - (A * tB + B * tA) + AB + BA  # C₁(A,B)
        adj_b[1, 2] = (tA * tC - trAC) * I₃ - (A * tC + C * tA) + AC + CA  # C₁(A,C)
    end

    if N ≥ 2
        adj_b[3, 1] = term_B - tr_term_B * I₃                      # adj(B)
        adj_b[1, 3] = term_C - tr_term_C * I₃                      # adj(C)

        # θ₁θ₂ coefficient: assembled entry-by-entry from 2×2 minor cross-terms.
        # For adj(J)[i,j] = (−1)^{i+j} · minor(rows≠j, cols≠i), the θ₁θ₂
        # coefficient of the 2×2 minor with rows {r1,r2} and cols {c1,c2} is:
        #   B[r1,c1]C[r2,c2] + C[r1,c1]B[r2,c2] − B[r1,c2]C[r2,c1] − C[r1,c2]B[r2,c1]
        # (independent of J₀: total minor degree 2 = 1+1 uses one factor from B, one from C)
        rows_excl = ((2, 3), (1, 3), (1, 2))
        cols_excl = ((2, 3), (1, 3), (1, 2))
        cof_12 = ntuple(q -> begin
            i = (q - 1) % 3 + 1
            j = (q - 1) ÷ 3 + 1
            r1, r2 = rows_excl[j]
            c1, c2 = cols_excl[i]
            s = iseven(i + j) ? 1.0 : -1.0
            s * (B[r1, c1] * C[r2, c2] + C[r1, c1] * B[r2, c2] -
                 B[r1, c2] * C[r2, c1] - C[r1, c2] * B[r2, c1])
        end, 9)
        adj_b[2, 2] = Tens3(cof_12)
    end

    return det_b, adj_b
end

# -----------------------------------------------------------------------
# Sanity check: J(θ₁,θ₂) · adj(J(θ₁,θ₂)) = det(J(θ₁,θ₂)) · I
# -----------------------------------------------------------------------
"""
    check_adj_det_bidentity(J₀, J₁, J₂, adj_b, det_b) -> Matrix{Float64}

Return a `(N+1)×(N+1)` matrix of Frobenius residuals
`‖[J(θ₁,θ₂) · adj(J(θ₁,θ₂)) − det(J(θ₁,θ₂)) · I]_{k₁,k₂}‖`
for all (k₁,k₂) with k₁+k₂ ≤ N.  All entries should be ≤ 1e-12.
"""
function check_adj_det_bidentity(J₀::Tens3, J₁::Tens3, J₂::Tens3,
    adj_b::AbstractMatrix{Tens3},
    det_b::AbstractMatrix{Float64})
    N = size(det_b, 1) - 1
    I₃ = one(Tens3)

    # Build bivariate series for J itself: degree-1
    J_b = Matrix{Tens3}(undef, N + 1, N + 1)
    fill!(J_b, zero(Tens3))
    J_b[1, 1] = J₀
    N ≥ 1 && (J_b[2, 1] = J₁)
    N ≥ 1 && (J_b[1, 2] = J₂)

    # J · adj(J) bivariate polynomial (using bpoly_dot for matrix products)
    Jadj = bpoly_dot(J_b, adj_b, N)

    # Residual norms per (k₁,k₂)
    resid = zeros(Float64, N + 1, N + 1)
    for k1 in 0:N, k2 in 0:N-k1
        target = det_b[k1+1, k2+1] * I₃
        resid[k1+1, k2+1] = norm(Jadj[k1+1, k2+1] - target)
    end
    return resid
end
