# =============================================================================
# Precomputed operator coefficient bundles
# =============================================================================

"""
    InvarianceOperators{T}

Precomputed column-polynomial coefficients for the invariance equation.
`C_coeffs[r]` is `FOM × ORD`; `E_coeffs[e]` is `FOM × ORD`.
"""
struct InvarianceOperators{T}
    C_coeffs::Vector{Matrix{T}}   # length ROM,   each FOM × ORD
    E_coeffs::Vector{Matrix{T}}   # length N_EXT, each FOM × ORD
end

"""
    OrthogonalityOperators{T}

Precomputed row and column-polynomial coefficients for the orthogonality conditions.
`J_coeffs[r]` is `ORD × FOM`; `C_coeffs[r]` is `(ORD-1) × ROM`; `E_coeffs[r]` is
`(ORD-1) × N_EXT`.
"""
struct OrthogonalityOperators{T}
    J_coeffs::Vector{Matrix{T}}   # length ROM, each ORD × FOM
    C_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × ROM
    E_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × N_EXT
end
