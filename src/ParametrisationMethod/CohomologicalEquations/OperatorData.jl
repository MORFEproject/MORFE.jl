# =============================================================================
# Precomputed operator coefficient bundles
# =============================================================================

"""
	InvarianceOperators{T}

Precomputed column-polynomial coefficients for the invariance equation.

`column_coeffs[r]` is the `FOM × ORD` coefficient matrix of the border *column*
`C_r(s)`; `E_coeffs[e]` is `FOM × ORD`.  Note the deliberate contrast with
[`OrthogonalityOperators`](@ref)`.corner_coeffs`, which is a different shape and a
different object — the two were both called `C_coeffs` and were easy to confuse.
"""
struct InvarianceOperators{T}
    column_coeffs::Vector{Matrix{T}}   # length ROM,   each FOM × ORD
    E_coeffs::Vector{Matrix{T}}   # length N_EXT, each FOM × ORD
end

"""
	OrthogonalityOperators{T}

Precomputed row and column-polynomial coefficients for the orthogonality conditions.

`J_coeffs[r]` is the `ORD × FOM` row operator `Ĵ_r(s)`; `corner_coeffs[r]` is the
`(ORD-1) × ROM` block landing in the bordered matrix's `ROM × ROM` *corner*;
`E_coeffs[r]` is `(ORD-1) × N_EXT`.  See [`InvarianceOperators`](@ref)`.column_coeffs`
for the differently-shaped border columns.
"""
struct OrthogonalityOperators{T}
    J_coeffs::Vector{Matrix{T}}        # length ROM, each ORD × FOM
    corner_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × ROM
    E_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × N_EXT
end
