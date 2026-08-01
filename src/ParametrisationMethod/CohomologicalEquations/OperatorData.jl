# =============================================================================
# Precomputed operator coefficient bundles
# =============================================================================

"""
	InvarianceOperators{T}

Precomputed column-polynomial coefficients for the invariance equation.

Both fields hold coefficients of polynomials in the superharmonic `s`, evaluated by
[`evaluate_column!`](@ref) once per monomial: the master columns become the `C(s)`
border of the bordered system, the external ones contribute to its right-hand side.
Precomputing them per order rather than per monomial is what keeps the inner solve
loop allocation-free.

# Fields

- `column_coeffs::Vector{Matrix{T}}` — one `FOM × ORD` matrix per master mode `r`,
  the coefficients of the border *column* `C_r(s)`.  Distinct in both shape and
  role from [`OrthogonalityOperators`](@ref)`.corner_coeffs`, which fills the
  `ROM × ROM` corner.
- `E_coeffs::Vector{Matrix{T}}` — one `FOM × ORD` matrix per external forcing mode
  `e`.  External amplitudes are known, so these never reach the matrix.
"""
struct InvarianceOperators{T}
    column_coeffs::Vector{Matrix{T}}   # length ROM,   each FOM × ORD
    E_coeffs::Vector{Matrix{T}}        # length N_EXT, each FOM × ORD
end

"""
	OrthogonalityOperators{T}

Precomputed row and column-polynomial coefficients for the orthogonality conditions.

Together the three fields supply the bottom `ROM` rows of the bordered system: one
row per master mode, assembled by [`assemble_orthogonality_matrix_and_rhs!`](@ref).
They are read off the left eigenvector order-blocks once per order, so evaluating a
row at a given `s` costs one Horner pass and no allocation.

# Fields

- `J_coeffs::Vector{Matrix{T}}` — one `ORD × FOM` matrix per master mode `r`, the
  coefficients of the row operator `Ĵ_r(s)` acting on `W[α]`.
- `corner_coeffs::Vector{Matrix{T}}` — one `(ORD-1) × ROM` matrix per master mode,
  evaluating to the `ROM × ROM` *corner* block `Ĉ(s)` that couples the orthogonality
  rows to the unknown reduced-dynamics coefficients.  See
  [`InvarianceOperators`](@ref)`.column_coeffs` for the differently-shaped border
  columns.
- `E_coeffs::Vector{Matrix{T}}` — one `(ORD-1) × N_EXT` matrix per master mode,
  contracting the known external amplitudes into the scalar right-hand side.
"""
struct OrthogonalityOperators{T}
    J_coeffs::Vector{Matrix{T}}        # length ROM, each ORD × FOM
    corner_coeffs::Vector{Matrix{T}}   # length ROM, each (ORD-1) × ROM
    E_coeffs::Vector{Matrix{T}}        # length ROM, each (ORD-1) × N_EXT
end
