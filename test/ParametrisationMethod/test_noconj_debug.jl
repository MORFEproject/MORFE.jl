"""Regression test for conjugate and no-conjugate solve equivalence."""

module NoConjugateRegression

using LinearAlgebra
using StaticArrays
using Test
using MORFE

# ─────────────────────────────────────────────────────────────────────────────
# System parameters
# ─────────────────────────────────────────────────────────────────────────────
const _FOM = 4
const _ROM = 2
const _N_EXT = 2
const _NVAR = _ROM + _N_EXT
const _max_degree = 3
const _ζ = 0.01

# Natural frequencies; first one is the master mode (conjugate pair λ₁, conj(λ₁))
const _ω_nat = [1.0, 3.2, 5.3, 7.1]

# ─────────────────────────────────────────────────────────────────────────────
# Linear matrices  (second-order system: M ẍ + C ẋ + K x = f, M = I)
# ─────────────────────────────────────────────────────────────────────────────
const _K = Matrix{Float64}(Diagonal(_ω_nat .^ 2))
const _C = Matrix{Float64}(Diagonal(2_ζ .* _ω_nat))
const _M = Matrix{Float64}(I(_FOM))

# ─────────────────────────────────────────────────────────────────────────────
# Eigenpairs (analytical; diagonal → mode shapes are standard basis vectors)
#   λ² + 2ζω λ + ω² = 0   →   λ = -ζω ± iω√(1-ζ²)
# ─────────────────────────────────────────────────────────────────────────────
const _λ₁ = complex(-_ζ * _ω_nat[1], _ω_nat[1] * sqrt(1 - _ζ^2))
const _λ₂ = conj(_λ₁)
const _master_eigenvalues = SVector{_ROM, ComplexF64}(_λ₁, _λ₂)

# Position component of right eigenvectors (DOF-1 mode shape = e₁, real)
const _master_modes = let
    m = zeros(ComplexF64, _FOM, _ROM)
    m[1, 1] = 1.0   # mode 1 position component
    m[1, 2] = 1.0   # mode 2 position component (conj = same for real mode shape)
    m
end

# Velocity component: λ × position for uncoupled diagonal system
const _master_modes_derivatives = let
    d = zeros(ComplexF64, _FOM, 1, _ROM)
    d[1, 1, 1] = _λ₁
    d[1, 1, 2] = _λ₂
    d
end

# For a symmetric real system, left eigenmodes = right eigenmodes
const _left_eigenmodes = _master_modes

# Lower-order left eigenvector blocks from the physical slice
const _left_modes_derivatives = MORFE.SpectralDecomposition.left_eigenmode_orders_from_slice(
    (_K, _C, _M), _left_eigenmodes, [_λ₁, _λ₂])[:, 1:1, :]

# The bundle the solve consumes: physical slices plus their companion blocks.
const _spectral = SpectralData(; eigenvalues = _master_eigenvalues,
    right_modes = _master_modes, right_derivatives = _master_modes_derivatives,
    left_modes = _left_eigenmodes, left_blocks = _left_modes_derivatives)

# ─────────────────────────────────────────────────────────────────────────────
# Multiindex set (shared; all monomials of total degree 1…max_degree in NVAR vars)
# ─────────────────────────────────────────────────────────────────────────────
const _mset = all_multiindices_up_to(_NVAR, _max_degree; min_degree = 1)

# External forcing frequency = imag(λ₁)  → exact resonance with e₃
const _Ω_force = imag(_λ₁)

# ─────────────────────────────────────────────────────────────────────────────
# Build model for a given f_vec
# ─────────────────────────────────────────────────────────────────────────────
function _make_model(f_vec::AbstractVector{ComplexF64})
    f_vec_r = real(f_vec)   # physical forcing is a real spatial vector
    term_quad = MultilinearMap((res, x, y) -> (res .+= x .* y), (2, 0))
    term_forcing = MultilinearMap((res, r) -> (res .+= f_vec_r * sum(r)), (0, 0), 1)
    ext_sys = ExternalSystem((im * _Ω_force, -im * _Ω_force))
    return NthOrderModel((_K, _C, _M), (term_quad, term_forcing), ext_sys)
end

function _make_resonance()
    ext_eigs = ComplexF64[im * _Ω_force, -im * _Ω_force]
    return resonance_set_from_complex_normal_form_style(
        _mset, Vector{ComplexF64}(_master_eigenvalues), 0.1;
        external_eigenvalues = ext_eigs)
end

# ─────────────────────────────────────────────────────────────────────────────
# Run and compare both solve paths, printing every diverging monomial
# ─────────────────────────────────────────────────────────────────────────────
function compare_paths(f_vec::AbstractVector{ComplexF64})
    model = _make_model(f_vec)
    res_set = _make_resonance()

    W_conj, R_conj = solve_cohomological_problem(
        model, _mset, _spectral, res_set;
        conjugate_permutation = [2, 1, 4, 3],
        options = ParametrisationOptions(show_progress = false)
    )
    W_noconj, R_noconj = solve_cohomological_problem(
        model, _mset, _spectral, res_set;
        conjugate_permutation = nothing,
        options = ParametrisationOptions(show_progress = false)
    )

    Wc1 = W_conj.poly.coefficients    # FOM × ORD × L
    Wc2 = W_noconj.poly.coefficients
    Rc1 = R_conj.poly.coefficients    # NVAR × L
    Rc2 = R_noconj.poly.coefficients

    n_diverge = 0

    for (l, α) in enumerate(_mset.exponents)
        dW = norm(Wc1[:, :, l] - Wc2[:, :, l])
        dR = norm(Rc1[:, l] - Rc2[:, l])
        (dW > 1e-10 || dR > 1e-10) || continue

        n_diverge += 1
    end
    return (; W_conj, R_conj, W_noconj, R_noconj, n_diverge)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
@testset "conjugate/no-conjugate equivalence" begin
    for force in (ComplexF64[1.0, 0.0, 0.0, 0.0],
        ComplexF64[1.0 + 0.3im, 0.0, 0.0, 0.0])
        result = compare_paths(force)
        @test result.n_diverge == 0
        @test result.W_conj.poly.coefficients ≈
              result.W_noconj.poly.coefficients atol=1e-10
        @test result.R_conj.poly.coefficients ≈
              result.R_noconj.poly.coefficients atol=1e-10
    end
end

end # module NoConjugateRegression
