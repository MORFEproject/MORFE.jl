"""
Coupled (upper-triangular, non-diagonal) external systems.

`ExternalSystem` requires an upper-triangular linear matrix because the solve is causal in
GrLex order.  Off-diagonal entries above the diagonal are legal, and this file checks the
solver actually honours them: the external forcing directions are pre-solved before `Φ_ext`
is known, so a coupled external block only comes out right if each direction feeds the
already-solved ones back into the partial context.

The reference is a *change of external coordinates*, not a recomputation.  For
`A = [λ₊ c; 0 λ₋]` with `λ₊ ≠ λ₋`, the similarity `A = T · diag(λ₊, λ₋) · T⁻¹` holds with the
unit-diagonal upper-triangular `T = [1 a; 0 1]`, `a = c / (λ₋ - λ₊)`.  Substituting `r = T r̃`
turns the coupled model into a *diagonal* one whose forcing matrix is `F·T` and whose
full-order dynamics are identical.  Both parametrise the same manifold, so their external
directions must satisfy `Φ_diag = Φ_coupled · T` exactly.  The diagonal model takes the
uncoupled fast path, so this pins the coupled path against a known-good reference.
"""

using Test
using LinearAlgebra
using StaticArrays
using MORFE

# ─────────────────────────────────────────────────────────────────────────────
# A 4-DOF second-order system: M ẍ + C ẋ + K x = f, M = I, all diagonal.
# Master pair = mode 1; two external variables carry the forcing.
# ─────────────────────────────────────────────────────────────────────────────
const _EC_FOM = 4
const _EC_ROM = 2
const _EC_N_EXT = 2
const _EC_NVAR = _EC_ROM + _EC_N_EXT
const _EC_DEGREE = 3
const _EC_ζ = 0.01
const _EC_ω = [1.0, 3.0, 5.0, 7.0]

const _EC_K = Matrix{Float64}(Diagonal(_EC_ω .^ 2))
const _EC_C = Matrix{Float64}(Diagonal(2_EC_ζ .* _EC_ω))
const _EC_M = Matrix{Float64}(I(_EC_FOM))

const _EC_λ = complex(-_EC_ζ * _EC_ω[1], _EC_ω[1] * sqrt(1 - _EC_ζ^2))
const _EC_master_eigenvalues = SVector{_EC_ROM, ComplexF64}(_EC_λ, conj(_EC_λ))

const _EC_master_modes = let m = zeros(ComplexF64, _EC_FOM, _EC_ROM)
    m[1, 1] = 1.0
    m[1, 2] = 1.0
    m
end

const _EC_master_modes_derivatives = let d = zeros(ComplexF64, _EC_FOM, 1, _EC_ROM)
    d[1, 1, 1] = _EC_λ
    d[1, 1, 2] = conj(_EC_λ)
    d
end

const _EC_left_eigenmodes = _EC_master_modes
const _EC_left_modes_derivatives = MORFE.Eigenproblems.left_eigenmode_orders_from_slice(
    (_EC_K, _EC_C, _EC_M), _EC_left_eigenmodes, [_EC_λ, conj(_EC_λ)])[:, 1:1, :]

const _EC_mset = all_multiindices_up_to(_EC_NVAR, _EC_DEGREE; min_degree = 1)

# Forcing frequency deliberately away from ω₁ = 1 and from every combination
# resonance, so no external monomial is flagged resonant.  R[1:ROM, ext] then stays
# zero in both models and the external directions are fixed by L(s)Φ = f alone —
# no bordered ambiguity to muddy the comparison.
const _EC_Ω = 2.1
const _EC_λ_ext = (im * _EC_Ω, -im * _EC_Ω)

# Spatial forcing shapes, one column per external variable.  Distinct columns so that
# mixing them via T actually changes the answer.
const _EC_F = ComplexF64[1.0 0.0; 0.0 1.0; 0.5 -0.25; 0.0 0.75]

"""
Vector-valued external polynomial ṙ = A r for a 2×2 matrix `A`.
"""
function _ec_linear_external_polynomial(A::AbstractMatrix{ComplexF64})
    ms = all_multiindices_up_to(2, 1; min_degree = 1)
    coeffs = zeros(ComplexF64, 2, length(ms))
    for (j, α) in enumerate(ms.exponents)
        col = findfirst(==(1), α)
        col === nothing && continue
        coeffs[:, j] = A[:, col]
    end
    return DensePolynomial(coeffs, ms)
end

"""
Build the model whose external dynamics are `A` and whose forcing shape matrix is `F`.
The nonlinear term is a plain quadratic so the reduction is not trivial.
"""
function _ec_make_model(A::AbstractMatrix{ComplexF64}, F::AbstractMatrix{ComplexF64})
    F_real = real(F)
    term_quad = MultilinearMap((res, x, y) -> (res .+= x .* y), (2, 0))
    term_forcing = MultilinearMap((res, r) -> (res .+= F_real * r), (0, 0), 1)
    ext_sys = ExternalSystem(_ec_linear_external_polynomial(A))
    return NDOrderModel((_EC_K, _EC_C, _EC_M), (term_quad, term_forcing), ext_sys)
end

function _ec_solve(A::AbstractMatrix{ComplexF64}, F::AbstractMatrix{ComplexF64})
    model = _ec_make_model(A, F)
    res_set = resonance_set_from_complex_normal_form_style(
        _EC_mset, Vector{ComplexF64}(_EC_master_eigenvalues), 0.1;
        external_eigenvalues = ComplexF64[_EC_λ_ext...])
    return solve_cohomological_problem(
        model, _EC_mset, _EC_master_eigenvalues, _EC_master_modes,
        _EC_left_eigenmodes, res_set;
        master_modes_derivatives = _EC_master_modes_derivatives,
        left_modes_derivatives = _EC_left_modes_derivatives,
        show_progress = false
    )
end

"Return the FOM × ORD × N_EXT block of external linear-monomial coefficients of `W`."
function _ec_external_blocks(W)
    offset = findfirst(α -> sum(α) == 1, _EC_mset.exponents) - 1
    return W.poly.coefficients[:, :, (offset + _EC_ROM + 1):(offset + _EC_NVAR)]
end

@testset "Coupled external system (upper triangular, non-diagonal)" begin
    λ₊, λ₋ = _EC_λ_ext
    c = 0.7 + 0.3im
    a = c / (λ₋ - λ₊)
    T = ComplexF64[1.0 a; 0.0 1.0]

    A_coupled = ComplexF64[λ₊ c; 0.0 λ₋]
    A_diag = ComplexF64[λ₊ 0.0; 0.0 λ₋]

    @testset "the two models really are the same system" begin
        @test A_coupled ≈ T * A_diag * inv(T)
        @test istriu(A_coupled)
        @test !isdiag(A_coupled)   # otherwise the test would take the fast path
    end

    W_coupled, R_coupled = _ec_solve(A_coupled, _EC_F)
    W_diag, R_diag = _ec_solve(A_diag, _EC_F * T)

    Φ_coupled = _ec_external_blocks(W_coupled)
    Φ_diag = _ec_external_blocks(W_diag)

    @testset "external directions transform as Φ_diag = Φ_coupled · T" begin
        # Contract the external axis of the FOM × ORD × N_EXT block with T.
        expected = similar(Φ_diag)
        for k in axes(Φ_diag, 2)
            expected[:, k, :] = Φ_coupled[:, k, :] * T
        end
        @test Φ_diag≈expected rtol=1e-10

        # The identity must have teeth: with the coupling dropped (the pre-fix
        # behaviour) the second direction is off by a·Φ_coupled[:, :, 1], which is
        # far outside the tolerance above.
        @test norm(Φ_coupled[:, :, 1] * a) > 1e-6 * norm(Φ_diag[:, :, 2])
    end

    @testset "the first direction is untouched by the coupling" begin
        # T[:, 1] = e₁, so the leading external direction must be identical.
        @test Φ_coupled[:, :, 1]≈Φ_diag[:, :, 1] rtol=1e-10
    end

    @testset "no external monomial is resonant, so R keeps zero master rows" begin
        offset = findfirst(α -> sum(α) == 1, _EC_mset.exponents) - 1
        for e in 1:_EC_N_EXT
            idx = offset + _EC_ROM + e
            @test norm(R_coupled.poly.coefficients[1:_EC_ROM, idx]) < 1e-12
            @test norm(R_diag.poly.coefficients[1:_EC_ROM, idx]) < 1e-12
        end
    end

    @testset "external rows of R carry the prescribed external dynamics" begin
        offset = findfirst(α -> sum(α) == 1, _EC_mset.exponents) - 1
        A_from_R = R_coupled.poly.coefficients[
            (_EC_ROM + 1):_EC_NVAR, (offset + _EC_ROM + 1):(offset + _EC_NVAR)]
        @test A_from_R ≈ A_coupled
    end
end
