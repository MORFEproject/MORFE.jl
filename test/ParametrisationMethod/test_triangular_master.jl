"""
Upper-triangular master blocks: Schur and Jordan coordinates, with and without conjugate
symmetry.

The master block `Λ̃` of the reduced linear dynamics may be any upper-triangular matrix:
`SpectralData(; upper_triangular = Λ̃, …)` carries it, its diagonal being the master
eigenvalues. Every reference here is exact:

- a change of master coordinates `z = T ẑ` (`T` upper triangular) maps one graph-style
  solution onto the other, `W_S = W ∘ T` and `R_S = T⁻¹ (R ∘ T)`;
- a critically damped Duffing oscillator in Jordan coordinates has a closed-form graph-style
  parametrisation;
- a solve with conjugate symmetry equals the same solve without it.
"""

using Test
using LinearAlgebra
using SparseArrays
using StaticArrays
using Random: MersenneTwister, randn
using MORFE

const _TM_SD = MORFE.SpectralDecomposition
const _TM_IE = MORFE.InvarianceError
const _TM_GRAPH = ResonanceConfig(style = :graph, warn_outer = false)
const _TM_CNF = ResonanceConfig(style = :complex_normal_form, tol = 0.05, warn_outer = false)
const _TM_QUIET = ParametrisationOptions(show_progress = false, verbose = false)

# Σ_γ c_γ z^γ, for coefficients whose last axis runs over the monomials.
function _tm_evaluate(coefficients, mset, z)
    out = zeros(ComplexF64, size(coefficients)[1:(end - 1)])
    for (k, γ) in enumerate(mset.exponents)
        out .+= selectdim(coefficients, ndims(coefficients), k) .* prod(z .^ γ)
    end
    return out
end

# Worst relative violation of W_S(y) = W(T y) and R_S(y) = T⁻¹ R(T y), where T acts on the
# master block and is the identity on the external variables.
function _tm_covariance_error(W, R, WS, RS, T)
    mset = W.poly.multiindex_set
    ROM, NVAR = size(T, 1), length(first(mset.exponents))
    T_full = Matrix{ComplexF64}(I, NVAR, NVAR)
    T_full[1:ROM, 1:ROM] .= T
    rng = MersenneTwister(11)
    worst = 0.0
    for _ in 1:5
        y = 0.3 .* complex.(randn(rng, NVAR), randn(rng, NVAR))
        z = T_full * y
        Wz = _tm_evaluate(W.poly.coefficients, mset, z)
        Rz = T_full \ _tm_evaluate(R.poly.coefficients, mset, z)
        Wy = _tm_evaluate(WS.poly.coefficients, WS.poly.multiindex_set, y)
        Ry = _tm_evaluate(RS.poly.coefficients, RS.poly.multiindex_set, y)
        worst = max(worst, norm(Wy - Wz) / norm(Wz), norm(Ry - Rz) / norm(Rz))
    end
    return worst
end

# T⁻¹ΛT for an upper-triangular T, with the diagonal kept exactly the eigenvalues.
_tm_schur(Λ::Diagonal, T) = Λ + triu(UpperTriangular(T) \ (Λ * T), 1)

function _tm_cubic(α, ORD = 2)
    MultilinearMap((res, x1, x2, x3) -> (@. res += -α * x1 * x2 * x3),
        ntuple(k -> k == 1 ? 3 : 0, ORD))
end

# A named kernel, so that a checkpointed solve can fingerprint it.
_tm_checkpoint_cubic!(res, x1, x2, x3) = (@. res += -0.4 * x1 * x2 * x3)
MORFE.checkpoint_fingerprint_data(::typeof(_tm_checkpoint_cubic!)) = (:tm_cubic, -0.4)

# ── The forced 4-DOF model, with a coupled external block (ORD = 2) ──────────
const _TM_ω = [1.0, 2.7, 4.3, 6.1]
const _TM_ζ = 0.01
const _TM_λ = complex(-_TM_ζ * _TM_ω[1], _TM_ω[1] * sqrt(1 - _TM_ζ^2))
const _TM_K = Matrix(Diagonal(_TM_ω .^ 2))
const _TM_C = Matrix(Diagonal(2_TM_ζ .* _TM_ω))
const _TM_M = Matrix(1.0I, 4, 4)

function _tm_forced_model()
    ms = all_multiindices_up_to(2, 1; min_degree = 1)
    A = ComplexF64[2.1im 0.7+0.3im; 0 -2.1im]
    coefficients = zeros(ComplexF64, 2, length(ms))
    for (j, α) in enumerate(ms.exponents)
        coefficients[:, j] = A[:, findfirst(==(1), α)]
    end
    F = ComplexF64[1.0 0.0; 0.0 1.0; 0.5 -0.25; 0.0 0.75]
    quadratic = MultilinearMap((res, x, y) -> (res .+= x .* y), (2, 0))
    forcing = MultilinearMap((res, r) -> (res .+= F * r), (0, 0), 1)
    return NthOrderModel((_TM_K, _TM_C, _TM_M), (quadratic, forcing),
        ExternalSystem(DensePolynomial(coefficients, ms)))
end

function _tm_forced_eigen_blocks()
    blocks = zeros(ComplexF64, 4, 2, 2)
    blocks[1, :, 1] .= (1, _TM_λ)
    blocks[1, :, 2] .= (1, conj(_TM_λ))
    left = _TM_SD.left_eigenmode_orders_from_slice(
        (_TM_K, _TM_C, _TM_M), blocks[:, 1, :], [_TM_λ, conj(_TM_λ)])
    return blocks, left
end

# ── An ORD = 3 augmented model: parametric mass M(θ) = M₀(1 + θ) ─────────────
function _tm_mass_model()
    K = [2.0 -1.0; -1.0 2.0]
    M0 = Matrix(1.0I, 2, 2)
    Z = zeros(2, 2)
    ep = spectrum(NthOrderModel((K, Z, M0)); solver = DefaultEigensolver())
    mass = MultilinearMap((res, x, r) -> (res .-= r[1] .* (M0 * x)), (0, 0, 1), 1)
    model = NthOrderModel((K, Z, M0, Z), (mass, _tm_cubic(0.3, 3)), ExternalSystem((0.0,)))
    return model, SpectralData(model, ep; master = master_by_sorting(2))
end

@testset "upper-triangular master block" begin
    @testset "master_right_blocks and optional left input" begin
        Φ = ComplexF64[1 2; 3 4; 5 6]
        λ = ComplexF64[-0.1 + 1im, -0.1 - 1im]
        # The diagonal case is the eigenvector convention ψ_{k+1} = λ ψ_k, bit for bit.
        @test master_right_blocks(Φ, Diagonal(λ), 3)[:, 3, :] ==
              (Φ .* transpose(λ)) .* transpose(λ)
        Λ = ComplexF64[λ[1] 0.5; 0 λ[2]]
        blocks = master_right_blocks(Φ, Λ, 3)
        @test blocks[:, 2, :] ≈ Φ * Λ
        @test blocks[:, 3, :] ≈ Φ * Λ^2
        @test_throws ArgumentError master_right_blocks(Φ, transpose(Λ), 2)

        sd = SpectralData(; diagonal = λ, right_modes = blocks)
        @test left_modes(sd) === nothing
        @test_throws ArgumentError check_biorthogonality(sd,
            NthOrderModel(
                (Matrix(1.0I, 3, 3), zeros(3, 3), zeros(3, 3), Matrix(1.0I, 3, 3))))
        @test_throws ArgumentError SpectralData(; diagonal = λ, right_modes = Φ,
            left_blocks = zeros(3, 1, 2))
    end

    @testset "Schur coordinates are covariant (graph style, no conjugate symmetry)" begin
        T = ComplexF64[1 0.3+0.2im; 0 0.8-0.1im]
        Λ = Diagonal(ComplexF64[_TM_λ, conj(_TM_λ)])
        # T upper triangular keeps the diagonal, so the same eigenvalues serve both solves.
        @test diag(UpperTriangular(T) \ (Λ * T)) ≈ diag(Λ)
        Λ_S = _tm_schur(Λ, T)

        model = _tm_forced_model()
        mset = all_multiindices_up_to(4, 3; min_degree = 1)
        blocks, left = _tm_forced_eigen_blocks()
        sd = SpectralData(; diagonal = diag(Λ), right_modes = blocks,
            left_modes = left[:, 2, :], left_blocks = left[:, 1:1, :])
        solve(spectral; kwargs...) = parametrise(model, spectral, mset;
            resonance = _TM_GRAPH, conjugate_permutation = nothing, options = _TM_QUIET,
            kwargs...)
        W, R = solve(sd)

        schur = master_right_blocks(blocks[:, 1, :] * T, Λ_S, 2)
        with_left = SpectralData(; upper_triangular = Λ_S, right_modes = schur,
            left_modes = left[:, 2, :], left_blocks = left[:, 1:1, :])
        without_left = SpectralData(; upper_triangular = Λ_S, right_modes = schur)
        @test master_linear_dynamics(without_left) == Λ_S
        @test master_eigenvalues(without_left) == diag(Λ)
        WA, RA = solve(with_left)       # route A
        WB, RB = solve(without_left)    # route B
        @test _tm_covariance_error(W, R, WA, RA, T) < 1e-10
        @test _tm_covariance_error(W, R, WB, RB, T) < 1e-10

        # The two routes give one dual basis, and it is dual and left-invariant.
        dual_A = master_left_basis(model, schur, Λ_S, with_left.master.left_blocks)
        dual_B = master_left_basis(model, schur, Λ_S)
        @test dual_A ≈ dual_B rtol = 1e-10
        A, B = linear_first_order_matrices(model)
        Φ, Ψ = reshape(schur, :, 2), reshape(dual_B, :, 2)
        @test Ψ' * B * Φ ≈ I atol = 1e-12
        @test norm(Ψ' * A - Λ_S * Ψ' * B) < 1e-10 * norm(Ψ' * A)

        # The state error is frame-independent, so the validation tool agrees with itself.
        rng = MersenneTwister(3)
        E = complex.(randn(rng, 4), randn(rng, 4))
        state_error(W, R, sd) = _TM_IE._state_error(
            _TM_IE._state_error_factorisation(model,
                _TM_IE._state_error_operators(model, W, R, sd), 0.9im), E,
            zeros(ComplexF64, 6), zeros(ComplexF64, 6))
        @test state_error(WB, RB, without_left) ≈ state_error(W, R, sd) rtol = 1e-10
    end

    @testset "Schur coordinates on an ORD = 3 augmented model" begin
        model, sd = _tm_mass_model()
        mset = MultiindexSet([SVector(a, b, c) for a in 0:3 for b in 0:3
                              for c in 0:2
                              if a + b ≤ 3 && a + b + c ≥ 1])
        λ = master_eigenvalues(sd)
        T = ComplexF64[1 -0.2+0.4im; 0 1.1+0.3im]
        Λ_S = _tm_schur(Diagonal(λ), T)
        schur = master_right_blocks(right_modes(sd) * T, Λ_S, 3)
        solve(spectral; kwargs...) = parametrise(model, spectral, mset;
            resonance = _TM_GRAPH, conjugate_permutation = nothing, options = _TM_QUIET,
            kwargs...)
        W, R = solve(sd)
        WS, RS = solve(SpectralData(; upper_triangular = Λ_S, right_modes = schur))
        @test _tm_covariance_error(W, R, WS, RS, T) < 1e-10
    end

    @testset "a pair coupled to itself is re-based onto the conjugate frame" begin
        ω, ζ = 1.3, 0.05
        λ = complex(-ζ * ω, ω * sqrt(1 - ζ^2))
        K, C, M = fill(ω^2, 1, 1), fill(2ζ * ω, 1, 1), fill(1.0, 1, 1)
        model = NthOrderModel((K, C, M), (_tm_cubic(0.4),))
        mset = all_multiindices_up_to(2, 5; min_degree = 1)
        eigen = master_right_blocks(ones(ComplexF64, 1, 2), Diagonal([λ, conj(λ)]), 2)
        left = _TM_SD.left_eigenmode_orders_from_slice((K, C, M), ones(ComplexF64, 1, 2),
            [λ, conj(λ)])
        sd = SpectralData(; diagonal = [λ, conj(λ)], right_modes = eigen,
            left_modes = left[:, 2, :], left_blocks = left[:, 1:1, :],
            conjugate_permutation = [2, 1])
        W, R = parametrise(model, sd, mset; resonance = _TM_CNF, options = _TM_QUIET)

        # Schur-like given frame: the second column is not the conjugate of the first.
        T0 = ComplexF64[1 0.4-0.3im; 0 0.7+0.2im]
        given = reshape(reshape(eigen, 2, 2) / T0, 1, 2, 2)
        coupling = triu(T0 * Diagonal([λ, conj(λ)]) / T0, 1)
        Λ̃ = Diagonal([λ, conj(λ)]) + coupling
        @test abs(coupling[1, 2]) > 0.1
        frame, Λ̂, T = conjugate_master_frame(given, Λ̃, [2, 1])
        @test T ≈ T0 rtol = 1e-12
        @test Λ̂ == Diagonal([λ, conj(λ)])
        @test frame[:, :, 2] == conj(frame[:, :, 1])

        # SpectralData re-bases at construction: the parametrisation is in the frame it holds.
        sd_given = @test_logs (:info, r"were not the conjugates") SpectralData(;
            upper_triangular = Λ̃, right_modes = given, conjugate_permutation = [2, 1])
        @test master_linear_dynamics(sd_given) == Diagonal([λ, conj(λ)])
        @test right_modes(sd_given) == right_modes(sd)
        WS, RS = parametrise(
            model, sd_given, mset; resonance = _TM_CNF, options = _TM_QUIET)
        @test WS.poly.coefficients ≈ W.poly.coefficients rtol = 1e-10
        @test RS.poly.coefficients ≈ R.poly.coefficients rtol = 1e-10
    end

    @testset "Jordan block at a complex eigenvalue, with conjugate symmetry" begin
        # Two identical damped oscillators, the first driven by the second: each of λ and
        # λ̄ is a double, defective eigenvalue, with the closed-form chain φ_a, φ_b.
        k, c, ε = 1.0, 0.1, 0.5
        λ = complex(-c / 2, sqrt(4k - c^2) / 2)
        K, C, M = [k ε; 0 k], c * Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2)
        model = NthOrderModel((K, C, M), (_tm_cubic(1.0),))
        φa, φb = ComplexF64[1, 0], ComplexF64[0, -(c + 2λ) / ε]
        Λ̃ = ComplexF64[λ 0 1 0; 0 conj(λ) 0 1; 0 0 λ 0; 0 0 0 conj(λ)]
        blocks = master_right_blocks(hcat(φa, conj(φa), φb, conj(φb)), Λ̃, 2)
        σ = [2, 1, 4, 3]
        sd = SpectralData(; upper_triangular = Λ̃, right_modes = blocks,
            conjugate_permutation = σ)
        mset = all_multiindices_up_to(4, 3; min_degree = 1)
        solve(; kwargs...) = parametrise(model, sd, mset; options = _TM_QUIET, kwargs...)
        for resonance in (_TM_GRAPH, _TM_CNF)
            Wσ, Rσ = solve(; resonance)
            W0, R0 = solve(; resonance, conjugate_permutation = nothing)
            @test Wσ.poly.coefficients ≈ W0.poly.coefficients rtol = 1e-10
            @test Rσ.poly.coefficients ≈ R0.poly.coefficients rtol = 1e-10
        end

        # The graph style is exact up to truncation, and real points map to real states.
        W, R = solve(resonance = _TM_GRAPH)
        a, b = 0.05 + 0.02im, -0.03 + 0.04im
        x = _tm_evaluate(W.poly.coefficients, mset, [a, conj(a), b, conj(b)])
        @test norm(imag(x)) < 1e-14 * norm(x)
        @test _TM_IE.invariance_error_norms(model, W, R; n_samples = 50,
            amplitude = 0.01, rng = MersenneTwister(5)).max < 1e-10

        # V4: blocks consistent with a coupling these modes do not have. (For a single
        # pair every coupling is realisable, by Cayley–Hamilton; a chain across two degrees
        # of freedom is not.)
        wrong = copy(Λ̃)
        wrong[1, 3], wrong[2, 4] = 1.5, 1.5
        @test_throws ArgumentError parametrise(model,
            SpectralData(; upper_triangular = wrong,
                right_modes = master_right_blocks(blocks[:, 1, :], wrong, 2),
                conjugate_permutation = σ),
            mset; resonance = _TM_GRAPH, options = _TM_QUIET)

        # The dual basis is conjugate-paired; the left eigenvectors cannot supply it.
        dual = master_left_basis(model, blocks, Λ̃; conjugate_permutation = σ)
        A, B = linear_first_order_matrices(model)
        Φ, Ψ = reshape(blocks, :, 4), reshape(dual, :, 4)
        @test Ψ' * B * Φ ≈ I atol = 1e-10
        @test norm(Ψ' * A - Λ̃ * Ψ' * B) < 1e-9 * norm(Ψ' * A)
        @test Ψ[:, σ] ≈ conj(Ψ) rtol = 1e-10
        ℓ = ComplexF64[0, 1]                      # the one left eigenvector of λ
        eigenvectors = _TM_SD.left_eigenmode_orders_from_slice(
            (K, C, M), hcat(ℓ, conj(ℓ), ℓ, conj(ℓ)), diag(Λ̃))
        fallback = @test_logs (:info, r"do not give the dual basis") master_left_basis(
            model, blocks, Λ̃, eigenvectors; conjugate_permutation = σ)
        @test fallback ≈ dual rtol = 1e-10
    end

    @testset "Jordan block at a real eigenvalue: critically damped Duffing, exact" begin
        # x = z₁ in Jordan coordinates: ẋ = −ωz₁ + z₂ and ż₂ = −ωz₂ − αz₁³ solve
        # ẍ + 2ωẋ + ω²x = −αx³ exactly, and the graph style leaves nothing in W at |γ| ≥ 2.
        ω, α = 1.2, 0.7
        model = NthOrderModel((fill(ω^2, 1, 1), fill(2ω, 1, 1), fill(1.0, 1, 1)),
            (_tm_cubic(α),))
        Λ̃ = ComplexF64[-ω 1; 0 -ω]
        blocks = master_right_blocks(ComplexF64[1 0], Λ̃, 2)
        @test blocks[1, 2, :] == [-ω, 1]
        mset = all_multiindices_up_to(2, 5; min_degree = 1)
        cubic = findfirst(==(SVector(3, 0)), mset.exponents)
        for σ in (nothing, [1, 2])
            sd = SpectralData(; upper_triangular = Λ̃, right_modes = blocks,
                conjugate_permutation = σ)
            W, R = parametrise(model, sd, mset; resonance = _TM_GRAPH, options = _TM_QUIET)
            expected = zeros(ComplexF64, 2, length(mset))
            expected[:, 1:2] .= Λ̃
            expected[2, cubic] = -α
            @test maximum(abs, R.poly.coefficients - expected) < 1e-12
            nonlinear = [k for (k, γ) in enumerate(mset.exponents) if sum(γ) ≥ 2]
            @test maximum(abs, W.poly.coefficients[:, :, nonlinear]) < 1e-12
        end
    end

    @testset "route B against the assembled first-order form" begin
        # The dual basis is computed with FOM-size solves only; check its two defining
        # identities against the assembled ORD·FOM pencil.
        function dual_identities(model, blocks, Λ)
            dual = master_left_basis(model, blocks, Λ)
            A, B = linear_first_order_matrices(model)
            Φ, Ψ = reshape(blocks, :, size(blocks, 3)), reshape(dual, :, size(blocks, 3))
            return norm(Ψ' * B * Φ - I), norm(Ψ' * A - Λ * Ψ' * B) / norm(Ψ' * A)
        end
        # A free rigid-body mode: λ = 0 is a Jordan block of the first-order form, so the
        # shift is what keeps the solve away from the singular P(0) = K.
        K, Z, M = [1.0 -1.0; -1.0 1.0], zeros(2, 2), Matrix(1.0I, 2, 2)
        Λ0 = ComplexF64[0 1; 0 0]
        rigid = master_right_blocks(hcat(ComplexF64[1, 1] / sqrt(2), zeros(2)), Λ0, 2)
        for terms in ((K, Z, M), sparse.((K, Z, M)))
            biorthogonality, invariance = dual_identities(NthOrderModel(terms), rigid, Λ0)
            @test biorthogonality < 1e-10
            @test invariance < 1e-10
        end
        # ORD = 3 with a zero top matrix (the augmented mass model), in Schur coordinates.
        model, sd = _tm_mass_model()
        T = ComplexF64[1 -0.2+0.4im; 0 1.1+0.3im]
        Λ_S = _tm_schur(Diagonal(master_eigenvalues(sd)), T)
        biorthogonality, invariance = dual_identities(model,
            master_right_blocks(right_modes(sd) * T, Λ_S, 3), Λ_S)
        @test biorthogonality < 1e-10
        @test invariance < 1e-10
    end

    @testset "conjugate_master_frame refuses what it cannot re-base" begin
        λ = complex(-0.05, 1.0)
        pair(φ, μ) = master_right_blocks(hcat(φ, conj(φ)), Diagonal([μ, conj(μ)]), 2)
        conjugate = pair(ComplexF64[1, 0.5im], λ)
        Λ = Diagonal([λ, conj(λ)])
        frame, Λ̂, T = conjugate_master_frame(conjugate, Λ, [2, 1])
        @test frame === conjugate && Λ̂ === Λ && T == I

        # Two identical oscillators: the conjugate of column 1 is column 4, after column 3.
        e1, e2 = ComplexF64[1, 0], ComplexF64[0, 1]
        double = master_right_blocks(hcat(e1, e2, conj(e2), conj(e1)),
            Diagonal([λ, λ, conj(λ), conj(λ)]), 2)
        @test_throws ArgumentError conjugate_master_frame(double,
            Diagonal([λ, λ, conj(λ), conj(λ)]), [3, 4, 1, 2])

        @test_throws ArgumentError conjugate_master_frame(
            reshape(ComplexF64[1, 1im], 1, 2, 1), fill(complex(-1.0), 1, 1), [1])

        # A pair at the real axis: φ and conj(φ) are almost parallel.
        μ = complex(-1.0, 1e-10)
        φ = ComplexF64[1, μ]
        q = ComplexF64[-conj(μ), 1] / norm(φ)
        near = reshape(hcat(φ, q), 1, 2, 2)
        @test_throws ArgumentError conjugate_master_frame(near,
            ComplexF64[μ 0.3; 0 conj(μ)], [2, 1])
    end

    @testset "validation" begin
        ω, ζ = 1.3, 0.05
        λ = complex(-ζ * ω, ω * sqrt(1 - ζ^2))
        model = NthOrderModel((fill(ω^2, 1, 1), fill(2ζ * ω, 1, 1), fill(1.0, 1, 1)),
            (MultilinearMap(_tm_checkpoint_cubic!, (3, 0)),))
        mset = all_multiindices_up_to(2, 3; min_degree = 1)
        T0 = ComplexF64[1 0.4-0.3im; 0 0.7+0.2im]
        eigen = master_right_blocks(ones(ComplexF64, 1, 2), Diagonal([λ, conj(λ)]), 2)
        given = reshape(reshape(eigen, 2, 2) / T0, 1, 2, 2)
        Λ̃ = Diagonal([λ, conj(λ)]) + triu(T0 * Diagonal([λ, conj(λ)]) / T0, 1)
        coupled(blocks; σ = nothing) = SpectralData(; upper_triangular = Λ̃,
            right_modes = blocks, conjugate_permutation = σ)
        solve(spectral; options = _TM_QUIET) = parametrise(model, spectral, mset;
            resonance = _TM_CNF, options)

        # The master block is given once: eigenvalues or an upper-triangular matrix.
        @test_throws ArgumentError SpectralData(; right_modes = given)
        @test_throws ArgumentError SpectralData(; diagonal = [λ, conj(λ)],
            upper_triangular = Λ̃, right_modes = given)
        @test_throws ArgumentError SpectralData(; upper_triangular = transpose(Λ̃),
            right_modes = given)
        @test_throws ArgumentError SpectralData(; upper_triangular = Λ̃[:, 1:1],
            right_modes = given)
        # A diagonal block is the eigenvector case, bit for bit.
        from_matrix = SpectralData(; upper_triangular = Diagonal([λ, conj(λ)]),
            right_modes = eigen)
        @test master_linear_dynamics(from_matrix) isa Diagonal
        Wd, Rd = solve(SpectralData(; diagonal = [λ, conj(λ)], right_modes = eigen))
        Wz, Rz = solve(from_matrix)
        @test Wz.poly.coefficients == Wd.poly.coefficients
        @test Rz.poly.coefficients == Rd.poly.coefficients

        # V3: the stored levels disagree with the block. (V4 needs more than one degree of
        # freedom; it is tested on the Jordan chain above.)
        skewed = copy(given)
        skewed[:, 2, :] .*= 1.01
        @test_throws ArgumentError solve(coupled(skewed))

        # V2 on supplied storage: R holds a coupled block, but W's columns are no pairs.
        W, R = solve(coupled(given))
        conjugate = SpectralData(; diagonal = [λ, conj(λ)], right_modes = eigen,
            conjugate_permutation = [2, 1])
        rset = MORFE.Resonance.build_resonance_set(model, mset, conjugate, _TM_CNF)
        @test_throws ArgumentError solve_parametrisation(model, mset, conjugate, rset;
            initial_solution = (W, R), options = _TM_QUIET)

        # The coupling is part of the checkpoint fingerprint: a resume needs the same block.
        mktempdir() do directory
            checkpoint = CheckpointOptions(joinpath(directory, "coupled");
                problem_id = "triangular")
            options = ParametrisationOptions(show_progress = false, verbose = false,
                checkpoint = checkpoint)
            W1, R1 = solve(coupled(given); options)
            @test_throws ArgumentError solve(
                SpectralData(; diagonal = [λ, conj(λ)], right_modes = given); options)
            W2, R2 = solve(coupled(given); options)
            @test R2.poly.coefficients == R1.poly.coefficients
            @test W2.poly.coefficients == W1.poly.coefficients
        end
    end
end
