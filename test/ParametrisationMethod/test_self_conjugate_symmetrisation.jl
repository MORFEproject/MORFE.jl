"""
Self-conjugate monomials are projected onto the conjugate-symmetric subspace.

A monomial γ with P·γ == γ (q₁q₂, r₁ for a real external variable r₁, q₁q₂r₂r₃, ...) has no
conjugate partner to be filled from, so the solver computes it directly. For a real FOM its
coefficients must satisfy

    W_γ real,        f_γ[r] == conj(f_γ[perm[r]])   (master rows r),

but the bordered system is solved in complex arithmetic and can amplify round-off along its
antisymmetric direction (1e-13 in the right-hand side became 8e-2 in the solution for a
finite-element turbine blade). The solver therefore projects after every such solve with
`symmetrise_self_conjugate_monomial!`: in the nonlinear schedule (`_fill_job_conjugate!`) and
for real external directions (`_solve_external_directions!`).

The projection makes both identities hold bit for bit: `real` drops the imaginary part
exactly, and `(a + conj(b))/2 == conj((b + conj(a))/2)` because IEEE addition commutes and
rounding is symmetric under negation. The integration tests therefore use `==`. Without the
projection the model below gives defects from 1e-19 up to 8e-12, which an `≈` test could not
tell apart from round-off.
"""

using Test
using LinearAlgebra
using StaticArrays: SVector
using Random: MersenneTwister, randn
using MORFE

const _SC_PS = MORFE.ParametrisationSolver

# ─────────────────────────────────────────────────────────────────────────────
# Unit test of the projection on hand-made coefficients.
# Variables (q₁, q₂, r₁, r₂, r₃): a master pair, a real external variable, an external pair.
# ─────────────────────────────────────────────────────────────────────────────
const _SC_ROM = 2
const _SC_N_EXT = 3
const _SC_NVAR = _SC_ROM + _SC_N_EXT
const _SC_perm = SVector{_SC_NVAR, Int}(2, 1, 3, 5, 4)

function _sc_symmetry_data(mset)
    mdict = Dict(mset.exponents[i] => i for i in eachindex(mset.exponents))
    return _SC_PS._build_conjugate_symmetry(_SC_perm, Set{Int}(), mset, mdict)
end

function _sc_random_objects(mset; FOM = 6, ORD = 2, seed = 0x5c)
    W, R = create_parametrisation_method_objects(
        mset, ORD, FOM, _SC_ROM, _SC_N_EXT, ComplexF64)
    rng = MersenneTwister(seed)
    W.poly.coefficients .= randn(rng, ComplexF64, size(W.poly.coefficients))
    R.poly.coefficients .= randn(rng, ComplexF64, size(R.poly.coefficients))
    return W, R
end

@testset "symmetrise_self_conjugate_monomial! is the conjugate-symmetric projection" begin
    mset = all_multiindices_up_to(_SC_NVAR, 3; min_degree = 1)
    sym = _sc_symmetry_data(mset)
    idx = findfirst(==(SVector(1, 1, 1, 0, 0)), mset.exponents)
    @test sym.monomial_map[idx] == idx   # q₁q₂r₁ really is self-conjugate

    W, R = _sc_random_objects(mset)
    W0, R0 = copy(W.poly.coefficients), copy(R.poly.coefficients)
    _SC_PS.symmetrise_self_conjugate_monomial!(W, R, idx, sym)
    Wc, Rc = W.poly.coefficients, R.poly.coefficients

    @testset "W becomes exactly its real part" begin
        @test Wc[:, :, idx] == real.(W0[:, :, idx])
        @test all(iszero ∘ imag, Wc[:, :, idx])
    end

    @testset "master rows become exactly conjugate-symmetric" begin
        for r in 1:_SC_ROM
            @test Rc[r, idx] == (R0[r, idx] + conj(R0[_SC_perm[r], idx])) / 2
            @test Rc[r, idx] == conj(Rc[_SC_perm[r], idx])
        end
    end

    @testset "external rows and every other monomial are untouched" begin
        @test Rc[(_SC_ROM + 1):end, idx] == R0[(_SC_ROM + 1):end, idx]
        others = setdiff(axes(Wc, 3), idx)
        @test Wc[:, :, others] == W0[:, :, others]
        @test Rc[:, others] == R0[:, others]
    end

    @testset "idempotent" begin
        W1, R1 = copy(Wc), copy(Rc)
        _SC_PS.symmetrise_self_conjugate_monomial!(W, R, idx, sym)
        @test W.poly.coefficients == W1
        @test R.poly.coefficients == R1
    end

    @testset "removes the antisymmetric part and keeps the symmetric one" begin
        # Split arbitrary master-row data into s (s[perm r] = conj s[r]) and
        # a (a[perm r] = -conj a[r]); the projection must return s.
        W, R = _sc_random_objects(mset; seed = 0x5d)
        s = ComplexF64[0.3 - 1.2im, 0.3 + 1.2im]
        a = ComplexF64[0.7 + 0.4im, -0.7 + 0.4im]
        R.poly.coefficients[1:_SC_ROM, idx] .= s .+ a
        w_real = randn(MersenneTwister(0x5e), size(W.poly.coefficients, 1),
            size(W.poly.coefficients, 2))
        W.poly.coefficients[:, :, idx] .= w_real .+ im .* 0.25 .* w_real
        _SC_PS.symmetrise_self_conjugate_monomial!(W, R, idx, sym)
        @test R.poly.coefficients[1:_SC_ROM, idx] ≈ s atol=4eps()
        @test W.poly.coefficients[:, :, idx] == w_real
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Integration: a forced 4-DOF model with a Lorenz-like external system, as in a structure
# driven by a chaotic load. External eigenvalues (μ, iΩ, -iΩ): r₁ real, (r₂, r₃) a pair.
#
# Non-diagonal M and K so the bordered solves mix all DOFs and round-off is generic.
# ω₁ ≈ 0.44 with ω₂ ≈ 2.23 and Ω = 0.37: no superharmonic of degree ≤ 4 comes within 10% of
# a non-master mode, so the solve raises no outer-resonance warning.
# ─────────────────────────────────────────────────────────────────────────────
const _SC_K = [0.2 -0.1 0.0 0.0; -0.1 6.0 -1.0 0.0; 0.0 -1.0 9.0 -2.0; 0.0 0.0 -2.0 14.0]
const _SC_M = [1.0 0.2 0.0 0.0; 0.2 1.0 0.2 0.0; 0.0 0.2 1.0 0.2; 0.0 0.0 0.2 1.0]
const _SC_C = 0.02 .* _SC_M .+ 0.005 .* _SC_K
const _SC_a2 = [0.7, -0.4, 0.25, 0.9]
const _SC_a3 = [-1.1, 0.5, 0.3, -0.6]
# Real physical forcing: real column for r₁, conjugate columns for the pair (r₂, r₃).
const _SC_F = let f = zeros(ComplexF64, 4, 3)
    f[:, 1] = [1.0, 0.3, -0.2, 0.5]
    f[:, 2] = [0.4 + 0.2im, -0.1 + 0.5im, 0.3, 0.2im]
    f[:, 3] = conj.(f[:, 2])
    f
end
const _SC_ORDER = 4

function _sc_model()
    quad = MultilinearMap((res, x, y) -> (@. res += _SC_a2 * x * y), (2, 0);
        fully_asymmetric = false)
    cubic = MultilinearMap((res, x, y, z) -> (@. res += _SC_a3 * x * y * z), (3, 0);
        fully_asymmetric = false)
    forcing = MultilinearMap((res, r) -> (res .+= _SC_F * r), (0, 0), 1)
    ext = ExternalSystem((complex(-1.7), 0.37im, -0.37im))
    return NthOrderModel((_SC_K, _SC_C, _SC_M), (quad, cubic, forcing), ext), ext
end

@testset "Self-conjugate monomials are exactly conjugate-symmetric after parametrise" begin
    model, ext = _sc_model()
    spectral = SpectralData(model, spectrum(model); master = [1, 2],
        conjugate_permutation = :detect)
    perm = full_conjugate_permutation(MORFE.master_conjugate_permutation(spectral), ext)
    @test perm == _SC_perm

    options = ParametrisationOptions(show_progress = false, verbose = false)
    mset = all_multiindices_up_to(_SC_NVAR, _SC_ORDER; min_degree = 1)
    self_conjugate = [j
                      for (j, γ) in enumerate(mset.exponents)
                      if SVector{_SC_NVAR, Int}(ntuple(k -> γ[perm[k]], _SC_NVAR)) == γ]
    r₁ = findfirst(==(SVector(0, 0, 1, 0, 0)), mset.exponents)
    @test r₁ in self_conjugate

    for style in (:graph, :complex_normal_form)
        @testset "style = :$style" begin
            resonance = ResonanceConfig(; style, tol = 0.05)
            # min_level = Warn: the model must stay free of outer resonances, otherwise the
            # comparison below would measure ill-posedness rather than symmetry.
            W, R = @test_logs min_level=Base.CoreLogging.Warn parametrise(
                model, spectral, _SC_ORDER; resonance, conjugate_permutation = perm, options)
            W_ref, R_ref = parametrise(model, spectral, _SC_ORDER; resonance,
                conjugate_permutation = nothing, options)
            Wc, Rc = W.poly.coefficients, R.poly.coefficients
            @test W.poly.multiindex_set.exponents == mset.exponents

            @testset "γ = $(Tuple(mset.exponents[j]))" for j in self_conjugate
                @test all(iszero ∘ imag, Wc[:, :, j])
                @test all(r -> Rc[r, j] == conj(Rc[perm[r], j]), 1:_SC_ROM)
            end

            @testset "the projection does not move the solution beyond round-off" begin
                # Same problem solved without conjugate symmetry: the projection removes only
                # the spurious antisymmetric part.
                @test Wc≈W_ref.poly.coefficients rtol=1e-10
                @test Rc≈R_ref.poly.coefficients rtol=1e-10
            end

            @testset "the self-conjugate coefficients are not trivially zero" begin
                nonzero = count(
                    j -> sum(mset.exponents[j]) ≥ 2 && norm(Wc[:, :, j]) > 1e-3,
                    self_conjugate)
                @test nonzero ≥ 8
            end
        end
    end

    @testset "real external direction r₁ (solved before the nonlinear schedule)" begin
        # Under :graph the r₁ direction is bordered by the master modes, so its solve runs in
        # complex arithmetic and needs the projection in `_solve_external_directions!`.
        # Under :complex_normal_form it is non-resonant and the solve is real anyway.
        W, R = parametrise(model, spectral, _SC_ORDER;
            resonance = ResonanceConfig(style = :graph, tol = 0.05),
            conjugate_permutation = perm, options)
        @test norm(R.poly.coefficients[1:_SC_ROM, r₁]) > 1e-3   # really bordered
        @test norm(W.poly.coefficients[:, :, r₁]) > 1e-3
        @test all(iszero ∘ imag, W.poly.coefficients[:, :, r₁])
        @test R.poly.coefficients[1, r₁] == conj(R.poly.coefficients[2, r₁])
    end
end
