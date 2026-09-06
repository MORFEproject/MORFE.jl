# `observable_polynomial` and `cycle_amplitude` — projecting a parametrisation onto one
# scalar quantity, and reading its amplitude off a periodic orbit.
#
# Both are checked against closed-form algebra rather than recorded output. A single
# harmonic `c·z₁` traces the circle `2|c|ρ·cos(φ + arg c)`, whose half peak-to-peak is
# exactly `2|c|ρ` at every phase offset; that identity carries most of the file, including
# the gauge-invariance property the reference blessing in MORFEFerrite's example 01 relies on.

using Test
using MORFE
using MORFE.Polynomials: DensePolynomial
using MORFE.ParametrisationMethod: Parametrisation
using StaticArrays

# A Parametrisation carrying prescribed displacement-level coefficients. `C1` is (FOM, L);
# the higher derivative levels are irrelevant to `observable_polynomial` and left zero.
function _parametrisation(C1::Matrix{ComplexF64}, mset)
    FOM, L = size(C1)
    coeffs = zeros(ComplexF64, FOM, 3, L)
    coeffs[:, 1, :] .= C1
    return Parametrisation(DensePolynomial(coeffs, mset), 0)
end

_mset2() = MultiindexSet([SVector(1, 0), SVector(0, 1), SVector(2, 1), SVector(1, 2)])

@testset "observable_polynomial picks one dof" begin
    mset = _mset2()
    C1 = ComplexF64[1+2im 3-1im 0.5im 0.25
                    -4 0.5 1+1im 2im
                    0 7im 0 1]
    W = _parametrisation(C1, mset)

    for i in 1:3
        u = observable_polynomial(W, i)
        @test MORFE.Polynomials.coefficients(u) == C1[i, :]
        @test MORFE.Polynomials.multiindex_set(u) === mset
    end

    # The functional method with a unit vector must agree with the index method: the two
    # are the same projection written two ways, and a transpose slip would show up here.
    for i in 1:3
        l = zeros(3)
        l[i] = 1.0
        @test MORFE.Polynomials.coefficients(observable_polynomial(W, l)) ≈ C1[i, :]
    end

    # and a general functional is the plain bilinear combination, not the sesquilinear one
    l = [2.0, -1.0, 0.5]
    @test MORFE.Polynomials.coefficients(observable_polynomial(W, l)) ≈
          vec(transpose(C1) * l)
end

@testset "observable_polynomial rejects a bad selector" begin
    W = _parametrisation(zeros(ComplexF64, 3, 4), _mset2())
    @test_throws ArgumentError observable_polynomial(W, 0)
    @test_throws ArgumentError observable_polynomial(W, 4)
    @test_throws ArgumentError observable_polynomial(W, ones(2))   # wrong length functional
end

@testset "cycle_amplitude of a single harmonic" begin
    # u(φ) = 2·Re(c ρ e^{iφ}) has half peak-to-peak 2|c|ρ, whatever arg c is.
    #
    # It is a *sampled* extremum, so the identity holds only to the sampling error. Near
    # its peak the signal is flat, u ≈ u_max(1 - δ²/2) with δ ≤ π/n_phase, which bounds the
    # relative error at (π/n_phase)²/2 ≈ 3e-7 for the default grid. That is the accuracy of
    # every amplitude this function returns, and it is worth stating rather than hiding
    # behind a loose tolerance: refining the grid recovers the exact value below.
    mset = MultiindexSet([SVector(1, 0), SVector(0, 1)])
    for c in (1.0 + 0.0im, -0.3 + 0.7im, 0.05im)
        P = DensePolynomial(ComplexF64[c, conj(c)], mset)
        for ρ in (0.0, 0.5, 3.0)
            @test cycle_amplitude(P, ρ)≈2 * abs(c) * ρ rtol=3e-7
            @test cycle_amplitude(P, ρ; n_phase = 1 << 20)≈2 * abs(c) * ρ rtol=1e-11
        end
    end
end

@testset "cycle_amplitude is invariant to the eigenvector gauge" begin
    # Rescaling the master eigenvector by exp(iψ) multiplies the (a,b) coefficient by
    # exp(i(a-b)ψ), which shifts the signal rigidly in φ. The continuous extrema do not
    # move, so the amplitude is gauge-free; this is the property that makes it safe to bless
    # in a reference when the eigensolver's phase is not reproducible between runs.
    #
    # On a finite grid the shift moves the samples relative to the signal, so the invariance
    # is recovered only as the grid refines. Both halves are asserted.
    mset = _mset2()
    c1, c3 = 0.4 - 0.9im, 0.03 + 0.02im
    P = DensePolynomial(ComplexF64[c1, conj(c1), c3, conj(c3)], mset)
    base = cycle_amplitude(P, 1.7)
    base_dense = cycle_amplitude(P, 1.7; n_phase = 1 << 20)

    for ψ in (0.3, -1.1, 2.5)
        rotated = ComplexF64[
            c1 * cis(ψ), conj(c1) * cis(-ψ), c3 * cis(ψ), conj(c3) * cis(-ψ)]
        Q = DensePolynomial(rotated, mset)
        @test cycle_amplitude(Q, 1.7)≈base rtol=1e-6
        @test cycle_amplitude(Q, 1.7; n_phase = 1 << 20)≈base_dense rtol=1e-11
    end
end

@testset "cycle_amplitude converges in n_phase, and needs a full cycle" begin
    # The coefficients carry a phase, so the peak does not sit on any of the coarse grids.
    # With a real signal the peak lands on φ = 0, which every grid contains, and the test
    # would pass on a bug that samples nothing but that point.
    mset = _mset2()
    c1, c3 = cis(0.37), 0.4 * cis(1.1)
    P = DensePolynomial(ComplexF64[c1, conj(c1), c3, conj(c3)], mset)
    ρ = 1.0

    dense = cycle_amplitude(P, ρ; n_phase = 1 << 20)
    e_coarse = abs(cycle_amplitude(P, ρ; n_phase = 64) - dense)
    e_fine = abs(cycle_amplitude(P, ρ; n_phase = 256) - dense)
    @test e_coarse > 0.0                     # the coarse grid really does miss the peak
    @test e_fine < e_coarse / 4              # and refining 4× gains more than a factor 4
    @test e_fine < 1e-4

    @test_throws ArgumentError cycle_amplitude(P, ρ; n_phase = 1)
end

@testset "cycle_amplitude carries the external coordinates" begin
    # z₁ z̄₁⁰ η: the amplitude is linear in η, so passing it wrongly is visible.
    mset = MultiindexSet([SVector(1, 0, 0), SVector(0, 1, 0),
        SVector(1, 0, 1), SVector(0, 1, 1)])
    P = DensePolynomial(ComplexF64[1, 1, 2, 2], mset)

    @test cycle_amplitude(P, 1.0, (0.0,))≈2.0 atol=1e-12
    @test cycle_amplitude(P, 1.0, (0.5,))≈2.0 * (1 + 2 * 0.5) atol=1e-12

    # a mismatched external length is an error, not a silent zero
    @test_throws ArgumentError cycle_amplitude(P, 1.0)
    @test_throws ArgumentError cycle_amplitude(P, 1.0, (0.1, 0.2))
end

@testset "the two functions compose into a backbone" begin
    # The end-to-end path the clamped-beam example uses: project W onto a dof, truncate it
    # per order, and read the amplitude along the ρ grid that `normal_form_branch` returns.
    mset = _mset2()
    W = _parametrisation(ComplexF64[0.5 0.5 0.01 0.01], mset)
    u = observable_polynomial(W, 1)

    Rmset = MultiindexSet([SVector(1, 0), SVector(0, 1), SVector(2, 1)])
    Rc = zeros(ComplexF64, 2, 3)
    Rc[1, 1], Rc[1, 3] = 3im, 0.25im
    Rc[2, 2] = -3im
    R = ReducedDynamics(DensePolynomial(Rc, Rmset), 0)

    b = normal_form_branch(R; amplitudes = 0:0.5:2)
    a = cycle_amplitude.(Ref(u), b.amplitude)

    @test b.frequency ≈ 3 .+ 0.25 .* b.amplitude .^ 2
    @test a[1] == 0.0
    @test issorted(a)                                   # hardening in amplitude
    # Truncating the observable to degree 1 drops the cubic, leaving the linear amplitude.
    linear = MORFE.Polynomials.restrict_polynomial_to_degree(u, 1)
    @test cycle_amplitude.(Ref(linear), b.amplitude) ≈ 2 * 0.5 .* b.amplitude
end
