# `normal_form_branch` — the polar reduction of a single-conjugate-pair normal form.
#
# Every case here is built on the Stuart-Landau normal form
#
#     ż = (σ + iω) z + (a + ib) z|z|²
#
# whose branch is closed-form: ρ̇ = ρ(σ + aρ²) vanishes at ρ* = √(-σ/a), and the frequency
# there is Ω = ω + bρ*². That makes every assertion below an EXACT comparison against
# algebra rather than against a previous run, and it needs no mesh and no solve, so the file
# is cheap enough to sit in the default suite.

using Test
using MORFE
using MORFE.Polynomials: DensePolynomial
using StaticArrays

# ż₁ = (σ + iω)z₁ + (a + ib)z₁|z₁|², row 2 its conjugate. NVAR = ROM = 2, no parameter.
function _stuart_landau(σ, ω, a, b)
    mset = MultiindexSet([SVector(1, 0), SVector(0, 1), SVector(2, 1)])
    C = zeros(ComplexF64, 2, 3)
    C[1, 1] = σ + im * ω
    C[1, 3] = a + im * b
    C[2, 2] = σ - im * ω
    return ReducedDynamics(DensePolynomial(C, mset), 0)
end

# The same, with the growth rate made linear in one external coordinate: σ → σ + cη.
# Re g = σ + cη + aρ² = 0 gives η(ρ) = -(σ + aρ²)/c in closed form.
function _stuart_landau_parametric(σ, ω, a, b, c)
    mset = MultiindexSet([SVector(1, 0, 0), SVector(0, 1, 0), SVector(1, 0, 1),
        SVector(0, 1, 1), SVector(2, 1, 0), SVector(1, 2, 0)])
    C = zeros(ComplexF64, 3, 6)
    C[1, 1] = σ + im * ω
    C[1, 3] = c
    C[1, 5] = a + im * b
    C[2, 2] = σ - im * ω
    C[2, 4] = c
    C[2, 6] = a - im * b
    return ReducedDynamics(DensePolynomial(C, mset), 1)
end

@testset "backbone, no parameter" begin
    σ, ω, a, b = 0.4, 3.0, -0.5, 0.25
    R = _stuart_landau(σ, ω, a, b)
    amps = collect(0:0.25:2)
    br = normal_form_branch(R; amplitudes = amps)

    # Ω(ρ) = ω + bρ² exactly — the phase equation is linear in the coefficients, so this is
    # an identity, not an approximation.
    @test br.frequency ≈ ω .+ b .* amps .^ 2 atol=1e-12
    @test br.amplitude == amps
    @test isempty(br.parameter)          # no parameter was solved for
    @test length(br.stable) == length(amps)
end

@testset "supercritical branch" begin
    σ, ω, a, b, c = -0.2, 3.0, -0.5, 0.25, 2.0
    R = _stuart_landau_parametric(σ, ω, a, b, c)
    amps = [0.0, 0.5, 1.0, 1.5]
    br = normal_form_branch(R; parameter = 1, amplitudes = amps)

    @test br.parameter ≈ [-(σ + a * ρ^2) / c for ρ in amps] atol=1e-12
    @test br.frequency ≈ ω .+ b .* amps .^ 2 atol=1e-12
    @test br.amplitude == amps
end

@testset "stability follows the sign of ∂ρ(ρ Re g)" begin
    σ, ω, b, c = -0.2, 3.0, 0.25, 2.0
    amps = [0.5, 1.0, 1.5]

    # a < 0 saturates: the limit cycle is stable.
    sup = normal_form_branch(_stuart_landau_parametric(σ, ω, -0.5, b, c);
        parameter = 1, amplitudes = amps)
    @test all(sup.stable)

    # a > 0 does not: the same branch is unstable, which is what makes the bifurcation
    # subcritical. Only this sign distinguishes the two, so it is the whole test.
    sub = normal_form_branch(_stuart_landau_parametric(σ, ω, 0.5, b, c);
        parameter = 1, amplitudes = amps)
    @test !any(sub.stable)
end

@testset "parameter_range clips, and empty is not an error" begin
    R = _stuart_landau_parametric(-0.2, 3.0, -0.5, 0.25, 2.0)
    br = normal_form_branch(R; parameter = 1, amplitudes = [1.0],
        parameter_range = (10.0, 20.0))
    @test isempty(br.parameter)
    @test isempty(br.amplitude)
    @test isempty(br.frequency)
    @test isempty(br.stable)
end

# ż₁ = z₁(σ + c₁η + c₂η²) + a z₁|z₁|². Quadratic in η, so Re g = 0 has TWO roots per
# amplitude: the sheet growing out of the bifurcation, and a far one the truncated series
# manufactures. The numbers are the Kármán ROM's, where the far root sits at Re ≈ 11.
function _two_sheets(σ = 0.004, c₁ = -212.0, c₂ = 2981.0, a = -0.111, ω = 16.86)
    mset = MultiindexSet([SVector(1, 0, 0), SVector(0, 1, 0), SVector(1, 0, 1),
        SVector(0, 1, 1), SVector(2, 1, 0), SVector(1, 2, 0),
        SVector(1, 0, 2), SVector(0, 1, 2)])
    C = zeros(ComplexF64, 3, 8)
    C[1, 1] = σ + im * ω
    C[2, 2] = σ - im * ω
    C[1, 3] = c₁
    C[2, 4] = c₁
    C[1, 5] = a
    C[2, 6] = a
    C[1, 7] = c₂
    C[2, 8] = c₂
    return ReducedDynamics(DensePolynomial(C, mset), 1)
end

@testset "sheet = :primary keeps one root per amplitude" begin
    R = _two_sheets()
    amps = collect(0:0.05:1.0)
    both = normal_form_branch(R; parameter = 1, amplitudes = amps,
        parameter_range = (-0.01, 0.1))
    one = normal_form_branch(R; parameter = 1, sheet = :primary, amplitudes = amps,
        parameter_range = (-0.01, 0.1))

    @test length(both.amplitude) == 2 * length(amps)   # both sheets
    @test length(one.amplitude) == length(amps)       # one of them
    @test one.amplitude == amps
    # It is the sheet born at the bifurcation, not the far one at η ≈ 0.071.
    @test all(<(1e-2), one.parameter)
    @test one.parameter[1]≈1.9e-5 atol=1e-6
    # and every kept point really is a root of the full problem
    @test issubset(Set(one.parameter), Set(both.parameter))
end

@testset "sheet = :primary stops at a gap instead of jumping" begin
    # Narrowing the window drops the tracked root at ρ ≈ 1.40. Only the far sheet is left
    # in range after that, and splicing the two into one curve is exactly the failure this
    # guards against, so the branch must end there.
    R = _two_sheets()
    amps = collect(0:0.05:3.0)
    one = normal_form_branch(R; parameter = 1, sheet = :primary, amplitudes = amps,
        parameter_range = (-0.001, 0.1))
    both = normal_form_branch(R; parameter = 1, amplitudes = amps,
        parameter_range = (-0.001, 0.1))

    @test 1.2 < one.amplitude[end] < 1.5
    @test one.amplitude[end] < amps[end]              # it stopped early
    @test both.amplitude[end] == amps[end]             # :all did not
    @test maximum(one.parameter) < 1e-2               # never crossed to the far sheet
end

@testset "sheet rejects an unknown symbol" begin
    @test_throws ArgumentError normal_form_branch(_two_sheets(); parameter = 1,
        sheet = :nearest)
end

@testset "graded truncation agrees with a truncated model" begin
    # The solve is graded, so restricting R and re-running must equal running the full R
    # when the extra monomials are above the retained degree. Add a quintic to have
    # something to drop.
    σ, ω, a, b = 0.4, 3.0, -0.5, 0.25
    mset = MultiindexSet([SVector(1, 0), SVector(0, 1), SVector(2, 1), SVector(3, 2)])
    C = zeros(ComplexF64, 2, 4)
    C[1, 1] = σ + im * ω
    C[1, 3] = a + im * b
    C[1, 4] = 0.05 - 0.02im            # degree 5, dropped at order 3
    C[2, 2] = σ - im * ω
    R = ReducedDynamics(DensePolynomial(C, mset), 0)

    amps = collect(0:0.5:2)
    cubic = normal_form_branch(restrict_ReducedDynamics_to_degree(R, 3); amplitudes = amps)
    @test cubic.frequency ≈
          _stuart_landau(σ, ω, a, b) |>
          R3 -> normal_form_branch(R3; amplitudes = amps).frequency atol=1e-12
    # and the quintic genuinely mattered, so the test is not vacuous
    @test !isapprox(normal_form_branch(R; amplitudes = amps).frequency, cubic.frequency;
        atol = 1e-8)
end

@testset "rejects a row that is not in normal form" begin
    # z₁² has a - b = 2, so R₁/z₁ keeps an exp(iθ) factor and the polar reduction is invalid.
    # Silently returning a number here is the failure mode the check exists to prevent.
    mset = MultiindexSet([SVector(1, 0), SVector(0, 1), SVector(2, 0)])
    C = zeros(ComplexF64, 2, 3)
    C[1, 1] = 0.1 + 3im
    C[1, 3] = 0.5                       # the offender
    C[2, 2] = 0.1 - 3im
    R = ReducedDynamics(DensePolynomial(C, mset), 0)

    err = try
        normal_form_branch(R)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("a - b", err.msg)
    @test occursin("(2, 0)", err.msg)   # the message names the offending exponent
end

@testset "rejects ROM ≠ 2" begin
    # Three master coordinates are not a single conjugate pair.
    mset = MultiindexSet([SVector(1, 0, 0), SVector(0, 1, 0), SVector(0, 0, 1)])
    R = ReducedDynamics(DensePolynomial(zeros(ComplexF64, 3, 3), mset), 0)
    @test_throws ArgumentError normal_form_branch(R)
end
