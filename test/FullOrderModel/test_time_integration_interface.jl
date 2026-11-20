# Tests for MORFE.TimeIntegrationInterface

using Test
using LinearAlgebra, SparseArrays, Random

using MORFE
using MORFE.FullOrderModel: NthOrderModel, linear_first_order_matrices
using MORFE.MultilinearMaps: MultilinearMap
using MORFE.ExternalSystems: ExternalSystem
using MORFE.Polynomials: DensePolynomial, coefficients
using MORFE.Multiindices: all_multiindices_up_to
using MORFE.TimeIntegrationInterface
const TI = MORFE.TimeIntegrationInterface

# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------

"""n×n matrices B_0 … B_ORD; B_ORD is symmetric positive definite (invertible mass)."""
function linear_terms(n, ord; sparse_mats = false, seed = 1)
    rng = MersenneTwister(seed)
    Bs = map(0:ord) do i
        G = randn(rng, n, n)
        i == ord ? G * G' + n * I : G
    end
    Bs = Tuple(Matrix{Float64}.(Bs))
    return sparse_mats ? map(sparse, Bs) : Bs
end

"""Polynomial dynamics ṙ = g(r) from `coeff(i, e)` = coefficient of monomial `e` in g_i.
Independent of the ordering of the multiindex set."""
function ext_polynomial(N, maxdeg, coeff)
    ms = all_multiindices_up_to(N, maxdeg)
    deleteat!(ms.exponents, 1)                       # drop the constant monomial
    C = zeros(Float64, N, length(ms.exponents))
    for (m, e) in enumerate(ms.exponents), i in 1:N

        C[i, m] = coeff(i, Tuple(e))
    end
    return DensePolynomial(C, ms)
end

# g = (λ₁ r₁ + c r₁², λ₂ r₂ + d r₁ r₂)
const λ1, λ2, cq, dq = -0.5, -0.3, 0.2, 0.1
function ext2()
    ExternalSystem(ext_polynomial(2, 2, (i, e) -> begin
        i == 1 && e == (1, 0) && return λ1
        i == 1 && e == (2, 0) && return cq
        i == 2 && e == (0, 1) && return λ2
        i == 2 && e == (1, 1) && return dq
        return 0.0
    end))
end
g2(r) = [λ1 * r[1] + cq * r[1]^2, λ2 * r[2] + dq * r[1] * r[2]]

"""Stand-in for `SciMLBase.NullParameters()`: neither a vector nor callable."""
struct NoParams end

# nonlinear terms (all accumulate into `res`)
function duffing(γ)
    MultilinearMap((res, a, b, c) -> (res .-= γ .* a .* b .* c), (3, 0), 0;
        fully_asymmetric = false)
end
function forcing(b)
    MultilinearMap((res, r) -> (res .+= b .* r[1]), (0, 0), 1;
        fully_asymmetric = false)
end
function coupling(c)
    MultilinearMap((res, r1, r2) -> (res .+= c .* r1[1] .* r2[2]), (0, 0), 2;
        fully_asymmetric = false)
end

# ----------------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------------

@testset "TimeIntegrationInterface" begin
    n = 3
    γ = 0.7
    b = [1.0, -2.0, 0.5]
    cc = [0.3, 0.1, -0.4]
    rng = MersenneTwister(42)

    # models used repeatedly
    B2 = linear_terms(n, 2)
    model_free = NthOrderModel(B2, (duffing(γ),))
    model_ext = NthOrderModel(B2, (duffing(γ), forcing(b), coupling(cc)), ext2())

    @testset "construction and bookkeeping" begin
        sys = FirstOrderSystem(model_free)
        @test state_dimension(sys) == 2n

        sys = FirstOrderSystem(model_ext)
        @test state_dimension(sys) == 2n + 2
        sys_in = FirstOrderSystem(model_ext; input = [2])
        @test state_dimension(sys_in) == 2n + 1
        sys_in2 = FirstOrderSystem(model_ext; input = [1, 2])
        @test state_dimension(sys_in2) == 2n

        @test_throws ArgumentError FirstOrderSystem(model_ext; input = [3])
        @test_throws ArgumentError FirstOrderSystem(model_ext; input = [0])
        @test_throws ArgumentError FirstOrderSystem(model_ext; input = [1, 1])
        @test_throws ArgumentError FirstOrderSystem(model_free; input = [1])
    end

    @testset "pack_state / unpack_state" begin
        sys = FirstOrderSystem(model_ext; input = [2])
        x, v = randn(rng, n), randn(rng, n)
        u = pack_state(sys, (x, v), [0.9])
        @test length(u) == state_dimension(sys)
        xs, r = unpack_state(sys, u)
        @test collect(xs[1]) == x
        @test collect(xs[2]) == v
        @test collect(r) == [0.9]

        @test_throws ArgumentError pack_state(sys, (x,), [0.9])         # wrong ORD
        @test_throws ArgumentError pack_state(sys, (x, v), [0.9, 1.0])  # wrong #ext
        @test_throws ArgumentError pack_state(sys, (x, v), ())          # missing ext
    end

    @testset "linear operators vs linear_first_order_matrices" begin
        for ord in 1:3, sp in (false, true)

            Bs = linear_terms(n, ord; sparse_mats = sp, seed = 10 + ord)
            model = NthOrderModel(Bs)
            sys = FirstOrderSystem(model)
            A, B = linear_first_order_matrices(model)
            X = randn(rng, ord * n)

            y = similar(X)
            @test TI._apply_linear!(y, model, X) ≈ A * X
            @test TI._apply_mass!(y, model, X) ≈ B * X
            @test Matrix(mass_matrix(sys)) ≈ Matrix(B)
        end
    end

    @testset "mass matrix with external variables" begin
        for input in (Int[], [2], [1, 2])
            sys = FirstOrderSystem(model_ext; input = input)
            N = state_dimension(sys)
            nS = N - 2n
            _, B = linear_first_order_matrices(model_ext)
            Mref = [Matrix(B) zeros(2n, nS); zeros(nS, 2n) I(nS)]

            @test mass_matrix(sys) isa SparseMatrixCSC
            @test Matrix(mass_matrix(sys)) ≈ Mref
        end
    end

    @testset "rhs: model without external system" begin
        sys = FirstOrderSystem(model_free)
        x, v = randn(rng, n), randn(rng, n)
        u = pack_state(sys, (x, v))
        du = zeros(2n)
        sys(du, u, nothing, 0.0)

        B0, B1, _ = B2
        @test du[1:n] ≈ v
        @test du[(n + 1):(2n)] ≈ -B0 * x - B1 * v - γ .* x .^ 3
    end

    @testset "rhs: ORD = 1" begin
        Bs = linear_terms(n, 1; seed = 5)
        # B₁ ẋ + B₀ x = -γ x³ is not multilinear for ORD=1 without a degree-3 term:
        term = MultilinearMap(
            (res, a, b_, c) -> (res .-= γ .* a .* b_ .* c), (3,), 0;
            fully_asymmetric = false)
        model = NthOrderModel(Bs, (term,))
        sys = FirstOrderSystem(model)
        x = randn(rng, n)
        du = zeros(n)
        sys(du, copy(x), nothing, 0.0)
        @test du ≈ -Bs[1] * x - γ .* x .^ 3
    end

    @testset "rhs: external variables integrated" begin
        sys = FirstOrderSystem(model_ext)
        x, v = randn(rng, n), randn(rng, n)
        r = [0.8, -0.6]
        u = pack_state(sys, (x, v), r)
        du = zeros(2n + 2)
        sys(du, u, nothing, 0.0)

        B0, B1, _ = B2
        F = -γ .* x .^ 3 .+ b .* r[1] .+ cc .* r[1] * r[2]
        @test du[1:n] ≈ v
        @test du[(n + 1):(2n)] ≈ -B0 * x - B1 * v + F
        @test du[(2n + 1):end] ≈ g2(r)
    end

    @testset "rhs: external variables as inputs" begin
        x, v = randn(rng, n), randn(rng, n)
        r = [0.8, -0.6]
        B0, B1, _ = B2

        sys_full = FirstOrderSystem(model_ext)
        du_full = zeros(2n + 2)
        sys_full(du_full, pack_state(sys_full, (x, v), r), nothing, 0.0)

        # r₂ comes from p, r₁ is integrated: X part and ṙ₁ agree with the full system
        sys_in = FirstOrderSystem(model_ext; input = [2])
        u = pack_state(sys_in, (x, v), [r[1]])
        du = zeros(2n + 1)
        sys_in(du, u, [r[2]], 0.0)
        @test du[1:(2n)] ≈ du_full[1:(2n)]
        @test du[end] ≈ du_full[end - 1]

        # p as a function of time
        du2 = zeros(2n + 1)
        sys_in(du2, u, t -> [r[2] + t], 0.5)
        sys_in(du, u, [r[2] + 0.5], 0.5)
        @test du2 ≈ du
        @test du2 != du_full[[1:(2n); 2n + 1]]               # p really enters

        # both inputs: no external state left in u
        sys_all = FirstOrderSystem(model_ext; input = [1, 2])
        du3 = zeros(2n)
        sys_all(du3, pack_state(sys_all, (x, v)), r, 0.0)
        @test du3 ≈ du_full[1:(2n)]

        # input order follows `input`, not the variable index
        sys_rev = FirstOrderSystem(model_ext; input = [2, 1])
        sys_rev(du3, pack_state(sys_rev, (x, v)), [r[2], r[1]], 0.0)
        @test du3 ≈ du_full[1:(2n)]

        # errors
        @test_throws ArgumentError sys_in(du, u, nothing, 0.0)       # p missing
        @test_throws ArgumentError sys_in(du, u, [1.0, 2.0], 0.0)    # wrong length
    end

    @testset "re-based (oscillatory) external system stays real" begin
        # ṙ₁ = ω r₂, ṙ₂ = -ω r₁ + κ r₁²: non-triangular linear part → complex basis Q
        ω, κ = 1.5, 0.2
        ext_osc = ExternalSystem(ext_polynomial(2, 2, (i, e) -> begin
            i == 1 && e == (0, 1) && return ω
            i == 2 && e == (1, 0) && return -ω
            i == 2 && e == (2, 0) && return κ
            return 0.0
        end))
        @test MORFE.ExternalSystems.external_basis(ext_osc) !== nothing
        model = NthOrderModel(B2, (duffing(γ), forcing(b), coupling(cc)), ext_osc)

        sys = FirstOrderSystem(model)
        @test eltype(coefficients(physical_external_dynamics(ext_osc))) == Float64
        x, v = randn(rng, n), randn(rng, n)
        r = [0.8, -0.6]
        u = pack_state(sys, (x, v), r)
        @test eltype(u) == Float64
        du = zeros(2n + 2)
        sys(du, u, nothing, 0.0)

        B0, B1, _ = B2
        F = -γ .* x .^ 3 .+ b .* r[1] .+ cc .* r[1] * r[2]     # terms see physical r
        @test du[1:n] ≈ v
        @test du[(n + 1):(2n)] ≈ -B0 * x - B1 * v + F
        @test du[(2n + 1):end] ≈ [ω * r[2], -ω * r[1] + κ * r[1]^2]

        # inputs are physical components too
        sys_in = FirstOrderSystem(model; input = [2])
        du_in = zeros(2n + 1)
        sys_in(du_in, pack_state(sys_in, (x, v), [r[1]]), [r[2]], 0.0)
        @test du_in ≈ du[1:(2n + 1)]

        # missing p (e.g. SciMLBase.NullParameters()) gives the ArgumentError
        @test_throws ArgumentError sys_in(du_in, pack_state(sys_in, (x, v), [r[1]]),
            NoParams(), 0.0)
    end

    @testset "argument checks of the callable" begin
        sys = FirstOrderSystem(model_ext; input = [2])
        u = zeros(state_dimension(sys))
        @test_throws AssertionError sys(zeros(length(u) + 1), u, [0.0], 0.0)
        @test_throws AssertionError sys(zeros(length(u)), zeros(length(u) + 1), [0.0], 0.0)
        sys0 = FirstOrderSystem(model_free)       # no external system: must not error
        @test sys0(zeros(2n), zeros(2n), nothing, 0.0) === nothing
    end

    @testset "time integration against an analytic solution" begin
        # 2 ẍ + 2 x = 2 r,  ṙ = -r,  r(0) = 1,  x(0) = 1/2, ẋ(0) = -1/2
        # ⇒ x(t) = e^{-t}/2  (non-trivial mass B₂ = 2 on purpose)
        Bs = ([2.0;;], [0.0;;], [2.0;;])
        ext1 = ExternalSystem(ext_polynomial(1, 1, (i, e) -> -1.0))
        model = NthOrderModel(Bs, (forcing([2.0]),), ext1)

        function rk4(sys, u0, p, T, dt)
            M = lu(Matrix(mass_matrix(sys)))
            f(u, t) = (du = zeros(eltype(u), length(u)); sys(du, u, p, t); M \ du)
            u, t = copy(u0), 0.0
            for _ in 1:round(Int, T / dt)
                k1 = f(u, t)
                k2 = f(u + dt / 2 * k1, t + dt / 2)
                k3 = f(u + dt / 2 * k2, t + dt / 2)
                k4 = f(u + dt * k3, t + dt)
                u += dt / 6 * (k1 + 2k2 + 2k3 + k4)
                t += dt
            end
            return u
        end
        T, dt = 2.0, 1e-3

        # r integrated
        sys = FirstOrderSystem(model)
        u = rk4(sys, pack_state(sys, ([0.5], [-0.5]), [1.0]), nothing, T, dt)
        @test u[1] ≈ exp(-T) / 2 atol = 1e-8
        @test u[2] ≈ -exp(-T) / 2 atol = 1e-8
        @test u[3] ≈ exp(-T) atol = 1e-8

        # harmonic forcing through a re-based system: 2ẍ + 8x = 2 r₁,
        # ṙ₁ = r₂, ṙ₂ = -r₁, r(0) = (1, 0) ⇒ r₁ = cos t, x(t) = cos(t)/3
        ext_h = ExternalSystem(ext_polynomial(2, 1, (i, e) -> begin
            i == 1 && e == (0, 1) && return 1.0
            i == 2 && e == (1, 0) && return -1.0
            return 0.0
        end))
        model_h = NthOrderModel(([8.0;;], [0.0;;], [2.0;;]), (forcing([2.0]),), ext_h)
        sys_h = FirstOrderSystem(model_h)
        u = rk4(sys_h, pack_state(sys_h, ([1 / 3], [0.0]), [1.0, 0.0]), nothing, T, dt)
        @test eltype(u) == Float64
        @test u[1] ≈ cos(T) / 3 atol = 1e-8
        @test u[2] ≈ -sin(T) / 3 atol = 1e-8
        @test u[3:4] ≈ [cos(T), -sin(T)] atol = 1e-8

        # r supplied as input signal r(t) = e^{-t}: same response
        sys_in = FirstOrderSystem(model; input = [1])
        u = rk4(sys_in, pack_state(sys_in, ([0.5], [-0.5])), t -> [exp(-t)], T, dt)
        @test u[1] ≈ exp(-T) / 2 atol = 1e-8
        @test u[2] ≈ -exp(-T) / 2 atol = 1e-8
    end
end
