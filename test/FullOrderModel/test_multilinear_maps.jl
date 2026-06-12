using Test
using MORFE
using MORFE.MultilinearMaps: evaluate_term!
using LinearAlgebra
using SparseArrays

function make_bilinear_elementwise()
    f!(res, x, y) = (res .+= x .* y)
    return f!
end
# Quadratic (symmetric):  res[i] += x[i]^2
function make_quadratic_elementwise()
    f!(res, x, y) = (res .+= x .* y)   # symmetric: f(x,x) = x.^2
    return f!
end

# Trilinear:  res[i] += x[i]*y[i]*z[i]
function make_trilinear_elementwise()
    f!(res, x, y, z) = (res .+= x .* y .* z)
    return f!
end

@testset "MultilinearMaps" begin
    @testset "MultilinearMap construction" begin
        @testset "first-order shorthand (no multiindex)" begin
            f!(res, x, y) = (res .+= x .* y)
            m = MultilinearMap(f!)
            @test m.deg == 2
            @test m.multiindex == (2,)
            @test m.multiplicity_external == 0
        end

        @testset "explicit multiindex, order-1 system" begin
            f!(res, x, y) = (res .+= x .* y)
            m = MultilinearMap(f!, (2,))
            @test m.deg == 2
            @test m.multiindex == (2,)
            @test m.multiplicity_external == 0
        end

        @testset "order-2 system, mixed derivatives" begin
            # multiindex = (1, 1) → one x^(0) and one x^(1); deg = 2
            f!(res, x0, x1) = (res .+= x0 .* x1)
            m = MultilinearMap(f!, (1, 1))
            @test m.deg == 2
            @test m.multiindex == (1, 1)
            @test m.multiplicity_external == 0
        end

        @testset "with external state (multiplicity_external = 1)" begin
            # multiindex = (1,), multiplicity_external = 1 → deg = 2
            f!(res, x, r) = (res .+= x .* r)
            m = MultilinearMap(f!, (1,), 1)
            @test m.deg == 2
            @test m.multiplicity_external == 1
        end

        @testset "purely external term (multiindex all-zero, me = 2)" begin
            # multiindex = (0,), multiplicity_external = 2 → deg = 2
            f!(res, r1, r2) = (res .+= r1 .* r2)
            m = MultilinearMap(f!, (0,), 2)
            @test m.deg == 2
            @test m.multiplicity_external == 2
        end

        @testset "degree-3 term" begin
            f!(res, a, b, c) = (res .+= a .* b .* c)
            m = MultilinearMap(f!, (3,))
            @test m.deg == 3
        end
    end
    @testset "MultilinearMap construction guards" begin
        @testset "negative multiindex entry" begin
            f!(res, x) = (res .+= x)
            @test_throws AssertionError MultilinearMap(f!, (-1,))
        end

        @testset "nargs mismatch (too few inputs declared)" begin
            # deg would be 2 but f! takes only 1 input beyond res
            f!(res, x) = (res .+= x)
            @test_throws AssertionError MultilinearMap(f!, (2,))
        end

        @testset "nargs mismatch (too many inputs declared)" begin
            f!(res, x, y, z) = (res .+= x .* y .* z)
            @test_throws AssertionError MultilinearMap(f!, (2,))
        end

        @testset "degree < 2 without external state" begin
            f!(res, x) = (res .+= x)
            @test_throws AssertionError MultilinearMap(f!, (1,))
        end

        @testset "negative multiplicity_external" begin
            f!(res, x, y) = (res .+= x .* y)
            @test_throws AssertionError MultilinearMap(f!, (1,), -1)
        end

        @testset "multiple methods on f! are rejected" begin
            # Julia allows multiple methods; the constructor must see exactly one.
            g!(res, x, y) = (res .+= x .* y)
            g!(res, x, y, z) = (res .+= x .* y .* z)
            # checks the assertion fired when nargs is ambiguous.
            h! = g!
            @test_throws AssertionError MultilinearMap(h!, (2,))   # nargs mismatch also fires
        end
    end
    @testset "evaluate_term! correctness" begin
        n = 4
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 6.0, 7.0, 8.0]
        z = [0.1, 0.2, 0.3, 0.4]
        r = [10.0, 20.0, 30.0, 40.0]

        @testset "degree-2 bilinear, no external" begin
            f!(res, a, b) = (res .+= a .* b)
            m = MultilinearMap(f!, (1, 1))  # order-2 system: one x^(0), one x^(1)
            xs = (x, y)
            res = zeros(n)
            evaluate_term!(res, m, xs, nothing)
            @test res ≈ x .* y
        end

        @testset "degree-2 quadratic on x^(0) only" begin
            f!(res, a, b) = (res .+= a .* b)
            m = MultilinearMap(f!, (2,))    # order-1: two slots from x^(0)
            xs = (x,)
            res = zeros(n)
            evaluate_term!(res, m, xs, nothing)
            @test res ≈ x .^ 2
        end

        @testset "degree-3 trilinear on x^(0)" begin
            f!(res, a, b, c) = (res .+= a .* b .* c)
            m = MultilinearMap(f!, (3,))
            xs = (x,)
            res = zeros(n)
            evaluate_term!(res, m, xs, nothing)
            @test res ≈ x .^ 3
        end

        @testset "mixed: one x^(0), one external r" begin
            f!(res, a, b) = (res .+= a .* b)
            m = MultilinearMap(f!, (1,), 1)
            xs = (x,)
            res = zeros(n)
            evaluate_term!(res, m, xs, r)
            @test res ≈ x .* r
        end

        @testset "purely external: two r slots" begin
            f!(res, a, b) = (res .+= a .* b)
            m = MultilinearMap(f!, (0,), 2)
            xs = (x,)   # xs present but not used
            res = zeros(n)
            evaluate_term!(res, m, xs, r)
            @test res ≈ r .^ 2
        end
    end

    @testset "evaluate_term! linearity" begin
        n = 5
        a = randn(n)
        b = randn(n)
        c = randn(n)
        α = 3.7

        @testset "linear in first slot" begin
            f!(res, x, y) = (res .+= x .* y)
            m = MultilinearMap(f!, (1, 1))

            res1 = zeros(n)
            evaluate_term!(res1, m, (a, b), nothing)
            res2 = zeros(n)
            evaluate_term!(res2, m, (c, b), nothing)
            res3 = zeros(n)
            evaluate_term!(res3, m, (α .* a .+ c, b), nothing)
            @test res3 ≈ α .* res1 .+ res2
        end

        @testset "linear in second slot" begin
            f!(res, x, y) = (res .+= x .* y)
            m = MultilinearMap(f!, (1, 1))

            res1 = zeros(n)
            evaluate_term!(res1, m, (a, b), nothing)
            res2 = zeros(n)
            evaluate_term!(res2, m, (a, c), nothing)
            res3 = zeros(n)
            evaluate_term!(res3, m, (a, α .* b .+ c), nothing)
            @test res3 ≈ α .* res1 .+ res2
        end

        @testset "symmetry under slot permutation" begin
            f!(res, x, y) = (res .+= x .* y)
            m = MultilinearMap(f!, (1, 1))

            res1 = zeros(n)
            evaluate_term!(res1, m, (a, b), nothing)
            res2 = zeros(n)
            evaluate_term!(res2, m, (b, a), nothing)
            @test res1 ≈ res2
        end

        @testset "trilinear: linear in each of three slots" begin
            f!(res, x, y, z) = (res .+= x .* y .* z)
            m = MultilinearMap(f!, (3,))

            # slot 1: fix y=b, z=c; vary x ↦ α*a + c
            r1 = zeros(n)
            evaluate_term!(r1, m, (a,), nothing)   # f(a,a,a)? No — all 3 slots come from xs[1]=a
            # For multiindex=(3,), all 3 args ARE the same vector xs[1].
            # So f(αa, αa, αa) = α^3 f(a,a,a)  — test cubic scaling instead.
            r_alpha = zeros(n)
            evaluate_term!(r_alpha, m, (α .* a,), nothing)
            @test r_alpha ≈ α^3 .* r1
        end
    end
    @testset "evaluate_term! external state error" begin
        n = 3
        x = ones(n)
        f!(res, a, b) = (res .+= a .* b)
        m = MultilinearMap(f!, (1,), 1)   # expects one external arg
        xs = (x,)

        @test_throws ErrorException evaluate_term!(zeros(n), m, xs, nothing)
    end
end #@testset "MultilinearMaps"