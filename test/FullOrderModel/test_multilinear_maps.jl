using Test
using MORFE
using MORFE.MultilinearMaps: evaluate_term!, _symmetry_label
using MORFE.MultilinearTerms: symmetry_type
using LinearAlgebra
using SparseArrays

# Callable struct: exercises the arity check on something that is not a `Function`.
struct ScaledProduct
    a::Float64
end
(s::ScaledProduct)(res, x, y) = (res .+= s.a .* x .* y)

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

        @testset "external-only forcing terms" begin
            # deg == 1 is legal only because the term reads the external state; this
            # branch of the degree guard had no coverage.
            f1!(res, r) = (res .+= r)
            m1 = MultilinearMap(f1!, (0,), 1)
            @test m1.deg == 1
            @test m1.multiplicity_external == 1

            m2 = MultilinearMap(f1!, (0, 0), 1)
            @test m2.deg == 1
            @test m2.multiindex == (0, 0)
        end
    end

    @testset "keyword constructor" begin
        @testset "round-trips against the positional forms" begin
            f2!(res, x, y) = (res .+= x .* y)
            f3!(res, x, y, z) = (res .+= x .* y .* z)

            two_arg = [(f2!, (2,)), (f2!, (1, 1)), (f3!, (3, 0)), (f2!, (0, 2)),
                (f3!, (2, 1))]
            for (f!, mi) in two_arg
                pos = MultilinearMap(f!, mi)
                kw = MultilinearMap(f!; multiindex = mi)
                @test typeof(kw) === typeof(pos)
                @test kw.f! === pos.f!
                @test kw.multiindex === pos.multiindex
                @test kw.multiplicity_external === pos.multiplicity_external
                @test kw.deg === pos.deg
                @test kw.fully_asymmetric === pos.fully_asymmetric
            end

            three_arg = [(f2!, (1,), 1), (f2!, (0,), 2), (f3!, (0, 0), 1)]
            for (f!, mi, me) in three_arg
                pos = MultilinearMap(f!, mi, me)
                kw = MultilinearMap(f!; multiindex = mi, multiplicity_external = me)
                @test typeof(kw) === typeof(pos)
                @test kw.multiindex === pos.multiindex
                @test kw.multiplicity_external === pos.multiplicity_external
                @test kw.deg === pos.deg
            end
        end

        @testset "bare call reproduces the first-order shorthand" begin
            f!(res, x, y) = (res .+= x .* y)
            m = MultilinearMap(f!)
            @test m.multiindex === (2,)
            # The shorthand used to build a UInt8 tuple that the inner constructor
            # silently converted; pin the element type so that cannot come back.
            @test typeof(m.multiindex) === NTuple{1, Int}
            @test m.deg == 2
            @test m.multiplicity_external == 0
            @test m.fully_asymmetric === nothing
        end

        @testset "fully_asymmetric round-trips through every form" begin
            f!(res, x, y) = (res .+= x .* y)
            for fa in (nothing, false, true)
                @test MultilinearMap(f!; fully_asymmetric = fa).fully_asymmetric === fa
                @test MultilinearMap(f!; multiindex = (2,),
                    fully_asymmetric = fa).fully_asymmetric === fa
                @test MultilinearMap(f!, (2,);
                    fully_asymmetric = fa).fully_asymmetric === fa
                @test MultilinearMap(f!, (1,), 1;
                    fully_asymmetric = fa).fully_asymmetric === fa
            end
        end

        @testset "multiindex accepts a vector" begin
            f!(res, x, y, z) = (res .+= x .* y .* z)
            m = MultilinearMap(f!; multiindex = [2, 1])
            @test m.multiindex == (2, 1)
            @test m isa MultilinearMap{2}
        end

        @testset "order keyword" begin
            f2!(res, x, y) = (res .+= x .* y)
            f3!(res, x, y, z) = (res .+= x .* y .* z)

            padded = MultilinearMap(f2!; multiindex = (2,), order = 3)
            @test padded.multiindex == (2, 0, 0)
            @test padded isa MultilinearMap{3}

            inferred = MultilinearMap(f2!; order = 3)
            @test inferred.multiindex == (2, 0, 0)

            from_degree = MultilinearMap(f3!; degree = 3, order = 3)
            @test from_degree.multiindex == (3, 0, 0)

            with_external = MultilinearMap(f2!; multiplicity_external = 1, order = 2)
            @test with_external.multiindex == (1, 0)
            @test with_external.deg == 2

            @test_throws "never truncate" MultilinearMap(
                f3!; multiindex = (2, 0, 1), order = 2)
            @test MultilinearMap(f2!; multiindex = (2,), order = 1).multiindex == (2,)
        end

        @testset "order padding is symmetry-neutral" begin
            f!(res, x, y) = (res .+= x .* y)
            short = MultilinearMap(f!, (2,))
            long = MultilinearMap(f!; multiindex = (2,), order = 3)
            @test typeof(symmetry_type(short)) === typeof(symmetry_type(long))
        end

        @testset "order keyword composes into an NDOrderModel" begin
            # The payoff: no hand-written trailing zeros, and the ORD=3 tuple still
            # unifies at model construction.
            q!(res, x, y) = (res .+= x .* y)
            t!(res, x, y, z) = (res .+= x .* y .* z)
            K = Matrix{Float64}(I, 3, 3)
            terms = (MultilinearMap(q!, (2,); order = 3),
                MultilinearMap(t!, (3, 0, 0); fully_asymmetric = false))
            @test NDOrderModel((K, K, K, K), terms) isa NDOrderModel{3}
        end

        @testset "derivatives keyword" begin
            f!(res, x, y, z) = (res .+= x .* y .* z)
            m = MultilinearMap(f!; derivatives = (0, 0, 1))
            @test m.multiindex == (2, 1)
            @test m.deg == 3

            # numerically identical to the multiindex spelling
            n = 4
            x = collect(1.0:n)
            xd = collect(n:-1.0:1.0)
            ref = zeros(n)
            got = zeros(n)
            evaluate_term!(ref, MultilinearMap(f!, (2, 1)), (x, xd), nothing)
            evaluate_term!(got, m, (x, xd), nothing)
            @test got ≈ ref

            @test MultilinearMap(f!; derivatives = (0, 0, 1), order = 4).multiindex ==
                  (2, 1, 0, 0)
            @test MultilinearMap(f!; derivatives = (0, 2, 2)).multiindex == (1, 0, 2)

            @test_throws "non-decreasing" MultilinearMap(f!; derivatives = (1, 0, 0))
            @test_throws "must be non-negative" MultilinearMap(
                f!; derivatives = (-1, 0, 0))
            @test_throws "not both" MultilinearMap(
                f!; multiindex = (2, 1), derivatives = (0, 0, 1))
        end

        @testset "degree keyword" begin
            va!(res, args...) = (res .+= reduce((a, b) -> a .* b, args))
            m = MultilinearMap(va!; degree = 3)
            @test m.deg == 3
            @test m.multiindex == (3,)

            ext = MultilinearMap(va!; degree = 3, multiplicity_external = 1)
            @test ext.multiindex == (2,)
            @test ext.deg == 3

            f3!(res, x, y, z) = (res .+= x .* y .* z)
            @test MultilinearMap(f3!; multiindex = (2, 1), degree = 3).deg == 3
            @test_throws "disagrees" MultilinearMap(f3!; multiindex = (2, 1), degree = 4)
            @test_throws "cannot infer" MultilinearMap(va!)
            @test_throws "smaller than multiplicity_external" MultilinearMap(
                va!; degree = 1, multiplicity_external = 2)
        end
    end

    @testset "show" begin
        f!(res, x, y, z) = (res .+= x .* y .* z)
        m = MultilinearMap(f!, (2, 1))

        compact = sprint(show, m)
        @test occursin("MultilinearMap{ORD=2}", compact)
        @test occursin("multiindex=(2, 1)", compact)
        @test occursin("deg=3", compact)
        @test occursin("GroupwiseSymmetric", compact)
        @test !occursin('\n', compact)

        verbose = sprint(show, MIME"text/plain"(), m)
        @test occursin("deg: 3", verbose)
        @test occursin("test_multilinear_maps.jl", verbose)
        @test occursin("f!(res, x^(0), x^(0), x^(1))", verbose)
        @test occursin("symmetry: GroupwiseSymmetric", verbose)

        forcing = MultilinearMap((res, r) -> (res .+= r), (0, 0), 1)
        @test occursin("f!(res, r)", sprint(show, MIME"text/plain"(), forcing))

        @testset "label agrees with symmetry_type" begin
            # `_symmetry_label` mirrors `symmetry_type`, which `MultilinearMaps` is
            # loaded too early to call.  Hold the two in step.
            f2!(res, x, y) = (res .+= x .* y)
            f3!(res, x, y, z) = (res .+= x .* y .* z)
            cases = [(f2!, (1, 1), nothing), (f2!, (2,), nothing), (f2!, (2, 0), false),
                (f3!, (2, 1), nothing), (f3!, (1, 1, 1), nothing), (f2!, (2,), true)]
            for (f!, mi, fa) in cases
                t = MultilinearMap(f!, mi; fully_asymmetric = fa)
                @test startswith(_symmetry_label(mi, fa),
                    string(nameof(typeof(symmetry_type(t)))))
            end
        end
    end
    @testset "MultilinearMap construction guards" begin
        @testset "negative multiindex entry" begin
            f!(res, x) = (res .+= x)
            @test_throws "must be non-negative" MultilinearMap(f!, (-1,))
            @test_throws ArgumentError MultilinearMap(f!, (-1,))
        end

        @testset "nargs mismatch (too few inputs declared)" begin
            # deg would be 2 but f! takes only 1 input beyond res
            f!(res, x) = (res .+= x)
            @test_throws "must accept 3 arguments" MultilinearMap(f!, (2,))
        end

        @testset "nargs mismatch (too many inputs declared)" begin
            f!(res, x, y, z) = (res .+= x .* y .* z)
            @test_throws "must accept 3 arguments" MultilinearMap(f!, (2,))
        end

        @testset "degree < 2 without external state" begin
            f!(res, x) = (res .+= x)
            @test_throws "degree at least 2" MultilinearMap(f!, (1,))
        end

        @testset "negative multiplicity_external" begin
            f!(res, x, y) = (res .+= x .* y)
            @test_throws "multiplicity_external must be non-negative" MultilinearMap(
                f!, (1,), -1)
        end

        @testset "empty multiindex" begin
            f!(res, x, y) = (res .+= x .* y)
            @test_throws "at least one entry" MultilinearMap(f!, ())
        end
    end

    @testset "arity check accepts what it should" begin
        @testset "multiple methods, one of matching arity" begin
            # `f!` may carry several methods; the constructor only requires that one of
            # them can be called with deg + 1 arguments.
            g!(res, x, y) = (res .+= x .* y)
            g!(res, x, y, z) = (res .+= x .* y .* z)
            h! = g!
            m2 = MultilinearMap(h!, (2,))
            @test m2.deg == 2
            m3 = MultilinearMap(h!, (3,))
            @test m3.deg == 3
        end

        @testset "multiple methods, none of matching arity" begin
            g!(res, x, y) = (res .+= x .* y)
            g!(res, x, y, z) = (res .+= x .* y .* z)
            h! = g!
            @test_throws "must accept 5 arguments" MultilinearMap(h!, (4,))
        end

        @testset "multi-method f! survives NDOrderModel construction" begin
            # Regression test for `_term_label`, which used to call `only(methods(f!))`
            # and threw inside the `@info` emitted by `_info_implicit_symmetry`.
            g!(res, x, y) = (res .+= x .* y)
            g!(res, x, y, z) = (res .+= x .* y .* z)
            h! = g!
            term = MultilinearMap(h!, (2,))
            K = Matrix{Float64}(I, 3, 3)
            @test NDOrderModel((K, K), (term,)) isa NDOrderModel
        end

        @testset "concretely annotated arguments" begin
            # `hasmethod(f!, NTuple{n, Any})` is false here, so the method scan decides.
            f!(res::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = (res .+= x .*
                                                                                        y)
            @test MultilinearMap(f!, (2,)).deg == 2
        end

        @testset "callable struct" begin
            m = MultilinearMap(ScaledProduct(2.0), (2,))
            @test m.deg == 2
            res = zeros(3)
            evaluate_term!(res, m, ([1.0, 2.0, 3.0],), nothing)
            @test res ≈ 2.0 .* [1.0, 4.0, 9.0]
        end

        @testset "varargs closure with explicit multiindex" begin
            # The shape produced by MORFESymbolicsExt: arity cannot be introspected, so
            # the caller must state the structure.
            f!(res, args...) = (res .+= reduce((a, b) -> a .* b, args))
            m = MultilinearMap(f!, (2, 1))
            @test m.deg == 3
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
