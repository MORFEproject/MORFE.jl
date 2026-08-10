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

# The keyword constructor reports every value it had to default.  Testsets that are not
# about that reporting wrap their calls in `quiet` to keep the output readable; the
# "assumption reporting" testset below asserts the messages themselves.
quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# Text of the single log record `f` is expected to emit.
only_message(f) = string(only(first(Test.collect_test_logs(f))).message)

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
        @testset "no multiindex: shape inferred, order defaults to 2" begin
            f!(res, x, y) = (res .+= x .* y)
            m = quiet(() -> MultilinearMap(f!))
            @test m.deg == 2
            @test m.multiindex == (2, 0)
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

            f1!(res, r) = (res .+= r)
            three_arg = [(f2!, (1,), 1), (f2!, (0,), 2), (f1!, (0, 0), 1)]
            for (f!, mi, me) in three_arg
                pos = MultilinearMap(f!, mi, me)
                kw = MultilinearMap(f!; multiindex = mi, multiplicity_external = me)
                @test typeof(kw) === typeof(pos)
                @test kw.multiindex === pos.multiindex
                @test kw.multiplicity_external === pos.multiplicity_external
                @test kw.deg === pos.deg
            end
        end

        @testset "bare call infers the shape and defaults to ORD = 2" begin
            f!(res, x, y) = (res .+= x .* y)
            m = @test_logs (:info, r"assumed multiindex") MultilinearMap(f!)
            @test m.multiindex === (2, 0)
            # The old shorthand built a UInt8 tuple that the inner constructor silently
            # converted; pin the element type so that cannot come back.
            @test typeof(m.multiindex) === NTuple{2, Int}
            @test m.deg == 2
            @test m.multiplicity_external == 0
            @test m.fully_asymmetric === nothing

            # `order = 1` is the way to get a first-order term, which `FirstOrderModel` needs.
            m1 = @test_logs (:info,) match_mode=:any MultilinearMap(f!; order = 1)
            @test m1.multiindex === (2,)
            @test m1 isa MultilinearMap{1}
        end

        @testset "fully_asymmetric round-trips through every form" begin
            f!(res, x, y) = (res .+= x .* y)
            for fa in (nothing, false, true)
                quiet() do
                    @test MultilinearMap(f!; fully_asymmetric = fa).fully_asymmetric === fa
                    @test MultilinearMap(f!; multiindex = (2,),
                        fully_asymmetric = fa).fully_asymmetric === fa
                end
                @test MultilinearMap(f!, (2,);
                    fully_asymmetric = fa).fully_asymmetric === fa
                @test MultilinearMap(f!, (1,), 1;
                    fully_asymmetric = fa).fully_asymmetric === fa
            end
        end

        @testset "multiindex accepts a vector" begin
            f!(res, x, y, z) = (res .+= x .* y .* z)
            m = MultilinearMap(f!; multiindex = [2, 1], multiplicity_external = 0)
            @test m.multiindex == (2, 1)
            @test m isa MultilinearMap{2}
        end

        @testset "order keyword" begin
            f2!(res, x, y) = (res .+= x .* y)
            f3!(res, x, y, z) = (res .+= x .* y .* z)

            padded = MultilinearMap(
                f2!; multiindex = (2,), order = 3, multiplicity_external = 0)
            @test padded.multiindex == (2, 0, 0)
            @test padded isa MultilinearMap{3}

            inferred = quiet(() -> MultilinearMap(f2!; order = 3))
            @test inferred.multiindex == (2, 0, 0)

            from_degree = quiet(() -> MultilinearMap(f3!; degree = 3, order = 3))
            @test from_degree.multiindex == (3, 0, 0)

            # A mixed internal/external split is never inferred — it would mean guessing
            # f!(res, x, r).  Stating the multiindex is what makes it legal.
            @test_throws "cannot infer how the 2 factors" MultilinearMap(
                f2!; multiplicity_external = 1, order = 2)
            stated = MultilinearMap(
                f2!; multiindex = (1, 0), multiplicity_external = 1)
            @test stated.multiindex == (1, 0)
            @test stated.deg == 2

            @test_throws "never truncate" MultilinearMap(
                f3!; multiindex = (2, 0, 1), order = 2)
            @test MultilinearMap(f2!; multiindex = (2,), order = 1,
                multiplicity_external = 0).multiindex == (2,)
        end

        @testset "order padding is symmetry-neutral" begin
            f!(res, x, y) = (res .+= x .* y)
            short = MultilinearMap(f!, (2,))
            long = MultilinearMap(
                f!; multiindex = (2,), order = 3, multiplicity_external = 0)
            @test typeof(symmetry_type(short)) === typeof(symmetry_type(long))
        end

        @testset "order keyword composes into an NthOrderModel" begin
            # The payoff: no hand-written trailing zeros, and the ORD=3 tuple still
            # unifies at model construction.
            q!(res, x, y) = (res .+= x .* y)
            t!(res, x, y, z) = (res .+= x .* y .* z)
            K = Matrix{Float64}(I, 3, 3)
            terms = (
                MultilinearMap(q!; multiindex = (2,), order = 3,
                    multiplicity_external = 0, fully_asymmetric = false),
                MultilinearMap(t!, (3, 0, 0); fully_asymmetric = false))
            @test NthOrderModel((K, K, K, K), terms) isa NthOrderModel{3}
        end

        @testset "derivatives keyword" begin
            f!(res, x, y, z) = (res .+= x .* y .* z)
            m = MultilinearMap(f!; derivatives = (0, 0, 1), multiplicity_external = 0,
                order = 2)
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

            @test quiet(() -> MultilinearMap(f!; derivatives = (0, 0, 1),
                order = 4)).multiindex == (2, 1, 0, 0)
            # `order` defaults to max(2, length(base)), so a three-order list is not
            # truncated by the default.
            @test quiet(() -> MultilinearMap(f!;
                derivatives = (0, 2, 2))).multiindex == (1, 0, 2)

            @test_throws "non-decreasing" MultilinearMap(f!; derivatives = (1, 0, 0))
            @test_throws "must be non-negative" MultilinearMap(
                f!; derivatives = (-1, 0, 0))
            @test_throws "not both" MultilinearMap(
                f!; multiindex = (2, 1), derivatives = (0, 0, 1))
        end

        @testset "degree keyword" begin
            va!(res, args...) = (res .+= reduce((a, b) -> a .* b, args))
            m = quiet(() -> MultilinearMap(va!; degree = 3))
            @test m.deg == 3
            @test m.multiindex == (3, 0)      # order defaults to 2

            # An inferred mixed split is refused; stating the multiindex allows it.
            @test_throws "cannot infer how the 3 factors" MultilinearMap(
                va!; degree = 3, multiplicity_external = 1)
            ext = MultilinearMap(va!; multiindex = (2,), multiplicity_external = 1)
            @test ext.multiindex == (2,)
            @test ext.deg == 3

            f3!(res, x, y, z) = (res .+= x .* y .* z)
            @test MultilinearMap(f3!; multiindex = (2, 1), degree = 3,
                multiplicity_external = 0).deg == 3
            @test_throws "disagrees" MultilinearMap(f3!; multiindex = (2, 1), degree = 4)
            @test_throws "cannot infer the degree" MultilinearMap(va!)
            @test_throws "smaller than multiplicity_external" MultilinearMap(
                va!; degree = 1, multiplicity_external = 2)
        end
    end

    # Fixtures for the resolution tables: 0, 1, 2 and 3 factors, plus a varargs closure
    # whose arity cannot be introspected.
    z!(res) = nothing
    r1!(res, r) = (res .+= r)
    q2!(res, x, y) = (res .+= x .* y)
    c3!(res, x, y, z) = (res .+= x .* y .* z)
    vararg!(res, args...) = (res .+= reduce((a, b) -> a .* b, args))

    @testset "forcing-term default" begin
        @testset "bare call on a one-factor f!" begin
            m = @test_logs (:info, r"pure external forcing term") MultilinearMap(r1!)
            @test m.multiindex == (0, 0)
            @test m.multiplicity_external == 1
            @test m.deg == 1
            @test m isa MultilinearMap{2}
        end

        @testset "equals the positional spelling field for field" begin
            m = quiet(() -> MultilinearMap(r1!))
            pos = MultilinearMap(r1!, (0, 0), 1)
            @test typeof(m) === typeof(pos)
            @test m.f! === pos.f!
            @test m.multiindex === pos.multiindex
            @test m.multiplicity_external === pos.multiplicity_external
            @test m.deg === pos.deg
            @test m.fully_asymmetric === pos.fully_asymmetric
        end

        @testset "multiplicity_external stated: only the rest is assumed" begin
            m = @test_logs (:info, r"assumed multiindex") MultilinearMap(
                r1!; multiplicity_external = 1)
            @test m.multiindex == (0, 0)
            @test m.deg == 1
        end

        @testset "purely external quadratic term" begin
            m = @test_logs (:info, r"assumed multiindex") MultilinearMap(
                q2!; multiplicity_external = 2)
            @test m.multiindex == (0, 0)
            @test m.multiplicity_external == 2
            @test m.deg == 2
        end

        @testset "degree = 1 triggers the same reading" begin
            m = @test_logs (:info, r"pure external forcing term") MultilinearMap(
                vararg!; degree = 1)
            @test m.multiindex == (0, 0)
            @test m.multiplicity_external == 1
        end

        @testset "silenced by stating the shape" begin
            @test_logs MultilinearMap(r1!; multiindex = (0,), multiplicity_external = 1)
            @test_logs MultilinearMap(
                r1!; multiindex = (0, 0, 0), multiplicity_external = 1)
            @test quiet(() -> MultilinearMap(r1!; multiplicity_external = 1,
                order = 3)).multiindex == (0, 0, 0)
            @test quiet(() -> MultilinearMap(r1!; multiplicity_external = 1,
                order = 1)).multiindex == (0,)
        end

        @testset "a stated linear term is still refused" begin
            # The shape was stated, so there is nothing to assume: multiindex = (1,) with
            # no external factors is unambiguously a linear state term.
            @test_throws "degree at least 2" MultilinearMap(r1!; multiindex = (1,))
            @test_throws "degree at least 2" MultilinearMap(r1!, (1,))
        end

        @testset "evaluates and composes into a forced model" begin
            n = 3
            m = quiet(() -> MultilinearMap(r1!))
            res = zeros(ComplexF64, n)
            evaluate_term!(res, m, (zeros(n), zeros(n)), ComplexF64[2.0, 0.0, 0.0])
            @test res ≈ ComplexF64[2.0, 0.0, 0.0]

            K = Matrix{Float64}(I, n, n)
            ext = MORFE.ExternalSystems.ExternalSystem((0.0 + 1.0im, 0.0 - 1.0im))
            @test NthOrderModel((K, K, K), (m,), ext) isa NthOrderModel
        end
    end

    @testset "mixed internal/external splits are never inferred" begin
        @test_throws "cannot infer how the 2 factors" MultilinearMap(
            q2!; multiplicity_external = 1)
        @test_throws "cannot infer how the 3 factors" MultilinearMap(
            c3!; multiplicity_external = 1)
        # ...but they are perfectly legal once stated.
        @test_logs MultilinearMap(q2!; multiindex = (1, 0), multiplicity_external = 1)
        @test MultilinearMap(q2!, (1, 0), 1).deg == 2
    end

    @testset "assumption reporting" begin
        @testset "order defaults to 2 for every inferred shape" begin
            @test quiet(() -> MultilinearMap(q2!)).multiindex == (2, 0)
            @test quiet(() -> MultilinearMap(c3!)).multiindex == (3, 0)
            @test quiet(() -> MultilinearMap(vararg!; degree = 3)).multiindex == (3, 0)
            @test quiet(() -> MultilinearMap(q2!; order = 1)).multiindex == (2,)
            @test quiet(() -> MultilinearMap(q2!; order = 3)).multiindex == (2, 0, 0)
        end

        @testset "compact message names exactly the assumed fields" begin
            msg = only_message(() -> MultilinearMap(q2!))
            @test occursin("multiindex = (2, 0)", msg)
            @test occursin("order = 2", msg)
            @test occursin("multiplicity_external = 0", msg)
            @test occursin("from the arity of f!", msg)
            @test !occursin("LINEAR", msg)
            @test !occursin("fully_asymmetric", msg)

            # `degree` is named as the source when it, not the arity, fixed the degree.
            @test occursin("from `degree`",
                only_message(() -> MultilinearMap(vararg!; degree = 3)))

            # A field that was actually stated must not be listed as assumed.
            omsg = only_message(() -> MultilinearMap(q2!; order = 1))
            @test !occursin("order =", omsg)
            @test occursin("multiplicity_external = 0", omsg)
        end

        @testset "forcing message explains the linear-term risk" begin
            msg = only_message(() -> MultilinearMap(r1!))
            @test occursin("LINEAR", msg)
            @test occursin("f!(res, r)", msg)
            @test occursin("linear_terms", msg)
            @test !occursin("fully_asymmetric", msg)
            # No inference from argument names: the factor is called `r` here, but the
            # message must never quote it.
            @test !occursin("named", msg)
        end

        @testset "every form the messages call silent really is silent" begin
            @test_logs MultilinearMap(r1!; multiindex = (0, 0), multiplicity_external = 1)
            @test_logs MultilinearMap(r1!, (0, 0), 1)
            @test_logs MultilinearMap(q2!; multiindex = (2, 0), multiplicity_external = 0)
        end

        @testset "positional forms never report" begin
            @test_logs MultilinearMap(q2!, (2, 0))
            @test_logs MultilinearMap(q2!, (2,))
            @test_logs MultilinearMap(q2!, (2, 0); fully_asymmetric = false)
            @test_logs MultilinearMap(r1!, (0, 0), 1)
            @test_logs MultilinearMap(r1!, (0, 0), 1; fully_asymmetric = true)
        end

        @testset "nothing is reported when construction fails" begin
            # `_info_assumed` runs after the term validates, so a failed build never
            # announces an assumption that did not survive.
            logs, _ = Test.collect_test_logs() do
                try
                    MultilinearMap(z!)
                catch
                end
            end
            @test isempty(logs)
            @test_throws "degree at least 2" MultilinearMap(z!)
        end

        @testset "fully_asymmetric is still reported where it matters" begin
            # Excluded from the constructor report, but `_info_implicit_symmetry` at
            # NthOrderModel still catches it — moved, not lost.
            K = Matrix{Float64}(I, 3, 3)
            term = MultilinearMap(q2!, (2, 0))
            @test_logs (:info, r"did not set `fully_asymmetric`") NthOrderModel(
                (K, K, K), (term,))
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

        @testset "multi-method f! survives NthOrderModel construction" begin
            # Regression test for `_term_label`, which used to call `only(methods(f!))`
            # and threw inside the `@info` emitted by `_info_implicit_symmetry`.
            g!(res, x, y) = (res .+= x .* y)
            g!(res, x, y, z) = (res .+= x .* y .* z)
            h! = g!
            term = MultilinearMap(h!, (2,))
            K = Matrix{Float64}(I, 3, 3)
            # `fully_asymmetric` is left unset on purpose: that is what makes
            # `_info_implicit_symmetry` — and hence `_term_label` — run at all.
            model = @test_logs (:info,) match_mode=:any NthOrderModel((K, K), (term,))
            @test model isa NthOrderModel
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
