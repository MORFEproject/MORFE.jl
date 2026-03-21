using Test
using Random
using StaticArrays: SVector

include(joinpath(@__DIR__, "../../src/MORFE.jl"))
using .MORFE.Multiindices
using .MORFE.Polynomials
using .MORFE.Realification
using .MORFE.Realification: _multinomial, _compositions, _reorder_canonical, _realify_term

# ---------- Helper functions for testing ----------

"""
    construct_polynomial(::Type{DensePolynomial}, dict::Dict{Vector{Int}, T}) -> DensePolynomial

Construct a polynomial from a dictionary mapping exponent vectors to coefficients.
The dictionary must be non‑empty. For empty polynomials, use explicit constructors.
"""
function construct_polynomial(::Type{DensePolynomial}, dict::Dict{Vector{Int}, T}) where T
    @assert !isempty(dict) "construct_polynomial does not support empty dictionaries"
    N = length(first(keys(dict)))
    exps_sv = [SVector{N,Int}(exp) for exp in keys(dict)]
    mset = MultiindexSet(exps_sv)                     # sorts and uniques
    exp_to_idx = Dict{SVector{N,Int}, Int}()
    for (i, exp_sv) in enumerate(mset.exponents)
        exp_to_idx[exp_sv] = i
    end
    sample_val = first(values(dict))
    coeffs = [zero(sample_val) for _ in 1:length(mset)]
    for (exp, val) in dict
        exp_sv = SVector{N,Int}(exp)
        coeffs[exp_to_idx[exp_sv]] = val
    end
    return DensePolynomial(coeffs, mset)
end

"""
    polynomial_to_dict(poly::DensePolynomial) -> Dict{Vector{Int}, eltype(poly)}

Convert a polynomial to a dictionary of non‑zero coefficients.
"""
function polynomial_to_dict(poly::DensePolynomial)
    d = Dict{Vector{Int}, eltype(poly)}()
    exps = multiindex_set(poly).exponents
    cs = coeffs(poly)
    for (j, exp_sv) in enumerate(exps)
        if !iszero(cs[j])
            d[collect(exp_sv)] = cs[j]
        end
    end
    return d
end

# ---------- Tests for internal helpers (type‑agnostic) ----------

@testset "multinomial" begin
    @test _multinomial(0, [0]) == 1
    @test _multinomial(5, [5]) == 1
    @test _multinomial(5, [0,5]) == 1
    @test _multinomial(5, [1,4]) == 5
    @test _multinomial(5, [2,3]) == 10
    @test _multinomial(5, [1,1,3]) == 20
    @test _multinomial(6, [2,2,2]) == 90
end

@testset "compositions" begin
    comps = collect(_compositions(2, 2))
    @test length(comps) == 3
    @test Set(comps) == Set([[0,2], [1,1], [2,0]])

    comps3 = collect(_compositions(3, 3))
    @test length(comps3) == 10
    for k in comps3
        @test length(k) == 3
        @test sum(k) == 3
        @test all(>=(0), k)
    end
end

# ---------- Tests for compose_linear on all implemented polynomial types ----------

@testset "compose_linear" begin
    @testset "for $PolyType" for PolyType in [DensePolynomial]
        @testset "linear identity" begin
            f = Dict([1,0] => 1.0, [0,1] => 2.0)
            M = [1 0; 0 1]
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M)
            @test result isa PolyType
            result_dict = polynomial_to_dict(result)
            expected = Dict([1,0] => 1.0, [0,1] => 2.0)
            @test result_dict == expected
        end

        @testset "quadratic example" begin
            f = Dict([2,0] => 3.0, [1,1] => 5.0)
            M = [1 2; 3 4]
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M)
            result_dict = polynomial_to_dict(result)
            expected = Dict([2,0] => 18.0, [1,1] => 62.0, [0,2] => 52.0)
            @test result_dict == expected
        end

        @testset "constant" begin
            f = Dict([0,0] => 7.0)
            M = [1 2; 3 4]
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M)
            result_dict = polynomial_to_dict(result)
            expected = Dict([0,0] => 7.0)
            @test result_dict == expected
        end

        @testset "higher degree" begin
            f = Dict([2,1,0] => 1.0, [0,0,3] => 1.0)
            M = [1 0; 0 1; 1 1]
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M)
            result_dict = polynomial_to_dict(result)
            expected = Dict([3,0] => 1.0, [2,1] => 4.0, [1,2] => 3.0, [0,3] => 1.0)
            @test result_dict == expected
        end

        @testset "vector output" begin
            # Use SVector instead of plain Vector for coefficients
            f = Dict([2,0] => SVector{2,Float64}(1.0, 0.0),
                     [1,1] => SVector{2,Float64}(0.0, 1.0))

            M = [1 2; 3 4]
            p = 2
            # x = z + 2w
            # y = 3z + 4w
            # f(z,w) = (1,0) (z^2 + 4zw + 4w^2) + (0,1) (3z^2 + 10zw + 8w^2) 
            #        = (1,3) z^2 + (4,10) zw + (4,8) w^2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M)
            result_dict = polynomial_to_dict(result)
            @test result_dict[[2,0]] ≈ SVector{2,Float64}(1.0, 3.0)
            @test result_dict[[1,1]] ≈ SVector{2,Float64}(4.0, 10.0)
            @test result_dict[[0,2]] ≈ SVector{2,Float64}(4.0, 8.0)
        end

        @testset "zero matrix" begin
            f = Dict([1,0] => 2.0, [0,1] => 3.0, [0,0] => 5.0)
            M = zeros(2, 2)
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M)
            result_dict = polynomial_to_dict(result)
            @test result_dict[[0,0]] ≈ 5.0
            for (k, v) in result_dict
                if k != [0,0]
                    @test v ≈ 0.0
                end
            end
        end

        @testset "identity with extra variables" begin
            f = Dict([2,0] => 1.0)
            M = [1 0 0; 0 1 0]
            p = 3
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M)
            result_dict = polynomial_to_dict(result)
            @test result_dict[[2,0,0]] ≈ 1.0
            for (k, v) in result_dict
                if k != [2,0,0]
                    @test v ≈ 0.0
                end
            end
        end

        @testset "binomial expansion" begin
            f = Dict([5] => 1.0)
            poly = construct_polynomial(PolyType, f)
            M = [1 1]
            p = 2
            result = compose_linear(poly, M)
            result_dict = polynomial_to_dict(result)
            for k in 0:5
                @test result_dict[[k, 5-k]] ≈ binomial(5, k)
            end
        end
    end
end

# ---------- Tests for realify on all implemented polynomial types ----------

@testset "realify" begin
    @testset "for $PolyType" for PolyType in [DensePolynomial]
        @testset "reorder_canonical - empty" begin
            mset = MultiindexSet(Vector{SVector{3,Int}}())
            poly = zero(DensePolynomial{Float64}, mset)
            conj_map = [2, 1, 3]
            canonical, n, m = _reorder_canonical(poly, conj_map)
            @test nvars(canonical) == 3
            @test n == 1
            @test m == 1
        end

        @testset "reorder_canonical - single pair" begin
            f = Dict([1,1] => 1.0)
            poly = construct_polynomial(PolyType, f)
            conj_map = [2, 1]
            newpoly, n, m = _reorder_canonical(poly, conj_map)
            @test length(newpoly) == 1
            @test n == 1
            @test m == 0
            d = polynomial_to_dict(newpoly)
            @test haskey(d, [1,1])
        end

        @testset "reorder_canonical - two pairs" begin
            f = Dict([2,1,1,0,0] => 2.0)
            poly = construct_polynomial(PolyType, f)
            conj_map = [3, 2, 1, 5, 4]
            newpoly, n, m = _reorder_canonical(poly, conj_map)
            @test n == 2
            @test m == 1
            d = polynomial_to_dict(newpoly)
            @test haskey(d, [2,0,1,0,1])
            @test d[[2,0,1,0,1]] ≈ 2.0
        end

        @testset "realify_term - z*zbar" begin
            exp_vec = SVector(1, 1)
            coeff = 2.0 + 3.0im
            n = 1
            result = _realify_term(exp_vec, coeff, n)
            @test result[[2,0]] ≈ coeff
            @test result[[0,2]] ≈ coeff
        end

        @testset "realify_term - z^2" begin
            exp_vec = SVector(2, 0)
            coeff = 1.0 + 0.0im
            n = 1
            result = _realify_term(exp_vec, coeff, n)
            @test result[[2,0]] ≈ 1.0 + 0.0im
            @test result[[0,2]] ≈ -1.0 + 0.0im
            @test result[[1,1]] ≈ 0.0 + 2.0im
        end

        @testset "realify_term - with real variable" begin
            exp_vec = SVector(1, 1, 2)
            coeff = 2.0 + 0.0im
            n = 1
            result = _realify_term(exp_vec, coeff, n)
            @test result[[2,0,2]] ≈ 2.0 + 0.0im
            @test result[[0,2,2]] ≈ 2.0 + 0.0im
        end

        @testset "realify - simple example from docs" begin
            f = Dict([1,1,0] => 2.0+1.0im, [2,0,0] => 1.0-1.0im, [0,0,0] => 3.0+0.0im)
            poly = construct_polynomial(PolyType, f)
            conj_map = [2, 1, 3]
            result = realify(poly, conj_map)
            @test result isa PolyType
            result_dict = polynomial_to_dict(result)
            @test result_dict[[2,0,0]] ≈ (2.0+1.0im) + (1.0-1.0im)
            @test result_dict[[1,1,0]] ≈ (1.0-1.0im)*2im
            @test result_dict[[0,2,0]] ≈ (2.0+1.0im) + (-1.0+1.0im)
            @test result_dict[[0,0,0]] ≈ 3.0 + 0.0im
        end

        @testset "realify - evaluation consistency" begin
            Random.seed!(42)
            for _ in 1:5
                n = 2
                m = 1
                N = 2n + m
                nterms = 5
                maxdeg = 3

                poly_orig = Dict{Vector{Int}, ComplexF64}()
                for _ in 1:nterms
                    exp = rand(0:maxdeg, N)
                    coeff = randn() + randn() * im
                    poly_orig[exp] = coeff
                end

                conj_map = [2, 1, 4, 3, 5]

                poly_canon, n_pairs, m_real = _reorder_canonical(construct_polynomial(PolyType, poly_orig), conj_map)
                poly_real = realify(construct_polynomial(PolyType, poly_orig), conj_map)

                x = randn(n_pairs)
                y = randn(n_pairs)
                w = randn(m_real)

                z = x + im*y
                z_conj = conj(z)
                vals_canon = vcat(z, z_conj, w)
                val_orig = evaluate(poly_canon, vals_canon)

                vals_real = vcat(x, y, w)
                val_real = evaluate(poly_real, vals_real)

                @test val_orig ≈ val_real atol=1e-10
            end
        end

        @testset "realify - idempotence (real variables only)" begin
            f = Dict([2,0] => 1.0, [1,1] => 2.0, [0,2] => 3.0)
            poly = construct_polynomial(PolyType, f)
            conj_map = [1, 2]
            result = realify(poly, conj_map)
            result_dict = polynomial_to_dict(result)
            @test result_dict[[2,0]] ≈ 1.0
            @test result_dict[[1,1]] ≈ 2.0
            @test result_dict[[0,2]] ≈ 3.0
        end
    end
end

# ---------- Tests for realify_via_linear on all implemented polynomial types ----------

@testset "realify_via_linear" begin
    @testset "for $PolyType" for PolyType in [DensePolynomial]
        @testset "equivalence with realify" begin
            Random.seed!(123)
            for _ in 1:5
                n = 2
                m = 1
                N = 2n + m
                nterms = 5
                maxdeg = 3

                f = Dict{Vector{Int}, ComplexF64}()
                for _ in 1:nterms
                    exp = rand(0:maxdeg, N)
                    coeff = randn() + randn() * im
                    f[exp] = coeff
                end

                conj_map = [2, 1, 4, 3, 5]
                poly = construct_polynomial(PolyType, f)

                res_direct = realify(poly, conj_map)
                res_linear = realify_via_linear(poly, conj_map)

                @test res_direct isa PolyType
                @test res_linear isa PolyType

                x = randn(n)
                y = randn(n)
                w = randn(m)
                vals = vcat(x, y, w)

                val_direct = evaluate(res_direct, vals)
                val_linear = evaluate(res_linear, vals)

                @test val_direct ≈ val_linear atol=1e-10
            end
        end
    end
end

# ---------- Additional tests for DensePolynomial ----------

@testset "DensePolynomial specific" begin
    @testset "basis sharing" begin
        f1 = Dict([1,0] => 1.0, [0,1] => 2.0)
        f2 = Dict([1,0] => 3.0, [0,1] => 4.0)
        p1 = construct_polynomial(DensePolynomial, f1)
        p2 = construct_polynomial(DensePolynomial, f2)
        @test multiindex_set(p1).exponents == multiindex_set(p2).exponents
    end

    @testset "empty polynomial" begin
        mset = MultiindexSet(Vector{SVector{3,Int}}())
        poly = DensePolynomial(Float64[], mset)
        @test length(poly) == 0
        @test nvars(poly) == 3
        @test polynomial_to_dict(poly) == Dict{Vector{Int}, Float64}()
    end
end

println("All tests passed!")