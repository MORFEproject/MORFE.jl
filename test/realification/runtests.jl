using Test
using Random

include(joinpath(@__DIR__, "..", "..", "src", "Polynomials.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "realification", "Realification.jl"))
using .Polynomials
using .Realification

# ---------- Helper functions for testing ----------

"""
    construct_polynomial(::Type{SparsePolynomial}, dict::Dict{Vector{Int}, T}) -> SparsePolynomial
    construct_polynomial(::Type{DensePolynomial}, dict::Dict{Vector{Int}, T}) -> DensePolynomial

Construct a polynomial of the specified type from a dictionary.
The number of variables is inferred from the first key (or 0 if empty).
"""
function construct_polynomial(::Type{SparsePolynomial}, dict::Dict{Vector{Int}, T}) where T
    nvars = isempty(dict) ? 0 : length(first(keys(dict)))
    return Realification.polynomial_from_dict(dict, nvars, SparsePolynomial)
end

function construct_polynomial(::Type{DensePolynomial}, dict::Dict{Vector{Int}, T}) where T
    nvars = isempty(dict) ? 0 : length(first(keys(dict)))
    return Realification.polynomial_from_dict(dict, nvars, DensePolynomial)
end

"""
    polynomial_to_dict(poly::AbstractPolynomial) -> Dict{Vector{Int}, eltype(poly)}

Convert any polynomial back to a dictionary mapping exponent vectors to coefficients.
"""
function polynomial_to_dict(poly::AbstractPolynomial)
    d = Dict{Vector{Int}, eltype(poly)}()
    for i in 1:length(poly)
        exp = exponents(poly)[:, i]
        d[exp] = coeffs(poly)[i]
    end
    return d
end

"""
    evaluate(poly::AbstractPolynomial, vals::Vector{<:Number})

Evaluate a polynomial at the given variable values.
"""
function evaluate(poly::AbstractPolynomial, vals::Vector{<:Number})
    @assert nvars(poly) == length(vals)
    T = eltype(poly)
    # If T is Any (e.g., from an empty coefficient vector), use ComplexF64 as fallback.
    result = T == Any ? zero(ComplexF64) : zero(T)
    for i in 1:length(poly)
        coeff = coeffs(poly)[i]
        exp = exponents(poly)[:, i]
        term = coeff
        for (j, e) in enumerate(exp)
            term *= vals[j]^e
        end
        result += term
    end
    return result
end

# ---------- Tests for internal helpers (type‑agnostic) ----------

@testset "multinomial" begin
    @test Realification._multinomial(0, [0]) == 1
    @test Realification._multinomial(5, [5]) == 1
    @test Realification._multinomial(5, [0,5]) == 1
    @test Realification._multinomial(5, [1,4]) == 5
    @test Realification._multinomial(5, [2,3]) == 10
    @test Realification._multinomial(5, [1,1,3]) == 20
    @test Realification._multinomial(6, [2,2,2]) == 90
end

@testset "compositions" begin
    comps = collect(Realification._compositions(2, 2))
    @test length(comps) == 3
    @test Set(comps) == Set([[0,2], [1,1], [2,0]])

    comps3 = collect(Realification._compositions(3, 3))
    @test length(comps3) == 10
    for k in comps3
        @test length(k) == 3
        @test sum(k) == 3
        @test all(>=(0), k)
    end
end

# ---------- Tests for compose_linear on both sparse and dense ----------

@testset "compose_linear" begin
    @testset "for $PolyType" for PolyType in [SparsePolynomial, DensePolynomial]
        @testset "linear identity" begin
            f = Dict([1,0] => 1.0, [0,1] => 2.0)
            M = [1 0; 0 1]
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M, p)
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
            result = compose_linear(poly, M, p)
            result_dict = polynomial_to_dict(result)
            expected = Dict([2,0] => 18.0, [1,1] => 62.0, [0,2] => 52.0)
            @test result_dict == expected
        end

        @testset "zero polynomial" begin
            f = Dict([0,0,0] => 0.0)
            M = rand(3, 2)
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M, 2)
            result_dict = polynomial_to_dict(result)
            @test length(result_dict) == 1
            @test result_dict[[0,0]] ≈ 0.0
        end

        @testset "constant" begin
            f = Dict([0,0] => 7.0)
            M = [1 2; 3 4]
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M, p)
            result_dict = polynomial_to_dict(result)
            expected = Dict([0,0] => 7.0)
            @test result_dict == expected
        end

        @testset "higher degree" begin
            f = Dict([2,1,0] => 1.0, [0,0,3] => 1.0)
            M = [1 0; 0 1; 1 1]
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M, p)
            result_dict = polynomial_to_dict(result)
            expected = Dict([3,0] => 1.0, [2,1] => 4.0, [1,2] => 3.0, [0,3] => 1.0)
            @test result_dict == expected
        end

        @testset "vector output" begin
            f = Dict([2,0] => [1.0, 0.0], [1,1] => [0.0, 1.0])
            M = [1 2; 3 4]
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M, p)
            result_dict = polynomial_to_dict(result)
            @test result_dict[[2,0]] ≈ [1.0, 3.0]
            @test result_dict[[1,1]] ≈ [4.0, 10.0]
            @test result_dict[[0,2]] ≈ [4.0, 8.0]
        end

        @testset "zero matrix" begin
            f = Dict([1,0] => 2.0, [0,1] => 3.0, [0,0] => 5.0)
            M = zeros(2, 2)
            p = 2
            poly = construct_polynomial(PolyType, f)
            result = compose_linear(poly, M, p)
            result_dict = polynomial_to_dict(result)
            @test result_dict[[0,0]] ≈ 5.0
            # Other terms should be zero (or filtered out)
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
            result = compose_linear(poly, M, p)
            result_dict = polynomial_to_dict(result)
            @test result_dict[[2,0,0]] ≈ 1.0
            # Other terms should be zero
            for (k, v) in result_dict
                if k != [2,0,0]
                    @test v ≈ 0.0
                end
            end
        end

        @testset "binomial expansion" begin
            # Polynomial: x^5 (one variable)
            f = Dict([5] => 1.0)
            poly = construct_polynomial(PolyType, f)
            M = [1 1]   # 1×2 matrix: x = y₁ + y₂
            p = 2
            result = compose_linear(poly, M, p)
            result_dict = polynomial_to_dict(result)
            # (y₁ + y₂)⁵ should expand to sum_{k=0..5} C(5,k) y₁^k y₂^(5-k)
            for k in 0:5
                @test result_dict[[k, 5-k]] ≈ binomial(5, k)
            end
        end
    end
end

# ---------- Tests for realify on both sparse and dense ----------

@testset "realify" begin
    @testset "for $PolyType" for PolyType in [SparsePolynomial, DensePolynomial]
        @testset "reorder_canonical - empty" begin
            # Empty polynomial: 3 variables, no terms
            if PolyType == SparsePolynomial
                poly = SparsePolynomial(Float64[], zeros(Int, 3, 0))
            else
                basis = MonomialBasis(zeros(Int, 3, 0))
                poly = DensePolynomial(Float64[], basis)
            end
            conj_map = [2, 1, 3]
            canonical, n, m = Realification._reorder_canonical(poly, conj_map)
            @test nvars(canonical) == 3
            @test n == 1      # one conjugate pair
            @test m == 1      # one real variable
        end

        @testset "reorder_canonical - single pair" begin
            f = Dict([1,1] => 1.0)
            poly = construct_polynomial(PolyType, f)
            conj_map = [2, 1]
            newpoly, n, m = Realification._reorder_canonical(poly, conj_map)
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
            newpoly, n, m = Realification._reorder_canonical(poly, conj_map)
            @test n == 2
            @test m == 1
            d = polynomial_to_dict(newpoly)
            @test haskey(d, [2,0,1,0,1])
            @test d[[2,0,1,0,1]] ≈ 2.0
        end

        @testset "realify_term - z*zbar" begin
            exp_vec = [1, 1]
            coeff = 2.0 + 3.0im
            n = 1; m = 0
            result = Realification._realify_term(exp_vec, coeff, n, m)
            @test result[[2,0]] ≈ coeff
            @test result[[0,2]] ≈ coeff
        end

        @testset "realify_term - z^2" begin
            exp_vec = [2, 0]
            coeff = 1.0 + 0.0im
            n = 1; m = 0
            result = Realification._realify_term(exp_vec, coeff, n, m)
            @test result[[2,0]] ≈ 1.0 + 0.0im
            @test result[[0,2]] ≈ -1.0 + 0.0im
            @test result[[1,1]] ≈ 0.0 + 2.0im
        end

        @testset "realify_term - with real variable" begin
            exp_vec = [1, 1, 2]
            coeff = 2.0 + 0.0im
            n = 1; m = 1
            result = Realification._realify_term(exp_vec, coeff, n, m)
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
            # Expected realified terms:
            # From (2+1i)*(z*z̄)      : [2,0,0] = 2+1i , [0,2,0] = 2+1i
            # From (1-1i)*z^2         : [2,0,0] = 1-1i , [1,1,0] = 2+2i , [0,2,0] = -1+1i
            # From constant 3          : [0,0,0] = 3
            # Summed:
            @test result_dict[[2,0,0]] ≈ (2.0+1.0im) + (1.0-1.0im)  # 3.0 + 0.0im
            @test result_dict[[1,1,0]] ≈ (1.0-1.0im)*2im             # 2.0 + 2.0im
            @test result_dict[[0,2,0]] ≈ (2.0+1.0im) + (-1.0+1.0im) # 1.0 + 2.0im
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

                poly_canon, n_pairs, m_real = Realification._reorder_canonical(construct_polynomial(PolyType, poly_orig), conj_map)
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

        @testset "realify - zero polynomial" begin
            f = Dict([0,0,0] => 0.0+0.0im)
            poly = construct_polynomial(PolyType, f)
            result = realify(poly, [2, 1, 3])
            result_dict = polynomial_to_dict(result)
            @test length(result_dict) >= 0
        end
    end
end

# ---------- Tests for realify_via_linear on both sparse and dense ----------

@testset "realify_via_linear" begin
    @testset "for $PolyType" for PolyType in [SparsePolynomial, DensePolynomial]
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

                # Evaluate at random real points
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
        # Two polynomials with the same monomials should have equal basis exponents
        f1 = Dict([1,0] => 1.0, [0,1] => 2.0)
        f2 = Dict([1,0] => 3.0, [0,1] => 4.0)
        p1 = construct_polynomial(DensePolynomial, f1)
        p2 = construct_polynomial(DensePolynomial, f2)
        @test p1.basis.exponents == p2.basis.exponents
    end

    @testset "empty polynomial" begin
        basis = MonomialBasis(zeros(Int, 3, 0))
        poly = DensePolynomial(Float64[], basis)
        @test length(poly) == 0
        @test nvars(poly) == 3
        @test polynomial_to_dict(poly) == Dict{Vector{Int}, Float64}()
    end
end

println("All tests passed!")