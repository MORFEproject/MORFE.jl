# test/runtests.jl (or a standalone test file for Polynomials module)

using Test
using StaticArrays: SVector

include(joinpath(@__DIR__, "../../src/MORFE.jl"))    # adjust path as needed
using .MORFE.Multiindices
using .MORFE.Multiindices: monomial_rank
using .MORFE.Polynomials

# ----------------------------
# Helper: check that two polynomials are approximately equal (for floats)
function poly_approx_equal(p1::DensePolynomial, p2::DensePolynomial; rtol=1e-12, atol=1e-12)
    p1.coefficients ≈ p2.coefficients && p1.multiindex_set == p2.multiindex_set
end

# ----------------------------
@testset "Polynomials module" begin

    # 1. Basic construction from dictionary
    @testset "Constructors from Dict" begin
        dict = Dict(
            [0,0] => 0.0,
            [2,0] => 3.1,
            [1,1] => 2.5,
            [0,2] => 1.3,
            [1,0] => 5.8,
            [0,1] => -1.9
        )
        p_dense = DensePolynomial(dict)

        # Expected multiindex set for 2 vars, max degree 2 (Grlex order)
        expected_set = all_multiindices_up_to(2, 2)
        @test p_dense.multiindex_set.exponents == expected_set.exponents

        # Dense coefficients should align with the set order
        exps = [collect(e) for e in expected_set.exponents]
        expected_dense = Float64[get(dict, e, nothing) for e in exps]
        @test p_dense.coefficients == expected_dense
    end

    @testset "Constructors from coefficient vector and MultiindexSet" begin
        set = all_multiindices_up_to(2, 2)
        # Build coefficient vector: zeros everywhere except at [2,0] and [0,1]
        coeffs = zeros(Int, length(set))
        idx_20 = find_in_set(set, [2,0])
        idx_01 = find_in_set(set, [0,1])
        coeffs[idx_20] = 5
        coeffs[idx_01] = -2
        p_dense = DensePolynomial(coeffs, set)

        @test p_dense.coefficients[idx_20] == 5
        @test p_dense.coefficients[idx_01] == -2
        @test all(i -> !(i in (idx_20, idx_01)) ? p_dense.coefficients[i] == 0 : true, 1:length(set))
    end

    @testset "polynomial_from_pairs" begin
        pairs = [[2,0] => 1.5, [0,1] => -3.0]
        p_dense = polynomial_from_pairs(DensePolynomial{Float64}, pairs)

        # The set contains exactly the given exponents, sorted in Grlex order.
        expected_set = MultiindexSet([[2,0], [0,1]])   # Grlex: [0,1] (deg1) then [2,0] (deg2)
        @test p_dense.multiindex_set.exponents == expected_set.exponents

        # Dense coeffs in set order: [0,1] first, then [2,0]
        @test p_dense.coefficients == [-3.0, 1.5]
    end

    @testset "Empty polynomial constructors" begin
        # From empty dict
        p_dense_empty = DensePolynomial(Dict{Vector{Int}, Float64}())
        @test length(p_dense_empty) == 0
        @test length(p_dense_empty.multiindex_set.exponents) == 0

        # From zero coefficient vector with existing set
        set = all_multiindices_up_to(2, 3)
        p_dense = DensePolynomial(zeros(Float64, length(set)), set)
        @test length(p_dense.coefficients) == length(set)
        @test all(iszero, p_dense.coefficients)
    end

    # 2. Accessors
    @testset "Accessors" begin
        set = all_multiindices_up_to(2, 2)
        # Build a polynomial on the full set with only two non-zero coefficients
        c = zeros(Int, length(set))
        idx_10 = find_in_set(set, [1,0])
        idx_00 = find_in_set(set, [0,0])
        c[idx_10] = 4
        c[idx_00] = 1
        p_dense = DensePolynomial(c, set)

        @test coefficients(p_dense) == p_dense.coefficients
        @test multiindex_set(p_dense) == set
        @test nvars(p_dense) == 2
        @test length(p_dense) == length(set)
    end

    # 3. Term lookup
    @testset "Term lookup" begin
        set = all_multiindices_up_to(3, 2)
        # Build polynomial with coefficients on a subset of exponents, zeros elsewhere
        c = zeros(Float64, length(set))
        idx_100 = find_in_set(set, [1,0,0])
        idx_011 = find_in_set(set, [0,1,1])
        idx_000 = find_in_set(set, [0,0,0])
        c[idx_100] = 2.5
        c[idx_011] = -1.2
        c[idx_000] = 7.0
        p_dense = DensePolynomial(c, set)

        # find_in_multiindex_set
        @test find_in_multiindex_set(p_dense, [1,0,0]) == idx_100
        @test isnothing(find_in_multiindex_set(p_dense, [5,0,0]))

        # has_term (any exponent in the set counts, even with zero coefficient)
        @test has_term(p_dense, [1,0,0]) == true
        @test has_term(p_dense, [2,0,0]) == true   # exponent exists in set, though coefficient is zero
        @test has_term(p_dense, [5,0,0]) == false  # outside set

        # coefficient
        @test coefficient(p_dense, [1,0,0]) == 2.5
        @test coefficient(p_dense, [0,0,0]) == 7.0
        @test coefficient(p_dense, [2,0,0]) == 0.0

        # find_term (returns index if exponent in set, even if coefficient zero)
        @test find_term(p_dense, [1,0,0]) == idx_100
        @test find_term(p_dense, [0,0,2]) == find_in_set(set, [0,0,2])  # exists in set, coefficient zero
        @test find_term(p_dense, [5,0,0]) === nothing
    end

    # 4. Zero polynomial constructors
    @testset "Zero polynomials" begin
        set = all_multiindices_up_to(2, 2)   # Grlex order

        zero_dense = zero(DensePolynomial{Int}, set)

        # New struct: DensePolynomial{T, NVAR, N, A} — check T and NVAR
        @test zero_dense isa DensePolynomial{Int}
        @test length(zero_dense) == 6
        @test length(zero_dense.multiindex_set.exponents) == 6
        @test nvars(zero_dense) == 2

        # Using zero with an existing set
        zero_dense_set = zero(DensePolynomial{Int}, set)

        @test zero_dense_set isa DensePolynomial{Int}
        @test length(zero_dense_set.coefficients) == length(set)
        @test all(iszero, zero_dense_set.coefficients)
        @test zero_dense_set.multiindex_set == set

        # Vector-valued zero: zero(DensePolynomial{T}, coeff_shape, mset)
        zero_vec = zero(DensePolynomial{ComplexF64}, (3,), set)
        @test zero_vec isa DensePolynomial{ComplexF64}
        @test coeff_shape(zero_vec) == (3,)
        @test nmonomials(zero_vec) == length(set)
        @test all(iszero, zero_vec.coefficients)
    end

    # 5. Error conditions
    @testset "Error conditions" begin
        set = all_multiindices_up_to(2, 2)

        # polynomial_from_pairs with mismatched exponent lengths (should error in MultiindexSet constructor)
        @test_throws Exception polynomial_from_pairs(DensePolynomial{Int}, [[1,0]=>1, [1,0,0]=>2])
    end

    # 6. Different element types
    @testset "Element types" begin
        for T in [Int, Float64, Rational{Int}]
            dict = Dict([1,0]=>T(2), [0,1]=>T(-3))
            p_dense = DensePolynomial(dict)
            @test eltype(p_dense) == T
            @test coefficient(p_dense, [1,0]) == T(2)
        end
    end

    # 7. Test that indices are correctly sorted according to the set's order (Grlex)
    @testset "Indices sorted" begin
        set = all_multiindices_up_to(3, 3)   # Grlex order
        expected_indices = [1, 4, 5, 6]
        @test expected_indices == [
            find_in_set(set, [0,0,0]),
            find_in_set(set, [0,0,1]),
            find_in_set(set, [2,0,0]),
            find_in_set(set, [1,1,0])
        ]
    end

    # 8. Test that polynomial_from_pairs with order works as expected (only Grlex now)
    @testset "polynomial_from_pairs Grlex ordering" begin
        pairs = [[1,0]=>1, [0,1]=>2, [0,0]=>3]
        p_grlex = polynomial_from_pairs(DensePolynomial{Int}, pairs)

        # Grlex order: first degree 0: [0,0], then degree 1: [1,0] then [0,1] (lex within degree)
        expected_grlex_set = MultiindexSet([[0,0], [1,0], [0,1]])
        @test p_grlex.multiindex_set.exponents == expected_grlex_set.exponents
        @test p_grlex.coefficients == [3, 1, 2]
    end

    # 9. Test monomial_rank (exported from Multiindices)
    @testset "monomial_rank" begin
        # In Grlex, rank of [1,0] in 2 vars with max degree 2:
        # All exponents up to degree 2 in Grlex: deg0: [0,0] (rank1), deg1: [1,0] (rank2), [0,1] (rank3), deg2: [2,0] (rank4), [1,1] (rank5), [0,2] (rank6)
        @test monomial_rank([1,0], 2, 2) == 2
        @test monomial_rank([0,1], 2, 2) == 3
        @test monomial_rank([2,0], 2, 2) == 4
        @test monomial_rank([0,2], 2, 2) == 6
    end

    # 10. coeff_shape and nmonomials accessors
    @testset "coeff_shape and nmonomials" begin
        set = all_multiindices_up_to(2, 3)
        L = length(set)

        p_scalar = zero(DensePolynomial{Float64}, set)
        @test coeff_shape(p_scalar) == ()
        @test nmonomials(p_scalar) == L

        p_vec = zero(DensePolynomial{ComplexF64}, (4,), set)
        @test coeff_shape(p_vec) == (4,)
        @test nmonomials(p_vec) == L
    end

    # 11. evaluate scalar and vector polynomials
    @testset "evaluate" begin
        # Scalar: p(x,y) = 3x² + 2xy - y
        dict = Dict([2,0]=>3.0, [1,1]=>2.0, [0,1]=>-1.0)
        p = DensePolynomial(dict)
        @test evaluate(p, [1.0, 1.0]) ≈ 3.0 + 2.0 - 1.0
        @test evaluate(p, [2.0, 0.0]) ≈ 12.0

        # Vector: coefficients are (K × L) matrix
set = all_multiindices_up_to(2, 1)
        p_vec = zero(DensePolynomial{Float64}, (2,), set)
        # set monomials: [0,0]=1, [1,0]=x, [0,1]=y  (Grlex)
        idx_10 = find_in_multiindex_set(p_vec, [1,0])
        idx_01 = find_in_multiindex_set(p_vec, [0,1])
        p_vec.coefficients[:, idx_10] = [1.0, 0.0]   # first component = x
        p_vec.coefficients[:, idx_01] = [0.0, 1.0]   # second component = y
        result = evaluate(p_vec, [3.0, 5.0])
        @test result ≈ [3.0, 5.0]
        @test evaluate(p_vec, [3.0, 5.0], 1) ≈ 3.0
        @test evaluate(p_vec, [3.0, 5.0], 2) ≈ 5.0
    end

    # 12. extract_component
    @testset "extract_component" begin
set = all_multiindices_up_to(2, 2)
        p = DensePolynomial([randn(SVector{3, Float64}) for _ in 1:length(set)], set)
        for k in 1:3
            p_k = extract_component(p, k)
            @test p_k isa DensePolynomial{Float64}
            @test nmonomials(p_k) == length(set)
            @test p_k.coefficients == p.coefficients[k, :]
        end
    end
end

println("All tests passed.")
