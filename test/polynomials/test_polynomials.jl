# test/runtests.jl (or a standalone test file for Polynomials module)

using Test
include(joinpath(@__DIR__, "../../src/Polynomials.jl"))    # adjust path as needed

using .Polynomials

# ----------------------------
# Helper: check that two polynomials are approximately equal (for floats)
function poly_approx_equal(p1::DensePolynomial, p2::DensePolynomial; rtol=1e-12, atol=1e-12)
    p1.coeffs ≈ p2.coeffs && p1.multiindex_set == p2.multiindex_set
end

# ----------------------------
@testset "Polynomials module" begin

    # 1. Basic construction from dictionary and order type
    @testset "Constructors from Dict and order" begin
        # Lex order
        dict_lex = Dict(
            [0,0] => 0.0,
            [2,0] => 3.1,
            [1,1] => 2.5,
            [0,2] => 1.3,
            [1,0] => 5.8,
            [0,1] => -1.9
        )
        p_dense_lex = DensePolynomial(dict_lex, Lex)

        # Expected multiindex set for 2 vars, max degree 2, Lex order
        expected_set_lex = all_multiindices_up_to(2, 2, Lex())
        @test p_dense_lex.multiindex_set.exponents == expected_set_lex.exponents

        # Dense coefficients should be: [coeff for [2,0], [1,1], [0,2], [1,0], [0,1], [0,0]? Wait, Lex order for max degree 2 includes all degrees ≤2, not just deg 2.
        # Lex order: exponents sorted descending lex: (2,0), (2,? no, first component dominates)
        # Actually Lex order: [2,0], [1,1], [1,0], [0,2], [0,1], [0,0]? Let's generate:
        exps_lex = [collect(e) for e in eachcol(expected_set_lex.exponents)]
        # We can manually compute expected dense coeffs: for each exponent in set, take coeff from dict if present else 0.
        expected_dense = Float64[get(dict_lex, e, 0) for e in exps_lex]   # using Float64 because dict values are Int but we want generic
        @test p_dense_lex.coeffs == expected_dense
    end

    @testset "Constructors from Dict and existing MultiindexSet" begin
        set = all_multiindices_up_to(2, 2, Grlex())
        dict = Dict([2,0] => 5, [0,1] => -2)
        p_dense = DensePolynomial(dict, set)

        # Check that coefficients are placed correctly
        idx_20 = find_in_set(set, [2,0]) # 2
        idx_01 = find_in_set(set, [0,1]) # 1
        @test p_dense.coeffs[idx_20] == 5
        @test p_dense.coeffs[idx_01] == -2
        @test all(i -> !(i in (idx_20, idx_01)) ? p_dense.coeffs[i] == 0 : true, 1:length(set))

        # Error: dict contains exponent not in set
        @test_throws ErrorException DensePolynomial(Dict([3,0] => 1), set)
    end

    @testset "polynomial_from_pairs" begin
        pairs = [[2,0] => 1.5, [0,1] => -3.0]
        p_dense = polynomial_from_pairs(DensePolynomial{Float64,Grlex}, pairs)

        set = all_multiindices_up_to(2, 2, Grlex())   # max degree inferred from exponents?
        # Actually polynomial_from_pairs creates a set containing exactly the given exponents (sorted).
        expected_set = MultiindexSet([[2,0], [0,1]], Grlex())  # Grlex order: [0,1] deg=1 before [2,0] deg=2? Wait: Grlex sorts by total degree first, so [0,1] (deg1) comes before [2,0] (deg2). Then within same degree? [0,1] and [2,0] are different degrees, so order: [0,1] then [2,0].
        # Let's compute: deg1 monomials: [1,0] and [0,1] in lex order? Actually for Grlex, within same degree it's lex, so [1,0] then [0,1]. But we only have [0,1] so it's fine.
        @test p_dense.multiindex_set.exponents == expected_set.exponents

        # Dense coeffs: length 2, in set order: first [0,1] then [2,0]? Wait set order: exponents sorted by Grlex: first [0,1] (deg1) then [2,0] (deg2). So coeffs[1] should be -3.0 (for [0,1]), coeffs[2] = 1.5 (for [2,0]).
        @test p_dense.coeffs == [-3.0, 1.5]
    end

    @testset "Empty polynomial constructors" begin
        # From empty dict
        p_dense_empty = DensePolynomial(Dict{Vector{Int}, Float64}(), Grlex)
        @test length(p_dense_empty) == 0
        @test size(p_dense_empty.multiindex_set.exponents, 2) == 0

        # From empty dict with existing set (should produce zero coefficients of correct length)
        set = all_multiindices_up_to(2, 3, Lex())
        p_dense = DensePolynomial(Dict{Vector{Int}, Float64}(), set)
        @test length(p_dense.coeffs) == length(set)
        @test all(iszero, p_dense.coeffs)
    end

    # 2. Accessors
    @testset "Accessors" begin
        set = all_multiindices_up_to(2, 2, Lex())
        dict = Dict([1,0] => 4, [0,0] => 1)
        p_dense = DensePolynomial(dict, set)

        @test coeffs(p_dense) == p_dense.coeffs
        @test multiindex_set(p_dense) == set
        @test nvars(p_dense) == 2
        @test length(p_dense) == length(set)
    end

    # 3. Term lookup
    @testset "Term lookup" begin
        set = all_multiindices_up_to(3, 2, Grlex())
        dict = Dict([1,0,0] => 2.5, [0,1,1] => -1.2, [0,0,0] => 7.0)
        p_dense = DensePolynomial(dict, set)

        # find_in_multiindex_set
        @test find_in_multiindex_set(p_dense, [1,0,0]) == find_in_set(set, [1,0,0])
        @test isnothing(find_in_multiindex_set(p_dense, [5,0,0]))

        # has_term
        # For dense, any exponent within the set is considered present (even with zero coefficient)
        @test has_term(p_dense, [1,0,0]) == true
        @test has_term(p_dense, [2,0,0]) == true   # exponent exists in set, though coefficient is zero
        @test has_term(p_dense, [5,0,0]) == false  # outside set

        # coefficient
        @test coefficient(p_dense, [1,0,0]) == 2.5
        @test coefficient(p_dense, [0,0,0]) == 7.0
        @test coefficient(p_dense, [2,0,0]) == 0.0

        # find_term
        # For dense, returns index if exponent in set (even if coefficient zero)
        @test find_term(p_dense, [1,0,0]) == find_in_set(set, [1,0,0])
        @test find_term(p_dense, [0,0,2]) == find_in_set(set, [0,0,2])  # exists in set, coefficient zero
        @test find_term(p_dense, [5,0,0]) === nothing
    end

    # 4. Zero polynomial constructors
    @testset "Zero polynomials" begin
        # From existing polynomial
        set = all_multiindices_up_to(2, 2, Lex())
        # (2,0), (1,1), (1,0), (0,2), (0,1), (0,0)

        p_dense = DensePolynomial(Dict([1,0]=>3), set)

        zero_dense = zero(DensePolynomial{Int}, set)

        @test zero_dense isa DensePolynomial{Int,Lex}
        @test length(zero_dense) == 6
        @test size(zero_dense.multiindex_set.exponents,2) == 6
        @test nvars(zero_dense) == 2

        # From type and existing set
        set_grlex = all_multiindices_up_to(2, 3, Grlex())
        zero_dense_set = zero(DensePolynomial{Int,Grlex}, set_grlex)

        @test zero_dense_set isa DensePolynomial{Int,Grlex}
        @test length(zero_dense_set.coeffs) == length(set_grlex)
        @test all(iszero, zero_dense_set.coeffs)
        @test zero_dense_set.multiindex_set == set_grlex
    end

    # 5. Error conditions
    @testset "Error conditions" begin
        set = all_multiindices_up_to(2, 2, Lex())

        # Mismatched nvars in coefficient lookup
        p = DensePolynomial(Dict([1,0]=>1), set)
        @test_throws AssertionError coefficient(p, [1,0,0])  # length mismatch

        # Using monomial_rank with Grevlex (error expected)
        using .Polynomials: monomial_rank
        @test_throws ErrorException monomial_rank([1,0], 2, 2, Grevlex())

        # polynomial_from_pairs with mismatched exponent lengths (should error in MultiindexSet constructor)
        @test_throws AssertionError polynomial_from_pairs(DensePolynomial{Int,Lex}, [[1,0]=>1, [1,0,0]=>2])
    end

    # 6. Different element types
    @testset "Element types" begin
        for T in [Int, Float64, Rational{Int}]
            dict = Dict([1,0]=>T(2), [0,1]=>T(-3))
            p_dense = DensePolynomial(dict, Grlex)
            @test eltype(p_dense) == T
            @test coefficient(p_dense, [1,0]) == T(2)
        end
    end

    # 7. Test that indices are correctly sorted according to the set's order
    @testset "Indices sorted" begin
        # For Grlex, the set is sorted by total degree then lex. We'll create a polynomial with terms of different degrees.
        set = all_multiindices_up_to(3, 3, Grlex())
        # (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2), etc.
        # Our set has [0,0,0] (index 1), [0,0,1] (index 4), [2,0,0] (index 5), [1,1,0] (index 6). Sorted indices: [1,4,5,6].
        expected_indices = [1,4,5,6]
        @test expected_indices == [find_in_set(set, [0,0,0]), find_in_set(set, [0,0,1]), find_in_set(set, [2,0,0]), find_in_set(set, [1,1,0])]
    end

    # 9. Test that polynomial_from_pairs with order works as expected
    @testset "polynomial_from_pairs order" begin
        pairs = [[1,0]=>1, [0,1]=>2, [0,0]=>3]
        p_lex = polynomial_from_pairs(DensePolynomial{Int,Lex}, pairs)
        p_grlex = polynomial_from_pairs(DensePolynomial{Int,Grlex}, pairs)

        # Lex order: [1,0] > [0,1] > [0,0]? Actually lex: larger first component first, so [1,0] then [0,1] then [0,0].
        # So set should be [[1,0],[0,1],[0,0]].
        expected_lex_set = MultiindexSet([[1,0],[0,1],[0,0]], Lex())
        @test p_lex.multiindex_set.exponents == expected_lex_set.exponents
        @test p_lex.coeffs == [1,2,3]

        # Grlex order: first deg0 [0,0], then deg1: [1,0] then [0,1] (lex within degree). So set: [[0,0],[1,0],[0,1]].
        expected_grlex_set = MultiindexSet([[0,0],[1,0],[0,1]], Grlex())
        @test p_grlex.multiindex_set.exponents == expected_grlex_set.exponents
        @test p_grlex.coeffs == [3,1,2]
    end
end

println("All tests passed.")