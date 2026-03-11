using Test
include(joinpath(@__DIR__, "../../src/Multiindices.jl"))
using .Multiindices
using .Multiindices: grlex_precede

# ============================================================================
# Helper functions for testing
# ============================================================================

"""
    is_grlex_sorted(set::MultiindexSet) -> Bool

Check that the columns of `set.exponents` are non‑decreasing in the Grlex order.
"""
function is_grlex_sorted(set::MultiindexSet)
    exps = set.exponents
    n = size(exps, 2)
    n ≤ 1 && return true
    for i in 1:n-1
        a = view(exps, :, i)
        b = view(exps, :, i+1)
        if !grlex_precede(a, b) && a != b
            return false
        end
    end
    return true
end

"""
    random_exponent(nvars::Int, max_deg::Int) -> Vector{Int}

Generate a random exponent vector with total degree ≤ max_deg.
"""
function random_exponent(nvars::Int, max_deg::Int)
    exp = zeros(Int, nvars)
    deg = rand(0:max_deg)
    for _ in 1:deg
        exp[rand(1:nvars)] += 1
    end
    return exp
end

"""
    factorization_less(a::Matrix{Int}, b::Matrix{Int}) -> Bool

Lexicographic comparison of two factorizations (matrices where columns are factors)
using Grlex order on the factor vectors. Returns `true` if `a` comes before `b`.
"""
function factorization_less(a::Matrix{Int}, b::Matrix{Int})
    @assert size(a) == size(b)
    for j in 1:size(a, 2)
        col_a = view(a, :, j)
        col_b = view(b, :, j)
        if grlex_precede(col_a, col_b)
            return true
        elseif grlex_precede(col_b, col_a)
            return false
        end
    end
    return false  # equal
end

# ============================================================================
# Test basic ordering functions
# ============================================================================
@testset "grlex_precede" begin
    # Deg 0
    a = [0,0,0]
    b = [0,0,0]
    @test !grlex_precede(a, b)

    # Different degrees
    a = [1,0,0]   # deg 1
    b = [0,1,1]   # deg 2
    @test grlex_precede(a, b)   # lower degree first
    @test !grlex_precede(b, a)

    # Same degree, lexicographic tie‑break
    a = [2,0,1]   # deg 3
    b = [1,2,0]   # deg 3
    # Lex order: compare first component: 2 > 1 → a precedes b
    @test grlex_precede(a, b)
    @test !grlex_precede(b, a)

    a = [1,2,0]   # deg 3
    b = [1,0,2]   # deg 3
    # First components equal (1), second: 2 > 0 → a precedes b
    @test grlex_precede(a, b)

    a = [1,0,2]   # deg 3
    b = [1,0,2]   # equal
    @test !grlex_precede(a, b) && !grlex_precede(b, a)

    # Additional random tests: compare with explicit sorting
    for _ in 1:100
        nvars = rand(2:5)
        max_deg = rand(1:5)
        exps = [random_exponent(nvars, max_deg) for _ in 1:10]
        sorted = sort(exps; lt=grlex_precede)
        for i in 1:length(sorted)-1
            @test grlex_precede(sorted[i], sorted[i+1]) || sorted[i] == sorted[i+1]
        end
    end
end

# ============================================================================
# Test MultiindexSet construction and sorting
# ============================================================================
@testset "MultiindexSet construction" begin
    # From a matrix
    mat = [1 2 0;
           0 1 1]   # 2 variables, 3 monomials
    set = MultiindexSet(mat)
    @test set isa MultiindexSet
    @test is_grlex_sorted(set)
    # Expected Grlex order: (1,0) deg1, (0,1) deg1, (2,1) deg3
    @test set.exponents == [1 0 2;
                            0 1 1]

    # From a vector of vectors
    vecs = [[1,0], [2,1], [0,1], [1,1], [1,2], [3,0]]
    set_vec = MultiindexSet(vecs)
    @test is_grlex_sorted(set_vec)
    # Expected: deg1: [1,0] < [0,1]; deg2: [1,1]; deg3: [3,0] < [2,1] < [1,2]
    expected = [1 0 1 3 2 1;
                0 1 1 0 1 2]
    @test set_vec.exponents == expected

    # Empty set
    set_empty = MultiindexSet(Matrix{Int}(undef, 0, 0))
    @test size(set_empty.exponents) == (0,0)
    @test length(set_empty) == 0

    # Single element
    set_single = MultiindexSet([[5,5,5]])
    @test is_grlex_sorted(set_single)
    @test set_single[1] == [5,5,5]

    # Duplicate elements (should be allowed, but sorted order may keep duplicates)
    dup = [[1,0], [1,0], [0,1]]
    set_dup = MultiindexSet(dup)
    @test is_grlex_sorted(set_dup)
    @test set_dup.exponents == [1 1 0;
                                0 0 1]   # duplicates allowed
end

# ============================================================================
# Test generation functions
# ============================================================================
@testset "Generation: all_multiindices_up_to" begin
    nvars = 3
    max_deg = 2
    set = all_multiindices_up_to(nvars, max_deg)

    # Expected number: binomial(2+3,3) = binomial(5,3) = 10
    @test length(set) == 10
    @test is_grlex_sorted(set)

    # Manually check first few
    @test set[1] == [0,0,0]                 # deg 0
    @test set[2] == [1,0,0]                 # deg 1
    @test set[3] == [0,1,0]                 # deg 1
    @test set[4] == [0,0,1]                 # deg 1
    @test set[5] == [2,0,0]                 # deg 2
    @test set[6] == [1,1,0]                 # deg 2
    @test set[7] == [1,0,1]                 # deg 2
    @test set[8] == [0,2,0]                 # deg 2
    @test set[9] == [0,1,1]                 # deg 2
    @test set[10] == [0,0,2]                # deg 2

    # Edge: nvars = 0
    set0 = all_multiindices_up_to(0, 5)
    @test size(set0.exponents) == (0,0)

    # Edge: max_deg = 0
    set_deg0 = all_multiindices_up_to(3, 0)
    @test length(set_deg0) == 1
    @test set_deg0[1] == [0,0,0]
end

@testset "Generation: multiindices_with_total_degree" begin
    nvars = 3
    deg = 2
    set = multiindices_with_total_degree(nvars, deg)

    # Number: binomial(2+3-1,3-1) = binomial(4,2) = 6
    @test length(set) == 6
    @test is_grlex_sorted(set)

    # Within fixed degree, order is lexicographic (larger first components first)
    # All vectors of deg 2 in lex order: [2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]
    expected = [2 1 1 0 0 0;
                0 1 0 2 1 0;
                0 0 1 0 1 2]
    @test set.exponents == expected

    # Edge: nvars = 0
    set0 = multiindices_with_total_degree(0, 0)
    @test size(set0.exponents) == (0,0)
    set0 = multiindices_with_total_degree(0, 1)
    @test size(set0.exponents) == (0,0)
end

@testset "Generation: all_multiindices_in_box" begin
    bound = [1,2]
    set = all_multiindices_in_box(bound)
    @test length(set) == prod(bound .+ 1) == 6
    @test is_grlex_sorted(set)

    # Expected vectors: (0,0),(1,0),(0,1),(1,1),(0,2),(1,2) sorted by Grlex.
    expected = [0 1 0 1 0 1;
                0 0 1 1 2 2]
    @test set.exponents == expected

    # Edge: empty bound
    set_empty = all_multiindices_in_box(Int[])
    @test size(set_empty.exponents) == (0,0)

    # Edge: zero bound components
    bound3 = [0,2,0]
    set3 = all_multiindices_in_box(bound3)
    @test size(set3.exponents, 2) == 3
    expected3 = [0 0 0;
                 0 1 2;
                 0 0 0]
    @test set3.exponents == expected3
end

# ============================================================================
# Test operations on MultiindexSet
# ============================================================================
@testset "Basic operations: length, getindex, iteration" begin
    set = all_multiindices_up_to(2, 2)
    @test length(set) == 6
    @test set[1] == [0,0]
    @test set[2] == [1,0]
    @test set[3] == [0,1]
    @test set[4] == [2,0]
    @test set[5] == [1,1]
    @test set[6] == [0,2]

    collected = collect(set)
    @test collected == [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]
    @test [v for v in set] == collected
end

@testset "find_in_set" begin
    set = all_multiindices_up_to(2, 2)
    @test find_in_set(set, [1,1]) == 5
    @test find_in_set(set, [2,0]) == 4
    @test find_in_set(set, [0,0]) == 1
    @test find_in_set(set, [3,0]) === nothing

    # Empty set
    empty = MultiindexSet(Matrix{Int}(undef, 0, 0))
    @test find_in_set(empty, [1,2]) === nothing

    # Single element
    single = MultiindexSet([[5,5]])
    @test find_in_set(single, [5,5]) == 1
    @test find_in_set(single, [0,0]) === nothing

    # Binary search correctness: random tests
    for _ in 1:50
        nvars = rand(2:4)
        max_deg = rand(1:5)
        set = all_multiindices_up_to(nvars, max_deg)
        idx = rand(1:length(set))
        exp = set[idx]
        @test find_in_set(set, exp) == idx
    end
end

@testset "indices_in_box_with_bounded_degree" begin
    set = all_multiindices_up_to(2, 3)   # all with total degree ≤ 3
    # indices: 1:[0,0] deg0, 2:[1,0] deg1, 3:[0,1] deg1, 4:[2,0] deg2, 5:[1,1] deg2,
    #          6:[0,2] deg2, 7:[3,0] deg3, 8:[2,1] deg3, 9:[1,2] deg3, 10:[0,3] deg3

    box = [2,2]
    # Find indices with 1 ≤ total degree ≤ 2 and within box
    result = indices_in_box_with_bounded_degree(set, box, 1, 3)   # total_deg_upper = 3 → degree <3
    @test result == [2,3,4,5,6]   # indices 2..6 have degree 1 or 2, all inside box

    # degree_lower_bound = 2, total_deg_upper = 3 → degree exactly 2
    result2 = indices_in_box_with_bounded_degree(set, box, 2, 3)
    @test result2 == [4,5,6]   # indices 4,5,6 have degree 2, all inside box

    # box that excludes some
    box_small = [1,1]
    result3 = indices_in_box_with_bounded_degree(set, box_small, 0, 3)   # all degrees <3
    # Inside [1,1]: [0,0](1), [1,0](2), [0,1](3), [1,1](5) → indices 1,2,3,5
    @test result3 == [1,2,3,5]

    # Empty set – must use empty box because set has zero variables
    empty = MultiindexSet(Matrix{Int}(undef, 0, 0))
    @test indices_in_box_with_bounded_degree(empty, Int[], 0, 3) == Int[]
end

# ============================================================================
# Test predicates: divides, is_constant
# ============================================================================
@testset "Predicates" begin
    # divides
    @test divides([1,0], [2,1]) == true
    @test divides([2,0], [1,1]) == false
    @test divides([0,0], [1,1]) == true

    # is_constant
    @test is_constant([0,0,0]) == true
    @test is_constant([0,1,0]) == false
end

# ============================================================================
# Test factorizations
# ============================================================================
@testset "factorizations" begin
    exp = [2,1]
    N = 2

    # Full set containing all vectors in box 0..exp
    full_set = all_multiindices_in_box(exp)
    facs = factorizations(full_set, exp, N)

    # Expect 6 factorizations
    @test length(facs) == 6
    # Verify each factorization sums to exp and all factors belong to set
    for f in facs
        @test size(f) == (2, N)
        s = zeros(Int, 2)
        for j in 1:N
            s .+= view(f, :, j)
        end
        @test s == exp
        @test all(j -> find_in_set(full_set, f[:, j]) !== nothing, 1:N)
    end

    # Outer list should be sorted lexicographically
    sorted_facs = sort(facs; lt=factorization_less)
    @test facs == sorted_facs

    # N = 1
    facs1 = factorizations(full_set, exp, 1)
    @test length(facs1) == 1
    @test facs1[1] == reshape([2,1], 2, 1)

    # N = 0
    @test factorizations(full_set, exp, 0) == []               # exp not zero
    @test factorizations(full_set, [0,0], 0) == [zeros(Int, 2, 0)]  # zero exponent

    # Set with missing vectors
    small_set = MultiindexSet([[0,0],[1,0],[2,0],[2,1]])
    facs_small = factorizations(small_set, exp, N)
    @test length(facs_small) == 2   # [0,0]+[2,1] and [2,1]+[0,0]
    for f in facs_small
        s = zeros(Int, 2)
        for j in 1:N
            s .+= view(f, :, j)
        end
        @test s == exp
        @test all(j -> find_in_set(small_set, f[:, j]) !== nothing, 1:N)
    end

    # Test with larger N and random exponents
    for _ in 1:20
        nvars = 2
        max_deg = 4
        set = all_multiindices_up_to(nvars, max_deg)
        exp = random_exponent(nvars, max_deg)
        N = rand(1:3)
        facs = factorizations(set, exp, N)
        for f in facs
            @test size(f) == (nvars, N)
            s = zeros(Int, nvars)
            for j in 1:N
                s .+= view(f, :, j)
            end
            @test s == exp
        end
        # Ensure sortedness
        if !isempty(facs)
            sorted_facs = sort(facs; lt=factorization_less)
            @test facs == sorted_facs
        end
    end
end

# ============================================================================
# Test combinatorial ranking
# ============================================================================
@testset "num_multiindices_up_to" begin
    @test num_multiindices_up_to(2, 3) == binomial(5,2) == 10
    @test num_multiindices_up_to(3, 2) == binomial(5,3) == 10
    @test num_multiindices_up_to(0, 5) == 0
    @test num_multiindices_up_to(1, 5) == binomial(6,1) == 6
end

@testset "monomial_rank" begin
    nvars = 3
    max_deg = 2
    set = all_multiindices_up_to(nvars, max_deg)

    # Check that rank matches index in generated set
    for (idx, exp) in enumerate(collect(set))
        @test monomial_rank(exp, nvars, max_deg) == idx
    end

    # Edge: max_deg = 0
    set0 = all_multiindices_up_to(2, 0)
    @test monomial_rank([0,0], 2, 0) == 1
    @test_throws AssertionError monomial_rank([1,0], 2, 0)   # degree exceeds max

    # Random tests
    for _ in 1:50
        nvars = rand(2:4)
        max_deg = rand(1:5)
        set = all_multiindices_up_to(nvars, max_deg)
        idx = rand(1:length(set))
        exp = set[idx]
        @test monomial_rank(exp, nvars, max_deg) == idx
    end
end

# ============================================================================
# Test that generated sets are always sorted
# ============================================================================
@testset "Generated sets are always sorted" begin
    for _ in 1:20
        nvars = rand(0:4)
        max_deg = rand(0:5)
        set1 = all_multiindices_up_to(nvars, max_deg)
        @test is_grlex_sorted(set1)

        if nvars > 0 && max_deg > 0
            deg = rand(0:max_deg)
            set2 = multiindices_with_total_degree(nvars, deg)
            @test is_grlex_sorted(set2)
        end

        if nvars > 0
            bound = rand(0:3, nvars)
            set3 = all_multiindices_in_box(bound)
            @test is_grlex_sorted(set3)
        end
    end
end

# ============================================================================
# Test zero_multiindex and multiindex convenience constructors
# ============================================================================
@testset "Convenience constructors" begin
    @test zero_multiindex(3) == [0,0,0]
    @test zero_multiindex(0) == Int[]
    @test multiindex(1,2,3) == [1,2,3]
    @test multiindex() == Int[]
end

println("All tests passed.")