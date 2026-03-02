# test_multiindices.jl

using Test
include(joinpath(@__DIR__, "../../src/Multiindices.jl"))
using .Multiindices

# ==================== Helper function to check sortedness ====================
function is_sorted_set(set::MultiindexSet{O}) where O
    exps = set.exponents
    n = size(exps, 2)
    n ≤ 1 && return true
    less = (a,b) -> precede(O(), a, b)
    for i in 1:n-1
        col_i = view(exps, :, i)
        col_j = view(exps, :, i+1)
        if !less(col_i, col_j) && col_i != col_j
            return false
        end
    end
    return true
end

# ==================== Ordering utilities ====================
@testset "Ordering utilities" begin

    # Lex: c < d < b < a
    # Grlex: c < d < a < b
    # Grevlex: c < a < d < b

    c = [2, 1, 1] # degree 4
    d = [1, 3, 2] # degree 6
    b = [1, 2, 4] # degree 7
    a = [1, 2, 3] # degree 6

    # lex_precede
    # Lex: c < d < b < a
    @test lex_precede(a, b) == false
    @test lex_precede(b, a) == true
    @test lex_precede(a, c) == false
    @test lex_precede(c, a) == true
    @test lex_precede(c, d) == true
    @test lex_precede(a, a) == false  # equal

    # grlex_precede
    a2 = [0, 3, 3] # degree 6
    d2 = [1, 2, 0]  # degree 3
    # Grlex: d2 < c < d < a < a2 < b
    @test grlex_precede(a, b) == true
    @test grlex_precede(b, a) == false
    @test grlex_precede(c, a) == true  
    @test grlex_precede(a, c) == false
    @test grlex_precede(d2, a) == true
    @test grlex_precede(a2, a) == lex_precede(a2, a)
    @test grlex_precede(a, a2) == true

    # grevlex_precede
    # Grevlex: d2 < c < a2 < a < d < b
    @test grevlex_precede(c, a) == true
    @test grevlex_precede(a, c) == false
    @test grevlex_precede(a2, a) == true
    @test grevlex_precede(d, a2) == false
    @test grevlex_precede(a, d) == true
    @test grevlex_precede(b, d) == false
end

# ==================== MultiindexSet construction ====================
@testset "MultiindexSet construction" begin
    # From matrix
    mat = [1 2 0;
           0 1 1]  # 2 variables, 3 monomials

    set_lex = MultiindexSet(mat, Lex())
    @test set_lex isa MultiindexSet{Lex}
    @test is_sorted_set(set_lex)
    @test set_lex.exponents == [2 1 0; 
                                1 0 1]

    set_grlex = MultiindexSet(mat, Grlex())
    @test set_grlex isa MultiindexSet{Grlex}
    @test is_sorted_set(set_grlex)
    @test set_grlex.exponents == [1 0 2; 
                                  0 1 1]

    set_grevlex = MultiindexSet(mat, Grevlex())
    @test set_grevlex isa MultiindexSet{Grevlex}
    @test is_sorted_set(set_grevlex)
    @test set_grevlex.exponents == [0 1 2; 
                                    1 0 1]

    # From vector of vectors
    vecs = [[1,0], [2,1], [0,1], [1,1], [1,2], [3,0]]
    set_vec = MultiindexSet(vecs, Grlex())
    # (1,0) -> (0,1) -> (2,0)X -> (1,1) -> (0,2)X -> (3,0) -> (2,1) -> (1,2) -> (0,3)X
    @test set_vec.exponents == [1 0 1 3 2 1; 
                                0 1 1 0 1 2]
    
    # Empty
    set_empty = MultiindexSet(Matrix{Int}(undef, 0, 0), Lex())
    @test size(set_empty.exponents) == (0,0)
end

# ==================== Generation functions ====================
@testset "Generation functions" begin
    nvars = 2
    max_deg = 2

    # all_multiindices_up_to
    set_lex = all_multiindices_up_to(nvars, max_deg, Lex())
    @test is_sorted_set(set_lex)
    @test size(set_lex.exponents, 2) == 6    
    expected_lex = [2 1 1 0 0 0; 
                    0 1 0 2 1 0]
    @test set_lex.exponents == expected_lex

    set_grlex = all_multiindices_up_to(nvars, max_deg, Grlex())
    @test is_sorted_set(set_grlex)
    expected_grlex = [0 1 0 2 1 0; 
                      0 0 1 0 1 2]
    @test set_grlex.exponents == expected_grlex

    set_grevlex = all_multiindices_up_to(nvars, max_deg, Grevlex())
    @test is_sorted_set(set_grevlex)
    expected_grevlex = [0 0 1 0 1 2; 
                        0 1 0 2 1 0]
    @test set_grevlex.exponents == expected_grevlex

    # multiindices_with_total_degree
    deg = 2
    set_fixed_lex = multiindices_with_total_degree(nvars, deg, Lex())
    @test is_sorted_set(set_fixed_lex)
    @test size(set_fixed_lex.exponents, 2) == 3
    @test set_fixed_lex.exponents == [2 1 0; 
                                      0 1 2]

    set_fixed_grlex = multiindices_with_total_degree(nvars, deg, Grlex())
    @test is_sorted_set(set_fixed_grlex)
    @test set_fixed_grlex.exponents == set_fixed_lex.exponents

    set_fixed_grevlex = multiindices_with_total_degree(nvars, deg, Grevlex())
    @test is_sorted_set(set_fixed_grevlex)
    @test set_fixed_grevlex.exponents == [0 1 2; 
                                          2 1 0]

    # all_multiindices_in_box with bound vector
    bound = [1,2]
    set_box = all_multiindices_in_box(bound, Grlex())
    @test is_sorted_set(set_box)
    @test size(set_box.exponents, 2) == 6
    expected_box_lex = [0 1 0 1 0 1; 
                        0 0 1 1 2 2]
    @test set_box.exponents == expected_box_lex

    # all_multiindices_in_box with matrix
    mat = [1 2; 
           0 1]
    set_from_mat = all_multiindices_in_box(mat, Lex())
    bound_from_mat = [2,1]  # max per row
    expected = all_multiindices_in_box(bound_from_mat, Lex())
    @test set_from_mat.exponents == expected.exponents
end

# ==================== Operations on MultiindexSet ====================
@testset "Operations on MultiindexSet" begin
    set = all_multiindices_up_to(2, 2, Grlex())  # 6 elements
    @test length(set) == 6
    @test set[1] == [0,0]
    @test set[2] == [1,0]
    @test collect(set) == [[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]] # sorted in grlex
    @test [v for v in set] == collect(set)

    # find_in_set
    @test find_in_set(set, [1,1]) == 5
    @test find_in_set(set, [2,0]) == 4
    @test find_in_set(set, [0,0]) == 1
    @test find_in_set(set, [3,0]) === nothing

    # preceding_indices
    @test preceding_indices(set, [0,2]) == 1:5
    @test preceding_indices(set, [1,1]) == 1:4
    @test preceding_indices(set, [0,0]) == 1:0
    @test preceding_indices(set, [2,0]) == 1:3

    # indices_in_box_and_after
    box_upper = [1,1]
    @test indices_in_box_and_after(set, box_upper, [0,0]) == [2,3,5]
    @test indices_in_box_and_after(set, box_upper, [0,3]) == Int[] # No indices after [0,3] in grelex
    @test indices_in_box_and_after(set, box_upper, [-1,-1]) == [1,2,3,5]  # All indices except index 4, which is not in box.

    # Edge: empty set
    empty_set = MultiindexSet(Matrix{Int}(undef, 0, 0), Grlex())
    @test length(empty_set) == 0
    @test collect(empty_set) == []
    @test find_in_set(empty_set, [1,2]) === nothing
    @test preceding_indices(empty_set, [1,2]) == 1:0
    @test indices_in_box_and_after(empty_set, [1,1], [0,0]) == Int[]
end

# ==================== Factorizations ====================
@testset "Factorizations" begin
    exp = [2,1]
    N = 2

    # Create a set containing all vectors in the box from 0 to exp
    full_set = all_multiindices_in_box(exp, Lex())  # order doesn't affect count
    facs = factorizations(full_set, exp, N)
    # All pairs of vectors (a,b) with a+b=exp
    # List all a from [0,0] to [2,1] componentwise:
    # a=[0,0] -> b=[2,1]
    # a=[0,1] -> b=[2,0]
    # a=[1,0] -> b=[1,1]
    # a=[1,1] -> b=[1,0]
    # a=[2,0] -> b=[0,1]
    # a=[2,1] -> b=[0,0]
    # So expected: 6 factorizations.
    @test length(facs) == 6
    # Check each is sorted lex (a ≤ b)
    for f in facs
        @test length(f) == 2
        @test f[1] + f[2] == exp
    end

    # With N=1
    facs1 = factorizations(full_set, exp, 1)
    @test facs1 == [[[2,1]]]

    # With N=0 should return empty
    @test factorizations(full_set, exp, 0) == []

    # Factorizations where some vectors missing
    set_small = MultiindexSet([[0,0],[1,0],[2,0],[2,1]], Grlex())
    facs_small = factorizations(set_small, exp, N)
    # a=[0,0] -> b=[2,1]
    # a=[1,0] -> b=[1,1] X 
    # a=[2,0] -> b=[0,1] X
    # a=[2,1] -> b=[0,0]
    @test length(facs_small) == 2 
    # Check each is sorted lex (a ≤ b)
    for f in facs
        @test length(f) == 2
        @test f[1] + f[2] == exp
    end
end

# ==================== Ranking functions ====================
@testset "Ranking functions" begin
    nvars = 2
    max_deg = 3
    set_lex = all_multiindices_up_to(nvars, max_deg, Lex())
    set_grlex = all_multiindices_up_to(nvars, max_deg, Grlex())

    # num_multiindices_up_to
    @test num_multiindices_up_to(nvars, max_deg) == binomial(max_deg + nvars, nvars) == 10

    # monomial_rank for lex
    for (idx, exp) in enumerate(collect(set_lex))
        @test monomial_rank(exp, nvars, max_deg, Lex()) == idx
    end

    # monomial_rank for grlex
    for (idx, exp) in enumerate(collect(set_grlex))
        @test monomial_rank(exp, nvars, max_deg, Grlex()) == idx
    end

    # grevlex should error
    @test_throws ErrorException monomial_rank([1,0], nvars, max_deg, Grevlex())
end

# ==================== Edge cases ====================
@testset "Edge cases" begin
    # Zero variables
    @test all_multiindices_up_to(0, 5, Lex()).exponents == zeros(Int, 0, 0)
    @test multiindices_with_total_degree(0, 0, Lex()).exponents == zeros(Int, 0, 0)
    @test multiindices_with_total_degree(0, 1, Lex()).exponents == zeros(Int, 0, 0)  # no vectors
    @test all_multiindices_in_box(Int[], Lex()).exponents == zeros(Int, 0, 0)

    # Zero max_degree
    set0 = all_multiindices_up_to(3, 0, Grlex())
    @test size(set0.exponents, 2) == 1
    @test set0[1] == [0,0,0]

    # Box with zero bound
    bound = [0,2,0]
    set_box0 = all_multiindices_in_box(bound, Grlex())
    @test size(set_box0.exponents, 2) == 3
    @test set_box0.exponents == [0 0 0; 
                                 0 1 2; 
                                 0 0 0]  # rows: first row all 0, second row 0,1,2, third row all 0.

    # divides
    @test divides([1,0], [2,1]) == true
    @test divides([2,0], [1,1]) == false
    @test divides([0,0], [1,1]) == true

    # is_constant
    @test is_constant([0,0,0]) == true
    @test is_constant([0,1,0]) == false

    # compare
    @test compare([1,3], [1,2], Lex()) == -1 # [1,3] comes after
    @test compare([1,2], [1,3], Lex()) == 1  # [1,2] comes before
    @test compare([1,2], [1,2], Lex()) == 0

    # MultiindexSet with single column
    single = MultiindexSet([[2,2,2]], Grlex())
    @test is_sorted_set(single)
    @test single[1] == [2,2,2]
    @test find_in_set(single, [2,2,2]) == 1
    @test find_in_set(single, [0,0,0]) === nothing
    @test preceding_indices(single, [1,2,3]) == 1:1  # [2,2,2] < [1,2,3] in Grlex
    @test preceding_indices(single, [0,0,0]) == 1:0
    @test indices_in_box_and_after(single, [2,2,2], [0,0,0]) == [1]

    # factorization with N=0 returns empty list
    @test factorizations(single, [1,1], 0) == []
end

println("All tests passed.")