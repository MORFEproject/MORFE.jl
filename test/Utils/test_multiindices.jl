using StaticArrays: SVector

using MORFE.Multiindices
using MORFE.Multiindices: grlex_precede, num_multiindices_up_to, monomial_rank, multiindex

# ============================================================================
# Helper functions for testing
# ============================================================================

"""
    is_grlex_sorted(set::MultiindexSet) -> Bool

Check that the exponents of `set` are non‑decreasing in the Grlex order.
"""
function is_grlex_sorted(set::MultiindexSet)
    exps = set.exponents
    n = length(exps)
    n ≤ 1 && return true
    for i in 1:(n - 1)
        a = exps[i]
        b = exps[i + 1]
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

# ============================================================================
# Test basic ordering functions
# ============================================================================
@testset "grlex_precede" begin
    # Deg 0
    a = [0, 0, 0]
    b = [0, 0, 0]
    @test !grlex_precede(a, b)

    # Different degrees
    a = [1, 0, 0]   # deg 1
    b = [0, 1, 1]   # deg 2
    @test grlex_precede(a, b)   # lower degree first
    @test !grlex_precede(b, a)

    # Same degree, lexicographic tie‑break
    a = [2, 0, 1]   # deg 3
    b = [1, 2, 0]   # deg 3
    # Lex order: compare first component: 2 > 1 → a precedes b
    @test grlex_precede(a, b)
    @test !grlex_precede(b, a)

    a = [1, 2, 0]   # deg 3
    b = [1, 0, 2]   # deg 3
    # First components equal (1), second: 2 > 0 → a precedes b
    @test grlex_precede(a, b)

    a = [1, 0, 2]   # deg 3
    b = [1, 0, 2]   # equal
    @test !grlex_precede(a, b) && !grlex_precede(b, a)

    # Additional random tests: compare with explicit sorting
    for _ in 1:100
        nvars = rand(2:5)
        max_deg = rand(1:5)
        exps = [random_exponent(nvars, max_deg) for _ in 1:10]
        sorted = sort(exps; lt = grlex_precede)
        for i in 1:(length(sorted) - 1)
            @test grlex_precede(sorted[i], sorted[i + 1]) || sorted[i] == sorted[i + 1]
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
    expected = [SVector(1, 0), SVector(0, 1), SVector(2, 1)]
    @test set.exponents == expected

    # From a vector of vectors
    vecs = [[1, 0], [2, 1], [0, 1], [1, 1], [1, 2], [3, 0]]
    set_vec = MultiindexSet(vecs)
    @test is_grlex_sorted(set_vec)
    # Expected: deg1: [1,0] < [0,1]; deg2: [1,1]; deg3: [3,0] < [2,1] < [1,2]
    expected_vec = [SVector(1, 0), SVector(0, 1), SVector(1, 1),
        SVector(3, 0), SVector(2, 1), SVector(1, 2)]
    @test set_vec.exponents == expected_vec

    # Empty set (zero variables)
    set_empty = MultiindexSet(Matrix{Int}(undef, 0, 0))
    @test length(set_empty) == 0
    @test isempty(set_empty.exponents)

    # Single element
    set_single = MultiindexSet([[5, 5, 5]])
    @test is_grlex_sorted(set_single)
    @test set_single[1] == SVector(5, 5, 5)

    # Sorted order should not keep duplicates
    dup = [[1, 0], [1, 0], [0, 1]]
    set_dup = MultiindexSet(dup)
    @test is_grlex_sorted(set_dup)
    expected_dup = [SVector(1, 0), SVector(0, 1)]
    @test set_dup.exponents == expected_dup
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
    @test set[1] == SVector(0, 0, 0)                 # deg 0
    @test set[2] == SVector(1, 0, 0)                 # deg 1
    @test set[3] == SVector(0, 1, 0)                 # deg 1
    @test set[4] == SVector(0, 0, 1)                 # deg 1
    @test set[5] == SVector(2, 0, 0)                 # deg 2
    @test set[6] == SVector(1, 1, 0)                 # deg 2
    @test set[7] == SVector(1, 0, 1)                 # deg 2
    @test set[8] == SVector(0, 2, 0)                 # deg 2
    @test set[9] == SVector(0, 1, 1)                 # deg 2
    @test set[10] == SVector(0, 0, 2)                # deg 2

    # Edge: nvars = 0
    set0 = all_multiindices_up_to(0, 5)
    @test length(set0) == 1                # only the empty exponent
    @test set0[1] == SVector{0, Int}()
    set0_neg = all_multiindices_up_to(0, -1)
    @test length(set0_neg) == 0

    # Edge: max_deg = 0
    set_deg0 = all_multiindices_up_to(3, 0)
    @test length(set_deg0) == 1
    @test set_deg0[1] == SVector(0, 0, 0)
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
    expected = [SVector(2, 0, 0), SVector(1, 1, 0), SVector(1, 0, 1),
        SVector(0, 2, 0), SVector(0, 1, 1), SVector(0, 0, 2)]
    @test set.exponents == expected

    # Edge: nvars = 0
    set0 = multiindices_with_total_degree(0, 0)
    @test length(set0) == 1
    @test set0[1] == SVector{0, Int}()
    set0_deg1 = multiindices_with_total_degree(0, 1)
    @test length(set0_deg1) == 0
end

@testset "Generation: all_multiindices_in_box" begin
    bound = [1, 2]
    set = all_multiindices_in_box(bound)
    @test length(set) == prod(bound .+ 1) == 6
    @test is_grlex_sorted(set)

    # Expected vectors: (0,0),(1,0),(0,1),(1,1),(0,2),(1,2) sorted by Grlex.
    expected = [SVector(0, 0), SVector(1, 0), SVector(0, 1),
        SVector(1, 1), SVector(0, 2), SVector(1, 2)]
    @test set.exponents == expected

    # Edge: empty bound (zero variables)
    set_empty = all_multiindices_in_box(Int[])
    @test length(set_empty) == 1
    @test set_empty[1] == SVector{0, Int}()

    # Edge: zero bound components
    bound3 = [0, 2, 0]
    set3 = all_multiindices_in_box(bound3)
    @test length(set3) == 3
    expected3 = [SVector(0, 0, 0), SVector(0, 1, 0), SVector(0, 2, 0)]
    @test set3.exponents == expected3
end

# ============================================================================
# Test operations on MultiindexSet
# ============================================================================
@testset "Basic operations: length, getindex, iteration" begin
    set = all_multiindices_up_to(2, 2)
    @test length(set) == 6
    @test set[1] == SVector(0, 0)
    @test set[2] == SVector(1, 0)
    @test set[3] == SVector(0, 1)
    @test set[4] == SVector(2, 0)
    @test set[5] == SVector(1, 1)
    @test set[6] == SVector(0, 2)

    collected = collect(set)
    @test collected == [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
    @test [v for v in set] == [SVector(0, 0), SVector(1, 0), SVector(0, 1),
        SVector(2, 0), SVector(1, 1), SVector(0, 2)]
end

@testset "find_in_set" begin
    set = all_multiindices_up_to(2, 2)
    @test find_in_set(set, [1, 1]) == 5
    @test find_in_set(set, [2, 0]) == 4
    @test find_in_set(set, [0, 0]) == 1
    @test find_in_set(set, [3, 0]) === nothing

    # Empty set
    empty = MultiindexSet(Matrix{Int}(undef, 0, 0))
    @test find_in_set(empty, [1, 2]) === nothing

    # Single element
    single = MultiindexSet([[5, 5]])
    @test find_in_set(single, [5, 5]) == 1
    @test find_in_set(single, [0, 0]) === nothing

    # Binary search correctness: random tests
    for _ in 1:50
        nvars = rand(2:4)
        max_deg = rand(1:5)
        set = all_multiindices_up_to(nvars, max_deg)
        idx = rand(1:length(set))
        exp = set[idx]
        @test find_in_set(set, exp) == idx
    end

    # Sets whose degree blocks are incomplete: the degree bracket must not assume
    # that a block contains every exponent of that degree.
    for incomplete in (all_multiindices_in_box([2, 1, 2]),
        MultiindexSet([[0, 2, 0], [3, 1, 0], [1, 1, 0], [0, 0, 4]]))
        exps = incomplete.exponents
        for (i, v) in enumerate(exps)
            @test find_in_set(incomplete, v) == i
            @test find_in_set(incomplete, Vector(v)) == i
        end
        # Non-members, including ones sharing a degree with a populated block
        for miss in ([9, 0, 0], [1, 0, 1], [0, 1, 1], [2, 2, 0], [0, 0, 5])
            @test find_in_set(incomplete, miss) ==
                  findfirst(v -> Vector(v) == miss, exps)
        end
    end

    # Degree-gapped set (the shape `parametrise` builds via min_degree)
    gapped = all_multiindices_up_to(3, 4; min_degree = 2)
    for (i, v) in enumerate(gapped.exponents)
        @test find_in_set(gapped, v) == i
    end
    @test find_in_set(gapped, [0, 0, 0]) === nothing
    @test find_in_set(gapped, [1, 0, 0]) === nothing
    @test find_in_set(gapped, [0, 1, 0]) === nothing
    @test find_in_set(gapped, [0, 0, 1]) === nothing

    # Tuple version agrees with the vector version, on members and non-members
    tset = all_multiindices_up_to(3, 3)
    for v in tset.exponents
        @test find_in_set(tset, Tuple(v)) == find_in_set(tset, Vector(v))
    end
    @test find_in_set(tset, (4, 0, 0)) === nothing
    @test find_in_set(tset, (2, 1, 0)) == find_in_set(tset, [2, 1, 0])

    # Wrong-length exponents are simply absent, not an error
    set = all_multiindices_up_to(2, 2)
    @test find_in_set(set, [1]) === nothing
    @test find_in_set(set, [1, 1, 0]) === nothing

    # Zero-variable set: the empty exponent is its single element
    @test find_in_set(all_multiindices_up_to(0, 2), ()) == 1
    @test find_in_set(all_multiindices_up_to(0, 2), Int[]) == 1
end

@testset "indices_in_box_with_bounded_degree" begin
    set = all_multiindices_up_to(2, 3)   # all with total degree ≤ 3
    # indices: 1:[0,0] deg0, 2:[1,0] deg1, 3:[0,1] deg1, 4:[2,0] deg2, 5:[1,1] deg2,
    #          6:[0,2] deg2, 7:[3,0] deg3, 8:[2,1] deg3, 9:[1,2] deg3, 10:[0,3] deg3

    box = [2, 2]
    # Find indices with 1 ≤ total degree ≤ 2 and within box
    result = indices_in_box_with_bounded_degree(set, box, 1, 3)   # total_deg_upper = 3 → degree <3
    @test result == [2, 3, 4, 5, 6]   # indices 2..6 have degree 1 or 2, all inside box

    # degree_lower_bound = 2, total_deg_upper = 3 → degree exactly 2
    result2 = indices_in_box_with_bounded_degree(set, box, 2, 3)
    @test result2 == [4, 5, 6]   # indices 4,5,6 have degree 2, all inside box

    # box that excludes some
    box_small = [1, 1]
    result3 = indices_in_box_with_bounded_degree(set, box_small, 0, 3)   # all degrees <3
    # Inside [1,1]: [0,0](1), [1,0](2), [0,1](3), [1,1](5) → indices 1,2,3,5
    @test result3 == [1, 2, 3, 5]

    # Empty set – must use empty box because set has zero variables
    empty = MultiindexSet(Matrix{Int}(undef, 0, 0))
    @test indices_in_box_with_bounded_degree(empty, Int[], 0, 3) == Int[]
end

# ============================================================================
# Test non-mutating set operations: delete_multiindices, filter
# ============================================================================
@testset "delete_multiindices: explicit exponents" begin
    set = all_multiindices_up_to(2, 2)   # [0,0],[1,0],[0,1],[2,0],[1,1],[0,2]

    # A list of exponents, in several accepted spellings
    @test collect(delete_multiindices(set, [[2, 0], [0, 2]])) ==
          [[0, 0], [1, 0], [0, 1], [1, 1]]
    @test delete_multiindices(set, [SVector{2, Int}(2, 0), SVector{2, Int}(0, 2)]) ==
          delete_multiindices(set, [[2, 0], [0, 2]])
    @test delete_multiindices(set, [(2, 0), (0, 2)]) ==
          delete_multiindices(set, [[2, 0], [0, 2]])

    # A single exponent is not a collection of exponents
    single = delete_multiindices(set, [1, 1])
    @test collect(single) == [[0, 0], [1, 0], [0, 1], [2, 0], [0, 2]]
    @test delete_multiindices(set, (1, 1)) == single
    @test delete_multiindices(set, SVector{2, Int}(1, 1)) == single

    # Another MultiindexSet: plain set difference
    @test collect(delete_multiindices(set, all_multiindices_up_to(2, 1))) ==
          [[2, 0], [1, 1], [0, 2]]

    # Exponents that are not members are ignored (setdiff semantics)
    @test delete_multiindices(set, [[9, 9]]) == set
    @test delete_multiindices(set, Vector{Int}[]) == set

    # Wrong number of components is an error, both spellings
    @test_throws ArgumentError delete_multiindices(set, [[1, 2, 3]])
    @test_throws ArgumentError delete_multiindices(set, [1, 2, 3])
    @test_throws ArgumentError delete_multiindices(set, all_multiindices_up_to(3, 1))
end

@testset "delete_multiindices: non-mutating" begin
    set = all_multiindices_up_to(3, 3)
    before_len = length(set)
    before_exps = copy(set.exponents)
    before_offs = copy(set.degree_offsets)

    reduced = delete_multiindices(set, [[1, 1, 1]])
    @test length(reduced) == before_len - 1

    # The input is untouched, and the result does not alias it
    @test length(set) == before_len
    @test set.exponents == before_exps
    @test set.degree_offsets == before_offs
    @test reduced.exponents !== set.exponents

    # Same for the predicate form and for the no-op path
    delete_multiindices(α -> sum(α) > 1, set)
    @test set.exponents == before_exps
    untouched = delete_multiindices(set, [[9, 9, 9]])
    @test untouched == set
    @test untouched.exponents !== set.exponents
end

@testset "delete_multiindices: invariants survive" begin
    set = all_multiindices_up_to(2, 3)

    # Removing an entire degree block leaves a gap the offset table must absorb
    gapped = delete_multiindices(α -> sum(α) == 2, set)
    @test is_grlex_sorted(gapped)
    @test collect(gapped) ==
          [[0, 0], [1, 0], [0, 1], [3, 0], [2, 1], [1, 2], [0, 3]]

    # find_in_set is degree-bracketed through degree_offsets: it must agree with a
    # linear scan on both surviving and removed exponents
    for (i, v) in enumerate(gapped.exponents)
        @test find_in_set(gapped, v) == i
    end
    @test find_in_set(gapped, [1, 1]) === nothing
    @test find_in_set(gapped, [2, 0]) === nothing

    # ... and so must the degree-bracketed box query
    @test indices_in_box_with_bounded_degree(gapped, [3, 3], 1, 4) == collect(2:7)
    @test indices_in_box_with_bounded_degree(gapped, [3, 3], 2, 3) == Int[]

    # Deleting the top degree shortens the offset table
    trimmed = delete_multiindices(α -> sum(α) == 3, set)
    @test is_grlex_sorted(trimmed)
    @test trimmed == all_multiindices_up_to(2, 2)
    @test find_in_set(trimmed, [3, 0]) === nothing

    # Deleting everything yields a usable empty set
    empty = delete_multiindices(α -> true, set)
    @test length(empty) == 0
    @test find_in_set(empty, [0, 0]) === nothing
    @test indices_in_box_with_bounded_degree(empty, [3, 3], 0, 4) == Int[]
end

@testset "delete_multiindices and filter: predicate forms" begin
    set = all_multiindices_up_to(3, 4)

    @test delete_multiindices(α -> sum(α) > 2, set) == all_multiindices_up_to(3, 2)
    @test filter(α -> sum(α) ≤ 2, set) == all_multiindices_up_to(3, 2)

    # The two are exact complements: they partition the set
    pred = α -> α[1] ≥ 2
    kept = filter(pred, set)
    dropped = delete_multiindices(pred, set)
    @test length(kept) + length(dropped) == length(set)
    @test isempty(intersect(Set(kept.exponents), Set(dropped.exponents)))
    @test union(Set(kept.exponents), Set(dropped.exponents)) == Set(set.exponents)
    @test is_grlex_sorted(kept)
    @test is_grlex_sorted(dropped)

    # The predicate sees an SVector
    @test filter(α -> α isa SVector{3, Int}, set) == set

    # An anisotropic condition: total degree in the first two variables, box in the third
    aniso = filter(α -> α[1] + α[2] ≤ 2 && α[3] ≤ 1, all_multiindices_in_box([2, 2, 1]))
    @test all(α -> α[1] + α[2] ≤ 2 && α[3] ≤ 1, aniso.exponents)
    @test is_grlex_sorted(aniso)
end

# ============================================================================
# Test is_downward_closed
# ============================================================================
@testset "is_downward_closed" begin
    # Combinatorial truncations are closed by construction
    @test is_downward_closed(all_multiindices_up_to(3, 4))
    @test is_downward_closed(all_multiindices_up_to(3, 4; min_degree = 1))
    @test is_downward_closed(all_multiindices_in_box([2, 3]))
    @test is_downward_closed(all_multiindices_in_box([1, 2, 3]))

    # Degenerate cases
    @test is_downward_closed(MultiindexSet(SVector{2, Int}[]))
    @test is_downward_closed(MultiindexSet(Matrix{Int}(undef, 0, 0)))

    # Removing a divisor of a retained member breaks closure
    set = all_multiindices_up_to(2, 2)
    @test !is_downward_closed(delete_multiindices(set, [[1, 0]]))
    @test !is_downward_closed(delete_multiindices(set, [[0, 1]]))

    # Removing a maximal element does not
    @test is_downward_closed(delete_multiindices(set, [[2, 0]]))

    # A single monomial with no divisors present
    @test !is_downward_closed(MultiindexSet([[2, 2]]))

    # The zero multiindex is exempt: a min_degree = 1 set is still closed
    @test is_downward_closed(delete_multiindices(set, [[0, 0]]))

    # A spectral-style cut generally is not closed; its downward closure is
    λ = [im, -im]
    radius = 1.5
    cut = filter(α -> abs(sum(λ .* α)) ≤ radius, all_multiindices_up_to(2, 3))
    @test !is_downward_closed(cut)
    closed = MultiindexSet(reduce(vcat,
        [[SVector{2, Int}(a, b) for a in 0:α[1] for b in 0:α[2]] for α in cut.exponents]))
    @test is_downward_closed(closed)
end

# ============================================================================
# Test predicates: divides, is_constant
# ============================================================================
@testset "Predicates" begin
    # divides
    @test divides([1, 0], [2, 1]) == true
    @test divides([2, 0], [1, 1]) == false
    @test divides([0, 0], [1, 1]) == true

    # is_constant
    @test is_constant([0, 0, 0]) == true
    @test is_constant([0, 1, 0]) == false
end

# ============================================================================
# Test factorisations
# ============================================================================
@testset "factorisations" begin
    exp = [2, 1]
    N = 2

    # Full set containing all vectors in box 0..exp
    full_set = all_multiindices_in_box(exp)
    candidate_indices = indices_in_box_with_bounded_degree(full_set, exp, 1, sum(exp))
    facs = factorisations_asymmetric(full_set, exp, N, candidate_indices)

    # Expect 4 factorisations
    @test length(facs) == 4
    # Verify each factorization sums to exp and all factors belong to set
    for f in facs
        @test length(f) == N
        s = zeros(Int, 2)
        for j in 1:N
            s .+= full_set[f[j]]
        end
        @test s == exp
    end

    # Set with missing vectors
    small_set = MultiindexSet([[0, 0], [0, 1], [2, 0], [2, 1]])
    candidate_indices = indices_in_box_with_bounded_degree(small_set, exp, 0, sum(exp))
    facs_small = factorisations_asymmetric(small_set, exp, N, candidate_indices)
    @test length(facs_small) == 2   # [0,1]+[2,0] and [2,0]+[0,1]
    for f in facs_small
        s = zeros(Int, 2)
        for j in 1:N
            s .+= small_set[f[j]]
        end
        @test s == exp
    end

    # Test with larger N and random exponents
    for _ in 1:20
        nvars = 2
        max_deg = 4
        set = all_multiindices_up_to(nvars, max_deg)
        candidate_indices = indices_in_box_with_bounded_degree(set, exp, 1, sum(exp))
        exp = random_exponent(nvars, max_deg)
        N = rand(1:3)
        facs = factorisations_asymmetric(set, exp, N, candidate_indices)
        for f in facs
            @test length(f) == N
            s = zeros(Int, nvars)
            for j in 1:N
                s .+= set[f[j]]
            end
            @test s == exp
        end
    end
end

# ============================================================================
# Test combinatorial ranking
# ============================================================================
@testset "num_multiindices_up_to" begin
    @test num_multiindices_up_to(2, 3) == binomial(5, 2) == 10
    @test num_multiindices_up_to(3, 2) == binomial(5, 3) == 10
    @test num_multiindices_up_to(0, 5) == 1
    @test num_multiindices_up_to(1, 5) == binomial(6, 1) == 6
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
    @test monomial_rank([0, 0], 2, 0) == 1
    @test_throws AssertionError monomial_rank([1, 0], 2, 0)   # degree exceeds max

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
    @test zero_multiindex(3) == [0, 0, 0]
    @test zero_multiindex(0) == Int[]
    @test multiindex(1, 2, 3) == [1, 2, 3]
    @test multiindex() == Int[]
end
