module Multiindices

export MultiindexSet,
       MonomialOrder, Lex, Grlex, Grevlex,
       precede,
       zero_multiindex, multiindex,
       all_multiindices_up_to, multiindices_with_total_degree,
       all_multiindices_in_box, all_multiindices_preceding,
       lex_precede, grlex_precede, grevlex_precede,  # keep for direct use if needed
       compare,
       divides, is_constant,
       find_in_set, preceding_indices, indices_in_box_and_after,
       factorizations, monomial_rank,
       num_multiindices_up_to

# ==================== Monomial order types ====================

"""
    abstract type MonomialOrder

Abstract type representing a monomial order. Concrete subtypes are
[`Lex`](@ref), [`Grlex`](@ref), and [`Grevlex`](@ref).
"""
abstract type MonomialOrder end

struct Lex <: MonomialOrder end
struct Grlex <: MonomialOrder end
struct Grevlex <: MonomialOrder end

# ==================== Core comparison functions ====================

"""
    lex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) -> Bool

Lexicographic order: `true` if `a` comes before `b` in the lexicographic order
(larger exponents in earlier components come first).
"""
function lex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    @assert length(a) == length(b)
    for i in eachindex(a)
        if a[i] != b[i]
            return a[i] > b[i]
        end
    end
    return false  # equal
end

"""
    grlex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) -> Bool

Graded lexicographic order: compare total degree first, then lexicographic.
"""
function grlex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    da, db = sum(a), sum(b)
    da != db && return da < db
    return lex_precede(a, b)
end

"""
    grevlex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) -> Bool

Graded reverse lexicographic order: compare total degree first,
then reverse lexicographic (compare from last component).
"""
function grevlex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    da, db = sum(a), sum(b)
    da != db && return da < db
    for i in length(a):-1:1 # reverse!
        if a[i] != b[i]
            return a[i] > b[i]
        end
    end
    return false
end

"""
    precede(::Lex, a, b) -> Bool
    precede(::Grlex, a, b) -> Bool
    precede(::Grevlex, a, b) -> Bool

Return `true` if `a` precedes `b` in the given monomial order.
"""
precede(::Lex, a, b) = lex_precede(a, b)
precede(::Grlex, a, b) = grlex_precede(a, b)
precede(::Grevlex, a, b) = grevlex_precede(a, b)

"""
    compare(a::Vector{Int}, b::Vector{Int}, order::MonomialOrder=Grlex()) -> Int

Return -1, 0, or 1 according to the given order (default `Grlex()`).
"""
function compare(a::Vector{Int}, b::Vector{Int}, order::MonomialOrder=Grlex())
    precede(order, a, b) && return -1
    precede(order, b, a) && return 1
    return 0
end

# ==================== MultiindexSet type ====================

"""
    MultiindexSet{O<:MonomialOrder}

A fixed collection of exponent vectors (multiindices) stored as columns of a matrix.
The set is guaranteed to be sorted according to the monomial order `O`.

# Fields
- `exponents::Matrix{Int}`: each column is an exponent vector.
  Rows correspond to variables, columns to monomials.
"""
struct MultiindexSet{O<:MonomialOrder}
    exponents::Matrix{Int}

    # Internal constructor with sortedness assertion (debug mode only)
    function MultiindexSet{O}(exponents::Matrix{Int}, ::Val{true}) where O
        if !is_sorted(exponents, O())
            error("MultiindexSet{$O} constructed with unsorted matrix")
        end
        new{O}(exponents)
    end
end

# Public constructor from matrix (sorts)
function MultiindexSet(exponents::Matrix{Int}, ::O) where O<:MonomialOrder
    exps_sorted = sort_exponents(exponents, O())
    MultiindexSet{O}(exps_sorted, Val(true))
end

# Constructor from vector of vectors (sorts)
function MultiindexSet(exponents::Vector{Vector{Int}}, ::O) where O<:MonomialOrder
    if isempty(exponents)
        return MultiindexSet{O}(Matrix{Int}(undef, 0, 0), Val(true))
    end
    nvars = length(exponents[1])
    @assert all(length(e) == nvars for e in exponents) "All exponent vectors must have the same length"
    mat = hcat(exponents...)
    return MultiindexSet(mat, O())
end

# Check that columns are non‑decreasing according to the order's strict less function.
function is_sorted(exponents::Matrix{Int}, order::MonomialOrder)
    n = size(exponents, 2)
    n ≤ 1 && return true
    less = (a,b) -> precede(order, a, b)
    for i in 1:n-1
        col_i = view(exponents, :, i)
        col_j = view(exponents, :, i+1)
        if !less(col_i, col_j) && col_i != col_j
            return false
        end
    end
    return true
end

# Helper to sort a matrix of exponents using views to avoid copies.
function sort_exponents(exponents::Matrix{Int}, order::MonomialOrder)
    n = size(exponents, 2)
    n == 0 && return exponents
    less = (a,b) -> precede(order, a, b)
    perm = sortperm(1:n; lt=(i, j) -> less(view(exponents, :, i), view(exponents, :, j)))
    return exponents[:, perm]
end

# ==================== Construction of multiindices ====================

"""
    zero_multiindex(n::Int) -> Vector{Int}

Return the zero exponent vector of length `n` (all components zero).
"""
zero_multiindex(n::Int) = zeros(Int, n)

"""
    multiindex(components::Int...) -> Vector{Int}

Convenience constructor for an exponent vector.
"""
multiindex(components::Int...) = collect(Int, components)

# ----- Generation in lex order (internal, preallocating) -----

# Generate all vectors of length n with total degree ≤ max_degree in lex order
# (as defined by lex_precede: larger exponents in earlier components come first).
function _generate_ascending_lex!(exponents::AbstractMatrix{Int}, n::Int, max_degree::Int)
    col = 1
    function recurse(prefix::Vector{Int}, remaining_vars::Int, remaining_deg::Int)
        if remaining_vars == 1
            for e in remaining_deg:-1:0
                @inbounds exponents[:, col] = vcat(prefix, e)
                col += 1
            end
        else
            for e in remaining_deg:-1:0
                recurse(vcat(prefix, e), remaining_vars-1, remaining_deg - e)
            end
        end
    end
    recurse(Int[], n, max_degree)
    return exponents
end

# Generate all vectors of length n with exact total degree in lex order.
function _generate_ascending_lex_fixed!(exponents::AbstractMatrix{Int}, n::Int, total_degree::Int)
    col = 1
    function recurse(prefix::Vector{Int}, remaining_vars::Int, remaining_deg::Int)
        if remaining_vars == 1
            @inbounds exponents[:, col] = vcat(prefix, remaining_deg)
            col += 1
        else
            for e in remaining_deg:-1:0
                recurse(vcat(prefix, e), remaining_vars-1, remaining_deg - e)
            end
        end
    end
    recurse(Int[], n, total_degree)
    return exponents
end

# Generate all vectors of length n with exact total degree in reverse lex order.
function _generate_ascending_revlex_fixed!(exponents::AbstractMatrix{Int}, n::Int, total_degree::Int)
    col = 1
    function recurse(prefix_rev::Vector{Int}, remaining_vars::Int, remaining_deg::Int)
        if remaining_vars == 1
            @inbounds exponents[:, col] = vcat(remaining_deg, reverse(prefix_rev))
            col += 1
        else
            for e in remaining_deg:-1:0
                recurse(vcat(prefix_rev, e), remaining_vars-1, remaining_deg - e)
            end
        end
    end
    recurse(Int[], n, total_degree)
    return exponents
end

# ----- Public generators returning MultiindexSet -----

"""
    all_multiindices_up_to(nvars::Int, max_degree::Int, order::MonomialOrder=Grlex()) -> MultiindexSet

Generate all exponent vectors with `nvars` variables whose total degree ≤ `max_degree`,
sorted according to the monomial order `order`.
Returns a `MultiindexSet`.
"""
function all_multiindices_up_to(nvars::Int, max_degree::Int, ::Lex)
    if nvars == 0
        return MultiindexSet{Lex}(Matrix{Int}(undef, 0, 0), Val(true))
    end
    total = binomial(max_degree + nvars, nvars)
    exponents = Matrix{Int}(undef, nvars, total)
    _generate_ascending_lex!(exponents, nvars, max_degree)
    return MultiindexSet{Lex}(exponents, Val(true))
end

function all_multiindices_up_to(nvars::Int, max_degree::Int, ::Grlex)
    if nvars == 0
        return MultiindexSet{Grlex}(Matrix{Int}(undef, 0, 0), Val(true))
    end
    # For Grlex, generate blocks by degree
    total = binomial(max_degree + nvars, nvars)
    exponents = Matrix{Int}(undef, nvars, total)
    col = 1
    for d in 0:max_degree
        block_size = binomial(d + nvars - 1, nvars - 1)
        block = view(exponents, :, col:col+block_size-1)
        _generate_ascending_lex_fixed!(block, nvars, d)
        col += block_size
    end
    return MultiindexSet{Grlex}(exponents, Val(true))
end

function all_multiindices_up_to(nvars::Int, max_degree::Int, ::Grevlex)
    if nvars == 0
        return MultiindexSet{Grevlex}(Matrix{Int}(undef, 0, 0), Val(true))
    end
    total = binomial(max_degree + nvars, nvars)
    exponents = Matrix{Int}(undef, nvars, total)
    col = 1
    for d in 0:max_degree
        block_size = binomial(d + nvars - 1, nvars - 1)
        block = view(exponents, :, col:col+block_size-1)
        _generate_ascending_revlex_fixed!(block, nvars, d)
        col += block_size
    end
    return MultiindexSet{Grevlex}(exponents, Val(true))
end

# Convenience method with default order
all_multiindices_up_to(nvars::Int, max_degree::Int) = all_multiindices_up_to(nvars, max_degree, Grlex())

"""
    multiindices_with_total_degree(nvars::Int, deg::Int, order::MonomialOrder=Grlex()) -> MultiindexSet

Generate all exponent vectors of length `nvars` with total degree exactly `deg`,
sorted according to the tie‑breaking part of the monomial order `order`.
For `Lex` and `Grlex` this is lexicographic order; for `Grevlex` it is reverse lexicographic.
Returns a `MultiindexSet`.
"""
function multiindices_with_total_degree(nvars::Int, degree::Int, ::Lex)
    if nvars == 0
        return MultiindexSet{Lex}(Matrix{Int}(undef, 0, 0), Val(true))
    end
    total = binomial(degree + nvars - 1, nvars - 1)
    exponents = Matrix{Int}(undef, nvars, total)
    _generate_ascending_lex_fixed!(exponents, nvars, degree)
    return MultiindexSet{Lex}(exponents, Val(true))
end

function multiindices_with_total_degree(nvars::Int, degree::Int, ::Grlex)
    if nvars == 0
        return MultiindexSet{Grlex}(Matrix{Int}(undef, 0, 0), Val(true))
    end
    # For Grlex, within fixed degree the order is lex, so same as Lex
    total = binomial(degree + nvars - 1, nvars - 1)
    exponents = Matrix{Int}(undef, nvars, total)
    _generate_ascending_lex_fixed!(exponents, nvars, degree)
    return MultiindexSet{Grlex}(exponents, Val(true))
end

function multiindices_with_total_degree(nvars::Int, degree::Int, ::Grevlex)
    if nvars == 0
        return MultiindexSet{Grevlex}(Matrix{Int}(undef, 0, 0), Val(true))
    end
    total = binomial(degree + nvars - 1, nvars - 1)
    exponents = Matrix{Int}(undef, nvars, total)
    _generate_ascending_revlex_fixed!(exponents, nvars, degree)
    return MultiindexSet{Grevlex}(exponents, Val(true))
end

multiindices_with_total_degree(nvars::Int, degree::Int) = multiindices_with_total_degree(nvars, degree, Grlex())

# ----- Box (hyperrectangle) generation -----

"""
    all_multiindices_in_box(bound::Vector{Int}, order::MonomialOrder=Grlex()) -> MultiindexSet

Generate all multi-indices `v` of length `length(bound)` such that
`0 ≤ v[i] ≤ bound[i]` for all `i`. The vectors are generated and then sorted
according to `order`.
"""
function all_multiindices_in_box(bound::Vector{Int}, ::O) where O<:MonomialOrder
    n = length(bound)
    if n == 0
        return MultiindexSet{O}(Matrix{Int}(undef, 0, 0), Val(true))
    end
    dims = bound .+ 1
    total = prod(dims)
    exponents = Matrix{Int}(undef, n, total)
    col = 1
    for idx in CartesianIndices(Tuple(dims))
        @inbounds for i in 1:n
            exponents[i, col] = idx[i] - 1
        end
        col += 1
    end
    return MultiindexSet(exponents, O())  # sort according to order
end

function all_multiindices_in_box(bound::Vector{Int})
    return all_multiindices_in_box(bound, Grlex())
end

"""
    all_multiindices_in_box(exponents::Matrix{Int}, order::MonomialOrder=Grlex()) -> MultiindexSet

Compute the component-wise maximum over the columns of `exponents` and generate
all multi-indices in the box from zero to that maximum.
"""
function all_multiindices_in_box(exponents::Matrix{Int}, order::MonomialOrder=Grlex())
    n = size(exponents, 1)
    bound = [maximum(exponents[i, :]) for i in 1:n]
    return all_multiindices_in_box(bound, order)
end

# ==================== Comparison predicates ====================

"""
    divides(a::Vector{Int}, b::Vector{Int}) -> Bool

Check whether `a` divides `b` componentwise, i.e., `a[i] ≤ b[i]` for all `i`.
"""
divides(a::Vector{Int}, b::Vector{Int}) = all(a .<= b)

"""
    is_constant(exp::Vector{Int}) -> Bool

Return true if the exponent vector is all zeros.
"""
is_constant(exp::Vector{Int}) = all(iszero, exp)

# ==================== Operations on MultiindexSet ====================

Base.length(S::MultiindexSet) = size(S.exponents, 2)
Base.getindex(S::MultiindexSet, i::Int) = S.exponents[:, i]
Base.iterate(S::MultiindexSet, state=1) = state > length(S) ? nothing : (S[state], state+1)
Base.collect(S::MultiindexSet) = [S.exponents[:, i] for i in 1:length(S)]

"""
    find_in_set(set::MultiindexSet{O}, exp::Vector{Int}) where O -> Union{Int, Nothing}

Return the column index of `exp` in `set.exponents` using binary search,
exploiting the fact that the set is sorted according to its stored order.
Returns `nothing` if `exp` is not present.
"""
function find_in_set(set::MultiindexSet{O}, exp::Vector{Int}) where O
    exps = set.exponents
    n = size(exps, 2)
    n == 0 && return nothing
    less = (a,b) -> precede(O(), a, b)
    lo, hi = 1, n
    while lo <= hi
        mid = (lo + hi) ÷ 2
        @inbounds v_mid = view(exps, :, mid)
        if v_mid == exp
            return mid
        elseif less(v_mid, exp)
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return nothing
end

"""
    preceding_indices(set::MultiindexSet{O}, exp::Vector{Int}) where O -> UnitRange{Int}

Return the indices of all vectors in `set` that are strictly less than `exp`
according to the set's monomial order. Assumes `set` is sorted by that order.
If `exp` itself is not in the set, the indices of all vectors less than `exp`
are returned (based on binary search).
"""
function preceding_indices(set::MultiindexSet{O}, exp::Vector{Int}) where O
    exps = set.exponents
    n = size(exps, 2)
    less = (a,b) -> precede(O(), a, b)
    lo, hi = 1, n
    pos = 0
    while lo <= hi
        mid = (lo + hi) ÷ 2
        @inbounds v_mid = view(exps, :, mid)
        if less(v_mid, exp)
            pos = mid
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return 1:pos
end

"""
    indices_in_box_and_after(set::MultiindexSet{O}, box_upper::Vector{Int}, other::Vector{Int}) where O -> Vector{Int}

Return the indices (column indices) of all multiindices in `set` that satisfy:
- `v[i] ≤ box_upper[i]` for all `i` (componentwise, "in the box sense"), and
- `v` appears after `other` in the sorted order of `set` (i.e., `other` is strictly less than `v` according to the set's monomial order).

If `other` is not in `set`, the condition is equivalent to `v` being greater than `other` in the order (using the set's comparison). The function performs a binary search to locate the start of the range of indices greater than `other`, then linearly scans the remainder of the set, checking the box condition.
"""
function indices_in_box_and_after(set::MultiindexSet{O}, box_upper::Vector{Int}, other::Vector{Int}) where O
    exps = set.exponents
    n = size(exps, 2)
    n == 0 && return Int[]

    nvars = size(exps, 1)
    @assert length(box_upper) == nvars "box_upper must have length $nvars"
    @assert length(other) == nvars "other must have length $nvars"

    less = (a,b) -> precede(O(), a, b)

    # Binary search for the last index with vector < other
    lo, hi = 1, n
    last_less = 0
    while lo <= hi
        mid = (lo + hi) ÷ 2
        @inbounds v_mid = view(exps, :, mid)
        if less(v_mid, other)
            last_less = mid
            lo = mid + 1
        else
            hi = mid - 1
        end
    end

    # Start index = first index with vector >= other
    start_idx = last_less + 1

    # Skip any vectors exactly equal to other (since we need strictly greater)
    while start_idx <= n
        @inbounds v_start = view(exps, :, start_idx)
        v_start == other ? (start_idx += 1) : break
    end

    # Linear scan from start_idx to end, collecting indices that satisfy box condition
    result = Int[]
    @inbounds for i in start_idx:n
        v = view(exps, :, i)
        if all(v .<= box_upper)
            push!(result, i)
        end
    end
    return result
end

# ==================== Factorizations ====================

"""
    factorizations(set::MultiindexSet{O}, exp::Vector{Int}, N::Int) where O -> Vector{Vector{Vector{Int}}}

Return all ordered `N`-tuples of exponent vectors from `set` whose sum equals `exp`.
Each factorization is represented as a vector of `N` exponent vectors (in the order they were selected).
The outer list is sorted by the same monomial order stored in `set`.
If no factorisation exists, an empty vector is returned.
"""
function factorizations(set::MultiindexSet{O}, exp::Vector{Int}, N::Int) where O
    # Quick return for N == 0
    if N == 0
        return iszero(exp) ? [Vector{Vector{Int}}()] : Vector{Vector{Vector{Int}}}()
    end

    exps = set.exponents
    M = size(exps, 2)
    nvars = length(exp)
    @assert size(exps, 1) == nvars "Length of exponent vector must match number of variables in the set"

    results = Vector{Vector{Vector{Int}}}()
    # Store indices of chosen factors to avoid copying vectors until final
    current_idxs = Vector{Int}(undef, N)
    current_sum = zeros(Int, nvars)

    function backtrack(depth::Int)
        if depth == 0
            if current_sum == exp
                # Build factorisation from indices
                fact = [exps[:, current_idxs[i]] for i in 1:N]
                push!(results, fact)
            end
            return
        end
        # Try every index – order matters, so always start from 1
        for i in 1:M
            v = view(exps, :, i)
            # Prune: any component would exceed target?
            any(>(0), v .+ current_sum .- exp) && continue
            # Add in place
            current_sum .+= v
            current_idxs[N-depth+1] = i
            backtrack(depth - 1)
            current_sum .-= v
        end
    end

    backtrack(N)

    isempty(results) && return Vector{Vector{Vector{Int}}}()

    # Sort outer list lexicographically using the order
    less_fact(a, b) = _fact_less(a, b, O())
    sort!(results, lt=less_fact)
    return results
end

# Lexicographic comparison of two factorisations (both internally sorted by construction)
function _fact_less(a::Vector{Vector{Int}}, b::Vector{Vector{Int}}, order::MonomialOrder)
    for (va, vb) in zip(a, b)
        if precede(order, va, vb)
            return true
        elseif precede(order, vb, va)
            return false
        end
    end
    return false  # equal
end

# ==================== Mathematical ranking for complete bases ====================

"""
    binomial(n::Int, k::Int) -> Integer

Safe binomial coefficient, returning 0 if `k < 0` or `k > n`.
"""
function binomial(n::Int, k::Int)
    if k < 0 || k > n
        return 0
    end
    return Base.binomial(n, k)
end

"""
    num_multiindices_up_to(nvars::Int, max_degree::Int) -> Integer

Return the number of exponent vectors of length `nvars` with total degree ≤ `max_degree`.
For `nvars = 0` the set is empty, so the function returns `0`.
"""
num_multiindices_up_to(nvars::Int, max_degree::Int) = nvars == 0 ? 0 : binomial(max_degree + nvars, nvars)

"""
    monomial_rank(exp::Vector{Int}, nvars::Int, max_degree::Int, order::MonomialOrder) -> Int

Return the 1‑based rank of the exponent vector `exp` in the complete set of all
multiindices of length `nvars` with total degree ≤ `max_degree`, sorted according to
the monomial order `order` (only `Lex` and `Grlex` are implemented combinatorially).
For `Grevlex`, use `find_in_set` on a generated `MultiindexSet` instead.

The vector `exp` must satisfy `length(exp) == nvars` and `sum(exp) ≤ max_degree`.
Uses combinatorial counting formulas for efficiency (O(nvars) time).

If the computed rank exceeds `typemax(Int)`, an error is thrown.
"""
function monomial_rank(exp::Vector{Int}, nvars::Int, max_degree::Int, ::Lex)
    if nvars == 0
        error("monomial_rank not defined for nvars = 0")
    end
    @assert length(exp) == nvars
    total_deg = sum(exp)
    @assert total_deg ≤ max_degree
    rank = _lex_rank(exp, nvars, max_degree)
    if rank > typemax(Int)
        throw(OverflowError("Computed rank $rank exceeds typemax(Int) ($(typemax(Int)))."))
    end
    return Int(rank)
end

function monomial_rank(exp::Vector{Int}, nvars::Int, max_degree::Int, ::Grlex)
    if nvars == 0
        error("monomial_rank not defined for nvars = 0")
    end
    @assert length(exp) == nvars
    total_deg = sum(exp)
    @assert total_deg ≤ max_degree
    rank = _grlex_rank(exp, nvars, max_degree)
    if rank > typemax(Int)
        throw(OverflowError("Computed rank $rank exceeds typemax(Int) ($(typemax(Int)))."))
    end
    return Int(rank)
end

function monomial_rank(exp::Vector{Int}, nvars::Int, max_degree::Int, ::Grevlex)
    error("Direct combinatorial rank for Grevlex not implemented. " *
          "Generate the set with `all_multiindices_up_to` and use `find_in_set`.")
end

# Lexicographic rank among all vectors of length n with total degree ≤ max_degree (1‑based)
function _lex_rank(exp::Vector{Int}, n::Int, max_degree::Int)
    if n == 1
        # descending order: e = max_degree, max_degree-1, ..., 0
        return max_degree - exp[1] + 1
    end
    a = exp[1]
    count_before = 0
    # count vectors with first component larger than a (they come before)
    for a0 in (a+1):max_degree
        count_before += num_multiindices_up_to(n - 1, max_degree - a0)
    end
    # vectors with first component equal to a: rank determined by the tail
    rest_rank = _lex_rank(exp[2:end], n-1, max_degree - a)
    return count_before + rest_rank
end

# Graded lexicographic rank among all vectors of length n with total degree ≤ max_degree.
function _grlex_rank(exp::Vector{Int}, n::Int, max_degree::Int)
    total_deg = sum(exp)
    count_before = 0
    for d in 0:(total_deg-1)
        count_before += binomial(d + n - 1, n - 1) 
    end
    lex_rank_within_deg = _lex_rank_fixed_degree(exp, n, total_deg) - 1
    return 1 + count_before + lex_rank_within_deg
end

# Lexicographic rank among vectors of exactly total_degree (1‑based)
function _lex_rank_fixed_degree(exp::Vector{Int}, n::Int, total_degree::Int)
    if n == 1
        @assert exp[1] == total_degree
        return 1
    end
    a = exp[1]
    count_before = 0
    # Vectors with first component > a come before
    for a0 in (a+1):total_degree
        count_before += binomial(total_degree - a0 + n - 2, n - 2)
    end
    # Recursive rank of the rest (first component = a)
    rest_rank = _lex_rank_fixed_degree(exp[2:end], n-1, total_degree - a)
    return count_before + rest_rank
end

end # module