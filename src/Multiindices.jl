module Multiindices

export MultiindexSet,
       zero_multiindex, multiindex,
       all_multiindices_up_to, multiindices_with_total_degree,
       all_multiindices_in_box,
       grlex_precede, compare, # if direct comparison is needed
       divides, is_constant,
       find_in_set, indices_in_box_and_after,
       factorizations, monomial_rank,
       num_multiindices_up_to

# ==================== Core comparison functions ====================

"""
    grlex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) -> Bool

Graded lexicographic order: compare total degree first, then lexicographic.
Returns `true` if `a` comes before `b` in this order.
"""
function grlex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    da, db = sum(a), sum(b)
    da != db && return da < db
    for i in eachindex(a)
        if a[i] != b[i]
            return a[i] > b[i]   # lexicographic (larger first component comes first)
        end
    end
    return false  # equal
end

"""
    compare(a::Vector{Int}, b::Vector{Int}) -> Int

Return -1, 0, or 1 according to the graded lexicographic order.
"""
function compare(a::Vector{Int}, b::Vector{Int})
    grlex_precede(a, b) && return -1
    grlex_precede(b, a) && return 1
    return 0
end

# ==================== MultiindexSet type ====================

"""
    MultiindexSet

A fixed collection of exponent vectors (multiindices) stored as columns of a matrix.
The set is guaranteed to be sorted according to the graded lexicographic (Grlex) order.

# Fields
- `exponents::Matrix{Int}`: each column is an exponent vector.
  Rows correspond to variables, columns to monomials.
"""
struct MultiindexSet
    exponents::Matrix{Int}

    # Internal constructor with sortedness assertion (debug mode only)
    function MultiindexSet(exponents::Matrix{Int}, ::Val{true})
        if !is_sorted(exponents)
            error("MultiindexSet constructed with unsorted matrix")
        end
        new(exponents)
    end
end

# Public constructor from matrix (sorts in Grlex)
function MultiindexSet(exponents::Matrix{Int})
    exps_sorted = sort_exponents(exponents)
    MultiindexSet(exps_sorted, Val(true))
end

# Constructor from vector of vectors (sorts in Grlex)
function MultiindexSet(exponents::Vector{Vector{Int}})
    if isempty(exponents)
        return MultiindexSet(Matrix{Int}(undef, 0, 0), Val(true))
    end
    nvars = length(exponents[1])
    @assert all(length(e) == nvars for e in exponents) "All exponent vectors must have the same length"
    mat = hcat(exponents...)
    return MultiindexSet(mat)
end

# Check that columns are non‑decreasing according to Grlex order.
function is_sorted(exponents::Matrix{Int})
    n = size(exponents, 2)
    n ≤ 1 && return true
    for i in 1:n-1
        col_i = view(exponents, :, i)
        col_j = view(exponents, :, i+1)
        if !grlex_precede(col_i, col_j) && col_i != col_j
            return false
        end
    end
    return true
end

# Helper to sort a matrix of exponents using views to avoid copies.
function sort_exponents(exponents::Matrix{Int})
    n = size(exponents, 2)
    n == 0 && return exponents
    perm = sortperm(1:n; lt=(i, j) -> grlex_precede(view(exponents, :, i), view(exponents, :, j)))
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

# ----- Public generators returning MultiindexSet -----

"""
    all_multiindices_up_to(nvars::Int, max_degree::Int) -> MultiindexSet

Generate all exponent vectors with `nvars` variables whose total degree ≤ `max_degree`,
sorted according to graded lexicographic order.
Returns a `MultiindexSet`.
"""
function all_multiindices_up_to(nvars::Int, max_degree::Int)
    if nvars == 0
        return MultiindexSet(Matrix{Int}(undef, 0, 0), Val(true))
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
    return MultiindexSet(exponents, Val(true))
end

"""
    multiindices_with_total_degree(nvars::Int, deg::Int) -> MultiindexSet

Generate all exponent vectors of length `nvars` with total degree exactly `deg`,
sorted in lexicographic order (the tie‑breaker for Grlex).
Returns a `MultiindexSet`.
"""
function multiindices_with_total_degree(nvars::Int, degree::Int)
    if nvars == 0
        return MultiindexSet(Matrix{Int}(undef, 0, 0), Val(true))
    end
    total = binomial(degree + nvars - 1, nvars - 1)
    exponents = Matrix{Int}(undef, nvars, total)
    _generate_ascending_lex_fixed!(exponents, nvars, degree)
    return MultiindexSet(exponents, Val(true))
end

# ----- Box (hyperrectangle) generation -----

"""
    all_multiindices_in_box(bound::Vector{Int}) -> MultiindexSet

Generate all multi-indices `v` of length `length(bound)` such that
`0 ≤ v[i] ≤ bound[i]` for all `i`. The vectors are generated and then sorted
according to graded lexicographic order.
"""
function all_multiindices_in_box(bound::Vector{Int})
    n = length(bound)
    if n == 0
        return MultiindexSet(Matrix{Int}(undef, 0, 0), Val(true))
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
    return MultiindexSet(exponents)  # sort according to Grlex
end

"""
    all_multiindices_in_box(exponents::Matrix{Int}) -> MultiindexSet

Compute the component-wise maximum over the columns of `exponents` and generate
all multi-indices in the box from zero to that maximum.
"""
function all_multiindices_in_box(exponents::Matrix{Int})
    n = size(exponents, 1)
    bound = [maximum(exponents[i, :]) for i in 1:n]
    return all_multiindices_in_box(bound)
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
    find_in_set(set::MultiindexSet, exp::Vector{Int}) -> Union{Int, Nothing}

Return the column index of `exp` in `set.exponents` using binary search,
exploiting the fact that the set is sorted according to Grlex.
Returns `nothing` if `exp` is not present.
"""
function find_in_set(set::MultiindexSet, exp::Vector{Int})
    exps = set.exponents
    n = size(exps, 2)
    n == 0 && return nothing
    lo, hi = 1, n
    while lo <= hi
        mid = (lo + hi) ÷ 2
        @inbounds v_mid = view(exps, :, mid)
        if v_mid == exp
            return mid
        elseif grlex_precede(v_mid, exp)
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return nothing
end

# -------------------------------------------------------------------
# Helper: first index strictly after `other` in Grlex order
# -------------------------------------------------------------------
"""
    _first_index_after(set::MultiindexSet, other::Vector{Int}) -> Int

Return the first column index `i` in `set.exponents` such that
`set[i]` is strictly greater than `other` according to Grlex order.
If no such index exists, return `size(set.exponents, 2) + 1`.
Assumes the set is sorted by Grlex.
"""
function _first_index_after(set::MultiindexSet, other::Vector{Int})
    exps = set.exponents
    n = size(exps, 2)
    n == 0 && return 1

    # Binary search for the last index with vector < other (Grlex)
    lo, hi = 1, n
    last_less = 0
    while lo <= hi
        mid = (lo + hi) ÷ 2
        @inbounds v_mid = view(exps, :, mid)
        if grlex_precede(v_mid, other)   # v_mid < other ?
            last_less = mid
            lo = mid + 1
        else
            hi = mid - 1
        end
    end

    start_idx = last_less + 1
    # Skip any vectors exactly equal to other (strictly greater required)
    while start_idx <= n
        @inbounds v_start = view(exps, :, start_idx)
        v_start == other ? (start_idx += 1) : break
    end
    return start_idx
end

# -------------------------------------------------------------------
# Helper: for Grlex, last index with total degree < max_total_deg
# -------------------------------------------------------------------
"""
    _last_index_below_degree(set::MultiindexSet, start_idx::Int, max_total_deg::Int) -> Int

Return the last column index `i` (≥ `start_idx`) in the Grlex‑sorted set
such that the total degree of `set[i]` is strictly less than `max_total_deg`.
If no such index exists, return `start_idx - 1`.
"""
function _last_index_below_degree(set::MultiindexSet, start_idx::Int, max_total_deg::Int)
    exps = set.exponents
    n = size(exps, 2)
    start_idx > n && return start_idx - 1

    lo, hi = start_idx, n
    last = start_idx - 1
    while lo <= hi
        mid = (lo + hi) ÷ 2
        @inbounds v_mid = view(exps, :, mid)
        if sum(v_mid) < max_total_deg
            last = mid
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return last
end

# -------------------------------------------------------------------
# Helper: collect indices in a range that satisfy v ≤ box_upper
# -------------------------------------------------------------------
"""
    _collect_indices_in_range_inside_box(exps::Matrix{Int}, range::UnitRange{Int},
                              box_upper::Vector{Int}) -> Vector{Int}

Scan the columns of `exps` over the given `range` and return the indices
of those vectors `v` for which `v[i] ≤ box_upper[i]` for all `i`.
The scan uses a short‑circuiting componentwise check to exit early on the first violation.
"""
function _collect_indices_in_range_inside_box(exps::Matrix{Int}, range::UnitRange{Int}, box_upper::Vector{Int})
    result = Int[]
    sizehint!(result, length(range))
    nvars = length(box_upper)

    @inbounds for col in range
        v = view(exps, :, col)
        # Manual short‑circuiting check: v ≤ box_upper componentwise
        ok = true
        for i in 1:nvars
            if v[i] > box_upper[i]
                ok = false
                break
            end
        end
        ok || continue
        push!(result, col)
    end
    return result
end

"""
    indices_in_box_and_after(set::MultiindexSet, box_upper::Vector{Int}, other::Vector{Int}) -> Vector{Int}

Return the column indices of all multiindices in `set` that satisfy:
- `v[i] ≤ box_upper[i]` for all `i` (componentwise),
- `v` appears after `other` in Grlex order,
- and implicitly `v ≠ box_upper` (enforced by the total‑degree bound).
"""
function indices_in_box_and_after(set::MultiindexSet, box_upper::Vector{Int}, other::Vector{Int})
    exps = set.exponents
    n = size(exps, 2)
    n == 0 && return Int[]

    nvars = size(exps, 1)
    @assert length(box_upper) == nvars "box_upper must have length $nvars"
    @assert length(other) == nvars "other must have length $nvars"

    # 1. Find the first index strictly greater than `other`
    start_idx = _first_index_after(set, other)
    start_idx > n && return Int[]

    # 2. Restrict to vectors with total degree < sum(box_upper)
    max_total_deg = sum(box_upper)
    end_idx = _last_index_below_degree(set, start_idx, max_total_deg)
    end_idx < start_idx && return Int[]

    # 3. Collect indices satisfying the box condition.
    #    No need to exclude box_upper explicitly because degree bound already does.
    return _collect_indices_in_range_inside_box(exps, start_idx:end_idx, box_upper)
end

# ==================== Factorizations ====================

"""
    factorizations(set::MultiindexSet, exp::Vector{Int}, N::Int) -> Vector{Vector{Vector{Int}}}

Return all ordered `N`-tuples of exponent vectors from `set` whose sum equals `exp`.
Each factorization is represented as a vector of `N` exponent vectors (in the order they were selected).
The outer list is sorted by graded lexicographic order (comparing factorizations lexicographically).
If no factorisation exists, an empty vector is returned.
"""
function factorizations(set::MultiindexSet, exp::Vector{Int}, N::Int)
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

    # Sort outer list lexicographically using Grlex
    less_fact(a, b) = _fact_less(a, b)
    sort!(results, lt=less_fact)
    return results
end

# Lexicographic comparison of two factorisations (both internally sorted by construction)
function _fact_less(a::Vector{Vector{Int}}, b::Vector{Vector{Int}})
    for (va, vb) in zip(a, b)
        if grlex_precede(va, vb)
            return true
        elseif grlex_precede(vb, va)
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
    monomial_rank(exp::Vector{Int}, nvars::Int, max_degree::Int) -> Int

Return the 1‑based rank of the exponent vector `exp` in the complete set of all
multiindices of length `nvars` with total degree ≤ `max_degree`, sorted according to
graded lexicographic order (Grlex).

The vector `exp` must satisfy `length(exp) == nvars` and `sum(exp) ≤ max_degree`.
Uses combinatorial counting formulas for efficiency (O(nvars) time).

If the computed rank exceeds `typemax(Int)`, an error is thrown.
"""
function monomial_rank(exp::Vector{Int}, nvars::Int, max_degree::Int)
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