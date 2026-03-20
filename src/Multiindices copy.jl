module Multiindices

export MultiindexSet, zero_multiindex,
       all_multiindices_up_to, multiindices_with_total_degree, 
       all_multiindices_in_box, indices_in_box_with_bounded_degree,
       divides, is_constant, find_in_set, build_exponent_index_map,
       factorisations_asymmetric, factorisations_fully_symmetric, factorisations_groupwise_symmetric

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
    grlex_precede(t::NTuple{N,Int}, v::AbstractVector{Int}) -> Bool

Compare a tuple (representing an exponent) with a vector in graded lexicographic order.
Useful for binary search without allocating a vector.
"""
function grlex_precede(t::NTuple{N,Int}, v::AbstractVector{Int}) where N
    dt = sum(t)
    dv = sum(v)
    dt != dv && return dt < dv
    for i in 1:N
        ti = t[i]
        vi = v[i]
        ti != vi && return ti > vi
    end
    return false
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
    # Precompute total degrees for faster comparison
    degrees = [sum(view(exponents, :, i)) for i in 1:n]
    for i in 1:n-1
        if degrees[i] != degrees[i+1]
            degrees[i] < degrees[i+1] || return false
        else
            col_i = view(exponents, :, i)
            col_j = view(exponents, :, i+1)
            for k in eachindex(col_i)
                if col_i[k] != col_j[k]
                    col_i[k] > col_j[k] || return false
                    break
                end
            end
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
    total = _binomial(max_degree + nvars, nvars)
    exponents = Matrix{Int}(undef, nvars, total)
    col = 1
    for d in 0:max_degree
        block_size = _binomial(d + nvars - 1, nvars - 1)
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
    total = _binomial(degree + nvars - 1, nvars - 1)
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


# ==================== Comparison predicates ====================

"""
    divides(a::AbstractVector{Int}, b::AbstractVector{Int}) -> Bool

Check whether `a` divides `b` componentwise, i.e., `a[i] ≤ b[i]` for all `i`.
"""
divides(a::AbstractVector{Int}, b::AbstractVector{Int}) = all(a .<= b)

"""
    is_constant(exp::AbstractVector{Int}) -> Bool

Return true if the exponent vector is all zeros.
"""
is_constant(exp::AbstractVector{Int}) = all(iszero, exp)

# ==================== Operations on MultiindexSet ====================

Base.length(S::MultiindexSet) = size(S.exponents, 2)
Base.getindex(S::MultiindexSet, i::Int) = S.exponents[:, i]
Base.iterate(S::MultiindexSet, state=1) = state > length(S) ? nothing : (S[state], state+1)
Base.collect(S::MultiindexSet) = [S.exponents[:, i] for i in 1:length(S)]

"""
    find_in_set(set::MultiindexSet, exp::AbstractVector{Int}) -> Union{Int, Nothing}

Return the column index of `exp` in `set.exponents` using binary search,
exploiting the fact that the set is sorted according to Grlex.
Returns `nothing` if `exp` is not present.
"""
function find_in_set(set::MultiindexSet, exp::AbstractVector{Int})
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

"""
    find_in_set(set::MultiindexSet, exp::NTuple{N,Int}) where N -> Union{Int, Nothing}

Tuple version – avoids allocating a vector for the exponent.
"""
function find_in_set(set::MultiindexSet, exp::NTuple{N,Int}) where N
    exps = set.exponents
    n = size(exps, 2)
    n == 0 && return nothing
    lo, hi = 1, n
    while lo <= hi
        mid = (lo + hi) ÷ 2
        @inbounds col = view(exps, :, mid)
        # Manual comparison for efficiency and type stability
        equal = true
        for i in 1:N
            if exp[i] != col[i]
                equal = false
                break
            end
        end
        if equal
            return mid
        elseif grlex_precede(exp, col)  # exp < col ?
            hi = mid - 1
        else
            lo = mid + 1
        end
    end
    return nothing
end

"""
    _last_index_below_degree(set::MultiindexSet, max_total_deg::Int) -> Int

Return the last column index `i` in the Grlex‑sorted set such that the total
degree of `set[i]` is strictly less than `max_total_deg`. If no such index
exists, return `0`.
"""
function _last_index_below_degree(set::MultiindexSet, max_total_deg::Int)
    exps = set.exponents
    n = size(exps, 2)
    lo, hi = 1, n
    last = 0
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

"""
    _last_index_below_degree(set::MultiindexSet, max_total_deg::Int,
                             allowed_indices::AbstractVector{Int}) -> Int

Same as the 2‑argument version, but the search is restricted to the indices
listed in `allowed_indices` (which must be sorted in increasing order).
The returned value is still an actual column index (or 0 if none qualifies).
"""
function _last_index_below_degree(set::MultiindexSet, max_total_deg::Int,
                                  allowed_indices::AbstractVector{Int})
    isempty(allowed_indices) && return 0
    exps = set.exponents
    lo, hi = 1, length(allowed_indices)
    last = 0
    while lo <= hi
        mid = (lo + hi) ÷ 2
        i = allowed_indices[mid]
        @inbounds v_mid = view(exps, :, i)
        if sum(v_mid) < max_total_deg
            last = i          # store the actual index, not the position
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return last
end

"""
    indices_in_box_with_bounded_degree(set::MultiindexSet, box_upper::AbstractVector{Int},
                                       degree_lower_bound::Int, total_deg_upper::Int) -> Vector{Int}

Return the column indices of all multiindices `v` in `set` such that
- `v ≤ box_upper` componentwise,
- `degree_lower_bound ≤ sum(v[i]) < total_deg_upper`.

Uses the degree bounds to limit the search to relevant columns.
"""
function indices_in_box_with_bounded_degree(set::MultiindexSet, box_upper::AbstractVector{Int},
                                            degree_lower_bound::Int, total_deg_upper::Int)
    @assert length(box_upper) == size(set.exponents, 1)
    exps = set.exponents
    n = size(exps, 2)
    n == 0 && return Int[]

    first_idx = _last_index_below_degree(set, degree_lower_bound) + 1
    last_idx  = _last_index_below_degree(set, total_deg_upper)

    result = Int[]
    @inbounds for i in first_idx:last_idx
        v = view(exps, :, i)
        if all(v .<= box_upper)
            push!(result, i)
        end
    end
    return result
end

"""
    indices_in_box_with_bounded_degree(set::MultiindexSet, box_upper::AbstractVector{Int},
                                       degree_lower_bound::Int, total_deg_upper::Int,
                                       allowed_indices::AbstractVector{Int}) -> Vector{Int}

Same as the 4‑argument version, but the search is restricted to the indices
listed in `allowed_indices` (which must be sorted). Only those indices that
also satisfy the componentwise bound are returned.
"""
function indices_in_box_with_bounded_degree(set::MultiindexSet, box_upper::AbstractVector{Int},
                                            degree_lower_bound::Int, total_deg_upper::Int,
                                            allowed_indices::AbstractVector{Int})
    @assert length(box_upper) == size(set.exponents, 1)
    exps = set.exponents
    isempty(allowed_indices) && return Int[]

    # Find the last allowed index with degree < degree_lower_bound
    # and the last allowed index with degree < total_deg_upper.
    last_below_lower = _last_index_below_degree(set, degree_lower_bound, allowed_indices)
    last_below_upper = _last_index_below_degree(set, total_deg_upper, allowed_indices)

    # The first allowed index with degree ≥ degree_lower_bound is the one
    # after last_below_lower. Convert to positions in allowed_indices.
    lo_pos = searchsortedfirst(allowed_indices, last_below_lower + 1)
    hi_pos = searchsortedlast(allowed_indices, last_below_upper)

    result = Int[]
    @inbounds for pos in lo_pos:hi_pos
        i = allowed_indices[pos]
        v = view(exps, :, i)
        if all(v .<= box_upper)
            push!(result, i)
        end
    end
    return result
end

# ==================== Factorizations ====================

"""
    factorisations_asymmetric(set::MultiindexSet, exp::AbstractVector{Int}, N::Int, candidate_indices::AbstractVector{Int}) -> Vector{NTuple{N,Int}}

Return all ordered `N`-tuples of indices (taken from `candidate_indices`) whose corresponding exponent vectors sum to `exp`.
Each factorisation is an `NTuple{N,Int}` where the `k`-th entry is the index of the `k`-th factor.
If no factorisation exists, an empty vector is returned.
"""
function factorisations_asymmetric(set::MultiindexSet, exp::AbstractVector{Int}, N::Int, candidate_indices::AbstractVector{Int})
    # Quick return for N == 0
    if N == 0
        return iszero(exp) ? NTuple{0,Int}[()] : NTuple{0,Int}[]
    end

    exps = set.exponents
    nvars = length(exp)
    @assert size(exps, 1) == nvars "Length of exponent vector must match number of variables in the set"

    results = NTuple{N,Int}[]
    current_idxs = Vector{Int}(undef, N)
    current_sum = zeros(Int, nvars)

    function backtrack(depth::Int)
        if depth == 0
            if current_sum == exp
                push!(results, Tuple(current_idxs))
            end
            return
        end
        for i in candidate_indices
            v = view(exps, :, i)
            # Prune: adding v would exceed exp in any component
            any(>(0), v .+ current_sum .- exp) && continue
            current_sum .+= v
            current_idxs[N-depth+1] = i
            backtrack(depth - 1)
            current_sum .-= v
        end
    end

    backtrack(N)

    return results
end

"""
    factorisations_fully_symmetric(set::MultiindexSet, exp::AbstractVector{Int}, N::Int, candidate_indices::AbstractVector{Int}) -> Vector{Tuple{NTuple{N,Int}, Int}}

Return all `N`-tuples of indices (taken from `candidate_indices`) with non‑decreasing indices whose corresponding exponent vectors sum to `exp`.
For each such combination, the result includes the tuple itself and the number of distinct ordered factorisations (permutations) that can be formed from it.

The number of permutations equals `N! / (m₁! m₂! … mₖ!)` where `mⱼ` are the multiplicities of each distinct index in the tuple.

The input `candidate_indices` is assumed to contain unique indices; duplicate entries are removed internally.

If no factorisation exists, an empty vector is returned.
"""
function factorisations_fully_symmetric(set::MultiindexSet, exp::AbstractVector{Int}, N::Int, candidate_indices::AbstractVector{Int})
    # Quick return for N == 0
    if N == 0
        return iszero(exp) ? [((), 1)] : Tuple{NTuple{0,Int},Int}[]
    end

    exps = set.exponents
    nvars = length(exp)
    @assert size(exps, 1) == nvars "Length of exponent vector must match number of variables in the set"

    # Remove duplicates and sort candidate indices to allow non‑decreasing enforcement
    sorted_candidates = sort(unique(candidate_indices))
    L = length(sorted_candidates)

    results = Vector{Tuple{NTuple{N,Int}, Int}}()
    current_idxs = Vector{Int}(undef, N)   # indices chosen so far
    current_sum = zeros(Int, nvars)

    function backtrack(depth::Int, start_pos::Int)
        if depth == 0
            if current_sum == exp
                # Compute number of permutations (_multinomial coefficient)
                # current_idxs is already sorted non‑decreasingly
                counts = Int[]
                i = 1
                while i <= N
                    j = i
                    while j <= N && current_idxs[j] == current_idxs[i]
                        j += 1
                    end
                    push!(counts, j - i)
                    i = j
                end
                perm_count = _multinomial(N, counts)
                push!(results, (Tuple(current_idxs), perm_count))
            end
            return
        end
        for pos in start_pos:L
            idx = sorted_candidates[pos]
            v = view(exps, :, idx)
            # Prune: adding v would exceed exp in any component
            any(>(0), v .+ current_sum .- exp) && continue
            current_sum .+= v
            current_idxs[N-depth+1] = idx
            backtrack(depth - 1, pos)      # allow same index again (non‑decreasing)
            current_sum .-= v
        end
    end

    backtrack(N, 1)

    return results
end

"""
    factorisations_groupwise_symmetric(set::MultiindexSet, exp::AbstractVector{Int},
                                        group_sizes::NTuple{M,Int},
                                        candidate_indices::AbstractVector{Int}) where M

Return factorisations of the exponent vector `exp` into `M` groups, where group `i`
has size `group_sizes[i]` and is symmetric within itself (permutations inside a group are equivalent).

The result is a vector of `(flat_indices, total_count)`:
- `flat_indices`: concatenation of the indices for each group in group order,
  with each group's indices sorted non‑decreasingly (canonical representation).
- `total_count`: number of ordered factorisations (full sequences) corresponding to this combination.

The algorithm recursively processes groups. For the current group of size `k` and remaining exponent `rem`,
it enumerates all sub‑exponents `s` with `0 ≤ s ≤ rem`. For each `s`, it calls
`factorisations_fully_symmetric(set, s, k, global_candidates)` to obtain all unordered `k`-tuples
(indices with multiplicities) summing to `s`. Their permutation counts are multiplied into the total,
and recursion continues with `rem - s`. When all groups are processed and `rem = 0`, the combination is recorded.

Returns an empty vector if no factorisation exists.
"""
function factorisations_groupwise_symmetric(set::MultiindexSet, exp::AbstractVector{Int},
                                    group_sizes::NTuple{M,Int},
                                    candidate_indices::AbstractVector{Int}) where M
    exps = set.exponents
    nvars = length(exp)
    @assert size(exps, 1) == nvars "Exponent vector length must match number of variables"

    # Global list of all possible indices (sorted, unique)
    global_candidates = sort(unique(candidate_indices))

    results = Vector{Tuple{Vector{Int}, Int}}()

    # Recursive function over groups
    function recurse_groups(group_idx::Int, remaining::Vector{Int},
                            current_flat::Vector{Int}, current_count::Int)
        if group_idx > M
            # All groups processed – success if nothing left
            iszero(remaining) && push!(results, (copy(current_flat), current_count))
            return
        end

        k = group_sizes[group_idx]
        if k == 0
            # Empty group: move to next without changing anything
            recurse_groups(group_idx + 1, remaining, current_flat, current_count)
            return
        end

        # Enumerate all possible sub‑exponents s with 0 ≤ s ≤ remaining (componentwise)
        ranges = [0:remaining[i] for i in 1:nvars]
        range_tuple = Tuple(ranges)  # convert to tuple for CartesianIndices

        for s_idx in CartesianIndices(range_tuple)
            s = [s_idx[i] for i in 1:nvars]   # current sub‑exponent for this group

            # Get all unordered factorisations of s with exactly k indices
            unordered = factorisations_fully_symmetric(set, s, k, global_candidates)

            for (tuple, perm_count) in unordered
                # tuple is an NTuple{k,Int} (sorted indices)
                new_remaining = remaining .- s
                if all(≥(0), new_remaining)
                    # Append this group's indices to the flat list
                    append!(current_flat, collect(tuple))
                    recurse_groups(group_idx + 1, new_remaining,
                                   current_flat, current_count * perm_count)
                    # Backtrack: remove the indices we just added
                    for _ in 1:k
                        pop!(current_flat)
                    end
                end
            end
        end
    end

    recurse_groups(1, exp, Int[], 1)
    return results
end

# ==================== Mathematical ranking for complete bases ====================

"""
    _binomial(n::Int, k::Int) -> Integer

Safe binomial coefficient, returning 0 if `k < 0` or `k > n`.
"""
function _binomial(n::Int, k::Int)
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
num_multiindices_up_to(nvars::Int, max_degree::Int) = nvars == 0 ? 0 : _binomial(max_degree + nvars, nvars)

"""
    monomial_rank(exp::AbstractVector{Int}, nvars::Int, max_degree::Int) -> Int

Return the 1‑based rank of the exponent vector `exp` in the complete set of all
multiindices of length `nvars` with total degree ≤ `max_degree`, sorted according to
graded lexicographic order (Grlex).

The vector `exp` must satisfy `length(exp) == nvars` and `sum(exp) ≤ max_degree`.
Uses combinatorial counting formulas for efficiency (O(nvars) time).

If the computed rank exceeds `typemax(Int)`, an error is thrown.
"""
function monomial_rank(exp::AbstractVector{Int}, nvars::Int, max_degree::Int)
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
function _grlex_rank(exp::AbstractVector{Int}, n::Int, max_degree::Int)
    total_deg = sum(exp)
    count_before = 0
    for d in 0:(total_deg-1)
        count_before += _binomial(d + n - 1, n - 1) 
    end
    lex_rank_within_deg = _lex_rank_fixed_degree(exp, n, total_deg) - 1
    return 1 + count_before + lex_rank_within_deg
end

# Lexicographic rank among vectors of exactly total_degree (1‑based)
function _lex_rank_fixed_degree(exp::AbstractVector{Int}, n::Int, total_degree::Int)
    rank = 1
    remaining_deg = total_degree
    for i in 1:n-1
        a = exp[i]
        # Count vectors that come before those with first component a
        # with first component > a (i.e., a0 from a+1 to remaining_deg)
        for a0 in (a+1):remaining_deg
            rank += _binomial(remaining_deg - a0 + n - i - 1, n - i - 1)
        end
        remaining_deg -= a
    end
    # Last component is fixed by remaining_deg
    @assert remaining_deg == exp[n]
    return rank
end

"""
    build_exponent_index_map(set::MultiindexSet) -> Dict{NTuple{N,Int}, Int} where N

Build a dictionary mapping each exponent (as a tuple) to its column index in the set.
Useful for O(1) lookups without repeated binary searches.

The keys are `NTuple{N,Int}` where `N` is the number of variables.
"""
function build_exponent_index_map(set::MultiindexSet)
    N = size(set.exponents, 1)
    exps = set.exponents
    ncols = size(exps, 2)
    d = Dict{NTuple{N,Int}, Int}()
    sizehint!(d, ncols)
    for j in 1:ncols
        d[Tuple(exps[:, j])] = j
    end
    return d
end

"""
    _multinomial(e::Int, k::Vector{Int}) -> Int

Multinomial coefficient: e! / (k₁! k₂! … kₚ!)  where sum(k) = e.
Uses iterative multiplication of binomial coefficients to avoid overflow.
"""
function _multinomial(e::Int, k::Vector{Int})::Int
    res = 1
    rem = e
    for ki in k
        res *= binomial(rem, ki)
        rem -= ki
    end
    return res
end

end # module