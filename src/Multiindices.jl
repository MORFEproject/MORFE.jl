module Multiindices

using StaticArrays: SVector

export MultiindexSet, zero_multiindex, # nvars,
	all_multiindices_up_to, multiindices_with_total_degree,
	all_multiindices_in_box, indices_in_box_with_bounded_degree,
	divides, is_constant, find_in_set, build_exponent_index_map,
	factorisations_asymmetric, factorisations_fully_symmetric,
	factorisations_groupwise_symmetric,
	bounded_index_tuples

# ==================== Core comparison functions ====================

"""
	grlex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) -> Bool

Graded lexicographic order: compare total degree first, then lexicographic descending.
Returns `true` if `a` comes before `b` in this order.
"""
@inline function grlex_precede(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
	da, db = sum(a), sum(b)
	da != db && return da < db
	for i in eachindex(a)
		if a[i] != b[i]
			return a[i] > b[i]   # descending lexicographic
		end
	end
	return false  # equal
end

"""
	grlex_precede(t::NTuple{N,Int}, v::AbstractVector{Int}) -> Bool

Compare a tuple (representing an exponent) with a vector in graded lexicographic order.
Useful for binary search without allocating a vector.
"""
@inline function grlex_precede(t::NTuple{N, Int}, v::AbstractVector{Int}) where {N}
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
	MultiindexSet{N}

A fixed collection of exponent vectors (multiindices) stored as a vector of `SVector{N, Int}`.
The set is guaranteed to be sorted according to the graded lexicographic (Grlex) order.

# Fields
- `exponents::Vector{SVector{N, Int}}`: each element is an exponent vector.
"""
struct MultiindexSet{N}
	exponents::Vector{SVector{N, Int}}

	# Internal constructor with sortedness assertion (debug mode only)
	function MultiindexSet(exponents::Vector{SVector{N, Int}}, ::Val{true}) where {N}
		if !is_sorted(exponents)
			error("MultiindexSet constructed with unsorted exponents")
		end
		new{N}(exponents)
	end
end

# Public constructor from vector of SVectors (sorts in Grlex)
function MultiindexSet(exponents::Vector{SVector{N, Int}}) where {N}
	exps_sorted = sort(exponents, lt = grlex_precede)
	unique!(exps_sorted)
	MultiindexSet(exps_sorted, Val(true))
end

# Constructor from matrix (converts columns to SVectors and sorts)
function MultiindexSet(exponents::Matrix{Int})
	nvars = size(exponents, 1)
	if nvars == 0
		exps_vec = SVector{0, Int}[]
	else
		exps_vec = [SVector{nvars, Int}(exponents[:, j]) for j in 1:size(exponents, 2)]
	end
	return MultiindexSet(exps_vec)
end

# Constructor from vector of vectors
function MultiindexSet(exponents::Vector{Vector{Int}})
	if isempty(exponents)
		return MultiindexSet(SVector{0, Int}[])
	end
	nvars = length(exponents[1])
	@assert all(length(e) == nvars for e in exponents) "All exponent vectors must have the same length"
	exps_vec = [SVector{nvars, Int}(e) for e in exponents]
	return MultiindexSet(exps_vec)
end

# Empty constructor
MultiindexSet() = MultiindexSet(SVector{0, Int}[])

# Check that the vector is non‑decreasing according to Grlex order.
function is_sorted(exponents::Vector{SVector{N, Int}}) where {N}
	for i in 1:(length(exponents)-1)
		if !grlex_precede(exponents[i], exponents[i+1])
			return false
		end
	end
	return true
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

# ----- Generation in lex order (internal, preallocating matrix) -----

# Generate all vectors of length n with exact total degree in descending lex order,
# filling a preallocated matrix column‑wise.
function _generate_ascending_lex_fixed!(
	exponents::AbstractMatrix{Int}, n::Int, total_degree::Int)
	col = 1
	function recurse(prefix::Vector{Int}, remaining_vars::Int, remaining_deg::Int)
		if remaining_vars == 1
			@inbounds exponents[:, col] = vcat(prefix, remaining_deg)
			col += 1
		else
			for e in remaining_deg:-1:0
				recurse(vcat(prefix, e), remaining_vars - 1, remaining_deg - e)
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
		return MultiindexSet(
			max_degree >= 0 ? [SVector{0, Int}()] : SVector{0, Int}[], Val(true))
	end
	total = binomial(max_degree + nvars, nvars)
	exponents = Matrix{Int}(undef, nvars, total)
	col = 1
	for d in 0:max_degree
		block_size = binomial(d + nvars - 1, nvars - 1)
		block = view(exponents, :, col:(col+block_size-1))
		_generate_ascending_lex_fixed!(block, nvars, d)
		col += block_size
	end
	# Convert columns to SVectors in the same order (already sorted by grlex)
	exps_vec = [SVector{nvars, Int}(exponents[:, i]) for i in 1:total]
	return MultiindexSet(exps_vec, Val(true))
end

"""
	multiindices_with_total_degree(nvars::Int, deg::Int) -> MultiindexSet

Generate all exponent vectors of length `nvars` with total degree exactly `deg`,
sorted in lexicographic order (the tie‑breaker for Grlex).
Returns a `MultiindexSet`.
"""
function multiindices_with_total_degree(nvars::Int, degree::Int)
	if nvars == 0
		return MultiindexSet(
			degree == 0 ? [SVector{0, Int}()] : SVector{0, Int}[], Val(true))
	end
	total = binomial(degree + nvars - 1, nvars - 1)
	exponents = Matrix{Int}(undef, nvars, total)
	_generate_ascending_lex_fixed!(exponents, nvars, degree)
	exps_vec = [SVector{nvars, Int}(exponents[:, i]) for i in 1:total]
	return MultiindexSet(exps_vec, Val(true))
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
		return MultiindexSet([SVector{0, Int}()])
	end
	dims = bound .+ 1
	total = prod(dims)
	exps_vec = Vector{SVector{n, Int}}(undef, total)
	idx = 1
	for i in CartesianIndices(Tuple(dims))
		exps_vec[idx] = SVector{n, Int}(ntuple(j -> i[j] - 1, n))
		idx += 1
	end
	return MultiindexSet(exps_vec)  # sorts in Grlex
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

Base.length(S::MultiindexSet) = length(S.exponents)
Base.getindex(S::MultiindexSet, i::Int) = S.exponents[i]
function Base.iterate(S::MultiindexSet, state = 1)
	state > length(S) ? nothing : (S[state], state + 1)
end
Base.collect(S::MultiindexSet) = [Vector(S.exponents[i]) for i in 1:length(S)]

"""
	find_in_set(set::MultiindexSet, exp::AbstractVector{Int}) -> Union{Int, Nothing}

Return the column index of `exp` in `set.exponents` using binary search,
exploiting the fact that the set is sorted according to Grlex.
Returns `nothing` if `exp` is not present.
"""
function find_in_set(set::MultiindexSet{N}, exp::AbstractVector{Int}) where {N}
	exps = set.exponents
	lo, hi = 1, length(exps)
	while lo <= hi
		mid = (lo + hi) ÷ 2
		v_mid = exps[mid]
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
	_sv_eq_tuple(v::SVector{N,Int}, t::NTuple{N,Int})

Helper functions that compares:  SVector ==(indexwise) NTuple
"""
@inline function _sv_eq_tuple(v::SVector{N, Int}, t::NTuple{N, Int}) where {N}
	for i in 1:N
		if v[i] != t[i]
			return false
		end
	end
	return true
end

"""
	find_in_set(set::MultiindexSet, exp::NTuple{N,Int}) where N -> Union{Int, Nothing}

Tuple version – avoids allocating a vector for the exponent.
"""
function find_in_set(set::MultiindexSet{N}, exp::NTuple{N, Int}) where {N}
	exps = set.exponents
	lo, hi = 1, length(exps)
	while lo <= hi
		mid = (lo + hi) ÷ 2
		v_mid = exps[mid]
		if _sv_eq_tuple(v_mid, exp)
			return mid
		elseif grlex_precede(exp, v_mid)  # exp < v_mid ?
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
function _last_index_below_degree(set::MultiindexSet{N}, max_total_deg::Int) where {N}
	exps = set.exponents
	lo, hi = 1, length(exps)
	last = 0
	while lo <= hi
		mid = (lo + hi) ÷ 2
		if sum(exps[mid]) < max_total_deg
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
function _last_index_below_degree(set::MultiindexSet{N}, max_total_deg::Int,
	allowed_indices::AbstractVector{Int}) where {N}
	isempty(allowed_indices) && return 0
	exps = set.exponents
	lo, hi = 1, length(allowed_indices)
	last = 0
	while lo <= hi
		mid = (lo + hi) ÷ 2
		i = allowed_indices[mid]
		if sum(exps[i]) < max_total_deg
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
function indices_in_box_with_bounded_degree(
	set::MultiindexSet{N}, box_upper::AbstractVector{Int},
	degree_lower_bound::Int, total_deg_upper::Int) where {N}
	@assert length(box_upper) == N
	exps = set.exponents
	n = length(exps)
	n == 0 && return Int[]

	first_idx = _last_index_below_degree(set, degree_lower_bound) + 1
	last_idx = _last_index_below_degree(set, total_deg_upper)

	result = Int[]
	@inbounds for i in first_idx:last_idx
		v = exps[i]
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
function indices_in_box_with_bounded_degree(
	set::MultiindexSet{N}, box_upper::AbstractVector{Int},
	degree_lower_bound::Int, total_deg_upper::Int,
	allowed_indices::AbstractVector{Int}) where {N}
	@assert length(box_upper) == N
	exps = set.exponents
	isempty(allowed_indices) && return Int[]

	last_below_lower = _last_index_below_degree(set, degree_lower_bound, allowed_indices)
	last_below_upper = _last_index_below_degree(set, total_deg_upper, allowed_indices)

	lo_pos = searchsortedfirst(allowed_indices, last_below_lower + 1)
	hi_pos = searchsortedlast(allowed_indices, last_below_upper)

	result = Int[]
	@inbounds for pos in lo_pos:hi_pos
		i = allowed_indices[pos]
		v = exps[i]
		if all(v .<= box_upper)
			push!(result, i)
		end
	end
	return result
end

# ==================== Factorizations ====================

"""
	factorisations_asymmetric(set::MultiindexSet, exp::AbstractVector{Int}, num_factors::Int, candidate_indices::AbstractVector{Int}) -> Vector{NTuple{num_factors,Int}}

Return all ordered `num_factors`-tuples of indices (taken from `candidate_indices`) whose corresponding exponent vectors sum to `exp`.
Each factorisation is an `NTuple{num_factors,Int}` where the `k`-th entry is the index of the `k`-th factor.
If no factorisation exists, an empty vector is returned.
"""
function factorisations_asymmetric(
	set::MultiindexSet{N}, exp::AbstractVector{Int}, num_factors::Int,
	candidate_indices::AbstractVector{Int}) where {N}
	if num_factors == 0
		return iszero(exp) ? NTuple{0, Int}[()] : NTuple{0, Int}[]
	end
	exp_svec = SVector{N, Int}(exp)
	exps = set.exponents
	results = NTuple{Int64(num_factors), Int}[]
	current_idxs = Vector{Int}(undef, num_factors)

	function backtrack(depth::Int, current_sum::SVector{N, Int})
		if depth == 0
			if current_sum == exp_svec
				push!(results, Tuple(current_idxs))
			end
			return
		end
		for i in candidate_indices
			v = exps[i]
			new_sum = current_sum + v
			# Prune: if any component exceeds exp
			if any(new_sum .> exp_svec)
				continue
			end
			current_idxs[num_factors-depth+1] = i
			backtrack(depth - 1, new_sum)
		end
	end

	backtrack(num_factors, zero(SVector{N, Int}))
	return results
end

"""
	factorisations_fully_symmetric(set::MultiindexSet, exp::AbstractVector{Int}, num_factors::Int, candidate_indices::AbstractVector{Int}) -> Vector{Tuple{NTuple{num_factors,Int}, Int}}

Return all `num_factors`-tuples of indices (taken from `candidate_indices`) with non‑decreasing indices whose corresponding exponent vectors sum to `exp`.
For each such combination, the result includes the tuple itself and the number of distinct ordered factorisations (permutations) that can be formed from it.

The number of permutations equals `num_factors! / (m₁! m₂! … mₖ!)` where `mⱼ` are the multiplicities of each distinct index in the tuple.

The input `candidate_indices` is assumed to contain unique indices; duplicate entries are removed internally.

If no factorisation exists, an empty vector is returned.
"""
function factorisations_fully_symmetric(
	set::MultiindexSet{N}, exp::AbstractVector{Int}, num_factors::Int,
	candidate_indices::AbstractVector{Int}) where {N}
	if num_factors == 0
		return iszero(exp) ? [((), 1)] : Tuple{NTuple{0, Int}, Int}[]
	end

	exp_svec = SVector{N, Int}(exp)
	exps = set.exponents

	sorted_candidates = sort(unique(candidate_indices))
	L = length(sorted_candidates)

	results = Vector{Tuple{NTuple{num_factors, Int}, Int}}()
	current_idxs = Vector{Int}(undef, num_factors)   # indices chosen so far

	function backtrack(depth::Int, start_pos::Int, current_sum::SVector{N, Int})
		if depth == 0
			if current_sum == exp_svec
				# Compute number of permutations (_multinomial coefficient)
				counts = Int[]
				i = 1
				while i <= num_factors
					j = i
					while j <= num_factors && current_idxs[j] == current_idxs[i]
						j += 1
					end
					push!(counts, j - i)
					i = j
				end
				perm_count = _multinomial(num_factors, counts)
				push!(results, (Tuple(current_idxs), perm_count))
			end
			return
		end
		for pos in start_pos:L
			idx = sorted_candidates[pos]
			v = exps[idx]
			new_sum = current_sum + v
			if any(new_sum .> exp_svec)
				continue
			end
			current_idxs[num_factors-depth+1] = idx
			backtrack(depth - 1, pos, new_sum)      # allow same index again (non‑decreasing)
		end
	end

	backtrack(num_factors, 1, zero(SVector{N, Int}))
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
function factorisations_groupwise_symmetric(
	set::MultiindexSet{N}, exp::AbstractVector{Int},
	group_sizes::NTuple{M, Int},
	candidate_indices::AbstractVector{Int}) where {N, M}
	exp_svec = SVector{N, Int}(exp)

	global_candidates = sort(unique(candidate_indices))

	results = Vector{Tuple{Vector{Int}, Int}}()

	function recurse_groups(group_idx::Int, remaining::SVector{N, Int},
		current_flat::Vector{Int}, current_count::Int)
		if group_idx > M
			if iszero(remaining)
				push!(results, (copy(current_flat), current_count))
			end
			return
		end

		k = group_sizes[group_idx]
		if k == 0
			recurse_groups(group_idx + 1, remaining, current_flat, current_count)
			return
		end

		# Enumerate all possible sub‑exponents s with 0 ≤ s ≤ remaining (componentwise)
		ranges = [0:remaining[i] for i in 1:N]
		range_tuple = Tuple(ranges)

		for s_idx in CartesianIndices(range_tuple)
			s = SVector{N, Int}(ntuple(i -> s_idx[i], N))
			# Get all unordered factorisations of s with exactly k indices
			unordered = factorisations_fully_symmetric(set, s, k, global_candidates)

			for (tuple, perm_count) in unordered
				new_remaining = remaining - s
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

	recurse_groups(1, exp_svec, Int[], 1)
	return results
end

# ==================== Mathematical ranking for complete bases ====================

"""
	num_multiindices_up_to(nvars::Int, max_degree::Int) -> Integer

Return the number of exponent vectors of length `nvars` with total degree ≤ `max_degree`.
For `nvars = 0` the set contains one element if `max_degree ≥ 0`, otherwise zero.
"""
num_multiindices_up_to(nvars::Int, max_degree::Int) = nvars == 0 ?
													  (max_degree >= 0 ? 1 : 0) :
													  binomial(max_degree + nvars, nvars)

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
		return 1  # only the empty exponent
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
		count_before += binomial(d + n - 1, n - 1)
	end
	lex_rank_within_deg = _lex_rank_fixed_degree(exp, n, total_deg) - 1
	return 1 + count_before + lex_rank_within_deg
end

# Lexicographic rank among vectors of exactly total_degree (1‑based) in descending order.
function _lex_rank_fixed_degree(exp::AbstractVector{Int}, n::Int, total_degree::Int)
	rank = 1
	remaining_deg = total_degree
	for i in 1:(n-1)
		a = exp[i]
		# Count vectors that come before those with first component a
		# (i.e., those with first component > a)
		for a0 in (a+1):remaining_deg
			rank += binomial(remaining_deg - a0 + n - i - 1, n - i - 1)
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
function build_exponent_index_map(set::MultiindexSet{N}) where {N}
	exps = set.exponents
	d = Dict{SVector{N, Int}, Int}()
	sizehint!(d, length(exps))
	for (j, v) in enumerate(exps)
		d[v] = j
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

"""
	bounded_index_tuples(M, exp::SVector{N,Int})

Enumerate all index tuples of length `M` whose component counts are bounded by `exp`.

This function generates all tuples `(i₁, ..., i_M)` with entries in `1:N` such that:

- Each index `k` appears at most `exp[k]` times
- The total length of the tuple is exactly `M`

Each tuple is represented in its canonical sorted form (non-decreasing order),
so that it uniquely corresponds to a count vector (multiindex).

For each valid tuple, the function also returns:
- The associated multiindex `alpha`, where `alpha[k]` is the number of times `k` appears
- The number of distinct permutations of the tuple

Returns a vector of tuples:
	(index_tuple, multiindex, permutation_count)

where:
- `index_tuple::NTuple{M,Int}` is the sorted tuple
- `multiindex::SVector{N,Int}` contains the counts
- `permutation_count::Int` is the number of distinct permutations
"""
function bounded_index_tuples(M::Int, exp::SVector{N, Int}) where {N}
	results = Vector{Tuple{NTuple{Int64(M), Int}, SVector{N, Int}, Int}}()

	counts = zeros(Int, N)

	function backtrack(pos::Int, remaining::Int)
		if pos == N
			if remaining <= exp[pos]
				counts[pos] = remaining

				# build canonical sorted tuple
				idx = Vector{Int}(undef, M)
				k = 1
				@inbounds for i in 1:N
					for _ in 1:counts[i]
						idx[k] = i
						k += 1
					end
				end

				tuple_repr = Tuple(idx)
				multi = SVector{N, Int}(counts)
				perm_count = _multinomial(Int64(M), counts)

				push!(results, (tuple_repr, multi, perm_count))
			end
			return
		end

		max_val = min(exp[pos], remaining)
		for v in 0:max_val
			counts[pos] = v
			backtrack(pos + 1, remaining - v)
		end
	end

	backtrack(1, M)

	return results
end

end # module
