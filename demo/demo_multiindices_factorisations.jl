include(joinpath(@__DIR__, "../src/Multiindices.jl"))
using .Multiindices

# -----------------------------------------------------------------------------
# Setup: multiindex set, target exponent, candidate indices
nvars, max_degree = 4, 20
multiindex_set = all_multiindices_up_to(nvars, max_degree)

exp = [2, 2, 0, 0]   # exponent vector to factorise
total_deg = sum(exp)
candidate_indices = indices_in_box_with_bounded_degree(multiindex_set, exp, 1, total_deg)

k = 3   # number of factor slots

println("Target exponent: $exp  (total degree $total_deg)")
println("Number of factors: $k\n")

# -----------------------------------------------------------------------------
# 1. Asymmetric factorisations — every permutation is a distinct entry (multiplier = 1)
result_asym = factorisations_asymmetric(multiindex_set, exp, k, candidate_indices)

println("Found ", length(result_asym), " asymmetric factorisation(s):\n")
for (i, entry) in enumerate(result_asym)
	println("  Factorisation $i (indices: $(entry.factor_indices)):")
	for (j, idx) in enumerate(entry.factor_indices)
		println("    factor $j: $(multiindex_set.exponents[idx])")
	end
end

println("\n" * "="^80 * "\n")

# -----------------------------------------------------------------------------
# 2. Fully symmetric factorisations — one representative per unordered combination;
#    multiplier counts the distinct ordered arrangements.
result_sym = factorisations_fully_symmetric(multiindex_set, exp, k, candidate_indices)

println("Found ", length(result_sym), " fully symmetric factorisation(s):\n")
for (i, entry) in enumerate(result_sym)
	println("  Factorisation $i: indices $(entry.factor_indices)  ($(entry.multiplier) ordered arrangement(s))")
	for (j, idx) in enumerate(entry.factor_indices)
		println("    factor $j: $(multiindex_set.exponents[idx])")
	end
end

println("\n" * "="^80 * "\n")

# -----------------------------------------------------------------------------
# 3. Groupwise symmetric factorisations — symmetry within each group, groups in fixed order.
#    Examples:
#      (1,1,1) → all arguments distinct            (same as asymmetric)
#      (3,)    → all three arguments symmetric     (same as fully symmetric)
#      (1,2)   → first argument alone, last two symmetric

group_sizes = (1, 2)
@assert sum(group_sizes) == k

result_group = factorisations_groupwise_symmetric(multiindex_set, exp, group_sizes, candidate_indices)

println("Found ", length(result_group), " groupwise symmetric factorisation(s) with group sizes $group_sizes:\n")
for (i, entry) in enumerate(result_group)
	println("  Factorisation $i: indices $(entry.factor_indices)  ($(entry.multiplier) ordered arrangement(s))")
	pos = 1
	for (g, sz) in enumerate(group_sizes)
		sz == 0 && continue
		group_indices = entry.factor_indices[pos:(pos+sz-1)]
		println("    Group $g (size $sz): indices $group_indices")
		for (j, idx) in enumerate(group_indices)
			println("      factor $(pos+j-1): $(multiindex_set.exponents[idx])")
		end
		pos += sz
	end
end

println("\n" * "="^80 * "\n")

# -----------------------------------------------------------------------------
# 4. Bounded index tuples — enumerate all sorted index tuples of length M whose
#    per-index counts are bounded by exp.  Each result contains:
#      - a canonical sorted index tuple
#      - the corresponding multiindex/count vector
#      - the number of distinct permutations
using StaticArrays: SVector
M = 6
exp_sv = SVector{length(exp), Int}(exp)

result_bounded = bounded_index_tuples(M, exp_sv)

println("Found ", length(result_bounded), " bounded index tuple(s) for M = $M and exp = $exp_sv:\n")
for (i, (idx_tuple, multiindex, perm_count)) in enumerate(result_bounded)
	println("  Tuple $i: $idx_tuple")
	println("    multiindex = $multiindex")
	println("    permutation count = $perm_count")
end

println("\n" * "="^80 * "\n")
println("Demo finished successfully.")
