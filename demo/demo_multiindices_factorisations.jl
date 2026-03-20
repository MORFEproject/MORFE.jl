include(joinpath(@__DIR__, "../src/Multiindices.jl"))
using .Multiindices

# -----------------------------------------------------------------------------
# Setup: multiindex set, target exponent, candidate indices
nvars, max_degree = 4, 20
multiindex_set = all_multiindices_up_to(nvars, max_degree)

exp = [2, 2, 0, 0] # exponent vector to factorise
total_deg = sum(exp)
candidate_indices = indices_in_box_with_bounded_degree(multiindex_set, exp, 1, total_deg)

k = 3   # number of terms in the factorisation

println("Target exponent: $exp  (total degree $total_deg)")
println("Number of terms: $k\n")

# -----------------------------------------------------------------------------
# 1. Asymmetric factorisations – every permutation is distinct
result_asym = factorisations_asymmetric(multiindex_set, exp, k, candidate_indices)

println("Found ", length(result_asym), " asymmetric factorisation(s):\n")
for (i, idx_tuple) in enumerate(result_asym)
    println("  Factorization $i (indices: $idx_tuple):")
    for (j, idx) in enumerate(idx_tuple)
        vec = multiindex_set.exponents[idx]
        println("    term $j: $vec")
    end
end

println("\n" * "="^80 * "\n")

# -----------------------------------------------------------------------------
# 2. Fully symmetric factorisations – all permutations equivalent
result_sym = factorisations_fully_symmetric(multiindex_set, exp, k, candidate_indices)

println("Found ", length(result_sym), " fully symmetric factorisation(s):\n")
for (i, (idx_tuple, perm_count)) in enumerate(result_sym)
    println("  Factorization $i: indices $idx_tuple  (yields $perm_count ordered factorisations)")
    for (j, idx) in enumerate(idx_tuple)
        vec = multiindex_set.exponents[idx]
        println("    term $j: $vec")
    end
end

println("\n" * "="^80 * "\n")

# -----------------------------------------------------------------------------
# 3. Groupwise symmetric factorisations – symmetry within each group, groups in fixed order
#    Examples: 
#       (1,1,1) means H(U,V,W) → all all arguments are different (same as assymetric case)
#       (3,) means H(U,U,U) → all three arguments are symmetric (same as fully symmetric)
#       (1,2) means H(U,V,V) → first argument is alone (no symmetry), last are two symmetric

group_sizes = (1,2)
@assert sum(group_sizes) == k

result_group = factorisations_groupwise_symmetric(multiindex_set, exp, group_sizes, candidate_indices)

println("Found ", length(result_group), " groupwise symmetric factorisation(s) with group sizes $group_sizes:\n")
for (i, (flat_indices, total_count)) in enumerate(result_group)
    println("  Factorisation $i: flat indices = $flat_indices  (total ordered = $total_count)")
    
    # Reconstruct groups from flat_indices according to group_sizes
    pos = 1
    for (g, size) in enumerate(group_sizes)
        if size == 0
            continue
        end
        group_indices = flat_indices[pos:pos+size-1]
        println("    Group $g (size $size): indices $group_indices")
        for (j, idx) in enumerate(group_indices)
            vec = multiindex_set.exponents[idx]
            println("      term $(pos + j - 1): $vec")
        end
        pos += size
    end
end

println("\n" * "="^80 * "\n")

println("Demo finished successfully.")