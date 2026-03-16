include(joinpath(@__DIR__, "../src/Multiindices.jl"))
using .Multiindices

nvars, max_degree = 4, 20
multiindex_set = all_multiindices_up_to(nvars, max_degree)

exp = [1, 2, 1, 1]

candidate_indices = indices_in_box_with_bounded_degree(multiindex_set, exp, 1, sum(exp))

k = 3   # number of terms that sum to exp

result = factorisations_ordered(multiindex_set, exp, k, candidate_indices)


println("Found ", length(result), " factorisation(s):")
for (i, idx_tuple) in enumerate(result)
    #println("Factorization $i (indices: $idx_tuple):")
    # Show the actual exponent vectors
    for (j, idx) in enumerate(idx_tuple)
        vec = multiindex_set.exponents[:, idx]
        println("  term $j: $vec")
    end
end

println("\n==========================================================================================\n")

result = factorisations_unordered(multiindex_set, exp, k, candidate_indices)

println("Found ", length(result), " factorisation(s):")
for (i, (idx_tuple, perm_count)) in enumerate(result)
    #println("Factorization $i: indices $idx_tuple (yields $perm_count ordered factorisations)")
    # Show the actual exponent vectors
    for (j, idx) in enumerate(idx_tuple)
        vec = multiindex_set.exponents[:, idx]
        println("  term $j: $vec")
    end
end
