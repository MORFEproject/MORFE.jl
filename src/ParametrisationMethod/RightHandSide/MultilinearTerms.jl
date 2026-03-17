function compute_sum(model::FullOrderModel, γ, k, W, multiindex_set, candidate_indices)
    result = zeros(size(W[1]))
    if model.nonlinearity_is_symmetric  # hypothetical flag
        """
        Symmetric case
        If the multilinear map is symmetric in its arguments (i.e. the value does not depend on the order of the inputs), 
        we can exploit this to avoid redundant evaluations. 
        Use factorisations_unordered, which returns a list of unique unordered tuples together with the number of ordered permutations that each unordered tuple yields
        """
        # Get unordered factorisations with permutation counts
        unordered_data = factorisations_unordered(multiindex_set, γ, k, candidate_indices)

        # Loop over each unordered tuple
        for (idx_tuple, perm_count) in unordered_data
            # Extract vectors (order inside idx_tuple is not important because N is symmetric)
            args = [W[idx] for idx in idx_tuple]
            # Evaluate N once
            temp = zeros(size(W[1]))
            evaluate_polynomial_nonlinearity!(temp, args...)
            # Add perm_count copies to the result
            result .+= perm_count * temp
        end
    else
        """
        General (non‑symmetric) case
        When the multilinear map is not symmetric, we must sum over all ordered tuples. 
        Use factorisations_ordered to obtain each tuple of indices
        """
        # Get all ordered factorisations
        ordered_tuples = factorisations_ordered(multiindex_set, γ, k, candidate_indices)

        # Allocate a result vector (same size as one W vector)
        result = zeros(size(W[1]))

        # Loop over each ordered tuple
        for idx_tuple in ordered_tuples
            # Extract the corresponding vectors
            args = [W[idx] for idx in idx_tuple]
            # Evaluate N and accumulate
            evaluate_polynomial_nonlinearity!(result, args...)
        end
    end
    return result
end