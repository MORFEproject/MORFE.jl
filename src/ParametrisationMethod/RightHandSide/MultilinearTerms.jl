# Canonical form for a tuple under symmetry labels:
# For each group of positions with the same label, sort the indices.
function canonical(tup, labels)
    idx = collect(tup)
    pos_groups = Dict{Int, Vector{Int}}()
    for (i, lbl) in enumerate(labels)
        push!(get!(pos_groups, lbl, Int[]), i)
    end
    for pos in values(pos_groups)
        sort!(view(idx, pos))
    end
    return Tuple(idx)
end

# Accumulate a fully symmetric term (all labels equal)
function accumulate_symmetric!(dest, term, γ, W, multiindex_set, candidates)
    m = length(term.symmetry)
    temp = similar(W[1])
    for (idx_tuple, count) in factorisations_unordered(multiindex_set, γ, m, candidates)
        fill!(temp, 0)
        term.func(temp, (view(W, i) for i in idx_tuple)...)   # pass views
        dest .+= count * temp
    end
end

# Accumulate a fully asymmetric term (all labels distinct)
function accumulate_asymmetric!(dest, term, γ, W, multiindex_set, candidates)
    m = length(term.symmetry)
    # Assume factorisations_ordered returns an iterator (lazy)
    for idx_tuple in factorisations_ordered(multiindex_set, γ, m, candidates)
        term.func(dest, (view(W, i) for i in idx_tuple)...)   # accumulates directly
    end
end

# Accumulate a partially symmetric term using a linear scan over sorted tuples.
function accumulate_partial!(dest, term, γ, W, multiindex_set, candidates)
    m = length(term.symmetry)
    labels = term.symmetry
    temp = similar(W[1])
    ordered = factorisations_ordered(multiindex_set, γ, m, candidates)
    # Since ordered is sorted, we can group consecutive tuples after canonicalisation.
    isempty(ordered) && return
    iter = iterate(ordered)
    iter === nothing && return
    prev_tup, state = iter
    prev_canon = canonical(prev_tup, labels)
    count = 1
    while true
        next = iterate(ordered, state)
        if next === nothing
            # Process last group
            fill!(temp, 0)
            term.func(temp, (view(W, i) for i in prev_canon)...)
            dest .+= count * temp
            break
        end
        curr_tup, state = next
        curr_canon = canonical(curr_tup, labels)
        if curr_canon == prev_canon
            count += 1
        else
            # Emit group
            fill!(temp, 0)
            term.func(temp, (view(W, i) for i in prev_canon)...)
            dest .+= count * temp
            prev_canon = curr_canon
            count = 1
        end
    end
end

# Main dispatcher for one term
function accumulate_term!(dest, term, γ, W, multiindex_set, candidates)
    m = length(term.symmetry)
    (m < 2 || m > sum(γ)) && return dest

    unique_labels = unique(term.symmetry)
    if length(unique_labels) == 1
        accumulate_symmetric!(dest, term, γ, W, multiindex_set, candidates)
    elseif length(unique_labels) == m
        accumulate_asymmetric!(dest, term, γ, W, multiindex_set, candidates)
    else
        accumulate_partial!(dest, term, γ, W, multiindex_set, candidates)
    end
    return dest
end

# Top‑level function: sum over all terms
function compute_total_sum(model::NDOrderModel, γ, W, multiindex_set, candidate_indices)
    result = zeros(size(W[1]))
    for term in model.terms
        accumulate_term!(result, term, γ, W, multiindex_set, candidate_indices)
    end
    return result
end