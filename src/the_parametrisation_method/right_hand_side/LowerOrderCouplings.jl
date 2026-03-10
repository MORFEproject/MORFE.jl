include(joinpath(@__DIR__, "../Polynomials.jl"))

using .Polynomials: indices_in_box_and_after
using .Polynomials: coeffs, multiindex_set, nvars

function compute_low_order_couplings(n, predecessor, box_upper, parametrisation::DensePolynomial{TW}, reduced_dynamics::DensePolynomial{TF}, B) where {TW, TF}
    # Check consistency
    S = multiindex_set(parametrisation)
    @assert nvars(parametrisation) == n "parametrisation has wrong number of variables"
    @assert nvars(f) == n "f has wrong number of variables"
    @assert length(predecessor) == n && length(box_upper) == n "predecessor and box_upper must have length n"

    # Determine the result type of one term (for type‑stable summation)
    S_f = eltype(TF)                     # scalar type of the reduced_dynamics's components
    T1 = Base.promote_op(*, typeof(B), TW)
    T2 = Base.promote_op(*, T1, Int)
    T_result = Base.promote_op(*, T2, S_f)
    summation = zero(T_result)

    # Build a dictionary for fast lookup of exponents in the parametrisation's multiindex set
    exps = S.exponents
    n_exps = size(exps, 2)
    exp2idx = Dict{Tuple{Vararg{Int}}, Int}()
    sizehint!(exp2idx, n_exps)
    for j in 1:n_exps
        exp2idx[Tuple(exps[:, j])] = j
    end

    # All multiindex satisfying multiindex > predecessor (order) and multiindex ≤ box_upper (box) – exclude box_upper itself
    multiindex_idxs_all = indices_in_box_and_after(S, box_upper, predecessor)
    multiindex_idxs = filter(!=(idx_box_upper), multiindex_idxs_all)

    # Main summation
    for multiindex_idx in multiindex_idxs
        multiindex_vec = view(exps, :, multiindex_idx)          # exponent vector of multiindex
        reduced_dynamics_coeff_multiindex = coeffs(reduced_dynamics)[multiindex_idx]          # coefficient of f at multiindex (tuple/vector)

        for i in 1:n
            reduced_dynamics_i = reduced_dynamics_coeff_multiindex[i]
            iszero(f_i) && continue

            box_upper_minus_multiindex_i = box_upper[i] - multiindex_vec[i]
            factor = box_upper_minus_multiindex_i + 1

            # Exponent for parametrisation: box_upper - multiindex + e_i
            exp_parametrisation_tuple = ntuple(j -> box_upper[j] - multiindex_vec[j] + (j == i ? 1 : 0), n)
            idx_parametrisation = get(exp2idx, exp_parametrisation_tuple, nothing)
            idx_parametrisation === nothing && continue

            coeff_parametrisation = coeffs(parametrisation)[idx_parametrisation]
            iszero(coeff_parametrisation) && continue

            term = B * coeff_parametrisation * factor * reduced_dynamics_i
            summation += term
        end
    end

    return summation
end