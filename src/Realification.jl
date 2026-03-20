module Realification

using LinearAlgebra

using ..Polynomials

export realify, compose_linear, realify_via_linear

# ------------------------------------------------------------
#  Internal helper functions
# ------------------------------------------------------------

"""
    _exponents_to_dict(poly::DensePolynomial) -> Dict{Vector{Int}, coeff_type}

Convert a polynomial to a dictionary mapping exponent vectors to coefficients.
Zero coefficients are omitted (only terms from `each_term` are included).
"""
function _exponents_to_dict(poly::DensePolynomial)
    d = Dict{Vector{Int}, eltype(coeffs(poly))}()
    for (exp, coeff) in each_term(poly)
        d[exp] = get(d, exp, zero(coeff)) + coeff
    end
    return d
end

"""
    _reorder_canonical(poly::DensePolynomial, conj_map::Vector{Int})
        -> (DensePolynomial, n, m)

Reorder variables according to a conjugation map `conj_map` of length `N`
(where `N` = number of variables).  
- `conj_map[i] = j` means variable `i` is conjugate to variable `j`.
- If variable `i` is real, then `conj_map[i] = i`.

The reordering groups variables as (z₁, …, zₙ, conj(z₁), …, conj(zₙ), w₁, …, wₘ)
where `n` is the number of conjugate pairs and `m` the number of real variables.
Terms with the same exponent after reordering are merged.

Returns the canonical polynomial (same concrete type as `poly`), `n`, and `m`.
"""
function _reorder_canonical(poly::DensePolynomial, conj_map::Vector{Int})
    N = length(conj_map)
    @assert nvars(poly) == N "number of variables must match length of conj_map"

    visited = falses(N)
    pairs = Tuple{Int,Int}[]
    unpaired = Int[]
    for i in 1:N
        visited[i] && continue
        j = conj_map[i]
        if j == i
            push!(unpaired, i)
            visited[i] = true
        else
            if i < j
                push!(pairs, (i, j))
            else
                push!(pairs, (j, i))
            end
            visited[i] = visited[j] = true
        end
    end

    n = length(pairs)
    m = length(unpaired)

    old2new = zeros(Int, N)
    for (k, (i, j)) in enumerate(pairs)
        old2new[i] = k
        old2new[j] = n + k
    end
    for (k, i) in enumerate(unpaired)
        old2new[i] = 2n + k
    end

    coeff_type = eltype(coeffs(poly))
    new_dict = Dict{Vector{Int}, coeff_type}()
    for (old_exp, coeff) in each_term(poly)
        new_exp = zeros(Int, N)
        for oldidx in 1:N
            new_exp[old2new[oldidx]] = old_exp[oldidx]
        end
        new_dict[new_exp] = get(new_dict, new_exp, zero(coeff)) + coeff
    end

    return similar_poly(new_dict, poly, N), n, m
end

"""
    _realify_term(exp_vec::AbstractVector{Int}, coeff::T, n::Int)
        -> Dict{Vector{Int}, T} where T

Transform a single term (exponent vector `exp_vec` and coefficient `coeff`)
of a polynomial in the canonical form (z, z̄, w) into a sum of real monomials.
Returns a dictionary of new exponent vectors (in the real variables) and their
coefficients.
"""
function _realify_term(exp_vec::AbstractVector{Int}, coeff::T, n::Int) where T
    α = exp_vec[1:n]
    β = exp_vec[n+1:2n]
    γ = exp_vec[2n+1:end]

    # states: (multiplier, x, y, w)
    states = [(coeff, zeros(Int, n), zeros(Int, n), copy(γ))]

    for i in 1:n
        a = α[i]
        b = β[i]
        new_states = []
        for (mult, x, y, w) in states
            for mi in 0:a
                for ni in 0:b
                    diff = mi - ni
                    # Use cis? No, just im^diff; careful with negative diff.
                    im_factor = diff >= 0 ? im^diff : (-im)^(-diff)
                    factor = binomial(a, mi) * binomial(b, ni) * im_factor
                    # Use broadcasting for multiplication: works for scalars and vectors
                    new_mult = mult .* factor
                    new_x = copy(x)
                    new_y = copy(y)
                    new_x[i] = a + b - mi - ni
                    new_y[i] = mi + ni
                    push!(new_states, (new_mult, new_x, new_y, w))
                end
            end
        end
        states = new_states
    end

    result_dict = Dict{Vector{Int}, T}()
    for (mult, x, y, w) in states
        new_exp = vcat(x, y, w)
        result_dict[new_exp] = get(result_dict, new_exp, zero(mult)) + mult
    end
    return result_dict
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
    _compositions(e::Int, p::Int) -> Channel{Vector{Int}}

Generate all compositions of the integer `e` into `p` non‑negative parts.
Yields vectors of length `p` whose sum is `e`.  The vectors are newly allocated
for each composition.
"""
function _compositions(e::Int, p::Int)
    out = Vector{Int}(undef, p)
    Channel() do ch
        function gen(pos::Int, remaining::Int)
            if pos == p
                out[pos] = remaining
                put!(ch, copy(out))
            else
                for v in 0:remaining
                    out[pos] = v
                    gen(pos+1, remaining - v)
                end
            end
        end
        gen(1, e)
    end
end

# ------------------------------------------------------------
#  Public API
# ------------------------------------------------------------

"""
    realify(poly::DensePolynomial, conj_map::Vector{Int}) -> DensePolynomial

Transform a complex‑valued polynomial (with variables that may be conjugate
pairs) into a polynomial in real variables.

# Arguments
- `poly`: a polynomial in variables `z₁, …, z_N`.
- `conj_map`: a vector of length `N` where `conj_map[i] = j` means variable `i`
  is the conjugate of variable `j`; if `i` is real, then `conj_map[i] = i`.

# Returns
A new polynomial in real variables `x₁, …, x_n, y₁, …, y_n, w₁, …, w_m`
with `n` conjugate pairs and `m` real variables. The transformation uses the
formulas `z = x + i y`, `z̄ = x - i y`. The returned polynomial has the same
concrete type as the input `poly`.
"""
function realify(poly::DensePolynomial, conj_map::Vector{Int})::DensePolynomial
    canonical_poly, n, m = _reorder_canonical(poly, conj_map)

    coeff_type = eltype(coeffs(poly))
    result_dict = Dict{Vector{Int}, coeff_type}()
    for (exp_vec, coeff) in each_term(canonical_poly)
        term_dict = _realify_term(exp_vec, coeff, n)
        for (exp, val) in term_dict
            result_dict[exp] = get(result_dict, exp, zero(val)) + val
        end
    end

    N_total = 2n + m
    return similar_poly(result_dict, poly, N_total)
end

"""
    compose_linear(poly::DensePolynomial, M::Matrix{TA}) where TA -> DensePolynomial

Compose a multivariate polynomial with a linear map.

# Arguments
- `poly`: polynomial in variables `x₁, …, x_n`. (The coefficient type can be
  numeric or array‑valued.)
- `M`: an `n × p` matrix. Composition means replacing `x_i` by
  `∑_{j=1}^p M[i,j] * y_j`, where `y₁, …, y_p` are new variables.

# Returns
A new polynomial in the variables `y₁, …, y_p`. The returned polynomial has
the same concrete type as the input `poly`.
"""
function compose_linear(poly::DensePolynomial, M::Matrix{TA}) where TA
    n = nvars(poly)
    @assert size(M, 1) == n "First dimension of M must match number of variables"
    p = size(M, 2)

    coeff_vec = coeffs(poly)
    coeff_is_vector = coeff_vec isa Vector{<:AbstractVector}
    if coeff_is_vector
        T = eltype(eltype(coeff_vec))
    else
        T = eltype(coeff_vec)
    end

    # 1. Compute maximum exponent for each original variable
    max_deg_i = zeros(Int, n)
    for (a, _) in each_term(poly)
        for i in 1:n
            max_deg_i[i] = max(max_deg_i[i], a[i])
        end
    end

    # 2. Precompute powers of M[i,j] for each i and j up to max_deg_i[i]
    # pow_M[i][j][d] = M[i,j]^d   for d = 1..max_deg_i[i]  (0 not stored)
    pow_M = [ [ Vector{TA}(undef, max_deg_i[i]) for j in 1:p ] for i in 1:n ]
    for i in 1:n
        for j in 1:p
            if max_deg_i[i] >= 1
                pow = pow_M[i][j]
                pow[1] = M[i, j]
                for d in 2:max_deg_i[i]
                    pow[d] = pow[d-1] * M[i, j]
                end
            end
        end
    end

    # 3. Precompute expansions (∑ M[i,j] y_j)^e for each i and e
    #    expansions[i][e] :: Dict{Vector{Int}, TA}  (e from 0 to max_deg_i[i])
    expansions = Vector{Vector{Dict{Vector{Int}, TA}}}(undef, n)
    for i in 1:n
        expansions_i = Dict{Vector{Int}, TA}[]
        for e in 0:max_deg_i[i]
            exp_dict = Dict{Vector{Int}, TA}()
            for k in _compositions(e, p)   # yields each composition as Vector{Int}
                mult = _multinomial(e, k)
                factor = mult
                for j in 1:p
                    kj = k[j]
                    if kj > 0
                        factor *= pow_M[i][j][kj]
                    end
                end
                exp_dict[k] = factor
            end
            push!(expansions_i, exp_dict)
        end
        expansions[i] = expansions_i
    end

    # 4. Initialise current polynomial: each term gets p trailing zeros
    current_dict = Dict{Vector{Int}, typeof(coeff_vec[1])}()
    for (a, coeff) in each_term(poly)
        key = vcat(a, zeros(Int, p))
        current_dict[key] = get(current_dict, key, zero(coeff)) + coeff
    end

    # 5. Compose variable by variable
    for i in 1:n
        next_dict = Dict{Vector{Int}, typeof(coeff_vec[1])}()
        expansions_i = expansions[i]
        for (key, coeff) in current_dict
            e = key[1]               # exponent of current variable
            rest = key[2:end]         # exponents of remaining variables + y

            if e == 0
                # nothing to do for this variable
                next_dict[rest] = get(next_dict, rest, zero(coeff)) + coeff
            else
                exp_dict = expansions_i[e+1]   # +1 because e is 0‑based index
                for (k, factor) in exp_dict
                    scaled_coeff = coeff .* factor   # broadcast for vector coefficients
                    y_part = rest[end-p+1:end]        # current y‑exponents
                    new_y = y_part .+ k                # add contributions from this variable
                    new_rest = vcat(rest[1:end-p], new_y)
                    next_dict[new_rest] = get(next_dict, new_rest, zero(scaled_coeff)) + scaled_coeff
                end
            end
        end
        current_dict = next_dict
    end

    # 6. Handle empty result (zero polynomial)
    if isempty(current_dict)
        if coeff_is_vector
            if !isempty(coeffs(poly))
                n_comp = length(coeffs(poly)[1])
                zero_coeff = zeros(T, n_comp)
            else
                error("Cannot determine component dimension for zero vector polynomial. " *
                      "Provide a non‑zero polynomial or use an explicit zero constructor.")
            end
        else
            zero_coeff = zero(T)
        end
        zero_dict = Dict(zeros(Int, p) => zero_coeff)
        return similar_poly(zero_dict, poly, p)
    end

    # 7. Convert final dictionary to polynomial of same type
    return similar_poly(current_dict, poly, p)
end

"""
    realify_via_linear(poly::DensePolynomial, conj_map::Vector{Int}) -> DensePolynomial

Transform a complex‑valued polynomial into a polynomial in real variables by
composing with the linear map that expresses complex variables in terms of real
and imaginary parts. This is an alternative implementation to `realify` that
uses the `compose_linear` function. The returned polynomial has the same concrete
type as the input `poly`.

See also: [`realify`](@ref), [`compose_linear`](@ref)
"""
function realify_via_linear(poly::DensePolynomial, conj_map::Vector{Int})::DensePolynomial
    canonical_poly, n, m = _reorder_canonical(poly, conj_map)
    N_orig = 2n + m
    N_new = 2n + m   # same number of real variables

    # Build transformation matrix: [z; z̄; w] = M * [x; y; w]
    M = zeros(Complex{Int}, N_orig, N_new)
    for i in 1:n
        M[i, i] = 1
        M[i, n + i] = im
        M[n + i, i] = 1
        M[n + i, n + i] = -im
    end
    for i in 1:m
        M[2n + i, 2n + i] = 1
    end

    return compose_linear(canonical_poly, M)
end

end # module