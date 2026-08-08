"""
Module `Realification` — convert complex-valued parametrisations and reduced dynamics
to real-valued form.

The cohomological equations are solved in `ComplexF64` to handle both damped and
undamped systems uniformly.  For systems with real matrices and complex-conjugate
master-mode pairs, the resulting `W` and `R` satisfy conjugate-symmetry relations
that allow an exact transformation to real arithmetic.  This module implements that
transformation so that subsequent time integration and post-processing can operate
entirely in real arithmetic.
"""
module Realification

using LinearAlgebra
using StaticArrays: SVector, MVector

using ..Polynomials: DensePolynomial, nvars, each_term, similar_poly, coefficient
using ..Polynomials: coefficients, multiindex_set, coeff_shape

# `compose_linear` and the coefficient-shape helpers live in `Polynomials`: they are pure
# polynomial algebra with no realification content, and `ExternalSystems` — which loads
# long before this module — needs `compose_linear` for its change of external coordinates.
# Re-exported below so `MORFE.Realification.compose_linear` keeps resolving.
using ..Polynomials: compose_linear, _coeff_type, _zero_like, _materialise

export realify, compose_linear, realify_via_linear

# ------------------------------------------------------------
#  Internal helper functions
# ------------------------------------------------------------

"""
	_reorder_canonical(poly::DensePolynomial{C,N}, conj_map::Vector{Int})
		-> (DensePolynomial{C,N}, n, m)

Reorder variables according to a conjugation map `conj_map` of length `N`
(where `N` = number of variables).  
- `conj_map[i] = j` means variable `i` is conjugate to variable `j`.
- If variable `i` is real, then `conj_map[i] = i`.

The reordering groups variables as (z₁, …, zₙ, conj(z₁), …, conj(zₙ), w₁, …, wₘ)
where `n` is the number of conjugate pairs and `m` the number of real variables.
Terms with the same exponent after reordering are merged.

Returns the canonical polynomial (same concrete type as `poly`), `n`, and `m`.
"""
function _reorder_canonical(poly::DensePolynomial{C, N}, conj_map::Vector{Int}) where {C, N}
    @assert nvars(poly) == length(conj_map) "Number of variables must match length of conj_map"

    visited = falses(N)
    pairs = Tuple{Int, Int}[]
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

    # Build permutation from old indices to new indices
    old2new = zeros(Int, N)
    for (k, (i, j)) in enumerate(pairs)
        old2new[i] = k
        old2new[j] = n + k
    end
    for (k, i) in enumerate(unpaired)
        old2new[i] = 2n + k
    end

    # Accumulate new exponents as SVector{N,Int}
    CoeffType = _coeff_type(poly)
    result_dict = Dict{SVector{N, Int}, CoeffType}()
    for (exp_sv, coeff) in each_term(poly)
        mat_coeff = _materialise(coeff)   # concrete copy (no-op for scalars)
        new_exp = zeros(Int, N)
        for idx in 1:N
            new_exp[old2new[idx]] = exp_sv[idx]
        end
        key = SVector{N, Int}(new_exp)
        result_dict[key] = get(result_dict, key, _zero_like(mat_coeff)) + mat_coeff
    end

    # Build polynomial from dictionary
    canonical = similar_poly(result_dict)
    return canonical, n, m
end

"""
	_realify_term(exp_vec::SVector{N,Int}, coeff::C, n::Int)
		-> Dict{SVector{N,Int}, C} where {C,N}

Transform a single term (exponent vector `exp_vec` and coefficient `coeff`)
of a polynomial in the canonical form (z, z̄, w) into a sum of real monomials.
Returns a dictionary mapping new exponent vectors (in the real variables)
to their coefficients.

Here `N = 2n + m`, with `n` conjugate pairs and `m` real variables.
"""
function _realify_term(exp_vec::SVector{N, Int}, coeff::C, n::Int) where {C, N}
    α = exp_vec[1:n]
    β = exp_vec[(n + 1):2n]
    γ = exp_vec[(2n + 1):end]

    # states: (multiplier, x, y, w)
    states = [(coeff, zeros(Int, n), zeros(Int, n), collect(γ))]

    for i in 1:n
        a = α[i]
        b = β[i]
        new_states = []
        for (mult, x, y, w) in states
            for mi in 0:a
                for ni in 0:b
                    diff = mi - ni
                    im_factor = diff >= 0 ? im^diff : (-im)^(-diff)
                    factor = binomial(a, mi) * binomial(b, ni) * im_factor
                    # Multiply factor (a number) with multiplier (may be scalar or SVector)
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

    zero_coeff = _zero_like(coeff)
    result_dict = Dict{SVector{N, Int}, typeof(zero_coeff)}()
    for (mult, x, y, w) in states
        new_exp = vcat(x, y, w)
        key = SVector{N, Int}(new_exp)
        result_dict[key] = get(result_dict, key, zero_coeff) + mult
    end
    return result_dict
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
concrete type as the input `poly` (including the same number of variables).
"""
function realify(poly::DensePolynomial, conj_map::Vector{Int})::DensePolynomial
    canonical_poly, n, _ = _reorder_canonical(poly, conj_map)

    N = nvars(canonical_poly)          # = 2n + m
    CoeffType = _coeff_type(canonical_poly)

    result_dict = Dict{SVector{N, Int}, CoeffType}()
    for (exp_vec, coeff) in each_term(canonical_poly)
        mat_coeff = _materialise(coeff)
        term_dict = _realify_term(exp_vec, mat_coeff, n)
        for (exp, val) in term_dict
            result_dict[exp] = get(result_dict, exp, _zero_like(val)) + val
        end
    end

    return similar_poly(result_dict)
end

"""
	realify_via_linear(poly::DensePolynomial, conj_map::Vector{Int}) -> DensePolynomial

Transform a complex‑valued polynomial into a polynomial in real variables by
composing with the linear map that expresses complex variables in terms of real
and imaginary parts. This is an alternative implementation to `realify` that
uses the `compose_linear` function. The returned polynomial has the same concrete
type as the input `poly` (real coefficients).

See also: [`realify`](@ref), [`compose_linear`](@ref)
"""
function realify_via_linear(poly::DensePolynomial, conj_map::Vector{Int})::DensePolynomial
    canonical_poly, n, m = _reorder_canonical(poly, conj_map)
    N = nvars(canonical_poly)

    # Build transformation matrix: [z; z̄; w] = M * [x; y; w]
    M = zeros(Complex{Int}, N, N)
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
