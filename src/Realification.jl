module Realification

using LinearAlgebra
using StaticArrays: SVector, MVector, StaticArray

using ..Polynomials: DensePolynomial, nvars, each_term, similar_poly, coefficient
using ..Polynomials: coefficients, multiindex_set

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
	result_dict = Dict{SVector{N, Int}, C}()
	for (exp_sv, coeff) in each_term(poly)
		new_exp = zeros(Int, N)
		for idx in 1:N
			new_exp[old2new[idx]] = exp_sv[idx]
		end
		key = SVector{N, Int}(new_exp)
		result_dict[key] = get(result_dict, key, zero(C)) + coeff
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
	β = exp_vec[(n+1):2n]
	γ = exp_vec[(2n+1):end]

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

	result_dict = Dict{SVector{N, Int}, C}()
	for (mult, x, y, w) in states
		new_exp = vcat(x, y, w)
		key = SVector{N, Int}(new_exp)
		result_dict[key] = get(result_dict, key, zero(C)) + mult
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
	canonical_poly, n, m = _reorder_canonical(poly, conj_map)

	N = nvars(canonical_poly)          # = 2n + m
	C = eltype(canonical_poly)

	result_dict = Dict{SVector{N, Int}, C}()
	for (exp_vec, coeff) in each_term(canonical_poly)
		term_dict = _realify_term(exp_vec, coeff, n)
		for (exp, val) in term_dict
			result_dict[exp] = get(result_dict, exp, zero(C)) + val
		end
	end

	return similar_poly(result_dict)
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
the same coefficient type as the input `poly`.
"""
function compose_linear(poly::DensePolynomial, M::Matrix{TA}) where {TA}
	n = nvars(poly)
	p = size(M, 2)
	@assert size(M, 1) == n "First dimension of M must match number of variables"

	# --- 1. Determine coefficient type and whether it's vector-valued ---
	C = eltype(poly)
	is_vector_coeff = C <: StaticArray
	T_coeff = is_vector_coeff ? eltype(C) : C

	# --- 2. Compute maximum exponent for each original variable ---
	max_exp = zeros(Int, n)
	for (a, _) in each_term(poly)
		for i in 1:n
			max_exp[i] = max(max_exp[i], a[i])
		end
	end

	# --- 3. Precompute powers of M[i,j] for each i,j ---
	# pow_M[i][j][d] = M[i,j]^d   for d = 1..max_exp[i]
	pow_M = [[Vector{TA}(undef, max_exp[i]) for _ in 1:p] for i in 1:n]
	for i in 1:n
		for j in 1:p
			if max_exp[i] >= 1
				pow = pow_M[i][j]
				pow[1] = M[i, j]
				for d in 2:max_exp[i]
					pow[d] = pow[d-1] * M[i, j]
				end
			end
		end
	end

	# --- 4. Precompute expansions (∑ M[i,j] y_j)^e for each i and e ---
	# expansions[i][e] :: Dict{Vector{Int}, TA}  where keys are compositions of e into p parts
	expansions = Vector{Vector{Dict{Vector{Int}, TA}}}(undef, n)
	for i in 1:n
		exp_i = Vector{Dict{Vector{Int}, TA}}(undef, max_exp[i] + 1)
		for e in 0:max_exp[i]
			exp_dict = Dict{Vector{Int}, TA}()
			for k in _compositions(e, p)
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
			exp_i[e+1] = exp_dict
		end
		expansions[i] = exp_i
	end

	# --- 5. Initialise current dictionary: keys are [a; zeros(p)] ---
	current_dict = Dict{Vector{Int}, C}()
	for (a, coeff) in each_term(poly)
		key = vcat(collect(a), zeros(Int, p))
		current_dict[key] = get(current_dict, key, zero(C)) + coeff
	end

	# --- 6. Compose variable by variable ---
	for i in 1:n
		next_dict = Dict{Vector{Int}, C}()
		expansions_i = expansions[i]
		for (key, coeff) in current_dict
			e = key[1]                     # exponent of the current variable
			rest = key[2:end]              # remaining variables + y
			if e == 0
				# nothing to do for this variable, just pass through
				next_dict[rest] = get(next_dict, rest, zero(C)) + coeff
			else
				exp_dict = expansions_i[e+1]
				for (k, factor) in exp_dict
					scaled_coeff = coeff .* factor
					# Update y‑exponents: last p entries of rest
					y_part = rest[(end-p+1):end]
					new_y = y_part .+ k
					new_rest = vcat(rest[1:(end-p)], new_y)
					next_dict[new_rest] = get(next_dict, new_rest, zero(C)) + scaled_coeff
				end
			end
		end
		current_dict = next_dict
	end

	# --- 7. Convert to final dictionary with SVector{p,Int} keys ---
	final_dict = Dict{SVector{p, Int}, C}()
	for (key, coeff) in current_dict
		# key is now a vector of length p (only y exponents)
		@assert length(key) == p
		sv_key = SVector{p, Int}(key)
		final_dict[sv_key] = get(final_dict, sv_key, zero(C)) + coeff
	end

	# --- 8. Return polynomial of same coefficient type ---
	if isempty(final_dict)
		# Zero polynomial with p variables
		return similar_poly(Dict{SVector{p, Int}, C}())
	else
		return similar_poly(final_dict)
	end
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
		M[i, n+i] = im
		M[n+i, i] = 1
		M[n+i, n+i] = -im
	end
	for i in 1:m
		M[2n+i, 2n+i] = 1
	end

	return compose_linear(canonical_poly, M)
end

# ------------------------------------------------------------
#  Helper functions for integer compositions
# ------------------------------------------------------------
function _multinomial(e::Int, k::Vector{Int})::Int
	res = 1
	rem = e
	for ki in k
		res *= binomial(rem, ki)
		rem -= ki
	end
	return res
end

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

end # module
