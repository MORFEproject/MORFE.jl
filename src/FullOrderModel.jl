module FullOrderModel

using LinearAlgebra
using SparseArrays

export NDOrderModel, FirstOrderModel,
       MultilinearMap,
       linear_first_order_matrices,
       evaluate_nonlinear_terms!

abstract type AbstractFullOrderModel end

"""
    MultilinearMap{N, F}

Represents a single monomial term of order deg in the nonlinear function of an `NDOrderModel`.

A term is represented using a multiindex stored in the NTupl
    'multiindex' = (i_0, ..., i_{N-1})  
where i_k is the multiplicity of the derivative x^(k). So the i_k specifies how many times the derivative x^(k) appears as an argument. 
During evaluation the multilinear map is called as

    f!(res,
       x^(0), ... repeated i₀ times,
       x^(1), ... repeated i₁ times,
       …
       x^(N-1), ...repeated i_{N-1} times)

# Important Notes
- Each `MultilinearMap` **must implement a multilinear map**, i.e., it should be linear in each of its arguments independently.
- The function `f!` accumulates (adds) into `res` and must be callable with the appropriate number of arguments.
- If one i_k is larger than 1 we assume the input arguments are symmetric by permutation. For example:
    multiindex = (0, 2,...)
    f!(res, x^(1)_1, x^(1)_2, ...) = f!(res, x^(1)_2, x^(1)_1, ...)

"""
struct MultilinearMap{N, F}
    f!::F
    multiindex::NTuple{N, Int}
    deg::Int
end

"""
    MultilinearMap{N}(f, multiindex)

Create a multilinear term for a system of order N.

# Arguments
- `f!`: in-place evaluation function
- `multiindex`: tuple specifying which derivatives are used
"""
function MultilinearMap(f!, multiindex::NTuple{N, Int}) where {N}
    deg = sum(multiindex)

    # Check if input arguments of f matches deg
    ms = methods(f!)
    @assert length(ms) == 1 "Function $(f!) must have exactly one method to determine number of inputs"
    @assert ms[1].nargs == deg + 2 "Function $(f!) must accept $(deg+1) arguments (`res` and $deg inputs) instead of $(ms[1].nargs - 1)"
    @assert deg >= 2 "Function $(f!) must have degree at least 2, but has degree $deg"

    return MultilinearMap{N, typeof(f!)}(f!, multiindex, deg)
end

# Create a multilinear term for a first order system.
function MultilinearMap(f!)
    ms = methods(f!)
    @assert length(ms) == 1 "Function $(f!) must have exactly one method to determine number of inputs"
    deg = ms[1].nargs - 2 # subtract the function itself and `res`
    @assert deg >= 2 "Function $(f!) must have degree at least 2, but has degree $deg"
    multiindex = (deg, )
    return MultilinearMap{1, typeof(f!)}(f!, multiindex, deg)
end

"""
    evaluate_term!(res, term, xs)

Evaluate a single `MultilinearMap` and accumulate (adds) the result into `res`.

# Arguments
- `res`: output vector (modified in-place)
- `term`: multilinear term
- `xs`: tuple `(x, x^(1), …, x^(N-1))` of state derivatives

"""
@inline function evaluate_term!(res, term::MultilinearMap{N}, xs) where {N}
    inds = term.multiindex
    args = ntuple(term.deg) do k
        s = 0
        for j in 1:N
            s += inds[j]
            if k ≤ s
                return @inbounds xs[j]
            end
        end
    end
    term.f!(res, args...)
end

"""

    NDOrderModel{N, NP1, N_NL, MT} <: AbstractFullOrderModel

Representation of an N-th (NP1=N+1) order dynamical system of the form

    B_N x^(N) + ... + B_1 x^(1) + B_0 x = F(x^(N-1)), ..., x^(1), x)

where:
- x^(n) is the n-th derivative of x (x^(n) = d_t^n x)
- x^(0) = x is the state vector
- B_i are the coefficient matrices
- F(x^(N-1)), ..., x^(1), x) represents the nonlinear terms that are assumed to be multilinear
- F is a polynomial function of the derivatives

# Generic type parameters

- `N` defines the order of the ODE.

- `NP1` is the number of linear terms (from 0 through N). It must satisfy NP1 == N+1.

- `N_NL` is the number of nonlinear terms in the tuple nonlinear_terms.

- `MT` is the type of the matrices in the NP1-tuple linear_terms.

# Representation

- Linear terms are stored as a tuple `(B_0, …, B_N)`
- Nonlinear terms are represented as a collection of `MultilinearMap`s

Each `MultilinearMap` defines:
- which derivatives are involved
- how they are combined via a user-provided function

# Notes
- All matrices must have identical size.
- The nonlinear structure is stored in sparse form (only active terms).
- TODO For large `K`, a `Vector` may be more appropriate than an `NTuple`.
"""
struct NDOrderModel{N, NP1, N_NL, MT <: AbstractMatrix} <: AbstractFullOrderModel
    linear_terms::NTuple{NP1, MT}
    nonlinear_terms::NTuple{N_NL, MultilinearMap{N}}

    """
    NDOrderModel(linear_terms, nonlinear_terms)

    Construct an `NDOrderModel` and validate consistency of the linear operators.

    # Checks performed
    - All matrices in `linear_terms` must have identical size
    - Correct relationship between `N` and `NP1`
    """
    function NDOrderModel(linear_terms::NTuple{NP1, MT},
            nonlinear_terms::NTuple{N_NL, MultilinearMap{N}}) where {
                N, NP1, N_NL, MT <: AbstractMatrix}
        @assert NP1 == N + 1
        @assert all(size(B) == size(linear_terms[1]) for B in linear_terms)

        return new{N, NP1, N_NL, MT}(linear_terms, nonlinear_terms)
    end
end

"""
    evaluate_nonlinear_terms!(res, model, order, state_vectors)

Evaluate all nonlinear terms of a given polynomial degree for an `NDOrderModel`.
"""
function evaluate_nonlinear_terms!(res, model::NDOrderModel{N, NP1, N_NL, MT}, 
    order, state_vectors) where {N, NP1, N_NL, MT}

    order <= 1 && return res

    for term in model.nonlinear_terms
        if term.deg == order
            evaluate_term!(res, term, state_vectors)
        end
    end
end

"""
    linear_first_order_matrices(model::NDOrderModel)

Construct the matrices A and B of the equivalent linear first-order system:

    B Ẋ = A X

obtained from the N-th order model

    B_N x^(N) + ... + B_1 x^(1) + B_0 x = F(...)

by introducing the augmented state vector

    X = [x, x^(1), ..., x^(N-1)].

and the (N*n x N*n)-block matrices

    B = [ I   0   0   ⋯   0
          0   I   0   ⋯   0
          ⋮       ⋱
          0   0   0   ⋯  B_N ]

and

    A = [ 0   I   0   ⋯   0
          0   0   I   ⋯   0
          ⋮       ⋱
         -B₀ -B₁ -B₂ ⋯ -B_{N-1} ]

where `I` is the `n × n` identity matrix.
"""
function linear_first_order_matrices(model::NDOrderModel{N, NP1, N_NL, MT}) where {N, NP1, N_NL, MT <: SparseMatrixCSC}
    n = size(model.linear_terms[1], 1)
    T = eltype(model.linear_terms[1])
    total = N * n

    B = spzeros(T, total, total)
    A = spzeros(T, total, total)
    Id = sparse(one(T)*I, n, n)

    # --- B matrix ---
    for i in 1:(N - 1)
        rows = ((i - 1) * n + 1):(i * n)
        B[rows, rows] .= Id
    end

    # last block
    rows = ((N - 1) * n + 1):(N * n)
    B[rows, rows] .= model.linear_terms[end]   # B_N

    # --- A matrix ---

    # shift identities
    for i in 1:(N - 1)
        rows = ((i - 1) * n + 1):(i * n)
        cols = (i * n + 1):((i + 1) * n)
        A[rows, cols] .= Id
    end

    # last row: -B0 ... -B_{N-1}
    lastrow = ((N - 1) * n + 1):(N * n)

    for i in 1:N
        cols = ((i - 1) * n + 1):(i * n)
        A[lastrow, cols] .= -model.linear_terms[i]
    end

    return A, B
end

function linear_first_order_matrices(model::NDOrderModel{N, NP1, N_NL, MT}) where {N, NP1, N_NL, MT <: AbstractMatrix}
    n = size(model.linear_terms[1], 1)
    T = eltype(model.linear_terms[1])
    total = N * n

    B = zeros(T, total, total)
    A = zeros(T, total, total)
    Id = Matrix{T}(I, n, n)

    # --- B matrix ---
    for i in 1:(N - 1)
        rows = ((i - 1) * n + 1):(i * n)
        B[rows, rows] .= Id
    end

    # last block
    rows = ((N - 1) * n + 1):(N * n)
    B[rows, rows] .= model.linear_terms[end]   # B_N

    # --- A matrix ---

    # shift identities
    for i in 1:(N - 1)
        rows = ((i - 1) * n + 1):(i * n)
        cols = (i * n + 1):((i + 1) * n)
        A[rows, cols] .= Id
    end

    # last row: -B0 ... -B_{N-1}
    lastrow = ((N - 1) * n + 1):(N * n)

    for i in 1:N
        cols = ((i - 1) * n + 1):(i * n)
        A[lastrow, cols] .= -model.linear_terms[i]
    end

    return A, B
end


"""
    FirstOrderModel{MT, N_NL} <: AbstractFullOrderModel

Optimised representation of a first‑order dynamical system

    B₁ ẋ + B₀ x = F(x)

where `F(x)` is a polynomial/multilinear function of `x`.

# Fields
- `B0`, `B1`: the linear coefficient matrices.
- `nonlinear_terms

# Construction
    FirstOrderModel((B0, B1), nonlinear_terms)

`nonlinear_terms` can be any iterable of `MultilinearMap{1}`.
"""
struct FirstOrderModel{MT, N_NL} <: AbstractFullOrderModel
    B0::MT
    B1::MT
    nonlinear_terms::NTuple{N_NL, MultilinearMap{1}}

    function FirstOrderModel(linear_terms::NTuple{2, MT},
                             nonlinear_terms::NTuple{N_NL, MultilinearMap{1}}) where {MT, N_NL}
        B0, B1 = linear_terms
        @assert size(B0) == size(B1) "Linear matrices must have identical size"
        new{MT, N_NL}(B0, B1, nonlinear_terms)
    end
end

"""
    evaluate_nonlinear_terms!(res, model::FirstOrderModel, order, state_vectors)

Evaluate all nonlinear terms of given `order` and accumulate into `res`.
`state_vectors` must be a 1‑tuple `(x,)`.
"""
function evaluate_nonlinear_terms!(res, model::FirstOrderModel,
                                   order::Int, state_vector)

    order <= 1 && return res

    @inbounds for term in model.nonlinear_terms
        deg = term.deg
        if deg == order
            term.f!(res, ntuple(_ -> state_vector, deg)...)
        end
    end
    return res
end

"""
    linear_first_order_matrices(model::FirstOrderModel)

Return the matrices `(A, B)` of the equivalent linear first‑order system
`B Ẋ = A X`.  Because the model is already first order, `X = x` and

    A = -B₀,    B = B₁.
"""
function linear_first_order_matrices(model::FirstOrderModel)
    return -model.B0, model.B1
end

end # module