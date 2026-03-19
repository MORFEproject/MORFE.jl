using SparseArrays

abstract type AbstractFullOrderModel end

"""

FullOrderModel{T} <: AbstractFullOrderModel

Full-Order model representation for a dynamical system of the form:
    B d_t X = AX + N(X)

where:
- X is the state vectors
- B is the inertia matrix
- A is the system matrix
- N(X) represents the nonlinear terms
- d_t X denotes the time derivative of x

The nonlinear terms are expected to be a polynomial function of X. 
The `evaluate_polynomial_nonlinearity!` function must be implemented to handle different
polynomial orders up to the order of `max_order_nonlinearity`:

- Order 2: `evaluate_polynomial_nonlinearity!(res, vec1, vec2)` - quadratic terms
- Order 3: `evaluate_polynomial_nonlinearity!(res, vec1, vec2, vec3)` - cubic terms
- And nothing for the rest
  `evaluate_polynomial_nonlinearity!(res, args...) = nothing``

TODO: Add inertia_singular_rows, forcing_rows or remove it

"""
struct FullOrderModel{T} <: AbstractFullOrderModel
    #B
    inertia_matrix::AbstractMatrix{T}
    #A
    system_matrix::AbstractMatrix{T}

    evaluate_polynomial_nonlinearity!::Function
    max_order_nonlinearity::Int

    inertia_singular_rows::Union{Vector{Int}, Nothing}
    forcing_rows::Union{Vector{Int}, Nothing}
end

"""
    MultilinearMap{N, F}

Represents a single monomial term of order deg in the nonlinear function of an `NDOrderModel`.

A term is represented using a multiindex stored in the NTupl
    'indices' = (i_0, ..., i_{N-1})  
where i_k is the multiplicity of the derivative x^(k). So the i_k specifies how many times the derivative x^(k) appears as an argument. 
During evaluation the multilinear map is called as

    f!(res,
       x^(0), ... repeated i₀ times,
       x^(1), ... repeated i₁ times,
       …
       x^(N-1), ...repeated i_{N-1} times)

# Important Notes
- Each `MultilinearMap` **must implement a multilinear map**, i.e., it should be linear in each of its arguments independently.
- The function `f!` accumulates (adds) into `res`.
- If one i_k is bigger than 2 we assume the input arguments are symmetric by permutation. For example:
    indices = (2,...)
    f!(res, v1, v2, ...) = f!(res, v2, v1, ...)

"""
struct MultilinearMap{N, F}
    f!::F
    indices::NTuple{N, Int}
    deg::Int
end

"""
    MultilinearMap{N}(f, indices)

Create a polynomial term and validate that all derivative indices
are within the allowed range `0:N`.

# Arguments
- `f!`: in-place evaluation function
- `indices`: tuple specifying which derivatives are used
"""
function MultilinearMap{N}(f!, indices::NTuple{N, Int}) where {N}
    deg = sum(indices)

    # Check if input arguments of f matches deg
    ms = methods(f!)
    @assert length(ms)==1 "Function f! must have exactly one method to determine number of inputs"
    nargs = ms[1].nargs - 1  # subtract function itself
    @assert nargs==deg + 1 "Function must accept $(deg+1) arguments ('res' and $deg inputs) instead of $nargs"

    return MultilinearMap{N, typeof(f!)}(f!, indices, deg)
end

"""
    evaluate_term!(res, term, xs)

Evaluate a single `MultilinearMap` and accumulate (adds) the result into `res`.

# Arguments
- `res`: output vector (modified in-place)
- `term`: polynomial term
- `xs`: tuple `(x, x^(1), …, x^(N-1))` of state derivatives

"""
@inline function evaluate_term!(res, term::MultilinearMap{N}, xs) where {N}
    inds = term.indices
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

    NDOrderModel{T, NP1, N_NL , MT} <: AbstractFullOrderModel

Representation of an N-th (NP1=N+1) order dynamical system of the form

    B_N x^(N) + ... + B_1 x^(1) + B_0 x = F(x^(N-1)), ..., x^(1), x)

where:
- x^(n) is the n-th derivative of x (x^(n) = d_t^n x)
- x^(0) = x is the state vector
- B_i are the coefficient matrices
- F(x^(N-1)), ..., x^(1), x) represents the nonlinear terms that are assumed to be multilinear
- F is a polynomial function of the derivatives

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

struct NDOrderModel{T, N, NP1, N_NL, MT <: AbstractMatrix{T}} <: AbstractFullOrderModel
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
            T, N, NP1, N_NL, MT <: AbstractMatrix{T}}
        @assert NP1 == N + 1
        @assert all(size(B) == size(linear_terms[1]) for B in linear_terms)

        return new{T, N, NP1, N_NL, MT}(linear_terms, nonlinear_terms)
    end
end

"""
    evaluate_nonlinear_terms!(res, model, order, state_vectors)

Evaluate all nonlinear terms of a given polynomial degree for an `NDOrderModel`.
"""
function evaluate_nonlinear_terms!(res, model::NDOrderModel{T, N, NP1, N_NL, MT},
        order, state_vectors) where {T, N, NP1, N_NL, MT}
    for term in model.nonlinear_terms
        if term.deg == order
            evaluate_term!(res, term, state_vectors)
        end
    end
end

"""
    to_first_order(model::NDOrderModel)

Transforms an N-th order system (NDOrderModel) 

    B_N x^(N) + ... + B_1 x^(1) + B_0 x = F(x^(N-1)), ..., x^(1), x)

into an equivalent first-order system

    B Ẋ = A X + F̃(X)

by introducing augmented state vector:

     X = [x_0, x_1, ...,  x_N-1] = [x, x^(1), ..., x^(N-1)]

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
function to_first_order(model::NDOrderModel{
        T, N, NP1, N_NL, <:SparseMatrixCSC}) where {T, N, NP1, N_NL}
    n = size(model.linear_terms[1], 1)
    total = N * n

    B = spzeros(T, total, total)
    A = spzeros(T, total, total)
    Id = sparse(1:n, 1:n, ones(T, n), n, n)

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

function to_first_order(model::NDOrderModel{
        T, N, NP1, N_NL, <:AbstractMatrix}) where {T, N, NP1, N_NL}
    n = size(model.linear_terms[1], 1)
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