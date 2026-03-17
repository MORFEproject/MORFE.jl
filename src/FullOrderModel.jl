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
    PolynomialTerm{N, Deg, F}

Represents a single monomial term of order Deg in the nonlinear function of an `NDOrderModel`.

A term corresponds to a function of the form

    f(x_(i1), … , x_(iDeg))

where each `ik ∈ {0, …, N}` specifies which derivative is used 
and the information of the i is stored in the tupel `indices`.
Example: F(x,x^(1)) -> indices=(0,1)

The tupel `symmetry` stores information, wether f is invariant und permutation of specific arguments.
Arguments that are commutable get the same label. Labels start from 1 and go to maximal Deg
Example: F(x,x,x^(1),x) -> symmetry=(1, 1, 2, 1)

# Type parameters
- `Deg`: maximum derivative order of the system
- `M`: polynomial degree (number of arguments of `f`)

# Example
A quadratic term depending on `x` and `ẋ`:

    PolynomialTerm{1}(f, (0, 1), (1, 2))

# Notes
- The function `f` must have signature:
  
      f(res, x₁, x₂, ..., x_Deg)

  and must accumulate into `res`.
"""
struct PolynomialTerm{N, Deg, F}
    f::F
    indices::NTuple{Deg, Int}
    symmetry::NTuple{Deg, Int}
end

"""
    PolynomialTerm{N}(f, indices)

Create a polynomial term and validate that all derivative indices
are within the allowed range `0:N`.

# Arguments
- `f`: in-place evaluation function
- `indices`: tuple specifying which derivatives are used

# Errors
Throws an error if any index is outside `0:N`.
"""
function PolynomialTerm{N}(f, indices::NTuple{Deg, Int};
        symmetry::Union{NTuple{Deg, Int}, Nothing} = nothing) where {N, Deg}
    @assert all(i -> 0 ≤ i < N, indices) "Index exceeds system order N=$(N)"

    # Check if input arguments of f matches Deg
    ms = methods(f)
    @assert length(ms)==1 "Function f must have exactly one method to determine number of inputs"
    nargs = ms[1].nargs - 1  # subtract function itself
    @assert nargs==Deg + 1 "Function must accept $(Deg+1) arguments ('res' and $Deg inputs) instead of $nargs"

    if isnothing(symmetry)
        # Assume no symmetry
        symmetry = ntuple(i -> i, Deg)
    else
        @assert length(indices)==length(symmetry) "Symmetry tuple must have the same length as indices"
        @assert all(s -> 0 < s ≤ Deg && s isa Int, symmetry) "Symmetry labels must be positive integers"
    end
    return PolynomialTerm{N, Deg, typeof(f)}(f, indices, symmetry)
end

"""
    evaluate_term!(res, term, xs)

Evaluate a single `PolynomialTerm` and accumulate the result into `res`.

# Arguments
- `res`: output vector (modified in-place)
- `term`: polynomial term
- `xs`: tuple `(x, ẋ, …, x⁽ⁿ⁾)` of state derivatives

# Notes
- No allocations are performed.
- Assumes that `xs[i+1]` corresponds to derivative `x⁽i⁾`.
"""
@inline function evaluate_term!(res, term::PolynomialTerm{N, Deg}, xs) where {N, Deg}
    inds = term.indices
    args = ntuple(i -> @inbounds(xs[inds[i] + 1]), Deg)
    term.f(res, args...)
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
- Nonlinear terms are represented as a collection of `PolynomialTerm`s

Each `PolynomialTerm` defines:
- which derivatives are involved
- how they are combined via a user-provided function

# Notes
- All matrices must have identical size.
- The nonlinear structure is stored in sparse form (only active terms).
- TODO For large `K`, a `Vector` may be more appropriate than an `NTuple`.
"""

struct NDOrderModel{T, N, NP1, N_NL, MT <: AbstractMatrix{T}} <: AbstractFullOrderModel
    linear_terms::NTuple{NP1, MT}
    nonlinear_terms::NTuple{N_NL, PolynomialTerm{N}}

    """
    NDOrderModel(linear_terms, nonlinear_terms)

    Construct an `NDOrderModel` and validate consistency of the linear operators.

    # Checks performed
    - All matrices in `linear_terms` must have identical size
    - Correct relationship between `N` and `NP1`
    """
    function NDOrderModel(linear_terms::NTuple{NP1, MT},
            nonlinear_terms::NTuple{N_NL, PolynomialTerm{N}}) where {
            T, N, NP1, N_NL, MT <: AbstractMatrix{T}}
            
        @assert NP1 == N + 1
        @assert all(size(B) == size(linear_terms[1]) for B in linear_terms)

        return new{T, N, NP1, N_NL, MT}(linear_terms, nonlinear_terms)
    end
end

"""
    to_first_order(model::NDOrderModel)

Builds First Order Model.
"""
function to_first_order(model::NDOrderModel)
end
