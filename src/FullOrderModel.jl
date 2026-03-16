abstract type AbstractFullOrderModel end

"""
Full-Order model representation for a dynamical system of the form:
    B d_t X = AX + N(X)

where:
- X is the state vectors
- B is the inertia matrix
- A is the Jacobian matrix
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
    inertia::AbstractMatrix{T}
    #A
    jacobian::AbstractMatrix{T}

    evaluate_polynomial_nonlinearity!::Function
    max_order_nonlinearity::Int

    inertia_singular_rows::Union{Vector{Int}, Nothing}
    forcing_rows::Union{Vector{Int}, Nothing}
end
