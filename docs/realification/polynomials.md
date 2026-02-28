# Polynomials Module Documentation

## Introduction

The `Polynomials` module provides abstract and concrete types for representing multivariate polynomials. It serves as a foundation for the `Realification` module, which transforms complex-valued polynomials into real-valued ones.

The module distinguishes between **sparse** and **dense** polynomial representations. All polynomial types implement the `AbstractPolynomial` interface, guaranteeing at least the methods `coeffs`, `exponents`, `nvars`, `length`, and `getindex`. This allows algorithms to work polymorphically.

## Module Overview

The module exports the following public symbols:

- **Abstract types**: `AbstractPolynomial`
- **Concrete types**: `SparsePolynomial`, `DensePolynomial`, `MonomialBasis`
- **Construction**: `polynomial_from_dict`
- **Accessors**: `coeffs`, `exponents`, `nvars`

## Types

### `AbstractPolynomial{T}`

Abstract supertype for multivariate polynomials with coefficient type `T`. Any concrete subtype must implement:
- `coeffs(p)`: return the coefficient vector (may be a `Vector` of numbers or of array‑valued coefficients).
- `exponents(p)`: return a matrix where each column is an exponent vector (row = variable index, column = term index).
- `nvars(p)`: number of variables.
- `Base.length(p)`: number of terms.
- `Base.getindex(p, i)`: return `(coeff, exponent_vector)` for the i‑th term.

### `SparsePolynomial{C<:AbstractVector} <: AbstractPolynomial{eltype(C)}`

Sparse representation storing coefficients and exponents explicitly.

**Fields**  
- `coeffs::C`: vector of coefficients (may be numeric or, e.g., `Vector{Vector{Float64}}` for matrix polynomials).  
- `indices::Matrix{Int}`: each column is an exponent vector; `indices[:, i]` corresponds to `coeffs[i]`.

**Constructor**  
`SparsePolynomial(coeffs, indices)`.

### `MonomialBasis`

A fixed set of monomials (exponent vectors) shared among dense polynomials.

**Fields**  
- `exponents::Matrix{Int}`: matrix whose columns are the exponent vectors. Rows = variables, columns = monomials.

**Constructor**  
`MonomialBasis(exponents)`.

### `DensePolynomial{T} <: AbstractPolynomial{T}`

Dense representation using a common monomial basis. The coefficient vector is aligned with the basis order.

**Fields**  
- `coeffs::Vector{T}`: coefficients in the order of the basis.  
- `basis::MonomialBasis`: the shared monomial basis.

**Constructor**  
`DensePolynomial(coeffs, basis)` – asserts that `length(coeffs) == size(basis.exponents, 2)`.

## Polynomial Construction from Dictionaries

The module provides several helpers to build polynomials from `Dict{Vector{Int},T}`.

- **Type‑based constructors** (for tests or external use):
  ```julia
  polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, ::Type{SparsePolynomial})
  polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, ::Type{DensePolynomial})
  ```
- **Instance‑based constructors** (used internally to preserve exact container type):
  ```julia
  polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, poly::SparsePolynomial)
  polynomial_from_dict(dict::Dict{Vector{Int}, T}, nvars::Int, poly::DensePolynomial)
  ```

These functions sort exponent vectors lexicographically and ensure a canonical ordering.

## Usage Examples

### Example 1: Create a simple polynomial

```julia
using Polynomials

# Sparse polynomial: f(z1, z2) = (2+3im)*z1*z2
coeffs = [2+3im]
exponents = [1 0; 0 1]   # two variables, one term: (z1)^1 * (z2)^1
poly_sparse = SparsePolynomial(coeffs, exponents)

# Access polynomial properties
nvars(poly_sparse)    # 2
length(poly_sparse)   # 1
coeffs(poly_sparse)   # [2+3im]
exponents(poly_sparse) # [1 0; 0 1]

# Get individual terms
poly_sparse[1]  # (2+3im, [1, 0])
```

### Example 2: Create a dense polynomial

```julia
# Dense polynomial with basis [x^2, x*y, y^2]
basis = MonomialBasis([2 1 0; 0 1 2])  # x^2, x*y, y^2
poly_dense = DensePolynomial([1.0, 2.0, 1.0], basis)

# This represents: f(x,y) = 1*x^2 + 2*x*y + 1*y^2
```

### Example 3: Construct from dictionary

```julia
# Build a polynomial from a dictionary
dict = Dict([1, 0] => 1.0, [0, 1] => 2.0, [1, 1] => 3.0)
poly = polynomial_from_dict(dict, 2, SparsePolynomial)

# Result: f(x,y) = 1*x + 2*y + 3*x*y
```
