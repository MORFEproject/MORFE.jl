# Polynomials.jl Documentation

This module provides two representations of multivariate polynomials: **sparse** (store only non‑zero coefficients) and **dense** (store all coefficients aligned with a fixed monomial set). Both representations share a common `MultiindexSet` object that defines the monomial basis and its ordering.

The module is part of a larger project and depends on the `Multiindices` submodule, which defines the `MultiindexSet` type and monomial orders.

## Exported Names

- Types: `AbstractPolynomial`, `SparsePolynomial`, `DensePolynomial`
- Constructors: `polynomial_from_dict`, `polynomial_from_pairs`
- Accessors: `coeffs`, `indices`, `multiindex_set`, `nvars`
- Queries: `coefficient`, `has_term`, `find_term`, `find_in_multiindex_set`
- Conversions: `convert_dense_to_sparse`, `convert_sparse_to_dense`

---

## Dependencies

The module automatically includes the `Multiindices` module:

```julia
include("Multiindices.jl")
using .Multiindices
```

All monomial orders (`O<:MonomialOrder`) and the `MultiindexSet` type are re‑exported or available through this dependency.

---

## Abstract Type: `AbstractPolynomial{T}`

All polynomial types inherit from `AbstractPolynomial{T}`, where `T` is the coefficient type (e.g., `Float64`, `Rational{Int}`).  
Common methods:

- `Base.eltype(::AbstractPolynomial{T}) = T`
- All concrete types must provide `multiindex_set(p)`.

---

## Concrete Types

### `SparsePolynomial{T,O}`

**Fields:**

- `coeffs::Vector{T}` – non‑zero coefficients
- `indices::Vector{Int}` – column indices in `multiindex_set.exponents` corresponding to each coefficient
- `multiindex_set::MultiindexSet{O}` – shared monomial set

**Invariants:**

- `length(coeffs) == length(indices)`
- Every index in `indices` is a valid column index of `multiindex_set.exponents`

### `DensePolynomial{T,O}`

**Fields:**

- `coeffs::Vector{T}` – coefficients for **all** monomials in the set, in the same order as the columns of `multiindex_set.exponents`
- `multiindex_set::MultiindexSet{O}` – shared monomial set

**Invariants:**

- `length(coeffs) == size(multiindex_set.exponents, 2)`

---

## Constructors

### 1. Direct construction from existing data

```julia
SparsePolynomial(coeffs::Vector{T}, indices::Vector{Int}, multiindex_set::MultiindexSet{O})
DensePolynomial(coeffs::Vector{T}, multiindex_set::MultiindexSet{O})
```

These constructors are **type‑stable** (they infer the monomial order `O` from the supplied `multiindex_set`). They perform basic sanity checks (length matching, indices in range).

### 2. From a dictionary of exponents → coefficients

```julia
SparsePolynomial(dict::Dict{Vector{Int}, T}, ::Type{O})
DensePolynomial(dict::Dict{Vector{Int}, T}, ::Type{O})
```

A new `MultiindexSet{O}` is built from the exponents present in the dictionary.  
The exponents are sorted according to the order `O`; the polynomial coefficients are aligned accordingly.

```julia
SparsePolynomial(dict::Dict{Vector{Int}, T}, multiindex_set::MultiindexSet{O})
DensePolynomial(dict::Dict{Vector{Int}, T}, multiindex_set::MultiindexSet{O})
```

Reuse an existing `multiindex_set`. All exponents in `dict` must be present in the set (otherwise an error is thrown). Coefficients for monomials not appearing in `dict` are treated as zero.

### 3. From a vector of pairs (convenience)

`polynomial_from_pairs` provides a convenient way to build a polynomial using `Pair`s:

```julia
polynomial_from_pairs(::Type{DensePolynomial{T,O}}, pairs::Vector{Pair{Vector{Int},T}})
polynomial_from_pairs(::Type{SparsePolynomial{T,O}}, pairs::Vector{Pair{Vector{Int},T}})
```

Example:

```julia
p = polynomial_from_pairs(DensePolynomial{Float64, LexOrder}, [ [1,0] => 2.5, [0,1] => -1.0 ])
```

---

## Accessor Functions

| Function               | Returns                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `coeffs(p)`            | Coefficient vector (dense: full length; sparse: non‑zero coefficients)  |
| `indices(p)`           | (Only for `SparsePolynomial`) column indices of non‑zero terms          |
| `multiindex_set(p)`    | The underlying `MultiindexSet` object                                    |
| `nvars(p)`             | Number of variables (size of each exponent vector)                      |
| `length(p)`            | Number of **stored** coefficients (for dense: total monomials; for sparse: non‑zeros) |
| `eltype(p)`            | Coefficient type `T`                                                     |

---

## Term Queries

All functions below work on both sparse and dense polynomials.

### `find_in_multiindex_set(p::AbstractPolynomial, exp::Vector{Int}) -> Union{Int,Nothing}`

Return the column index of the exponent vector `exp` in the polynomial’s `multiindex_set`, or `nothing` if the exponent is not part of the monomial set. (This function does **not** check whether the coefficient is non‑zero.)

### `has_term(p::AbstractPolynomial, exp::Vector{Int}) -> Bool`

Return `true` if the polynomial contains a term with the given exponent and (for sparse) the coefficient is stored as non‑zero; for dense it only checks that the exponent exists in the monomial set (coefficient may be zero).

### `coefficient(p::AbstractPolynomial, exp::Vector{Int}) -> T`

Return the coefficient associated with `exp`. If the exponent is not in the monomial set, or (for sparse) the term is not stored, return `zero(T)`.

### `find_term(p::AbstractPolynomial, exp::Vector{Int}) -> Union{Int,Nothing}`

Return the **storage index** of the term in the polynomial’s `coeffs` array:

- For `DensePolynomial`: the column index in the `multiindex_set` (or `nothing` if the exponent is not in the set).
- For `SparsePolynomial`: the position in `p.indices` (i.e., the index into `p.coeffs`) or `nothing` if the term is not present.

---

## Conversions Between Representations

### `convert_sparse_to_dense(p::SparsePolynomial) -> DensePolynomial`

Create a dense polynomial that shares the same `multiindex_set`. Zero coefficients are filled explicitly.

### `convert_dense_to_sparse(p::DensePolynomial; tol=0) -> SparsePolynomial`

Create a sparse polynomial by dropping coefficients whose absolute value ≤ `tol`. The same `multiindex_set` is reused.

---

## Convenience Function: `polynomial_from_dict`

This function is simply an alias for the dictionary constructors, provided for consistency with the `polynomial_from_pairs` interface.

```julia
polynomial_from_dict(::Type{DensePolynomial}, args...; kwargs...)
polynomial_from_dict(::Type{SparsePolynomial}, args...; kwargs...)
```

It forwards all arguments to the corresponding `DensePolynomial`/`SparsePolynomial` dictionary constructor.

---

## Examples

### Building a sparse polynomial from a dictionary

```julia
using .Polynomials
dict = Dict([1,0] => 2.0, [0,1] => -3.0, [2,0] => 1.5)
p_sparse = SparsePolynomial(dict, LexOrder)

coeffs(p_sparse)       # [2.0, -3.0, 1.5]  (order depends on LexOrder)
indices(p_sparse)      # [1, 2, 3] if those are the positions of [1,0], [0,1], [2,0]
nvars(p_sparse)        # 2
```

### Building a dense polynomial with an existing multiindex set

```julia
# Suppose we already have a multiindex set (e.g., from another polynomial)
ms = multiindex_set(p_sparse)

# Create a new polynomial using the same set, but with different coefficients
dict2 = Dict([1,0] => 5.0, [2,0] => -1.0)
p_dense = DensePolynomial(dict2, ms)   # [0,1] term will be zero automatically
```

### Querying terms

```julia
coefficient(p_dense, [0,1])   # 0.0 (because not in dict2)
has_term(p_sparse, [2,0])     # true
find_term(p_sparse, [0,1])    # 2 (if [0,1] is the second term in the sparse storage)
```

### Conversion

```julia
p_dense_from_sparse = convert_sparse_to_dense(p_sparse)
p_sparse_from_dense = convert_dense_to_sparse(p_dense; tol=1e-10)
```

### Using `polynomial_from_pairs`

```julia
pairs = [ [1,0] => 2.0, [0,1] => -3.0, [2,0] => 1.5 ]
p = polynomial_from_pairs(SparsePolynomial{Float64, GrlexOrder}, pairs)
```