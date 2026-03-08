# Polynomials Module Documentation

The `Polynomials` module provides a flexible and efficient representation for multivariate polynomials. It is built around a dense storage scheme where coefficients are stored in a fixed order defined by a **multiindex set** – a collection of exponent vectors sorted according to a chosen monomial order.

## Key Concepts

- **MultiindexSet** – a container for exponent vectors, typically sorted by a monomial order (e.g., `Grlex`, `Lex`, `Grevlex`). It serves as the fixed basis for a dense polynomial.
- **DensePolynomial** – a mutable structure holding a coefficient vector aligned with the columns of a multiindex set. Zero coefficients are stored explicitly, making term lookup by exponent O(log n) if the set is sorted.
- **MonomialOrder** – an abstract type with concrete subtypes `Grlex`, `Lex`, `Grevlex`. The order determines how exponent vectors are sorted in the multiindex set.

## Main Types

### `AbstractPolynomial{T}`
An abstract supertype for all polynomial implementations. Currently the only concrete subtype is `DensePolynomial`.

### `DensePolynomial{T,O<:MonomialOrder} <: AbstractPolynomial{T}`
- `coeffs::Vector{T}` – coefficients, one per monomial in the multiindex set.
- `multiindex_set::MultiindexSet{O}` – the reference set of exponent vectors (columns = monomials).

## Constructors

### From a dictionary
```julia
DensePolynomial(dict::Dict{Vector{Int}, T}, ::Type{O})
```
Builds a polynomial from a dictionary mapping exponent vectors to coefficients. The multiindex set is created automatically using the order `O`.

```julia
p = DensePolynomial(Dict([1,0] => 2.0, [0,1] => -3.0), Grlex)
```

### From a dictionary and an existing multiindex set
```julia
DensePolynomial(dict::Dict{Vector{Int}, T}, multiindex_set::MultiindexSet{O})
```
Checks that all dictionary keys belong to the set and fills the coefficient vector accordingly.

### From pairs (vector of `Pair{Vector{Int},T}`)
```julia
polynomial_from_pairs(::Type{DensePolynomial{T,O}}, pairs)
```
A convenience function that converts a vector of pairs into a dictionary and then constructs the polynomial.

```julia
pairs = [Vector{Int}([2,0]) => 1.5, [0,1] => -0.5]
p = polynomial_from_pairs(DensePolynomial{Float64,Grlex}, pairs)
```

### Zero polynomial
```julia
zero(DensePolynomial{T}, set::MultiindexSet{O})
zero(DensePolynomial{T,O}, set::MultiindexSet{O})
```
Returns a polynomial with all coefficients zero for the given multiindex set.

```julia
mset = MultiindexSet([[0,0],[1,0],[0,1]], Grlex)
z = zero(DensePolynomial{Float64}, mset)   # zero polynomial with 3 terms
```

## Basic Accessors

- `coeffs(p)` – the coefficient vector.
- `multiindex_set(p)` – the underlying multiindex set.
- `nvars(p)` – number of variables (length of each exponent vector).
- `length(p)` – number of monomials in the basis (size of the coefficient vector).
- `eltype(p)` – element type of the coefficients.

## Term Lookup and Coefficient Access

- `find_in_multiindex_set(p, exp)` – returns the column index of `exp` in the multiindex set, or `nothing`.
- `has_term(p, exp)` – checks whether a term with exponent `exp` exists (even if coefficient is zero? Actually it checks presence in the set, not whether coefficient is non‑zero – the current implementation uses `find_in_multiindex_set`, so it returns `true` if the exponent is in the basis, regardless of coefficient value. This may be intended.)
- `coefficient(p, exp)` – returns the coefficient for `exp` (zero if the exponent is not in the basis).
- `find_term(p, exp)` – same as `find_in_multiindex_set` (returns index or nothing).

```julia
p = DensePolynomial(Dict([2,0]=>3, [0,1]=>5), Grlex)
coefficient(p, [2,0])   # 3
has_term(p, [1,1])      # false (exponent not in basis)
find_term(p, [0,1])     # returns the column index (e.g., 3)
```

## Evaluation

### Scalar evaluation
```julia
evaluate(poly::AbstractPolynomial, vals::Vector{<:Number})
```
Evaluates the polynomial at the given variable values. The result type is deduced from the coefficients.

```julia
p = DensePolynomial(Dict([2,0]=>2, [0,1]=>-3), Grlex)
evaluate(p, [1.5, 2.0])   # 2*(1.5^2) + (-3)*(2.0) = 2*2.25 -6 = 4.5-6 = -1.5
```

### Component evaluation (for tuple coefficients)
If coefficients are tuples (e.g., for vector‑valued polynomials), `evaluate(poly, vals, idx)` evaluates only the `idx`‑th component.

```julia
dict = Dict([1,0]=>(1.0,2.0), [0,1]=>(3.0,4.0))
p = DensePolynomial(dict, Grlex)
evaluate(p, [2.0, 3.0], 1)   # 1.0*2.0 + 3.0*3.0 = 2+9 = 11
evaluate(p, [2.0, 3.0], 2)   # 2.0*2.0 + 4.0*3.0 = 4+12 = 16
```

### Extracting a component polynomial
```julia
extract_component(poly::DensePolynomial{NTuple{L,T},O}, idx::Int)
```
Returns a new polynomial (with scalar coefficients) containing the `idx`‑th component of each tuple coefficient.

```julia
p = DensePolynomial(Dict([1,0]=>(1.0,2.0), [0,1]=>(3.0,4.0)), Grlex)
p1 = extract_component(p, 1)   # polynomial with coefficients 1.0 and 3.0
```

## Iterating Over Non‑Zero Terms

The function `each_term(poly)` returns a generator that yields `(exponent_vector, coefficient)` for every term with a non‑zero coefficient. Zero coefficients are skipped automatically.

```julia
p = DensePolynomial(Dict([2,0]=>0, [0,1]=>5), Grlex)   # first term has zero coefficient
for (exp, coeff) in each_term(p)
    println("exp = $exp, coeff = $coeff")
end
# Output: exp = [0,1], coeff = 5
```

## Constructing a Similar Polynomial

```julia
similar_poly(dict::Dict{Vector{Int}, C}, poly::DensePolynomial, nvars::Int)
```
Creates a new polynomial of the same concrete type and monomial order as `poly`, using the coefficients from `dict`. The new polynomial will have `nvars` variables. If `dict` is empty, an empty multiindex set with the correct number of rows is created.

```julia
p1 = DensePolynomial(Dict([1,0]=>2.0, [0,1]=>-1.0), Grlex)
dict2 = Dict([2,0]=>0.5, [1,1]=>1.2)
p2 = similar_poly(dict2, p1, 2)   # p2 is a DensePolynomial{Float64,Grlex} with exponents [2,0] and [1,1]
```

## Dependencies

The module relies on two internal submodules:
- `Multiindices` – provides `MultiindexSet` and the monomial order types.
- `ArrayAlgebra` – (likely provides additional array utilities, but not used directly in the shown code).

## Complete Example

```julia
using .Polynomials

# Create a polynomial in two variables with Grlex order
coeff_dict = Dict(
    [2,0] => 3.0,
    [1,1] => -2.5,
    [0,2] => 1.0
)
p = DensePolynomial(coeff_dict, Grlex)

# Access information
println("Variables: ", nvars(p))
println("Monomial basis: ")
for exp in eachcol(multiindex_set(p).exponents)
    println("  ", exp)
end

# Evaluate at (x,y) = (2,3)
val = evaluate(p, [2.0, 3.0])
println("p(2,3) = ", val)

# Get coefficient of x*y
c = coefficient(p, [1,1])
println("Coefficient of x*y: ", c)

# Iterate over non‑zero terms
for (exp, coeff) in each_term(p)
    println("Term: x^$(exp[1]) y^$(exp[2])  coeff=$coeff")
end
```

## Notes

- The dense representation is most efficient when the polynomial uses a large fraction of the monomials in the chosen basis. For very sparse polynomials, a sparse representation might be more appropriate.
- The module exports `MultiindexSet`, `Grlex`, `Lex`, `Grevlex`, and the main polynomial types and functions.
- All constructors and functions are designed to be type‑stable and generic.

```julia
# You can also create a zero polynomial from an existing multiindex set
mset = MultiindexSet([[0,0],[1,0],[0,1]], Grevlex)
z = zero(DensePolynomial{Float64}, mset)
```

For more details, refer to the source code and the documentation of the `Multiindices` submodule.
```