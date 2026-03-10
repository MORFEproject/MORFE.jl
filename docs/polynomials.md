# Polynomials.jl

A module for multivariate polynomials with a **dense** representation: coefficients stored in a fixed order defined by a `MultiindexSet` (Grlex‑sorted). Built on `Multiindices.jl`.

---

## Main Type

```julia
mutable struct DensePolynomial{T} <: AbstractPolynomial{T}
    coeffs::Vector{T}
    multiindex_set::MultiindexSet
end
```

- `coeffs`: one coefficient per monomial in the set.
- `multiindex_set`: sorted set of exponent vectors (columns = monomials).

---

## Constructors

| Constructor | Description |
|-------------|-------------|
| `DensePolynomial(dict::Dict{Vector{Int},T})` | From dictionary (exponent → coeff). Set built from keys. |
| `DensePolynomial(dict, mset::MultiindexSet)` | Fill existing set; keys must be in `mset`. |
| `polynomial_from_pairs(::Type{DensePolynomial{T}}, pairs)` | From vector of `exp => coeff`. |
| `polynomial_from_dict(::Type{DensePolynomial}, args...)` | Alias for `DensePolynomial(args...)`. |
| `zero(DensePolynomial{T}, mset::MultiindexSet)` | Zero polynomial with given monomial basis. |
| `similar_poly(dict, poly::DensePolynomial, nvars)` | New polynomial of same type as `poly`, using `dict`. Empty dict → empty set. |

**Examples**:
```julia
p = DensePolynomial(Dict([1,0]=>2.0, [0,1]=>-3.0))
pairs = [Vector{Int}([2,0])=>1.5, [0,1]=>-0.5]
p2 = polynomial_from_pairs(DensePolynomial{Float64}, pairs)
mset = MultiindexSet([[0,0],[1,0],[0,1]])
z = zero(DensePolynomial{Float64}, mset)   # three zero coefficients
```

---

## Basic Accessors

- `coeffs(p)`, `multiindex_set(p)`, `nvars(p)`, `length(p)`, `eltype(p)`

---

## Term Lookup & Coefficient Access

| Function | Description |
|----------|-------------|
| `find_in_multiindex_set(p, exp)` | Column index of `exp` in set, or `nothing`. |
| `has_term(p, exp)` | `true` if exponent exists in basis (coeff may be zero). |
| `coefficient(p, exp)` | Coefficient (zero if exponent absent). |
| `find_term(p, exp)` | Same as `find_in_multiindex_set`. |

```julia
coefficient(p, [1,0])   # 2.0
has_term(p, [1,1])      # false
```

---

## Evaluation

### Scalar
```julia
evaluate(poly, vals::Vector{<:Number})
```
Evaluates polynomial at given variable values.

```julia
evaluate(p, [1.5, 2.0])   # 2*1.5^2 + (-3)*2.0 = -1.5
```

### Component (tuple coefficients)
```julia
evaluate(poly, vals, idx)   # evaluate idx‑th component of tuple‑valued coefficients
extract_component(poly, idx) → DensePolynomial{<:Number}
```

```julia
p_vec = DensePolynomial(Dict([1,0]=>(1.0,2.0), [0,1]=>(3.0,4.0)))
evaluate(p_vec, [2.0,3.0], 1)   # 11.0
p1 = extract_component(p_vec, 1) # polynomial with coeffs 1.0 and 3.0
```

---

## Iterating Non‑Zero Terms

`each_term(p)` yields `(exponent_vector, coefficient)` for every term with `!iszero(coeff)`.

```julia
for (exp, coeff) in each_term(p)
    println("$exp → $coeff")
end
```

---

## Complete Example

```julia
using .Polynomials

# Build polynomial
p = DensePolynomial(Dict(
    [2,0] => 3.0,
    [1,1] => -2.5,
    [0,2] => 1.0
))

# Inspect
println("Variables: ", nvars(p)) # 2
println("Monomials: ", collect(eachcol(multiindex_set(p).exponents)))

# Evaluate
val = evaluate(p, [2.0, 3.0]) # 3.0*(2.0^2)  -2.5*(2.0*3.0) + 1.0*(3.0^2) = 6.0

# Coefficient lookup
c = coefficient(p, [1,1]) # -2.5

# Iterate terms
for (exp, coeff) in each_term(p)
    println("x^$(exp[1]) y^$(exp[2]) : $coeff")
end
```

---

## Notes

- Dense representation is efficient when polynomial uses most monomials in the basis.
- All sets are Grlex‑sorted (see `Multiindices.jl`).
- For very sparse polynomials, consider a sparse representation (not provided).