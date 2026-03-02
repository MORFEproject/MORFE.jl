# Multiindices.jl Documentation

The `Multiindices` module provides a comprehensive framework for working with multi-indices (exponent vectors) in Julia. It supports three common monomial orders, generation of multi-index sets, efficient binary search operations, combinatorial ranking, and factorisation of exponent vectors into sums of elements from a given set.

## Installation

The module is self-contained; simply include it in your project and import the desired functions:

```julia
using .Multiindices
```

## Monomial Orders

Monomial orders determine how exponent vectors are compared. The module defines an abstract type `MonomialOrder` with three concrete subtypes:

- `Lex` – Lexicographic order (higher exponents in earlier components come first).
- `Grlex` – Graded lexicographic order (total degree first, then lexicographic).
- `Grevlex` – Graded reverse lexicographic order (total degree first, then reverse lexicographic from the last component).

### Core comparison functions

```julia
lex_precede(a, b)          # true if a ≺ b in Lex order
grlex_precede(a, b)        # true if a ≺ b in Grlex order
grevlex_precede(a, b)      # true if a ≺ b in Grevlex order
precede(order, a, b)       # generic version using a monomial order instance
compare(a, b, order=Grlex()) # returns -1, 0, or 1
```

Example:
```julia
a = [2,1,0]
b = [1,3,0]
lex_precede(a, b)      # true, because a[1] > b[1]
grlex_precede(a, b)    # false, total degrees are 3 vs 4, so b comes first
```

## The `MultiindexSet` Type

A `MultiindexSet{O}` is a sorted collection of exponent vectors stored as columns of a matrix. The type parameter `O` is one of the monomial orders, and the set is guaranteed to be sorted according to that order.

```julia
struct MultiindexSet{O<:MonomialOrder}
    exponents::Matrix{Int}   # each column is an exponent vector
end
```

Constructors:
- `MultiindexSet(exponents::Matrix{Int}, order)`
- `MultiindexSet(exponents::Vector{Vector{Int}}, order)`

The set is automatically sorted upon creation. Use `length(S)` to get the number of multi-indices, and iterate or index to obtain individual vectors (e.g., `S[3]` returns the third column as a vector).

## Generating Multi-index Sets

### All multi-indices with total degree ≤ max_degree

```julia
all_multiindices_up_to(nvars, max_degree, order=Grlex())
```

Generates all vectors of length `nvars` whose components sum to at most `max_degree`, sorted by `order`.

Example:
```julia
S = all_multiindices_up_to(2, 2, Lex())
# Columns: [2,0] [1,1] [1,0] [0,2] [0,1] [0,0]   (Lex order)
```

### All multi-indices with exact total degree

```julia
multiindices_with_total_degree(nvars, degree, order=Grlex())
```

Returns all vectors of length `nvars` whose components sum exactly to `degree`, sorted by the tie‑breaking part of the order.

### Box (hyperrectangle) generation

```julia
all_multiindices_in_box(bound, order=Grlex())
```

`bound` is a vector `[b₁, b₂, …, bₙ]`. Generates all vectors `v` with `0 ≤ v[i] ≤ bᵢ`. The result is sorted according to `order`.

A second method accepts a matrix of exponents; it computes the componentwise maximum and uses that as the box bound.

## Operations on `MultiindexSet`

### Searching

```julia
find_in_set(set, exp)   # returns column index or nothing (binary search)
```

Uses binary search to locate an exponent vector in the sorted set.

### Preceding indices

```julia
preceding_indices(set, exp)
```

Returns a `UnitRange` of all column indices whose vectors are strictly less than `exp` according to the set’s order. If `exp` is not in the set, the range includes all vectors less than `exp`.

### Indices in a box and after a given vector

```julia
indices_in_box_and_after(set, box_upper, other)
```

Returns the indices of all vectors `v` in `set` that satisfy:
- `v[i] ≤ box_upper[i]` for all `i` (componentwise box condition)
- `v` comes after `other` in the sorted order (strictly greater).

This function first binary searches to find the starting point of vectors greater than `other`, then scans linearly to collect those that also lie inside the box.

## Predicates

```julia
divides(a, b)      # true iff a[i] ≤ b[i] for all i
is_constant(exp)   # true iff exp is the zero vector
```

## Factorisations

```julia
factorizations(set, exp, N)
```

Returns all ordered `N`-tuples of vectors from `set` whose sum equals `exp`. The result is a vector of factorisations, each factorisation being a vector of `N` exponent vectors. The outer list is sorted lexicographically using the same monomial order stored in `set`.

The search uses backtracking with pruning; it respects the order of factors (i.e., `(a,b)` is different from `(b,a)` if the vectors differ). Returns an empty vector if no factorisation exists.

Example:
```julia
S = all_multiindices_up_to(2, 2, Grlex())
facts = factorizations(S, [2,1], 2)
# e.g., ([1,0] + [1,1]) and ([2,0] + [0,1]) etc.
```

## Combinatorial Ranking for Complete Bases

For the full set of all multi-indices of length `nvars` with total degree ≤ `max_degree`, combinatorial formulas allow direct computation of the 1‑based rank without generating the set. These are efficient (`O(nvars)` time).

```julia
num_multiindices_up_to(nvars, max_degree)
```

Returns the total number of such vectors (binomial coefficient). For `nvars = 0` the result is 0.

```julia
monomial_rank(exp, nvars, max_degree, order)
```

Computes the 1‑based rank of `exp` in the complete set sorted by `order`. Implemented for `Lex` and `Grlex`. For `Grevlex` the function throws an error because a direct combinatorial formula is not provided; use `find_in_set` on a generated set instead.

The vector `exp` must satisfy `length(exp) == nvars` and `sum(exp) ≤ max_degree`. If the computed rank exceeds `typemax(Int)` an overflow error is thrown.

Example:
```julia
rank = monomial_rank([2,1], 2, 3, Grlex())   # returns an integer
```

## Utility Functions

- `zero_multiindex(n)` – returns a zero vector of length `n`.
- `multiindex(components...)` – convenience constructor for an exponent vector.

## Internal Details

The module uses internal recursive generation functions (e.g., `_generate_ascending_lex!`) that preallocate a matrix and fill it in the required order. Sorting of a `MultiindexSet` is performed by a column‑wise permutation using `sortperm` with a custom less‑function based on the monomial order. Binary search operations rely on the sortedness invariant.

## Example Session

```julia
using .Multiindices

# Create a set of all (a,b) with total degree ≤ 3, sorted by Grlex
S = all_multiindices_up_to(2, 3, Grlex())

# Get the 5th vector
v = S[5]          # e.g., [1,1]

# Find its index
idx = find_in_set(S, v)   # returns 5

# Get all vectors before [1,1]
preceding = S[preceding_indices(S, [1,1])]

# Check if [1,0] divides [2,1]
divides([1,0], [2,1])   # true

# Factorisations of [2,1] into two vectors from S
facts = factorizations(S, [2,1], 2)

# Combinatorial rank of [2,1] in the full set up to degree 3
rank = monomial_rank([2,1], 2, 3, Grlex())   # should equal the index we found earlier
```

## Performance Notes

- Generation functions use preallocation and avoid copies; they are suitable for moderately sized sets (up to several hundred thousand multi-indices).
- Binary search and `preceding_indices` are `O(log n)`.
- `factorizations` may be expensive for large `N` because it explores all ordered tuples; prune early to keep it feasible.
- The combinatorial rank functions are very fast and do not generate the set.

---

*This documentation was automatically generated from the module source.*