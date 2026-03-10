# Multiindices.jl

A Julia module for multi‑indices (exponent vectors) with **graded lexicographic (Grlex)** order.  
Provides a sorted container, efficient generation, queries, factorizations, and combinatorial counting.

---

## `MultiindexSet` – Sorted container

```julia
struct MultiindexSet
    exponents::Matrix{Int}   # each column is a multi‑index
end
```

**Constructors** (automatically sort by Grlex):
```julia
MultiindexSet(exponents::Matrix{Int})                # from columns
MultiindexSet(exponents::Vector{Vector{Int}})        # from vector of vectors
```

**Access**:
- `length(S)`, `S[i]` (returns view), `collect(S)`, iteration `for v in S`.

---

## Generation functions

| Function | Description |
|----------|-------------|
| `zero_multiindex(n)` | zero vector of length `n` |
| `multiindex(components...)` | e.g. `multiindex(1,0,2)` |
| `all_multiindices_up_to(n, max_deg)` | all vectors with total degree ≤ `max_deg` (Grlex) |
| `multiindices_with_total_degree(n, deg)` | exact degree `deg`, lexicographic |
| `all_multiindices_in_box(bound)` | `0 ≤ v[i] ≤ bound[i]`, Grlex sorted |

**Examples**:
```julia
all_multiindices_up_to(2, 2)   # 6 vectors: [0,0], [1,0], [0,1], [2,0], [1,1], [0,2]
all_multiindices_in_box([2,1]) # same as above but stops at [2,1]
```

---

## Predicates & comparisons

| Function | Returns `true` if |
|----------|-------------------|
| `grlex_precede(a, b)` | `a` comes before `b` in Grlex |
| `compare(a, b)` | `-1`, `0`, `1` (Grlex) |
| `divides(a, b)` | `a[i] ≤ b[i]` ∀i |
| `is_constant(exp)` | `exp == [0,...]` |

**Example**:
```julia
grlex_precede([2,0], [1,1])   # true (same degree, larger first)
divides([1,1], [2,2])         # true
```

---

## Queries on a `MultiindexSet`

| Function | Description |
|----------|-------------|
| `find_in_set(S, exp)` | column index of `exp` (binary search) or `nothing` |
| `indices_in_box_and_after(S, box, other)` | indices `i` with `S[i] < box` and `S[i]` strictly after `other` in Grlex |

**Example**:
```julia
S = all_multiindices_up_to(2, 3)
find_in_set(S, [1,1])                 # 5
indices_in_box_and_after(S, [2,2], [1,1])   # [6,8,9]  ([0,2], [2,1], [1,2])
```

---

## Factorizations

```julia
factorizations(S, exp, N) -> Vector{Vector{Vector{Int}}}
```

Ordered `N`-tuples of vectors from `S` summing to `exp`.  
Outer list sorted lexicographically (tuple order). Returns empty if none.

**Example**:
```julia
S = all_multiindices_up_to(2, 2)
factorizations(S, [2,1], 2)
# 2 solutions: [[1,0], [1,1]] and [[2,0], [0,1]]
```

---

## Combinatorial utilities

| Function | Description |
|----------|-------------|
| `num_multiindices_up_to(n, max_deg)` | number of vectors with total degree ≤ `max_deg` (`binomial(max_deg+n, n)`) |
| `monomial_rank(exp, n, max_deg)` | 1‑based rank of `exp` in the complete Grlex‑sorted set (counting formula) |

**Example**:
```julia
num_multiindices_up_to(3, 4)          # 35
monomial_rank([2,0], 2, 3)            # 4
monomial_rank([1,1], 2, 3)            # 5
```

---

## Notes

- All sets are stored and guaranteed sorted in **Grlex** order.
- `find_in_set` uses binary search – efficient for large sets.
- `indices_in_box_and_after` uses binary search + range scan; the box condition is componentwise `≤`, and total degree is implicitly `< sum(box)`.
- `factorizations` uses backtracking with pruning; number of results can be huge.
- `monomial_rank` is purely combinatorial, O(nvars), no set construction.
- Internal helpers (prefixed `_`) are documented in the source.

---

## Complete example

```julia
using Multiindices

# All monomials in x,y up to degree 4
mons = all_multiindices_up_to(2, 4)

# Find index of x³y
idx = find_in_set(mons, [3,1])

# Check divisibility
divides([2,0], [3,1]) # true

# Factor x³y into two monomials from the set
facts = factorizations(mons, [3,1], 2)
foreach(println, facts)                # prints 4 ordered factorisations
```