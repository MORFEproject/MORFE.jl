# Realification Module Documentation

## Introduction

The `Realification` module provides tools for transforming complex-valued multivariate polynomials into equivalent real-valued polynomials. This is essential in fields such as dynamical systems and reduced-order modeling, where complex normal coordinates naturally arise but final models must be expressed in real variables. The module also supports composing polynomials with linear maps, enabling coordinate transformations and realification via linear substitution.

The design distinguishes between **sparse** and **dense** polynomial representations, and all public functions preserve the concrete type of the input polynomial. The mathematical foundation follows the binomial expansion of monomials involving conjugate variables, as derived in the referenced scientific document.

The module builds on the `Polynomials` module which defines the polynomial type hierarchy.

## Mathematical Background

Consider a complex polynomial in variables \(\mathbf{z} = (z_1,\dots,z_n)\) and their complex conjugates \(\mathbf{z}^\star\). Because the system is real, the variables satisfy \(\overline{z_k} = z_k^\star\). By introducing real and imaginary parts

\[
z_k = x_k + \mathrm{i}y_k,\qquad z_k^\star = x_k - \mathrm{i}y_k,
\]

any monomial \(\mathbf{z}^{\boldsymbol{\alpha}}(\mathbf{z}^\star)^{\boldsymbol{\beta}}\) expands into a sum of real monomials \(\mathbf{x}^{\boldsymbol{\gamma}}\mathbf{y}^{\boldsymbol{\delta}}\). Using the binomial theorem twice:

\[
z_k^{\alpha_k}(z_k^\star)^{\beta_k}
 = \sum_{m_k=0}^{\alpha_k}\binom{\alpha_k}{m_k}x_k^{\alpha_k-m_k}(\mathrm{i}y_k)^{m_k}
   \sum_{n_k=0}^{\beta_k}\binom{\beta_k}{n_k}x_k^{\beta_k-n_k}(-\mathrm{i}y_k)^{n_k}.
\]

Multiplying over all \(k\) yields the general formula (see Eq.~(4) in the reference):

\[
\mathbf{z}^{\boldsymbol{\alpha}}(\mathbf{z}^\star)^{\boldsymbol{\beta}}
 = \sum_{\substack{\mathbf{0}\le\mathbf{m}\le\boldsymbol{\alpha}\\
                    \mathbf{0}\le\mathbf{n}\le\boldsymbol{\beta}}}
   \binom{\boldsymbol{\alpha}}{\mathbf{m}}\binom{\boldsymbol{\beta}}{\mathbf{n}}
   \mathrm{i}^{|\mathbf{m}|-|\mathbf{n}|}
   \mathbf{x}^{\boldsymbol{\alpha}+\boldsymbol{\beta}-\mathbf{m}-\mathbf{n}}
   \mathbf{y}^{\mathbf{m}+\mathbf{n}}.
\]

Realification collects coefficients of like monomials \(\mathbf{x}^{\boldsymbol{\gamma}}\mathbf{y}^{\boldsymbol{\delta}}\) by summing over all \(\boldsymbol{\alpha},\boldsymbol{\beta},\mathbf{m}\) satisfying
\(\boldsymbol{\gamma} = \boldsymbol{\alpha}+\boldsymbol{\beta}-\mathbf{m}-\mathbf{n}\) and \(\boldsymbol{\delta} = \mathbf{m}+\mathbf{n}\), with \(\mathbf{n} = \boldsymbol{\delta}-\mathbf{m}\) and appropriate bounds. This leads to the coefficient transformation rule.

The module also handles **unpaired real variables** (variables without a conjugate counterpart). A conjugation map `conj_map` of length \(N\) encodes the pairing:
- `conj_map[i] = j` means variable \(i\) is the conjugate of variable \(j\);
- if variable \(i\) is real, `conj_map[i] = i`.

Internally, variables are reordered to a canonical order:
\[
(z_1,\dots,z_n,\ \bar{z}_1,\dots,\bar{z}_n,\ w_1,\dots,w_m)
\]
with \(n\) conjugate pairs and \(m\) real variables. This simplifies the realification algorithm.

## Module Overview

The module exports the following public symbols:

- **Core functions**: `realify`, `compose_linear`, `realify_via_linear`

All functions work with any polynomial type implementing the `AbstractPolynomial` interface (see the `Polynomials` module).

## Core Functions

### `realify(poly::AbstractPolynomial, conj_map::Vector{Int}) -> AbstractPolynomial`

Transforms a complex‑valued polynomial (in variables that may be conjugate pairs) into a polynomial in real variables.

**Arguments**  
- `poly`: input polynomial in \(N\) variables \(z_1,\dots,z_N\).  
- `conj_map`: vector of length \(N\) describing conjugacy: `conj_map[i] = j` means variable \(i\) is the conjugate of variable \(j\); for a real variable, `conj_map[i] = i`.

**Returns**  
A new polynomial in real variables  
\[
x_1,\dots,x_n,\ y_1,\dots,y_n,\ w_1,\dots,w_m
\]
where \(n\) is the number of conjugate pairs and \(m\) the number of real variables. The transformation follows the mathematical derivation above. The returned polynomial has the same concrete type (sparse or dense) as the input.

**Algorithm outline**  
1. Reorder variables to canonical form using `_reorder_canonical`.  
2. For each term in the canonical polynomial, expand using `_realify_term`.  
3. Accumulate results in a dictionary and construct the output polynomial via `polynomial_from_dict`.

### `compose_linear(poly::AbstractPolynomial, M::Matrix{TA}, p::Int) where TA -> AbstractPolynomial`

Composes a polynomial with a linear map: replaces each original variable \(x_i\) by \(\sum_{j=1}^p M[i,j]\,y_j\), where \(y_1,\dots,y_p\) are new variables.

**Arguments**  
- `poly`: polynomial in \(n\) variables.  
- `M`: \(n \times p\) matrix (element type `TA`).  
- `p`: number of new variables (must equal `size(M,2)`).

**Returns**  
A new polynomial in the variables \(y_1,\dots,y_p\). The coefficient type is appropriately promoted. The returned polynomial has the same concrete type as the input.

**Algorithm**  
- Build a dictionary of terms, each keyed by an exponent vector of length \(n+p\) (first \(n\) entries for original variables, last \(p\) for new variables).  
- Iterate over each original variable index \(i\) (from 1 to \(n\)):
  - For every term with exponent \(e\) in that variable, generate all compositions of \(e\) into \(p\) non‑negative parts (using `_compositions`).  
  - For each composition \(\mathbf{k} = (k_1,\dots,k_p)\), compute the multinomial coefficient \(\frac{e!}{k_1!\cdots k_p!}\) and the product \(\prod_{j=1}^p M[i,j]^{k_j}\).  
  - Multiply the term's coefficient by this factor and update the exponent vector (remove the \(i\)-th entry and add \(\mathbf{k}\) to the new‑variable part).  
- After processing all variables, construct the final polynomial from the resulting dictionary.

### `realify_via_linear(poly::AbstractPolynomial, conj_map::Vector{Int}) -> AbstractPolynomial`

Alternative realification that uses `compose_linear` with a specific linear map. This is mathematically equivalent to `realify` but implemented via composition.

**Arguments**  
Same as `realify`.

**Returns**  
A real‑valued polynomial in the same variables as returned by `realify`, with the same concrete type.

**Algorithm**  
- Reorder variables to canonical form (same as `realify`).  
- Build a linear transformation matrix \(M\) of size \((2n+m) \times (2n+m)\) that expresses the complex variables in terms of real and imaginary parts:
  \[
  \begin{aligned}
  z_i &= x_i + \mathrm{i} y_i,\\
  \bar{z}_i &= x_i - \mathrm{i} y_i,\\
  w_i &= w_i.
  \end{aligned}
  \]
- Apply `compose_linear(canonical_poly, M, N_new)`.

This function demonstrates how `compose_linear` can be used to implement realification.

## Internal Helper Functions (Brief)

- **`_exponents_to_dict(poly)`**: converts any polynomial to a dictionary mapping exponent vectors to coefficients.
- **`_reorder_canonical(poly, conj_map)`**: reorders variables according to the conjugation map, grouping conjugate pairs together. Returns the canonical polynomial, the number of pairs \(n\), and the number of real variables \(m\).
- **`_realify_term(exp_vec, coeff, n, m)`**: expands a single term of a canonical polynomial into a dictionary of real monomials.
- **`_multinomial(e, k)`**: computes \(\frac{e!}{k_1! \cdots k_p!}\) where `sum(k) == e`.
- **`_compositions(e, p)`**: generates all compositions of the integer `e` into `p` non‑negative parts (used in `compose_linear`).

## Usage Examples

### Example 1: Realify a simple polynomial

```julia
using Realification

# Polynomial: f(z1, z2) = (2+3im)*z1*z2
# Suppose variables are conjugate pairs: z1 and z2 are conjugates? 
# For a single pair (n=1), we have variables [z, z̄] in the input.
# conj_map = [2,1] meaning var1 ↔ var2.

coeffs = [2+3im]
exponents = [1 0; 0 1]   # two variables, one term: (z1)^1 * (z2)^1
poly_sparse = SparsePolynomial(coeffs, exponents)

conj_map = [2, 1]   # var1 conjugate to var2, var2 conjugate to var1
poly_real = realify(poly_sparse, conj_map)

# Result: polynomial in (x, y) (since n=1, m=0)
# Expected expansion: (2+3i)*(x+iy)*(x-iy) = (2+3i)*(x^2+y^2)
# => coefficient 2+3i for x^2, same for y^2 (they are separate monomials)
# The module will produce two terms: (2+3i)*x^2 and (2+3i)*y^2 (since x^2 and y^2 are different monomials).
```

### Example 2: Compose with a linear map

```julia
# Polynomial f(x1,x2) = x1^2 + 2*x1*x2
coeffs = [1, 2]
exponents = [2 1; 0 1]  # [2,0]' for x1^2, [1,1]' for x1*x2
poly = SparsePolynomial(coeffs, exponents)

# Linear map: x1 = y1 + y2, x2 = y1 - y2   (M = [1 1; 1 -1])
M = [1 1; 1 -1]
p = 2   # new variables y1, y2

poly_y = compose_linear(poly, M, p)

# The result is a polynomial in y1, y2: (y1+y2)^2 + 2*(y1+y2)*(y1-y-y2) = ...
```

### Example 3: Realify via linear map

```julia
poly_real2 = realify_via_linear(poly_sparse, conj_map)
# Should give the same result as realify(poly_sparse, conj_map)
```

## Notes

- Coefficients can be **array‑valued** (e.g., vectors or matrices). All functions handle such coefficients correctly by performing elementwise operations where necessary (e.g., scaling a vector coefficient by a scalar factor).
- The module does **not** simplify numerical values (e.g., `im^2` is left as `-1` only after multiplication with coefficients). The result may contain explicit `im` factors, but they are real numbers multiplied by the coefficient (e.g., `(2+3im)*im` becomes `-3+2im`). This is acceptable because the final polynomial is complex‑valued but expressed in real variables.
- The ordering of monomials in the output follows the lexicographic order used internally by `polynomial_from_dict`.
