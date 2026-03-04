The `Realification` module transforms complex‑valued multivariate polynomials into equivalent real‑valued polynomials. This is useful when complex normal coordinates appear in dynamical systems or reduced‑order models, but the final equations must be expressed in real variables. The module also provides general linear composition of polynomials, which can be used for coordinate changes and as an alternative realification method.

## What does it do?

- **`realify(poly, conj_map)`**  
  Takes a polynomial in complex variables (which may be conjugate pairs or real) and a conjugation map, and returns a polynomial in real variables $x_i$, $y_i$ (for each conjugate pair) and $w_i$ (for each real variable).  
  Example: if $z$ and $\bar z$ are a conjugate pair, they are replaced by $x+iy$ and $x-iy$.

- **`compose_linear(poly, M, p)`**  
  Replaces each original variable $x_i$ by a linear combination $\sum_{j=1}^p M_{ij} y_j$ of new variables $y_1,\dots,y_p$. The matrix $M$ is $n \times p$, where $n$ is the number of variables in `poly`.

- **`realify_via_linear(poly, conj_map)`**  
  An alternative implementation that builds the linear map $z = x+iy$, $\bar z = x-iy$ and uses `compose_linear`. Mathematically equivalent to `realify`.

All functions preserve the concrete polynomial type (`SparsePolynomial` or `DensePolynomial`) of the input and work with both scalar‑ and array‑valued coefficients (e.g., vectors or tuples).

## What is the maths?

Given variables grouped as $(z_1,\dots,z_n,\bar z_1,\dots,\bar z_n,w_1,\dots,w_m)$ after canonical reordering, each monomial

\[
z_1^{\alpha_1}\cdots z_n^{\alpha_n}\;\bar z_1^{\beta_1}\cdots\bar z_n^{\beta_n}\;w_1^{\gamma_1}\cdots w_m^{\gamma_m}
\]

is expanded using the binomial theorem for each pair $(z_k,\bar z_k)$:

\[
z_k^{\alpha_k}\bar z_k^{\beta_k}
= \sum_{m_k=0}^{\alpha_k}\sum_{n_k=0}^{\beta_k}
   \binom{\alpha_k}{m_k}\binom{\beta_k}{n_k}
   i^{\,m_k-n_k}\,
   x_k^{\alpha_k+\beta_k-m_k-n_k}\,
   y_k^{m_k+n_k}.
\]

Multiplying over all $k$ yields a sum of real monomials $x^{\gamma_x}y^{\gamma_y}w^{\gamma_w}$. Coefficients are accumulated from all combinations $(\mathbf m,\mathbf n)$ satisfying
$\gamma_x = \alpha+\beta-\mathbf m-\mathbf n$ and $\gamma_y = \mathbf m+\mathbf n$.

The module also handles unpaired real variables directly. The conjugation map `conj_map` (length = number of original variables) encodes:
- `conj_map[i] = j`  means variable $i$ is the conjugate of variable $j$;
- `conj_map[i] = i`   means variable $i$ is real.

Variables are internally reordered to the canonical order $(z,\bar z,w)$ for efficient processing.

## How does one use the module?

### 1. Setup
Load the module. It re‑exports the needed polynomial types and basic functions from an underlying `Polynomials` module.

```julia
using Realification   # brings in realify, compose_linear, etc.
```

### 2. Create a polynomial
Use either `SparsePolynomial` or `DensePolynomial`. Coefficients can be numbers, tuples, vectors, etc. The polynomial’s variables are implicitly the columns of the exponent matrix.

```julia
# Dense example: 3 variables, max degree 2
miset = all_multiindices_up_to(3, 2)   # uses Grlex order
poly = zero(DensePolynomial{Tuple{2,ComplexF64}}, miset)   # 2‑component coefficients
```

### 3. Realify
Define a conjugation map of length equal to the number of variables. For a system with variables $z_1$, $z_2$, $z_3$ where $z_2$ is the conjugate of $z_1$ and $z_3$ is real:

```julia
conj_map = [2, 1, 3]   # conj(z1)=z2, conj(z2)=z1, conj(z3)=z3
real_poly = realify(poly, conj_map)
```

The new polynomial has variables $(x_1, y_1, w_1)$ (since $n=1$, $m=1$). Evaluation now expects real numbers:

```julia
x = real(z1); y = imag(z1); w = real(z3)
val = evaluate(real_poly, [x, y, w])
```

### 4. Compose with a linear map
If you have a polynomial in variables $x_1,\dots,x_n$ and want to express it in new variables $y_1,\dots,y_p$ via $x_i = \sum_j M_{ij} y_j$:

```julia
M = [1 1; 1 -1]   # 2×2 matrix
p = 2
new_poly = compose_linear(poly, M, p)   # result in y1,y2
```

### 5. Extract a single component
If coefficients are array‑valued (e.g., tuples for vector fields), you can extract one component as a scalar polynomial:

```julia
comp1_poly = extract_component(poly, 1)   # polynomial for the first component
```

### 6. Evaluate
`evaluate` works with the full coefficient array or a specific component:

```julia
full   = evaluate(poly, [z1,z2,z3])          # returns the tuple
comp   = evaluate(poly, [z1,z2,z3], 1)       # returns first component
scalar = evaluate(comp1_poly, [z1,z2,z3])    # returns a number
```

## Important notes

- All functions preserve the polynomial type (`Sparse`/`Dense`) and work with arbitrary coefficient types (numbers, vectors, tuples).
- The module does not simplify powers of `im` beyond normal arithmetic; the final coefficients may still contain `im` but multiplied by real‑valued numbers, which is acceptable because the polynomial is complex‑valued but expressed in real variables.
- The canonical variable ordering after `realify` is always $(x_1,\dots,x_n, y_1,\dots,y_n, w_1,\dots,w_m)$. This order is used in the exponent vectors of the returned polynomial.
- The internal functions `_reorder_canonical`, `_realify_term`, `_multinomial`, `_compositions` are for module use only; the public API is what is exported.

For a complete working example, see the demo file that accompanies the module.