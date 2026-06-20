# Parametric FEM — Mathematical Derivation

This document describes the mathematics behind the polynomial-in-parameter FEM
assembly used in `examples/04_parametric_clamped_beam/`. The goal is to express
the stiffness $K$, mass $M$, and nonlinear elastic forces as polynomial functions
of two geometric parameters $(\theta_1, \theta_2)$, assembled **once** from the
reference mesh rather than re-assembled at every parameter value.

---

## 1. Reference Configuration and Geometric Parameters

### 1.1 Physical Setup

The beam occupies a reference domain $\Omega_0 \subset \mathbb{R}^3$. Two scalar
parameters control the geometry:

- **$\theta_1$** — uniform axial stretch along $x_1$  
- **$\theta_2$** — arch pre-deformation in the shape of the first bending eigenmode $\varphi_1$

### 1.2 Reference Map and Jacobian

The parametric reference map reads

$$x(\theta_1, \theta_2, x_0) = x_0 + \theta_1 J_1 x_0 + \theta_2 \varphi_1(x_0)$$

with Jacobian (gradient with respect to $x_0$)

$$J(\theta_1, \theta_2, x_0) = I + \theta_1 J_1 + \theta_2 J_2(x_0)$$

where

$$J_1 = e_1 \otimes e_1 \quad (\text{constant axial projection}), \qquad
  J_2(x_0) = \nabla_0 \varphi_1(x_0) \quad (\text{gradient of arch mode, varies per QP}).$$

The key distinction: $J_1$ is constant across the mesh, so the univariate family
$J(\theta_1) = I + \theta_1 J_1$ has an affine (QP-independent) structure.
By contrast, $J_2(x_0)$ must be re-evaluated at every quadrature point, which
drives the bivariate assembly in Section 5.

---

## 2. Polynomial Series Algebra

Since $J$ is linear in $(\theta_1, \theta_2)$, any function of $J$ — determinant,
adjugate, inverse — is polynomial in the parameters and can be manipulated via
truncated power-series arithmetic.

### 2.1 Univariate Series ($\theta$-polynomials)

A scalar or tensor-valued series is stored as a vector of coefficients:

$$p(\theta) = \sum_{k=0}^{N} p_k \, \theta^k, \qquad \texttt{p[k+1]} = p_k$$

**Truncated product** (Cauchy convolution, capped at degree $N$):

$$[p \cdot q]_k = \sum_{j=0}^{k} p_j \, q_{k-j}, \quad k = 0,\ldots,N$$

Three named variants dispatch on the bilinear operation:

| Function | Operation | Typical use |
|----------|-----------|-------------|
| `poly_mul` | scalar `*` | scalar series, or scalar × tensor |
| `poly_dot` | single contraction `⋅` | matrix products $J(\theta) \cdot J^{-1}(\theta)$ |
| `poly_contract` | double contraction `⊡` | strain–stress inner products $\varepsilon \mathbin{:} \sigma$ |

**Reciprocal series** ($q = 1/p$):

Matching coefficients of $\theta^k$ in $p \cdot q \equiv 1 \pmod{\theta^{N+1}}$ gives the O($N^2$) recurrence

$$q_0 = \frac{1}{p_0}, \qquad q_k = -\frac{1}{p_0} \sum_{j=1}^{k} p_j \, q_{k-j}, \quad k \geq 1.$$

Implemented in `reciprocal_series` (`theta_polynomials.jl`). This is the only
place a rational function (specifically $1/\det J(\theta)$) is converted to a
polynomial series; no tensor-valued reciprocal series is ever built.

### 2.2 Bivariate Series ($(\theta_1, \theta_2)$-polynomials)

A bivariate series is stored as an $(N+1)\times(N+1)$ matrix:

$$p(\theta_1, \theta_2) = \sum_{k_1, k_2 \geq 0} p_{k_1,k_2} \, \theta_1^{k_1} \theta_2^{k_2}, \qquad
  \texttt{A[k1+1, k2+1]} = p_{k_1,k_2}$$

Only entries with $k_1 + k_2 \leq N$ are meaningful; the rest are zero.

**Bivariate truncated product**:

$$[p \cdot q]_{k_1,k_2} = \sum_{j_1=0}^{k_1}\sum_{j_2=0}^{k_2} p_{j_1,j_2} \, q_{k_1-j_1,\, k_2-j_2}$$

kept only for $k_1+k_2 \leq N$. The same three operation variants exist:
`bpoly_mul`, `bpoly_dot`, `bpoly_contract` (`bivariate_polynomials.jl`).

**Bivariate reciprocal series** ($q = 1/p$):

The condition $p \cdot q \equiv 1$ gives

$$p_{00} \, q_{k_1,k_2} = \delta_{k_1,0}\delta_{k_2,0}
  - \sum_{\substack{(j_1,j_2)\neq(0,0) \\ j_1 \leq k_1,\; j_2 \leq k_2}} p_{j_1,j_2} \, q_{k_1-j_1,\, k_2-j_2}$$

computed in order of increasing total degree $d = k_1 + k_2$ so lower-degree
values are always available (implemented in `breciprocal_series`).

**Integer power**: `bpoly_power(p, n, N)` computes $p^n$ via $n$ repeated
bivariate products.

---

## 3. Determinant and Adjugate as Polynomial Series

### 3.1 The Cofactor Identity

The key algebraic device is the exact identity

$$J^{-1} = \frac{\operatorname{adj}(J)}{\det(J)}$$

where $\operatorname{adj}(J)$ is the classical adjugate (transpose of the
cofactor matrix). This decomposes the rational tensor $J^{-1}(\theta)$ into a
**polynomial** tensor series (adjugate) and a **scalar** reciprocal series
(1/det), side-stepping any tensor-valued reciprocal series.

### 3.2 Shared Intermediates

For any $3\times 3$ matrix $X$ define:

$$\operatorname{term}_X = X^2 - X \cdot \operatorname{tr}(X), \qquad
  \tau_X = \tfrac{1}{2}\operatorname{tr}(\operatorname{term}_X) = \tfrac{1}{2}\!\left[\operatorname{tr}(X^2) - \operatorname{tr}(X)^2\right]$$

These two quantities appear in both the determinant and adjugate formulas and
are computed once per pair of matrices.

### 3.3 Univariate Case: $J(\theta) = J_0 + \theta J_1$

With $A = J_0$, $B = J_1$, set $A^2 = J_0 J_0$, $B^2 = J_1 J_1$,
$AB = J_0 J_1$, $BA = J_1 J_0$, and abbreviate traces as $t_A = \operatorname{tr}(J_0)$, etc.

**Determinant** (exact degree 3, 4 non-zero coefficients):

$$\det J(\theta) = c_0 + c_1\theta + c_2\theta^2 + c_3\theta^3$$

$$\begin{aligned}
c_0 &= \det(J_0) \\
c_1 &= \operatorname{tr}(J_0^2 J_1) - \tau_A\,\operatorname{tr}(J_1) - \operatorname{tr}(J_0)\,\operatorname{tr}(J_0 J_1) \\
c_2 &= \operatorname{tr}(J_1^2 J_0) - \tau_B\,\operatorname{tr}(J_0) - \operatorname{tr}(J_1)\,\operatorname{tr}(J_0 J_1) \\
c_3 &= \det(J_1)
\end{aligned}$$

**Adjugate** (exact degree 2, 3 non-zero coefficients):

$$\operatorname{adj} J(\theta) = C_0 + \theta C_1 + \theta^2 C_2$$

$$\begin{aligned}
C_0 &= \operatorname{term}_A - \tau_A\,I \\
C_1 &= \bigl(\operatorname{tr}(J_0)\operatorname{tr}(J_1) - \operatorname{tr}(J_0 J_1)\bigr) I
       - \bigl(J_0\operatorname{tr}(J_1) + J_1\operatorname{tr}(J_0)\bigr)
       + J_0 J_1 + J_1 J_0 \\
C_2 &= \operatorname{term}_B - \tau_B\,I
\end{aligned}$$

These are computed in a single pass by `det_and_adj_series` in
`parametric_geometry.jl`, sharing all matrix products.

### 3.4 Bivariate Case: $J(\theta_1,\theta_2) = J_0 + \theta_1 J_1 + \theta_2 J_2(x_0)$

With $A = J_0$, $B = J_1$, $C = J_2$, the bivariate series have at most
10 non-zero determinant entries and 6 non-zero adjugate entries.

**Bivariate determinant** (computed in `det_and_adj_bseries`, `bivariate_geometry.jl`):

| Monomial | Coefficient |
|----------|------------|
| $1$ | $\det(A)$ |
| $\theta_1$ | $\operatorname{tr}(A^2 B) - \tau_A\operatorname{tr}(B) - \operatorname{tr}(A)\operatorname{tr}(AB)$ |
| $\theta_2$ | $\operatorname{tr}(A^2 C) - \tau_A\operatorname{tr}(C) - \operatorname{tr}(A)\operatorname{tr}(AC)$ |
| $\theta_1^2$ | $\operatorname{tr}(B^2 A) - \tau_B\operatorname{tr}(A) - \operatorname{tr}(B)\operatorname{tr}(AB)$ |
| $\theta_2^2$ | $\operatorname{tr}(C^2 A) - \tau_C\operatorname{tr}(A) - \operatorname{tr}(C)\operatorname{tr}(AC)$ |
| $\theta_1\theta_2$ | $\operatorname{tr}(A)\operatorname{tr}(B)\operatorname{tr}(C) - \operatorname{tr}(AC)\operatorname{tr}(B) - \operatorname{tr}(C)\operatorname{tr}(AB) - \operatorname{tr}(A)\operatorname{tr}(BC) + \operatorname{tr}(ACB) + \operatorname{tr}(ABC)$ |
| $\theta_1^3$ | $\det(B)$ |
| $\theta_2^3$ | $\det(C)$ |
| $\theta_1^2\theta_2$ | $\operatorname{tr}(B^2 C) - \operatorname{tr}(B)\operatorname{tr}(BC) - \tau_B\operatorname{tr}(C) = \operatorname{tr}(\operatorname{adj}(B)\, C)$ |
| $\theta_1\theta_2^2$ | $\operatorname{tr}(C^2 B) - \operatorname{tr}(C)\operatorname{tr}(BC) - \tau_C\operatorname{tr}(B) = \operatorname{tr}(\operatorname{adj}(C)\, B)$ |

The mixed $\theta_1\theta_2$ coefficient follows from
$\partial^2\det/(\partial\theta_1\,\partial\theta_2)\big|_{\theta=0}$ via
Jacobi's formula.

**Bivariate adjugate** (6 non-zero coefficients):

| Monomial | Coefficient |
|----------|------------|
| $1$ | $\operatorname{term}_A - \tau_A I$ |
| $\theta_1$ | $\bigl(\operatorname{tr}(A)\operatorname{tr}(B)-\operatorname{tr}(AB)\bigr)I - \bigl(A\operatorname{tr}(B)+B\operatorname{tr}(A)\bigr) + AB+BA$ |
| $\theta_2$ | $\bigl(\operatorname{tr}(A)\operatorname{tr}(C)-\operatorname{tr}(AC)\bigr)I - \bigl(A\operatorname{tr}(C)+C\operatorname{tr}(A)\bigr) + AC+CA$ |
| $\theta_1^2$ | $\operatorname{term}_B - \tau_B I$ |
| $\theta_2^2$ | $\operatorname{term}_C - \tau_C I$ |
| $\theta_1\theta_2$ | entry-by-entry from $2\times 2$ minor cross-terms (see below) |

The $\theta_1\theta_2$ adjugate entry $(i,j)$ is

$$[\operatorname{adj}(J)]_{ij}^{(1,1)} = (-1)^{i+j}
  \bigl(B_{r_1 c_1}C_{r_2 c_2} + C_{r_1 c_1}B_{r_2 c_2}
       - B_{r_1 c_2}C_{r_2 c_1} - C_{r_1 c_2}B_{r_2 c_1}\bigr)$$

where $(r_1, r_2)$ are the row indices $\neq j$ and $(c_1, c_2)$ the column
indices $\neq i$. This is the $2\times 2$ minor contribution with one factor
from $B = J_1$ and one from $C = J_2$, independent of $J_0$ (since total
minor degree equals $1+1 = 2$).

---

## 4. Pull-Back Integrals and the adj/det Substitution

### 4.1 Standard Pull-Back

For a parametrised reference domain with Jacobian $J(\theta)$, the change of
variables from deformed ($\Omega$) to reference ($\Omega_0$) coordinates gives:

$$dV = \det(J)\,dV_0$$

For a **scalar** field $\phi$, the physical gradient (a column vector) transforms as
$\nabla_x \phi = J^{-T}\nabla_0\phi$. For a **vector** displacement field $u$, the
displacement gradient matrix (rows = field components, columns = spatial indices)
transforms differently: right-multiplying by $J^{-1}$ pulls back column indices,

$$\nabla_x u = \nabla_0 u \cdot J^{-1}$$

Substituting $J^{-1} = \operatorname{adj}(J)/\det(J)$ keeps the tensor part polynomial:

$$\nabla_x u = \frac{\nabla_0 u \cdot \operatorname{adj}(J)}{\det(J)}$$

### 4.2 Linear Stiffness

The St. Venant–Kirchhoff linear stiffness (small-strain) pulled back to the
reference domain is:

$$a(u, v; \theta) = \int_{\Omega_0}
  \underbrace{\operatorname{sym}\!\bigl(\nabla_0 v \cdot J^{-1}\bigr)}_{\varepsilon_{J^{-1}}(v)}
  \mathbin{:} \mathbb{C} \mathbin{:}
  \underbrace{\operatorname{sym}\!\bigl(\nabla_0 u \cdot J^{-1}\bigr)}_{\varepsilon_{J^{-1}}(u)}
  \det(J)\, dV_0$$

Substituting $J^{-1} = \operatorname{adj}(J)/\det(J)$:

$$a(u, v; \theta) = \int_{\Omega_0}
  \underbrace{\operatorname{sym}\!\bigl(\nabla_0 v \cdot \operatorname{adj}(J)\bigr)}_{\varepsilon_{\text{adj}}(v)}
  \mathbin{:} \mathbb{C} \mathbin{:}
  \underbrace{\operatorname{sym}\!\bigl(\nabla_0 u \cdot \operatorname{adj}(J)\bigr)}_{\varepsilon_{\text{adj}}(u)}
  \cdot \frac{1}{\det J(\theta)}\, dV_0$$

One $\det(J)$ factor in the numerator (volume element) cancels one from the
denominator (two $J^{-1}$ bring $1/\det^2$, then $\det \cdot 1/\det^2 = 1/\det$).
The result has exactly **one** $1/\det$ factor. The integrand is a polynomial
in $\theta$ since $\varepsilon_{\text{adj}}$ involves the polynomial series
$\operatorname{adj}(J)$ and $1/\det$ is the reciprocal series from Section 2.

The material law is isotropic (Lamé parameters $\lambda$, $\mu$):

$$\mathbb{C}\varepsilon = \lambda\operatorname{tr}(\varepsilon)I + 2\mu\varepsilon$$

$\mathbb{C}$ is **spatially uniform** ($\lambda$ and $\mu$ are constants; the
position-dependence of the stiffness integrand comes entirely from $\operatorname{adj}(J)$
and $\det(J)$). The isotropic Lamé form satisfies **major symmetry**
$\mathbb{C}_{ijkl} = \mathbb{C}_{klij}$, which implies $a(u,v;\theta)=a(v,u;\theta)$
and hence each coefficient matrix $K_{k_1,k_2}$ is symmetric.

### 4.3 Linear Mass

The mass form contains no spatial gradients:

$$m(u,v;\theta) = \int_{\Omega_0} \rho\, u \cdot v \cdot \det J(\theta)\,dV_0$$

This carries **one positive** power of $\det(J)$ and no reciprocal factor.

### 4.4 Counting Reciprocal Factors

After the $dV$ cancellation, an elastic form with $N_{\text{input}}$
displacement inputs (each contributes one factor $\nabla_0(\cdot)\cdot J^{-1}$) carries:

| Form | Displacement inputs | Factor |
|------|---------------------|--------|
| Linear stiffness $K$ | 1 | $(1/\det)^1$ |
| Linear mass $M$ | 0 (no $\nabla$) | $(\det)^{+1}$ |
| Quadratic $g$ | 2 | $(1/\det)^2$ |
| Cubic $h$ | 3 | $(1/\det)^3$ |

This is the central bookkeeping identity: **an $N_{\text{input}}$-displacement
elastic form carries $(1/\det)^{N_{\text{input}}}$** after the volume element
absorbs one factor.

### 4.5 Bivariate Coefficient Matrices

The complete bivariate polynomial stiffness and mass are:

$$K(\theta_1,\theta_2) = \sum_{\substack{k_1,k_2 \geq 0 \\ k_1+k_2 \leq N_\theta}} K_{k_1,k_2}\,\theta_1^{k_1}\theta_2^{k_2}, \qquad
  M(\theta_1,\theta_2) = \sum_{\substack{k_1,k_2 \geq 0 \\ k_1+k_2 \leq N_\theta}} M_{k_1,k_2}\,\theta_1^{k_1}\theta_2^{k_2}$$

where the coefficient matrices are assembled QP-by-QP:

$$[K_{k_1,k_2}]_{ij} = \int_{\Omega_0}
  \left[\varepsilon_{\text{adj}}(N_i) \mathbin{:} \mathbb{C} \mathbin{:} \varepsilon_{\text{adj}}(N_j) \cdot \frac{1}{\det J}\right]_{k_1,k_2} dV_0$$

$$[M_{k_1,k_2}]_{ij} = \int_{\Omega_0}
  \rho\, N_i \cdot N_j \cdot [\det J]_{k_1,k_2}\, dV_0$$

At each quadrature point, `det_and_adj_bseries` is called with the per-QP
$J_2(x_0) = \nabla_0\varphi_1(x_0)$, yielding a local bivariate polynomial; the
coefficient-by-coefficient contributions accumulate into `K_b[k1+1, k2+1]` and
`M_b[k1+1, k2+1]` (implemented in `assemble_K_M_bivariate!`,
`parametric_assembly.jl`).

---

## 5. Nonlinear Geometric Maps

### 5.1 Adj-Scaled Displacement Gradient

Define the **adj-scaled gradient** (no $1/\det$ yet) at a quadrature point:

$$\widehat\nabla u = \nabla_0 u \cdot \operatorname{adj}(J(\theta))$$

This is a polynomial series in $(\theta_1,\theta_2)$ of degree $\leq 2$ (one
factor of $\operatorname{adj}$, which is degree $\leq 2$).

The true parametric gradient is $\nabla_\theta u = \widehat\nabla u / \det J$,
so each factor of $\nabla_\theta$ brings one power of $1/\det$.

### 5.2 Nonlinear Strain Piece

The Green–Lagrange nonlinear strain contribution from two displacements
$(u_A, u_B)$ — before the $1/\det$ factors — is:

$$\widehat{E}_{\text{nl}}(u_A, u_B) = \frac{1}{4}\bigl[\widehat\nabla u_A^T \cdot \widehat\nabla u_B + \widehat\nabla u_B^T \cdot \widehat\nabla u_A\bigr]$$

This is a polynomial in $\theta$ of degree $\leq 4$ (product of two degree-2
series). The true nonlinear strain is $E_{\text{nl}} = \widehat{E}_{\text{nl}}/\det^2$.

### 5.3 Quadratic Internal Force

The quadratic Galerkin form (symmetric in $(u_1, u_2)$ by the three-term
expansion of Green–Lagrange) is:

$$g(u_1, u_2, v;\theta) = \int_{\Omega_0}
  \bigl[t_1 + \tfrac{1}{2}(t_2 + t_3)\bigr] \cdot \frac{1}{\det^2 J}\,dV_0$$

with:

$$\begin{aligned}
t_1 &= \varepsilon_{\text{adj}}(v) \mathbin{:} \mathbb{C} \mathbin{:} \widehat{E}_{\text{nl}}(u_1, u_2) \\
t_2 &= \operatorname{sym}\!\bigl(\widehat\nabla u_1^T \widehat\nabla v\bigr) \mathbin{:} \mathbb{C} \mathbin{:} \varepsilon_{\text{adj}}(u_2) \\
t_3 &= \operatorname{sym}\!\bigl(\widehat\nabla u_2^T \widehat\nabla v\bigr) \mathbin{:} \mathbb{C} \mathbin{:} \varepsilon_{\text{adj}}(u_1)
\end{aligned}$$

Each bracket is a bivariate polynomial series assembled via `bpoly_contract` and
`bpoly_dot`. The $(1/\det)^2$ factor is then applied as:

```
with_invdet2 = bpoly_mul(integ_b, inv_det2_b, N_θ)
```

**Symmetry in $(u_1,u_2)$**: $g$ is symmetric in its two displacement arguments,
and this holds at every individual coefficient $(k_1,k_2)$ of the $\theta$ expansion.
For $t_1$: each convolution summand $\varepsilon_\text{adj}^{(m)}(v):\mathbb{C}:\hat{E}_\text{nl}^{(n)}(u_1,u_2)$
is symmetric because $\hat{E}_\text{nl}^{(n)}$ is symmetric at every individual degree $n$
(swapping $u_1\leftrightarrow u_2$ and relabelling the adj-degree summation index leaves
the expression unchanged).
For $t_2+t_3$: for each convolution pair $(m,n)$, the $t_2$ contribution at $(m,n)$
equals the $t_3$ contribution with $u_1\leftrightarrow u_2$, so their sum is symmetric
pair-by-pair. The code does not perform any additional symmetrization — it falls out
of the algebraic structure automatically.

### 5.4 Cubic Internal Force

The cubic Galerkin form (symmetric in $(u_1, u_2, u_3)$) is:

$$h(u_1, u_2, u_3, v;\theta) = \int_{\Omega_0}
  \frac{1}{3}\bigl[s_1 + s_2 + s_3\bigr] \cdot \frac{1}{\det^3 J}\,dV_0$$

with:

$$\begin{aligned}
s_1 &= \operatorname{sym}\!\bigl(\widehat\nabla u_1^T \widehat\nabla v\bigr) \mathbin{:} \mathbb{C} \mathbin{:} \widehat{E}_{\text{nl}}(u_2, u_3) \\
s_2 &= \operatorname{sym}\!\bigl(\widehat\nabla u_2^T \widehat\nabla v\bigr) \mathbin{:} \mathbb{C} \mathbin{:} \widehat{E}_{\text{nl}}(u_1, u_3) \\
s_3 &= \operatorname{sym}\!\bigl(\widehat\nabla u_3^T \widehat\nabla v\bigr) \mathbin{:} \mathbb{C} \mathbin{:} \widehat{E}_{\text{nl}}(u_1, u_2)
\end{aligned}$$

**Symmetry in $(u_1,u_2,u_3)$**: $h$ is symmetric in all three displacement arguments
at every coefficient $(k_1,k_2)$ of the $\theta$ expansion. Any transposition of
$(u_1,u_2,u_3)$ merely permutes the three terms $s_1,s_2,s_3$.

### 5.5 Reciprocal Powers per Quadrature Point

The scalar series $(1/\det)^{N_{\text{input}}}$ is computed **per quadrature point**
for the bivariate assembly:

```
inv_det2_b = bpoly_power(inv_det_b, 2, N_θ)   # for quadratic, per QP
inv_det3_b = bpoly_power(inv_det_b, 3, N_θ)   # for cubic, per QP
```

This must be done per QP because $\det J(\theta_1,\theta_2;x_0)$ depends on
$J_2(x_0)=\nabla_0\varphi_1(x_0)$, which varies by quadrature point. (For the
univariate-only case $J=I+\theta_1 J_1$ with $J_1$ constant, the det/adj series are
QP-independent and could in principle be precomputed once.)
The integrand bracket is then completed in a single `bpoly_mul` call per QP.

---

## 6. ROM Linear Corrections

### 6.1 Parametric Equations of Motion

On the reference domain, the equations of motion are:

$$M(\theta)\ddot{u} + C(\theta)\dot{u} + K(\theta)\,u = f_{\text{nl}}(u;\theta)$$

where $C(\theta) = \alpha M(\theta) + \beta K(\theta)$ (Rayleigh damping) and
$f_{\text{nl}}$ contains the quadratic and cubic internal forces from Section 5.
Expanding:

$$\bigl[K_{00} + \sum_{k_1+k_2 \geq 1} K_{k_1,k_2}\,\theta_1^{k_1}\theta_2^{k_2}\bigr]\,u + \cdots$$

$K_{00}$ and $M_{00}$ become the reference-configuration operators of the
`NDOrderModel`. Every remaining coefficient matrix $K_{k_1,k_2}$ (with
$k_1+k_2 \geq 1$) becomes a `MultilinearMap` linear correction.

### 6.2 Wrapping as MultilinearMaps

Each correction term $K_{k_1,k_2}\,\theta_1^{k_1}\theta_2^{k_2}\,u$ is wrapped as a
`MultilinearMap` with:

| Property | Value |
|----------|-------|
| modal arity $(a_{\text{pos}}, a_{\text{vel}}, a_{\text{acc}})$ | $(1,0,0)$ for $K$; $(0,1,0)$ for $C$; $(0,0,1)$ for $M$ |
| external arity | $k_1 + k_2$ |
| sign | negative (MORFE places nonlinear terms on the RHS) |

### 6.3 Permutation Scaling

MORFE internally sums over all $\binom{k_1+k_2}{k_1}$ sorted assignments of
$(k_1+k_2)$ external slots to the two parameters $(\theta_1, \theta_2)$. To
cancel this overcounting, each closure is pre-scaled by

$$\text{scale} = \frac{1}{\binom{k_1+k_2}{k_1}} = \frac{k_1!\,k_2!}{(k_1+k_2)!}$$

The closure factories `_wrap_linK2(Val(k1), Val(k2), Kk)` (generated at
file-load time via `@eval` metaprogramming in `parametric_assembly.jl`) handle
the arity matching and permutation scaling automatically.

### 6.4 External States (Frozen Dynamics)

The two parameters $(\theta_1, \theta_2)$ enter the DPIM system as **external
states** with eigenvalues $\lambda_{\text{ext}} = 0$ (frozen dynamics,
$\dot\theta = 0$):

```julia
ext_sys = ExternalSystem((complex(0.0, 0.0), complex(0.0, 0.0)))
```

The augmented reduced coordinates are $(z_1, z_2, \theta_1, \theta_2)$, giving
NVAR = 4. An anisotropic truncation separates the dynamical and parametric
orders:

$$\deg(z_1) + \deg(z_2) \leq \texttt{max\_degree\_z}, \qquad
  \deg(\theta_1) + \deg(\theta_2) \leq \texttt{max\_degree\_}\theta$$

---

## 7. Verification Identities

### 7.1 Polynomial Cofactor Identity

The identity $J \cdot \operatorname{adj}(J) = \det(J)\,I$ must hold **at each
monomial coefficient**. This is verified by computing the bivariate convolution
$J_b \star \operatorname{adj}_b$ and checking that every residual norm

$$\bigl\|[J(\theta_1,\theta_2)\cdot\operatorname{adj}(J(\theta_1,\theta_2)) - \det(J(\theta_1,\theta_2))\,I]_{k_1,k_2}\bigr\|_F \leq 10^{-12}$$

implemented in `check_adj_det_bidentity` (`bivariate_geometry.jl`). This test
exercises only the exact polynomial algebra — independent of the reciprocal
series — and provides a machine-precision check of the det and adj formulas.

### 7.2 Modal Stiffness/Mass Diagnostic

For a mode $\varphi$ (not necessarily mass-normalised, $\varphi^T M_0 \varphi \neq 1$),
the parametric sensitivity of the eigenfrequency at $\theta = 0$ is:

$$\frac{\partial\omega}{\partial\theta_i}\bigg|_{\theta=0}
  = \frac{\varphi^T K_{e_i} \varphi - \omega_0^2 \,\varphi^T M_{e_i} \varphi}{2\omega_0\,\varphi^T M_0 \varphi}$$

where $K_{e_i}$, $M_{e_i}$ are the first-order bivariate coefficient matrices
in direction $\theta_i$.

**Expected values** (checked in `main.jl` before the DPIM solve):

| Sensitivity | Expected | Reason |
|-------------|----------|--------|
| $\partial\omega/\partial\theta_1$ | $\approx -2\omega_0$ | Bending eigenfrequency scales as $(1+\theta_1)^{-2}$ under uniform axial stretch |
| $\partial\omega/\partial\theta_2$ | $\approx 0$ | Arch pre-deformation does not alter the linearised frequency to first order |

The $-2\omega_0$ value can also be read off from the DPIM output: the $(1,0,1,0)$
monomial coefficient (one $z_1$, zero $z_2$, one $\theta_1$, zero $\theta_2$)
of the reduced dynamics $R$ should equal $-2\lambda_1 = -2\omega_0$ for the
first bending mode.

---

## File Map

| File | Content |
|------|---------|
| `fem/theta_polynomials.jl` | Scalar univariate series: `poly_mul`, `poly_contract`, `reciprocal_series` |
| `fem/parametric_geometry.jl` | Univariate det/adj: `det_and_adj_series`, `check_adj_det_identity` |
| `fem/bivariate_polynomials.jl` | Bivariate series: `bpoly_mul`, `breciprocal_series`, `bpoly_power` |
| `fem/bivariate_geometry.jl` | Bivariate det/adj: `det_and_adj_bseries`, `check_adj_det_bidentity` |
| `fem/parametric_assembly.jl` | FEM assembly: `assemble_K_M_bivariate!`, `ParametricGeometricNonlinearity2D`, `multilinear_maps`, correction builders |
| `main.jl` | Pipeline driver: mesh → eigenproblem → bivariate assembly → DPIM solve |
