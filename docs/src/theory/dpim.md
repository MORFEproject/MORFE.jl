# Mathematical Background: DPIM

This page derives the **Direct Parametrisation of Invariant Manifolds (DPIM)** method from first
principles, connecting the mathematics to the corresponding MORFE.jl data types and solver steps.

---

## 1. Problem Statement

Finite element discretisation of structures with geometric nonlinearities yields a large second-order
ODE in $\mathbb{R}^n$ (with $n \sim 10^4$–$10^6$ in practical models):

$$M\ddot{x} + C\dot{x} + Kx = F_{\text{nl}}(x,\dot{x}) + f_{\text{ext}}(t), \tag{1}$$

where $M, C, K \in \mathbb{R}^{n \times n}$ are the mass, damping and stiffness matrices,
$F_{\text{nl}}$ collects the nonlinear restoring and damping forces, and $f_{\text{ext}}(t)$ is
external excitation.  Direct time-integration of $(1)$ is feasible but expensive; continuation of
periodic orbits or frequency-response curves is impractical.

**Goal.** Find a low-dimensional ($m \ll n$) **reduced-order model (ROM)** that faithfully captures
the near-resonant dynamics and permits rapid parameter studies.

The DPIM achieves this by computing a polynomial map from $m$ reduced coordinates to the full
$n$-dimensional state, whose image is an *invariant manifold* of the flow.

!!! note "Scope of MORFE.jl"
    MORFE.jl implements DPIM for $N$-th order ODEs of the general form
    $\sum_{k=0}^N B_k x^{(k)} = F(x, \dot x, \ldots, x^{(N-1)}, r)$
    where $r$ are autonomous external variables (see §6).  The second-order case $(N=2)$ covers
    virtually all structural dynamics models.

---

## 2. First-Order Reformulation

Introduce the extended state vector $u = (x,\, \dot x,\, \ldots,\, x^{(N-1)}) \in \mathbb{R}^{Nn}$
and rewrite $(1)$ in first-order companion form:

$$B\dot u = Au + G(u) + \Phi r, \tag{2}$$

where

| Symbol | Size | Description |
|--------|------|-------------|
| $A, B$ | $Nn \times Nn$ | Companion-form linear matrices |
| $G(u)$ | $\mathbb{R}^{Nn}$ | Polynomial nonlinear terms |
| $\Phi$ | $Nn \times N_\text{ext}$ | Forcing-direction columns |
| $r$ | $\mathbb{C}^{N_\text{ext}}$ | External forcing variables |

For the standard second-order case $N=2$ with $n$ DOF:

$$A = \begin{pmatrix} -K & 0 \\ 0 & M \end{pmatrix}, \quad
  B = \begin{pmatrix} C & M \\ M & 0 \end{pmatrix}, \quad
  u = \begin{pmatrix} x \\ \dot x \end{pmatrix}.$$

In MORFE.jl, `linear_first_order_matrices(model)` returns $(A, B)$ from an
`NDOrderModel`.

---

## 3. Spectral Decomposition

The linear part of $(2)$ defines the **generalised eigenproblem**:

$$(A - \lambda B)\varphi = 0, \qquad B^\top\!(A - \lambda B)^{-\top}\psi = 0,$$

yielding right eigenvectors $\varphi_i$ and left eigenvectors $\psi_i$ with
$\langle \psi_i, B\varphi_j \rangle = \delta_{ij}$.

For a lightly-damped structure the eigenvalues come in complex-conjugate pairs:

$$\lambda_{2k-1} = \mu_k + i\omega_k, \qquad \lambda_{2k} = \bar\lambda_{2k-1}, \qquad
\mu_k < 0,\; \omega_k > 0.$$

The real part $\mu_k < 0$ is the decay rate and $\omega_k$ is the damped natural frequency of the
$k$-th mode pair.

**Master mode selection.**  Choose $m/2$ conjugate pairs whose eigenvalues are closest to the
imaginary axis (least damped) or are resonantly driven.  This gives $m$ **master eigenvalues**
$\lambda_1,\ldots,\lambda_m$ and corresponding **master modes** $\Phi_m = [\varphi_1,\ldots,\varphi_m]$.
The complementary $2Nn - m$ modes are **slave modes**.

In MORFE.jl:

```julia
A, B    = linear_first_order_matrices(model)
result  = generalised_eigenpairs(A, B; nev = 4, sigma = 0.0)
# or use Julia's built-in eigen for small systems:
result  = eigen(A, B)
```

---

## 4. Invariant Manifolds and the Spectral Gap

!!! info "Existence of a smooth invariant manifold (Cabré, de la Llave & Jorba 2003)"
    Let $E_m$ be a spectral subspace of $A$ associated with eigenvalues
    $\Lambda_m = \{\lambda_1,\ldots,\lambda_m\}$.  If the **non-resonance** (spectral gap)
    condition holds for all slave eigenvalues $\lambda_j$, then there exists a unique smooth
    invariant manifold $\mathcal{W}(E_m)$ tangent to $E_m$ at the origin, and its dynamics can
    be expressed as a convergent power series in the reduced coordinates.

The non-resonance condition rules out integer combinations of master eigenvalues from coinciding
with slave eigenvalues.  For a system with slave eigenvalues $\lambda_j$ ($j \notin \Lambda_m$)
the condition is:

$$\lambda_j \neq \sum_{i=1}^m k_i\,\lambda_i, \quad
  k_i \in \mathbb{N},\; \sum_i k_i \geq 1, \tag{3}$$

for all slave modes $j$.  In structural dynamics this is almost always satisfied for
well-separated mode pairs.

**Why invariant manifolds are useful.**  The restriction of the full $Nn$-dimensional flow to the
$m$-dimensional invariant manifold $\mathcal{W}(E_m)$ is an *exact* reduced-order model — it
captures all orbits that start on the manifold forever, without projection error.  The
*parametrisation method* (Cabré, de la Llave & Jorba 2003; see also §5–7) computes a polynomial
approximation to this manifold up to user-chosen order $p$.

---

## 5. The Parametrisation Ansatz

DPIM seeks a pair of polynomial maps $(W, R)$ satisfying:

$$u(t) = W(\mathbf{z}(t),\, r(t)), \qquad
\dot z_i = R_i(\mathbf{z}(t),\, r(t)), \quad i = 1,\ldots,m, \tag{4}$$

expanded as multivariate polynomials in $\text{NVAR} = m + N_\text{ext}$ variables:

$$W(\mathbf{z}, r) = \sum_{\alpha \in \mathcal{A}} W_\alpha\, \mathbf{z}^{\alpha_z} r^{\alpha_r},
\qquad
R_i(\mathbf{z}, r) = \sum_{\alpha \in \mathcal{A}} R_{i,\alpha}\, \mathbf{z}^{\alpha_z} r^{\alpha_r},
\tag{5}$$

where $\alpha = (\alpha_z, \alpha_r) \in \mathbb{N}^m \times \mathbb{N}^{N_\text{ext}}$ is a
multiindex and $\mathcal{A}$ is the **multiindex set** of all monomials up to total degree $p$.

The initialisation conditions fix the linear part:

$$W_\alpha = \varphi_r \text{ for } \alpha = e_r \quad (r = 1,\ldots,m), \qquad
R_{r,\alpha} = \lambda_r \text{ for } \alpha = e_r, \tag{6}$$

so that $W$ is tangent to $E_m$ at the origin and $R$ reduces to the linearised flow on $E_m$.

**In MORFE.jl:**

| Concept | Type |
|---------|------|
| $W$ | `Parametrisation{ORD, NVAR}` — coefficients shape $(n,\, N,\, L)$ |
| $R$ | `ReducedDynamics{NVAR}` — coefficients shape $(\text{NVAR},\, L)$ |
| $\mathcal{A}$ | `MultiindexSet` sorted in graded lex order |

---

## 6. The Invariance Equation

The image of $W$ is an invariant manifold of the flow $(2)$ if and only if $W$ and $R$ satisfy the
**invariance equation**:

$$\boxed{
B\,\frac{\partial W}{\partial \mathbf{z}}\,R(\mathbf{z}, r)
+ B\,\frac{\partial W}{\partial r}\,E(r)
= A\,W(\mathbf{z}, r) + G(W(\mathbf{z}, r)) + \Phi r
} \tag{7}$$

where $E(r) = \dot r$ is the autonomous dynamics of the external forcing variables.  For harmonic
excitation at frequency $\Omega$: $r = e^{i\Omega t}$, so $E(r) = i\Omega\,r$ and equation $(7)$
captures the *steady-state forced response on the SSM*.

**Derivation.**  The chain rule gives
$\dot u = \tfrac{\partial W}{\partial \mathbf{z}}\dot{\mathbf{z}} + \tfrac{\partial W}{\partial r}\dot r
= \tfrac{\partial W}{\partial \mathbf{z}}R + \tfrac{\partial W}{\partial r}E$,
which upon substitution into $(2)$ yields $(7)$.

---

## 7. Cohomological Equations

### 7.1 Monomial decomposition

Substitute the polynomial ansatz $(5)$ into the invariance equation $(7)$ and match coefficients of
each monomial $\mathbf{z}^\alpha r^{\alpha_r}$.  On the left-hand side of $(7)$, the product
$\tfrac{\partial W}{\partial \mathbf{z}} R$ contributes at monomial $\alpha$ through all pairs
$(\beta, \gamma)$ with $\beta + \gamma = \alpha$ and $|\gamma| = 1$:

$$\sum_{r=1}^m R_{r,\,\alpha-e_r}\,B\,(s_r W_{\alpha-e_r})$$

where $s_r = \partial_{\mathbf{z}_r}$ denotes differentiation in the $r$-th coordinate.  Collecting
all terms at order $\alpha$ yields the **cohomological equation** for $W_\alpha$:

$$\underbrace{[A - s B]}_{\mathcal{L}(s)}\, W_\alpha
= \underbrace{N_\alpha(W_{|\beta|<|\alpha|})}_{\text{nonlinear RHS}}
- B\sum_{r \in \mathcal{R}_\alpha} R_{r,\alpha}\,\varphi_r
- B\,\Phi_\text{ext}\,R_\alpha^\text{ext}
\tag{8}$$

### 7.2 Superharmonic frequency

The **superharmonic frequency** is the inner product of the master eigenvalue vector with the
multiindex:

$$s = \langle\lambda, \alpha\rangle = \sum_{i=1}^{\text{NVAR}} \lambda_i\,\alpha_i \tag{9}$$

It appears as the evaluation point of the linear operator $\mathcal{L}(s) = A - sB$.

### 7.3 The nonlinear right-hand side

$N_\alpha$ collects all contributions of the nonlinear terms $G(u)$ to monomial $\alpha$.  For a
multilinear map $t$ of degree $d$ (e.g., a cubic stiffness is $d=3$), the contribution is a sum
over all *factorisations* $(\beta_1,\ldots,\beta_d)$ with $\beta_1 + \cdots + \beta_d = \alpha$
and $|\beta_k| < |\alpha|$:

$$[N_\alpha]_t = \sum_{\substack{|\beta_1| + \cdots + |\beta_d| = |\alpha| \\ |\beta_k| \geq 1}}
c(\beta_1,\ldots,\beta_d)\cdot t(W_{\beta_1},\ldots,W_{\beta_d}) \tag{10}$$

where $c$ is the symmetry multiplicity (multinomial coefficient for equal slots).  Since each
$|\beta_k| < |\alpha|$, all required $W_{\beta_k}$ are known when equation $(8)$ is solved —
the graded lex ordering of $\mathcal{A}$ guarantees **causality**.

In MORFE.jl, `compute_multilinear_terms` evaluates $(10)$ for all nonlinear terms simultaneously,
using cached factorisations.

### 7.4 Resonances

The linear operator $\mathcal{L}(s)$ is **singular** when $s = \lambda_r$ for some master
eigenvalue $\lambda_r$ (i.e., when the superharmonic coincides with a master eigenvalue).  In this
case equation $(8)$ has no solution unless the right-hand side is orthogonal to the corresponding
left eigenvector $\psi_r$ — this is the **resonance condition**.

Three approaches are available in MORFE.jl:

| Style | `resonance_set_from_*` | Description |
|-------|------------------------|-------------|
| **Graph style** | `graph_style` | Resonant if $|\lambda_r - s| < \varepsilon\cdot|\lambda_r|$ |
| **Normal form** | `complex_normal_form_style` | Resonant if the monomial is in the normal-form basis |
| **Condition number** | `condition_number_style` | Resonant if $\|\mathcal{L}(s)^{-1}\|$ exceeds a threshold |

When mode $r$ is resonant at $\alpha$, the reduced-dynamics coefficient $R_{r,\alpha}$ becomes an
**additional unknown** and the cohomological equation is augmented with the orthogonality condition:

$$\langle \psi_r,\, [A - sB]\,W_\alpha + B\,R_{r,\alpha}\,\varphi_r \rangle
= \langle \psi_r,\, N_\alpha \rangle, \tag{11}$$

yielding a stacked $(n + n_\mathcal{R}) \times (n + n_\mathcal{R})$ square system for
$(W_\alpha,\, R_{\mathcal{R},\alpha})$.

---

## 8. External Forcing as Autonomous Dynamics

Harmonic excitation $f_\text{ext}(t) = \hat{f}\,e^{i\Omega t}$ is embedded as an **autonomous
external variable** $r(t) = e^{i\Omega t}$ satisfying $\dot r = i\Omega\,r$, so

$$E(r) = i\Omega\,r, \qquad \lambda_\text{ext} = i\Omega. \tag{12}$$

The extended NVAR-variable space then includes $r$ as an extra coordinate alongside the $m$ master
reduced coordinates $\mathbf{z}$.  Monomials in $r$ of degree $k$ correspond to the $k$-th
super-harmonic of the excitation.

The ExternalSystem additionally supports quasi-periodic forcing and autonomous nonlinear oscillators
as external variables, by specifying a polynomial $\dot r = g(r)$ (e.g., $g(r) = i\Omega\,r +
\varepsilon r^2$ for a van der Pol driver).

```julia
external_system = ExternalSystem(
    DensePolynomial(                              # ṙ = iΩ·r
        ComplexF64[1.0im],                        # coefficient
        MultiindexSet([[1]]),                     # monomial r¹
    )
)
```

---

## 9. DPIM Algorithm Summary

Putting the pieces together, MORFE.jl solves the following sequence:

```
Input:  system matrices (B₀,…,B_N), nonlinear terms, ExternalSystem
        polynomial degree p, resonance tolerance ε

1.  Compute (A, B)              ← linear_first_order_matrices(model)
2.  Solve (A − λB)φ = 0        ← generalised_eigenpairs / eigen
3.  Select m master modes       ← user choice (e.g., least damped)
4.  Build MultiindexSet         ← all_multiindices_up_to(NVAR, p)
5.  Build ResonanceSet          ← resonance_set_from_graph_style(...)
6.  Initialise W, R at degree 1 ← φᵣ, λᵣ  (§5, eq. 6)
7.  For each α ∈ 𝒜 in GrLex order (|α| ≥ 2):
      a. s ← ⟨λ, α⟩              (superharmonic)
      b. resonances ← 𝒮_α         (bitmask from ResonanceSet)
      c. Nα ← compute_multilinear_terms(model, α, W)
      d. ξ  ← compute_lower_order_couplings(...)
      e. Assemble stacked system  [L(s) C(s); L̂(s) Ĉ(s)]
      f. Solve for (W[α], R_res[α])
      g. Recover higher-derivative slices of W[α]
8.  Return (W, R)

Output: Parametrisation W  (shape n × N × L),
        ReducedDynamics R  (shape NVAR × L)
```

The GrLex ordering ensures step 7c can always use already-solved lower-order coefficients.

---

## 10. Realification

MORFE.jl computes $W$ and $R$ in **complex normal coordinates** $z_i \in \mathbb{C}$.  For
conjugate mode pairs $(\varphi_r, \bar\varphi_r)$ with eigenvalues $(\lambda_r, \bar\lambda_r)$,
the physical state is real: $x = W(\mathbf{z})$ with $\bar{\mathbf{z}}$ being the complex
conjugate of $\mathbf{z}$.

The `Realification` module converts a polynomial in complex variables $(z, \bar z)$ to an
equivalent polynomial in real variables $(a, b)$ via $z = a + ib$, $\bar z = a - ib$, using the
binomial expansion:

$$z^p \bar z^q = \sum_{j=0}^p\sum_{k=0}^q \binom{p}{j}\binom{q}{k} i^{j-k}
  a^{p+q-j-k}\, b^{j+k}. \tag{13}$$

After realification, the reduced-order model can be integrated or continued in real arithmetic.

---

## 11. Worked Example: 2-DOF Duffing System

Consider two coupled Duffing oscillators:

$$\begin{pmatrix}1&0\\0&1\end{pmatrix}\ddot{x}
+\begin{pmatrix}0.01&0\\0&0.01\end{pmatrix}\dot{x}
+\begin{pmatrix}2&-1\\-1&2\end{pmatrix}x
+\begin{pmatrix}x_1^3\\0\end{pmatrix}
= \begin{pmatrix}1\\1\end{pmatrix}e^{i\Omega t}. \tag{14}$$

**Eigenvalues** (from the first-order form, $4\times 4$ eigenproblem):
The two undamped mode frequencies are $\omega_1 \approx 1.0$, $\omega_2 \approx \sqrt{3} \approx 1.73$.
With light damping $\zeta = 0.005$: $\lambda_{1,2} \approx -0.005 \pm i$.

**SSM selection**: choose the first conjugate pair as master modes ($m = 2$, ROM = 2).
With one external variable (forcing at $\Omega$): NVAR = 3.

**Polynomial degree** $p = 7$ gives $L = \binom{7+3}{3} = 120$ monomials per coefficient.

The DPIM then produces:
- $W: \mathbb{C}^3 \to \mathbb{C}^4$ — the SSM embedding (120 coefficient matrices of size $4 \times 2$)
- $R: \mathbb{C}^3 \to \mathbb{C}^2$ — the reduced flow on the SSM (120 coefficient vectors)

The frequency-response curve near $\Omega \approx \omega_1$ is obtained by continuing equilibria of
$\dot{\mathbf{z}} = R(\mathbf{z}, e^{i\Omega t})$ in the forcing amplitude or frequency.

!!! tip "Interactive notebook"
    The full computation for this system is implemented as a Pluto.jl notebook in
    `notebooks/duffing_demo.jl`.  Launch it interactively via:

    [![Open in Pluto on Binder](https://img.shields.io/badge/launch-Pluto%20notebook-4063D8?logo=julia)](https://binder.plutojl.org/v2/gh/MORFEproject/MORFE.jl/main?path=notebooks%2Fduffing_demo.jl)

---

## 12. References

1. **Cabré, de la Llave & Jorba** (2003a). *The parameterization method for invariant manifolds I:
   Manifolds associated to non-resonant subspaces.* Indiana University Mathematics Journal 52(2),
   283–328.
   [doi:10.1512/iumj.2003.52.2245](https://doi.org/10.1512/iumj.2003.52.2245)

2. **Cabré, de la Llave & Jorba** (2003b). *The parameterization method for invariant manifolds II:
   Regularity with respect to parameters.* Indiana University Mathematics Journal 52(2), 329–360.
   [doi:10.1512/iumj.2003.52.2346](https://doi.org/10.1512/iumj.2003.52.2346)

3. **Cabré, de la Llave & Jorba** (2005). *The parameterization method for invariant manifolds III:
   Overview and applications.* Journal of Differential Equations 218(2), 444–515.
   [doi:10.1016/j.jde.2004.12.003](https://doi.org/10.1016/j.jde.2004.12.003)

4. **Opreni, Vizzaccaro, Touzé & Frangi** (2021). *Model order reduction based on direct normal
   form: application to large finite element MEMS structures vibrating at resonance.*
   npj Computational Materials 7, 161.
   [doi:10.1038/s41524-021-00630-9](https://doi.org/10.1038/s41524-021-00630-9)

5. **Vizzaccaro, Opreni, Salles, Frangi & Touzé** (2022). *High order direct parametrisation of
   invariant manifolds for model order reduction of mechanical systems: application to generic
   types of nonlinearity.* Nonlinear Dynamics 110, 1941–2023.
   [doi:10.1007/s11071-022-07651-9](https://doi.org/10.1007/s11071-022-07651-9)

6. **Opreni, Vizzaccaro, Frangi & Touzé** (2023). *High-order direct parametrisation of invariant
   manifolds for model order reduction of finite element structures: application to generic
   mechanical systems.* Journal of Sound and Vibration 567, 117813.
   [doi:10.1016/j.jsv.2023.117813](https://doi.org/10.1016/j.jsv.2023.117813)
