# Backbone curve computation — mathematical derivation

## 1. Full-order model

The parametric beam is governed by the second-order ODE

$$M(\theta)\,\ddot{x} + C(\theta)\,\dot{x} + K(\theta)\,x = f_{\mathrm{nl}}(x,\dot{x};\theta),$$

where $x \in \mathbb{R}^N$ is the displacement vector ($N \approx 6000$ free DOFs),
$\theta = (\theta_1, \theta_2)$ are the two scalar parameters (axial stretch and arch amplitude),
and $f_{\mathrm{nl}}$ collects the quadratic and cubic geometric nonlinearities.
The matrices $K(\theta)$, $M(\theta)$, $C(\theta)$ are assembled as bivariate polynomial series
in $(\theta_1, \theta_2)$ up to total degree $N_\theta$.

---

## 2. DPIM parametrisation

The **Direct Parametrisation of Invariant Manifolds** (DPIM) seeks a two-dimensional
Spectral Submanifold (SSM) attached to the first complex-conjugate mode pair
$(\lambda_1, \lambda_2 = \bar\lambda_1)$.  It produces two polynomial maps

$$x(t) = W(z_1(t),\, z_2(t),\, \theta_1,\, \theta_2), \qquad W:\mathbb{C}^2\times\mathbb{R}^2\to\mathbb{R}^N,$$

$$\dot{z} = R(z_1, z_2, \theta_1, \theta_2), \qquad R:\mathbb{C}^2\times\mathbb{R}^2\to\mathbb{C}^2,$$

satisfying the invariance equation $DW\cdot R = \mathcal{F}(W, \dot{W};\theta)$ order by order
in a graded-lex monomial expansion (cohomological equations).

The four variables $(z_1, z_2, \theta_1, \theta_2)$ are treated uniformly; the two external
states $\theta_i$ have frozen dynamics $\dot\theta_i = 0$ and eigenvalue $0$.

---

## 3. Normal-form structure of the ROM

In the **complex normal form** style, the resonance condition for the first component $R_1$
(governing $\dot z_1$) requires

$$a - b = 1, \qquad a,b \ge 0,$$

for every retained monomial $z_1^a\,z_2^b\,\theta_1^{k_1}\,\theta_2^{k_2}$.
All other monomials are non-resonant and are eliminated into $W$.
This leaves

$$\dot z_1 = R_1(z_1, z_2, \theta_1, \theta_2)
           = \sum_{\substack{a-b=1\\k_1,k_2\ge 0}} c_{a,b,k_1,k_2}\,
             z_1^a\, z_2^b\, \theta_1^{k_1}\, \theta_2^{k_2}.$$

At linear order (the only term with $a+b=1$) this reduces to
$\dot z_1 \approx \lambda(\theta)\,z_1$, where $\lambda(\theta)$ is the first eigenvalue of
the linearised system at the given parameters.

---

## 4. Backbone derivation

### 4.1 Polar decomposition of the normal-form flow

Write the complex amplitude in polar form: $z_1(t) = A(t)\,e^{i\phi(t)}$.  Then

$$\dot z_1 = \bigl(\dot A + i A\dot\phi\bigr)\,e^{i\phi}.$$

The normal-form structure (every resonant monomial $z_1^a z_2^b$ satisfies $a-b=1$)
implies that multiplying $z_1 \to e^{i\phi} z_1$, $z_2 \to e^{-i\phi} z_2$ factors out
a global $e^{i\phi}$:

$$R_1\!\left(A e^{i\phi},\, A e^{-i\phi};\,\theta\right)
= e^{i\phi}\,R_1(A,\, A;\,\theta).$$

Substituting into $\dot z_1 = R_1$:

$$\bigl(\dot A + i A\dot\phi\bigr)\,e^{i\phi} = e^{i\phi}\,R_1(A,\,A;\,\theta).$$

Cancelling $e^{i\phi}$ and separating real and imaginary parts gives the
**amplitude–phase equations**:

$$\dot A = \mathrm{Re}\bigl[R_1(A,\,A;\,\theta)\bigr], \qquad
\dot\phi = \frac{\mathrm{Im}\bigl[R_1(A,\,A;\,\theta)\bigr]}{A}.$$

Both right-hand sides depend only on the instantaneous amplitude $A$ and the
parameters $\theta$ — not on the phase $\phi$.  This decoupling is a direct
consequence of the normal form.

### 4.2 When is $r$ constant?

The amplitude is constant ($\dot A = 0$) precisely when

$$\mathrm{Re}\bigl[R_1(r,\,r;\,\theta)\bigr] = 0.$$

- **Undamped systems**: the governing PDE is conservative, so every coefficient
  $c_{a,b,k_1,k_2}$ in the normal form is purely imaginary.  Hence
  $\mathrm{Re}[R_1] = 0$ for all $r$, and every orbit is exactly periodic with
  constant $r$.

- **Lightly damped systems** (this beam): $\mathrm{Re}[\lambda] = \delta < 0$ and
  $\mathrm{Re}[R_1(r,r)] \approx \delta \cdot r \neq 0$, so the amplitude decays
  slowly.  The orbit is not exactly periodic, but the **backbone** is still defined
  as the instantaneous frequency at amplitude $r$.  For light damping ($|\delta| \ll
  \omega_0$) this is an excellent approximation of the peak-frequency locus in forced
  response.

### 4.3 Backbone formula

Assuming constant amplitude $r$ (exact or approximate), the phase equation gives
the backbone frequency directly:

$$\boxed{\Omega(r;\theta_1,\theta_2) = \frac{\mathrm{Im}\bigl[R_1(r,\,r;\,\theta_1,\theta_2)\bigr]}{r}.}$$

The backbone curve is the locus $\{(\Omega(r;\theta), r) : r\ge 0\}$ in the
frequency–amplitude plane at fixed $\theta$.

---

## 5. Realification and practical evaluation

The ROM is stored in the **realified** coordinates $(x, y, \theta_1, \theta_2)$ with
$z_1 = x + iy$, $z_2 = x - iy$, obtained by substituting $z_i = x_i + i y_i$ into the
complex polynomial.  Let

$$\texttt{R1\_cplx}(x, y, \theta_1, \theta_2) := R_1(x+iy,\; x-iy;\; \theta_1, \theta_2).$$

Evaluating at the phase $t = 0$ of the backbone orbit gives $(x, y) = (r, 0)$, i.e.
$z_1 = r$ (real), $z_2 = r$, recovering

$$\texttt{R1\_cplx}(r,\, 0,\, \theta_1, \theta_2) = R_1(r,\, r;\, \theta_1, \theta_2)
= i\Omega r$$

so the backbone formula in code is:

```julia
backbone_Ω(r, θ₁, θ₂) = imag(evaluate(R1_cplx, [r, 0.0, θ₁, θ₂])) / r
```

---

## 6. Linear eigenfrequency

As $r \to 0$ all nonlinear terms vanish and the backbone frequency tends to the
**linear eigenfrequency** at the given parameters.  Because the linear part of
$\texttt{R1\_cplx}$ at the origin is $\lambda(\theta_1,\theta_2)\,(x+iy)$,

$$\left.\frac{\partial\,\texttt{R1\_cplx}}{\partial x}\right|_{(0,0,\theta_1,\theta_2)}
= \lambda(\theta_1,\theta_2),$$

and therefore

```julia
ω₀_of(θ₁, θ₂) = imag(evaluate(dR1dx, [0.0, 0.0, θ₁, θ₂]))
```

where `dR1dx` $= \partial\,\texttt{R1\_cplx}/\partial x$ is obtained by symbolic
polynomial differentiation.

### Axial-stretch scaling

For a beam under uniform axial stretch $\theta_1$, dimensional analysis gives

$$\omega_0(\theta_1) = \omega_0(0)\,(1+\theta_1)^{-2},$$

with Taylor coefficients $(-1)^k(k+1)\,\omega_0(0)$ at order $k$.  This is the
**EB reference** used in `omega0_slope_minus2.png`.

---

## 7. Parametric dependence and truncation order

The ROM polynomial is computed to total degree $p = $ `max_degree` in all four variables
$(z_1, z_2, \theta_1, \theta_2)$, but the bivariate FEM operators $(K_b, M_b)$ are
only assembled to parametric degree $N_\theta$.

A monomial $z_1 \theta_1^k$ has total degree $k+1$, so reliable parametric coefficients
in the ROM require

$$k + 1 \le p \quad\text{and}\quad k \le N_\theta.$$

The binding constraint is $k \le N_\theta$: coefficients at $k > N_\theta$ are not backed
by assembled FEM data and accumulate truncation errors from the cohomological solve.
**This is why the order-$N_\theta$ truncation of the ROM outperforms the full
degree-$p$ polynomial for $|\theta_1| \gtrsim 0.1$** when $N_\theta < p - 1$.

Setting $N_\theta = p - 1$ (e.g. $N_\theta = 4$, $p = 5$) restores consistency and
should recover accurate coefficients up to $k = 4$.

---

## 8. Numerical validation via BifurcationKit

As a cross-check, the backbone is also traced by pseudo-arclength continuation of

$$G(r,\Omega;\theta) := \Omega - \frac{\mathrm{Im}[\texttt{R1\_cplx}(r, 0, \theta)]}{r} = 0,$$

with Jacobian

$$\frac{\partial G}{\partial r} = \frac{r\,\mathrm{Im}[\partial_r R_1] - \mathrm{Im}[R_1]}{r^2}.$$

Scatter points in the backbone figures are the BifurcationKit solution; solid lines
are the analytical formula above.  Agreement between the two confirms the correctness
of the realification and the backbone formula.
