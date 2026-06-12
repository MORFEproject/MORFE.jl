# The physical model — article-style description

Draft section for a *Nonlinear Dynamics* article. Equations in LaTeX; numbering follows
the order of appearance. Companion derivation details: `implementation_plan_detailed.md` §1.

---

## 2. A third-order electromechanical model problem: the dielectric elastomer cantilever

### 2.1 Physical setting

We consider a slender cantilevered beam of length $L$ made of a soft silicone elastomer,
operated as a dielectric elastomer actuator (DEA). Compliant electrodes are deposited on
the upper and lower faces of the beam so that the structure forms a deformable parallel-plate
capacitor. When a voltage is applied across the electrodes, the resulting Maxwell stress
compresses the elastomer through its thickness and, owing to the asymmetric (unimorph)
layout, generates a distributed bending moment along the span. The beam is clamped at one
end and free at the other; the quantity of interest is the transverse tip displacement.

Such actuators constitute a canonical building block of soft robotics, and their dynamics
is genuinely *electro*-mechanical: for large-area DEAs with compliant (carbon-grease or
hydrogel) electrodes, the electrode sheet resistance and the driver output impedance are of
the order of kilo-ohms while the device capacitance reaches microfarads, so the electrical
relaxation time $\tau = R C$ falls in the millisecond range — comparable to the period of
the fundamental bending mode. Neither degree of freedom can be adiabatically eliminated,
and the coupled system retains a memory of the charging dynamics that manifests itself at
the level of the *jerk* of the structure.

### 2.2 Governing equations

The mechanical substrate is an Euler–Bernoulli beam with Kelvin–Voigt viscoelasticity. With
$w(x,t)$ the transverse deflection, $\rho A$ the mass per unit length, $EI$ the bending
stiffness and $\eta I$ the viscoelastic modulus,

The electrodes are deposited with a **linearly tapered coverage**, full width at the clamp
and vanishing at the tip — a standard design choice for unimorph benders that distributes
the actuation authority along the span and shapes the curvature field. The actuation couple
per unit length is therefore

$$
m_b(x) = \bar m\,\Bigl(1 - \frac{x}{L}\Bigr),
\tag{1a}
$$

and the beam equation, with the actuation entering through the distributed couple
$m_b(x)\,Q^2$ (weak form $\int_0^L m_b\,Q^2\,\delta w'\,\mathrm{d}x$), reads

$$
\rho A\,\ddot{w} + \eta I\,\dot{w}'''' + EI\,w'''' = -\,m_b'(x)\,Q^2(t)
= \frac{\bar m}{L}\,Q^2(t),
\tag{1b}
$$

with clamped conditions $w(0) = w'(0) = 0$ and natural conditions
$EI w''(L) = m_b(L)\,Q^2 = 0$, $EI w'''(L) = 0$: the taper converts the actuation into a
*uniform distributed transverse load* proportional to $Q^2$, exciting the entire span. The
quadratic dependence on the electrode charge $Q(t)$ is the signature of the Maxwell stress,
which depends on the square of the electric field. After finite-element discretisation the
actuation becomes a constant, fully populated load vector $b$ times $Q^2$ (see §2.4).

The electrical side is the charging circuit of a *strain-dependent capacitor* in series
with the resistance $R$:

$$
R\,\dot{Q} + \frac{Q}{C(w)} = V(t),
\qquad
C(w) = \frac{C_0}{1-\alpha_c\,\langle w'' \rangle},
\tag{2}
$$

where $C_0$ is the capacitance of the undeformed device and $\langle w'' \rangle$ is the
*coverage-weighted mean curvature*: bending strain modulates the local electrode gap, and
each station contributes in proportion to its electrode area, i.e. to the same taper (1a),

$$
\langle w'' \rangle
= \frac{\int_0^L \bigl(1-x/L\bigr)\, w''\,\mathrm{d}x}{\int_0^L \bigl(1-x/L\bigr)\,\mathrm{d}x}
= \frac{2\,w(L)}{L^2},
\tag{2a}
$$

the closed form following from integration by parts with the clamped boundary conditions
$w(0)=w'(0)=0$ and the vanishing taper at the tip. The distributed capacitive feedback thus
collapses *exactly* to a tip-displacement functional — a convenient property of the linear
taper, not an additional modelling assumption. Writing $c_0 = 1/C_0$, the circuit equation
is *exactly bilinear*,

$$
R\,\dot{Q} + c_0\,Q\,\bigl(1-\alpha_c \langle w''\rangle\bigr) = V(t).
\tag{2'}
$$

Equations (1)–(2′) define a coupled system that is second order in the structure and first
order in the charge, with two nonlinear couplings of clear physical origin: the
*electrostatic forcing* $\propto Q^2$ in (1), and the *capacitive feedback*
$\propto Q\,\langle w''\rangle$ in (2′).

### 2.3 Operating point and voltage drive

The actuator is operated about a DC bias, $V(t) = V_0 + v(t)$ with
$v(t) = v_a \cos\Omega t$ and $v_a \ll V_0$, as is standard practice for DEAs (the bias
linearises the leading actuation path: perturbations of the force
$\propto (Q_0+q)^2$ contain a term $2Q_0 q$ linear in the charge fluctuation). The static
equilibrium $(w_0, Q_0)$ solves a scalar cubic balance between elastic restoring force and
Maxwell stress; the upper solution branch terminates in the classical electrostatic
*pull-in* instability, and we operate safely below it. Expanding (1)–(2′) about
$(w_0, Q_0)$ yields, for the fluctuations $(u, q)$, an exactly polynomial system: the
mechanical equation is quadratic in $q$, the circuit equation bilinear in $(q, u)$, and the
bias additionally produces a *rank-one electrostatic softening* of the structural stiffness
together with a non-symmetric circuit–structure coupling.

### 2.4 Exact reduction to a single third-order field equation

Spatial discretisation with cubic Hermite finite elements ($n$ degrees of freedom collecting
nodal deflections and rotations) gives

$$
M\ddot{u} + D\dot{u} + Ku = \beta\,q + b\,q^2,
\qquad
R\dot q + \hat c\,q = v(t) + c_0 Q_0\, g^{\!\top}u + c_0\, q\, g^{\!\top}u,
\tag{3}
$$

with $\beta = 2Q_0 b$ the linearised coupling vector, $b$ the (fully populated) consistent
load vector assembled from the tapered couple distribution (1a), $g$
the discrete form of the weighted-curvature functional (2a) — by that identity, a
tip-displacement functional, with the sensitivity $\alpha_c$ absorbed in $g$ — and
$\hat c = c_0\bigl(1 - g^{\!\top}x_0\bigr)$ the effective electrical stiffness at the bias
point, $x_0$ being the static displacement vector.

Rather than integrating the mixed-order system (3), we eliminate the charge and obtain a
**single third-order equation in the structural field**. Applying the operator
$R\,\mathrm{d}/\mathrm{d}t + \hat c$ to the mechanical equation produces the charge
contribution $\beta\,(R\dot q + \hat c\, q)$, which the circuit equation replaces exactly by
$\beta\,(v + c_0 Q_0\, g^{\!\top}u + c_0\, q\, g^{\!\top}u)$ — the linear closure. The
differentiation also generates the nonlinear term $2 R\, b\, q\dot q$; its $\dot q$ is
likewise eliminated with the circuit equation, leaving all remaining terms polynomial in
$(q, u, v)$ with no charge rate. The residual charge factors — which now appear only inside
terms of degree two and higher — are then removed through the *charge-proxy identity*

$$
\sigma(u,\dot u,\ddot u) := \ell_0^{\!\top}u + \ell_1^{\!\top}\dot u + \ell_2^{\!\top}\ddot u
\;=\; q + \frac{q^2}{2Q_0},
\qquad
\ell_k = \frac{(K, D, M)_k\, b}{2 Q_0\, b^{\!\top} b},
\tag{4}
$$

which holds *exactly on trajectories of (3)* — it is the projection of the mechanical
equation onto the coupling direction. Inverting the scalar series (4) and truncating at
cubic order yields the model problem in its final form,

$$
\underbrace{R M}_{B_3}\, \dddot{u}
+ \underbrace{(R D + \hat c M)}_{B_2}\, \ddot u
+ \underbrace{(R K + \hat c D)}_{B_1}\, \dot u
+ \underbrace{\bigl(\hat c K - 2 c_0 Q_0^2\, b\,g^{\!\top}\bigr)}_{B_0}\, u
= F(u, \dot u, \ddot u, v),
\tag{5}
$$

$$
F = 2Q_0\, b\, v \;+\; 2 b\, \sigma v \;+\; 4 c_0 Q_0\, b\, \sigma\gamma
\;-\; \hat c\, b\, \sigma^2 \;+\; \frac{\hat c}{Q_0}\, b\, \sigma^3
\;-\; \frac{1}{Q_0}\, b\, \sigma^2 v,
\qquad \gamma = g^{\!\top} u,
\tag{6}
$$

with truncation error $\mathcal{O}(\|\cdot\|^4)$ in the fluctuation amplitude (the cubic
$\sigma^2\gamma$ contributions cancel identically in the closure).

Equation (5) is third order in time *by physics, not by construction*: the leading
coefficient $B_3 = RM$ is the product of structural inertia and electrical resistance, and
the third derivative is the rate of change of the inertial force filtered through the RC
charging dynamics. In the limits $R \to 0$ (ideal driver) or $C_0 \to \infty$, (5) degenerates
to the familiar second-order structural problem; at $\omega_1 \tau = \mathcal{O}(1)$ — the
regime studied here — the charging dynamics shifts and damps the structural spectrum and
introduces a branch of $n$ relaxation eigenvalues clustered near $-\hat c/R$, interleaved
with the $n$ underdamped bending pairs of the cubic eigenvalue problem
$(\lambda^3 B_3 + \lambda^2 B_2 + \lambda B_1 + B_0)\phi = 0$, solved in practice by
companion linearisation to a $3n \times 3n$ generalised eigenvalue problem.

### 2.5 Classification of the nonlinear terms

All nonlinearities in (6) are electromechanical in origin and share the rank-one structure
$b \times (\text{scalar})$, the scalar being a polynomial in the charge proxy $\sigma$, the
mean strain $\gamma$, and the voltage perturbation $v$:

The quadratic terms $4 c_0 Q_0\, b\,\sigma\gamma$ and $-\hat c\, b\,\sigma^2$ descend,
respectively, from the capacitive feedback (strain-modulated charging) and from the Maxwell
forcing $b q^2$; because $\sigma$ mixes displacement, velocity and acceleration, they
populate every quadratic monomial in $(u,\dot u,\ddot u)$ — including
acceleration-squared terms, a feature alien to purely structural models. The cubic term
$(\hat c/Q_0)\, b\, \sigma^3$ is the third-order correction of the charge–proxy inversion
and is the leading source of amplitude dependence (backbone curvature) of the biased
actuator. The terms $2 b\,\sigma v$ and $-(1/Q_0)\, b\, \sigma^2 v$ are *parametric*
forcings: the AC drive multiplies the response, reflecting that voltage modulates the
stiffness of an electrostatically softened structure. Geometric beam nonlinearities can be
appended without altering the structure of (5), but are deliberately omitted so that every
nonlinear effect in the results is attributable to the electromechanical coupling.

### 2.6 Non-autonomous structure and reduction

The only explicit time dependence in (5)–(6) is the scalar drive
$v(t) = v_a\cos\Omega t$. Following the direct parametrisation of invariant manifolds
(DPIM) for non-autonomous systems, the drive is represented as an autonomous *external
system*: a pair of complex variables $r = (r_1, r_2)$ obeying
$\dot r_{1,2} = \pm\,\mathrm{i}\Omega\, r_{1,2}$, $r(0) = (1,1)$, so that
$v = \tfrac{v_a}{2}(r_1 + r_2)$. The full-order problem is thus a system of $n$ coupled
third-order scalar equations — phase-space dimension $3n$ in first-order form — driven by a
two-dimensional linear external system, for a total phase space of dimension $3n+2$: a
high-dimensional autonomous part with minimal non-autonomous content, the configuration for
which DPIM-style reduction is most effective. The reduced model is constructed on the
four-dimensional spectral submanifold associated with the fundamental bending pair
$(\lambda_1, \bar\lambda_1)$ of the cubic pencil and the forcing pair $\pm\mathrm{i}\Omega$,
solved order by order from the cohomological equations in the four reduced coordinates
$(z, \bar z, r_1, r_2)$. Near primary resonance $\Omega \approx \operatorname{Im}\lambda_1$
the reduced dynamics yields the forced response, its backbone, and the phase lag introduced
by the RC dynamics — quantities that the second-order truncation $R \to 0$ misrepresents
precisely when $\omega_1\tau = \mathcal{O}(1)$.

A remark on the alternative formulations is in order. The coupled system (3) can of course
be integrated directly as a first-order system of dimension $2n+1$, and we use it in this
capacity as the ground truth for validation. The third-order formulation is not adopted to
economise on dimension — its companion phase space, $3n$, is in fact larger — but for
structure: it is a single equation of uniform temporal order in the structural unknowns
alone, posed on the unchanged mechanical mesh of dimension $n$, with the electrical
dynamics absorbed into the coefficient matrices $B_0,\dots,B_3$; it exposes the jerk
explicitly, which is relevant in its own right since elastomer fatigue correlates with
jerk; and it furnishes a genuine, physically motivated benchmark for parametrisation
methods formulated directly for ODE systems of order $\nu > 2$, avoiding ad hoc first-order
inflation. The closure is exact to cubic order in the fluctuation amplitude, as verified
numerically against direct integration of (3).
