# Pressure Lift Polynomial L(z)

## Physical setting

We study incompressible flow past a cylinder in 2-D.
The main flow direction is $x$; lift acts in the $y$-direction.
The cylinder surface $\Gamma_\text{cyl}$ is the no-slip wall.
We use a P2/P1 Taylor-Hood finite element discretisation
(quadratic velocity, linear pressure) on a triangular mesh.

The full-order state is the DOF vector

$$\mathbf{u} \in \mathbb{R}^{N}, \quad N = N_u + N_p,$$

where the first $N_u$ components are velocity DOFs and the last $N_p$ are pressure DOFs
(blocked ordering, as set by the Ferrite `DofHandler`).

---

## Full-state reconstruction from ROM coordinates

The DPIM parametrisation produces:

- A **base flow** $\mathbf{s}_0 \in \mathbb{R}^N$ (steady Navier–Stokes solution).
- A **parametrisation** $\mathbf{W} : \mathbb{C}^{n_\text{var}} \to \mathbb{C}^{n_\text{free}}$,
  a polynomial map from ROM coordinates $\mathbf{z}$ to the free-DOF subspace.

The free-DOF index set $\mathcal{F} \subset \{1,\dots,N\}$ excludes velocity DOFs on the
cylinder surface and channel walls (Dirichlet no-slip conditions);
the inlet boundary is *not* excluded (perturbations are allowed there).

The physical state at reduced coordinate $\mathbf{z}$ is reconstructed as

$$\mathbf{u}(\mathbf{z}) = \mathbf{s}_0 + \mathbf{E}\,\mathbf{W}(\mathbf{z}),$$

where $\mathbf{E} \in \mathbb{R}^{N \times n_\text{free}}$ is the Boolean extension operator
that inserts the free-DOF values into the full DOF vector (zero elsewhere).

In code:

```julia
u_full = s0_full .+ expand(evaluate(W1, z), free_dpim, ndofs_total)
```

---

## Lift as a linear functional of the state

The aerodynamic force on the cylinder is

$$\mathbf{F} = \int_{\Gamma_\text{cyl}} \boldsymbol{\sigma} \cdot \mathbf{n} \, \mathrm{d}\Gamma,$$

where $\mathbf{n}$ is the **outward normal from the fluid domain** at the cylinder surface
(pointing into the cylinder), and $\boldsymbol{\sigma} = -p\mathbf{I} + \nu(\nabla\mathbf{u} + \nabla\mathbf{u}^\top)$
is the Cauchy stress tensor.

The **pressure contribution** to lift (the $y$-component) is

$$F_L^\text{pres} = \int_{\Gamma_\text{cyl}} (-p)\, n_y \, \mathrm{d}\Gamma.$$

Expanding the pressure in P1 basis functions $\{\psi_i^p\}$,

$$p(\mathbf{x}) = \sum_{i \in \mathcal{P}} p_i\, \psi_i^p(\mathbf{x}),$$

gives

$$F_L^\text{pres} = \sum_{i \in \mathcal{P}} \underbrace{\left(-\int_{\Gamma_\text{cyl}} n_y(\mathbf{x})\,\psi_i^p(\mathbf{x})\,\mathrm{d}\Gamma\right)}_{l_i} p_i
                 = \mathbf{l}^\top \mathbf{u},$$

where the **lift weight vector** $\mathbf{l} \in \mathbb{R}^N$ has

$$l_i = \begin{cases}
  -\displaystyle\int_{\Gamma_\text{cyl}} n_y\,\psi_i^p\,\mathrm{d}\Gamma & \text{DOF } i \text{ is pressure and } \psi_i^p|_{\Gamma_\text{cyl}} \neq 0, \\
  0 & \text{otherwise.}
\end{cases}$$

Velocity DOFs contribute zero (only pressure enters the formula).
Pressure DOFs away from the cylinder also contribute zero because their basis functions
vanish on $\Gamma_\text{cyl}$.

The weight vector is assembled by Gaussian quadrature over every boundary edge
$(c, f) \in \mathcal{F}_\text{cyl}$ (cell $c$, local facet $f$):

```julia
for (cell_idx, local_facet_idx) in getfacetset(fom.grid, "Cylinder")
    cell = CellCache(fom.dh); reinit!(cell, cell_idx)
    gdofs = celldofs(cell)
    reinit!(fv_pres, cell, local_facet_idx)
    for q in 1:getnquadpoints(fv_pres)
        dΓ  = getdetJdV(fv_pres, q)
        n_y = getnormal(fv_pres, q)[2]
        for (i, li) in enumerate(fom.dof_range_p)   # i = 1,2,3; li = 13,14,15
            l[gdofs[li]] += (-n_y) * shape_value(fv_pres, q, i) * dΓ
        end
    end
end
```

---

## Polynomial structure of $L(\mathbf{z})$

Substituting the DPIM expansion into $F_L^\text{pres} = \mathbf{l}^\top \mathbf{u}(\mathbf{z})$:

$$F_L^\text{pres}(\mathbf{z})
  = \mathbf{l}^\top \mathbf{s}_0
    + \mathbf{l}^\top \mathbf{E}\, \mathbf{W}(\mathbf{z})
  = L_0 + \mathbf{l}_\mathcal{F}^\top \mathbf{W}(\mathbf{z}),$$

where $\mathbf{l}_\mathcal{F} = \mathbf{l}[\mathcal{F}]$ is the restriction of $\mathbf{l}$ to
free DOFs and $L_0 = \mathbf{l}_\mathcal{F}^\top \mathbf{s}_{0,\mathcal{F}}$ is the lift of the
base flow.

Writing $\mathbf{W}(\mathbf{z}) = \mathbf{W}_1\,\mathbf{m}(\mathbf{z})$ (coefficient matrix
times monomial vector):

$$\boxed{F_L^\text{pres}(\mathbf{z}) = L_0 + \mathbf{L}_\text{coeffs}^\top \mathbf{m}(\mathbf{z}),
\qquad \mathbf{L}_\text{coeffs} = \mathbf{W}_1^\top \mathbf{l}_\mathcal{F} \in \mathbb{C}^{L}.}$$

This is a scalar polynomial of the same degree and monomial structure as $\mathbf{W}$.
**No additional FEM solve is needed** — $\mathbf{L}_\text{coeffs}$ is a single matrix–vector
product.

In code:

```julia
L_coeffs = vec(W1_coeffs' * l_free)       # (L,) ComplexF64
L0       = dot(l_free, real.(s₀_full[fom.free_dpim]))
```

The imaginary parts of `L_coeffs` cancel when the state is conjugate-symmetric
($z_2 = \bar{z}_1$), so `real(dot(L_coeffs, m(z)))` is the physical lift.

---

## Evaluation

Load the saved polynomial and evaluate at any ROM coordinate `z`:

```julia
lr = deserialize(joinpath(DATA_DIR, "lift_polynomial.jls"))  # (; L0, L_coeffs, mset)

using MORFE
L_poly = MORFE.DensePolynomial(reshape(lr.L_coeffs, 1, :), lr.mset)

function lift_at(z, lr, L_poly)
    return lr.L0 + only(real.(MORFE.evaluate(L_poly, z)))
end

# Periodic orbit at phase θ, amplitude A:
z1 = A * cis(θ)
z  = SVector(z1, conj(z1), 0.0 + 0im)
Fl_pres = lift_at(z, lr, L_poly)
Cl_pres = -2.0 * Fl_pres / (U_MEAN^2 * _CYL_D)   # pressure lift coefficient
```

---

## Sign convention and normalisation

| Quantity | Definition | Code |
|---|---|---|
| $\mathbf{n}$ | Outward normal from fluid at $\Gamma_\text{cyl}$ | `getnormal(fv, q)` |
| $l_i$ | $-\int n_y \psi_i^p \,\mathrm{d}\Gamma$ | `l[gdofs[li]]` |
| $F_L^\text{pres}$ | $\mathbf{l}^\top \mathbf{u}$ | `L0 + real(dot(L_coeffs, m(z)))` |
| $C_L^\text{pres}$ | $-2 F_L^\text{pres} / (U^2 D)$ | consistent with `compute_drag_lift` |

The factor $-2$ in $C_L$ arises because `compute_drag_lift` in `steady_state.jl`
accumulates `Fl += (-p_q) * n[2] * dΓ` for the pressure part and returns `Cl = -2*Fl/ref`,
so $F_L^\text{pres} = \mathbf{l}^\top \mathbf{u}$ and $C_L = -2 F_L^\text{pres} / (U^2 D)$
are fully consistent.

---

## Limitations

This polynomial captures only the **pressure** contribution to lift.
The viscous contribution

$$F_L^\text{visc} = \int_{\Gamma_\text{cyl}} \nu\bigl(\nabla\mathbf{u}+\nabla\mathbf{u}^\top\bigr)\mathbf{n} \cdot \mathbf{e}_y \, \mathrm{d}\Gamma$$

is also linear in the DOF vector (hence also polynomial in $\mathbf{z}$), but requires
assembling a separate weight vector using `FacetValues` for the velocity field with
`Lagrange{RefTriangle, 2}()^2`. At moderate–high Reynolds numbers the pressure term
dominates and the viscous contribution is small.
