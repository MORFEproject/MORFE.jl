### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# This Pluto notebook is part of MORFE.jl.
# It demonstrates the MultilinearTerms module with step-by-step hand computations.
#
# ── Running locally ────────────────────────────────────────────────────────────
#   julia> import Pkg; Pkg.add("Pluto")
#   julia> import Pluto; Pluto.run()
#   Then open this file from the Pluto interface.
#
# The notebook activates the notebooks/Project.toml environment, which lists
# MORFE as a local path dependency (resolves to the repo root automatically).

# ╔═╡ bb000001-0001-4001-b001-000000000001
begin
	import Pkg
	Pkg.activate(@__DIR__)
	Pkg.instantiate()
end

# ╔═╡ bb000002-0002-4002-b002-000000000002
begin
	using MORFE.Multiindices: all_multiindices_up_to
	using MORFE.FullOrderModel: NDOrderModel, MultilinearMap
	using MORFE.ParametrisationMethod: create_parametrisation_method_objects
	using MORFE.MultilinearTerms: compute_multilinear_terms, build_multilinear_terms_cache
	using StaticArrays: SVector
	using LinearAlgebra
end

# ╔═╡ bb000003-0003-4003-b003-000000000003
md"""
# MORFE.jl — `MultilinearTerms` Step-by-Step Demo

This notebook walks through `compute_multilinear_terms` by hand, verifying each
result against the Julia implementation.

**What is `compute_multilinear_terms`?**
Given a partial parametrisation $W$ and a target exponent $\alpha$, it computes
the nonlinear right-hand-side contribution $N_\alpha$ to the cohomological equation

$$\mathcal{L}(W_\alpha) = N_\alpha(W_{|\beta| < |\alpha|})$$

$N_\alpha$ is a sum over all multilinear terms $t$ in the model, each evaluated at
combinations of already-computed parametrisation coefficients whose multiindices sum
to $\alpha$.

**System setup.** 2-DOF second-order system with:
- $m = 2$ reduced (internal) coordinates $z_1, z_2$
- $s = 1$ external forcing variable $u$
- `NVAR = 3` variables total
- 5 nonlinear terms covering all symmetry classes

Each section picks one target exponent $\alpha$ and derives $N_\alpha$ by hand,
then confirms the result numerically.
"""

# ╔═╡ bb000004-0004-4004-b004-000000000004
md"""
## 1. System Definition

We consider five multilinear terms covering all three symmetry classes:

| Term | Expression | `multiindex` | $m_e$ | Symmetry class |
|------|-----------|-------------|-------|----------------|
| `term1` | $x \odot \dot{x}$ | `(1,1)` | 0 | `FullyAsymmetric` |
| `term2` | $0.5\,\dot{x} \odot \dot{x}$ | `(0,2)` | 0 | `FullySymmetric` |
| `term3` | $0.5\,x \odot x \odot \dot{x}$ | `(2,1)` | 0 | `GroupwiseSymmetric` |
| `term4` | $2\,x \odot u$ | `(1,0)` | 1 | `FullyAsymmetric` |
| `term5` | $[100,\,200] \cdot u$ | `(0,0)` | 1 | `FullyAsymmetric` |

The `multiindex` field records how many factor slots belong to each derivative order:
`(n_pos, n_vel)` where `n_pos` = position slots and `n_vel` = velocity slots.
A term with $m_e > 0$ has additional **external forcing slots** that consume the
forcing-variable budget of $\alpha$.

The model matrices are all identity for simplicity (they do not affect this demo).
"""

# ╔═╡ bb000005-0005-4005-b005-000000000005
begin
	FOM          = 2     # full-order DOF
	ORD          = 2     # system order (position + velocity)
	ROM          = 2     # reduced coordinates
	FORCING_SIZE = 1     # external forcing variables
	NVAR         = ROM + FORCING_SIZE  # = 3

	term1 = MultilinearMap((res, x, xdot) -> (@. res += x * xdot),          (1, 1))
	term2 = MultilinearMap((res, v1, v2)  -> (@. res += 0.5 * v1 * v2),     (0, 2))
	term3 = MultilinearMap((res, x1, x2, xdot) -> (@. res += 0.5*x1*x2*xdot), (2, 1))
	term4 = MultilinearMap((res, x, r)    -> (@. res += 2.0 * x * r),       (1, 0), 1)
	term5 = MultilinearMap((res, r)       -> (@. res += [100.0, 200.0] * r), (0, 0), 1)

	Id    = Matrix{Float64}(I, FOM, FOM)
	model = NDOrderModel((Id, Id, Id), (term1, term2, term3, term4, term5))
end

# ╔═╡ bb000006-0006-4006-b006-000000000006
md"""
## 2. Multiindex Set

We work with all monomials in $(z_1, z_2, u)$ up to total degree 3.
The graded-lex ordering assigns the following indices:

| Index | Exponent $(z_1, z_2, u)$ | Shorthand |
|-------|--------------------------|-----------|
| 1 | (0,0,0) | constant |
| 2 | (1,0,0) | $z_1$ |
| 3 | (0,1,0) | $z_2$ |
| 4 | (0,0,1) | $u$ |
| 5 | (2,0,0) | $z_1^2$ |
| 6 | (1,1,0) | $z_1 z_2$ |
| 7 | (1,0,1) | $z_1 u$ |
| 8 | (0,2,0) | $z_2^2$ |
| 9 | (0,1,1) | $z_2 u$ |
| 10 | (0,0,2) | $u^2$ |
| 11 | (3,0,0) | $z_1^3$ |
| ⋮  | ⋮ | ⋮ |
"""

# ╔═╡ bb000007-0007-4007-b007-000000000007
begin
	mset = all_multiindices_up_to(NVAR, 3)
	println("Multiindex set: $(length(mset.exponents)) monomials in $NVAR variables, degree ≤ 3")
	println("\nFirst 12 exponents:")
	for (i, e) in enumerate(mset.exponents[1:min(12, end)])
		println("  idx $i → $e")
	end
end

# ╔═╡ bb000008-0008-4008-b008-000000000008
md"""
## 3. Parametrisation Coefficients

The parametrisation $W : (z_1, z_2, u) \mapsto (x, \dot{x})$ is stored as a
polynomial whose coefficient at multiindex $\alpha$ is a matrix
$W_\alpha \in \mathbb{C}^{n \times \text{ORD}}$:

- `W.poly.coefficients[:, 1, idx]` — position part $W_\alpha^{(x)}$
- `W.poly.coefficients[:, 2, idx]` — velocity part $W_\alpha^{(\dot x)}$

We populate linear and selected quadratic/cubic coefficients:

**Linear part** (standard modal expansion):
$$W_{(1,0,0)} = \begin{pmatrix}\varphi_1 \\ \lambda_1\varphi_1\end{pmatrix}, \quad
  W_{(0,1,0)} = \begin{pmatrix}\varphi_2 \\ \lambda_2\varphi_2\end{pmatrix}, \quad
  W_{(0,0,1)} = \begin{pmatrix}b \\ \mathrm{i}\,b\end{pmatrix}$$

with $\varphi_1 = [2,3]^\top$, $\varphi_2 = [4,5]^\top$, $b = [6,7]^\top$,
$\lambda_1 = -0.1 + 10\mathrm{i}$, $\lambda_2 = \bar\lambda_1$.

**Quadratic part** (arbitrary values for testing):

| Exponent | Position | Velocity |
|----------|----------|----------|
| $(2,0,0)$ | $0.1\,\varphi_1$ | $b + \varphi_2$ |
| $(1,1,0)$ | $0.05\,\varphi_1$ | $0.05\,\varphi_2$ |
| $(1,0,1)$ | $0.1\,\varphi_1$ | $-0.3\,\varphi_2$ |
| $(0,2,0)$ | $b + \varphi_2$ | $0.2\,\varphi_2$ |

**Cubic part:** $W_{(3,0,0)} = (500\,\varphi_1,\; -500\mathrm{i}\,\varphi_2)$.
"""

# ╔═╡ bb000009-0009-4009-b009-000000000009
begin
	W, _ = create_parametrisation_method_objects(mset, ORD, FOM, ROM, FORCING_SIZE, ComplexF64)

	λ₁ = -0.1 + 10.0im
	λ₂ = conj(λ₁)
	φ₁ = ComplexF64[2.0, 3.0]
	φ₂ = ComplexF64[4.0, 5.0]
	b  = ComplexF64[6.0, 7.0]

	# Linear
	W.poly.coefficients[:, 1, 2] = φ₁;          W.poly.coefficients[:, 2, 2] = λ₁ * φ₁
	W.poly.coefficients[:, 1, 3] = φ₂;          W.poly.coefficients[:, 2, 3] = λ₂ * φ₂
	W.poly.coefficients[:, 1, 4] = b;            W.poly.coefficients[:, 2, 4] = 1.0im * b

	# Quadratic
	W.poly.coefficients[:, 1, 5] = 0.1  * φ₁;   W.poly.coefficients[:, 2, 5] = b .+ φ₂
	W.poly.coefficients[:, 1, 6] = 0.05 * φ₁;   W.poly.coefficients[:, 2, 6] = 0.05 * φ₂
	W.poly.coefficients[:, 1, 7] = 0.1  * φ₁;   W.poly.coefficients[:, 2, 7] = -0.3 * φ₂
	W.poly.coefficients[:, 1, 8] = b .+ φ₂;     W.poly.coefficients[:, 2, 8] = 0.2  * φ₂

	# Cubic
	W.poly.coefficients[:, 1, 11] = 500.0 * φ₁; W.poly.coefficients[:, 2, 11] = -500im * φ₂

	println("Parametrisation populated.  Non-zero entries:")
	for idx in [2,3,4,5,6,7,8,11]
		p = W.poly.coefficients[:, 1, idx]
		v = W.poly.coefficients[:, 2, idx]
		println("  $(mset.exponents[idx])  pos=$p  vel=$v")
	end
end

# ╔═╡ bb000010-0010-4010-b010-000000000010
md"""
## 4. Target $\alpha = (1,0,0)$: Linear in $z_1$

**Claim:** $N_{(1,0,0)} = 0$.

For any term of degree $d \geq 2$ to contribute at $\alpha = (1,0,0)$, we would need
to factorise $(1,0,0)$ into $d \geq 2$ multiindices from the set, all of which sum to
$(1,0,0)$. The only way is to have one factor at $(1,0,0)$ and the rest at $(0,0,0)$.
But $W_{(0,0,0)} = 0$ (the parametrisation is centred at the origin), so every such
factorisation contributes zero.

The same argument holds for all degree-1 targets: nonlinear terms (degree $\geq 2$)
cannot build a degree-1 exponent from lower-order coefficients without using the
zero constant term.
"""

# ╔═╡ bb000011-0011-4011-b011-000000000011
begin
	exp100  = SVector(1, 0, 0)
	r100    = compute_multilinear_terms(model, exp100, W)
	manual100 = ComplexF64[0.0, 0.0]
	println("exp = $exp100")
	println("  computed : $r100")
	println("  manual   : $manual100")
	println("  ‖error‖  : $(norm(r100 - manual100))")
end

# ╔═╡ bb000012-0012-4012-b012-000000000012
md"""
## 5. Target $\alpha = (0,0,1)$: Linear in $u$

**Contributing terms:**

Only `term5` can contribute at degree 1 in the forcing variable $u$, because it has
$m_e = 1$ external forcing slot and zero internal slots.

**Forcing split for `term5`** ($m_e = 1$, $\alpha_u = 1$):

The one external slot is assigned to $u$ (the only forcing variable):
- $f_\text{idx} = (1,)$, $f_\text{count} = 1$, $\hat\beta = (0,0,1)$
- $\text{rem} = (0,0,1) - (0,0,1) = (0,0,0)$

Since `term5` has zero internal slots ($d_\text{int} = 0$), the only valid
factorisation of $\text{rem} = (0,0,0)$ is the empty tuple. The call is:

$$\text{term5.f!}(\text{temp},\; e_1) \quad\text{where } e_1 = 1 \in \mathbb{R}$$

$$\Rightarrow \text{temp} = [100,\,200] \cdot 1 = [100,\,200]$$

**`term1`–`term4`:** require at least one internal slot with $|\alpha| \geq 1$.
At $\alpha = (0,0,1)$ the internal budget after the forcing split is zero, so no
internal factorisation is possible for $d_\text{int} \geq 1$. No contribution.

$$\boxed{N_{(0,0,1)} = [100,\,200]^\top}$$
"""

# ╔═╡ bb000013-0013-4013-b013-000000000013
begin
	exp001    = SVector(0, 0, 1)
	r001      = compute_multilinear_terms(model, exp001, W)
	manual001 = ComplexF64[100.0, 200.0]
	println("exp = $exp001")
	println("  computed : $r001")
	println("  manual   : $manual001")
	println("  ‖error‖  : $(norm(r001 - manual001))")
end

# ╔═╡ bb000014-0014-4014-b014-000000000014
md"""
## 6. Target $\alpha = (2,0,0)$: Quadratic in $z_1$

$\alpha = (2,0,0)$, $\deg_\text{max} = 2$.  Terms with $\deg > 2$ are skipped.

**Degree filter:** `term3` has $\deg = 3 > 2$ → **skipped**.
`term4`, `term5`: $m_e = 1$, need $\alpha_u \geq 1$, but $\alpha_u = 0$ → **skipped**.

---

### `term1`: $x \odot \dot{x}$, `multiindex = (1,1)`, `FullyAsymmetric`

Slot assignments: slot 1 → position ($k=1$), slot 2 → velocity ($k=2$).

Enumerate all *ordered* pairs $(\alpha_1, \alpha_2)$ with $\alpha_1 + \alpha_2 = (2,0,0)$:

| $\alpha_1$ (pos) | $\alpha_2$ (vel) | $W_{\alpha_1}^{(x)}$ | $W_{\alpha_2}^{(\dot x)}$ | Contribution |
|-------------------|-------------------|----------------------|--------------------------|-------------|
| $(1,0,0)$ | $(1,0,0)$ | $\varphi_1$ | $\lambda_1\varphi_1$ | $\varphi_1 \odot \lambda_1\varphi_1 = \lambda_1\,\varphi_1 \odot \varphi_1$ |
| $(2,0,0)$ | $(0,0,0)$ | $0.1\varphi_1$ | **0** | 0 |
| $(0,0,0)$ | $(2,0,0)$ | **0** | $b+\varphi_2$ | 0 |

$$\text{term1 contribution} = \lambda_1\,\varphi_1 \odot \varphi_1$$

---

### `term2`: $0.5\,\dot{x} \odot \dot{x}$, `multiindex = (0,2)`, `FullySymmetric`

Both slots draw from velocity.  Enumerate *unordered multisets* $\{\alpha_1, \alpha_2\}$
with $\alpha_1 + \alpha_2 = (2,0,0)$:

| Multiset | $\text{sym\_count}$ | $W_{\alpha_1}^{(\dot x)}$ | $W_{\alpha_2}^{(\dot x)}$ | Contribution |
|----------|---------------------|--------------------------|--------------------------|-------------|
| $\{(1,0,0),(1,0,0)\}$ | $1\;(= 2!/2!)$ | $\lambda_1\varphi_1$ | $\lambda_1\varphi_1$ | $1 \times 0.5\,\lambda_1^2\,\varphi_1 \odot \varphi_1$ |
| $\{(2,0,0),(0,0,0)\}$ | $2\;(= 2!/1!1!)$ | $b+\varphi_2$ | **0** | 0 |

$$\text{term2 contribution} = 0.5\,\lambda_1^2\,\varphi_1 \odot \varphi_1$$

---

### Total

$$\boxed{N_{(2,0,0)} = \bigl(\lambda_1 + 0.5\,\lambda_1^2\bigr)\,\varphi_1 \odot \varphi_1}$$

With $\lambda_1 = -0.1 + 10\mathrm{i}$:
$\lambda_1^2 = (-0.1)^2 - 100 + 2(-0.1)(10)\mathrm{i} = -99.99 - 2\mathrm{i}$,
so $\lambda_1 + 0.5\lambda_1^2 = -50.095 + 9\mathrm{i}$.

For $\varphi_1 \odot \varphi_1 = [4,\,9]^\top$:

$$N_{(2,0,0)} \approx [-200.38 + 36\mathrm{i},\; -450.855 + 81\mathrm{i}]^\top$$
"""

# ╔═╡ bb000015-0015-4015-b015-000000000015
begin
	exp200 = SVector(2, 0, 0)
	r200   = compute_multilinear_terms(model, exp200, W)

	manual200_term1 = λ₁ .* φ₁ .* φ₁
	manual200_term2 = 0.5 * λ₁^2 .* φ₁ .* φ₁
	manual200       = manual200_term1 + manual200_term2

	println("exp = $exp200")
	println("  computed         : $r200")
	println("  manual term1     : $manual200_term1")
	println("  manual term2     : $manual200_term2")
	println("  manual total     : $manual200")
	println("  ‖error‖          : $(norm(r200 - manual200))")
end

# ╔═╡ bb000016-0016-4016-b016-000000000016
md"""
## 7. Target $\alpha = (1,0,1)$: Mixed $z_1 \cdot u$

$\alpha = (1,0,1)$, $\deg_\text{max} = 2$.
**Degree filter:** `term3` ($\deg=3$) → **skipped**.

The forcing slice is $\alpha_u = 1$.

---

### `term1`: $x \odot \dot{x}$, `FullyAsymmetric`, $m_e = 0$

No forcing split ($m_e = 0$, rem $= (1,0,1)$).

| $\alpha_1$ (pos) | $\alpha_2$ (vel) | Contribution |
|-------------------|-------------------|-------------|
| $(1,0,0)$ | $(0,0,1)$ | $\varphi_1 \odot \mathrm{i}\,b$ |
| $(0,0,1)$ | $(1,0,0)$ | $b \odot \lambda_1\varphi_1$ |
| $(1,0,1)$ | $(0,0,0)$ | $0.1\varphi_1 \odot 0 = 0$ |
| $(0,0,0)$ | $(1,0,1)$ | $0 \odot (-0.3\varphi_2) = 0$ |

$$\text{term1} = \mathrm{i}\,\varphi_1 \odot b + \lambda_1\,b \odot \varphi_1 = (\mathrm{i} + \lambda_1)\,\varphi_1 \odot b$$

---

### `term2`: $0.5\,\dot{x} \odot \dot{x}$, `FullySymmetric`, $m_e = 0$

Unordered pairs summing to $(1,0,1)$ via velocity:

| Multiset | $\text{sym\_count}$ | Contribution |
|----------|---------------------|-------------|
| $\{(1,0,0),(0,0,1)\}$ | $2$ | $2 \times 0.5\,(\lambda_1\varphi_1) \odot (\mathrm{i}\,b) = \mathrm{i}\lambda_1\,\varphi_1 \odot b$ |

$$\text{term2} = \mathrm{i}\lambda_1\,\varphi_1 \odot b$$

---

### `term4`: $2\,x \odot u$, `FullyAsymmetric`, $m_e = 1$

Forcing split: assign the 1 external slot to $u$ ($f_\text{idx} = (1,)$, $f_\text{count} = 1$):
- $\hat\beta = (0,0,1)$, $\text{rem} = (1,0,1) - (0,0,1) = (1,0,0)$
- $\text{args\_ext} = (e_1 = 1,)$

One internal $x$-slot must draw $\alpha_1 = (1,0,0)$, giving $W_{(1,0,0)}^{(x)} = \varphi_1$:

$$\text{term4.f!}(\text{temp},\, \varphi_1,\, 1) \;\Rightarrow\; \text{temp} = 2\varphi_1$$

$$\text{term4} = 1 \times 2\varphi_1 = 2\varphi_1$$

---

### `term5`: $[100,200] \cdot u$, $m_e = 1$, $d_\text{int} = 0$

After the forcing split rem $= (1,0,0) \neq (0,0,0)$, so the empty factorisation
(the only option for $d_\text{int} = 0$) does not match. **No contribution.**

---

### Total

$$\boxed{N_{(1,0,1)} = (\mathrm{i} + \lambda_1 + \mathrm{i}\lambda_1)\,\varphi_1 \odot b + 2\varphi_1}$$
"""

# ╔═╡ bb000017-0017-4017-b017-000000000017
begin
	exp101 = SVector(1, 0, 1)
	r101   = compute_multilinear_terms(model, exp101, W)

	manual101_term1 = (1.0im + λ₁) .* φ₁ .* b
	manual101_term2 = (0.5 * 1.0im * λ₁ .* φ₁ .* b) * 2
	manual101_term4 = 2.0 .* φ₁
	manual101       = manual101_term1 + manual101_term2 + manual101_term4

	println("exp = $exp101")
	println("  computed         : $r101")
	println("  manual term1     : $manual101_term1")
	println("  manual term2     : $manual101_term2")
	println("  manual term4     : $manual101_term4")
	println("  manual total     : $manual101")
	println("  ‖error‖          : $(norm(r101 - manual101))")
end

# ╔═╡ bb000018-0018-4018-b018-000000000018
md"""
## 8. Target $\alpha = (3,0,1)$: Cubic in $z_1$, Linear in $u$

$\alpha = (3,0,1)$, $\deg_\text{max} = 4$.  All five terms are within range.

The forcing slice is $\alpha_u = 1$.

---

### `term1`: $x \odot \dot{x}$, `FullyAsymmetric`, $m_e = 0$, $\text{deg} = 2$

No forcing split. Enumerate ordered pairs $(\alpha_1, \alpha_2)$ summing to $(3,0,1)$
with $W_{\alpha_i} \neq 0$:

| $\alpha_1$ (pos) | $\alpha_2$ (vel) | Contribution |
|------------------|-----------------|-------------|
| $(3,0,0)$ | $(0,0,1)$ | $(500\varphi_1) \odot (\mathrm{i}\,b)$ |
| $(0,0,1)$ | $(3,0,0)$ | $b \odot (-500\mathrm{i}\,\varphi_2)$ |
| $(2,0,0)$ | $(1,0,1)$ | $(0.1\varphi_1) \odot (-0.3\varphi_2)$ |
| $(1,0,1)$ | $(2,0,0)$ | $(0.1\varphi_1) \odot (b + \varphi_2)$ |

All other pairs involve a zero coefficient (e.g. $W_{(2,0,1)} = 0$).

---

### `term2`: $0.5\,\dot{x} \odot \dot{x}$, `FullySymmetric`, $m_e = 0$, $\text{deg} = 2$

Unordered pairs $\{\alpha_1, \alpha_2\}$ summing to $(3,0,1)$ via velocity with non-zero
coefficients:

| Multiset | sym_count | Contribution |
|----------|-----------|-------------|
| $\{(3,0,0),(0,0,1)\}$ | $2$ | $2 \times 0.5\,(-500\mathrm{i}\,\varphi_2) \odot (\mathrm{i}\,b)$ |
| $\{(2,0,0),(1,0,1)\}$ | $2$ | $2 \times 0.5\,(b+\varphi_2) \odot (-0.3\varphi_2)$ |

---

### `term3`: $0.5\,x \odot x \odot \dot{x}$, `GroupwiseSymmetric`, $m_e = 0$, $\text{deg} = 3$

Two $x$-slots (symmetric, position) and one $\dot{x}$-slot (velocity).
Enumerate over the $\dot{x}$-slot exponent $\alpha_3$ and symmetric pairs $\{\alpha_1,\alpha_2\}$
for the $x$-slots such that $\alpha_1 + \alpha_2 + \alpha_3 = (3,0,1)$:

| $\alpha_3$ (vel) | $\{\alpha_1, \alpha_2\}$ (pos) | count | Contribution |
|-----------------|-------------------------------|-------|-------------|
| $(0,0,1)$ | $\{(2,0,0),(1,0,0)\}$ | $2$ | $2 \times 0.5\,(0.1\varphi_1)\odot\varphi_1\odot(\mathrm{i}\,b)$ |
| $(1,0,0)$ | $\{(2,0,0),(0,0,1)\}$ | $2$ | $2 \times 0.5\,(0.1\varphi_1)\odot b\odot(\lambda_1\varphi_1)$ |
| $(2,0,0)$ | $\{(1,0,0),(0,0,1)\}$ | $2$ | $2 \times 0.5\,\varphi_1\odot b\odot(b+\varphi_2)$ |
| $(1,0,1)$ | $\{(1,0,0),(1,0,0)\}$ | $1$ | $1 \times 0.5\,\varphi_1\odot\varphi_1\odot(-0.3\varphi_2)$ |
| $(1,0,0)$ | $\{(1,0,0),(1,0,1)\}$ | $2$ | $2 \times 0.5\,\varphi_1\odot(0.1\varphi_1)\odot(\lambda_1\varphi_1)$ |

The count for each groupwise-symmetric factorisation is the within-group multinomial:
- Pair with two *distinct* elements: $2!/1!1! = 2$.
- Pair with two *identical* elements: $2!/2! = 1$.

---

### `term4`: $2\,x \odot u$, `FullyAsymmetric`, $m_e = 1$, $\text{deg} = 2$

Forcing split: $f_\text{idx} = (1,)$, $\hat\beta = (0,0,1)$, rem $= (3,0,0)$.

The single $x$-slot must be $\alpha_1 = (3,0,0)$ (the only non-zero entry):

$$\text{term4.f!}(\text{temp},\, 500\varphi_1,\, 1) \;\Rightarrow\; \text{term4} = 2 \times 500\varphi_1 = 1000\varphi_1$$

---

### `term5`: no contribution

After the forcing split rem $= (3,0,0) \neq (0,0,0)$; $d_\text{int} = 0$ requires
rem $= (0,0,0)$. **Skipped.**

---

### Total

$$\boxed{N_{(3,0,1)} = \text{term1} + \text{term2} + \text{term3} + \text{term4}}$$

(see code cell below for numerical values)
"""

# ╔═╡ bb000019-0019-4019-b019-000000000019
begin
	exp301 = SVector(3, 0, 1)
	r301   = compute_multilinear_terms(model, exp301, W)

	# term1: four ordered pairs
	t1_a = (500.0 * φ₁) .* (1.0im * b)
	t1_b = b .* (-500im * φ₂)
	t1_c = (0.1 * φ₁) .* (-0.3 * φ₂)
	t1_d = (0.1 * φ₁) .* (b .+ φ₂)
	manual301_term1 = t1_a + t1_b + t1_c + t1_d

	# term2: two unordered pairs, each with sym_count=2
	t2_a = 0.5 * (-500im * φ₂) .* (1.0im * b) * 2
	t2_b = 0.5 * (b .+ φ₂)     .* (-0.3 * φ₂) * 2
	manual301_term2 = t2_a + t2_b

	# term3: five groupwise-symmetric factorisations
	t3_a = 0.5 * (0.1 * φ₁) .* φ₁ .* (1.0im * b) * 2
	t3_b = 0.5 * (0.1 * φ₁) .* b  .* (λ₁ * φ₁)  * 2
	t3_c = 0.5 * φ₁          .* b  .* (b .+ φ₂)   * 2
	t3_d = 0.5 * φ₁          .* φ₁ .* (-0.3 * φ₂) * 1
	t3_e = 0.5 * φ₁ .* (0.1 * φ₁)  .* (λ₁ * φ₁)   * 2
	manual301_term3 = t3_a + t3_b + t3_c + t3_d + t3_e

	# term4: single x-slot at (3,0,0)
	manual301_term4 = 2.0 * (500.0 * φ₁)

	manual301 = manual301_term1 + manual301_term2 + manual301_term3 + manual301_term4

	println("exp = $exp301")
	println("  computed     : $r301")
	println("  manual term1 : $manual301_term1")
	println("  manual term2 : $manual301_term2")
	println("  manual term3 : $manual301_term3")
	println("  manual term4 : $manual301_term4")
	println("  manual total : $manual301")
	println("  ‖error‖      : $(norm(r301 - manual301))")
end

# ╔═╡ bb000020-0020-4020-b020-000000000020
md"""
## 9. Cached Path Verification

`build_multilinear_terms_cache` pre-computes all factorisation structures for the
full multiindex set.  The cached path `compute_multilinear_terms(model, idx, W, cache)`
accepts an integer index into the multiindex set rather than an `SVector`.

The two paths must produce **bit-identical** results for every exponent.
"""

# ╔═╡ bb000021-0021-4021-b021-000000000021
begin
	cache     = build_multilinear_terms_cache(model, W)
	mset_exps = mset.exponents

	all_match = true
	for (label, sv) in [
		("(1,0,0)", SVector(1,0,0)),
		("(0,0,1)", SVector(0,0,1)),
		("(2,0,0)", SVector(2,0,0)),
		("(1,0,1)", SVector(1,0,1)),
		("(3,0,1)", SVector(3,0,1)),
	]
		idx      = findfirst(==(sv), mset_exps)
		r_direct = compute_multilinear_terms(model, sv,  W)
		r_cached = compute_multilinear_terms(model, idx, W, cache)
		err      = norm(r_cached - r_direct)
		status   = err == 0 ? "✓" : "✗"
		println("$status  exp $label : ‖cached − direct‖ = $err")
		all_match = all_match && (err == 0)
	end
	println()
	all_match ? println("All cached results match direct computation.") :
	            println("MISMATCH detected — check above.")
end

# ╔═╡ bb000022-0022-4022-b022-000000000022
md"""
## Summary

| $\alpha$ | $N_\alpha$ formula | Key features exercised |
|----------|-------------------|------------------------|
| $(1,0,0)$ | $0$ | zero contribution from degree-$\geq 2$ terms at degree-1 target |
| $(0,0,1)$ | $[100,200]^\top$ | $m_e=1$ forcing constant term, $d_\text{int}=0$ |
| $(2,0,0)$ | $(\lambda_1+0.5\lambda_1^2)\,\varphi_1\odot\varphi_1$ | `FullyAsymmetric` + `FullySymmetric` |
| $(1,0,1)$ | $(\mathrm{i}+\lambda_1+\mathrm{i}\lambda_1)\,\varphi_1\odot b + 2\varphi_1$ | forcing split in `term4`, mixed internal/external |
| $(3,0,1)$ | sum of 4 terms | `GroupwiseSymmetric`, all five terms contribute |

**Design principles confirmed:**

- **Degree filter:** terms with $\deg > |\alpha|$ are skipped entirely.
- **Forcing split:** only terms with $m_e \leq \alpha_u$ can contribute to a target
  exponent with forcing component $\alpha_u$.
- **Symmetry counting:** `FullySymmetric` and `GroupwiseSymmetric` routes correctly
  weight unordered multisets by their multinomial coefficients.
- **Cached path:** produces bit-identical results to the direct path.
"""

# ╔═╡ Cell order:
# ╟─bb000003-0003-4003-b003-000000000003
# ╟─bb000004-0004-4004-b004-000000000004
# ╠═bb000005-0005-4005-b005-000000000005
# ╟─bb000006-0006-4006-b006-000000000006
# ╠═bb000007-0007-4007-b007-000000000007
# ╟─bb000008-0008-4008-b008-000000000008
# ╠═bb000009-0009-4009-b009-000000000009
# ╟─bb000010-0010-4010-b010-000000000010
# ╠═bb000011-0011-4011-b011-000000000011
# ╟─bb000012-0012-4012-b012-000000000012
# ╠═bb000013-0013-4013-b013-000000000013
# ╟─bb000014-0014-4014-b014-000000000014
# ╠═bb000015-0015-4015-b015-000000000015
# ╟─bb000016-0016-4016-b016-000000000016
# ╠═bb000017-0017-4017-b017-000000000017
# ╟─bb000018-0018-4018-b018-000000000018
# ╠═bb000019-0019-4019-b019-000000000019
# ╟─bb000020-0020-4020-b020-000000000020
# ╠═bb000021-0021-4021-b021-000000000021
# ╟─bb000022-0022-4022-b022-000000000022
# ╠═bb000001-0001-4001-b001-000000000001
# ╠═bb000002-0002-4002-b002-000000000002
