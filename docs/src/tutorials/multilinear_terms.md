# Tutorial: `MultilinearTerms` Step-by-Step

This tutorial walks through [`compute_multilinear_terms`](@ref) by hand, verifying
each result against the Julia implementation.

**What is `compute_multilinear_terms`?**
Given a partial parametrisation $W$ and a target exponent $\alpha$, it computes
the nonlinear right-hand-side contribution $N_\alpha$ to the cohomological equation

$$\mathcal{L}(W_\alpha) = N_\alpha\!\left(W_{\lvert\beta\rvert < \lvert\alpha\rvert}\right)$$

$N_\alpha$ is a sum over all multilinear terms $t$ in the model, each evaluated at
combinations of already-computed parametrisation coefficients whose multiindices sum
to $\alpha$.

**System setup.** 2-DOF second-order system with:
- $m = 2$ reduced (internal) coordinates $z_1, z_2$
- $s = 1$ external forcing variable $u$
- `NVAR = 3` variables total
- 5 nonlinear terms covering all three symmetry classes

Each section picks one target exponent $\alpha$ and derives $N_\alpha$ by hand,
then confirms the result numerically.

---

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

```@example multilinear_terms
using MORFE.Multiindices: all_multiindices_up_to
using MORFE.FullOrderModel: NDOrderModel, MultilinearMap
using MORFE.ParametrisationMethod: create_parametrisation_method_objects
using MORFE.MultilinearTerms: compute_multilinear_terms, build_multilinear_terms_cache
using StaticArrays: SVector
using LinearAlgebra

FOM          = 2
ORD          = 2
ROM          = 2
FORCING_SIZE = 1
NVAR         = ROM + FORCING_SIZE  # = 3

term1 = MultilinearMap((res, x, xdot)          -> (@. res += x * xdot),           (1, 1))
term2 = MultilinearMap((res, v1, v2)            -> (@. res += 0.5 * v1 * v2),      (0, 2))
term3 = MultilinearMap((res, x1, x2, xdot)     -> (@. res += 0.5 * x1 * x2 * xdot), (2, 1))
term4 = MultilinearMap((res, x, r)             -> (@. res += 2.0 * x * r),         (1, 0), 1)
term5 = MultilinearMap((res, r)                -> (@. res += [100.0, 200.0] * r),   (0, 0), 1)

Id    = Matrix{Float64}(I, FOM, FOM)
model = NDOrderModel((Id, Id, Id), (term1, term2, term3, term4, term5))

println("Model: FOM=$FOM, ORD=$ORD, ROM=$ROM, FORCING_SIZE=$FORCING_SIZE, NVAR=$NVAR")
```

---

## 2. Multiindex Set

We work with all monomials in $(z_1, z_2, u)$ up to total degree 3.
The graded-lex ordering assigns the following indices:

| Index | Exponent $(z_1, z_2, u)$ |
|-------|--------------------------|
| 1  | $(0,0,0)$ |
| 2  | $(1,0,0)$ — $z_1$ |
| 3  | $(0,1,0)$ — $z_2$ |
| 4  | $(0,0,1)$ — $u$ |
| 5  | $(2,0,0)$ — $z_1^2$ |
| 6  | $(1,1,0)$ — $z_1 z_2$ |
| 7  | $(1,0,1)$ — $z_1 u$ |
| 8  | $(0,2,0)$ — $z_2^2$ |
| 9  | $(0,1,1)$ — $z_2 u$ |
| 10 | $(0,0,2)$ — $u^2$ |
| 11 | $(3,0,0)$ — $z_1^3$ |

```@example multilinear_terms
mset = all_multiindices_up_to(NVAR, 3)
println("$(length(mset.exponents)) monomials in $NVAR variables, degree ≤ 3")
println("\nFirst 12 exponents:")
for (i, e) in enumerate(mset.exponents[1:12])
    println("  idx $i → $e")
end
```

---

## 3. Parametrisation Coefficients

The parametrisation $W : (z_1, z_2, u) \mapsto (x, \dot{x})$ is stored as a
polynomial whose coefficient at multiindex $\alpha$ is a matrix
$W_\alpha \in \mathbb{C}^{n \times \text{ORD}}$:

- `W.poly.coefficients[:, 1, idx]` — position part $W_\alpha^{(x)}$
- `W.poly.coefficients[:, 2, idx]` — velocity part $W_\alpha^{(\dot x)}$

**Linear part** (standard modal expansion):

$$W_{(1,0,0)} = \begin{pmatrix}\varphi_1 \\ \lambda_1\varphi_1\end{pmatrix}, \quad
  W_{(0,1,0)} = \begin{pmatrix}\varphi_2 \\ \lambda_2\varphi_2\end{pmatrix}, \quad
  W_{(0,0,1)} = \begin{pmatrix}b \\ \mathrm{i}\,b\end{pmatrix}$$

with $\varphi_1 = [2,3]^\top$, $\varphi_2 = [4,5]^\top$, $b = [6,7]^\top$,
$\lambda_1 = -0.1 + 10\mathrm{i}$, $\lambda_2 = \bar\lambda_1$.

**Selected quadratic/cubic coefficients** (arbitrary values for testing):

| Exponent | Position | Velocity |
|----------|----------|----------|
| $(2,0,0)$ | $0.1\,\varphi_1$ | $b + \varphi_2$ |
| $(1,1,0)$ | $0.05\,\varphi_1$ | $0.05\,\varphi_2$ |
| $(1,0,1)$ | $0.1\,\varphi_1$ | $-0.3\,\varphi_2$ |
| $(0,2,0)$ | $b + \varphi_2$ | $0.2\,\varphi_2$ |
| $(3,0,0)$ | $500\,\varphi_1$ | $-500\mathrm{i}\,\varphi_2$ |

All other coefficients are zero.

```@example multilinear_terms
W, _ = create_parametrisation_method_objects(mset, ORD, FOM, ROM, FORCING_SIZE, ComplexF64)

λ₁ = -0.1 + 10.0im
λ₂ = conj(λ₁)
φ₁ = ComplexF64[2.0, 3.0]
φ₂ = ComplexF64[4.0, 5.0]
b  = ComplexF64[6.0, 7.0]

# Linear (indices 2, 3, 4 = z₁, z₂, u)
W.poly.coefficients[:, 1, 2] = φ₁;         W.poly.coefficients[:, 2, 2] = λ₁ * φ₁
W.poly.coefficients[:, 1, 3] = φ₂;         W.poly.coefficients[:, 2, 3] = λ₂ * φ₂
W.poly.coefficients[:, 1, 4] = b;          W.poly.coefficients[:, 2, 4] = 1.0im * b

# Quadratic (indices 5..8 = z₁², z₁z₂, z₁u, z₂²)
W.poly.coefficients[:, 1, 5] = 0.1  * φ₁; W.poly.coefficients[:, 2, 5] = b .+ φ₂
W.poly.coefficients[:, 1, 6] = 0.05 * φ₁; W.poly.coefficients[:, 2, 6] = 0.05 * φ₂
W.poly.coefficients[:, 1, 7] = 0.1  * φ₁; W.poly.coefficients[:, 2, 7] = -0.3 * φ₂
W.poly.coefficients[:, 1, 8] = b .+ φ₂;   W.poly.coefficients[:, 2, 8] = 0.2  * φ₂

# Cubic (index 11 = z₁³)
W.poly.coefficients[:, 1, 11] = 500.0 * φ₁; W.poly.coefficients[:, 2, 11] = -500im * φ₂

println("Non-zero W coefficients:")
for idx in [2, 3, 4, 5, 6, 7, 8, 11]
    p = W.poly.coefficients[:, 1, idx]
    v = W.poly.coefficients[:, 2, idx]
    println("  $(mset.exponents[idx])  pos=$p   vel=$v")
end
```

---

## 4. Target $\alpha = (1,0,0)$: Linear in $z_1$

**Claim:** $N_{(1,0,0)} = 0$.

For any degree-$d \geq 2$ term to contribute at $\alpha = (1,0,0)$, we need to
factorise $(1,0,0)$ into $d \geq 2$ multiindices that sum to $(1,0,0)$. The only
way is one factor at $(1,0,0)$ and the rest at $(0,0,0)$. But
$W_{(0,0,0)} = 0$ (the parametrisation is centred at the origin), so every
such factorisation is zero.

This argument holds for all degree-1 targets: nonlinear terms (degree $\geq 2$)
cannot build a degree-1 exponent without using the zero constant term.

```@example multilinear_terms
exp100 = SVector(1, 0, 0)
r100   = compute_multilinear_terms(model, exp100, W)

println("exp = $exp100")
println("  computed : $r100")
println("  manual   : $(ComplexF64[0, 0])")
println("  ‖error‖  : $(norm(r100))")
```

---

## 5. Target $\alpha = (0,0,1)$: Linear in $u$

**Contributing terms:** only `term5`, which has $m_e = 1$ external forcing slot
and $d_\text{int} = 0$ internal slots.

**Forcing split for `term5`** ($m_e = 1$, $\alpha_u = 1$):

The one external slot is assigned to $u$:
$f_\text{idx} = (1,)$, $f_\text{count} = 1$, $\hat\beta = (0,0,1)$,
$\text{rem} = (0,0,0)$.

Since $d_\text{int} = 0$, the only factorisation of rem $= (0,0,0)$ is the
empty tuple. The call is

$$\texttt{term5.f!}(\texttt{temp},\; e_1=1) \;\Rightarrow\; \texttt{temp} = [100,\,200]$$

**`term1`–`term4`:** after the forcing split, rem $= (0,0,0)$ with
$d_\text{int} \geq 1$ — no internal factorisation possible. No contribution.

$$\boxed{N_{(0,0,1)} = [100,\,200]^\top}$$

```@example multilinear_terms
exp001 = SVector(0, 0, 1)
r001   = compute_multilinear_terms(model, exp001, W)

manual001 = ComplexF64[100.0, 200.0]
println("exp = $exp001")
println("  computed : $r001")
println("  manual   : $manual001")
println("  ‖error‖  : $(norm(r001 - manual001))")
```

---

## 6. Target $\alpha = (2,0,0)$: Quadratic in $z_1$

$\alpha = (2,0,0)$, $\deg_\text{max} = 2$.

**Degree filter:** `term3` has $\deg = 3 > 2$ → **skipped**.
`term4`, `term5` require $\alpha_u \geq 1$, but $\alpha_u = 0$ → **skipped**.

---

### `term1`: $x \odot \dot{x}$, `multiindex = (1,1)`, `FullyAsymmetric`

Slot ordering: slot 1 → position ($k=1$), slot 2 → velocity ($k=2$).
All **ordered** pairs $(\alpha_1, \alpha_2)$ with $\alpha_1 + \alpha_2 = (2,0,0)$:

| $\alpha_1$ (pos) | $\alpha_2$ (vel) | $W_{\alpha_1}^{(x)}$ | $W_{\alpha_2}^{(\dot x)}$ | Contribution |
|--|--|--|--|--|
| $(1,0,0)$ | $(1,0,0)$ | $\varphi_1$ | $\lambda_1\varphi_1$ | $\lambda_1\,\varphi_1 \odot \varphi_1$ |
| $(2,0,0)$ | $(0,0,0)$ | $0.1\varphi_1$ | $\mathbf{0}$ | $0$ |
| $(0,0,0)$ | $(2,0,0)$ | $\mathbf{0}$ | $b+\varphi_2$ | $0$ |

$$\text{term1} = \lambda_1\,\varphi_1 \odot \varphi_1$$

---

### `term2`: $0.5\,\dot{x} \odot \dot{x}$, `multiindex = (0,2)`, `FullySymmetric`

Both slots draw from velocity. **Unordered multisets** $\{\alpha_1, \alpha_2\}$
summing to $(2,0,0)$:

| Multiset | sym\_count | Contribution |
|--|--|--|
| $\{(1,0,0),(1,0,0)\}$ | $1\;(=2!/2!)$ | $0.5\,\lambda_1^2\,\varphi_1 \odot \varphi_1$ |
| $\{(2,0,0),(0,0,0)\}$ | $2\;(=2!/1!1!)$ | $0$ (velocity at $(0,0,0)$ is zero) |

$$\text{term2} = 0.5\,\lambda_1^2\,\varphi_1 \odot \varphi_1$$

---

### Total

$$\boxed{N_{(2,0,0)} = \bigl(\lambda_1 + 0.5\,\lambda_1^2\bigr)\,\varphi_1 \odot \varphi_1}$$

With $\lambda_1 = -0.1 + 10\mathrm{i}$:
$\lambda_1^2 = -99.99 - 2\mathrm{i}$, so the scalar prefactor is
$\lambda_1 + 0.5\lambda_1^2 = -50.095 + 9\mathrm{i}$.

```@example multilinear_terms
exp200 = SVector(2, 0, 0)
r200   = compute_multilinear_terms(model, exp200, W)

manual200 = λ₁ .* φ₁ .* φ₁ + 0.5 * λ₁^2 .* φ₁ .* φ₁

println("exp = $exp200")
println("  term1 (λ₁·φ₁⊙φ₁)        : $(λ₁ .* φ₁ .* φ₁)")
println("  term2 (0.5λ₁²·φ₁⊙φ₁)    : $(0.5*λ₁^2 .* φ₁ .* φ₁)")
println("  manual total             : $manual200")
println("  computed                 : $r200")
println("  ‖error‖                  : $(norm(r200 - manual200))")
```

---

## 7. Target $\alpha = (1,0,1)$: Mixed $z_1 \cdot u$

$\alpha = (1,0,1)$, $\deg_\text{max} = 2$.
**Degree filter:** `term3` ($\deg=3$) → **skipped**.
Forcing slice: $\alpha_u = 1$.

---

### `term1`: `FullyAsymmetric`, $m_e = 0$

No forcing split; rem $= (1,0,1)$.

| $\alpha_1$ (pos) | $\alpha_2$ (vel) | Contribution |
|--|--|--|
| $(1,0,0)$ | $(0,0,1)$ | $\varphi_1 \odot \mathrm{i}\,b$ |
| $(0,0,1)$ | $(1,0,0)$ | $b \odot \lambda_1\varphi_1$ |
| $(1,0,1)$ | $(0,0,0)$ | $0.1\varphi_1 \odot \mathbf{0} = 0$ |
| $(0,0,0)$ | $(1,0,1)$ | $\mathbf{0} \odot (-0.3\varphi_2) = 0$ |

$$\text{term1} = (\mathrm{i} + \lambda_1)\,\varphi_1 \odot b$$

---

### `term2`: `FullySymmetric`, $m_e = 0$

Unordered pairs summing to $(1,0,1)$ via velocity:

| Multiset | sym\_count | Contribution |
|--|--|--|
| $\{(1,0,0),(0,0,1)\}$ | $2$ | $2 \times 0.5\,(\lambda_1\varphi_1) \odot (\mathrm{i}\,b) = \mathrm{i}\lambda_1\,\varphi_1 \odot b$ |

$$\text{term2} = \mathrm{i}\lambda_1\,\varphi_1 \odot b$$

---

### `term4`: $2\,x \odot u$, $m_e = 1$

Forcing split: $f_\text{idx} = (1,)$, $f_\text{count} = 1$,
$\hat\beta = (0,0,1)$, rem $= (1,0,0)$.

The single $x$-slot must be $\alpha_1 = (1,0,0)$:
$\texttt{term4.f!}(\texttt{temp},\,\varphi_1,\,1) \Rightarrow \texttt{temp} = 2\varphi_1$.

**`term5`:** after the forcing split rem $= (1,0,0) \neq (0,0,0)$, so the empty
internal factorisation (required for $d_\text{int}=0$) does not match. **No contribution.**

---

### Total

$$\boxed{N_{(1,0,1)} = (\mathrm{i} + \lambda_1 + \mathrm{i}\lambda_1)\,\varphi_1 \odot b + 2\varphi_1}$$

```@example multilinear_terms
exp101 = SVector(1, 0, 1)
r101   = compute_multilinear_terms(model, exp101, W)

manual101_term1 = (1.0im + λ₁) .* φ₁ .* b
manual101_term2 = 1.0im * λ₁ .* φ₁ .* b
manual101_term4 = 2.0 .* φ₁
manual101       = manual101_term1 + manual101_term2 + manual101_term4

println("exp = $exp101")
println("  term1 ((i+λ₁)·φ₁⊙b)     : $manual101_term1")
println("  term2 (iλ₁·φ₁⊙b)        : $manual101_term2")
println("  term4 (2φ₁)              : $manual101_term4")
println("  manual total             : $manual101")
println("  computed                 : $r101")
println("  ‖error‖                  : $(norm(r101 - manual101))")
```

---

## 8. Target $\alpha = (3,0,1)$: Cubic in $z_1$, Linear in $u$

$\alpha = (3,0,1)$, $\deg_\text{max} = 4$. All five terms are within the degree bound.
Forcing slice: $\alpha_u = 1$.

---

### `term1`: `FullyAsymmetric`, $m_e = 0$

Ordered pairs $(\alpha_1, \alpha_2)$ summing to $(3,0,1)$ with both $W_{\alpha_i} \neq 0$:

| $\alpha_1$ (pos) | $\alpha_2$ (vel) | Contribution |
|--|--|--|
| $(3,0,0)$ | $(0,0,1)$ | $(500\varphi_1) \odot (\mathrm{i}\,b)$ |
| $(0,0,1)$ | $(3,0,0)$ | $b \odot (-500\mathrm{i}\,\varphi_2)$ |
| $(2,0,0)$ | $(1,0,1)$ | $(0.1\varphi_1) \odot (-0.3\varphi_2)$ |
| $(1,0,1)$ | $(2,0,0)$ | $(0.1\varphi_1) \odot (b + \varphi_2)$ |

---

### `term2`: `FullySymmetric`, $m_e = 0$

Unordered pairs summing to $(3,0,1)$ via velocity:

| Multiset | sym\_count | Contribution |
|--|--|--|
| $\{(3,0,0),(0,0,1)\}$ | $2$ | $2 \times 0.5\,(-500\mathrm{i}\varphi_2) \odot (\mathrm{i}\,b)$ |
| $\{(2,0,0),(1,0,1)\}$ | $2$ | $2 \times 0.5\,(b+\varphi_2) \odot (-0.3\varphi_2)$ |

---

### `term3`: `GroupwiseSymmetric`, $m_e = 0$

Two $x$-slots (symmetric) and one $\dot{x}$-slot. Enumerate over the velocity slot
$\alpha_3$ and symmetric pairs $\{\alpha_1,\alpha_2\}$ for the two $x$-slots such
that $\alpha_1 + \alpha_2 + \alpha_3 = (3,0,1)$:

| $\alpha_3$ (vel) | $\{\alpha_1,\alpha_2\}$ (pos) | count | Contribution |
|--|--|--|--|
| $(0,0,1)$ | $\{(2,0,0),(1,0,0)\}$ | $2$ | $2\times0.5\,(0.1\varphi_1)\odot\varphi_1\odot(\mathrm{i}\,b)$ |
| $(1,0,0)$ | $\{(2,0,0),(0,0,1)\}$ | $2$ | $2\times0.5\,(0.1\varphi_1)\odot b\odot(\lambda_1\varphi_1)$ |
| $(2,0,0)$ | $\{(1,0,0),(0,0,1)\}$ | $2$ | $2\times0.5\,\varphi_1\odot b\odot(b+\varphi_2)$ |
| $(1,0,1)$ | $\{(1,0,0),(1,0,0)\}$ | $1$ | $1\times0.5\,\varphi_1\odot\varphi_1\odot(-0.3\varphi_2)$ |
| $(1,0,0)$ | $\{(1,0,0),(1,0,1)\}$ | $2$ | $2\times0.5\,\varphi_1\odot(0.1\varphi_1)\odot(\lambda_1\varphi_1)$ |

The count for each groupwise-symmetric factorisation is the within-group multinomial:
two **distinct** elements give $2!/1!1! = 2$; two **identical** elements give $2!/2! = 1$.

---

### `term4`: $m_e = 1$

Forcing split: rem $= (3,0,0)$. Single $x$-slot $\alpha_1 = (3,0,0)$:

$$\texttt{term4.f!}(\texttt{temp},\,500\varphi_1,\,1) \Rightarrow \text{term4} = 2 \times 500\varphi_1 = 1000\varphi_1$$

**`term5`:** rem $= (3,0,0) \neq (0,0,0)$; $d_\text{int}=0$ requires rem $= 0$. **Skipped.**

```@example multilinear_terms
exp301 = SVector(3, 0, 1)
r301   = compute_multilinear_terms(model, exp301, W)

# term1: four ordered pairs
manual301_term1 = (500.0 * φ₁) .* (1.0im * b) +
                  b .* (-500im * φ₂) +
                  (0.1 * φ₁) .* (-0.3 * φ₂) +
                  (0.1 * φ₁) .* (b .+ φ₂)

# term2: two unordered pairs with sym_count=2
manual301_term2 = 0.5 * (-500im * φ₂) .* (1.0im * b) * 2 +
                  0.5 * (b .+ φ₂) .* (-0.3 * φ₂) * 2

# term3: five groupwise-symmetric factorisations
manual301_term3 = 0.5 * (0.1 * φ₁) .* φ₁ .* (1.0im * b) * 2 +
                  0.5 * (0.1 * φ₁) .* b  .* (λ₁ * φ₁)   * 2 +
                  0.5 * φ₁          .* b  .* (b .+ φ₂)    * 2 +
                  0.5 * φ₁          .* φ₁ .* (-0.3 * φ₂)  * 1 +
                  0.5 * φ₁ .* (0.1 * φ₁)  .* (λ₁ * φ₁)   * 2

# term4: single x-slot at (3,0,0)
manual301_term4 = 2.0 * (500.0 * φ₁)

manual301 = manual301_term1 + manual301_term2 + manual301_term3 + manual301_term4

println("exp = $exp301")
println("  term1 : $manual301_term1")
println("  term2 : $manual301_term2")
println("  term3 : $manual301_term3")
println("  term4 : $manual301_term4")
println("  manual total : $manual301")
println("  computed     : $r301")
println("  ‖error‖      : $(norm(r301 - manual301))")
```

---

## 9. Cached Path Verification

`build_multilinear_terms_cache` pre-computes all factorisation structures for the
full multiindex set. The cached variant `compute_multilinear_terms(model, idx, W, cache)`
takes an integer index rather than an `SVector`. The two paths must produce
**bit-identical** results.

```@example multilinear_terms
cache     = build_multilinear_terms_cache(model, W)
mset_exps = mset.exponents

for (label, sv) in [
    ("(1,0,0)", SVector(1,0,0)),
    ("(0,0,1)", SVector(0,0,1)),
    ("(2,0,0)", SVector(2,0,0)),
    ("(1,0,1)", SVector(1,0,1)),
    # (3,0,1) has total degree 4 — outside the degree-3 set; only tested via direct path above
]
    idx      = findfirst(==(sv), mset_exps)
    r_direct = compute_multilinear_terms(model, sv,  W)
    r_cached = compute_multilinear_terms(model, idx, W, cache)
    err      = norm(r_cached - r_direct)
    println("$(err == 0 ? "✓" : "✗")  exp $label : ‖cached − direct‖ = $err")
end
```

---

## Summary

| $\alpha$ | $N_\alpha$ | Key features |
|----------|-----------|--------------|
| $(1,0,0)$ | $0$ | degree filter: nonlinear terms cannot build degree-1 from degree-0 |
| $(0,0,1)$ | $[100,200]^\top$ | $m_e=1$, $d_\text{int}=0$ forcing constant |
| $(2,0,0)$ | $(\lambda_1+0.5\lambda_1^2)\,\varphi_1\odot\varphi_1$ | `FullyAsymmetric` + `FullySymmetric` |
| $(1,0,1)$ | $(\mathrm{i}+\lambda_1+\mathrm{i}\lambda_1)\,\varphi_1\odot b + 2\varphi_1$ | forcing split in `term4`, mixed external/internal |
| $(3,0,1)$ | sum of 4 terms | `GroupwiseSymmetric`, full forcing split, all terms contribute |
