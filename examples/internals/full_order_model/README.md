# full_order_model — Building a full-order model

Every DPIM computation starts from the same three objects. `main.jl` builds all three
from scratch and draws what each one means: nonlinear terms → the driver → the assembled
model → its response.

| Section | Shows |
|---------|-------|
| 1 | `MultilinearMap`: the `multiindex` as a calling convention, evaluated with `evaluate_term!` — hardening/softening cubics, quadratic drag, a mixed `x·ẋ` term, and the `@info` reporting assumed constructor defaults |
| 2 | `ExternalSystem`: harmonic `(iΩ, −iΩ)`, quasi-periodic with incommensurate frequencies, and a *nonlinear* upper-triangular cascade — all integrated through `evaluate(sys.first_order_dynamics, r)` |
| 3 | Lorenz: why the constructor rejects it, and the shift-and-diagonalise coordinate change that repairs the full nonlinear system |
| 4 | `NthOrderModel`: assembling a forced Duffing oscillator, `linear_first_order_matrices`, `evaluate_nonlinear_terms!`, and the external-system check |

No FEM backend and no solve: the script runs in a couple of seconds against the
repository's root environment.

## How to run

From the repository root:

```bash
julia --project -e 'include("examples/internals/full_order_model/main.jl")'
```

## The upper-triangularity constraint

Section 3 is the one worth reading if you are about to write your own driver. An
`ExternalSystem`'s **linear** part must be upper triangular — its nonlinear part is
unconstrained. The reason is causality: the cohomological equations are solved monomial
by monomial in GrLex order, and the `|β| = 1` lower-order coupling reads only the strictly
upper triangle of `Λ`, so an entry below the diagonal would be discarded without trace.

Lorenz fails this at the origin, and no reordering of its variables helps — both
off-diagonal entries of the x–y block are non-zero, so one is always below the diagonal.
But the constraint is a property of the *coordinates*, not of the system, so the
constructor repairs it instead of rejecting it: it finds a basis `Q` in which the linear
part is triangular and re-expresses the **whole** polynomial as `ṙ′ = U r′ + Q⁻¹g(Q r′)`,
storing `Q` in the `basis` field. Because Lorenz's Jacobian is real, the eigenvector basis
is chosen — which additionally makes `U` diagonal and preserves the conjugate pairing that
realification depends on.

Section 3 still shifts the origin to a non-trivial fixed point `C₊`, because that is *not*
something a change of basis can do: it removes the constant term, and an `ExternalSystem`
polynomial has none. It then diagonalises the Jacobian by hand, `J = T Λ T⁻¹`, so the
automatic result has something independent to be checked against.

The script checks both coordinate changes rather than asserting them: the hand-derived
modal RHS mapped back through `T` matches Lorenz's own to `~1e-13`, the automatic basis
reproduces the centred field to the same order, and an orbit integrated as an
`ExternalSystem` tracks a direct integration of Lorenz. The figure draws both, so the
butterfly is visibly the same curve.

## Output

Four standalone HTML figures in `results/figures/`:

| File | Content |
|------|---------|
| `fig1_nonlinear_terms.html` | restoring-force curves, quadratic drag, and a bilinear `(1, 1)` term — every curve produced by `evaluate_term!` |
| `fig2_external_systems.html` | harmonic, quasi-periodic and nonlinear-cascade drivers, as time series and phase portraits |
| `fig3_lorenz.html` | the attractor in physical and modal coordinates, drawn over a direct integration |
| `fig4_forced_response.html` | the assembled Duffing model's forced response, transient and steady state |

Each file is self-contained — no CDN, no external stylesheet or script — so it opens
straight from the file system. Curves are drawn as inline SVG with hover readout; panels
switch from the button bar. `viz.jl` imports nothing at all, which is what lets the whole
tutorial run under a plain `julia --project`: the root environment carries no plotting
package.

The figures embedded in the website live in `website/tutorials/assets/full_order_model/`.
Refresh them after changing the script with:

```bash
MORFE_FOM_OUT=website/tutorials/assets/full_order_model \
  julia --project -e 'include("examples/internals/full_order_model/main.jl")'
```

## The other scripts here

`demo_NthOrderModel.jl` and `demo_external_system.jl` are terse, copy-pasteable API demos
of the same types, without figures or narration. Start with `main.jl`; reach for those
when you want the shortest possible snippet to adapt.
