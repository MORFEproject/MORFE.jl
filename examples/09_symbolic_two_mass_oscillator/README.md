# 09 — Symbolic full-order model

This notebook builds MORFE full-order models by writing their equations as **symbolic
expressions** instead of assembling matrices and `MultilinearMap` closures by hand. The model
is the two-degree-of-freedom nonlinear oscillator of Shaw and Pierre: two coupled masses with
linear stiffness and damping, the first carrying an additional cubic restoring force.

The example is intentionally limited to the public API of the Symbolics extension:

```julia
model = model_from_symbolics(exprs, groups)                      # autonomous
model = model_from_symbolics(exprs, groups, ext_var, ext_exprs)  # coupled to a driver
sys   = externalsystem_from_symbolics(ext_exprs, ext_var)        # a driver on its own
```

There is no example-specific extraction, eigensolver, cohomological solver or equation printer.

`MORFESymbolicsExt` is a package extension: it loads by itself as soon as both `MORFE` and
`Symbolics` are in the same session, so nothing is imported by name.

## What it covers

| Section | Shows |
|---------|-------|
| 1–2 | `model_from_symbolics(exprs, groups)` on the unforced oscillator, and the `B₀, B₁, B₂` matrices and `MultilinearMap` multiindices it extracted |
| 3 | Harmonic forcing routed through an `ExternalSystem`, built and coupled in one call |
| 4 | A gallery of drivers: harmonic, quasi-periodic, multiharmonic (which triggers the re-basing `@info`), and the chaotic Lorenz system shifted to a non-trivial equilibrium |
| 5 | The DifferentialEquations.jl-shaped layer — `f!(dᴺu, …, du, u, p, t)` and `g!(dr, r, p, t)` — checked against the symbolic objects built earlier |
| 6 | A damped oscillator driven by each system in turn, integrated with `Tsit5` and plotted |
| 7 | A run summary written next to the figure |

The driver right-hand sides are never retyped for the integration: they come out of the
`ExternalSystem` itself through `evaluate`, with `external_basis` mapping the initial condition
in and `to_physical_external` reading the state back.

## Run

From the repository root, initialise this example's environment once:

```bash
julia --project=examples/09_symbolic_two_mass_oscillator -e \
  'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

This uses the MORFE source in the current checkout. The generated `Manifest.toml` stays local
and is not committed, so it carries no machine-specific paths into the repository.

Then open and execute [`symbolic_model.ipynb`](symbolic_model.ipynb), or from a shell:

```bash
cd examples/09_symbolic_two_mass_oscillator
jupyter nbconvert --execute --to notebook --inplace symbolic_model.ipynb
```

It needs no FEM backend and no mesh, and runs in well under a minute once the environment is
precompiled.

## Output

```text
results/
  summary.txt
  figures/
    forced_response.png
```

`summary.txt` records the extracted linear matrices and multiindices, the size and re-basing
status of each driver, the two closed-form integration checks, and the Julia version, MORFE
commit and timestamp.

## Two things worth knowing

**The expression method validates nothing.** `externalsystem_from_symbolics(exprs, var)` does
not check that its right-hand side is polynomial, and does not check that the origin is an
equilibrium — a stray constant term is absorbed silently. Those checks (`check_expr`, and
through it `is_polynomial` and `check_constant_terms`) run only on the `model_from_symbolics`
path. This is why section 4 shifts Lorenz to its non-trivial equilibrium explicitly rather
than relying on being told.

**The function layout is real-only.** `_differential_equations_helper_external` builds a
`Vector{Num}`, so a complex right-hand side such as `dr[1] = im*Ω*r[1]` raises
`InexactError`. Section 5 writes the harmonic driver in its equivalent real form
`ṙ₁ = Ω r₂, ṙ₂ = −Ω r₁`, which traces the same circle. The expression method
`externalsystem_from_symbolics(exprs, var)` takes complex coefficients without trouble.

## The website tutorial

The [Symbolic full-order model](https://morfeproject.github.io/MORFE.jl/tutorials/symbolics_ext.html)
tutorial walks through the same material. Its embedded figures are generated separately by
`website/tutorials/assets/symbolics_ext/generate_assets.jl`, which uses the site's
dependency-free renderer rather than Plots.
