# internals — Low-level API demos

Fast, self-contained scripts that exercise individual MORFE subsystems.
Useful for understanding internals, testing custom extensions, or debugging.

| Script | Subsystem | Runtime |
|--------|-----------|---------|
| `demo_polynomials.jl` | `DensePolynomial` algebra and evaluation | <5 s |
| `demo_multiindices_factorisations.jl` | `MultiindexSet` construction and factorisation | <5 s |
| `multiindex_sets/main.jl` | Tutorial: building `MultiindexSet`s, deletion, spectral truncation | <5 s |
| `eigensolver/demo_eigenproblem.jl` | `Eigenproblem` API | <5 s |
| `eigensolver/demo_eigensolver.jl` | `ArpackEigensolver` on a sparse K/M pair | <30 s |
| `eigensolver/demo_propagation.jl` | Eigenvector propagation for 2nd-order systems | <5 s |
| `full_order_model/demo_NDOrderModel.jl` | `NDOrderModel` construction | <5 s |
| `full_order_model/demo_external_system.jl` | `ExternalSystem` setup | <5 s |
| `parametrisation_method/demo_resonances.jl` | `ResonanceSet` construction | <5 s |
| `parametrisation_method/demo_parametrisation_method.jl` | Full cohomological solve (small synthetic FOM) | <1 min |

## How to run

From the repository root (using the root project environment):

```bash
julia --project -e 'include("examples/internals/demo_polynomials.jl")'
julia --project -e 'include("examples/internals/demo_multiindices_factorisations.jl")'
```
