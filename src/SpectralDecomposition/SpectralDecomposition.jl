"""
Module `SpectralDecomposition` — solve the generalised eigenproblem for an
`NDOrderModel` and package the result for a parametrisation.

This is the whole spectral layer in one module. It replaces the former split between
`Eigensolvers` (a stub that held no solvers) and `Eigenproblems` (which held them),
a division that named neither half for what it contained.

## Contents

- **Eigensolvers.** `AbstractEigensolver` and its concrete subtypes, plus the
  `eigensolve` / `eigensolve_left` interface each must implement. `generalised_eigenpairs`
  is the Arpack-backed shift-invert entry point (activated by `using Arpack, LinearMaps`).
- **`Spectrum`.** The raw solver output: eigenvalues and the full right/left companion
  order-blocks, for every mode computed. No selection state.
- **Sorting and normalisation.** `sort_by_magnitude!`, `sort_left_eigenmodes`,
  `normalise_biorthogonal!`.
- **Left-block reconstruction.** `left_eigenmode_orders_from_slice` for solvers that
  return only the physical left slice.

`SpectralData` — the *selected*, model-reconciled bundle that a parametrisation actually
consumes — lives alongside in `SpectralDataTypes`, and the conjugate involution in
`ConjugatePermutation`.

## Interface contract — full order-blocks

`eigensolve` and `eigensolve_left` must return eigenvectors as `FOM × ORD × n` arrays
containing ALL companion order-blocks, not just the physical slice:

- right: `(λB − A) ψ = 0`, blocks `ψ = [ψ_1; …; ψ_ORD]` with `ψ_{k+1} = λ ψ_k`;
- left (sesquilinear): `φᴴ (λB − A) = 0`, reported eigenvalue `λ` (the pencil eigenvalue
  of the adjoint problem is `conj(λ)`).

The eigensolver is the single owner of eigenvalue knowledge: it uses `λ` to *define* the
eigenvector blocks, and downstream code reads the blocks without ever folding eigenvalues.
"""
module SpectralDecomposition

using ..FullOrderModel

using LinearAlgebra
using SparseArrays

export AbstractEigensolver, DefaultEigensolver, ArpackEigensolver, MorfeEigensolver,
       StructureModalDampingEigensolver
export eigensolve, eigensolve_left, generalised_eigenpairs
export sort_by_magnitude!, normalise_biorthogonal!, sort_left_eigenmodes
export Spectrum, spectrum, left_eigenmode_orders_from_slice

include("Eigensolvers.jl")
include("Spectrum.jl")

end # module
