"""
Module `SpectralDecomposition` — solve the generalised eigenproblem for an
`NDOrderModel` and package the result for a parametrisation.

The whole spectral layer, in one module. It replaces the former split between
`Eigensolvers` (a stub that held no solvers) and `Eigenproblems` (which held them) —
a division that named neither half for what it contained.

## Files

| File | Contents |
|------|----------|
| `Eigensolvers.jl` | `AbstractEigensolver` and its subtypes, the `eigensolve` / `eigensolve_left` interface, `generalised_eigenpairs`, the `Spectrum` container, sorting/normalisation, and left-block reconstruction |
| `SpectralData.jl` | `ModeBundle` and `SpectralData` — the selected, model-reconciled bundle a parametrisation consumes |
| `ConjugatePermutation.jl` | the conjugate involution derived from a spectrum and an external system |

## Two layers, deliberately

`Spectrum` is raw solver output: every mode computed, no selection state.
[`SpectralData`](@ref) is what a parametrisation actually consumes: the *selected*
master modes, reconciled against a specific model's order, plus the outer eigenvalues
resonance detection reads. Keeping them apart means a spectrum can be solved once and
several reductions taken from it, and that selection is a pure operation rather than a
mutation.

## Interface contract — full order-blocks

`eigensolve` and `eigensolve_left` must return eigenvectors as `FOM × ORD × n` arrays
containing ALL companion order-blocks, not just the physical slice:

- right: `(λB − A) ψ = 0`, blocks `ψ = [ψ_1; …; ψ_ORD]` with `ψ_{k+1} = λ ψ_k`;
- left (sesquilinear): `φᴴ (λB − A) = 0`, reported eigenvalue `λ` (the pencil eigenvalue
  of the adjoint problem is `conj(λ)`).

The eigensolver is the single owner of eigenvalue knowledge: it uses `λ` to *define* the
eigenvector blocks, and downstream code reads the blocks without ever folding eigenvalues.
Solvers producing only the physical left slice can reconstruct the rest with
[`left_eigenmode_orders_from_slice`](@ref).

## Naming: spectral data, not eigen-data

`SpectralData` and `ModeBundle` store `right_blocks` / `left_blocks`, not "eigenvectors",
and the names are chosen to keep room. A master set need not consist of eigenpairs: for a
defective eigenvalue the invariant subspace is spanned by a **Jordan chain** of
generalised eigenvectors, and such a chain would sit in exactly these fields alongside
its eigenvalue. Nothing here implements Jordan chains today, but nothing here assumes
their absence either — the block layout and the accessors carry over unchanged.
"""
module SpectralDecomposition

using ..FullOrderModel: NDOrderModel, linear_first_order_matrices
using ..ExternalSystems: ExternalSystem, external_basis

using LinearAlgebra
using SparseArrays
using StaticArrays: SVector

# ── Eigensolvers, the Spectrum container, sorting and normalisation ──────────
export AbstractEigensolver, DefaultEigensolver, ArpackEigensolver, MorfeEigensolver,
       StructureModalDampingEigensolver
export eigensolve, eigensolve_left, generalised_eigenpairs
export Spectrum, spectrum
export sort_by_magnitude!, sort_left_eigenmodes, normalise_biorthogonal!
export left_eigenmode_orders_from_slice
export select_master_modes_by_hand, select_master_modes_by_sorting,
       select_master_modes_by_target_frequency

# ── The bundle a parametrisation consumes ────────────────────────────────────
export ModeBundle, SpectralData,
       right_modes, left_modes, right_mode_derivatives, left_mode_blocks,
       master_eigenvalues, outer_eigenvalues, master_bundle, outer_bundle,
       check_biorthogonality, indices,
       physical_mode, spectrum_entries,
       master_conjugate_permutation, outer_conjugate_permutation

# ── Conjugate structure ──────────────────────────────────────────────────────
export detect_conjugate_permutation, external_conjugate_permutation,
       full_conjugate_permutation

include("Eigensolvers.jl")
include("ConjugatePermutation.jl")
include("SpectralData.jl")

end # module
