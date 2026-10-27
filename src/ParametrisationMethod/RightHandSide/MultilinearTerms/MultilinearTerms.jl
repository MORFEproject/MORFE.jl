"""
Module `MultilinearTerms` — efficient evaluation of the nonlinear right-hand side
of the cohomological equations.

For each monomial `α` the nonlinear contribution is

	Σₜ  Σ_{β₁+…+βₖ=α}  multiplier · t.f!(W[β₁], …, W[βₖ], r₁, …, rₘ)

where the outer sum runs over all nonlinear terms `t` of the model and the inner
sum enumerates factorisations of `α` into `k` sub-exponents from already-computed
`W` columns.  This module provides two evaluation paths:

- **Non-cached** (`compute_multilinear_terms` with an `SVector` exponent):
  calls the factorisation routines on every invocation.  Simple but allocating.

- **Cached** (`build_multilinear_terms_cache` + `compute_multilinear_terms!`):
  precomputes all factorisation bookkeeping in `MultilinearTermsCache` once before
  the solve loop, then replays it allocation-free at each monomial.

FEM-backed terms (`FEMMultilinearMap`) are grouped by element loop (`fem_element_loop`), and
each group is assembled in a **single pass** over its cells per monomial: each coefficient is
scattered to the quadrature points once per element, and at each quadrature point the fluxes
of every factorisation of every term of the group are summed and tested against the shape
functions once.

Three symmetry strategies (`FullyAsymmetric`, `FullySymmetric`, `GroupwiseSymmetric`)
are dispatched at compile time from the `MultilinearMap.multiindex` field.
"""
module MultilinearTerms

using LinearAlgebra: axpy!
using StaticArrays: SVector

using ..Multiindices: indices_in_box_with_bounded_degree,
                      factorisations_asymmetric, factorisations_fully_symmetric,
                      factorisations_groupwise_symmetric,
                      bounded_index_tuples, FactorisationEntry
using ..ParametrisationObjects: Parametrisation
using ..FullOrderModel: NthOrderModel, MultilinearMap
using ..MultilinearMaps: AbstractMultilinearMap, FEMMultilinearMap,
                         fem_elements, fem_n_qp, fem_ndofs_per_cell,
                         fem_reinit!, scatter_qp!, accumulate_qp!, assemble_element!,
                         fem_getdetJdV, fem_qp_buffer, fem_element_loop, fem_flux_zero,
                         flux_qp, contract_qp!
import ..MultilinearMaps: accumulate_terms_qp!
# The external argument a term receives is materialised here, in *physical* external
# coordinates: unit vectors normally, the columns of Q after a change of external basis.
using ..ExternalSystems: external_argument_vectors

export compute_multilinear_terms, compute_multilinear_terms!, build_multilinear_terms_cache,
       MultilinearTermsCache

include("Symmetry.jl")
include("Structs.jl")
include("NonCachedEval.jl")
include("CacheBuilder.jl")
include("CachedEval.jl")

end # module
