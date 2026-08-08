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

For FEM-backed terms (`FEMMultilinearMap`) an additional **O4 combined element loop**
merges all me=0 FEM contributions into a single mesh traversal per monomial,
avoiding redundant `fem_reinit!` and `scatter_qp!` calls.

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
using ..ParametrisationMethod: Parametrisation
using ..FullOrderModel: NDOrderModel, MultilinearMap
using ..MultilinearMaps: AbstractMultilinearMap, FEMMultilinearMap,
                         fem_elements, fem_n_qp, fem_ndofs_per_cell,
                         fem_reinit!, scatter_qp!, accumulate_qp!, assemble_element!,
                         fem_getdetJdV, fem_qp_buffer
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
