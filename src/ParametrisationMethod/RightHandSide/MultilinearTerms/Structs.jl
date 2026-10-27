# -----------------------------------------------------------------------
# Factorisation cache — structs
# -----------------------------------------------------------------------
#
# Use case: the parametrisation solve loop calls compute_multilinear_terms
# once per monomial (O(L) calls, where L = |mset|).  Each call invokes the
# factorisation routines, which enumerate index tuples and allocate Vectors.
# For large L this GC pressure is measurable.
#
# Since factorisation results depend only on the multiindex set and the model
# structure — not on the parametrisation coefficients W — they can be
# precomputed once and replayed on every call.
#
# What IS cached: for each (monomial, term, external-split) triple,
#   - which coefficient indices to load from W,
#   - their symmetry multipliers,
#   - which external arguments to pass as forcing arguments (unit vectors, or the
#     columns of the change of basis when the external system was re-based).
#
# What is NOT cached: the O(FOM) arithmetic (fill!, t.f!, axpy!).
# Those operations read from W, which changes at every solve step.

"""
	CachedSplit

Precomputed bookkeeping for one `(monomial l, term t, external-split)` triple.

# Fields

- `ext_count::Int` — multiplicity of this external-variable split (from
  `bounded_index_tuples`); always 1 when `me = 0`.
- `args_ext_indices::Vector{Int}` — indices into the cache's `external_arguments` that
  reconstruct the external forcing arguments; empty when `me = 0`.
- `is_asymmetric::Bool` — true iff the term is [`FullyAsymmetric`](@ref), in which
  case replaying it needs no scratch buffer.
- `orders::Vector{Int}` — derivative index for each factor slot; length is the
  internal degree of the term.
- `entries::Vector{FactorisationEntry}` — one entry per factorisation of the
  remainder exponent, each carrying its own symmetry multiplier.
"""
struct CachedSplit
    ext_count::Int
    args_ext_indices::Vector{Int}
    is_asymmetric::Bool
    orders::Vector{Int}
    entries::Vector{FactorisationEntry}
end

"""
	FEMEntry{DEG, ME}

One factorisation of one FEM term at one monomial, for the single-pass assembly.

# Fields

- `multiplier::Int`: the symmetry count of the factorisation, as in `FactorisationEntry`,
  times the multiplicity `ext_count` of its external split.
- `factors::NTuple{DEG, Int}`: for each internal slot, the index into the monomial's
  `cols` in its [`FEMGroup`](@ref), which is also the row of `qp_buffer` holding that
  coefficient's quadrature-point values.
- `externals::NTuple{ME, Int}`: for each external slot, the index into
  `external_arguments`.

`DEG` is the internal degree and `ME` the external multiplicity of the term, so the
argument tuple has `DEG + ME == t.deg` entries and is built with compile-time lengths.
"""
struct FEMEntry{DEG, ME}
    multiplier::Int
    factors::NTuple{DEG, Int}
    externals::NTuple{ME, Int}
end

"""
	FEMGroup{TT, FE, QP, T}

The FEM terms of a model that share one element loop (equal `fem_element_loop` labels),
with what the single-pass assembly needs to traverse that loop once per monomial for all of
them.

Type parameters:
- `TT`: the tuple type of `terms`.
- `FE`: the tuple type of every `entries[l]`, the same at each monomial.
- `QP`: element type of `qp_buffer`, taken from `fem_qp_buffer` of the driver (e.g.
  `Tensor{2,3,ComplexF64,9}` for Ferrite SVK), so that element access in the hot loop is
  statically typed.
- `T`: element type of `Fe`.

# Fields

- `terms::TT`: the terms of the group, in model order, as a concretely typed tuple; the
  first drives the element loop.
- `cols::Vector{Vector{Tuple{Int, Int}}}`: per monomial, the distinct
  `(derivative order, W column)` pairs that the factorisations of the group read.
- `entries::Vector{FE}`: per monomial, one `Vector{`[`FEMEntry`](@ref)`}` per term, aligned
  with `terms`.
- `qp_buffer::Matrix{QP}`: quadrature-point values of the current element, one row per
  entry of `cols[l]` (filled by `scatter_qp!`) and one column per quadrature point.
- `Fe::Vector{T}`: the element vector, shared by the terms of the group.
"""
struct FEMGroup{TT <: Tuple, FE <: Tuple, QP, T}
    terms::TT
    cols::Vector{Vector{Tuple{Int, Int}}}
    entries::Vector{FE}
    qp_buffer::Matrix{QP}
    Fe::Vector{T}
end

"""
	MultilinearTermsCache{T, EV, G}

All precomputed factorisation bookkeeping for a given `(model, parametrisation)` pair.

**Build** once before the solve loop with `build_multilinear_terms_cache`.
**Use** by passing to `compute_multilinear_terms!` inside the loop.  The cache is valid as
long as the multiindex set and the model structure are unchanged (i.e. across all solve
steps).

Closure terms (`MultilinearMap`) are replayed split by split. FEM terms
(`FEMMultilinearMap`) are assembled in a single pass per monomial over each element loop,
together with the other terms on that loop.

Type parameters:
- `T`: element type of FOM-length vectors (e.g. `ComplexF64`).
- `EV`: element type of `external_arguments`, i.e. `SVector{N_EXT, Int}` normally and
  `SVector{N_EXT, eltype(Q)}` when the external system was re-based, so that the hot
  loop's argument tuple is inferred rather than `Any`.
- `G`: the tuple type of `fem_groups`.

# Fields

- `splits::Vector{Vector{Vector{CachedSplit}}}`: `splits[l][t_idx]` lists the
  [`CachedSplit`](@ref) values of closure term `t_idx` at monomial `l`; empty for FEM terms
  and when the term degree exceeds the monomial degree.
- `scratch_buffer::Vector{T}`: length `FOM`, working space for symmetric closure terms.
- `temp_buffer::Vector{T}`: length `FOM`, holds one external split of a closure term.
- `external_arguments::Vector{EV}`: the external argument passed for each external
  variable: the unit vector `eⱼ` in the model's own coordinates, or the column `Q[:, j]` of
  the change of basis when the external system was re-based (see
  `ExternalSystems.external_argument_vectors`).  Applying `Q` here, once per solve, is why
  no term ever has to know about the change of coordinates.
- `fem_groups::G`: one [`FEMGroup`](@ref) per element loop, in the order the loops first
  appear among the model's FEM terms; empty without FEM terms.
"""
struct MultilinearTermsCache{T, EV, G <: Tuple}
    splits::Vector{Vector{Vector{CachedSplit}}}
    scratch_buffer::Vector{T}
    temp_buffer::Vector{T}
    external_arguments::Vector{EV}                 # eⱼ, or Q[:, j]; empty when N_EXT == 0
    fem_groups::G
end
