"""
Module `Resonance` — resonance detection for the parametrisation method.

## Eigenvalue roles

Three eigenvalue groups are distinguished:

- **`master_eigenvalues`** (required): the ROM eigenvalues.  They enter the
  superharmonic `s = ⟨λ, α⟩` **and** define the inner resonance targets (rows 1:ROM
  of `inner_resonances`).

- **`external_eigenvalues`** (optional): eigenvalues of the external forcing system.
  They enter `s` through the multiindex coefficients but do **not** produce a target
  row.  Pass them so that `s` is computed over the full `NVAR = ROM + N_EXT` index.

- **`outer_eigenvalues`** (optional): additional resonance targets (eigenvalues not included in `master_eigenvalues`, tested for near-resonance).  They define the rows of `outer_resonances`
  but do **not** enter `s`.

## Choosing a resonance style

Four constructors are provided:

- **`resonance_set_from_graph_style`**: every monomial of total degree ≥ 2 is
  automatically resonant with all master modes (inner resonances); outer resonances are
  flagged by eigenvalue proximity.  Use for non-autonomous SSMs with harmonic forcing.

- **`resonance_set_from_complex_normal_form_style`**: inner resonances determined by
  eigenvalue proximity to `master_eigenvalues`; suitable for autonomous SSMs with
  complex conjugate reduced variables.

- **`resonance_set_from_real_normal_form_style`**: like CNF but conjugate pairs share
  the resonance flag; use when building a real-valued ROM.

- **`resonance_set_from_condition_number_estimate`**: flags near-resonances using a
  condition-number criterion rather than a fixed tolerance.
"""
module Resonance

using Printf: @sprintf

using ..Multiindices: MultiindexSet, find_in_set
using ..FullOrderModel: NthOrderModel
using ..SpectralDecomposition: Spectrum, SpectralData, outer_bundle, indices,
                               outer_conjugate_permutation, physical_mode
using ..ExternalSystems: external_basis

export ResonanceSet,
       build_resonance_set,
       resonance_set_from_graph_style,
       resonance_set_from_complex_normal_form_style,
       resonance_set_from_real_normal_form_style,
       resonance_set_from_condition_number_estimate,
       empty_resonance_set,
       set_resonance!,
       is_resonant,
       n_internal,
       resonant_targets,
       resonant_multiindices,
       EigenvalueCondition,
       RealEigenvalueCondition,
       ConditionNumberEstimateCondition,
       GraphInternal,
       NormalFormInternal,
       ResonanceConfig,
       resolve_tolerances

# ======================================================================
# ResonanceSet
# ======================================================================

"""
	ResonanceSet{ROM, N_EXT, M}

Boolean look-up table recording which monomials are resonant with which master-mode
or outer-mode targets.

Type parameters: `ROM` = number of master modes, `N_EXT` = external system size,
`M` = matrix type (typically `BitMatrix`).

Resonance decides, per monomial, whether the cohomological system gets a border on a
given master row — so this table is consulted once per monomial per master mode, and
is precomputed as bits rather than re-tested against tolerances during the solve.

Use one of the `resonance_set_from_*` constructors rather than building this directly.

# Fields

- `multiindices::MultiindexSet` — the set over which resonances are defined; its
  `NVAR` must equal `ROM + N_EXT`, which the constructor enforces.
- `inner_resonances::M` — `ROM × NMON`; entry `(r, k)` is `true` when monomial `k`
  is resonant with master mode `r`.
- `outer_resonances::Union{Nothing, M}` — `n_out × NMON` for outer (non-master)
  targets, or `nothing` when there are none.  Outer resonance is not something the
  border can absorb; it signals that the master set is too small.
"""
struct ResonanceSet{ROM, N_EXT, M <: AbstractMatrix{Bool}}
    multiindices::MultiindexSet               # NVAR = ROM + N_EXT, enforced at construction
    inner_resonances::M                       # (ROM, NMON)
    outer_resonances::Union{Nothing, M}       # (n_out, NMON) or nothing

    function ResonanceSet{ROM, N_EXT, M}(
            multiindices::MultiindexSet{NVAR},
            inner::M,
            outer::Union{Nothing, M}) where {ROM, N_EXT, NVAR, M <: AbstractMatrix{Bool}}
        @assert NVAR == ROM + N_EXT "NVAR=$NVAR but ROM=$ROM + N_EXT=$N_EXT = $(ROM+N_EXT)"
        NMON = length(multiindices)
        @assert size(inner) == (ROM, NMON) "inner_resonances size $(size(inner)) ≠ ($ROM, $NMON)"
        if outer !== nothing
            @assert size(outer, 2) == NMON "outer_resonances column count $(size(outer,2)) ≠ $NMON"
        end
        new{ROM, N_EXT, M}(multiindices, inner, outer)
    end
end

"""
	n_internal(rs::ResonanceSet{ROM}) -> ROM

Return the number of internal master modes (compile-time constant from type parameter).
"""
n_internal(::ResonanceSet{ROM}) where {ROM} = ROM

"""
	empty_resonance_set(multiindices, n_internal, n_outer=0) -> ResonanceSet

Construct a `ResonanceSet` with all resonance flags set to `false`.
`n_internal` = number of master modes (ROM); `n_outer` = number of outer targets (0 = none).
"""
function empty_resonance_set(
        multiindices::MultiindexSet{NVAR}, n_int::Int, n_out::Int = 0) where {NVAR}
    N_EXT = NVAR - n_int
    NMON = length(multiindices)
    inner = falses(n_int, NMON)
    outer = n_out > 0 ? falses(n_out, NMON) : nothing
    ResonanceSet{n_int, N_EXT, BitMatrix}(multiindices, inner, outer)
end

"""
	set_resonance!(rs, target, idx, value) -> rs
	set_resonance!(rs, target, mi, value)  -> rs

Set the resonance flag for target `target` and monomial `idx` (or multiindex vector
`mi`) to `value`.  Targets `1:ROM` address `inner_resonances`; targets `> ROM` address
`outer_resonances`.  Returns `rs` for chaining.  Warns if `mi` is not found.
"""
function set_resonance!(rs::ResonanceSet{ROM}, target::Int,
        idx::Int, value::Bool) where {ROM}
    if target ≤ ROM
        rs.inner_resonances[target, idx] = value
    else
        rs.outer_resonances[target - ROM, idx] = value
    end
    return rs
end
function set_resonance!(rs::ResonanceSet{ROM}, target::Int,
        mi::Vector{Int}, value::Bool) where {ROM}
    idx = find_in_set(rs.multiindices, mi)
    idx === nothing && @warn "Multiindex $mi not found" && return rs
    return set_resonance!(rs, target, idx, value)
end

"""
	is_resonant(rs, idx, target) -> Bool
	is_resonant(rs, mi, target)  -> Bool

Return `true` when the monomial at position `idx` (or exponent vector `mi`) is resonant
with target `target`.  Targets `1:ROM` query `inner_resonances`; targets `> ROM` query
`outer_resonances` (returns `false` when outer is `nothing`).
"""
function is_resonant(rs::ResonanceSet{ROM}, idx::Int, target::Int)::Bool where {ROM}
    if target ≤ ROM
        return rs.inner_resonances[target, idx]
    elseif rs.outer_resonances !== nothing
        return rs.outer_resonances[target - ROM, idx]
    else
        return false
    end
end
function is_resonant(rs::ResonanceSet{ROM}, mi::Vector{Int}, target::Int) where {ROM}
    idx = find_in_set(rs.multiindices, mi)
    idx === nothing && return false
    return is_resonant(rs, idx, target)
end

"""
	resonant_targets(rs, idx) -> AbstractVector{Bool}
	resonant_targets(rs, mi)  -> Union{AbstractVector{Bool}, Nothing}

Return a boolean vector indicating which targets are resonant with the monomial at
position `idx` (or exponent vector `mi`).  Concatenates inner and outer rows.
Returns `nothing` when `mi` is not in the multiindex set.
"""
function resonant_targets(rs::ResonanceSet, idx::Int)
    rs.outer_resonances === nothing && return rs.inner_resonances[:, idx]
    return vcat(rs.inner_resonances[:, idx], rs.outer_resonances[:, idx])
end
function resonant_targets(rs::ResonanceSet, mi::Vector{Int})
    idx = find_in_set(rs.multiindices, mi)
    idx === nothing && return nothing
    return resonant_targets(rs, idx)
end

"""
	resonant_multiindices(rs, target) -> Vector{Int}

Return the positions of all monomials resonant with `target`.
Targets `1:ROM` query `inner_resonances`; targets `> ROM` query `outer_resonances`.
"""
function resonant_multiindices(rs::ResonanceSet{ROM}, target::Int) where {ROM}
    if target ≤ ROM
        return findall(rs.inner_resonances[target, :])
    end
    rs.outer_resonances === nothing && return Int[]
    return findall(rs.outer_resonances[target - ROM, :])
end

# ======================================================================
# Internal strategies
# ======================================================================

"""
	InternalResonance

Abstract supertype for strategies that decide which monomials are resonant with the
*inner* (ROM) master modes.
"""
abstract type InternalResonance end

"""
	GraphInternal <: InternalResonance

Every monomial of total degree ≥ 2 is marked resonant with all inner master modes.
Linear monomials `eᵣ` are resonant only with their own mode `r`.
"""
struct GraphInternal <: InternalResonance end

"""
	NormalFormInternal <: InternalResonance

No monomial is automatically marked resonant with inner modes; resonance is determined
entirely by the eigenvalue-proximity condition.
"""
struct NormalFormInternal <: InternalResonance end

"""
	apply_internal_resonances!(mat, strategy, mi, n_int, k)

Set inner-resonance flags in column `k` of the inner matrix `mat` for the monomial
with exponent vector `mi`.

- `GraphInternal`: all `n_int` rows flagged for degree ≥ 2; for a linear monomial `eᵣ`
  only row `r` (if `r ≤ n_int`) is flagged.
- `NormalFormInternal`: no-op.
"""
function apply_internal_resonances!(::AbstractMatrix{Bool}, ::NormalFormInternal,
        ::AbstractVector{Int}, ::Int, ::Int)
    return
end
function apply_internal_resonances!(mat::AbstractMatrix{Bool}, ::GraphInternal,
        mi::AbstractVector{Int}, n_int::Int, k::Int)
    deg = sum(mi)
    if deg == 1
        pos = findfirst(!iszero, mi)
        if pos ≤ n_int
            mat[pos, k] = true
        else
            for j in 1:n_int
                mat[j, k] = true
            end
        end
    elseif deg > 1
        for j in 1:n_int
            mat[j, k] = true
        end
    end
end

# ======================================================================
# Outer resonance conditions
#
# All methods use **local indexing**: `target` is a local row index within
# the condition's own eigenvalue array (always 1:n for some n).
# ======================================================================

"""
	OuterResonanceCondition

Abstract supertype for conditions that test whether a monomial is resonant with a
mode at superharmonic frequency `s`.

All concrete subtypes must implement
`is_resonant(cond, target::Int, s::Number, k::Int) -> Bool`.
"""
abstract type OuterResonanceCondition end

"""
	EigenvalueCondition <: OuterResonanceCondition

Flags a monomial as resonant when `|λⱼ - s| < tol`.

# Fields

- `eigenvalues::Vector{ComplexF64}` — the target eigenvalues, in local indexing.
- `tol::Union{Float64, Vector{Vector{Float64}}}` — a scalar tolerance, or a
  per-monomial, per-target table when the threshold has to vary.
- `target_indices::Vector{Int}` — which local targets this condition applies to,
  typically `1:n`.  Kept explicit so several conditions can cover disjoint targets.
"""
struct EigenvalueCondition <: OuterResonanceCondition
    eigenvalues::Vector{<:Number}
    tol::Union{Float64, Vector{Vector{Float64}}}
    target_indices::Vector{Int}
    function EigenvalueCondition(eig, tol, target_indices = 1:length(eig))
        new(eig, tol, collect(target_indices))
    end
end

"""
	RealEigenvalueCondition <: OuterResonanceCondition

Flags a monomial as resonant when `|λⱼ - s| < tol` **or** `|λ_{conj(j)} - s| < tol`,
so that conjugate eigenvalue pairs share the resonance flag.

# Fields

- `eigenvalues::Vector{ComplexF64}` — the target eigenvalues, in local indexing.
- `conjugacy_map::Vector{Int}` — `conjugacy_map[i]` is the local index of the
  conjugate of eigenvalue `i`.  Pairing them keeps the flags symmetric, which a
  real-valued full-order model requires.
- `tol::Union{Float64, Vector{Vector{Float64}}}` — a scalar tolerance, or a
  per-monomial, per-target table.
- `target_indices::Vector{Int}` — which local targets this condition applies to.
"""
struct RealEigenvalueCondition <: OuterResonanceCondition
    eigenvalues::Vector{<:Number}
    conjugacy_map::Vector{Int}
    tol::Union{Float64, Vector{Vector{Float64}}}
    target_indices::Vector{Int}
    function RealEigenvalueCondition(eig, conj, tol, target_indices = 1:length(eig))
        new(eig, conj, tol, collect(target_indices))
    end
end

"""
	ConditionNumberEstimateCondition <: OuterResonanceCondition

Flags a monomial as resonant using the criterion:

	|λⱼ - s| * max_cond < spectral_radius * κ(λⱼ)

Scaling the test by the spectral radius and by each eigenvalue's own conditioning
makes the criterion dimensionless, so it transfers across models without retuning a
raw distance tolerance.

# Fields

- `eigenvalues::Vector{ComplexF64}` — the target eigenvalues, in local indexing.
- `spectral_radius::Float64` — spectral radius of the full-order system, setting the
  scale against which `|λⱼ - s|` is judged.
- `condition_numbers::Vector{Float64}` — per-target eigenvalue condition numbers
  `κ(λⱼ)`.
- `max_cond::Float64` — the largest condition number tolerated for the cohomological
  operator before the monomial counts as resonant.
- `target_indices::Vector{Int}` — which local targets this condition applies to.
- `conjugacy_map::Union{Nothing, Vector{Int}}` — optional local conjugacy map, used
  as in [`RealEigenvalueCondition`](@ref); `nothing` when pairing is not wanted.
"""
struct ConditionNumberEstimateCondition <: OuterResonanceCondition
    eigenvalues::Vector{<:Number}
    spectral_radius::Float64
    condition_numbers::Vector{Float64}
    max_cond::Float64
    target_indices::Vector{Int}
    conjugacy_map::Union{Nothing, Vector{Int}}
    function ConditionNumberEstimateCondition(
            eig, spectral_radius, eigenvalue_condition_number,
            max_cond, target_indices, conj = nothing)
        new(eig, spectral_radius, eigenvalue_condition_number,
            max_cond, collect(target_indices), conj)
    end
end

@inline _local_index(cond::OuterResonanceCondition, target::Int) = findfirst(==(target), cond.target_indices)

function is_resonant(cond::EigenvalueCondition, target::Int, s::Number, k::Int)::Bool
    local_idx = _local_index(cond, target)
    local_idx === nothing && return false
    eig = cond.eigenvalues[local_idx]
    tol = cond.tol
    return tol isa Float64 ? abs(eig - s) < tol : abs(eig - s) < tol[k][local_idx]
end

function is_resonant(cond::RealEigenvalueCondition, target::Int, s::Number, k::Int)::Bool
    local_idx = _local_index(cond, target)
    local_idx === nothing && return false
    local_conj = cond.conjugacy_map[local_idx]
    eig1 = cond.eigenvalues[local_idx]
    eig2 = cond.eigenvalues[local_conj]
    tol = cond.tol
    if tol isa Float64
        return (abs(eig1 - s) < tol) || (abs(eig2 - s) < tol)
    else
        return (abs(eig1 - s) < tol[k][local_idx]) || (abs(eig2 - s) < tol[k][local_conj])
    end
end

function is_resonant(cond::ConditionNumberEstimateCondition, target::Int, s::Number,
        ::Int)::Bool
    local_idx = _local_index(cond, target)
    local_idx === nothing && return false
    eig = cond.eigenvalues[local_idx]
    κ = cond.condition_numbers[local_idx]
    ρ = cond.spectral_radius
    mc = cond.max_cond
    if cond.conjugacy_map === nothing
        return abs(eig - s) * mc < ρ * κ
    else
        local_conj = cond.conjugacy_map[local_idx]
        eig_c = cond.eigenvalues[local_conj]
        κ_c = cond.condition_numbers[local_conj]
        return (abs(eig - s) * mc < ρ * κ) || (abs(eig_c - s) * mc < ρ * κ_c)
    end
end

# ======================================================================
# Private build helpers
# ======================================================================

# Compute superharmonics s_k = ⟨super_eigenvalues, α_k⟩ for all monomials.
function _superharmonics(super_eigenvalues, multiindices::MultiindexSet)
    [sum(super_eigenvalues .* mi) for mi in multiindices.exponents]
end

"""
Build the `n_int × NMON` inner resonance matrix.

`strategy` applies graph/normal-form unconditional flags; `inner_cond` (optional)
applies an eigenvalue-proximity check on the master eigenvalues.
"""
function _build_inner_matrix(
        strategy::InternalResonance,
        inner_cond::Union{Nothing, OuterResonanceCondition},
        super_eigenvalues, multiindices::MultiindexSet, n_int::Int)
    exps = multiindices.exponents
    NMON = length(exps)
    mat = falses(n_int, NMON)
    s_vec = _superharmonics(super_eigenvalues, multiindices)
    for k in 1:NMON
        mi = exps[k]
        s = s_vec[k]
        apply_internal_resonances!(mat, strategy, mi, n_int, k)
        if inner_cond !== nothing
            for r in 1:n_int
                is_resonant(inner_cond, r, s, k) && (mat[r, k] = true)
            end
        end
    end
    return mat
end

"""
Build the `n_out × NMON` outer resonance matrix using `outer_cond`.
"""
function _build_outer_matrix(
        outer_cond::OuterResonanceCondition,
        super_eigenvalues, multiindices::MultiindexSet, n_out::Int)
    exps = multiindices.exponents
    NMON = length(exps)
    mat = falses(n_out, NMON)
    s_vec = _superharmonics(super_eigenvalues, multiindices)
    for k in 1:NMON
        s = s_vec[k]
        for j in 1:n_out
            is_resonant(outer_cond, j, s, k) && (mat[j, k] = true)
        end
    end
    return mat
end

# ======================================================================
# Public constructors
# ======================================================================

"""
	resonance_set_from_graph_style(
		multiindices, master_eigenvalues, external_eigenvalues,
		outer_eigenvalues, tol)

Build a `ResonanceSet` using the **graph style**: every monomial of total degree ≥ 2 is
marked resonant with all `ROM` master modes.  Outer targets are flagged by eigenvalue
proximity `|λⱼ - s| < tol`.

- `master_eigenvalues`: ROM eigenvalues; enter `s` and are inner targets.
- `external_eigenvalues`: enter `s` only (e.g. forcing frequencies in the multiindex, since external_eigenvalues cant be targets).
- `outer_eigenvalues`: outer targets (eigenvalues not included in `master_eigenvalues`, tested for near-resonance).
  Pass `ComplexF64[]` when there are no outer targets.
"""
function resonance_set_from_graph_style(
        multiindices::MultiindexSet{NVAR},
        master_eigenvalues::AbstractVector{<:Number},
        external_eigenvalues::AbstractVector{<:Number},
        outer_eigenvalues::AbstractVector{<:Number},
        tol::Union{Float64, Vector{Vector{Float64}}}) where {NVAR}
    n_int = length(master_eigenvalues)
    n_out = length(outer_eigenvalues)
    N_EXT = NVAR - n_int
    _super = vcat(master_eigenvalues, external_eigenvalues)
    @assert length(_super) == NVAR "length(master) + length(external) ≠ NVAR"
    inner = _build_inner_matrix(GraphInternal(), nothing, _super, multiindices, n_int)
    outer = if n_out > 0
        outer_cond = EigenvalueCondition(outer_eigenvalues, tol, 1:n_out)
        _build_outer_matrix(outer_cond, _super, multiindices, n_out)
    else
        nothing
    end
    return ResonanceSet{n_int, N_EXT, BitMatrix}(multiindices, inner, outer)
end

"""
	resonance_set_from_complex_normal_form_style(
		multiindices, master_eigenvalues, tol;
		external_eigenvalues, outer_eigenvalues)

Build a `ResonanceSet` using the **complex normal form style**: inner resonances
flagged by `|λᵣ - s| < tol` for each master mode `r`; outer resonances (if any)
flagged by proximity to `outer_eigenvalues`.

Suitable for autonomous SSMs with complex conjugate reduced variables; add
`external_eigenvalues` for non-autonomous systems where the multiindex includes
forcing directions.
"""
function resonance_set_from_complex_normal_form_style(
        multiindices::MultiindexSet{NVAR},
        master_eigenvalues::AbstractVector{<:Number},
        tol::Union{Float64, Vector{Vector{Float64}}};
        external_eigenvalues::AbstractVector{<:Number} = ComplexF64[],
        outer_eigenvalues::AbstractVector{<:Number} = ComplexF64[],
        outer_tol::Union{Nothing, Float64, Vector{Vector{Float64}}} = nothing) where {NVAR}
    n_int = length(master_eigenvalues)
    n_out = length(outer_eigenvalues)
    N_EXT = NVAR - n_int
    _super = vcat(master_eigenvalues, external_eigenvalues)
    @assert length(_super) == NVAR "length(master) + length(external) ≠ NVAR"
    otol = _resolve_outer_tol(tol, outer_tol, n_out, n_int)
    inner_cond = EigenvalueCondition(master_eigenvalues, tol, 1:n_int)
    inner = _build_inner_matrix(
        NormalFormInternal(), inner_cond, _super, multiindices, n_int)
    outer = if n_out > 0
        outer_cond = EigenvalueCondition(outer_eigenvalues, otol, 1:n_out)
        _build_outer_matrix(outer_cond, _super, multiindices, n_out)
    else
        nothing
    end
    return ResonanceSet{n_int, N_EXT, BitMatrix}(multiindices, inner, outer)
end

"""
	_resolve_outer_tol(tol, outer_tol, n_out, n_int)

Pick the tolerance for the outer targets, and reject the one combination that used to be
a silent bounds error.

A per-target tolerance is read `tol[k][local_idx]` with `local_idx` **local to its own
condition** — `1:n_int` for the inner block, `1:n_out` for the outer one. The two blocks
are built from separate condition objects, so they can perfectly well carry separate
tolerances; only the public constructors' single `tol` argument tied them together, and a
per-target vector sized for `n_int` then overran whenever `n_out > n_int`.

`outer_tol === nothing` means "reuse `tol`", which is exactly right for a scalar (it
applies to any number of targets) and impossible for a per-target vector — hence the
error, which tells the caller what to pass instead of failing deep inside `is_resonant`.
"""
function _resolve_outer_tol(tol, outer_tol, n_out::Int, n_int::Int)
    n_out == 0 && return tol
    outer_tol !== nothing && return outer_tol
    tol isa Float64 && return tol
    throw(ArgumentError("""
        A per-target `tol` is indexed by the target number *within its own block*, so a
        vector sized for the $n_int inner targets cannot also serve the $n_out outer targets.
        Pass `outer_tol` as well: a scalar, or a per-monomial vector whose entries have
        $n_out elements (e.g. `[[rel * abs(λ_outer[j]) for j in 1:$n_out] for _ in 1:NMON]`).
        """))
end

"""
	resonance_set_from_real_normal_form_style(
		multiindices, master_eigenvalues, conjugacy_map, tol;
		external_eigenvalues, outer_eigenvalues)

Build a `ResonanceSet` using the **real normal form style**: like CNF but conjugate
pairs share the resonance flag — monomial `k` is resonant with target `j` when
`|λⱼ - s| < tol` OR `|λ_{conj(j)} - s| < tol`.

`conjugacy_map` has `length(master_eigenvalues) + length(outer_eigenvalues)` entries;
the first `n_int` cover inner targets, the remainder cover outer targets (re-indexed
locally).  `conjugacy_map[i]` is the local index of the conjugate of target `i`.
"""
function resonance_set_from_real_normal_form_style(
        multiindices::MultiindexSet{NVAR},
        master_eigenvalues::AbstractVector{<:Number},
        conjugacy_map::Vector{Int},
        tol::Union{Float64, Vector{Vector{Float64}}};
        external_eigenvalues::AbstractVector{<:Number} = ComplexF64[],
        outer_eigenvalues::AbstractVector{<:Number} = ComplexF64[],
        outer_tol::Union{Nothing, Float64, Vector{Vector{Float64}}} = nothing) where {NVAR}
    n_int = length(master_eigenvalues)
    n_out = length(outer_eigenvalues)
    N_EXT = NVAR - n_int
    _super = vcat(master_eigenvalues, external_eigenvalues)
    @assert length(_super) == NVAR "length(master) + length(external) ≠ NVAR"
    @assert length(conjugacy_map) == n_int + n_out "conjugacy_map length ≠ n_int + n_out"
    otol = _resolve_outer_tol(tol, outer_tol, n_out, n_int)
    inner_conj = conjugacy_map[1:n_int]
    inner_cond = RealEigenvalueCondition(master_eigenvalues, inner_conj, tol, 1:n_int)
    inner = _build_inner_matrix(
        NormalFormInternal(), inner_cond, _super, multiindices, n_int)
    outer = if n_out > 0
        # re-index outer conjugacy map entries to local 1:n_out
        outer_conj = conjugacy_map[(n_int + 1):end] .- n_int
        outer_cond = RealEigenvalueCondition(outer_eigenvalues, outer_conj, otol, 1:n_out)
        _build_outer_matrix(outer_cond, _super, multiindices, n_out)
    else
        nothing
    end
    return ResonanceSet{n_int, N_EXT, BitMatrix}(multiindices, inner, outer)
end

"""
	resonance_set_from_condition_number_estimate(
		multiindices, master_eigenvalues, spectral_radius,
		target_condition_numbers, max_cond;
		external_eigenvalues, outer_eigenvalues,
		inner_target_indices, outer_target_indices, conjugacy_map)

Build a `ResonanceSet` using a condition-number criterion:

	|λⱼ - s| * max_cond < spectral_radius * κ(λⱼ)

`target_condition_numbers` has `n_int + n_out` entries: the first `n_int` are for
the master modes, the remainder for the outer modes.

`inner_target_indices` / `outer_target_indices` restrict which rows in each sub-matrix
are populated (default: all).
"""
function resonance_set_from_condition_number_estimate(
        multiindices::MultiindexSet{NVAR},
        master_eigenvalues::AbstractVector{<:Number},
        spectral_radius::Float64,
        target_condition_numbers::Vector{Float64},
        max_cond::Float64;
        external_eigenvalues::AbstractVector{<:Number} = ComplexF64[],
        outer_eigenvalues::AbstractVector{<:Number} = ComplexF64[],
        inner_target_indices::Union{Nothing, UnitRange{Int}, Vector{Int}} = nothing,
        outer_target_indices::Union{Nothing, UnitRange{Int}, Vector{Int}} = nothing,
        conjugacy_map::Union{Nothing, Vector{Int}} = nothing) where {NVAR}
    n_int = length(master_eigenvalues)
    n_out = length(outer_eigenvalues)
    N_EXT = NVAR - n_int
    _super = vcat(master_eigenvalues, external_eigenvalues)
    @assert length(_super) == NVAR "length(master) + length(external) ≠ NVAR"
    @assert length(target_condition_numbers) == n_int + n_out
    inner_κ = target_condition_numbers[1:n_int]
    outer_κ = target_condition_numbers[(n_int + 1):end]
    eff_inner = inner_target_indices === nothing ? (1:n_int) : inner_target_indices
    inner_cond = ConditionNumberEstimateCondition(
        master_eigenvalues, spectral_radius, inner_κ, max_cond, collect(eff_inner),
        conjugacy_map)
    inner = _build_inner_matrix(
        NormalFormInternal(), inner_cond, _super, multiindices, n_int)
    outer = if n_out > 0
        eff_outer = outer_target_indices === nothing ? (1:n_out) : outer_target_indices
        outer_cond = ConditionNumberEstimateCondition(
            outer_eigenvalues, spectral_radius, outer_κ, max_cond, collect(eff_outer), nothing)
        _build_outer_matrix(outer_cond, _super, multiindices, n_out)
    else
        nothing
    end
    return ResonanceSet{n_int, N_EXT, BitMatrix}(multiindices, inner, outer)
end

"""
	resonance_set_from_graph_style(multiindices, master_eigenvalues, external_eigenvalues, outer_condition)

Advanced overload that accepts a pre-built `OuterResonanceCondition` for the outer rows.
`outer_condition.target_indices` must use **local** indices `1:n_out`.
`n_out` is inferred as `maximum(outer_condition.target_indices)` (or 0 if empty).
"""
function resonance_set_from_graph_style(
        multiindices::MultiindexSet{NVAR},
        master_eigenvalues::AbstractVector{<:Number},
        external_eigenvalues::AbstractVector{<:Number},
        outer_condition::OuterResonanceCondition) where {NVAR}
    n_int = length(master_eigenvalues)
    n_out = isempty(outer_condition.target_indices) ? 0 :
            maximum(outer_condition.target_indices)
    N_EXT = NVAR - n_int
    _super = vcat(master_eigenvalues, external_eigenvalues)
    @assert length(_super) == NVAR "length(master) + length(external) ≠ NVAR"
    inner = _build_inner_matrix(GraphInternal(), nothing, _super, multiindices, n_int)
    outer = n_out > 0 ?
            _build_outer_matrix(outer_condition, _super, multiindices, n_out) : nothing
    return ResonanceSet{n_int, N_EXT, BitMatrix}(multiindices, inner, outer)
end

function Base.show(io::IO, rs::ResonanceSet{ROM, N_EXT, M}) where {ROM, N_EXT, M}
    n_out = rs.outer_resonances === nothing ? 0 : size(rs.outer_resonances, 1)
    print(io, "ResonanceSet{ROM=", ROM, ",N_EXT=", N_EXT, "} with ",
        length(rs.multiindices), " multiindices, ",
        count(rs.inner_resonances), " inner resonances",
        n_out > 0 ? ", $(count(rs.outer_resonances)) outer resonances" : "")
end

# =============================================================================
# ResonanceConfig — every resonance knob in one validated object
# =============================================================================

"""
	ResonanceConfig(; style, tol, tol_relative, conjugacy_map, outer_targets, warn_outer)

Every knob that controls resonance detection, in one place.

Previously these were loose keyword arguments spread across `parametrise`
(`resonance`, `resonance_tol`, `conjugacy_map`) and re-implemented again in
MORFEFerrite's structural backend (`resonance_tol`, `resonance_tol_rel`, plus a separate
off-manifold warning pass). Gathering them means there is one thing to read, and
combinations that cannot work are rejected where they are written rather than deep inside
the solve.

# Fields

- `style::Symbol = :graph` — `:graph`, `:complex_normal_form`, or `:real_normal_form`.
- `tol::Union{Nothing, Real, AbstractVector} = nothing` — absolute detuning threshold.
  `nothing` means "not specified" and resolves to the style's default.
- `tol_relative::Union{Nothing, Real} = nothing` — when set, replaces `tol` by the
  per-target threshold `tol_relative * |λⱼ|`, judging each target on its own frequency
  scale. This is the physically meaningful criterion when the master modes span decades.
- `conjugacy_map::Union{Nothing, Vector{Int}} = nothing` — required by
  `:real_normal_form`, and an error with any other style rather than silently ignored.
- `outer_targets::Bool = false` — also flag resonances against the non-master
  eigenvalues, populating the `outer_resonances` block. Diagnostic: the solve reads only
  the inner block.
- `warn_outer::Bool = true` — warn when a monomial is near-resonant with an off-manifold
  mode, whose direction is then solved through a near-singular operator.
- `eigenvalue_projection::Symbol = :full` — which part of the master eigenvalues detection
  compares. `:full` uses them as they are; `:imaginary_part_only` replaces `λ` by
  `i·Im(λ)`, so near-resonance is judged on **frequency alone** and the growth rate is
  ignored. See below.

## `eigenvalue_projection`

For an oscillatory reduction about a marginally stable state — a Hopf normal form, say —
what makes a monomial resonant is that its frequency combination `⟨Im λ, α⟩` lands on a
target frequency. The growth rate `Re λ` is small, and it *varies with the continuation
parameter*, so letting it enter the detuning makes the flag pattern depend on where in a
parameter sweep you happen to be. `:imaginary_part_only` removes that dependence.

It applies to the **master eigenvalues only** — the external and outer eigenvalues are
compared as they are. Note it also changes what `tol_relative` means, and helpfully so:
`tol_relative * |λ|` becomes `tol_relative * |Im λ|`, a tolerance measured on the same
frequency scale the detection now uses.

The default is `:full`, so nothing changes unless it is asked for.

## Why `tol` defaults to `nothing` rather than `1e-2`

The guards below fire on *explicitly set* values. With a numeric default there would be no
way to tell "the user asked for this tolerance" from "nobody said anything", so a plain
`ResonanceConfig()` would emit a spurious "tolerance unused" notice on every run — and
guards that cry wolf get ignored. `nothing` is the only honest "unspecified".
"""
struct ResonanceConfig
    style::Symbol
    tol::Union{Nothing, Real, AbstractVector}
    tol_relative::Union{Nothing, Real}
    conjugacy_map::Union{Nothing, Vector{Int}}
    outer_targets::Bool
    warn_outer::Bool
    eigenvalue_projection::Symbol

    function ResonanceConfig(; style::Symbol = :graph,
            tol::Union{Nothing, Real, AbstractVector} = nothing,
            tol_relative::Union{Nothing, Real} = nothing,
            conjugacy_map::Union{Nothing, Vector{Int}} = nothing,
            outer_targets::Bool = false,
            warn_outer::Bool = true,
            eigenvalue_projection::Symbol = :full)
        style in (:graph, :complex_normal_form, :real_normal_form) || throw(ArgumentError(
            "unknown resonance style :$style; choose :graph, :complex_normal_form or " *
            ":real_normal_form"))
        eigenvalue_projection in (:full, :imaginary_part_only) || throw(ArgumentError(
            "unknown eigenvalue_projection :$eigenvalue_projection; choose :full or " *
            ":imaginary_part_only"))
        if style === :real_normal_form && conjugacy_map === nothing
            throw(ArgumentError(
                ":real_normal_form pairs conjugate targets, so it requires `conjugacy_map`"))
        end
        if style !== :real_normal_form && conjugacy_map !== nothing
            throw(ArgumentError(
                "`conjugacy_map` is only read by :real_normal_form, but style is :$style. " *
                "It would be silently ignored, so it is rejected instead."))
        end
        if tol isa Real && tol <= 0
            throw(ArgumentError("resonance tol must be positive, got $tol"))
        end
        if tol_relative !== nothing && tol_relative <= 0
            throw(ArgumentError("tol_relative must be positive, got $tol_relative"))
        end
        # NOTE: `tol_relative` together with `outer_targets` is deliberately NOT rejected.
        # It used to be impossible because one per-target tolerance vector was handed to
        # both target families; `outer_tol` now sizes them separately, so the combination
        # is both legal and the physically obvious reading of "relative detuning".
        return new(style, tol, tol_relative, conjugacy_map, outer_targets, warn_outer,
            eigenvalue_projection)
    end
end

function Base.show(io::IO, c::ResonanceConfig)
    print(io, "ResonanceConfig(style = :", c.style)
    c.tol === nothing || print(io, ", tol = ", c.tol)
    c.tol_relative === nothing || print(io, ", tol_relative = ", c.tol_relative)
    c.conjugacy_map === nothing || print(io, ", conjugacy_map = ", c.conjugacy_map)
    c.outer_targets && print(io, ", outer_targets = true")
    c.warn_outer || print(io, ", warn_outer = false")
    c.eigenvalue_projection === :full ||
        print(io, ", eigenvalue_projection = :", c.eigenvalue_projection)
    print(io, ")")
end

"""
	project_eigenvalues(λ, projection::Symbol) -> Vector

Apply a [`ResonanceConfig`](@ref) eigenvalue projection: `:full` is the identity,
returning `Vector{ComplexF64}`; `:imaginary_part_only` returns the imaginary parts
`Im(λ)` as `Vector{Float64}`.

A projected eigenvalue `i·Im(λ)` is carried by the real number `Im(λ)` — the two are
isomorphic for this purpose, because for real `a, b`

    |i·a − i·b| = |a − b|

and the whole of detection is `abs(eig - s)`. So carrying the frequencies as `Float64`
reproduces the pure-imaginary complex computation *exactly*, with no `i·Im`
reconstruction and one code path serving both modes.

**Project every family the same way** — master, external and outer. Mixing a projected
family with an unprojected one promotes the superharmonic `s = ⟨λ, α⟩` back to complex and
reinstates the growth rates through the back door, which is precisely what the projection
was asked to remove.

Always returns a plain `Vector`. The master eigenvalues arrive as a static vector, and a
comprehension over one stays static — a `SizedVector` that the `resonance_set_from_*`
signatures reject. The loops below are the same reason the caller used
`Vector{ComplexF64}(…)` rather than `collect`.
"""
function project_eigenvalues(λ::AbstractVector, projection::Symbol)
    projection === :full && return Vector{ComplexF64}(λ)
    if projection === :imaginary_part_only
        out = Vector{Float64}(undef, length(λ))
        @inbounds for (i, l) in enumerate(λ)
            out[i] = imag(l)
        end
        return out
    end
    throw(ArgumentError("unknown eigenvalue_projection :$projection"))
end

# Style default for an unspecified tolerance.
_default_tol(style::Symbol) = style === :graph ? 0.0 : 1e-2

"""
	_first_close_pair(eigenvalues, tol, n) -> (i, j, gap)

The first pair of eigenvalues separated by no more than `tol`, or `(0, 0, 0.0)` if there is
none. Allocation-free, and stops at the first hit.

Split out from the caller so the `@info` that reports it sits *outside* the loop: string
interpolation inside a loop body captures the loop variables and boxes them on every
iteration, even when the branch is not taken and nothing is logged.
"""
function _first_close_pair(eigenvalues::AbstractVector, tol::Float64, n::Int)
    for i in 1:n, j in (i + 1):n

        gap = abs(eigenvalues[i] - eigenvalues[j])
        if gap > 0 && tol >= gap
            return i, j, gap
        end
    end
    return 0, 0, 0.0
end

"""
	resolve_tolerances(config, master_eigenvalues, outer_eigenvalues, n_monomials)
		-> (inner_tol, outer_tol)

Turn a [`ResonanceConfig`](@ref) into the two correctly-sized tolerance objects the
resonance-set constructors want, and emit `@info` for settings that will have no effect.

With `tol_relative`, each family is sized for its **own** target count:

	inner[k][r] = tol_relative * |λ_master[r]|      length n_int
	outer[k][j] = tol_relative * |λ_outer[j]|       length n_out

which is why the two can now be combined at all.
"""
function resolve_tolerances(config::ResonanceConfig,
        master_eigenvalues::AbstractVector, outer_eigenvalues::AbstractVector,
        n_monomials::Int)
    n_int = length(master_eigenvalues)
    n_out = length(outer_eigenvalues)

    if config.style === :graph &&
       (config.tol !== nothing || config.tol_relative !== nothing)
        @info "ResonanceConfig: style = :graph marks every monomial of degree ≥ 2 as " *
              "resonant with all master modes, so the tolerance you set is not used for " *
              "the inner block. It still applies to outer targets when `outer_targets = true`."
    end
    if config.outer_targets && n_out == 0
        @info "ResonanceConfig: `outer_targets = true` but the spectral data carries no " *
              "outer eigenvalues, so no off-manifold targets can be flagged."
    end

    if config.tol_relative !== nothing
        rel = Float64(config.tol_relative)
        inner = [[rel * abs(master_eigenvalues[r]) for r in 1:n_int] for _ in 1:n_monomials]
        outer = [[rel * abs(outer_eigenvalues[j]) for j in 1:n_out] for _ in 1:n_monomials]
        return inner, outer
    end

    tol = config.tol === nothing ? _default_tol(config.style) : config.tol
    if tol isa Real
        t = Float64(tol)
        # A tolerance wider than the gaps between master eigenvalues makes essentially
        # every monomial read as resonant, which is rarely what anyone means.
        # The guard asks a yes/no question — "does ANY pair of master eigenvalues sit within
        # `t` of each other?" — so it neither materialises the ROM(ROM-1)/2 distances nor
        # reduces them to a minimum. The search is a separate function, and the logging
        # happens outside it: an `@info` in the loop body interpolates the loop variables,
        # which boxes them on every iteration even when nothing is logged.
        i, j, gap = _first_close_pair(master_eigenvalues, t, n_int)
        if i != 0
            @info "ResonanceConfig: tol = $t is at least the spacing between master " *
                  "eigenvalues $i and $j ($gap), so nearly every monomial will be " *
                  "flagged resonant. Did you mean a smaller tolerance, or `tol_relative`?"
        end
        return t, t
    end
    return tol, nothing   # explicit per-target vector: outer_tol must be supplied upstream
end

"""
	build_resonance_set(model, mset, spectral::SpectralData, config::ResonanceConfig)
		-> ResonanceSet

Build the resonance set from a [`SpectralData`](@ref) bundle and a
[`ResonanceConfig`](@ref) — the single entry point for resonance construction.

Reads the master and outer eigenvalues off `spectral` (replacing the eigenproblem plus
master-mask plumbing), resolves the config's tolerances into correctly-sized inner and
outer objects via [`resolve_tolerances`](@ref), and warns about off-manifold
near-resonances when `config.warn_outer` is set.
"""
function build_resonance_set(model::NthOrderModel, mset::MultiindexSet,
        spectral, config::ResonanceConfig)
    # `Vector{ComplexF64}`, not `collect`: the master eigenvalues are an `SVector`, and
    # `collect` would give a `SizedVector` that the constructors' signatures reject.
    #
    # The projection is applied HERE, before `resolve_tolerances`, so it governs both what
    # detection compares and what `tol_relative` is relative to — with
    # `:imaginary_part_only` that makes the threshold `tol_relative * |Im λ|`, measured on
    # the same frequency scale the detection now uses. All three eigenvalue families are
    # projected: master, external and outer, so the detection compares frequencies alone
    # everywhere.
    master_eigs = project_eigenvalues(spectral.master.eigenvalues,
        config.eigenvalue_projection)
    all_outer = project_eigenvalues(spectral.outer.eigenvalues,
        config.eigenvalue_projection)
    # The EMPTY cases must carry the projected element type too, or a `Float64[]` family
    # meets a `ComplexF64[]` one and the superharmonic promotes back to complex — which
    # would silently undo the projection for exactly the models that have no external
    # system or no outer targets.
    T = eltype(master_eigs)
    external_eigs = model.external_system === nothing ? T[] :
                    project_eigenvalues(Vector(model.external_system.eigenvalues),
                        config.eigenvalue_projection)

    # Outer eigenvalues serve two distinct purposes: populating the diagnostic
    # `outer_resonances` block (opt-in), and driving the off-manifold warning (on by
    # default). Only the first puts them in the returned set.
    target_outer = config.outer_targets ? all_outer : T[]
    inner_tol, outer_tol = resolve_tolerances(
        config, master_eigs, target_outer, length(mset))

    rset = if config.style === :graph
        resonance_set_from_graph_style(
            mset, master_eigs, external_eigs, target_outer,
            outer_tol === nothing ? inner_tol : outer_tol)
    elseif config.style === :complex_normal_form
        resonance_set_from_complex_normal_form_style(
            mset, master_eigs, inner_tol;
            external_eigenvalues = external_eigs,
            outer_eigenvalues = target_outer, outer_tol = outer_tol)
    else
        resonance_set_from_real_normal_form_style(
            mset, master_eigs, config.conjugacy_map, inner_tol;
            external_eigenvalues = external_eigs,
            outer_eigenvalues = target_outer, outer_tol = outer_tol)
    end

    config.warn_outer && _warn_outer_resonances(
        mset, master_eigs, all_outer, external_eigs, config, spectral)
    return rset
end

"""
	_warn_outer_resonances(mset, master_eigs, outer_eigs, external_eigs, config, spectral)

Warn when a monomial is near-resonant with a physical mode that is *not* on the manifold.

That direction is then solved through a near-singular operator, so the ROM loses accuracy
there regardless of how the load is shaped: `solve_single_monomial!` builds its operator
from `s = ⟨λ, α⟩` alone, which makes the conditioning independent of the right-hand side.
Rounding injects a component along the near-null direction and `1/(λ_s - s)` amplifies it.

Runs for autonomous and forced models alike — a monomial built purely from master
coordinates that lands on an off-manifold eigenvalue is exactly as near-singular as a
forced one, and that its cause is the chosen master set makes it more worth surfacing.

## Why this does not build a `ResonanceSet`

It used to, and that made an on-by-default diagnostic cost `O(NMON × n_outer)` with a large
constant: a whole second set including an `n_int × NMON` inner block that was discarded,
`_superharmonics` allocating a temporary per monomial *twice*, and `_local_index`'s
`findfirst` inside `is_resonant` turning the outer build into `O(NMON × n_outer²)`.

The criterion needs none of that — it is one distance test per (monomial, outer target) — so
it is written out directly. A run that flags nothing now costs a **constant 64 bytes**
(measured, unchanged from `|mset| = 20` to `54`) against the 246 400 the probe cost at
`|mset| = 35` with 58 outer modes. The test itself is unchanged: `|λ_outer[j] - s| < tol`,
the same `EigenvalueCondition` comparison the probe applied, for every style. Reusing the
already-built outer block instead was rejected deliberately: under `:real_normal_form` that
block ORs each target with its conjugate, which would silently change what gets reported.
"""
function _warn_outer_resonances(mset::MultiindexSet, master_eigs, outer_eigs,
        external_eigs, config::ResonanceConfig, spectral)
    isempty(outer_eigs) && return nothing
    tol = _outer_warn_tolerance(config, outer_eigs)
    tol === nothing && return nothing
    hits = _scan_outer_resonances(mset, master_eigs, outer_eigs, external_eigs, tol)
    hits === nothing && return nothing
    return _warn_flagged_outer_modes(mset, outer_eigs, hits, spectral)
end

"""
	_outer_warn_tolerance(config, outer_eigenvalues)
		-> Union{Nothing, Float64, Vector{Float64}}

The threshold the off-manifold scan compares `|λ_outer[j] - s|` against, resolved *without*
going through [`resolve_tolerances`](@ref).

That function is wrong for this caller three times over: it re-emits every `@info` guard a
second time, `tol_relative` makes it build `n_monomials` identical tolerance rows where the
scan wants one, and for an explicitly per-target `tol` it returns `nothing` for the outer
family — a vector sized for the *inner* targets cannot be indexed by an outer target number.
That last case used to reach `_resolve_outer_tol` and **throw**, so a per-target tolerance
combined with the default `warn_outer = true` aborted the solve. It now returns `nothing`,
and the scan is skipped with a notice rather than taking the run down with it.
"""
function _outer_warn_tolerance(config::ResonanceConfig, outer_eigenvalues::AbstractVector)
    if config.tol_relative !== nothing
        rel = Float64(config.tol_relative)
        return [rel * abs(λ) for λ in outer_eigenvalues]
    end
    tol = config.tol === nothing ? _default_tol(config.style) : config.tol
    tol isa Real && return Float64(tol)
    @info "ResonanceConfig: `tol` is a per-target vector sized for the inner targets, so " *
          "it cannot be indexed by an outer target. The off-manifold near-resonance scan " *
          "is skipped — pass a scalar `tol` or `tol_relative` to re-enable it."
    return nothing
end

@inline _tol_at(tol::Float64, ::Int) = tol
@inline _tol_at(tol::Vector{Float64}, j::Int) = tol[j]

# s = ⟨λ, α⟩ for one monomial, accumulated in place: no `vcat` of the master and external
# eigenvalues, and no per-monomial temporary. `α` has NVAR = n_int + N_EXT entries, the
# master ones first, exactly as `_superharmonics` contracts them.
@inline function _superharmonic(master_eigs, external_eigs, α)
    s = zero(ComplexF64)
    n_int = length(master_eigs)
    @inbounds for i in 1:n_int
        s += master_eigs[i] * α[i]
    end
    @inbounds for i in eachindex(external_eigs)
        s += external_eigs[i] * α[n_int + i]
    end
    return s
end

# Every (outer target, monomial) pair that is near-resonant, or `nothing` when there are
# none. The result vector is created on the FIRST hit, so a quiet run allocates a constant
# handful of bytes whatever the size of the monomial set or the outer spectrum.
function _scan_outer_resonances(
        mset::MultiindexSet, master_eigs, outer_eigs, external_eigs,
        tol::Union{Float64, Vector{Float64}})
    hits = nothing
    exps = mset.exponents
    @inbounds for k in eachindex(exps)
        s = _superharmonic(master_eigs, external_eigs, exps[k])
        for j in eachindex(outer_eigs)
            abs(outer_eigs[j] - s) < _tol_at(tol, j) || continue
            hits === nothing && (hits = Tuple{Int, Int}[])
            push!(hits, (j, k))
        end
    end
    return hits
end

# ── Reporting ───────────────────────────────────────────────────────────────
#
# Conjugate partners are one physical mode, so they warn once, not twice. The pairing comes
# from the spectral bundle's own involution; `spectral` is duck-typed here, so anything that
# is not `SpectralData` falls back to one warning per target.

_outer_pairing(sd::SpectralData) = outer_conjugate_permutation(sd)
_outer_pairing(::Any) = nothing

_outer_entry(sd::SpectralData, j::Int) = indices(outer_bundle(sd))[j]
_outer_entry(::Any, j::Int) = j

_outer_mode_number(sd::SpectralData, entry::Int) = physical_mode(sd, entry)
_outer_mode_number(::Any, ::Int) = nothing

# Both the mode's description and how to name it in the remedy: a pair is "mode pair p"
# with both spectrum entries and λ written as a ± bi; a self-paired (real) or unpaired
# target is singular.
function _outer_mode_description(spectral, rep::Int, partner::Int, outer_eigs)
    entry = _outer_entry(spectral, rep)
    p = _outer_mode_number(spectral, entry)
    λ = outer_eigs[rep]
    if partner != rep
        entries = sort!([entry, _outer_entry(spectral, partner)])
        # Under `:imaginary_part_only` the targets are REAL frequencies; rendering one
        # as `re ± im·i` would print a bare `ω + 0.0i` and invite the reader to think
        # the growth rate had been measured and found to be zero.
        lam = λ isa Real ? @sprintf("±%.3ei (frequency only)", abs(λ)) :
              @sprintf("%.3e ± %.3ei", real(λ), abs(imag(λ)))
        head = p === nothing ? "an outer conjugate mode pair" :
               "outer physical mode pair $p"
        subject = p === nothing ? "those modes" : "mode $p"
        return ("$head (spectrum entries $(join(entries, ", ")); λ = $lam)", subject)
    end
    lam = λ isa Real ? @sprintf("%.3ei (frequency only)", λ) :
          @sprintf("%.3e %s %.3ei", real(λ), imag(λ) < 0 ? "-" : "+", abs(imag(λ)))
    head = p === nothing ? "an outer mode" : "outer physical mode $p"
    subject = p === nothing ? "that mode" : "mode $p"
    return ("$head (spectrum entry $entry; λ = $lam)", subject)
end

function _warn_flagged_outer_modes(mset::MultiindexSet, outer_eigs,
        hits::Vector{Tuple{Int, Int}}, spectral)
    σ = _outer_pairing(spectral)
    paired = !(σ === nothing || isempty(σ))
    # Group by the lower-numbered member of each conjugate pair, so both conjugates of a
    # flagged mode land in the same bucket however the eigensolver ordered them.
    groups = Dict{Int, Vector{Int}}()
    for (j, k) in hits
        rep = paired ? min(j, σ[j]) : j
        push!(get!(() -> Int[], groups, rep), k)
    end
    for rep in sort!(collect(keys(groups)))
        partner = paired ? σ[rep] : rep
        cols = sort!(unique!(groups[rep]))
        monomials = join((string(Tuple(mset.exponents[c])) for c in cols), ", ")
        description, subject = _outer_mode_description(spectral, rep, partner, outer_eigs)
        @warn """
          Monomials are near-resonant with $description. That mode is not on the manifold, \
          so its direction is solved through a near-singular operator and the ROM will lose \
          accuracy there regardless of how the load is shaped. Offending monomial exponents: \
          $monomials. Add $subject to the master set, detune the forcing, or add damping."""
    end
    return nothing
end

end # module
