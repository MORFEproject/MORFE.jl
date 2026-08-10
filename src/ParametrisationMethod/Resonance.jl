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

using ..Multiindices: MultiindexSet, find_in_set
using ..FullOrderModel: NDOrderModel
using ..SpectralDecomposition: Spectrum
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
`is_resonant(cond, target::Int, s::ComplexF64, k::Int) -> Bool`.
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
    eigenvalues::Vector{ComplexF64}
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
    eigenvalues::Vector{ComplexF64}
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
    eigenvalues::Vector{ComplexF64}
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

function is_resonant(cond::EigenvalueCondition, target::Int, s::ComplexF64, k::Int)::Bool
    local_idx = _local_index(cond, target)
    local_idx === nothing && return false
    eig = cond.eigenvalues[local_idx]
    tol = cond.tol
    return tol isa Float64 ? abs(eig - s) < tol : abs(eig - s) < tol[k][local_idx]
end

function is_resonant(cond::RealEigenvalueCondition, target::Int, s::ComplexF64, k::Int)::Bool
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

function is_resonant(cond::ConditionNumberEstimateCondition, target::Int, s::ComplexF64,
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
        master_eigenvalues::Vector{ComplexF64},
        external_eigenvalues::Vector{ComplexF64},
        outer_eigenvalues::Vector{ComplexF64},
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
        master_eigenvalues::Vector{ComplexF64},
        tol::Union{Float64, Vector{Vector{Float64}}};
        external_eigenvalues::Vector{ComplexF64} = ComplexF64[],
        outer_eigenvalues::Vector{ComplexF64} = ComplexF64[],
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
        master_eigenvalues::Vector{ComplexF64},
        conjugacy_map::Vector{Int},
        tol::Union{Float64, Vector{Vector{Float64}}};
        external_eigenvalues::Vector{ComplexF64} = ComplexF64[],
        outer_eigenvalues::Vector{ComplexF64} = ComplexF64[],
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
        master_eigenvalues::Vector{ComplexF64},
        spectral_radius::Float64,
        target_condition_numbers::Vector{Float64},
        max_cond::Float64;
        external_eigenvalues::Vector{ComplexF64} = ComplexF64[],
        outer_eigenvalues::Vector{ComplexF64} = ComplexF64[],
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
        master_eigenvalues::Vector{ComplexF64},
        external_eigenvalues::Vector{ComplexF64},
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

"""
	_check_external_eigenvalues(supplied, sys)

Reject an explicitly-supplied `external_eigenvalues` vector that disagrees with the external
system's own eigenvalues, when the system was re-based.

Resonance detection contracts these against multiindex components **position by position**
(`_superharmonics`), so a permuted or stale vector does not fail — it silently detects the
wrong resonances and yields a wrong ROM.  A change of external coordinates re-orders the
spectrum into the new basis, which is exactly what makes a hand-written literal go stale.

Only checked when `external_basis(sys) !== nothing`, so no system that was left in its own
coordinates can trip it.
"""
function _check_external_eigenvalues(supplied, sys)
    sys === nothing && return nothing
    external_basis(sys) === nothing && return nothing
    actual = Vector(sys.eigenvalues)
    length(supplied) == length(actual) && all(isapprox.(supplied, actual)) && return nothing
    throw(ArgumentError("""
       `external_eigenvalues` was given as $(supplied), but the external system's \
       eigenvalues are $(actual).
       The external system was re-based (its linear matrix was not upper triangular), so \
       its spectrum is now expressed in the new coordinates and an explicit vector written \
       for the original ones is stale.  Resonance detection contracts these against \
       multiindex components position by position, so a stale vector silently detects the \
       wrong resonances.
       Drop the `external_eigenvalues` argument — the default reads them from the model.
       """))
end

"""
	build_resonance_set(model, style, mset, eigenproblem, tol, conjugacy_map;
	                    external_eigenvalues = nothing)

Build a `ResonanceSet` from a solved `Spectrum` according to the chosen
parametrisation `style`. Accepted styles:
- `:graph`
- `:complex_normal_form`
- `:real_normal_form` (requires `conjugacy_map`)

`external_eigenvalues` overrides the eigenvalues of the external system used
in resonance detection (default: taken from `model.external_system`).
"""
function build_resonance_set(
        model::NDOrderModel,
        style::Symbol,
        mset::MultiindexSet,
        eigenproblem::Spectrum,
        tol::Float64,
        conjugacy_map::Union{Nothing, Vector{Int}};
        external_eigenvalues::Union{Nothing, Vector{ComplexF64}} = nothing
)
    master_mask = eigenproblem.master_modes
    outer_mask = .!eigenproblem.master_modes
    master_eigenvalues = eigenproblem.eigenvalues[master_mask]
    outer_eigenvalues = eigenproblem.eigenvalues[outer_mask]
    if external_eigenvalues === nothing
        external_eigenvalues = model.external_system === nothing ? ComplexF64[] :
                               Vector(model.external_system.eigenvalues)
    else
        _check_external_eigenvalues(external_eigenvalues, model.external_system)
    end

    if style === :graph
        return resonance_set_from_graph_style(
            mset, master_eigenvalues, external_eigenvalues, outer_eigenvalues, tol)

    elseif style === :complex_normal_form
        return resonance_set_from_complex_normal_form_style(
            mset, master_eigenvalues, tol;
            external_eigenvalues, outer_eigenvalues)

    elseif style === :real_normal_form
        @assert !isnothing(conjugacy_map) ":real_normal_form requires conjugacy_map to be set"
        return resonance_set_from_real_normal_form_style(
            mset, master_eigenvalues, conjugacy_map, tol;
            external_eigenvalues, outer_eigenvalues)
    else
        throw(ArgumentError("Unknown resonance_style :$style. Choose :graph or :complex_normal_form"))
    end
    #TODO resonance_set_from_condition_number_estimate
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

    function ResonanceConfig(; style::Symbol = :graph,
            tol::Union{Nothing, Real, AbstractVector} = nothing,
            tol_relative::Union{Nothing, Real} = nothing,
            conjugacy_map::Union{Nothing, Vector{Int}} = nothing,
            outer_targets::Bool = false,
            warn_outer::Bool = true)
        style in (:graph, :complex_normal_form, :real_normal_form) || throw(ArgumentError(
            "unknown resonance style :$style; choose :graph, :complex_normal_form or " *
            ":real_normal_form"))
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
        return new(style, tol, tol_relative, conjugacy_map, outer_targets, warn_outer)
    end
end

function Base.show(io::IO, c::ResonanceConfig)
    print(io, "ResonanceConfig(style = :", c.style)
    c.tol === nothing || print(io, ", tol = ", c.tol)
    c.tol_relative === nothing || print(io, ", tol_relative = ", c.tol_relative)
    c.conjugacy_map === nothing || print(io, ", conjugacy_map = ", c.conjugacy_map)
    c.outer_targets && print(io, ", outer_targets = true")
    c.warn_outer || print(io, ", warn_outer = false")
    print(io, ")")
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
function build_resonance_set(model::NDOrderModel, mset::MultiindexSet,
        spectral, config::ResonanceConfig)
    # `Vector{ComplexF64}`, not `collect`: the master eigenvalues are an `SVector`, and
    # `collect` would give a `SizedVector` that the constructors' signatures reject.
    master_eigs = Vector{ComplexF64}(spectral.master.eigenvalues)
    all_outer = Vector{ComplexF64}(spectral.outer.eigenvalues)
    external_eigs = model.external_system === nothing ? ComplexF64[] :
                    Vector(model.external_system.eigenvalues)

    # Outer eigenvalues serve two distinct purposes: populating the diagnostic
    # `outer_resonances` block (opt-in), and driving the off-manifold warning (on by
    # default). Only the first puts them in the returned set.
    target_outer = config.outer_targets ? all_outer : ComplexF64[]
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
        mset, master_eigs, all_outer, external_eigs, config)
    return rset
end

"""
	_warn_outer_resonances(mset, master_eigs, outer_eigs, external_eigs, config)

Warn when a monomial is near-resonant with a physical mode that is *not* on the manifold.

That direction is then solved through a near-singular operator, so the ROM loses accuracy
there regardless of how the load is shaped: `solve_single_monomial!` builds its operator
from `s = ⟨λ, α⟩` alone, which makes the conditioning independent of the right-hand side.
Rounding injects a component along the near-null direction and `1/(λ_s - s)` amplifies it.

Runs for autonomous and forced models alike — a monomial built purely from master
coordinates that lands on an off-manifold eigenvalue is exactly as near-singular as a
forced one, and that its cause is the chosen master set makes it more worth surfacing.

Diagnostic only: this builds its own resonance set and discards it.
"""
function _warn_outer_resonances(mset::MultiindexSet, master_eigs, outer_eigs,
        external_eigs, config::ResonanceConfig)
    isempty(outer_eigs) && return nothing
    inner_tol, outer_tol = resolve_tolerances(config, master_eigs, outer_eigs, length(mset))
    probe = resonance_set_from_complex_normal_form_style(
        mset, master_eigs, inner_tol;
        external_eigenvalues = external_eigs, outer_eigenvalues = outer_eigs,
        outer_tol = outer_tol)
    n_int = length(master_eigs)
    flagged = Dict{Int, Vector{Int}}()
    for j in eachindex(outer_eigs)
        cols = resonant_multiindices(probe, n_int + j)
        isempty(cols) || (flagged[j] = cols)
    end
    isempty(flagged) && return nothing
    for j in sort!(collect(keys(flagged)))
        monomials = join([string(Tuple(mset.exponents[c])) for c in sort!(flagged[j])], ", ")
        @warn """
        Monomials are near-resonant with a non-master eigenvalue \
        (λ = $(outer_eigs[j])). That mode is not on the manifold, so its direction is \
        solved through a near-singular operator and the ROM will lose accuracy there \
        regardless of how the load is shaped. Offending monomial exponents: $monomials. \
        Add that mode to the master set, detune the forcing, or add damping."""
    end
    return nothing
end

end # module
