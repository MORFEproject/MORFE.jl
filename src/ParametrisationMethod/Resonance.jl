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

- **`outer_eigenvalues`** (optional): additional resonance targets (e.g. forcing
  frequencies tested for near-resonance).  They define the rows of `outer_resonances`
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

export ResonanceSet,
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
	NormalFormInternal

# ======================================================================
# ResonanceSet
# ======================================================================

"""
	ResonanceSet{ROM, N_EXT, M}

Boolean look-up table recording which monomials are resonant with which master-mode
or outer-mode targets.

- `multiindices`: the `MultiindexSet` over which resonances are defined (NVAR = ROM + N_EXT).
- `inner_resonances`: `ROM × NMON` `BitMatrix`; row `r`, column `k` is `true` when
  monomial `k` is resonant with master mode `r`.
- `outer_resonances`: `n_out × NMON` `BitMatrix` for outer (forcing) targets, or
  `nothing` when there are no outer targets.

Type parameters: `ROM` = number of master modes, `N_EXT` = external system size,
`M` = matrix type (typically `BitMatrix`).

Use one of the `resonance_set_from_*` constructors rather than building this directly.
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
		rs.outer_resonances[target-ROM, idx] = value
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
		return rs.outer_resonances[target-ROM, idx]
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
	return findall(rs.outer_resonances[target-ROM, :])
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

- `eigenvalues`: target eigenvalues (local indexing).
- `tol`: scalar tolerance, or `Vector{Vector{Float64}}` for per-monomial per-target.
- `target_indices`: local target indices this condition applies to (typically `1:n`).
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

- `conjugacy_map`: local index map; `conjugacy_map[i]` is the local index of the
  conjugate of eigenvalue `i`.
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

- `spectral_radius`: spectral radius of the full-order system.
- `condition_numbers`: per-target eigenvalue condition numbers `κ(λⱼ)`.
- `max_cond`: maximum acceptable condition number for the cohomological operator.
- `conjugacy_map`: optional local conjugacy map.
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

@inline _local_index(cond::OuterResonanceCondition, target::Int) =
	findfirst(==(target), cond.target_indices)

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
- `outer_eigenvalues`: outer targets (e.g. forcing eigenvalues tested for near-resonance).
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
	outer_eigenvalues::Vector{ComplexF64} = ComplexF64[]) where {NVAR}
	n_int = length(master_eigenvalues)
	n_out = length(outer_eigenvalues)
	N_EXT = NVAR - n_int
	_super = vcat(master_eigenvalues, external_eigenvalues)
	@assert length(_super) == NVAR "length(master) + length(external) ≠ NVAR"
	inner_cond = EigenvalueCondition(master_eigenvalues, tol, 1:n_int)
	inner = _build_inner_matrix(NormalFormInternal(), inner_cond, _super, multiindices, n_int)
	outer = if n_out > 0
		outer_cond = EigenvalueCondition(outer_eigenvalues, tol, 1:n_out)
		_build_outer_matrix(outer_cond, _super, multiindices, n_out)
	else
		nothing
	end
	return ResonanceSet{n_int, N_EXT, BitMatrix}(multiindices, inner, outer)
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
	outer_eigenvalues::Vector{ComplexF64} = ComplexF64[]) where {NVAR}
	n_int = length(master_eigenvalues)
	n_out = length(outer_eigenvalues)
	N_EXT = NVAR - n_int
	_super = vcat(master_eigenvalues, external_eigenvalues)
	@assert length(_super) == NVAR "length(master) + length(external) ≠ NVAR"
	@assert length(conjugacy_map) == n_int + n_out "conjugacy_map length ≠ n_int + n_out"
	inner_conj = conjugacy_map[1:n_int]
	inner_cond = RealEigenvalueCondition(master_eigenvalues, inner_conj, tol, 1:n_int)
	inner = _build_inner_matrix(NormalFormInternal(), inner_cond, _super, multiindices, n_int)
	outer = if n_out > 0
		# re-index outer conjugacy map entries to local 1:n_out
		outer_conj = conjugacy_map[(n_int+1):end] .- n_int
		outer_cond = RealEigenvalueCondition(outer_eigenvalues, outer_conj, tol, 1:n_out)
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
	outer_κ = target_condition_numbers[(n_int+1):end]
	eff_inner = inner_target_indices === nothing ? (1:n_int) : inner_target_indices
	inner_cond = ConditionNumberEstimateCondition(
		master_eigenvalues, spectral_radius, inner_κ, max_cond, collect(eff_inner),
		conjugacy_map)
	inner = _build_inner_matrix(NormalFormInternal(), inner_cond, _super, multiindices, n_int)
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

end # module
