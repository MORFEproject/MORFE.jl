"""
Module `Resonance` — resonance detection for the parametrisation method.

## Choosing a resonance style

Four constructors are provided; choose based on the problem type:

- **`resonance_set_from_graph_style`** (default for non-autonomous SSMs):
  Detects internal resonances relative to the master-mode eigenvalue combination
  `s = Σ kᵢ λᵢ`, plus autonomous/forced external resonances.  Use for
  mechanical systems with harmonic forcing or whenever an external eigenvalue
  (forcing frequency) is present.

- **`resonance_set_from_complex_normal_form_style`** (autonomous ROMs, complex form):
  Targets only near-identity monomials; suitable for computing autonomous SSMs
  when all reduced variables are complex conjugate pairs.

- **`resonance_set_from_real_normal_form_style`** (autonomous ROMs, real form):
  Variant of CNF style operating on real eigenvalue pairs; use when building a
  real-valued reduced-order model without explicit complex conjugation.

- **`resonance_set_from_condition_number_estimate`** (near-resonance uncertainty):
  Flags a monomial as resonant when the cohomological operator is nearly singular
  (condition number exceeds a threshold).  Use when eigenvalues are approximate
  or when the resonance gap is small and purely geometric criteria are unreliable.
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
	resonant_targets,
	resonant_multiindices,
	EigenvalueCondition,
	RealEigenvalueCondition,
	ConditionNumberEstimateCondition,
	GraphInternal,
	NormalFormInternal,
	ResonanceStyle

# ======================================================================
# ResonanceSet
# ======================================================================

"""
	ResonanceSet{NVAR, N_TARGETS, M}

Boolean look-up table recording which monomials are resonant with which
master-mode targets.

- `multiindices`: the `MultiindexSet{NVAR}` over which resonances are defined
- `resonances`: `N_TARGETS × NMON` `BitMatrix`; `resonances[r, k] == true` means
  monomial `k` is resonant with target `r`
- `n_internal`: number of internal master modes (rows 1:`n_internal` correspond to
  the ROM eigenvalues; remaining rows correspond to external/forcing eigenvalues)

Use one of the `resonance_set_from_*` constructors rather than building this directly.
"""
struct ResonanceSet{NVAR, N_TARGETS, M <: AbstractMatrix{Bool}}
	multiindices::MultiindexSet{NVAR}
	resonances::M
	n_internal::Int
	function ResonanceSet{NVAR, N_TARGETS, M}(
		multiindices::MultiindexSet{NVAR}, resonances::M,
		n_internal::Int) where {NVAR, N_TARGETS, M <: AbstractMatrix{Bool}}
		NMON = length(multiindices)
		@assert size(resonances) == (N_TARGETS, NMON)
		@assert 0 ≤ n_internal ≤ N_TARGETS
		new{NVAR, N_TARGETS, M}(multiindices, resonances, n_internal)
	end
end

"""
	empty_resonance_set(multiindices, nmodes, n_internal) -> ResonanceSet

Construct a `ResonanceSet` with all resonance flags set to `false`.
Useful as a starting point for manual `set_resonance!` calls.
"""
function empty_resonance_set(multiindices::MultiindexSet{NVAR}, nmodes::Int, n_internal::Int) where {NVAR}
	ResonanceSet{NVAR, nmodes, BitMatrix}(multiindices, falses(nmodes, length(multiindices)), n_internal)
end

"""
	set_resonance!(rs, target, idx, value) -> rs
	set_resonance!(rs, target, mi, value)  -> rs

Set the resonance flag for target mode `target` and monomial `idx` (or multiindex
vector `mi`) to `value`.  Returns `rs` for chaining.  Warns if `mi` is not found.
"""
function set_resonance!(rs::ResonanceSet{NVAR, N_TARGETS}, target::Int,
	idx::Int, value::Bool) where {NVAR, N_TARGETS}
	(rs.resonances[target, idx] = value; rs)
end
function set_resonance!(rs::ResonanceSet{NVAR, N_TARGETS}, target::Int,
	mi::Vector{Int}, value::Bool) where {NVAR, N_TARGETS}
	idx = find_in_set(rs.multiindices, mi)
	idx === nothing && @warn "Multiindex $mi not found" && return rs
	return set_resonance!(rs, target, idx, value)
end

"""
	is_resonant(rs, idx, target) -> Bool
	is_resonant(rs, mi, target)  -> Bool

Return `true` when the monomial at position `idx` (or with exponent vector `mi`)
is resonant with master-mode target `target`.
"""
is_resonant(rs::ResonanceSet, idx::Int, target::Int) = rs.resonances[target, idx]
function is_resonant(rs::ResonanceSet{NVAR, N_TARGETS}, mi::Vector{Int}, target::Int) where {
	NVAR, N_TARGETS}
	idx = find_in_set(rs.multiindices, mi)
	idx === nothing && return false
	return rs.resonances[target, idx]
end

"""
	resonant_targets(rs, idx) -> AbstractVector{Bool}
	resonant_targets(rs, mi)  -> Union{AbstractVector{Bool}, Nothing}

Return the column `rs.resonances[:, idx]` indicating which targets are resonant
with the monomial at position `idx` (or with exponent vector `mi`).
Returns `nothing` when `mi` is not in the multiindex set.
"""
function resonant_targets(rs::ResonanceSet{NVAR, N_TARGETS}, idx::Int) where {
	NVAR, N_TARGETS}
	rs.resonances[:, idx]
end
function resonant_targets(rs::ResonanceSet{NVAR, N_TARGETS}, mi::Vector{Int}) where {
	NVAR, N_TARGETS}
	idx = find_in_set(rs.multiindices, mi)
	idx === nothing && return nothing
	return resonant_targets(rs, idx)
end

"""
	resonant_multiindices(rs, target) -> Vector{Int}

Return the positions in the multiindex set of all monomials resonant with `target`.
"""
resonant_multiindices(rs::ResonanceSet, target::Int) = findall(rs.resonances[target, :])

# ======================================================================
# Internal strategies
# ======================================================================

"""
	InternalResonance

Abstract supertype for strategies that decide which monomials are resonant
with the *internal* (ROM) master modes.  Two concrete subtypes are provided:
`GraphInternal` and `NormalFormInternal`.
"""
abstract type InternalResonance end

"""
	GraphInternal <: InternalResonance

Internal-resonance strategy for the **graph style**: every monomial of total
degree ≥ 2 is marked resonant with all internal master modes.  Linear monomials
`eᵣ` are resonant only with their own mode `r`.
"""
struct GraphInternal <: InternalResonance end

"""
	NormalFormInternal <: InternalResonance

Internal-resonance strategy for the **normal form style**: no monomial is
automatically marked resonant with internal modes — resonance is determined
entirely by the outer eigenvalue-proximity condition.
"""
struct NormalFormInternal <: InternalResonance end

"""
	apply_internal_resonances!(resonances, strategy, mi, n_internal, k)

Set the internal-resonance flags in column `k` of `resonances` for the
monomial with exponent vector `mi`, according to `strategy`.

- `GraphInternal`: marks all `n_internal` rows resonant for degree ≥ 2;
  for a linear monomial `eᵣ` marks only row `r` (if `r ≤ n_internal`).
- `NormalFormInternal`: no-op — leaves all internal flags unchanged.
"""
function apply_internal_resonances!(resonances::AbstractMatrix{Bool}, ::NormalFormInternal,
	mi::AbstractVector{Int}, n_internal::Int, k::Int)
	return
end
function apply_internal_resonances!(resonances::AbstractMatrix{Bool}, ::GraphInternal,
	mi::AbstractVector{Int}, n_internal::Int, k::Int)
	deg = sum(mi)
	if deg == 1
		pos = findfirst(!iszero, mi)
		if pos ≤ n_internal
			resonances[pos, k] = true
		else
			for j in 1:n_internal
				resonances[j, k] = true
			end
		end
	elseif deg > 1
		for j in 1:n_internal
			resonances[j, k] = true
		end
	end
end

# ======================================================================
# Outer conditions
#
# All `is_resonant(cond, target, s, k)` methods use **local indexing**:
# `target` is a global target index; the condition first looks it up in
# `cond.target_indices` to obtain a local index, then uses that local
# index to address `cond.eigenvalues` (and `cond.condition_numbers`).
# This decouples the size of the stored arrays from the global N_TARGETS.
# ======================================================================

"""
	OuterResonanceCondition

Abstract supertype for conditions that test whether a monomial is resonant with
an *outer* (forcing or target) mode at superharmonic frequency `s`.

All concrete subtypes must implement
`is_resonant(cond, target::Int, s::ComplexF64, k::Int) -> Bool`.
"""
abstract type OuterResonanceCondition end

"""
	EigenvalueCondition <: OuterResonanceCondition

Flags a monomial as resonant when `|λⱼ - s| < tol`.

- `eigenvalues`: target eigenvalues (local indexing).
- `tol`: scalar global tolerance, or `Vector{Vector{Float64}}` for per-monomial,
  per-target tolerances.
- `target_indices`: global target indices this condition applies to.
"""
struct EigenvalueCondition <: OuterResonanceCondition
	eigenvalues::Vector{ComplexF64}
	tol::Union{Float64, Vector{Vector{Float64}}}
	target_indices::Vector{Int}   # global target indices this condition applies to
	function EigenvalueCondition(eig, tol, target_indices = 1:length(eig))
		new(eig, tol, collect(target_indices))
	end
end

"""
	RealEigenvalueCondition <: OuterResonanceCondition

Flags a monomial as resonant when `|λⱼ - s| < tol` **or** `|λ_{conj(j)} - s| < tol`,
so that conjugate eigenvalue pairs share the same resonance flag.

- `conjugacy_map`: local index map; `conjugacy_map[i]` is the local index of the
  conjugate of eigenvalue `i`.
"""
struct RealEigenvalueCondition <: OuterResonanceCondition
	eigenvalues::Vector{ComplexF64}
	conjugacy_map::Vector{Int}    # local index map: conjugacy_map[local_i] = local_j
	tol::Union{Float64, Vector{Vector{Float64}}}
	target_indices::Vector{Int}
	function RealEigenvalueCondition(eig, conj, tol, target_indices = 1:length(eig))
		new(eig, conj, tol, collect(target_indices))
	end
end

"""
	ConditionNumberEstimateCondition <: OuterResonanceCondition

Flags a monomial as resonant using a condition-number criterion:

	|λⱼ - s| * max_cond < spectral_radius * κ(λⱼ)

This catches near-resonances that a fixed-tolerance eigenvalue check would miss
when the cohomological operator is close to singular.

- `spectral_radius`: spectral radius of the full-order system.
- `condition_numbers`: per-target eigenvalue condition numbers `κ(λⱼ)`.
- `max_cond`: maximum acceptable condition number for the cohomological operator.
- `conjugacy_map`: optional local conjugacy map; when set, both `λⱼ` and its
  conjugate are tested.
"""
struct ConditionNumberEstimateCondition <: OuterResonanceCondition
	eigenvalues::Vector{ComplexF64}
	spectral_radius::Float64
	condition_numbers::Vector{Float64}
	max_cond::Float64
	target_indices::Vector{Int}
	conjugacy_map::Union{Nothing, Vector{Int}}  # local index map
	function ConditionNumberEstimateCondition(
		eig, spectral_radius, eigenvalue_condition_number,
		max_cond, target_indices, conj = nothing)
		new(eig, spectral_radius, eigenvalue_condition_number,
			max_cond, collect(target_indices), conj)
	end
end

"""
	_local_index(cond, target) -> Union{Int, Nothing}

Return the local array index for global target `target` in `cond.target_indices`,
or `nothing` if this condition does not apply to that target.
"""
@inline _local_index(cond::OuterResonanceCondition, target::Int) = findfirst(==(target), cond.target_indices)

"""
	is_resonant(cond::EigenvalueCondition, target, s, k) -> Bool
	is_resonant(cond::RealEigenvalueCondition, target, s, k) -> Bool
	is_resonant(cond::ConditionNumberEstimateCondition, target, s, k) -> Bool

Test whether superharmonic frequency `s` of monomial `k` is resonant with
target mode `target` under the given outer condition.  Returns `false` immediately
when `target` is not in `cond.target_indices`.
"""
function is_resonant(cond::EigenvalueCondition, target::Int, s::ComplexF64, k::Int)::Bool
	local_idx = _local_index(cond, target)
	local_idx === nothing && return false
	eig = cond.eigenvalues[local_idx]
	tol = cond.tol
	if tol isa Float64
		return abs(eig - s) < tol
	else
		return abs(eig - s) < tol[k][local_idx]
	end
end

function is_resonant(cond::RealEigenvalueCondition, target::Int, s::ComplexF64, k::Int)::Bool
	local_idx = _local_index(cond, target)
	local_idx === nothing && return false
	# conjugacy_map[local_idx] gives the local index of the conjugate target
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

function is_resonant(cond::ConditionNumberEstimateCondition, target::Int, s::ComplexF64, k::Int)::Bool
	local_idx = _local_index(cond, target)
	local_idx === nothing && return false
	spectral_radius = cond.spectral_radius
	max_cond = cond.max_cond
	eig = cond.eigenvalues[local_idx]
	eigenvalue_condition_number = cond.condition_numbers[local_idx]
	if cond.conjugacy_map === nothing
		return abs(eig - s) * max_cond < spectral_radius * eigenvalue_condition_number
	else
		local_conj = cond.conjugacy_map[local_idx]
		eig_conj = cond.eigenvalues[local_conj]
		eigenvalue_condition_number_conj = cond.condition_numbers[local_conj]
		return (abs(eig - s) * max_cond < spectral_radius * eigenvalue_condition_number) ||
			   (abs(eig_conj - s) * max_cond <
				spectral_radius * eigenvalue_condition_number_conj)
	end
end

# ======================================================================
# ResonanceStyle and generic constructor
# ======================================================================

"""
	AbstractResonanceStyle

Abstract supertype for resonance style objects.  A concrete subtype bundles an
`InternalResonance` strategy with an `OuterResonanceCondition` and is passed to
`resonance_set_from_style` to build a `ResonanceSet`.
"""
abstract type AbstractResonanceStyle end

"""
	ResonanceStyle{INT, EXT} <: AbstractResonanceStyle

Composed resonance specification: combines an `InternalResonance` strategy `INT`
with an `OuterResonanceCondition` `EXT`.

# Fields
- `super_eigenvalues`: eigenvalues used to compute the superharmonic `s = ⟨λ, α⟩`.
- `internal_strategy`: controls internal-mode resonance marking.
- `outer_condition`: controls outer-mode resonance marking.
- `n_internal`: number of internal (ROM) master modes.
- `n_targets`: total number of target rows in the resonance matrix (`n_internal + n_outer`).
"""
struct ResonanceStyle{INT <: InternalResonance, EXT <: OuterResonanceCondition} <:
	   AbstractResonanceStyle
	super_eigenvalues::Vector{ComplexF64}
	internal_strategy::INT
	outer_condition::EXT
	n_internal::Int
	n_targets::Int
end

"""
	n_internal(style::ResonanceStyle) -> Int

Return the number of internal master modes in `style`.
"""
n_internal(style::ResonanceStyle) = style.n_internal

"""
	build_resonances_matrix(style, multiindices) -> BitMatrix

Build the `N_TARGETS × NMON` resonance matrix from a `ResonanceStyle` and a
`MultiindexSet`.  For each monomial, the superharmonic `s` is computed, then
`apply_internal_resonances!` and the outer `is_resonant` condition are applied.
"""
function build_resonances_matrix(style::ResonanceStyle{INT, EXT},
	multiindices::MultiindexSet{NVAR}) where {NVAR, INT, EXT}
	exps = multiindices.exponents
	NMON = length(exps)
	superharmonics = [sum(style.super_eigenvalues .* mi) for mi in exps]
	n_int = style.n_internal
	N_TARGETS = style.n_targets
	resonances = falses(N_TARGETS, NMON)

	for k in 1:NMON
		mi = exps[k]
		s = superharmonics[k]
		apply_internal_resonances!(resonances, style.internal_strategy, mi, n_int, k)
		for j in 1:N_TARGETS
			if is_resonant(style.outer_condition, j, s, k)
				resonances[j, k] = true
			end
		end
	end
	return resonances
end

"""
	resonance_set_from_style(style, multiindices) -> ResonanceSet

Build a `ResonanceSet` from any `AbstractResonanceStyle` and a `MultiindexSet`.
This is the generic entry point used internally by all `resonance_set_from_*` constructors.
"""
function resonance_set_from_style(style::AbstractResonanceStyle, multiindices::MultiindexSet{NVAR}) where {NVAR}
	resonances = build_resonances_matrix(style, multiindices)
	N_TARGETS = size(resonances, 1)
	return ResonanceSet{NVAR, N_TARGETS, BitMatrix}(multiindices, resonances, n_internal(style))
end

# ======================================================================
# Public constructors (convenience)
# ======================================================================

"""
	resonance_set_from_graph_style(
		n_internal, multiindices, super_eigenvalues,
		outer_eigenvalues, tol)

Build a `ResonanceSet` using the **graph style**: every monomial of total degree ≥ 2
is marked resonant with all `n_internal` internal master modes.  Outer (forcing)
modes are flagged by eigenvalue proximity: monomial `k` is resonant with outer
target `j` when `|λⱼ - s| < tol`, where `s = ⟨super_eigenvalues, α⟩`.

This is the recommended choice for non-autonomous SSMs with external forcing.
"""
function resonance_set_from_graph_style(n_internal::Int, multiindices::MultiindexSet{NVAR},
	super_eigenvalues::Vector{ComplexF64}, outer_eigenvalues::Vector{ComplexF64},
	tol::Union{Float64, Vector{Vector{Float64}}}) where {NVAR}
	n_outer = length(outer_eigenvalues)
	N_TARGETS = n_internal + n_outer
	ext_cond = EigenvalueCondition(outer_eigenvalues, tol, (n_internal+1):N_TARGETS)
	style = ResonanceStyle(
		super_eigenvalues, GraphInternal(), ext_cond, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

"""
	resonance_set_from_complex_normal_form_style(
		n_internal, multiindices, super_eigenvalues,
		target_eigenvalues, tol)

Build a `ResonanceSet` using the **complex normal form style**: no automatic
internal resonances; a monomial is resonant with target `j` only when
`|λⱼ - s| < tol`.  Suitable for autonomous SSMs with complex conjugate
reduced variables.
"""
function resonance_set_from_complex_normal_form_style(
	n_internal::Int, multiindices::MultiindexSet{NVAR},
	super_eigenvalues::Vector{ComplexF64}, target_eigenvalues::Vector{ComplexF64},
	tol::Union{Float64, Vector{Vector{Float64}}}) where {NVAR}
	N_TARGETS = length(target_eigenvalues)
	ext_cond = EigenvalueCondition(target_eigenvalues, tol, 1:N_TARGETS)
	style = ResonanceStyle(
		super_eigenvalues, NormalFormInternal(), ext_cond, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

"""
	resonance_set_from_real_normal_form_style(
		n_internal, multiindices, super_eigenvalues,
		target_eigenvalues, conjugacy_map, tol)

Build a `ResonanceSet` using the **real normal form style**: like the complex
normal form style but conjugate pairs share resonance — monomial `k` is resonant
with target `j` when `|λⱼ - s| < tol` OR `|λ_{conj(j)} - s| < tol`.
`conjugacy_map[i]` is the local index of the conjugate of target `i`.
Use when building a real-valued ROM without explicit complex conjugation.
"""
function resonance_set_from_real_normal_form_style(
	n_internal::Int,
	multiindices::MultiindexSet{NVAR},
	super_eigenvalues::Vector{ComplexF64},
	target_eigenvalues::Vector{ComplexF64},
	conjugacy_map::Vector{Int},
	tol::Union{Float64, Vector{Vector{Float64}}},
) where {NVAR}
	N_TARGETS = length(target_eigenvalues)
	ext_cond = RealEigenvalueCondition(target_eigenvalues, conjugacy_map, tol, 1:N_TARGETS)
	style = ResonanceStyle(
		super_eigenvalues, NormalFormInternal(), ext_cond, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

"""
	resonance_set_from_condition_number_estimate(
		n_internal, multiindices, super_eigenvalues,
		target_eigenvalues, spectral_radius,
		target_condition_numbers, max_cond
		[, target_indices, conjugacy_map])

Build a `ResonanceSet` that flags near-resonant monomials by a condition-number
criterion: monomial `k` is resonant with target `j` when

	|λⱼ - s| * max_cond < spectral_radius * κ(λⱼ)

where `κ(λⱼ)` is the eigenvalue condition number and `spectral_radius` is the
spectral radius of the system.  Use when eigenvalues are approximate and purely
geometric proximity criteria are unreliable.
"""
function resonance_set_from_condition_number_estimate(
	n_internal::Int,
	multiindices::MultiindexSet{NVAR},
	super_eigenvalues::Vector{ComplexF64},
	target_eigenvalues::Vector{ComplexF64},
	spectral_radius::Float64,
	target_condition_numbers::Vector{Float64},
	max_cond::Float64,
	target_indices::Union{UnitRange{Int}, Vector{Int}} = 1:length(target_eigenvalues),
	conjugacy_map::Union{Nothing, Vector{Int}} = nothing,
) where {NVAR}
	N_TARGETS = length(target_eigenvalues)
	ext_cond = ConditionNumberEstimateCondition(
		target_eigenvalues, spectral_radius, target_condition_numbers,
		max_cond, collect(target_indices), conjugacy_map)
	style = ResonanceStyle(
		super_eigenvalues, NormalFormInternal(), ext_cond, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

"""
	resonance_set_from_graph_style(n_internal, multiindices, super_eigenvalues, outer_condition)

Advanced overload of `resonance_set_from_graph_style` that accepts any pre-built
`OuterResonanceCondition`.  `n_targets` is inferred as
`max(n_internal, maximum(outer_condition.target_indices))`.
"""
function resonance_set_from_graph_style(n_internal::Int, multiindices::MultiindexSet{NVAR},
	super_eigenvalues::Vector{ComplexF64},
	outer_condition::OuterResonanceCondition) where {NVAR}
	N_TARGETS = isempty(outer_condition.target_indices) ?
				n_internal : max(n_internal, maximum(outer_condition.target_indices))
	style = ResonanceStyle(
		super_eigenvalues, GraphInternal(), outer_condition, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

function Base.show(io::IO, rs::ResonanceSet{NVAR, N_TARGETS, M}) where {NVAR, N_TARGETS, M}
	print(io, "ResonanceSet{$NVAR,$N_TARGETS} with ",
		length(rs.multiindices), " multiindices, ", count(rs.resonances),
		" resonances (", rs.n_internal, " internal)")
end

end # module
