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
struct ResonanceSet{NVAR, N_TARGETS, M <: AbstractMatrix{Bool}}
	multiindices::MultiindexSet{NVAR}
	resonances::M
	n_internal::Int
	function ResonanceSet{NVAR, N_TARGETS, M}(multiindices::MultiindexSet{NVAR}, resonances::M, n_internal::Int) where {NVAR, N_TARGETS, M <: AbstractMatrix{Bool}}
		NMON = length(multiindices)
		@assert size(resonances) == (N_TARGETS, NMON)
		@assert 0 ≤ n_internal ≤ N_TARGETS
		new{NVAR, N_TARGETS, M}(multiindices, resonances, n_internal)
	end
end

empty_resonance_set(multiindices::MultiindexSet{NVAR}, nmodes::Int, n_internal::Int) where {NVAR} =
	ResonanceSet{NVAR, nmodes, BitMatrix}(multiindices, falses(nmodes, length(multiindices)), n_internal)

set_resonance!(rs::ResonanceSet{NVAR, N_TARGETS}, target::Int, idx::Int, value::Bool) where {NVAR, N_TARGETS} = (rs.resonances[target, idx] = value; rs)
function set_resonance!(rs::ResonanceSet{NVAR, N_TARGETS}, target::Int, mi::Vector{Int}, value::Bool) where {NVAR, N_TARGETS}
	idx = find_in_set(rs.multiindices, mi)
	idx === nothing && @warn "Multiindex $mi not found" && return rs
	return set_resonance!(rs, target, idx, value)
end

is_resonant(rs::ResonanceSet, idx::Int, target::Int) = rs.resonances[target, idx]
function is_resonant(rs::ResonanceSet{NVAR, N_TARGETS}, mi::Vector{Int}, target::Int) where {NVAR, N_TARGETS}
	idx = find_in_set(rs.multiindices, mi)
	idx === nothing && return false
	return rs.resonances[target, idx]
end
resonant_targets(rs::ResonanceSet{NVAR, N_TARGETS}, idx::Int) where {NVAR, N_TARGETS} = rs.resonances[:, idx]
function resonant_targets(rs::ResonanceSet{NVAR, N_TARGETS}, mi::Vector{Int}) where {NVAR, N_TARGETS}
	idx = find_in_set(rs.multiindices, mi)
	idx === nothing && return nothing
	return resonant_targets(rs, idx)
end
resonant_multiindices(rs::ResonanceSet, target::Int) = findall(rs.resonances[target, :])

# ======================================================================
# Internal strategies
# ======================================================================
abstract type InternalResonance end
struct GraphInternal <: InternalResonance end
struct NormalFormInternal <: InternalResonance end

function apply_internal_resonances!(resonances::AbstractMatrix{Bool}, ::NormalFormInternal, mi::AbstractVector{Int}, n_internal::Int, k::Int)
	return
end
function apply_internal_resonances!(resonances::AbstractMatrix{Bool}, ::GraphInternal, mi::AbstractVector{Int}, n_internal::Int, k::Int)
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
# External conditions
#
# All `is_resonant(cond, target, s, k)` methods use **local indexing**:
# `target` is a global target index; the condition first looks it up in
# `cond.target_indices` to obtain a local index, then uses that local
# index to address `cond.eigenvalues` (and `cond.condition_numbers`).
# This decouples the size of the stored arrays from the global N_TARGETS.
# ======================================================================
abstract type ExternalResonanceCondition end

struct EigenvalueCondition <: ExternalResonanceCondition
	eigenvalues::Vector{ComplexF64}
	tol::Union{Float64, Vector{Vector{Float64}}}
	target_indices::Vector{Int}   # global target indices this condition applies to
	function EigenvalueCondition(eig, tol, target_indices = 1:length(eig))
		new(eig, tol, collect(target_indices))
	end
end

struct RealEigenvalueCondition <: ExternalResonanceCondition
	eigenvalues::Vector{ComplexF64}
	conjugacy_map::Vector{Int}    # local index map: conjugacy_map[local_i] = local_j
	tol::Union{Float64, Vector{Vector{Float64}}}
	target_indices::Vector{Int}
	function RealEigenvalueCondition(eig, conj, tol, target_indices = 1:length(eig))
		new(eig, conj, tol, collect(target_indices))
	end
end

struct ConditionNumberEstimateCondition <: ExternalResonanceCondition
	eigenvalues::Vector{ComplexF64}
	spectral_radius::Float64
	condition_numbers::Vector{Float64}
	max_cond::Float64
	target_indices::Vector{Int}
	conjugacy_map::Union{Nothing, Vector{Int}}  # local index map
	function ConditionNumberEstimateCondition(eig, spectral_radius, eigenvalue_condition_number, max_cond, target_indices, conj = nothing)
		new(eig, spectral_radius, eigenvalue_condition_number, max_cond, collect(target_indices), conj)
	end
end

# Local-index helper: returns nothing if target is not in the condition.
@inline _local_index(cond::ExternalResonanceCondition, target::Int) =
	findfirst(==(target), cond.target_indices)

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
			   (abs(eig_conj - s) * max_cond < spectral_radius * eigenvalue_condition_number_conj)
	end
end

# ======================================================================
# ResonanceStyle and generic constructor
# ======================================================================
abstract type AbstractResonanceStyle end

# n_targets: total number of target modes (rows in the resonance matrix).
# This may exceed length(external_condition.eigenvalues) when, e.g., internal
# resonances are handled by GraphInternal but the external condition only stores
# the outer eigenvalues.
struct ResonanceStyle{INT <: InternalResonance, EXT <: ExternalResonanceCondition} <: AbstractResonanceStyle
	super_eigenvalues::Vector{ComplexF64}
	internal_strategy::INT
	external_condition::EXT
	n_internal::Int
	n_targets::Int
end
n_internal(style::ResonanceStyle) = style.n_internal

function build_resonances_matrix(style::ResonanceStyle{INT, EXT}, multiindices::MultiindexSet{NVAR}) where {NVAR, INT, EXT}
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
			if is_resonant(style.external_condition, j, s, k)
				resonances[j, k] = true
			end
		end
	end
	return resonances
end

function resonance_set_from_style(style::AbstractResonanceStyle, multiindices::MultiindexSet{NVAR}) where {NVAR}
	resonances = build_resonances_matrix(style, multiindices)
	N_TARGETS = size(resonances, 1)
	return ResonanceSet{NVAR, N_TARGETS, BitMatrix}(multiindices, resonances, n_internal(style))
end

# ======================================================================
# Public constructors (convenience)
# ======================================================================

# Graph style: internal modes use GraphInternal (all-resonant for degree ≥ 2),
# outer modes use an eigenvalue proximity check.
# The EigenvalueCondition stores only the outer eigenvalues (local indices 1..n_outer)
# with target_indices = (n_internal+1):(n_internal+n_outer).
function resonance_set_from_graph_style(n_internal::Int, multiindices::MultiindexSet{NVAR}, super_eigenvalues::Vector{ComplexF64}, outer_eigenvalues::Vector{ComplexF64}, tol::Union{Float64, Vector{Vector{Float64}}}) where {NVAR}
	n_outer = length(outer_eigenvalues)
	N_TARGETS = n_internal + n_outer
	ext_cond = EigenvalueCondition(outer_eigenvalues, tol, (n_internal+1):N_TARGETS)
	style = ResonanceStyle(super_eigenvalues, GraphInternal(), ext_cond, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

# Complex normal form: no automatic internal resonances; all targets checked via
# eigenvalue proximity.
function resonance_set_from_complex_normal_form_style(n_internal::Int, multiindices::MultiindexSet{NVAR}, super_eigenvalues::Vector{ComplexF64}, target_eigenvalues::Vector{ComplexF64}, tol::Union{Float64, Vector{Vector{Float64}}}) where {NVAR}
	N_TARGETS = length(target_eigenvalues)
	ext_cond = EigenvalueCondition(target_eigenvalues, tol, 1:N_TARGETS)
	style = ResonanceStyle(super_eigenvalues, NormalFormInternal(), ext_cond, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

# Real normal form: conjugate pairs share resonance via RealEigenvalueCondition.
# `conjugacy_map[i]` is the **local** index of the conjugate of target i.
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
	style = ResonanceStyle(super_eigenvalues, NormalFormInternal(), ext_cond, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

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
	ext_cond = ConditionNumberEstimateCondition(target_eigenvalues, spectral_radius, target_condition_numbers, max_cond, collect(target_indices), conjugacy_map)
	style = ResonanceStyle(super_eigenvalues, NormalFormInternal(), ext_cond, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

# Advanced: graph style with any pre-built external condition.
# n_targets is inferred as max(n_internal, maximum(cond.target_indices)).
function resonance_set_from_graph_style(n_internal::Int, multiindices::MultiindexSet{NVAR}, super_eigenvalues::Vector{ComplexF64}, external_condition::ExternalResonanceCondition) where {NVAR}
	N_TARGETS = isempty(external_condition.target_indices) ?
				n_internal : max(n_internal, maximum(external_condition.target_indices))
	style = ResonanceStyle(super_eigenvalues, GraphInternal(), external_condition, n_internal, N_TARGETS)
	return resonance_set_from_style(style, multiindices)
end

function Base.show(io::IO, rs::ResonanceSet{NVAR, N_TARGETS, M}) where {NVAR, N_TARGETS, M}
	print(io, "ResonanceSet{$NVAR,$N_TARGETS} with ", length(rs.multiindices), " multiindices, ", count(rs.resonances), " resonances (", rs.n_internal, " internal)")
end

end # module
