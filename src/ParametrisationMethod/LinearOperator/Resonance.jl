module Resonance

using ..Multiindices: MultiindexSet, find_in_set, _last_index_below_degree
using StaticArrays: SVector, MVector, setindex

export SingleResonance, ResonanceSet,
	resonance_set_from_graph_style,
	resonance_set_from_complex_normal_form_style,
	update_resonance!,
	is_resonant, resonant_targets, resonant_multiindices,
	tolerance_from_local_estimate,
	empty_resonance_set

# ----------------------------------------------------------------------------
# Concrete resonance rules
# ----------------------------------------------------------------------------

"""
	GraphStyleRule{NVAR, NO, T}

Rule following a "graph style" for internal modes and complex normal‑form style for outer modes:

- Internal modes:
	* degree‑1 monomials: if the non‑zero index ≤ `n_internal` → self‑resonance;
	  if the non‑zero index > `n_internal` (external) → resonance with all internal modes.
	* all higher‑degree monomials → resonance with all internal modes.
- Outer modes: set according to `|λ_outer[j] - multiindex·λ_super| < tol` for each outer mode j.

The tolerance `tol` can be a scalar `Float64` (same for all (multiindex,outer‑mode) pairs)
or a `Vector{SVector{NO,Float64}}` giving per‑multiindex tolerances.

# Fields
- `super_eigenvalues::SVector{NVAR,ComplexF64}`: eigenvalues of the internal+external modes.
- `outer_eigenvalues::SVector{NO,ComplexF64}`: eigenvalues of the outer modes.
- `n_internal::Int`: number of internal modes (the first `n_internal` targets).
- `tol::T`: tolerance (scalar or vector of per‑multiindex tolerances).
"""
struct GraphStyleRule{NVAR, NO, T}
	super_eigenvalues::SVector{NVAR, ComplexF64}
	outer_eigenvalues::SVector{NO, ComplexF64}
	n_internal::Int
	tol::T
end

n_internal(rule::GraphStyleRule) = rule.n_internal

function build_resonances(rule::GraphStyleRule{NVAR, NO, Float64},
	multiindices::MultiindexSet{NVAR}) where {NVAR, NO}
	n_internal = rule.n_internal
	N_TARGETS = n_internal + NO
	exps = multiindices.exponents
	NMON = length(exps)
	degrees = [sum(mi) for mi in exps]
	superharmonics = [sum(rule.super_eigenvalues .* mi) for mi in exps]
	tol = rule.tol
	resonances = Vector{SVector{N_TARGETS, Bool}}(undef, NMON)
	@inbounds for k in 1:NMON
		mi = exps[k]
		deg = degrees[k]
		s = superharmonics[k]
		vec = MVector{N_TARGETS, Bool}(ntuple(_ -> false, N_TARGETS))

		# Internal modes (graph style)
		if deg == 1
			pos = findfirst(!iszero, mi)
			if pos ≤ n_internal
				vec[pos] = true
			elseif pos > n_internal
				for j in 1:n_internal
					vec[j] = true
				end
			end
		else   # deg > 1
			for j in 1:n_internal
				vec[j] = true
			end
		end

		# Outer modes (normal‑form style)
		for j in 1:NO
			if abs(rule.outer_eigenvalues[j] - s) < tol
				vec[n_internal+j] = true
			end
		end

		resonances[k] = SVector{N_TARGETS, Bool}(vec)
	end
	return resonances
end

function build_resonances(rule::GraphStyleRule{NVAR, NO, Vector{SVector{NO, Float64}}},
	multiindices::MultiindexSet{NVAR}) where {NVAR, NO}
	n_internal = rule.n_internal
	N_TARGETS = n_internal + NO
	exps = multiindices.exponents
	NMON = length(exps)
	@assert length(rule.tol) == NMON
	degrees = [sum(mi) for mi in exps]
	superharmonics = [sum(rule.super_eigenvalues .* mi) for mi in exps]
	resonances = Vector{SVector{N_TARGETS, Bool}}(undef, NMON)
	@inbounds for k in 1:NMON
		mi = exps[k]
		deg = degrees[k]
		s = superharmonics[k]
		tol_vec = rule.tol[k]
		vec = MVector{N_TARGETS, Bool}(ntuple(_ -> false, N_TARGETS))

		# Internal modes (graph style)
		if deg == 1
			pos = findfirst(!iszero, mi)
			if pos ≤ n_internal
				vec[pos] = true
			elseif pos > n_internal
				for j in 1:n_internal
					vec[j] = true
				end
			end
		else   # deg > 1
			for j in 1:n_internal
				vec[j] = true
			end
		end

		# Outer modes with per‑multiindex tolerances
		for j in 1:NO
			if abs(rule.outer_eigenvalues[j] - s) < tol_vec[j]
				vec[n_internal+j] = true
			end
		end

		resonances[k] = SVector{N_TARGETS, Bool}(vec)
	end
	return resonances
end

"""
	ComplexNormalFormRule{NVAR, N_TARGETS, T}

Rule based on the complex normal form condition:
`|λ_target[j] - multiindex·λ_super| < tol` for all targets j = 1..N_TARGETS.

- `λ_super` is the vector of eigenvalues for internal+external modes (length `NVAR`).
- `λ_target` is the vector of eigenvalues for internal+outer modes (length `N_TARGETS`).

The tolerance `tol` can be a scalar `Float64` (same for all (multiindex,target) pairs)
or a `Vector{SVector{N_TARGETS,Float64}}` giving per‑multiindex tolerances.
"""
struct ComplexNormalFormRule{NVAR, N_TARGETS, T}
	super_eigenvalues::SVector{NVAR, ComplexF64}
	target_eigenvalues::SVector{N_TARGETS, ComplexF64}
	n_internal::Int
	tol::T

	function ComplexNormalFormRule(super_eigenvalues::SVector{NVAR, ComplexF64},
		target_eigenvalues::SVector{N_TARGETS, ComplexF64},
		n_internal::Int,
		tol) where {NVAR, N_TARGETS}
		@assert 0 ≤ n_internal ≤ N_TARGETS
		new{NVAR, N_TARGETS, typeof(tol)}(super_eigenvalues, target_eigenvalues, n_internal, tol)
	end
end

n_internal(rule::ComplexNormalFormRule) = rule.n_internal

function build_resonances(rule::ComplexNormalFormRule{NVAR, N_TARGETS, Float64},
	multiindices::MultiindexSet{NVAR}) where {NVAR, N_TARGETS}
	exps = multiindices.exponents
	NMON = length(exps)
	superharmonics = [sum(rule.super_eigenvalues .* mi) for mi in exps]
	tol = rule.tol
	resonances = Vector{SVector{N_TARGETS, Bool}}(undef, NMON)
	@inbounds for k in 1:NMON
		s = superharmonics[k]
		vec = ntuple(j -> abs(rule.target_eigenvalues[j] - s) < tol, N_TARGETS)
		resonances[k] = SVector{N_TARGETS, Bool}(vec)
	end
	return resonances
end

function build_resonances(rule::ComplexNormalFormRule{NVAR, N_TARGETS, Vector{SVector{N_TARGETS, Float64}}},
	multiindices::MultiindexSet{NVAR}) where {NVAR, N_TARGETS}
	exps = multiindices.exponents
	NMON = length(exps)
	@assert length(rule.tol) == NMON
	superharmonics = [sum(rule.super_eigenvalues .* mi) for mi in exps]
	resonances = Vector{SVector{N_TARGETS, Bool}}(undef, NMON)
	@inbounds for k in 1:NMON
		s = superharmonics[k]
		tol_vec = rule.tol[k]
		vec = ntuple(j -> abs(rule.target_eigenvalues[j] - s) < tol_vec[j], N_TARGETS)
		resonances[k] = SVector{N_TARGETS, Bool}(vec)
	end
	return resonances
end

# ----------------------------------------------------------------------------
# SingleResonance and ResonanceSet
# ----------------------------------------------------------------------------

"""
	SingleResonance{NVAR}

Saves a single resonance condition: the `target`-th eigenvalue is resonant with the linear
combination of eigenvalues given by the `multiindex`. The `value` indicates whether the
resonance is active (true) or not (false). By default `value = true`.

# Fields
- `target::Int`: index of the eigenvalue that is resonant (must be in 1:N_TARGETS).
- `multiindex::SVector{NVAR,Int}`: multiindex defining the linear combination multiindex·λ.
- `value::Bool`: whether this resonance should be set (true) or cleared (false).
"""
struct SingleResonance{NVAR}
	target::Int
	multiindex::SVector{NVAR, Int}
	value::Bool
	SingleResonance(target, multiindex, value = true) = new{length(multiindex)}(target, multiindex, value)
end

"""
	ResonanceSet{NVAR, N_TARGETS}

Stores all resonance information for a given set of multiindices. For each multiindex,
a boolean vector of length `N_TARGETS` indicates which targets are resonant.

# Fields
- `multiindices::MultiindexSet{NVAR}`: the multiindex set used for the expansion.
- `resonances::Vector{SVector{N_TARGETS,Bool}}`: for each multiindex, an `SVector` of booleans.
- `n_internal::Int`: number of internal modes (first `n_internal` targets are internal).
"""
struct ResonanceSet{NVAR, N_TARGETS}
	multiindices::MultiindexSet{NVAR}
	resonances::Vector{SVector{N_TARGETS, Bool}}
	n_internal::Int

	function ResonanceSet(multiindices::MultiindexSet{NVAR}, resonances::Vector{SVector{N_TARGETS, Bool}}, n_internal::Int) where {NVAR, N_TARGETS}
		@assert length(resonances) == length(multiindices)
		@assert 0 ≤ n_internal ≤ N_TARGETS
		new{NVAR, N_TARGETS}(multiindices, resonances, n_internal)
	end
end

# ----------------------------------------------------------------------------
# Construction from rules
# ----------------------------------------------------------------------------

"""
	resonance_set_from_rule(rule, multiindices) -> ResonanceSet

Build a resonance set by applying the given resonance rule.
"""
function resonance_set_from_rule(rule, multiindices::MultiindexSet)
	resonances = build_resonances(rule, multiindices)
	return ResonanceSet(multiindices, resonances, n_internal(rule))
end

# ----------------------------------------------------------------------------
# Public constructors for the two styles
# ----------------------------------------------------------------------------

"""
	resonance_set_from_graph_style(n_internal, multiindices, super_eigenvalues, outer_eigenvalues, tol)

Build a resonance set where:
- Internal modes follow the graph‑style rules.
- Outer modes are set according to `|outer_eigenvalues[j] - multiindex·super_eigenvalues| < tol`.

`super_eigenvalues` must have length `NVAR` (internal + external eigenvalues).
`outer_eigenvalues` must have length `NO` (outer modes).
`tol` can be a scalar (same for all (multiindex,outer‑mode) pairs) or a vector of length NMON
where each entry is an `SVector{NO,Float64}`.
"""
function resonance_set_from_graph_style(n_internal::Int,
	multiindices::MultiindexSet{NVAR},
	super_eigenvalues::SVector{NVAR, ComplexF64},
	outer_eigenvalues::SVector{NO, ComplexF64},
	tol::Union{Float64, Vector{SVector{NO, Float64}}}) where {NVAR, NO}
	rule = GraphStyleRule{NVAR, NO, typeof(tol)}(super_eigenvalues, outer_eigenvalues, n_internal, tol)
	return resonance_set_from_rule(rule, multiindices)
end

"""
	resonance_set_from_complex_normal_form_style(n_internal, multiindices, super_eigenvalues, target_eigenvalues, tol)

Build a resonance set by testing `|target_eigenvalues[j] - multiindex·super_eigenvalues| < tol`
for every multiindex and every target j = 1..N_TARGETS.

`super_eigenvalues` must have length `NVAR` (internal + external eigenvalues).
`target_eigenvalues` must have length `N_TARGETS` (internal + outer eigenvalues).
`tol` can be a scalar (same for all (multiindex,target) pairs) or a vector of length NMON
where each entry is an `SVector{N_TARGETS,Float64}`.
"""
function resonance_set_from_complex_normal_form_style(n_internal::Int,
	multiindices::MultiindexSet{NVAR},
	super_eigenvalues::SVector{NVAR, ComplexF64},
	target_eigenvalues::SVector{N_TARGETS, ComplexF64},
	tol::Union{Float64, Vector{SVector{N_TARGETS, Float64}}}) where {NVAR, N_TARGETS}
	@assert 0 ≤ n_internal ≤ N_TARGETS
	rule = ComplexNormalFormRule(super_eigenvalues, target_eigenvalues, n_internal, tol)
	return resonance_set_from_rule(rule, multiindices)
end

# ----------------------------------------------------------------------------
# Empty set and creation from SingleResonance list
# ----------------------------------------------------------------------------

"""
	empty_resonance_set(multiindices::MultiindexSet{NVAR}, nmodes::Int, n_internal::Int) -> ResonanceSet{NVAR, nmodes}

Create a `ResonanceSet` with all resonances initially set to `false`.
"""
function empty_resonance_set(multiindices::MultiindexSet{NVAR}, nmodes::Int, n_internal::Int) where {NVAR}
	@assert 0 ≤ n_internal ≤ nmodes
	resonances = [SVector{nmodes, Bool}(ntuple(_ -> false, nmodes)) for _ in 1:length(multiindices)]
	return ResonanceSet(multiindices, resonances, n_internal)
end

"""
	ResonanceSet(multiindices::MultiindexSet{NVAR}, nmodes::Int, n_internal::Int,
				 resonances::AbstractVector{<:SingleResonance}) -> ResonanceSet

Build a `ResonanceSet` from a list of `SingleResonance` objects. The set is initialised with all
resonances set to `false`. First, the automatic first‑order resonance rule is applied:
for each degree‑1 monomial whose non‑zero index ≤ `n_internal`, that internal target is set to `true`.
Then each user‑provided `SingleResonance` is applied (updates are cumulative).
"""
function ResonanceSet(multiindices::MultiindexSet{NVAR},
	nmodes::Int,
	n_internal::Int,
	resonances::AbstractVector{<:SingleResonance}) where {NVAR}
	rs = empty_resonance_set(multiindices, nmodes, n_internal)

	# First‑order resonances: internal modes for degree‑1 monomials
	for idx in 1:_last_index_below_degree(multiindices, 1)
		mi = multiindices.exponents[idx]
		target = findfirst(!iszero, mi)
		if isnothing(target) || target > n_internal
			continue
		end
		# Set the internal target to true
		new_vec = setindex(rs.resonances[idx], true, target)
		rs.resonances[idx] = new_vec
	end

	# Apply user‑provided resonances
	update_resonance!(rs, resonances)

	return rs
end

# ----------------------------------------------------------------------------
# Update function
# ----------------------------------------------------------------------------

"""
	update_resonance!(rs::ResonanceSet, srs::AbstractVector{<:SingleResonance}) -> rs

Update a `ResonanceSet` by applying a list of `SingleResonance` objects. Each resonance is processed
in order.
"""
function update_resonance!(rs::ResonanceSet{NVAR, N_TARGETS}, srs::AbstractVector{<:SingleResonance}) where {NVAR, N_TARGETS}
	for sr in srs
		idx = find_in_set(rs.multiindices, sr.multiindex)
		if idx === nothing
			@warn "Multiindex $(sr.multiindex) not found in ResonanceSet; skipping resonance"
			continue
		end
		if !(1 <= sr.target <= N_TARGETS)
			@warn "Target $(sr.target) out of range [1, $N_TARGETS]; skipping resonance"
			continue
		end
		new_vec = setindex(rs.resonances[idx], sr.value, sr.target)
		rs.resonances[idx] = new_vec
	end
	return rs
end

# ----------------------------------------------------------------------------
# Query functions
# ----------------------------------------------------------------------------

function is_resonant(rs::ResonanceSet, idx::Int, target::Int)
	return rs.resonances[idx][target]
end

function is_resonant(rs::ResonanceSet{NVAR, N_TARGETS}, multiindex::SVector{NVAR, Int}, target::Int) where {NVAR, N_TARGETS}
	idx = find_in_set(rs.multiindices, multiindex)
	idx === nothing && return false
	return rs.resonances[idx][target]
end

resonant_targets(rs::ResonanceSet, idx::Int) = rs.resonances[idx]

function resonant_targets(rs::ResonanceSet{NVAR, N_TARGETS}, multiindex::SVector{NVAR, Int}) where {NVAR, N_TARGETS}
	idx = find_in_set(rs.multiindices, multiindex)
	idx === nothing && return nothing
	return rs.resonances[idx]
end

function resonant_multiindices(rs::ResonanceSet, target::Int)
	return [i for i in 1:length(rs.resonances) if rs.resonances[i][target]]
end

# ----------------------------------------------------------------------------
# Tolerance calculation using local estimate near a simple eigenvalue
# ----------------------------------------------------------------------------

function tolerance_from_local_estimate(
	multiindices::MultiindexSet{NVAR},
	super_eigenvalues::SVector{NVAR, ComplexF64},
	spectral_radius::Float64,
	target_condition_numbers::SVector{N_TARGETS, Float64},
	max_cond::Float64,
	target_indices::Union{UnitRange{Int}, Vector{Int}},
) where {NVAR, N_TARGETS}
	@assert max_cond > 0 "max_cond must be positive"
	exps = multiindices.exponents
	NMON = length(exps)
	superharmonics = [sum(super_eigenvalues .* mi) for mi in exps]
	NROM = length(target_indices)
	tolerances = Vector{SVector{NROM, Float64}}(undef, NMON)
	@inbounds for k in 1:NMON
		s = superharmonics[k]
		tol_vec = SVector{NROM, Float64}(
			ntuple(j -> spectral_radius * target_condition_numbers[target_indices[j]] / max_cond, NROM),
		)
		tolerances[k] = tol_vec
	end
	return tolerances
end

# ----------------------------------------------------------------------------
# Show method
# ----------------------------------------------------------------------------

function Base.show(io::IO, rs::ResonanceSet{NVAR, N_TARGETS}) where {NVAR, N_TARGETS}
	print(io, "ResonanceSet{$NVAR,$N_TARGETS} with ")
	print(io, length(rs.multiindices), " multiindices, ")
	nz = count(r -> any(r), rs.resonances)
	print(io, nz, " non‑zero resonance entries (", rs.n_internal, " internal, ", N_TARGETS - rs.n_internal, " outer targets)")
end

end # module
