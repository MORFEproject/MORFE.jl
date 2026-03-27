module Resonance

using ..Multiindices: MultiindexSet, find_in_set, _last_index_below_degree
using StaticArrays: SVector, MVector

export SingleResonance, ResonanceSet, resonance_set, resonance_set_from_complex_normal_form_style

"""
	SingleResonance

saves single resonance of the style
	λ_j = γ ⋅ (λ_1,..., λ_NVAR)
with target=j and multiindex=γ
"""
struct SingleResonance{NVAR}
	target::Int
	multiindex::SVector{NVAR, Int}
end

"""
	Resonances

saves all resonances for a specific MultiindexSet
"""
struct ResonanceSet{NVAR}
	multiindices::MultiindexSet{NVAR}
	resonances::Vector{SVector{NVAR, Bool}}

	function ResonanceSet(
		multiindices::MultiindexSet{NVAR}, resonances::Vector{SVector{NVAR, Bool}}) where {NVAR}
		@assert length(resonances) == length(multiindices)
		new{NVAR}(multiindices, resonances)
	end
end

"""
	resonance_set(multiindices::MultiindexSet{NVAR}, resonance_tuple::NTuple(SingleResonance{NVAR}) -> ResonanceSet

Builds ResonanceSet boolean matrix from tuple of SingleResonance's
First order (trivial) resonances are automatically filled.
"""
function resonance_set(
	multiindices::MultiindexSet{NVAR},
	resonance_tuple::NTuple{NSR, SingleResonance{NVAR}}) where {NVAR, NSR}

	tmp_resonances = [MVector{NVAR, Bool}(ntuple(_ -> false, NVAR))
					  for _ in 1:length(multiindices)]
	#fill resonances from function argument
	for resonance in resonance_tuple
		index = find_in_set(multiindices, resonance.multiindex)
		if index === nothing
			@warn "Resonance multiindex ($(resonance.multiindex)) not contained in MultiindexSet"
			continue
		end
		@inbounds tmp_resonances[index][resonance.target] = true
	end

	#fill first order resonances
	for index in 1:_last_index_below_degree(multiindices, 1)
		multiindex = multiindices.exponents[index]
		target = findfirst(!iszero, multiindex)
		if isnothing(target)
			continue
		end
		@inbounds tmp_resonances[index][target] = true
	end
	resonances = SVector{NVAR, Bool}.(tmp_resonances)
	return ResonanceSet(multiindices, resonances)
end

"""
	resonance_set_from_eigenvalues(multiindices::MultiindexSet{N}, eigenvalues::SVector{N, Complex}, tol::Float64) -> ResonanceSet

Builds resonance matrix from MultiindexSet by testing for a tolerance.
A multiindex γ and a eigenvalue λ are said to be resonant if:
	| λ - γ⋅eigenvalues | < tol 
"""
function resonance_set_from_complex_normal_form_style(
	multiindices::MultiindexSet{NVAR},
	eigenvalues::SVector{NVAR, ComplexF64},
	tol::Float64) where {NVAR}

	exps = multiindices.exponents
	NMON = length(exps)
	resonances = Vector{SVector{NVAR, Bool}}(undef, NMON)
	@inbounds for k in 1:NMON
		multiindex = exps[k]
		superharmonic = sum(eigenvalues .* multiindex)
		for j in 1:NVAR
			resonances[k] = SVector{NVAR, Bool}(
				ntuple(j -> abs(eigenvalues[j] - superharmonic) < tol, NVAR),
			)
		end
	end
	return ResonanceSet(multiindices, resonances)
end

"""
	resonance_set_from_complex_normal_form_style(multiindices::MultiindexSet{NVAR}, eigenvalues::SVector{NVAR, Complex}, tol::Vector{SVector{NVAR, Float64}}) -> ResonanceSet

Builds resonance matrix from MultiindexSet by testing for a tolerance.
A multiindex γ and a eigenvalue λ are said to be resonant if:
	| λ - γ⋅eigenvalues | < tol 
"""
function resonance_set_from_complex_normal_form_style(
	multiindices::MultiindexSet{NVAR},
	eigenvalues::SVector{NVAR, Complex},
	tol::Vector{SVector{NVAR, Float64}}) where {NVAR}

	NMON = length(exps)
	exps = multiindices.exponents
	resonances = Vector{SVector{NVAR, Bool}}(undef, NMON)
	@inbounds for k in 1:NMON
		multiindex = eigenvalues[k]
		superharmonic = sum(eigenvalues .* multiindex)
		resonances[k] = SVector{NVAR, Bool}(
			ntuple(j -> abs(eigenvalues[j] - superharmonic) < tol[k][j], N),
		)
	end
	return ResonanceSet(multiindices, resonances)
end

end # module
