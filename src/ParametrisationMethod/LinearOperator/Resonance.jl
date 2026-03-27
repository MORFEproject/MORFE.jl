module Resonance

using ..Multiindices
using StaticArrays: SVector, MVector
export SingleResonance, ResonanceSet, resonance_set, resonance_set_from_eigenvalues

"""
    SingleResonance

saves single resonance of the style
    λ_j = γ ⋅ (λ_1,..., λ_N)
with target=j and multiindex=γ
"""
struct SingleResonance{N}
    target::Int
    multiindex::SVector{N, Int}
end

"""
    Resonances

saves all resonances for a specific MultiindexSet
"""
struct ResonanceSet{N}
    multiindices::MultiindexSet{N}
    resonances::Vector{SVector{N, Bool}}

    function ResonanceSet(
            multiindices::MultiindexSet{N}, resonances::Vector{SVector{N, Bool}}) where {N}
        @assert length(resonances) == length(multiindices)
        new{N}(multiindices, resonances)
    end
end

"""
    resonance_set(multiindices::MultiindexSet{N}, resonance_tuple::NTuple(SingleResonance{N}) -> ResonanceSet

Builds ResonanceSet boolean matrix from tuple of SingleResonance's
First order (trivial) resonances are automatically filled.
"""
function resonance_set(
        multiindices::MultiindexSet{N},
        resonance_tuple::NTuple{M, SingleResonance{N}}) where {N, M}
    tmp_resonances = [MVector{N, Bool}(ntuple(_ -> false, N))
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
    for index in 1:Multiindices._last_index_below_degree(multiindices, 1)
        multiindex = multiindices.exponents[index]
        target = findfirst(!iszero, multiindex)
        if isnothing(target)
            continue
        end
        @inbounds tmp_resonances[index][target] = true
    end
    resonances = SVector{N, Bool}.(tmp_resonances)
    return ResonanceSet(multiindices, resonances)
end

"""
    resonance_set_from_eigenvalues(multiindices::MultiindexSet{N}, eigs::SVector{N, Complex}, tol::Float64) -> ResonanceSet

Builds resonance matrix from MultiindexSet by testing for a tolerance.
A multiindex γ and a eigenvalue λ are said to be resonant if:
    | λ - γ⋅eigs | < tol 
"""
function resonance_set_from_eigenvalues(
        multiindices::MultiindexSet{N},
        eigs::SVector{N, ComplexF64},
        tol::Float64) where {N}
    exps = multiindices.exponents
    M = length(exps)
    resonances = Vector{SVector{N, Bool}}(undef, M)
    @inbounds for k in 1:M
        γ = exps[k]
        dot = sum(eigs .* γ)
        for j in 1:N
            resonances[k] = SVector{N, Bool}(
                ntuple(j -> abs(eigs[j] - dot) < tol, N)
            )
        end
    end
    return ResonanceSet(multiindices, resonances)
end
"""
    resonance_set_from_eigenvalues(multiindices::MultiindexSet{N}, eigs::SVector{N, Complex}, tol::Vector{SVector{N, Float64}}) -> ResonanceSet

Builds resonance matrix from MultiindexSet by testing for a tolerance.
A multiindex γ and a eigenvalue λ are said to be resonant if:
    | λ - γ⋅eigs | < tol 
"""
function resonance_set_from_eigenvalues(
        multiindices::MultiindexSet{N},
        eigs::SVector{N, Complex},
        tol::Vector{SVector{N, Float64}}) where {N}
    exps = multiindices.exponents
    M = length(exps)
    resonances = Vector{SVector{N, Bool}}(undef, M)
    @inbounds for k in 1:M
        γ = exps[k]
        dot = eigs * γ
        resonances[k] = SVector{N, Bool}(
            ntuple(j -> abs(eigs[j] - d) < tol[k][j], N)
        )
    end
    return ResonanceSet(multiindices, resonances)
end

end # module