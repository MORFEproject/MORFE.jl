# Causal monomial jobs and exact structural factor groups.

"""
	_SolveJob

A primary monomial solve and its optional conjugate reconstruction target.

# Fields

- `index::Int` — multiindex-set position solved by the cohomological system.
- `conjugate_target::Int` — secondary position filled by conjugation afterwards, or `0`
  when no reconstruction is required.
"""
struct _SolveJob
    index::Int
    conjugate_target::Int
end

"""Internal dispatch root for direct and exact-factor-grouped solve plans."""
abstract type _AbstractSolvePlan end

"""
	_DirectSolvePlan

Causal jobs executed as independent singleton factorisation groups.

# Fields

- `jobs::Vector{_SolveJob}` — solve jobs in graded-lexicographic causal order.
"""
struct _DirectSolvePlan <: _AbstractSolvePlan
    jobs::Vector{_SolveJob}
end

"""
	_SolveGroup{T}

Jobs sharing one canonical superharmonic and exactly the same bordered matrix.

# Fields

- `superharmonic::T` — common value of `s = dot(alpha, lambda_diag)` used to assemble and
  factor the first job.
- `jobs::Vector{_SolveJob}` — jobs that reuse that numeric factorisation in causal order.
"""
struct _SolveGroup{T}
    superharmonic::T
    jobs::Vector{_SolveJob}
end

"""
	_GroupedSolvePlan{T}

Degree-local groups whose jobs share exactly identical bordered matrices.

# Fields

- `groups::Vector{_SolveGroup{T}}` — ordered factor-reuse groups, never spanning degrees.
- `n_jobs::Int` — total primary solves across all groups, used for progress reporting.
"""
struct _GroupedSolvePlan{T} <: _AbstractSolvePlan
    groups::Vector{_SolveGroup{T}}
    n_jobs::Int
end

"""
	StructuralFactorKey{NVAR, ROM}

Exact mathematical identity of a bordered matrix for factor-reuse grouping.

# Fields

- `exponents::SVector{NVAR, Int}` — monomial powers accumulated by equal nonzero
  eigenvalue; powers belonging to zero eigenvalues do not affect the superharmonic.
- `resonance::SVector{ROM, Bool}` — complete master-mode border mask.
"""
struct StructuralFactorKey{NVAR, ROM}
    exponents::SVector{NVAR, Int}
    resonance::SVector{ROM, Bool}
end

"""
	_eigenvalue_representatives(lambda_diag) -> Vector{Int}

Assign equal nonzero eigenvalues the same representative index and assign zero eigenvalues
the sentinel `0`. The result defines the exact exponent aggregation used by factor keys.
"""
function _eigenvalue_representatives(lambda_diag)
    representatives = collect(eachindex(lambda_diag))
    for i in eachindex(lambda_diag)
        if iszero(lambda_diag[i])
            representatives[i] = 0
            continue
        end
        for j in firstindex(lambda_diag):(i - 1)
            if isequal(lambda_diag[i], lambda_diag[j])
                representatives[i] = representatives[j]
                break
            end
        end
    end
    return representatives
end

"""
	_has_structural_factor_reuse(lambda_diag) -> Bool

Return whether zero or repeated eigenvalues make two distinct monomials capable of sharing
an identical superharmonic expression and numeric factorisation.
"""
function _has_structural_factor_reuse(lambda_diag)
    for i in eachindex(lambda_diag)
        iszero(lambda_diag[i]) && return true
        for j in firstindex(lambda_diag):(i - 1)
            isequal(lambda_diag[i], lambda_diag[j]) && return true
        end
    end
    return false
end

"""
	_structural_factor_key(multi, resonance, representatives) -> StructuralFactorKey

Build the exact factor-reuse key from aggregated monomial powers and the full resonance
mask. Equality of these keys implies equality of the bordered matrix, not merely closeness
of the floating-point superharmonics.
"""
function _structural_factor_key(multi::SVector{NVAR, Int},
        resonance::SVector{ROM, Bool}, representatives) where {NVAR, ROM}
    exponents = zeros(MVector{NVAR, Int})
    for i in 1:NVAR
        representative = representatives[i]
        representative == 0 || (exponents[representative] += multi[i])
    end
    return StructuralFactorKey(SVector(exponents), resonance)
end

"""
	_job_target(sym, index) -> Int

Return the larger conjugate-secondary position reconstructed after solving `index`, or `0`
when symmetry is inactive, the monomial is self-conjugate, or no partner is present.
"""
_job_target(::ConjugateSymmetryData{NoConjugatePermutation}, ::Int) = 0
function _job_target(sym::ConjugateSymmetryData{<:SVector}, index::Int)
    target = @inbounds sym.monomial_map[index]
    return target > index ? target : 0
end

"""
	_build_solve_jobs(sym) -> Vector{_SolveJob}

Build jobs from the *current* skip mask. Checkpoint restoration and external-direction
initialisation mutate that mask after symmetry discovery, so jobs must not be cached in
`ConjugateSymmetryData`.
"""
function _build_solve_jobs(sym::ConjugateSymmetryData)
    jobs = _SolveJob[]
    sizehint!(jobs, count(!, sym.skip_bits))
    for index in eachindex(sym.skip_bits)
        @inbounds sym.skip_bits[index] && continue
        push!(jobs, _SolveJob(index, _job_target(sym, index)))
    end
    return jobs
end

"""
	_group_solve_jobs(ctx, mset, jobs, Val(ROM)) -> Vector{_SolveGroup}

Partition causal jobs into degree-local groups with equal [`StructuralFactorKey`](@ref).
Group order follows the first occurrence of each key, and job order is preserved within a
group.
"""
function _group_solve_jobs(ctx, mset, jobs::Vector{_SolveJob}, ::Val{ROM}) where {ROM}
    representatives = _eigenvalue_representatives(ctx.lambda_diag)
    T = eltype(ctx.lambda_diag)
    NVAR = length(ctx.lambda_diag)
    Key = StructuralFactorKey{NVAR, ROM}
    ordered_groups = Vector{_SolveGroup{T}}()

    first_job = firstindex(jobs)
    while first_job <= lastindex(jobs)
        degree = sum(mset[jobs[first_job].index])
        keys = Key[]
        groups = Dict{Key, _SolveGroup{T}}()
        next_job = first_job
        while next_job <= lastindex(jobs) &&
              sum(mset[jobs[next_job].index]) == degree
            job = jobs[next_job]
            resonance = _resonance_vector(ctx.resonance_set, job.index, Val(ROM))
            key = _structural_factor_key(
                mset[job.index], resonance, representatives)
            if !haskey(groups, key)
                groups[key] = _SolveGroup(
                    _superharmonic(mset[job.index], ctx.lambda_diag), _SolveJob[])
                push!(keys, key)
            end
            push!(groups[key].jobs, job)
            next_job += 1
        end
        append!(ordered_groups, (groups[key] for key in keys))
        first_job = next_job
    end
    return ordered_groups
end

"""
	_build_solve_plan(ctx, sym, mset, grouping, Val(ROM)) -> _AbstractSolvePlan

Create a causal direct or grouped plan. Grouping is exact and never crosses a total
degree. `:auto` returns the direct plan unless structural reuse is possible and actually
reduces the number of numeric factorisations.
"""
function _build_solve_plan(ctx, sym, mset, grouping::Symbol, ::Val{ROM}) where {ROM}
    grouping in (:off, :on, :auto) || throw(ArgumentError(
        "grouping must be :auto, :off, or :on"))
    jobs = _build_solve_jobs(sym)
    grouping == :off && return _DirectSolvePlan(jobs)
    grouping == :auto && !_has_structural_factor_reuse(ctx.lambda_diag) &&
        return _DirectSolvePlan(jobs)

    groups = _group_solve_jobs(ctx, mset, jobs, Val(ROM))
    grouping == :auto && length(groups) == length(jobs) &&
        return _DirectSolvePlan(jobs)
    return _GroupedSolvePlan(groups, length(jobs))
end
