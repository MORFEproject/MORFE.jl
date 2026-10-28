# Causal monomial jobs, their data dependencies, and exact structural factor groups.

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
	_CrossDegreeSolvePlan{T}

Factor-reuse groups that may span total degrees. Each group is a maximal run of
consecutive jobs with one [`StructuralFactorKey`](@ref), ordered by a topological sort of
the jobs' data dependencies, so every coefficient a job reads is final before it runs.

# Fields

- `groups::Vector{_SolveGroup{T}}` — ordered factor-reuse groups; one group may hold jobs
  of several degrees.
- `n_jobs::Int` — total primary solves across all groups, used for progress reporting.
- `jobs_per_degree::Dict{Int, Int}` — number of jobs of each total degree, so the executor
  can report a degree complete when its last job has run, whichever group holds it.
"""
struct _CrossDegreeSolvePlan{T} <: _AbstractSolvePlan
    groups::Vector{_SolveGroup{T}}
    n_jobs::Int
    jobs_per_degree::Dict{Int, Int}
end

"""Return the number of primary jobs a solve plan executes."""
_plan_job_count(plan::_DirectSolvePlan) = length(plan.jobs)
_plan_job_count(plan::Union{_GroupedSolvePlan, _CrossDegreeSolvePlan}) = plan.n_jobs

"""Return the number of numeric factorisations a solve plan performs."""
_plan_factorisation_count(plan::_DirectSolvePlan) = length(plan.jobs)
function _plan_factorisation_count(plan::Union{_GroupedSolvePlan, _CrossDegreeSolvePlan})
    length(plan.groups)
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
	_job_positions(jobs, L) -> Vector{Int}

Map each of the `L` multiindex-set positions to the number of the job that makes it final:
the job solving it, or the job whose conjugate reconstruction fills it. Positions that no
job touches (linear, restored or external-direction monomials) map to `0`, meaning
available before the first job.
"""
function _job_positions(jobs::Vector{_SolveJob}, L::Int)
    positions = zeros(Int, L)
    for (number, job) in enumerate(jobs)
        positions[job.index] = number
        job.conjugate_target == 0 || (positions[job.conjugate_target] = number)
    end
    return positions
end

"""
	_structurally_nonzero(reduced, resonance_set, i, index, ROM) -> Bool

Return whether `R[i, index]` may be nonzero once the solve is complete. Master rows of a
non-resonant monomial are exact zeros by the style choice, so the resonance mask decides
them; every other entry (external rows, and anything already final) is read directly.
"""
function _structurally_nonzero(reduced, resonance_set, i::Int, index::Int, ROM::Int)
    i <= ROM && is_resonant(resonance_set, index, i) && return true
    return !iszero(reduced[i, index])
end

"""
	_job_dependencies(ctx, R, mset, jobs, Val(ROM)) -> Vector{Vector{Int}}

For each job, the sorted numbers of the earlier jobs whose coefficients it reads. The
reads are those of the monomial pipeline:

- the nonlinear right-hand side reads every componentwise-smaller monomial, and the
  lower-order coupling reads `R[:, β]` for every such `β` of degree at least two; these are
  exactly `ctx.lower_order.candidate_indices`;
- the lower-order coupling also reads `W[γ − β + eᵢ]` wherever `R[i, β]` is structurally
  nonzero, with `|β| = 1` restricted to `i < j` exactly as in `_sum_degree_one_terms!`.

`R` must hold the final external rows and degree-one coefficients, which is the case once
the external directions are solved. A job never depends on itself or on positions marked
available by [`_job_positions`](@ref).
"""
function _job_dependencies(ctx, R, mset::MultiindexSet{NVAR}, jobs::Vector{_SolveJob},
        ::Val{ROM}) where {NVAR, ROM}
    positions = _job_positions(jobs, length(mset))
    dictionary = ctx.lower_order.multiindex_dict
    candidates = ctx.lower_order.candidate_indices
    units = ctx.lower_order.unit_vectors
    reduced = R.poly.coefficients
    dependencies = Vector{Vector{Int}}(undef, length(jobs))
    for (number, job) in enumerate(jobs)
        found = Int[]
        gamma = mset[job.index]
        for j in 1:NVAR
            gamma[j] >= 1 || continue
            unit = get(dictionary, units[j], 0)
            unit == 0 && continue
            for i in 1:(j - 1)
                iszero(reduced[i, unit]) && continue
                target = get(dictionary, gamma - units[j] + units[i], 0)
                target == 0 || push!(found, positions[target])
            end
        end
        for beta_index in candidates[job.index]
            push!(found, positions[beta_index])
            beta = mset[beta_index]
            for i in 1:NVAR
                _structurally_nonzero(reduced, ctx.resonance_set, i, beta_index, ROM) ||
                    continue
                target = get(dictionary, gamma - beta + units[i], 0)
                target == 0 || push!(found, positions[target])
            end
        end
        dependencies[number] = filter!(p -> p != 0 && p != number, unique!(sort!(found)))
    end
    return dependencies
end

"""
	_schedule_key_runs(keys, key_priority, dependencies) -> Union{Nothing, Vector{Vector{Int}}}

Topologically sort job numbers `1:n` so that jobs sharing a key run consecutively whenever
the dependencies allow, and return the maximal same-key runs in execution order.

`keys[k]` is the integer key of job `k` and `key_priority[key]` its rank (lower opens
first). The sort is Kahn's algorithm with two preferences. While a ready job carries the
current key, the smallest-numbered such job runs next. Otherwise a new run opens, on the
key of lowest priority among those that no unfinished job of another key holds back,
which therefore finish in one run; only if there is none, on the ready key of lowest
priority. Returns `nothing` if the dependencies contain a cycle.
"""
function _schedule_key_runs(keys::Vector{Int}, key_priority::Vector{Int},
        dependencies::Vector{Vector{Int}})
    n = length(keys)
    n_keys = length(key_priority)
    successors = [Int[] for _ in 1:n]
    pending = zeros(Int, n)
    # unfinished dependencies of each key's jobs on jobs of other keys
    foreign = zeros(Int, n_keys)
    for number in 1:n, dependency in dependencies[number]

        push!(successors[dependency], number)
        pending[number] += 1
        keys[dependency] == keys[number] || (foreign[keys[number]] += 1)
    end
    ready = [BitSet() for _ in 1:n_keys]
    for number in 1:n
        pending[number] == 0 && push!(ready[keys[number]], number)
    end
    keys_by_priority = sortperm(key_priority)

    runs = Vector{Vector{Int}}()
    current = 0
    scheduled = 0
    while scheduled < n
        if current == 0 || isempty(ready[current])
            next = findfirst(key -> !isempty(ready[key]) && foreign[key] == 0,
                keys_by_priority)
            next === nothing &&
                (next = findfirst(key -> !isempty(ready[key]), keys_by_priority))
            next === nothing && return nothing
            current = keys_by_priority[next]
            push!(runs, Int[])
        end
        number = first(ready[current])
        delete!(ready[current], number)
        push!(last(runs), number)
        scheduled += 1
        for successor in successors[number]
            pending[successor] -= 1
            keys[successor] == current || (foreign[keys[successor]] -= 1)
            pending[successor] == 0 && push!(ready[keys[successor]], successor)
        end
    end
    return runs
end

"""
	_group_solve_jobs(ctx, mset, jobs, dependencies, Val(ROM); across_degrees = false)
		-> Union{Nothing, Vector{_SolveGroup}}

Partition jobs into factor-reuse groups of equal [`StructuralFactorKey`](@ref), ordered so
that every job runs after its `dependencies` ([`_job_dependencies`](@ref)). The
topological sort [`_schedule_key_runs`](@ref) keeps a group together wherever the
dependencies allow and splits it only where one requires it.

By default each total degree is scheduled on its own, with keys opened in order of first
occurrence, so the groups complete each degree before the next. Where no dependency
intervenes, this gives one group per key and degree.

With `across_degrees = true` a group may span degrees. Keys are then opened in order of
their aggregated degree, then first occurrence, which runs every master key in one group
when the only zero eigenvalues are frozen parameters. Returns `nothing` when the
dependency graph cannot be ordered.
"""
function _group_solve_jobs(ctx, mset, jobs::Vector{_SolveJob},
        dependencies::Vector{Vector{Int}}, ::Val{ROM};
        across_degrees::Bool = false) where {ROM}
    representatives = _eigenvalue_representatives(ctx.lambda_diag)
    T = eltype(ctx.lambda_diag)
    NVAR = length(ctx.lambda_diag)
    structural_keys = [_structural_factor_key(mset[job.index],
                           _resonance_vector(ctx.resonance_set, job.index, Val(ROM)),
                           representatives) for job in jobs]
    # Within degrees, each degree is scheduled on its own: its jobs are contiguous in the
    # causal order, and their dependencies on lower degrees are already met.
    blocks = across_degrees ? [1:length(jobs)] :
             _degree_blocks([sum(mset[job.index]) for job in jobs])
    runs = Vector{Vector{Int}}()
    for block in blocks
        key_ids = Dict{StructuralFactorKey{NVAR, ROM}, Int}()
        keys = Vector{Int}(undef, length(block))
        key_priority_data = Tuple{Int, Int}[]
        for (local_number, number) in enumerate(block)
            key = structural_keys[number]
            keys[local_number] = get!(key_ids, key) do
                push!(key_priority_data,
                    (across_degrees ? sum(key.exponents) : 0, local_number))
                length(key_priority_data)
            end
        end
        offset = first(block) - 1
        local_dependencies = [[d - offset for d in dependencies[number] if d in block]
                              for number in block]
        block_runs = _schedule_key_runs(
            keys, invperm(sortperm(key_priority_data)), local_dependencies)
        block_runs === nothing && return nothing
        append!(runs, [run .+ offset for run in block_runs])
    end
    return [_SolveGroup(_superharmonic(mset[jobs[first(run)].index], ctx.lambda_diag),
                jobs[run]) for run in runs]::Vector{_SolveGroup{T}}
end

"""
	_degree_blocks(degrees) -> Vector{UnitRange{Int}}

Split positions `1:length(degrees)` into maximal runs of equal degree.
"""
function _degree_blocks(degrees::Vector{Int})
    blocks = UnitRange{Int}[]
    start = 1
    for k in 2:(length(degrees) + 1)
        if k > length(degrees) || degrees[k] != degrees[start]
            push!(blocks, start:(k - 1))
            start = k
        end
    end
    return blocks
end

"""
	_build_solve_plan(ctx, R, sym, mset, grouping, Val(ROM)) -> _AbstractSolvePlan

Create a causal direct or grouped plan.

- `:off` returns the direct graded-lexicographic plan.
- `:on` groups equal keys within each total degree.
- `:across_degrees` lets groups span degrees.
- `:auto` returns, among the direct, within-degree and cross-degree plans, the one with the
  fewest numeric factorisations, preferring the degree-ordered plans on ties. Without zero
  or repeated eigenvalues no reuse is possible, and it returns the direct plan at once.

Both groupings come from [`_group_solve_jobs`](@ref), which follows the jobs' data
dependencies ([`_job_dependencies`](@ref)) and splits a group only where one of them
requires it. Within a degree that happens when the reduced linear dynamics couple
variables with distinct eigenvalues.
"""
function _build_solve_plan(ctx, R, sym, mset, grouping::Symbol, ::Val{ROM}) where {ROM}
    grouping in (:off, :on, :auto, :across_degrees) || throw(ArgumentError(
        "grouping must be :auto, :off, :on, or :across_degrees"))
    jobs = _build_solve_jobs(sym)
    grouping == :off && return _DirectSolvePlan(jobs)
    grouping in (:auto, :across_degrees) &&
        !_has_structural_factor_reuse(ctx.lambda_diag) &&
        return _DirectSolvePlan(jobs)

    dependencies = _job_dependencies(ctx, R, mset, jobs, Val(ROM))
    within = grouping == :across_degrees ? nothing :
             _group_solve_jobs(ctx, mset, jobs, dependencies, Val(ROM))
    across = grouping == :on ? nothing :
             _group_solve_jobs(ctx, mset, jobs, dependencies, Val(ROM);
        across_degrees = true)
    if grouping == :on
        within === nothing && return _DirectSolvePlan(jobs)
        return _GroupedSolvePlan(within, length(jobs))
    elseif grouping == :across_degrees
        across === nothing && return _DirectSolvePlan(jobs)
        return _cross_degree_plan(across, jobs, mset)
    end

    # Each group costs one numeric factorisation and each direct job one.
    n_within = within === nothing ? length(jobs) : length(within)
    across !== nothing && length(across) < n_within &&
        return _cross_degree_plan(across, jobs, mset)
    n_within < length(jobs) && return _GroupedSolvePlan(within, length(jobs))
    return _DirectSolvePlan(jobs)
end

"""
	_cross_degree_plan(groups, jobs, mset) -> _CrossDegreeSolvePlan

Wrap cross-degree `groups` with the number of jobs per total degree, which lets the plan
report a degree complete when its last job runs.
"""
function _cross_degree_plan(groups, jobs::Vector{_SolveJob}, mset)
    jobs_per_degree = Dict{Int, Int}()
    for job in jobs
        degree = sum(mset[job.index])
        jobs_per_degree[degree] = get(jobs_per_degree, degree, 0) + 1
    end
    return _CrossDegreeSolvePlan(groups, length(jobs), jobs_per_degree)
end
