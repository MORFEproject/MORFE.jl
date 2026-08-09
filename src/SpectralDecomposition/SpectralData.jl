"""
Module `SpectralDataTypes` — the spectral input to a parametrisation, in one object.

[`SpectralData`](@ref) bundles everything the cohomological solve needs to know about the
spectrum: the master eigenvalues and their right/left eigenvector blocks, the non-master
("outer") eigenvalues that resonance detection reads, and the conjugate involution on the
master block.  It replaces the five loose, mutually-dependent arguments
(`master_eigenvalues`, `master_modes`, `master_modes_derivatives`, `left_eigenmodes`,
`left_modes_derivatives`) that callers previously had to keep consistent by hand.

Master and outer modes are held symmetrically, as two [`ModeBundle`](@ref)s.

## The mirrored index convention

Right and left blocks are both `FOM × ORD × n`, but they index **oppositely**:

| | physical slice | the other blocks |
|---|---|---|
| right | `[:, 1, :]` | `[:, 2:ORD, :]` — time derivatives `ψ_{k+1} = λ ψ_k` |
| left  | `[:, ORD, :]` | `[:, 1:(ORD-1), :]` — orthogonality row operators |

Because both arrays have the same shape, getting this backwards is type-correct and
compiles.  It is therefore encoded in exactly two places and nowhere else: the
`ModeBundle` constructor, which caches the two physical slices, and the four accessors
[`right_modes`](@ref), [`left_modes`](@ref), [`right_mode_derivatives`](@ref) and
[`left_mode_blocks`](@ref).  After construction there is no `[:, 1, :]` or `[:, ORD, :]`
left in the codebase to get wrong.

[`check_biorthogonality`](@ref) is the numerical guard: `φᵣᴴ B ψₛ = δᵣₛ` fails loudly
under any swap.
"""
module SpectralDataTypes

using LinearAlgebra
using StaticArrays: SVector

using ..FullOrderModel: NDOrderModel, linear_first_order_matrices
using ..Eigenproblems: Eigenproblem, left_eigenmode_orders_from_slice
using ..ConjugatePermutation: detect_conjugate_permutation

export ModeBundle, SpectralData,
       right_modes, left_modes, right_mode_derivatives, left_mode_blocks,
       master_eigenvalues, outer_eigenvalues, master_bundle, outer_bundle,
       check_biorthogonality

# =============================================================================
# ModeBundle
# =============================================================================

"""
	ModeBundle{ORD, EV}

One family of modes — eigenvalues plus, optionally, their right and left eigenvector
order-blocks.

`EV` is the eigenvalue container, which is what lets a single type serve both families:
the master bundle uses `SVector{ROM, ComplexF64}` because the solve requires a statically
sized vector, while the outer bundle uses `Vector{ComplexF64}` because its length varies
run to run and must not drive recompilation.

# Fields

- `eigenvalues::EV`
- `right_blocks`, `left_blocks` — `FOM × ORD × n`, or `nothing` when modes were not kept
  (outer bundles default to eigenvalues only).
- `right_physical`, `left_physical` — `FOM × n` **materialised copies** of the physical
  slices.

## Why the physical slices are cached copies, not views

Three reasons, all load-bearing:

1. A `Bool`-mask selection is not strided, so a view of it would drop downstream
   BLAS/sparse products onto the slow generic path — the reason the previous code took
   copies explicitly.
2. `solve_cohomological_problem` types its right modes as a concrete `Matrix{ComplexF64}`.
3. An accessor that sliced would allocate on every call rather than once.

The cost is `2·FOM·n` complex numbers, negligible beside the `FOM × ORD × L`
parametrisation.
"""
struct ModeBundle{ORD, EV <: AbstractVector{ComplexF64}}
    eigenvalues::EV
    right_blocks::Union{Nothing, Array{ComplexF64, 3}}
    left_blocks::Union{Nothing, Array{ComplexF64, 3}}
    right_physical::Union{Nothing, Matrix{ComplexF64}}
    left_physical::Union{Nothing, Matrix{ComplexF64}}

    function ModeBundle{ORD}(eigenvalues::EV,
            right_blocks::Union{Nothing, AbstractArray{<:Complex, 3}},
            left_blocks::Union{Nothing, AbstractArray{<:Complex, 3}}
    ) where {ORD, EV <: AbstractVector{ComplexF64}}
        n = length(eigenvalues)
        rb = right_blocks === nothing ? nothing : Array{ComplexF64, 3}(right_blocks)
        lb = left_blocks === nothing ? nothing : Array{ComplexF64, 3}(left_blocks)
        for (name, b) in (("right", rb), ("left", lb))
            b === nothing && continue
            size(b, 2) == ORD || throw(ArgumentError(
                "$name blocks have $(size(b, 2)) order-slices but ORD = $ORD"))
            size(b, 3) == n || throw(ArgumentError(
                "$name blocks have $(size(b, 3)) modes but there are $n eigenvalues"))
        end
        # THE mirrored convention, written once. Right physical is the FIRST slice,
        # left physical is the LAST.
        rp = rb === nothing ? nothing : Matrix{ComplexF64}(rb[:, 1, :])
        lp = lb === nothing ? nothing : Matrix{ComplexF64}(lb[:, ORD, :])
        return new{ORD, EV}(eigenvalues, rb, lb, rp, lp)
    end
end

"""
	right_modes(b::ModeBundle) -> Matrix{ComplexF64}

The physical-space right eigenvectors, `FOM × n`. Cached at construction; no indexing.
"""
right_modes(b::ModeBundle) = b.right_physical

"""
	left_modes(b::ModeBundle) -> Matrix{ComplexF64}

The physical-space left eigenvectors, `FOM × n` — the **highest**-order block, mirroring
[`right_modes`](@ref), which is the lowest. Cached at construction.
"""
left_modes(b::ModeBundle) = b.left_physical

"""
	right_mode_derivatives(b::ModeBundle) -> view or nothing

Blocks `2:ORD` of the right eigenvectors — the time derivatives `ψ_{k+1} = λ ψ_k`.
`nothing` when `ORD == 1`.
"""
function right_mode_derivatives(b::ModeBundle{ORD}) where {ORD}
    (ORD == 1 || b.right_blocks === nothing) && return nothing
    return @view b.right_blocks[:, 2:ORD, :]
end

"""
	left_mode_blocks(b::ModeBundle) -> view or nothing

Blocks `1:(ORD-1)` of the left eigenvectors — the ones feeding the orthogonality row
operators. `nothing` when `ORD == 1`.
"""
function left_mode_blocks(b::ModeBundle{ORD}) where {ORD}
    (ORD == 1 || b.left_blocks === nothing) && return nothing
    return @view b.left_blocks[:, 1:(ORD - 1), :]
end

Base.length(b::ModeBundle) = length(b.eigenvalues)

# =============================================================================
# SpectralData
# =============================================================================

"""
	SpectralData{ORD, ROM}

The complete spectral input to a parametrisation: a master [`ModeBundle`](@ref) (always
carrying modes), an outer bundle (eigenvalues always, modes optional), and the conjugate
involution on the master block.

Element type is pinned to `ComplexF64` rather than left as a parameter, because the solve
is `ComplexF64` throughout; a generic element type would only create a silent conversion
boundary.

# Fields

- `master::ModeBundle{ORD, SVector{ROM, ComplexF64}}`
- `outer::ModeBundle{ORD, Vector{ComplexF64}}` — the non-master eigenvalues that outer
  resonance detection reads.
- `conjugate_permutation::Union{Nothing, Vector{Int}}` — length `ROM`, the **master block
  only**. The full `NVAR` vector is assembled at solve time from the model's external
  system, because the eigenproblem is generally solved on the autonomous operator before
  the forced model exists.

Mode rescaling for conditioning (e.g. multiplying both sides by `1e-2`) is a caller-side
operation on the raw arrays before construction; there is deliberately no `scale` field.
"""
struct SpectralData{ORD, ROM}
    master::ModeBundle{ORD, SVector{ROM, ComplexF64}}
    outer::ModeBundle{ORD, Vector{ComplexF64}}
    conjugate_permutation::Union{Nothing, Vector{Int}}
end

master_bundle(sd::SpectralData) = sd.master
outer_bundle(sd::SpectralData) = sd.outer
master_eigenvalues(sd::SpectralData) = sd.master.eigenvalues
outer_eigenvalues(sd::SpectralData) = sd.outer.eigenvalues

right_modes(sd::SpectralData) = right_modes(sd.master)
left_modes(sd::SpectralData) = left_modes(sd.master)
right_mode_derivatives(sd::SpectralData) = right_mode_derivatives(sd.master)
left_mode_blocks(sd::SpectralData) = left_mode_blocks(sd.master)

function Base.show(io::IO, sd::SpectralData{ORD, ROM}) where {ORD, ROM}
    print(io, "SpectralData{ORD=$ORD, ROM=$ROM}(", length(sd.outer),
        " outer eigenvalues, ",
        sd.conjugate_permutation === nothing ? "no conjugate permutation" :
        "conjugate_permutation = $(sd.conjugate_permutation)", ")")
end

"""
	SpectralData(; eigenvalues, right_modes, left_modes,
				 outer_eigenvalues = ComplexF64[], conjugate_permutation = nothing)

Build `SpectralData` directly from raw arrays — for eigensolvers that never construct an
`Eigenproblem` (a shift-invert Hopf solve, say).

`right_modes` and `left_modes` are `FOM × ORD × ROM` block arrays, or `FOM × ROM` matrices
when `ORD == 1`.  Remember the mirrored convention: the left array's **last** slice is the
physical one.

`conjugate_permutation` is taken as given here (no `:detect`): a caller assembling raw
arrays is in the best position to know the pairing, and eigenvalue-based detection is not
sufficient on its own.
"""
function SpectralData(; eigenvalues::AbstractVector,
        right_modes::AbstractArray,
        left_modes::AbstractArray,
        outer_eigenvalues::AbstractVector = ComplexF64[],
        conjugate_permutation::Union{Nothing, AbstractVector{Int}} = nothing)
    ROM = length(eigenvalues)
    rb = _as_blocks(right_modes, ROM, "right_modes")
    lb = _as_blocks(left_modes, ROM, "left_modes")
    ORD = size(rb, 2)
    size(lb, 2) == ORD || throw(ArgumentError(
        "right_modes has $ORD order-blocks but left_modes has $(size(lb, 2))"))
    λ = SVector{ROM, ComplexF64}(ComplexF64.(eigenvalues))
    master = ModeBundle{ORD}(λ, rb, lb)
    outer = ModeBundle{ORD}(Vector{ComplexF64}(outer_eigenvalues), nothing, nothing)
    perm = conjugate_permutation === nothing ? nothing : collect(Int, conjugate_permutation)
    _validate_master_permutation(perm, ROM)
    return SpectralData{ORD, ROM}(master, outer, perm)
end

# FOM × n (ORD = 1) or FOM × ORD × n, normalised to the 3-D form.
function _as_blocks(a::AbstractArray, n::Int, what::AbstractString)
    if ndims(a) == 2
        size(a, 2) == n || throw(ArgumentError(
            "$what has $(size(a, 2)) columns but there are $n eigenvalues"))
        return reshape(Array{ComplexF64}(a), size(a, 1), 1, n)
    elseif ndims(a) == 3
        return Array{ComplexF64, 3}(a)
    else
        throw(ArgumentError("$what must be a FOM × n matrix or a FOM × ORD × n array"))
    end
end

function _validate_master_permutation(perm, ROM::Int)
    perm === nothing && return nothing
    length(perm) == ROM || throw(ArgumentError(
        "conjugate_permutation has $(length(perm)) entries but ROM = $ROM. It covers the " *
        "MASTER BLOCK ONLY; the external block is appended at solve time from the model's " *
        "external system."))
    sort(collect(perm)) == collect(1:ROM) || throw(ArgumentError(
        "conjugate_permutation must be a permutation of 1:$ROM, got $(collect(perm))"))
    all(i -> perm[perm[i]] == i, 1:ROM) || throw(ArgumentError(
        "conjugate_permutation must be an involution, got $(collect(perm))"))
    return nothing
end

"""
	SpectralData(model, eigenproblem; master,
				 conjugate_permutation = nothing, keep_outer_modes = false)

Select the master modes out of a solved eigenproblem and reconcile their blocks against
`model`'s order.

`master` is either a `Vector{Int}` of indices or a `Vector{Bool}` mask.

## Three rules that carry correctness weight

**Order is never changed.** `master`'s index order *is* the reduced-coordinate order — it
determines the conjugate permutation, the monomial-set variable roles and the resonance
target numbering. Nothing here sorts; sorting is a property of the eigenproblem, applied
before selection.

**Blocks are sliced when the orders match, extended only when they don't.** If the
eigenproblem was solved on an operator of the same order as `model`, the stored blocks are
used as-is — *not* recomputed, because they share whatever scaling the biorthogonal
normalisation applied and a recomputation can drift from that. When `model` has the higher
order (an augmented `(K, C, M, 0)` fed by a second-order structural eigenproblem), the
missing right blocks are generated by multiplying the **last available** block by `λ` —
not by forming a fresh `λ^{k-1} ψ` — and the left blocks are rebuilt from the physical
slice against `model.linear_terms`.

**`conjugate_permutation` defaults to `nothing`.** Pass `:detect` to derive it from the
master eigenvalues; unlike bare `detect_conjugate_permutation`, that path also **verifies
the eigenvectors** (`Ψ[:, σ(r)] ≈ conj(Ψ[:, r])` on every order-block, both sides) and
returns `nothing` with an `@info` if they disagree — eigenvalue pairing alone is necessary
but not sufficient, and a wrong permutation silently corrupts `W` and `R`. The default is
`nothing` so that enabling conjugate symmetry is always a deliberate act.
"""
function SpectralData(model::NDOrderModel, eigenproblem::Eigenproblem;
        master,
        conjugate_permutation = nothing,
        keep_outer_modes::Bool = false,
        atol::Real = 1e-8)
    n_eigs = size(eigenproblem.eigenmodes, 3)
    idx = _master_indices(master, n_eigs)
    ROM = length(idx)
    ROM > 0 || throw(ArgumentError("no master modes selected"))
    allunique(idx) || throw(ArgumentError("master indices must be distinct, got $idx"))

    ORD_model = length(model.linear_terms) - 1
    ORD_spec = size(eigenproblem.eigenmodes, 2)
    ORD_spec <= ORD_model || throw(ArgumentError(
        "the eigenproblem carries $ORD_spec order-blocks but the model has ORD = $ORD_model; " *
        "a higher-order eigenproblem cannot be reconciled to a lower-order model"))

    λ_master = SVector{ROM, ComplexF64}(ComplexF64.(eigenproblem.eigenvalues[idx]))

    right = _reconcile_right(eigenproblem, idx, λ_master, ORD_spec, ORD_model)
    left = _reconcile_left(model, eigenproblem, idx, λ_master, ORD_spec, ORD_model)

    master_b = ModeBundle{ORD_model}(λ_master, right, left)

    outer_idx = setdiff(1:n_eigs, idx)
    λ_outer = Vector{ComplexF64}(eigenproblem.eigenvalues[outer_idx])
    outer_b = if keep_outer_modes && !isempty(outer_idx)
        ro = _reconcile_right(eigenproblem, outer_idx,
            ComplexF64.(eigenproblem.eigenvalues[outer_idx]), ORD_spec, ORD_model)
        lo = _reconcile_left(model, eigenproblem, outer_idx,
            ComplexF64.(eigenproblem.eigenvalues[outer_idx]), ORD_spec, ORD_model)
        ModeBundle{ORD_model}(λ_outer, ro, lo)
    else
        ModeBundle{ORD_model}(λ_outer, nothing, nothing)
    end

    perm = _resolve_permutation(conjugate_permutation, λ_master, right, left, ROM, atol)
    return SpectralData{ORD_model, ROM}(master_b, outer_b, perm)
end

function _master_indices(master::AbstractVector{Bool}, n_eigs::Int)
    (
        length(master) == n_eigs ||
            throw(ArgumentError("master mask has $(length(master)) entries but there are $n_eigs modes"));
        findall(master))
end
function _master_indices(master::AbstractVector{<:Integer}, n_eigs::Int)
    (
        all(i -> 1 <= i <= n_eigs, master) ||
            throw(ArgumentError("master indices $master out of range 1:$n_eigs"));
        collect(Int, master))
end

# Right blocks: slice, then extend by λ · (last available block) if the model has a
# higher order than the eigenproblem.
function _reconcile_right(ep, idx, λ, ORD_spec::Int, ORD_model::Int)
    FOM = size(ep.eigenmodes, 1)
    blocks = Array{ComplexF64}(undef, FOM, ORD_model, length(idx))
    for (r, i) in enumerate(idx), k in 1:ORD_spec

        blocks[:, k, r] .= @view ep.eigenmodes[:, k, i]
    end
    for r in eachindex(idx), k in (ORD_spec + 1):ORD_model
        # λ · the block just below, NOT a fresh λ^{k-1}ψ: the eigensolver's own block
        # carries its numerical content and must be the thing that is scaled.
        @views blocks[:, k, r] .= λ[r] .* blocks[:, k - 1, r]
    end
    return blocks
end

# Left blocks: slice when the orders match; otherwise rebuild from the physical slice
# against the MODEL's linear terms (the augmented tuple), which is what defines the
# orthogonality operators at the model's order.
function _reconcile_left(model, ep, idx, λ, ORD_spec::Int, ORD_model::Int)
    if ORD_spec == ORD_model
        ep.left_eigenmodes_orders === nothing && throw(ArgumentError("""
            The eigenproblem stores only the physical left slice, but ORD = $ORD_model needs
            the full left order-blocks. Use an eigensolver that returns them, or rebuild with
            `left_eigenmode_orders_from_slice(model.linear_terms, slice, eigenvalues)`.
            """))
        return Array{ComplexF64, 3}(ep.left_eigenmodes_orders[:, :, idx])
    end
    slice = Matrix{ComplexF64}(ep.left_eigenmodes[:, idx])
    return left_eigenmode_orders_from_slice(model.linear_terms, slice, collect(λ))
end

_resolve_permutation(::Nothing, args...) = nothing
function _resolve_permutation(perm::AbstractVector{Int}, λ, right, left, ROM, atol)
    (_validate_master_permutation(collect(Int, perm), ROM); collect(Int, perm))
end

function _resolve_permutation(sym::Symbol, λ, right, left, ROM, atol)
    sym === :detect || throw(ArgumentError(
        "conjugate_permutation must be `nothing`, `:detect`, or a Vector{Int}; got :$sym"))
    σ = detect_conjugate_permutation(collect(λ); atol = atol)
    if σ === nothing
        @info "conjugate_permutation = :detect — the master eigenvalues are not closed " *
              "under conjugation, so no permutation was derived. Proceeding without " *
              "conjugate symmetry."
        return nothing
    end
    # Eigenvalue pairing is necessary but NOT sufficient: verify the vectors, on every
    # order-block and on both sides, exactly as external_conjugate_permutation verifies
    # the external basis columns.
    for (name, blocks) in (("right", right), ("left", left))
        blocks === nothing && continue
        for r in 1:ROM, k in axes(blocks, 2)

            if !isapprox(@view(blocks[:, k, σ[r]]), conj(@view(blocks[:, k, r])); atol = atol)
                @info "conjugate_permutation = :detect — the master eigenvalues pair up, " *
                      "but the $name eigenvectors do not satisfy Ψ[:, σ(r)] = conj(Ψ[:, r]) " *
                      "(mode $r, order-block $k). Proceeding without conjugate symmetry; " *
                      "pass an explicit permutation to override."
                return nothing
            end
        end
    end
    return collect(Int, σ)
end

"""
	check_biorthogonality(sd::SpectralData, model) -> Matrix{ComplexF64}

Return the master-block biorthogonality matrix `G[r, s] = φᵣᴴ B ψₛ`, which should be the
identity for biorthogonally normalised eigenvectors.

This is the numerical guard on the mirrored right/left index convention: swapping the
physical slice for a derivative block, or the two sides for each other, destroys `G ≈ I`
loudly, whatever accessor was misused.

**Diagnostic only** — deliberately not called from `parametrise` or the solve, so it adds
no cost to a normal run. Call it in tests, or when a result looks wrong.
"""
function check_biorthogonality(sd::SpectralData{ORD, ROM}, model::NDOrderModel) where {
        ORD, ROM}
    _, B = linear_first_order_matrices(model)
    right = sd.master.right_blocks
    left = sd.master.left_blocks
    (right === nothing || left === nothing) &&
        throw(ArgumentError("check_biorthogonality needs both right and left blocks"))
    G = Matrix{ComplexF64}(undef, ROM, ROM)
    for r in 1:ROM, s in 1:ROM

        φ = vec(@view left[:, :, r])
        ψ = vec(@view right[:, :, s])
        G[r, s] = dot(φ, B * ψ)
    end
    return G
end

end # module
