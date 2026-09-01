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
2. `solve_parametrisation` types its right modes as a concrete `Matrix{ComplexF64}`.
3. An accessor that sliced would allocate on every call rather than once.

The cost is `2·FOM·n` complex numbers, negligible beside the `FOM × ORD × L`
parametrisation.
"""
struct ModeBundle{ORD, EV <: AbstractVector{ComplexF64}, RD, LD}
    eigenvalues::EV
    right_blocks::Union{Nothing, Array{ComplexF64, 3}}
    left_blocks::Union{Nothing, Array{ComplexF64, 3}}
    right_physical::Union{Nothing, Matrix{ComplexF64}}
    left_physical::Union{Nothing, Matrix{ComplexF64}}
    # The derivative/order sub-blocks, cached as concretely-typed views. Building them per
    # call allocated a SubArray on every solve; the type parameters keep the fields
    # concrete so reading them is free.
    right_derivatives::RD
    left_order_blocks::LD
    indices::Vector{Int}

    function ModeBundle{ORD}(eigenvalues::EV,
            right_blocks::Union{Nothing, AbstractArray{<:Complex, 3}},
            left_blocks::Union{Nothing, AbstractArray{<:Complex, 3}},
            indices::AbstractVector{<:Integer} = 1:length(eigenvalues)
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
        length(indices) == n || throw(ArgumentError(
            "indices has $(length(indices)) entries but there are $n eigenvalues"))
        # THE mirrored convention, written once. Right physical is the FIRST slice,
        # left physical is the LAST.
        rp = rb === nothing ? nothing : Matrix{ComplexF64}(rb[:, 1, :])
        lp = lb === nothing ? nothing : Matrix{ComplexF64}(lb[:, ORD, :])
        # Same convention, other end: right derivatives are blocks 2:ORD, left order-blocks
        # are 1:ORD-1. Empty for ORD == 1, which the accessors report as `nothing`.
        rd = (rb === nothing || ORD == 1) ? nothing : @view rb[:, 2:ORD, :]
        ld = (lb === nothing || ORD == 1) ? nothing : @view lb[:, 1:(ORD - 1), :]
        return new{ORD, EV, typeof(rd), typeof(ld)}(
            eigenvalues, rb, lb, rp, lp, rd, ld, collect(Int, indices))
    end
end

"""
	indices(b::ModeBundle) -> Vector{Int}

The positions this bundle's modes occupy in the **source spectrum**.

Selecting master modes discards the original numbering, so it is recorded here. Two things
need it: restricting the spectrum-wide conjugate involution to this bundle, and reporting a
mode by the entry a user would index in their own spectrum. Conjugate partners need not be
adjacent, so these are not recoverable by arithmetic.
"""
indices(b::ModeBundle) = b.indices

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
right_mode_derivatives(b::ModeBundle) = b.right_derivatives

"""
	left_mode_blocks(b::ModeBundle) -> view or nothing

Blocks `1:(ORD-1)` of the left eigenvectors — the ones feeding the orthogonality row
operators. `nothing` when `ORD == 1`.
"""
left_mode_blocks(b::ModeBundle) = b.left_order_blocks

Base.length(b::ModeBundle) = length(b.eigenvalues)

# =============================================================================
# SpectralData
# =============================================================================

"""
	SpectralData{ORD, ROM}

The complete spectral input to a parametrisation: a master [`ModeBundle`](@ref) (always
carrying modes), an outer bundle (eigenvalues always, modes optional), and the conjugate
structure of the spectrum they were selected from.

Element type is pinned to `ComplexF64` rather than left as a parameter, because the solve
is `ComplexF64` throughout; a generic element type would only create a silent conversion
boundary.

# Fields

- `master::ModeBundle{ORD, SVector{ROM, ComplexF64}}`
- `outer::ModeBundle{ORD, Vector{ComplexF64}}` — the eigenvalues left off the manifold, which
  outer resonance detection reads.
- `conjugate_permutation::Union{Nothing, Vector{Int}}` — the involution `σ` over the **whole
  spectrum**, `1:n_eigs`, with `λ[σ(i)] = conj(λ[i])`; `nothing` when the spectrum has no
  conjugate structure or none was requested.
- `master_permutation`, `outer_permutation` — `σ` restricted to each bundle and re-indexed,
  computed once in the constructor. Read them through
  [`master_conjugate_permutation`](@ref) / [`outer_conjugate_permutation`](@ref).
- `mode_numbers::Vector{Int}` — physical mode number of each spectrum entry, derived from
  `σ`'s orbits (see below).

## One involution, detected once — and each restriction computed once

Conjugacy is a property of the *spectrum*, not of a bundle: master and outer modes are
subsets of one index set. Detecting per bundle would establish the same fact twice, so `σ`
is detected once over all eigenvalues.

The restrictions are then computed **once, in the constructor, and stored** — not derived per
call. The master restriction is what every solve reads, so recomputing it would put an
avoidable allocation on that path. Whether `σ` was detected or the master pairing was supplied
explicitly, the result is settled at construction and thereafter only read.

A restriction is well defined only if the index set is **closed under `σ`** — both members of
a pair selected, or neither. Splitting a pair across the master/outer boundary is rejected at
construction, where the offending entry can be named, rather than surfacing later inside the
solve.

Mode rescaling for conditioning (e.g. multiplying both sides by `1e-2`) is a caller-side
operation on the raw arrays before construction; there is deliberately no `scale` field.
"""
struct SpectralData{ORD, ROM}
    master::ModeBundle{ORD, SVector{ROM, ComplexF64}}
    outer::ModeBundle{ORD, Vector{ComplexF64}}
    conjugate_permutation::Union{Nothing, Vector{Int}}
    master_permutation::Union{Nothing, Vector{Int}}
    outer_permutation::Union{Nothing, Vector{Int}}
    mode_numbers::Vector{Int}

    # The only constructor: takes the canonical facts (bundles + σ) and settles everything
    # derived from them here, once. Restriction failures are raised at this point, where the
    # offending spectrum entry can be named.
    function SpectralData{ORD, ROM}(master::ModeBundle{ORD, SVector{ROM, ComplexF64}},
            outer::ModeBundle{ORD, Vector{ComplexF64}},
            σ::Union{Nothing, AbstractVector{Int}},
            n_eigs::Int) where {ORD, ROM}
        σv = σ === nothing ? nothing : collect(Int, σ)
        mp = _restrict_permutation(σv, master.indices, "master")
        op = _restrict_permutation(σv, outer.indices, "outer")
        return new{ORD, ROM}(master, outer, σv, mp, op, _mode_numbers(σv, n_eigs))
    end
end

master_bundle(sd::SpectralData) = sd.master
outer_bundle(sd::SpectralData) = sd.outer
master_eigenvalues(sd::SpectralData) = sd.master.eigenvalues
outer_eigenvalues(sd::SpectralData) = sd.outer.eigenvalues

right_modes(sd::SpectralData) = right_modes(sd.master)
left_modes(sd::SpectralData) = left_modes(sd.master)
right_mode_derivatives(sd::SpectralData) = right_mode_derivatives(sd.master)
left_mode_blocks(sd::SpectralData) = left_mode_blocks(sd.master)

# ── Physical mode numbering ──────────────────────────────────────────────────

"""
	_mode_numbers(σ, n_eigs) -> Vector{Int}

Number the physical modes of a spectrum: entry `i` gets `out[i]`, the index of the conjugate
pair it belongs to.

The pairs are the orbits of `σ`, numbered by **first appearance** — walk the spectrum in
order and, on reaching an unvisited entry, assign the next number to it and to `σ(i)`. A
self-paired entry (`σ(i) = i`, a real eigenvalue) is its own mode.

This deliberately does **not** compute `⌈i/2⌉`. That formula assumes the eigensolver emits
conjugate partners adjacently, which is a convention of some solvers rather than a fact about
spectra — a shift-invert or filtered solve can return `{1, 5}` as a pair. Deriving the
numbering from `σ` agrees with the adjacent case and stays correct otherwise.

With no conjugate structure (`σ === nothing`), every entry is its own mode.
"""
function _mode_numbers(σ::Union{Nothing, AbstractVector{Int}}, n_eigs::Int)
    σ === nothing && return collect(1:n_eigs)
    out = zeros(Int, n_eigs)
    next = 0
    for i in 1:n_eigs
        out[i] == 0 || continue
        next += 1
        out[i] = next
        out[σ[i]] = next          # no-op when σ[i] == i (a real, self-paired mode)
    end
    return out
end

"""
	physical_mode(sd::SpectralData, i::Integer) -> Int

The physical mode number of spectrum entry `i` — conjugate partners share a number.
"""
physical_mode(sd::SpectralData, i::Integer) = sd.mode_numbers[i]

"""
	spectrum_entries(sd::SpectralData, p::Integer) -> Vector{Int}

The spectrum entries making up physical mode `p` — one for a real mode, two for a conjugate
pair. **Not necessarily consecutive**, which is why they are looked up rather than computed.
"""
spectrum_entries(sd::SpectralData, p::Integer) = findall(==(p), sd.mode_numbers)

"""
	master_conjugate_permutation(sd) -> Union{Nothing, Vector{Int}}

`σ` restricted to the master modes and re-indexed to `1:ROM` — the form the cohomological
solve consumes, before it is extended over the external variables.

Computed once in the constructor; this is a field read. Every solve consults it, so it is
not something to re-derive per call.
"""
master_conjugate_permutation(sd::SpectralData) = sd.master_permutation

"""
	outer_conjugate_permutation(sd) -> Union{Nothing, Vector{Int}}

`σ` restricted to the outer modes and re-indexed to `1:n_outer`. Used to group conjugates
when reporting off-manifold near-resonances.

Computed once in the constructor; this is a field read.
"""
outer_conjugate_permutation(sd::SpectralData) = sd.outer_permutation

# Restrict a spectrum-wide involution to a subset, re-indexed to 1:length(subset).
# Well defined only when the subset is closed under σ.
function _restrict_permutation(σ::Union{Nothing, AbstractVector{Int}},
        idx::AbstractVector{Int}, what::AbstractString)
    σ === nothing && return nothing
    isempty(idx) && return Int[]
    out = Vector{Int}(undef, length(idx))
    # Linear scan rather than a Dict: this runs once per solve on a handful of master
    # modes, where building a hash table costs more than it saves — and it kept the
    # per-solve setup allocation flat.
    for (l, g) in enumerate(idx)
        partner = findfirst(==(σ[g]), idx)
        partner === nothing && throw(ArgumentError(
            "the $what modes are not closed under the conjugate involution: spectrum entry " *
            "$g is included but its conjugate partner $(σ[g]) is not. Select both or " *
            "neither — a half-pair has no conjugate symmetry to exploit."))
        out[l] = partner
    end
    return out
end

function Base.show(io::IO, sd::SpectralData{ORD, ROM}) where {ORD, ROM}
    print(io, "SpectralData{ORD=$ORD, ROM=$ROM}(", length(sd.outer),
        " outer eigenvalues, ",
        sd.conjugate_permutation === nothing ? "no conjugate structure" :
        "conjugate pairs detected", ")")
end

"""
	SpectralData(; eigenvalues, right_modes, left_modes,
				 right_derivatives = nothing, left_blocks = nothing,
				 outer_eigenvalues = ComplexF64[], conjugate_permutation = nothing)

Build `SpectralData` directly from raw arrays — for eigensolvers that never construct a
`Spectrum` (a shift-invert Hopf solve, say).

Two shapes are accepted:

- **Whole blocks.** `right_modes` and `left_modes` are `FOM × ORD × ROM` arrays (or
  `FOM × ROM` matrices when `ORD == 1`), with `right_derivatives` and `left_blocks` left
  `nothing`. Remember the mirrored convention: the left array's **last** slice is the
  physical one.
- **Physical slices plus their companions.** `right_modes` and `left_modes` are the
  `FOM × ROM` physical slices, `right_derivatives` is `FOM × (ORD-1) × ROM` holding
  `W^(k)[eᵣ]`, and `left_blocks` is `FOM × (ORD-1) × ROM` holding the lower-order left
  blocks `φ_{r,j}`. This is the shape callers of the old positional solve already had, and
  taking it here means the mirrored convention is applied in **one** place instead of at
  every call site — where a swap is type-correct and compiles silently.

`conjugate_permutation` is taken as given here (no `:detect`): a caller assembling raw
arrays is in the best position to know the pairing, and eigenvalue-based detection is not
sufficient on its own.

It is accepted at **either of two lengths**:

- `ROM` — the master block. Outer entries are left self-paired, the honest statement that
  raw arrays say nothing about their conjugate structure.
- `ROM + length(outer_eigenvalues)` — the involution over the whole synthetic spectrum,
  used verbatim. Pass this when the outer modes' pairing IS known, so that
  `physical_mode` numbers physical modes rather than eigenvalues and a conjugate pair
  among the outer targets reports once rather than twice. A real (non-oscillatory) outer
  mode is its own conjugate and maps to itself.
"""
function SpectralData(; eigenvalues::AbstractVector,
        right_modes::AbstractArray,
        left_modes::AbstractArray,
        right_derivatives::Union{Nothing, AbstractArray} = nothing,
        left_blocks::Union{Nothing, AbstractArray} = nothing,
        outer_eigenvalues::AbstractVector = ComplexF64[],
        conjugate_permutation::Union{Nothing, AbstractVector{Int}} = nothing)
    ROM = length(eigenvalues)
    rb = _stack_blocks(right_modes, right_derivatives, ROM, :right)
    lb = _stack_blocks(left_modes, left_blocks, ROM, :left)
    ORD = size(rb, 2)
    size(lb, 2) == ORD || throw(ArgumentError(
        "right_modes has $ORD order-blocks but left_modes has $(size(lb, 2))"))
    λ = SVector{ROM, ComplexF64}(ComplexF64.(eigenvalues))
    n_out = length(outer_eigenvalues)
    # Raw arrays carry no spectrum of their own, so define one: masters occupy entries
    # 1:ROM and the outer eigenvalues follow. Those positions are what `physical_mode`
    # and the off-manifold warning report.
    master = ModeBundle{ORD}(λ, rb, lb, 1:ROM)
    outer = ModeBundle{ORD}(Vector{ComplexF64}(outer_eigenvalues), nothing, nothing,
        (ROM + 1):(ROM + n_out))
    perm = conjugate_permutation === nothing ? nothing : collect(Int, conjugate_permutation)
    # Accepted at either length, as in `_resolve_permutation`: a ROM-length MASTER block
    # (outer entries then self-paired), or the involution over the whole synthetic
    # spectrum `1:(ROM + n_out)`, used verbatim. The second is what lets a caller state
    # the pairing of its OUTER modes — without it every outer entry is its own physical
    # mode, and a conjugate pair among them warns twice instead of once.
    σ = if perm !== nothing && n_out > 0 && length(perm) == ROM + n_out
        _validate_spectrum_permutation(perm, ROM + n_out)
        perm
    else
        _validate_master_permutation(perm, ROM)
        _embed_master_permutation(perm, ROM, ROM + n_out)
    end
    return SpectralData{ORD, ROM}(master, outer, σ, ROM + n_out)
end

# Widen a ROM-length master-block involution to the full spectrum, fixing the outer entries.
# The master block occupies 1:ROM here by construction, so this is a straight copy; the
# outer entries are self-paired, which is the honest statement that nothing is known about
# their conjugate structure from raw arrays alone.
function _embed_master_permutation(perm::Union{Nothing, Vector{Int}}, ROM::Int, n_eigs::Int)
    perm === nothing && return nothing
    σ = collect(1:n_eigs)
    σ[1:ROM] .= perm
    return σ
end

# FOM × n (ORD = 1) or FOM × ORD × n, normalised to the 3-D form.
"""
	_stack_blocks(physical, companions, ROM, side) -> Array{ComplexF64, 3}

Assemble one side's order-blocks, applying the mirrored convention **here and nowhere
else**: the right physical slice is block 1 with its derivatives in `2:ORD`, the left
physical slice is block `ORD` with its orthogonality blocks in `1:(ORD-1)`.

With `companions === nothing` the caller already holds whole blocks and they are used as
given.
"""
function _stack_blocks(physical::AbstractArray, companions::Union{Nothing, AbstractArray},
        ROM::Int, side::Symbol)
    what = side === :right ? "right_modes" : "left_modes"
    companions === nothing && return _as_blocks(physical, ROM, what)

    ndims(physical) == 2 || throw(ArgumentError(
        "$what must be the FOM × ROM physical slice when " *
        "$(side === :right ? "right_derivatives" : "left_blocks") is given"))
    ndims(companions) == 3 || throw(ArgumentError(
        "$(side === :right ? "right_derivatives" : "left_blocks") must be a " *
        "FOM × (ORD-1) × ROM array"))
    size(physical, 2) == ROM || throw(ArgumentError(
        "$what has $(size(physical, 2)) columns but there are $ROM eigenvalues"))
    size(companions, 3) == ROM || throw(ArgumentError(
        "$(side === :right ? "right_derivatives" : "left_blocks") has " *
        "$(size(companions, 3)) modes but there are $ROM eigenvalues"))
    size(companions, 1) == size(physical, 1) || throw(ArgumentError(
        "$what has $(size(physical, 1)) rows but its companion blocks have " *
        "$(size(companions, 1))"))

    FOM = size(physical, 1)
    ORD = size(companions, 2) + 1
    blocks = Array{ComplexF64, 3}(undef, FOM, ORD, ROM)
    if side === :right
        blocks[:, 1, :] .= physical
        blocks[:, 2:ORD, :] .= companions
    else
        blocks[:, 1:(ORD - 1), :] .= companions
        blocks[:, ORD, :] .= physical
    end
    return blocks
end

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

# The involution over the whole spectrum, `1:n_eigs`. Unlike the master block this one is
# not widened — it is stored as given, so every entry, master and outer alike, carries its
# true conjugate partner.
function _validate_spectrum_permutation(perm, n_eigs::Int)
    perm === nothing && return nothing
    sort(collect(perm)) == collect(1:n_eigs) || throw(ArgumentError(
        "conjugate_permutation must be a permutation of 1:$n_eigs, got $(collect(perm))"))
    all(i -> perm[perm[i]] == i, 1:n_eigs) || throw(ArgumentError(
        "conjugate_permutation must be an involution, got $(collect(perm))"))
    return nothing
end

function _validate_master_permutation(perm, ROM::Int)
    perm === nothing && return nothing
    length(perm) == ROM || throw(ArgumentError(
        "conjugate_permutation has $(length(perm)) entries but ROM = $ROM. It covers the " *
        "MASTER BLOCK ONLY (or, from a solved Spectrum, the whole spectrum — one entry per " *
        "eigenvalue); the external block is appended at solve time from the model's " *
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

An explicit vector is accepted at **either of two lengths**:

- `n_eigs` — the involution over the whole spectrum, stored verbatim. Prefer this whenever
  the solver's pairing is known for every entry. A structural solver returning adjacent
  conjugate pairs, for instance, has `σ = reduce(vcat, [[2p, 2p-1] for p in 1:n_pairs])`
  exactly. Outer entries then carry their true partner, so [`physical_mode`](@ref) numbers
  physical modes rather than individual eigenvalues, and per-mode diagnostics (the
  outer-resonance warning) name the pair instead of warning once per conjugate.
- `ROM` — the master block only. Outer entries are left **self-paired**, the honest reading
  of "the caller stated the master pairing, not the spectrum's"; an outer conjugate pair
  then reports as two separate modes.

Both derive the same master restriction, so moving a call site from the second form to the
first leaves the solve bit-identical.
"""
function SpectralData(model::NthOrderModel, eigenproblem::Spectrum;
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

    master_b = ModeBundle{ORD_model}(λ_master, right, left, idx)

    # `setdiff` returns ascending indices, so the outer bundle stays in spectrum order.
    outer_idx = setdiff(1:n_eigs, idx)
    λ_outer = Vector{ComplexF64}(eigenproblem.eigenvalues[outer_idx])
    outer_b = if keep_outer_modes && !isempty(outer_idx)
        ro = _reconcile_right(eigenproblem, outer_idx,
            ComplexF64.(eigenproblem.eigenvalues[outer_idx]), ORD_spec, ORD_model)
        lo = _reconcile_left(model, eigenproblem, outer_idx,
            ComplexF64.(eigenproblem.eigenvalues[outer_idx]), ORD_spec, ORD_model)
        ModeBundle{ORD_model}(λ_outer, ro, lo, outer_idx)
    else
        ModeBundle{ORD_model}(λ_outer, nothing, nothing, outer_idx)
    end

    # ONE detection, over the WHOLE spectrum — master and outer alike. The master and outer
    # restrictions are derived from it on demand, so the pairing is never computed twice.
    σ = _resolve_permutation(
        conjugate_permutation, eigenproblem, idx, right, left, ROM, atol)
    # The constructor settles the restrictions and the mode numbering, and raises here —
    # where the offending spectrum entry can be named — if a conjugate pair was split.
    sd = SpectralData{ORD_model, ROM}(master_b, outer_b, σ, n_eigs)
    return sd
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

# An explicitly supplied permutation, accepted at either of two lengths.
#
#   length == n_eigs — the involution over the WHOLE spectrum, used verbatim. Prefer this
#       when the solver's pairing is known for every entry (an adjacent-pair structural
#       spectrum, say): outer entries then carry their true conjugate partner, so
#       `_mode_numbers` numbers physical modes rather than individual eigenvalues, and
#       anything reporting per-mode (the outer-resonance warning) names the pair.
#
#   length == ROM — the master block only, widened to the full spectrum by leaving every
#       outer entry SELF-PAIRED. That is the honest reading of "the caller stated the
#       master pairing, not the spectrum's", but it does mean an outer conjugate pair
#       reports as two separate modes.
#
# Both derive the same master restriction, so switching a call site from the ROM-length
# form to the spectrum-wide one leaves the solve bit-identical.
function _resolve_permutation(perm::AbstractVector{Int}, ep, idx, right, left, ROM, atol)
    p = collect(Int, perm)
    n_eigs = length(ep.eigenvalues)
    if length(p) == n_eigs && n_eigs != ROM
        _validate_spectrum_permutation(p, n_eigs)
        return p
    end
    _validate_master_permutation(p, ROM)
    σ = collect(1:n_eigs)
    for (l, g) in enumerate(idx)
        σ[g] = idx[p[l]]        # local pairing re-expressed in spectrum indices
    end
    return σ
end

function _resolve_permutation(sym::Symbol, ep, idx, right, left, ROM, atol)
    sym === :detect || throw(ArgumentError(
        "conjugate_permutation must be `nothing`, `:detect`, or a Vector{Int}; got :$sym"))
    # Detect over the WHOLE spectrum, once. The master and outer restrictions are derived
    # from this single vector, so the pairing is never computed a second time.
    σ = detect_conjugate_permutation(collect(ep.eigenvalues); atol = atol)
    if σ === nothing
        @info "conjugate_permutation = :detect — the spectrum is not closed under " *
              "conjugation, so no involution was derived. Proceeding without conjugate " *
              "symmetry."
        return nothing
    end
    # Eigenvalue pairing is necessary but NOT sufficient: verify the vectors, on every
    # order-block and on both sides, exactly as external_conjugate_permutation verifies
    # the external basis columns. Only the MASTER modes are checked — they are the ones the
    # solve exploits; the outer restriction is used for diagnostics only.
    position = Dict(g => l for (l, g) in enumerate(idx))
    for (name, blocks) in (("right", right), ("left", left))
        blocks === nothing && continue
        for (r, g) in enumerate(idx)
            partner = get(position, σ[g], 0)
            partner == 0 && continue   # half-pair; reported by _restrict_permutation
            for k in axes(blocks, 2)
                if !isapprox(@view(blocks[:, k, partner]), conj(@view(blocks[:, k, r]));
                    atol = atol)
                    @info "conjugate_permutation = :detect — the eigenvalues pair up, but " *
                          "the $name eigenvectors do not satisfy Ψ[:, σ(r)] = conj(Ψ[:, r]) " *
                          "(spectrum entry $g, order-block $k). Proceeding without " *
                          "conjugate symmetry; pass an explicit permutation to override."
                    return nothing
                end
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
function check_biorthogonality(sd::SpectralData{ORD, ROM}, model::NthOrderModel) where {
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
