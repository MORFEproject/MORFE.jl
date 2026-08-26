"""
Constructing MultiindexSets

A `MultiindexSet` is the object that decides **which monomials the DPIM solve
computes**. Every parametrisation `W` and reduced dynamics `R` carries one, and
`parametrise(model, spectral, mset)` lets you supply your own instead of the
default graded expansion — that single argument is the difference between an isotropic
expansion, an anisotropic parametric ROM, and a spectrally truncated one.

This tutorial builds multiindex sets six ways, from the automatic generators to a
spectral truncation driven by the superharmonics `s(α) = ⟨λ, α⟩`:

 1. the automatic generators
 2. the same sets built by hand from a vector of exponents
 3. combining conditions: anisotropic total degree × per-parameter box
 4. removing multiindices with `delete_multiindices` and `filter`
 5. bounding the superharmonics by a spectral radius
 6. the `parametrise` contract, and how `validate_multiindex_set` enforces it

Each section writes an interactive lattice view to `results/figures/`. Nothing is
solved here — the script runs in a couple of seconds and needs no FEM backend.

Run it from the repository root:

	julia --project -e 'include("examples/internals/multiindex_sets/main.jl")'
"""

using MORFE.Multiindices
using MORFE: validate_multiindex_set
using StaticArrays: SVector

include(joinpath(@__DIR__, "viz.jl"))

const FIGDIR = get(ENV, "MORFE_LATTICE_OUT", joinpath(@__DIR__, "results", "figures"))

# Compact printing: a set is small enough to show in full throughout this tutorial.
show_set(name, S) = println(rpad(name, 34), lpad(length(S), 4), "  ", collect(S))

# ------------------------------------------------------------------------------
# 1. The automatic generators
#
# A multiindex α ∈ ℕᴺ is the exponent vector of the monomial z₁^α₁ ⋯ z_N^α_N. A
# MultiindexSet stores them sorted in graded-lexicographic (GrLex) order: total degree
# ascending first, then lexicographic *descending* within a degree. That order is what
# makes the cohomological solve causal — when the solver reaches α, every coefficient it
# needs at lower degree has already been computed.
# ------------------------------------------------------------------------------

println("="^92)
println("1. Automatic generators\n")

# `min_degree = 1` throughout: the expansion is centred on the fixed point, so the
# constant monomial α = 0 is never part of a DPIM set. Every set below excludes it.
total_deg = all_multiindices_up_to(2, 4; min_degree = 1)      # 1 ≤ |α| ≤ 4
exact_deg = multiindices_with_total_degree(2, 4)              # |α| = 4
box = filter(α -> sum(α) ≥ 1, all_multiindices_in_box([3, 3]))   # 0 ≤ αᵢ ≤ 3

show_set("up_to(2, 4; min_degree = 1)", total_deg)
show_set("multiindices_with_total_degree(2, 4)", exact_deg)
show_set("in_box([3, 3]), origin dropped", box)

# `all_multiindices_up_to` without `min_degree` does include the constant — which is
# exactly the one monomial DPIM refuses.
println("\nwith the constant:    ", length(all_multiindices_up_to(2, 4)), " monomials")
println("without it:           ", length(total_deg), " monomials")

# The GrLex order is visible in the listing above: degree 1 is [1,0] then [0,1], degree 2
# is [2,0], [1,1], [0,2]. Position in that order is the column index used everywhere else
# in MORFE, and `find_in_set` recovers it through a degree-bracketed binary search.
println("\nfind_in_set(total_deg, [1, 1])  = ", find_in_set(total_deg, [1, 1]))
println("find_in_set(total_deg, [9, 9])  = ", find_in_set(total_deg, [9, 9]), "  (absent)")
println("total_deg[5]                    = ", total_deg[5])

# `indices_in_box_with_bounded_degree` is the workhorse query: componentwise bound plus a
# HALF-OPEN degree window, degree_lower_bound ≤ |α| < total_deg_upper.
in_window = indices_in_box_with_bounded_degree(total_deg, [2, 2], 1, 3)
println("indices with 1 ≤ |α| < 3 inside [2,2]: ", in_window)
println("  → ", [Vector(total_deg[i]) for i in in_window])

write_lattice(joinpath(FIGDIR, "fig1_generators.html"),
    [
        LatticePanel("1 ≤ |α| ≤ 4", total_deg;
            note = "all_multiindices_up_to(2, 4; min_degree = 1) — the graded simplex."),
        LatticePanel("|α| = 4", exact_deg;
            note = "multiindices_with_total_degree(2, 4) — one degree slice, the line α₁ + α₂ = 4."),
        LatticePanel("1 ≤ |α|, α ≤ [3,3]", box;
            note = "all_multiindices_in_box([3, 3]) — a hyperrectangle: degree |α| = 6 on the corner.")];
    title = "Generators",
    caption = "The three automatic generators. The constant α = 0 is excluded throughout.")

# ------------------------------------------------------------------------------
# 2. The same sets, built by hand
#
# The generators are conveniences over one public constructor that takes the exponents
# directly. Use it whenever the shape you want is not a simplex or a box.
# ------------------------------------------------------------------------------

println("\n" * "="^92)
println("2. Manual construction\n")

# Vector of SVectors — the storage representation. Order does not matter: the
# constructor sorts into GrLex and deduplicates for you.
by_hand = MultiindexSet([SVector{2, Int}(a, b)
                         for a in 0:4 for b in 0:4 if 1 ≤ a + b ≤ 4])
println("hand-built == up_to(2, 4; min_degree = 1): ", by_hand == total_deg)

# Deliberately unsorted, with a duplicate:
messy = MultiindexSet([SVector{2, Int}(0, 2), SVector{2, Int}(1, 0),
    SVector{2, Int}(0, 2), SVector{2, Int}(2, 0)])
show_set("sorted and deduplicated", messy)

# Two more spellings of the same thing. Note that the matrix form takes exponents as
# COLUMNS, so it is `nvars × nmonomials`.
from_vecs = MultiindexSet([[1, 0], [0, 1], [1, 1]])
from_matrix = MultiindexSet([1 0 1; 0 1 1])
println("Vector{Vector{Int}} == Matrix{Int}: ", from_vecs == from_matrix)

# Equality compares the exponents only — `degree_offsets` is a derived field.
println("equality ignores degree_offsets:    ",
    MultiindexSet([[2, 0], [1, 0]]) == MultiindexSet([[1, 0], [2, 0]]))

# And a set no generator can produce: a hand-picked selection.
handpicked = MultiindexSet([[1, 0], [0, 1], [3, 1], [1, 3]])
show_set("hand-picked", handpicked)

# ------------------------------------------------------------------------------
# 3. Combining conditions
#
# The parametric ROMs are the real use case. There the reduced coordinates split into
# master coordinates z, which deserve a total-degree bound, and parameter coordinates θ,
# which deserve a per-parameter box — an expansion of order 4 in z but only 2 in each θ.
# The two conditions simply intersect.
# ------------------------------------------------------------------------------

println("\n" * "="^92)
println("3. Combining conditions: anisotropic z-total × θ-box\n")

const MAXZ = 4   # total degree budget in the master coordinates
const MAXT = 2   # per-parameter budget in each θ

# Route A — a comprehension stating both conditions directly. This is the pattern the
# parametric examples use in their config files.
aniso_A = MultiindexSet([SVector{3, Int}(a, b, c)
                         for a in 0:MAXZ for b in 0:MAXZ
                         for c in 0:MAXT
                         if a + b ≤ MAXZ && 1 ≤ a + b + c])

# Route B — start from the enclosing box and narrow it with `filter`.
aniso_B = filter(α -> α[1] + α[2] ≤ MAXZ && sum(α) ≥ 1,
    all_multiindices_in_box([MAXZ, MAXZ, MAXT]))

println("route A (comprehension) == route B (filter): ", aniso_A == aniso_B)
println("enclosing box:      ", length(all_multiindices_in_box([MAXZ, MAXZ, MAXT])))
println("anisotropic set:    ", length(aniso_A))
println("isotropic |α| ≤ 4:  ", length(all_multiindices_up_to(3, MAXZ; min_degree = 1)),
    "  (a different set — it allows α₃ = 3, 4 but forbids |α| = 5, 6)")

# Two parameters, i.e. the four-variable set of the parametric beam example. Same two
# conditions, one more θ.
println("two parameters (NVAR = 4): ",
    length(filter(α -> α[1] + α[2] ≤ MAXZ && sum(α) ≥ 1,
        all_multiindices_in_box([MAXZ, MAXZ, MAXT, MAXT]))), " monomials")

# ----- The lattice figure: one universe, composable conditions -----
#
# Every button acts on the SAME lattice. The red ones are filters and INTERSECT — turn
# on two and only the monomials satisfying both remain. The green ones are unions and
# ADD their monomials back on top of whatever the filters left. `reset` clears both.
# Whatever a condition excludes stays visible as a hollow marker, and the camera holds
# still across a toggle, so the shapes can be compared directly.

universe = filter(α -> sum(α) ≥ 1, all_multiindices_in_box([MAXZ, MAXZ, MAXT]))

println("\nuniverse (the box, minus the origin): ", length(universe), " monomials")
for (name, pred) in ("|α| ≤ 4" => (α -> sum(α) ≤ MAXZ),
    "α₁ + α₂ ≤ 4" => (α -> α[1] + α[2] ≤ MAXZ),
    "α₁ ≤ 2" => (α -> α[1] ≤ 2),
    "α ≤ (2,2,1)" => (α -> all(α .≤ [2, 2, 1])),
    "α₃ = 0 face" => (α -> α[3] == 0))
    println("  ", rpad(name, 16), count(pred, universe.exponents))
end

write_lattice(joinpath(FIGDIR, "fig2_anisotropic.html"),
    LatticeConditions(universe;
        filters = ["|α| ≤ 4" => (α -> sum(α) ≤ MAXZ),
            "α₁ + α₂ ≤ 4" => (α -> α[1] + α[2] ≤ MAXZ),
            "α₁ ≤ 2" => (α -> α[1] ≤ 2),
            "α ≤ (2,2,1)" => (α -> all(α .≤ [2, 2, 1]))],
        unions = ["α₃ = 0 face" => (α -> α[3] == 0),
            "α₃ = 2 face" => (α -> α[3] == 2)]);
    title = "Filtering a lattice",
    caption = "Red filters intersect · green buttons add points back · reset clears both.")

# ------------------------------------------------------------------------------
# 4. Removing multiindices
#
# `delete_multiindices` returns a NEW set — there is no in-place variant, so a set you
# have already handed to a parametrisation can never be mutated underneath it.
# ------------------------------------------------------------------------------

println("\n" * "="^92)
println("4. Deleting multiindices\n")

S = all_multiindices_up_to(2, 3; min_degree = 1)
show_set("S", S)

# By explicit exponent. Accepts a single exponent, a list of them, or another set.
no_corners = delete_multiindices(S, [[3, 0], [0, 3]])
show_set("delete [3,0] and [0,3]", no_corners)
show_set("delete a single exponent", delete_multiindices(S, [1, 1]))
show_set("set difference",
    delete_multiindices(S, all_multiindices_up_to(2, 1; min_degree = 1)))

# Exponents that are not members are simply ignored, as with `setdiff`.
println("deleting an absent exponent is a no-op: ",
    delete_multiindices(S, [[9, 9]]) == S)

# S itself never changes.
println("S is untouched: ", length(S), " monomials, still ", collect(S)[end])

# By predicate — and `filter` keeps exactly what `delete_multiindices` drops.
odd_only = delete_multiindices(α -> iseven(sum(α)), S)
even_only = filter(α -> iseven(sum(α)), S)
show_set("odd total degree", odd_only)
show_set("even total degree", even_only)
println("the two partition S: ", length(odd_only) + length(even_only) == length(S))

# Deletion cannot reorder a sorted list, so the result skips the re-sort — but the
# degree table is rebuilt, and lookups stay correct even when a whole degree vanishes.
gapped = delete_multiindices(α -> sum(α) == 2, S)
show_set("degree-2 block removed", gapped)
println("find_in_set(gapped, [3, 0]) = ", find_in_set(gapped, [3, 0]),
    "   find_in_set(gapped, [1, 1]) = ", find_in_set(gapped, [1, 1]))

write_lattice(joinpath(FIGDIR, "fig3_deletion.html"),
    [LatticePanel("S = 1 ≤ |α| ≤ 3", S; note = "The starting set."),
        LatticePanel("explicit deletion", S;
            marks = Dict([3, 0] => "dropped", [0, 3] => "dropped"),
            note = "delete_multiindices(S, [[3,0], [0,3]]) — removed exponents shown hollow."),
        LatticePanel("odd degree kept", S;
            marks = Dict(Vector(α) => (isodd(sum(α)) ? "kept" : "dropped")
            for α in S),
            note = "delete_multiindices(α -> iseven(sum(α)), S)."),
        LatticePanel("even degree kept", S;
            marks = Dict(Vector(α) => (iseven(sum(α)) ? "kept" : "dropped")
            for α in S),
            note = "filter(α -> iseven(sum(α)), S) — the exact complement.")];
    title = "Deletion",
    caption = "Deletion is non-mutating: every panel is a new set derived from the first.")

# ------------------------------------------------------------------------------
# 5. Bounding the superharmonics by a spectral radius
#
# Every monomial has a superharmonic s(α) = ⟨λ, α⟩ — the eigenvalue of the linear part
# its forced response sits at, obtained by adding the eigenvalues head to tail, α_k
# copies of λ_k. This is exactly what `Resonance._superharmonics` computes, and it
# drives resonance detection: α is resonant with master mode r when |λ_r − s(α)| < tol.
#
# It also supports a truncation that no degree or box bound can express — keep only the
# monomials whose response lands inside a band |s(α)| < R of the spectrum.
# ------------------------------------------------------------------------------

println("\n" * "="^92)
println("5. Spectral bandwidth cut\n")

# A damped spectrum: one oscillatory conjugate pair and one real decaying mode.
λ = [-1 + 2im, -1 - 2im, -1.0 + 0im]
superharmonic(α) = sum(λ .* α)   # s(α) = ⟨λ, α⟩

full = all_multiindices_up_to(3, 4; min_degree = 1)
R = 4.0

band = filter(α -> abs(superharmonic(α)) < R, full)
println("|α| ≤ 4:         ", length(full), " monomials")
println("|s(α)| < ", R, ":     ", length(band), " monomials")
@assert length(full) == 34 && length(band) == 13

# Since every eigenvalue has the same real part, Re s = −|α| and Im s = 2(α₁ − α₂):
#
#   |s|² = (α₁+α₂+α₃)² + 4(α₁−α₂)² = 5α₁² + 5α₂² + α₃² − 6α₁α₂ + 2α₁α₃ + 2α₂α₃
#
# The α₁α₂ coefficient is NEGATIVE. Multiplying by z̄ cancels imaginary part contributed
# by z, pulling s back towards the real axis — so |s| is not monotone under divisibility,
# and a monomial can sit inside the band while one of its factors does not.
println("\nα                s(α)              |s|     kept")
for α in full.exponents
    sum(α) ≤ 3 || continue
    s = superharmonic(α)
    println("  ", rpad(string(Vector(α)), 12), rpad(string(round(s, digits = 3)), 18),
        rpad(string(round(abs(s), digits = 3)), 8), abs(s) < R ? "yes" : "no")
end

# Because every Re λ is equal, s(α) = λ_r forces |α| = 1: this spectrum has no
# NONTRIVIAL resonances at all. They appear when the eigenvalues sit on the imaginary
# axis, which is the conservative case the bordered solve exists for.
println("\nresonant monomials of degree > 1: ",
    count(α -> sum(α) > 1 && any(r -> abs(λ[r] - superharmonic(α)) < 1e-8, eachindex(λ)),
        full.exponents))

# So the band is NOT downward closed. The graded solve reads W[α − eᵢ] while working on
# α, so DPIM rejects a set with a missing factor.
println("\nis_downward_closed(band): ", is_downward_closed(band))
@assert !is_downward_closed(band)

unit(i, n) = SVector{n, Int}(ntuple(j -> j == i ? 1 : 0, n))

# Degree 1 is exempt — its only factor is the constant, which min_degree = 1 removed.
for α in band.exponents, i in 1:3

    (α[i] > 0 && sum(α) > 1) || continue
    β = α - unit(i, 3)
    find_in_set(band, β) === nothing || continue
    println("  ", Vector(α), " is kept (|s| = ", round(abs(superharmonic(α)), digits = 3),
        "), but its factor ", Vector(β), " has |s| = ",
        round(abs(superharmonic(β)), digits = 3), " ≥ ", R)
end

# The fix: take the downward closure — add back every factor of every member. The
# result is legal, and still far smaller than the full expansion.
function downward_closure(S::MultiindexSet{N}) where {N}
    closed = Set{SVector{N, Int}}()
    for α in S.exponents
        for idx in CartesianIndices(ntuple(i -> 0:α[i], N))
            push!(closed, SVector{N, Int}(Tuple(idx)))
        end
    end
    delete!(closed, zero(SVector{N, Int}))   # DPIM needs min_degree ≥ 1
    return MultiindexSet(collect(closed))
end

band_closed = downward_closure(band)
println("\ndownward closure:   ", length(band_closed), " monomials, added ",
    [Vector(α) for α in band_closed.exponents if find_in_set(band, α) === nothing])
println("is_downward_closed: ", is_downward_closed(band_closed))
println("still below |α| ≤ 4: ", length(band_closed), " < ", length(full))
@assert length(band_closed) == 15 && is_downward_closed(band_closed)

write_lattice(joinpath(FIGDIR, "fig4_spectral.html"),
    LatticeSpectrum(full, λ, R;
        note = "Hover over a multiindex α to visualise its superharmonic: s(α) = α₁ λ₁ + α₂ λ₂ + α₃ λ₃.");
    title = "Superharmonics",
    caption = "Left: the lattice, coloured by |s(α)|. Right: the same monomials as points " *
              "in the complex plane, with the band |s| < 4 dashed.")

# ------------------------------------------------------------------------------
# 6. Handing a custom set to `parametrise`
#
# `parametrise(model, spectral, mset)` enforces the whole contract before it
# solves anything, through `validate_multiindex_set`. That function needs no model, so
# a set can be checked the moment it is built rather than after a long assembly.
#
# It throws an ArgumentError naming the offending exponent. Both `parametrise` and
# `solve_cohomological_problem` take `validate_mset = false` to skip the check.
# ------------------------------------------------------------------------------

println("\n" * "="^92)
println("6. Checking the parametrise contract\n")

# The two closure predicates on their own, when a Bool is all you want:
println("is_downward_closed(band_closed)          = ", is_downward_closed(band_closed))
println("is_conjugate_closed(aniso_A, [2, 1, 3])  = ",
    is_conjugate_closed(aniso_A, [2, 1, 3]))
# Swapping z₁ ↔ z₂ maps the anisotropic set onto itself: both conditions defining it
# (α₁ + α₂ ≤ MAXZ, and the θ box) are symmetric in α₁ and α₂.

# And the whole contract, exactly as `parametrise` applies it.
function report(name, S; nvar = length(first(S.exponents)), rom = 2, perm = nothing)
    try
        validate_multiindex_set(S, nvar, rom; conjugate_permutation = perm)
        println(rpad(name, 26), "OK — usable as mset")
    catch err
        err isa ArgumentError || rethrow()
        # First sentence only; the full message also states why it matters.
        println(rpad(name, 26), "rejected — ", first(split(err.msg, ". ")))
    end
end

report("isotropic |α| ≤ 4", all_multiindices_up_to(2, 4; min_degree = 1))
report("anisotropic z × θ", aniso_A; rom = 2, perm = [2, 1, 3])
report("spectral band (raw)", band; rom = 3)
report("spectral band (closed)", band_closed; rom = 3)
report("band, wrong NVAR", band; nvar = 4, rom = 3)
report("not conjugate closed",
    delete_multiindices(all_multiindices_up_to(2, 3; min_degree = 1), [[1, 2]]);
    rom = 2, perm = [2, 1])

println("\n" * "="^92)
println("Figures written to ", FIGDIR)
foreach(f -> println("  ", f), sort(readdir(FIGDIR)))
println("Demo finished successfully.")
