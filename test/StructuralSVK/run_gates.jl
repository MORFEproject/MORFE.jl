# Full-mesh equivalence gates for MORFEStructuralSVK (run manually; ~minutes).
# Usage, from the repository root:
#   julia --project=examples/01_clamped_beam_ferrite test/StructuralSVK/run_gates.jl
#
# Gate A: high-level main.jl ≡ low-level low_level.jl (order 9, full mesh, 1e-10).
# Gate B: forced(amplitude = 0) ≡ autonomous (order 5, full mesh, 1e-10).

using DelimitedFiles

const EX = joinpath(@__DIR__, "..", "..", "examples", "01_clamped_beam_ferrite")

# ── Gate A ───────────────────────────────────────────────────────────────────
println("═"^60, "\nGate A: low-level reference run\n", "═"^60)
include(joinpath(EX, "low_level.jl"))
low_csv = joinpath(tempdir(), "R_low_gateA.csv")
cp(joinpath(EX, "results", "data", "R_coefficients.csv"), low_csv; force = true)

println("═"^60, "\nGate A: high-level run\n", "═"^60)
include(joinpath(EX, "main.jl"))

a = Float64.(readdlm(low_csv, ',', skipstart = 1))
b = Float64.(readdlm(joinpath(EX, "results", "data", "R_coefficients.csv"), ',',
    skipstart = 1))
size(a) == size(b) || error("GATE A FAILED: table size $(size(a)) vs $(size(b))")
devA = maximum(abs.(a .- b) ./ max.(abs.(a), 1e-12))
println("Gate A max rel dev: $devA")
devA < 1e-10 || error("GATE A FAILED: high-level path diverges from low-level ($devA)")
println("Gate A PASSED ✓")

# ── Gate B ───────────────────────────────────────────────────────────────────
println("═"^60, "\nGate B: zero-amplitude forcing consistency (order 5)\n", "═"^60)
using StaticArrays: SVector
rom0 = SVK.parametrise(beam; master = [1], order = 5)
romf = SVK.parametrise(beam; master = [1], order = 5,
    forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.0))
exps0 = rom0.R.poly.multiindex_set.exponents
expsf = romf.R.poly.multiindex_set.exponents
lookup = Dict(e => i for (i, e) in enumerate(expsf))
maxdev = 0.0
for (i, e) in enumerate(exps0)
    j = get(lookup, SVector{4, Int}(e[1], e[2], 0, 0), nothing)
    j === nothing && error("GATE B FAILED: monomial $e missing in forced mset")
    c0 = rom0.R.poly.coefficients[:, i]
    cf = romf.R.poly.coefficients[1:2, j]
    global maxdev = max(maxdev, maximum(abs.(c0 .- cf) ./ max.(abs.(c0), 1e-12)))
end
println("Gate B max rel dev: $maxdev")
maxdev < 1e-10 || error("GATE B FAILED: forcing block corrupts autonomous dynamics ($maxdev)")
println("Gate B PASSED ✓")
