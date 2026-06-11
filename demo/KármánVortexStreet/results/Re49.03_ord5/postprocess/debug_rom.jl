"""
debug_rom.jl — Diagnostics for the KVS DPIM ROM.

Loads R.jls + W.jls and writes debug.log with:
  - All resonant monomials of ż₁ (α₁ − α₂ = 1)
  - Physical interpretation: λ₁, c₁₀₁, c₂₁₀, c₃₂₀
  - Sign structure of Re(F(ρ,η)) for the amplitude equation
  - Eigenvector / parametrisation norm
  - Comparison of realify path vs direct complex path
"""

using Pkg: Pkg
const _backbone_env = joinpath(@__DIR__, "../../../backbone_env")
Pkg.activate(_backbone_env)
Pkg.instantiate()

using Serialization
using StaticArrays: SVector
using LinearAlgebra
using Printf
using MORFE
using MORFE.Polynomials: each_term, evaluate, extract_component
using MORFE.Realification: realify

const _results = joinpath(@__DIR__, "..")
const Re₀     = 49.03   # must match config.jl

println("Loading ROM ...")
R = deserialize(joinpath(_results, "R.jls"))
W = deserialize(joinpath(_results, "W.jls"))
println("Done.")

open(joinpath(@__DIR__, "debug.log"), "w") do io

# ─── 1. Structural info ───────────────────────────────────────────────────────
println(io, "=== KVS DPIM Debug Log ===\n")
println(io, "typeof(R)         : $(typeof(R))")
println(io, "typeof(R.poly)    : $(typeof(R.poly))")
println(io, "R.poly size       : coefficients $(size(R.poly.coefficients)), $(length(R.poly.multiindex_set.exponents)) monomials")

exps = R.poly.multiindex_set.exponents   # Vector of SVector{3,Int}
c    = R.poly.coefficients               # (ROM, L) matrix, row 1 = ż₁

# ─── 2. All resonant monomials of ż₁ ─────────────────────────────────────────
println(io, "\n--- Resonant monomials of ż₁  (α₁ − α₂ = 1) ---")
@printf(io, "  %-12s  %-14s  %-16s  %-16s\n", "monomial", "ρ^p · η^q", "Re(c₁)", "Im(c₁)")
for (k, α) in enumerate(exps)
    α[1] - α[2] == 1 || continue
    ck = c[1, k]
    @printf(io, "  [%d,%d,%d]       ρ^%d · η^%d      %+.6g        %+.6g·i\n",
        α[1], α[2], α[3], α[1]+α[2]-1, α[3], real(ck), imag(ck))
end

# ─── 3. Physical key coefficients ────────────────────────────────────────────
println(io, "\n--- Key coefficients ---")

find_coeff(a, b, c_) = let idx = findfirst(α -> α[1]==a && α[2]==b && α[3]==c_, exps)
    isnothing(idx) ? 0.0+0.0im : c[1, idx]
end

c100 = find_coeff(1,0,0); σ₀ = real(c100); ω₀ = imag(c100)
c101 = find_coeff(1,0,1)
c210 = find_coeff(2,1,0)
c320 = find_coeff(3,2,0)

@printf(io, "  λ₁  = c₁₀₀ = %+.8f %+.8f·i   (σ₀ = %.6f)\n", σ₀, ω₀, σ₀)
@printf(io, "  c₁₀₁ = %+.6g %+.6g·i   Re(c₁₀₁) = %+.4g  %s\n",
    real(c101), imag(c101), real(c101),
    real(c101) < 0 ? "← correct (more visc → more stable)" : "← WRONG SIGN")
@printf(io, "  c₂₁₀ = %+.6g %+.6g·i   Re(c₂₁₀) = %+.4g  %s\n",
    real(c210), imag(c210), real(c210),
    real(c210) < 0 ? "← supercritical ✓" : "← subcritical or wrong sign ✗")
@printf(io, "  c₃₂₀ = %+.6g %+.6g·i   Re(c₃₂₀) = %+.4g\n",
    real(c320), imag(c320), real(c320))

# Linear Re_c estimate
if !iszero(real(c101))
    η_c = -σ₀ / real(c101)
    Re_c_lin = 1.0 / (1.0/Re₀ + η_c)
    @printf(io, "\n  Linear Re_c ≈ %.4f  (η_c = %.6g, above Re₀=%g: %s)\n",
        Re_c_lin, η_c, Re₀, Re_c_lin > Re₀ ? "YES (σ₀<0 at Re₀)" : "NO")
end

# ─── 4. Amplitude function F(ρ,η) = ∑_{α₁-α₂=1} cα · ρ^{α₁+α₂-1} · η^{α₃} ──
function F_complex(ρ::Float64, η::Float64; max_ord::Int=99)
    val = zero(ComplexF64)
    for (k, α) in enumerate(exps)
        α[1] - α[2] == 1 || continue
        sum(α) <= max_ord || continue
        val += c[1,k] * ρ^(α[1]+α[2]-1) * η^α[3]
    end
    return val
end

# ─── 5. Sign scan: does Re(F) change sign for any ρ? ─────────────────────────
println(io, "\n--- Re(F(ρ,η)) sign scan — looking for a limit cycle ---")
@printf(io, "  %-8s  %-12s  %-10s  %-10s  %-10s  %-10s  %-10s  %s\n",
    "Re", "η", "F(0)", "F(1e-12)", "F(1e-8)", "F(1e-5)", "F(1e-2)", "sign_change?")

ρ_grid = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 1e3, 1e4]

for Re_test in [49.0, 50.0, 51.0, 52.0, 55.0, 60.0, 70.0]
    η_t = 1.0/Re_test - 1.0/Re₀
    vals = [real(F_complex(ρ, η_t)) for ρ in ρ_grid]
    sc = any(vals[i]*vals[i+1] < 0 for i in 1:length(vals)-1)
    @printf(io, "  %5.1f  %+.4e  %+.4g  %+.4g  %+.4g  %+.4g  %+.4g  %s\n",
        Re_test, η_t,
        vals[1], vals[2], vals[8], vals[10], vals[11],
        sc ? "YES ← limit cycle" : "no")
end

# ─── 6. Full log-scan at Re=55 ────────────────────────────────────────────────
println(io, "\n--- Full log-scan at Re=55 ---")
η_55 = 1.0/55.0 - 1.0/Re₀
rhos = [10.0^x for x in range(-15, 4, 80)]
vals_55 = [real(F_complex(ρ, η_55)) for ρ in rhos]
prev_sign = sign(vals_55[1])
for i in eachindex(rhos)
    s = sign(vals_55[i])
    marker = s != prev_sign ? "  ← SIGN CHANGE" : ""
    @printf(io, "  ρ=%.3e  Re(F)=%+.6g%s\n", rhos[i], vals_55[i], marker)
    prev_sign = s
end

# ─── 7. Realify path comparison ──────────────────────────────────────────────
println(io, "\n--- Realify vs. direct comparison at Re=55, several ρ ---")
conj_map = [2, 1, 3]
R1_full = extract_component(realify(R.poly, conj_map), 1)
println(io, "  Realified R1_full type: $(typeof(R1_full))")
for ρ_t in [1e-10, 1e-5, 1e-2, 0.1]
    direct_val  = real(F_complex(ρ_t, η_55))
    realify_val = real(evaluate(R1_full, SVector(ρ_t, 0.0, η_55)))
    @printf(io, "  ρ=%.1e  direct Re(F)=%+.6g   realify Re(R1(ρ,0,η))=%+.6g   ratio=%.4g\n",
        ρ_t, direct_val, realify_val,
        abs(realify_val) < 1e-300 ? NaN : realify_val / direct_val)
end

# ─── 8. W parametrisation: eigenvector norm ──────────────────────────────────
println(io, "\n--- W parametrisation structure ---")
println(io, "  typeof(W)         : $(typeof(W))")
println(io, "  fieldnames(W)     : $(fieldnames(typeof(W)))")
try
    println(io, "  typeof(W.poly)    : $(typeof(W.poly))")
    println(io, "  W.poly coeff size : $(size(W.poly.coefficients))")
    exps_W = W.poly.multiindex_set.exponents
    println(io, "  W multiindex count: $(length(exps_W))")
    # [1,0,0] monomial gives the first eigenvector φ₁ restricted to free DOFs
    idx100_W = findfirst(α -> α[1]==1 && α[2]==0 && α[3]==0, exps_W)
    if !isnothing(idx100_W)
        col = W.poly.coefficients[:, idx100_W]   # FOM × 1 → all master modes collapsed?
        println(io, "  W[:,idx_100] size : $(size(col))")
        @printf(io, "  ||φ₁|| (col 1 of W[1,0,0]) = %.6g\n", norm(col))
        @printf(io, "  max|φ₁| = %.6g\n", maximum(abs, col))
    end
catch e
    println(io, "  ERROR accessing W.poly: $e")
end

println(io, "\n=== End debug log ===")
end  # open

println("Debug log written to: ", joinpath(@__DIR__, "debug.log"))
