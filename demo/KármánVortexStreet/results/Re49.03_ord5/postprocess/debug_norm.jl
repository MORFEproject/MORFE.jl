"""
debug_norm.jl — Check eigenvector norm and physical amplitude scale.

Reads W.jls, finds φ₁ = W coefficient for monomial [1,0,0],
prints ‖φ₁‖₂ and computes what the physical limit-cycle amplitude
looks like after rescaling.
"""

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "../../../backbone_env"))
Pkg.instantiate()

using Serialization, Printf, LinearAlgebra, StaticArrays
using MORFE
using MORFE.Polynomials: each_term
using Roots

const _results = joinpath(@__DIR__, "..")
const Re₀ = 49.03

println("Loading ROM...")
R = deserialize(joinpath(_results, "R.jls"))
W = deserialize(joinpath(_results, "W.jls"))
println("Done.")

open(joinpath(@__DIR__, "debug_norm.log"), "w") do io

println(io, "=== Normalization Debug ===\n")

# ── W structure ──────────────────────────────────────────────────────────────
println(io, "W type     : $(typeof(W))")
println(io, "W.poly type: $(typeof(W.poly))")
coeff_W = W.poly.coefficients        # should be (FOM, 1, L) or similar
println(io, "W.poly.coefficients size: $(size(coeff_W))")

# ── Find monomial [1,0,0] index ───────────────────────────────────────────────
exps_W  = W.poly.multiindex_set.exponents
idx_100 = findfirst(α -> α[1]==1 && α[2]==0 && α[3]==0, exps_W)
idx_010 = findfirst(α -> α[1]==0 && α[2]==1 && α[3]==0, exps_W)

println(io, "Index of [1,0,0] in W: $idx_100")
println(io, "Index of [0,1,0] in W: $idx_010")
println(io, "Total monomials in W  : $(length(exps_W))")

# ── Extract φ₁ ────────────────────────────────────────────────────────────────
# W.poly.coefficients is (FOM, 1, L):  axis 1=dof, axis 2=????, axis 3=monomial
nd = ndims(coeff_W)
println(io, "ndims(coeff_W) = $nd")

if nd == 3
    φ₁ = coeff_W[:, 1, idx_100]
elseif nd == 2
    φ₁ = coeff_W[:, idx_100]
else
    error("Unexpected ndims = $nd")
end

φ₁_norm = norm(φ₁)
@printf(io, "\n‖φ₁‖₂     = %.6g\n", φ₁_norm)
@printf(io, "max|φ₁_i| = %.6g\n", maximum(abs, φ₁))
@printf(io, "min|φ₁_i| = %.6g\n", minimum(abs, φ₁))

# ── Amplitude function F(ρ,η) ─────────────────────────────────────────────────
exps = R.poly.multiindex_set.exponents
c1   = R.poly.coefficients[1, :]

function F(ρ::Float64, η::Float64)
    val = zero(ComplexF64)
    for (k, α) in enumerate(exps)
        α[1] - α[2] == 1 || continue
        val += c1[k] * ρ^(α[1]+α[2]-1) * η^α[3]
    end
    return val
end

# ── Find ρ* at several Re values ──────────────────────────────────────────────
println(io, "\n--- Limit cycle amplitudes ---")
@printf(io, "  %-6s  %-12s  %-14s  %-14s  %-12s  %s\n",
    "Re", "η", "ρ* (DPIM)", "ρ_phys=ρ·‖φ₁‖", "Ω (rad/s)", "St=Ω·D/U")

for Re_test in [51.5, 52.0, 53.0, 55.0, 58.0, 60.0, 65.0, 70.0]
    η_t = 1.0/Re_test - 1.0/Re₀
    f(ρ) = real(F(ρ, η_t))

    # log-scan
    rhos = [10.0^x for x in range(-15, 4, 120)]
    vals = f.(rhos)
    ρ_star = NaN
    for i in 1:length(rhos)-1
        if vals[i]*vals[i+1] < 0
            ρ_star = find_zero(f, (rhos[i], rhos[i+1]), Bisection())
            break
        end
    end

    if !isnan(ρ_star)
        Ω = imag(F(ρ_star, η_t))
        St = Ω * 0.1 / (2π)  # St = f·D/U = (Ω/2π)·D/U
        @printf(io, "  %5.1f  %+.4e  %.6e  %.6e  %+10.5f  %.4f\n",
            Re_test, η_t, ρ_star, ρ_star*φ₁_norm, Ω, St)
    else
        @printf(io, "  %5.1f  %+.4e  no limit cycle found\n", Re_test, η_t)
    end
end

# ── Re_c ─────────────────────────────────────────────────────────────────────
println(io, "\n--- Re_c (linear stability boundary) ---")
c100 = c1[findfirst(α -> α[1]==1 && α[2]==0 && α[3]==0, exps)]
c101 = c1[findfirst(α -> α[1]==1 && α[2]==0 && α[3]==1, exps)]
@printf(io, "  σ₀      = Re(c₁₀₀) = %.6f\n", real(c100))
@printf(io, "  dσ/dη   = Re(c₁₀₁) = %.6g\n", real(c101))
η_c  = -real(c100)/real(c101)
Re_c = 1.0/(1.0/Re₀ + η_c)
@printf(io, "  η_c     = %.6g  →  Re_c = %.4f\n", η_c, Re_c)

# Verify by bisection on Re(F(ε,η))
ε = 1e-14
σ(Re) = real(F(ε, 1.0/Re - 1.0/Re₀))
bracket = (44.0, 62.0)
if σ(bracket[1])*σ(bracket[2]) < 0
    Re_c_bisect = find_zero(σ, bracket, Bisection())
    @printf(io, "  Re_c (bisection) = %.4f\n", Re_c_bisect)
else
    println(io, "  Re_c bisection: no sign change in $(bracket)")
    @printf(io, "  σ(%.1f)=%.4g  σ(%.1f)=%.4g\n",
        bracket[1], σ(bracket[1]), bracket[2], σ(bracket[2]))
end

println(io, "\n=== End ===")
end  # open

println("Written to debug_norm.log")
