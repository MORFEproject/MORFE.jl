"""
debug_conv.jl — Invariance-equation and FluidConvection sign tests.

Three tests:
  Test 1: Orthogonality  ψ₁^H B₁ W[2,1,0] ≈ 0     (MORFE internal consistency)
  Test 2: Invariance eq   L(2λ₁) W[2,0,0] == ml_result  (sign check for FluidConvection)
  Test 3: FD sign         FluidConvection(φ₁,φ₁) ≈ -N₂_FD  (physical sign)

Writes debug_conv.log.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "../../.."))
Pkg.instantiate()

using Ferrite, FerriteGmsh, Gmsh
using LinearAlgebra, SparseArrays, Printf, Serialization, StaticArrays, KLU
using MORFE

const _demo = joinpath(@__DIR__, "../../..")
include(joinpath(_demo, "config.jl"))
include(joinpath(_demo, "mesh.jl"))
include(joinpath(_demo, "fem_setup.jl"))
include(joinpath(_demo, "steady_state.jl"))
include(joinpath(_demo, "linear_operators.jl"))
include(joinpath(_demo, "fluid_maps.jl"))
include(joinpath(_demo, "eigensolver.jl"))

const _results = joinpath(@__DIR__, "..")

open(joinpath(@__DIR__, "debug_conv.log"), "w") do io

println(io, "=== FluidConvection / Invariance Equation Debug ===\n")

# ── Setup ─────────────────────────────────────────────────────────────────────
println(io, "[Setup] Mesh + FEM + Newton + linear operators + eigenproblem...")
meshfile = generate_mesh(; h_cyl=MESH_H_CYL, h_wake=MESH_H_WAKE, h_bulk=MESH_H_BULK)
fom = setup_fem(meshfile)
(_, _, s₀_full) = solve_steady_state(fom; Re0=Re₀)
B₀, B₁, A_lin = assemble_linear_operators(s₀_full, fom; Re0=Re₀)
(λs, master_modes, left_modes) =
    solve_hopf_eigenproblem(-B₀, B₁; nev=EIG_NEV, sigma_re=EIG_SIGMA_RE, sigma_im=EIG_SIGMA_IM)

φ₁ = master_modes[:, 1]    # right eigenvector, normalized so ψ₁ᴴ B₁ φ₁ = 1
ψ₁ = left_modes[:, 1]      # left eigenvector (ARPACK norm, not divided by α)
λ₁ = λs[1]

@printf(io, "λ₁ = %+.8f %+.8fi\n", real(λ₁), imag(λ₁))
@printf(io, "‖φ₁‖ = %.6g,  ‖ψ₁‖ = %.6g\n", norm(φ₁), norm(ψ₁))

# Sanity: normalization condition ψ₁ᴴ B₁ φ₁ must equal 1
norm_check = dot(ψ₁, B₁ * φ₁)
@printf(io, "ψ₁ᴴ B₁ φ₁ = %+.6g %+.6gi  (should be 1+0i)\n\n", real(norm_check), imag(norm_check))

# ── Load saved ROM ────────────────────────────────────────────────────────────
println(io, "[Load] Reading W.jls and R.jls...")
W_rom = deserialize(joinpath(_results, "W.jls"))
R_rom = deserialize(joinpath(_results, "R.jls"))

coeff_W = W_rom.poly.coefficients   # (FOM, 1, L)  FOM = n_free_dpim = 57954
exps_W  = W_rom.poly.multiindex_set.exponents

find_W(a,b,c) = let idx=findfirst(α->α[1]==a && α[2]==b && α[3]==c, exps_W)
    isnothing(idx) ? nothing : coeff_W[:, 1, idx]
end

φ₁_W = find_W(1,0,0)
W200 = find_W(2,0,0)
W110 = find_W(1,1,0)
W210 = find_W(2,1,0)

@printf(io, "‖φ₁ from eigen‖ = %.6g\n", norm(φ₁))
@printf(io, "‖φ₁ from W[1,0,0]‖ = %.6g  (should match)\n", norm(φ₁_W))
@printf(io, "‖W[2,0,0]‖ = %.6g\n", norm(W200))
@printf(io, "‖W[1,1,0]‖ = %.6g\n", norm(W110))
@printf(io, "‖W[2,1,0]‖ = %.6g\n\n", norm(W210))

# ── Test 1: Orthogonality ψ₁ᴴ B₁ W[α] = 0 ───────────────────────────────────
println(io, "=== Test 1: Orthogonality ψ₁ᴴ B₁ W[α] == 0 ===")

B₁_cplx = ComplexF64.(B₁)
B₀_cplx = ComplexF64.(B₀)

function orth_check(label, W_vec)
    v = B₁_cplx * W_vec
    proj = dot(ψ₁, v)
    @printf(io, "  ψ₁ᴴ B₁ W[%s] = %+.4e %+.4ei\n", label, real(proj), imag(proj))
    return proj
end

orth_check("1,0,0", φ₁_W)   # should be 1
orth_check("2,0,0", W200)    # should be 0
orth_check("1,1,0", W110)    # should be 0
orth_check("2,1,0", W210)    # should be 0 (orthogonality condition)

println(io)

proj_100 = dot(ψ₁, B₁_cplx * φ₁_W)
proj_210 = dot(ψ₁, B₁_cplx * W210)
@printf(io, "  Normalization: ψ₁ᴴ B₁ φ₁ = %+.8g %+.8gi\n", real(proj_100), imag(proj_100))
@printf(io, "  Orthogonality deviation: ψ₁ᴴ B₁ W[2,1,0] = %+.4e %+.4ei\n\n",
    real(proj_210), imag(proj_210))

# ── Test 2: Invariance equation at [2,0,0] (non-resonant) ─────────────────────
println(io, "=== Test 2: Invariance equation at [2,0,0] ===")

s_200 = 2 * λ₁
@printf(io, "  s[2,0,0] = %+.6f %+.6fi\n", real(s_200), imag(s_200))

LW200 = B₀_cplx * W200 .+ s_200 .* (B₁_cplx * W200)
@printf(io, "  ‖L(2λ₁) W[2,0,0]‖ = %.6g\n", norm(LW200))
println(io, "  (ml_result[2,0,0] estimated from FD below)")
println(io)

# ── Test 3: FD test — N₂(φ₁,φ₁) via NSE residual ────────────────────────────
println(io, "=== Test 3: FD sign test ===")

φ_re = real(φ₁_W)   # from W, length n_free_dpim
φ_im = imag(φ₁_W)

K_scratch = allocate_matrix(fom.dh)
R_scratch = zeros(ndofs(fom.dh))

function nse_residual_dpim(s_full_in)
    fill!(K_scratch, 0.0)
    fill!(R_scratch, 0.0)
    assemble_steady_nse!(K_scratch, R_scratch, s_full_in, fom, Re₀)
    apply_zero!(K_scratch, R_scratch, fom.ch_hom)
    return copy(R_scratch[fom.free_dpim])
end

R₀ = nse_residual_dpim(s₀_full)
@printf(io, "  ‖R(u₀)‖ at free_dpim = %.4g  (should be ≈0)\n", norm(R₀))

ε_fd = 1e-5 / norm(φ_re)

# Centered FD: N₂(φ_re, φ_re)
s_plus  = copy(s₀_full); s_plus[fom.free_dpim]  .+= ε_fd .* φ_re
s_minus = copy(s₀_full); s_minus[fom.free_dpim] .-= ε_fd .* φ_re
N2_rr = (nse_residual_dpim(s_plus) .+ nse_residual_dpim(s_minus)) ./ (2ε_fd^2)

# Centered FD: N₂(φ_im, φ_im)
s_plus_i  = copy(s₀_full); s_plus_i[fom.free_dpim]  .+= ε_fd .* φ_im
s_minus_i = copy(s₀_full); s_minus_i[fom.free_dpim] .-= ε_fd .* φ_im
N2_ii = (nse_residual_dpim(s_plus_i) .+ nse_residual_dpim(s_minus_i)) ./ (2ε_fd^2)

# Cross-term N₂(φ_re, φ_im)
s_plus_ri  = copy(s₀_full); s_plus_ri[fom.free_dpim]  .+= ε_fd .* (φ_re .+ φ_im)
s_minus_ri = copy(s₀_full); s_minus_ri[fom.free_dpim] .-= ε_fd .* (φ_re .+ φ_im)
N2_ri = ((nse_residual_dpim(s_plus_ri) .+ nse_residual_dpim(s_minus_ri)) ./ (2ε_fd^2)
         .- N2_rr .- N2_ii)

# N₂(φ₁,φ₁) complex
N2_FD = ComplexF64.(N2_rr .- N2_ii) .+ 2im .* ComplexF64.(N2_ri)
@printf(io, "  ‖N₂_FD(φ₁,φ₁)‖ = %.6g\n", norm(N2_FD))

# Compare L(2λ₁) W[2,0,0] with +ml_result = -N₂_FD (if FluidConvection = -N₂)
# or with +N₂_FD (if FluidConvection = +N₂)
@printf(io, "  ‖LW200‖ = %.6g\n", norm(LW200))
rel_neg = norm(LW200 .+ N2_FD) / (norm(N2_FD) + 1e-30)    # LW200 = -N₂?
rel_pos = norm(LW200 .- N2_FD) / (norm(N2_FD) + 1e-30)    # LW200 = +N₂?

@printf(io, "\n  |LW200 - (-N₂_FD)| / ‖N₂‖ = %.4g  (small → FC = -N₂, sign correct)\n", rel_neg)
@printf(io, "  |LW200 - (+N₂_FD)| / ‖N₂‖ = %.4g  (small → FC = +N₂, sign WRONG)\n\n", rel_pos)

if rel_neg < rel_pos
    println(io, "  → LW200 ≈ -N₂ → FluidConvection correctly negates N₂ ✓")
elseif rel_pos < rel_neg
    println(io, "  → LW200 ≈ +N₂ → FluidConvection has WRONG sign (should negate N₂)")
else
    println(io, "  → Magnitudes too different to determine sign from LW200 comparison")
end

println(io)

# Scale ratio: does LW200 have expected magnitude?
@printf(io, "  ‖LW200‖/‖N₂_FD‖ = %.4g  (expect 1.0 if multiplier correct)\n",
    norm(LW200)/norm(N2_FD))

# ── Test 4: Reduced dynamics coefficients ────────────────────────────────────
println(io, "\n=== Test 4: Reduced dynamics coefficients ===")
exps_R = R_rom.poly.multiindex_set.exponents
c1 = R_rom.poly.coefficients[1, :]
find_c(a,b,c) = let idx=findfirst(α->α[1]==a&&α[2]==b&&α[3]==c, exps_R)
    isnothing(idx) ? 0.0+0im : c1[idx]
end

c100 = find_c(1,0,0); c101 = find_c(1,0,1); c210 = find_c(2,1,0)
@printf(io, "  c₁₀₀ = λ₁     = %+.6g %+.6gi\n", real(c100), imag(c100))
@printf(io, "  c₁₀₁          = %+.6g %+.6gi\n", real(c101), imag(c101))
@printf(io, "  c₂₁₀ (Landau) = %+.6g %+.6gi\n", real(c210), imag(c210))
@printf(io, "\n  Re(c₂₁₀) sign: %s  (expected NEGATIVE for supercritical Hopf)\n",
    real(c210) < 0 ? "NEGATIVE ✓" : "POSITIVE ✗")

# ── Test 5: Schur complement denominator ─────────────────────────────────────
println(io, "\n=== Test 5: Schur complement denominator [2,1,0] ===")
s_210 = 2λ₁ + conj(λ₁)
@printf(io, "  s[2,1,0] = %+.8f %+.8fi\n", real(s_210), imag(s_210))
@printf(io, "  |s[2,1,0] - λ₁| = %.6g\n", abs(s_210 - λ₁))

# Leading-order approximation: Schur denom ≈ 1/(s-λ₁)
denom_approx = 1.0 / (s_210 - λ₁)
@printf(io, "  Approx Schur denom ≈ 1/(s-λ₁) = %+.4g %+.4gi\n",
    real(denom_approx), imag(denom_approx))

# W[2,0,0] magnitude check
@printf(io, "\n  Expected ‖W[2,0,0]‖ ≈ ‖N₂‖/|2λ₁| = %.4g/%.4g = %.4g\n",
    norm(N2_FD), abs(s_200), norm(N2_FD)/abs(s_200))
@printf(io, "  Actual   ‖W[2,0,0]‖ = %.4g\n", norm(W200))
@printf(io, "  ‖W[2,0,0]‖ × |2λ₁| / ‖N₂_FD‖ = %.4g  (should be ≈1)\n",
    norm(W200) * abs(s_200) / norm(N2_FD))

println(io, "\n=== End ===")

end  # open io

println("Written to debug_conv.log")
