"""
Diagnose the -7.583× factor between MORFE.jl and legacy reduced dynamics.

Computes and compares the cohomological bordered-system operators
(C_r, J_r, Schur complement S) for the monomial [2,1,0,0] using:
  - MORFE.jl convention  (what benchmark_ferrite.jl currently computes)
  - Legacy convention    (what legacy MORFE2.0 homological_HALF! computes)

Run from the repo root:
	julia --project benchmark/ferrite/diagnose_operators.jl
"""

import Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "MORFE")
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
    Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays",
        "BenchmarkTools", "Gmsh"])
end
Pkg.instantiate()

using MORFE
using Ferrite
using FerriteGmsh
using SparseArrays
using LinearAlgebra
using Arpack
using LinearMaps
using StaticArrays
using Printf

include(joinpath(@__DIR__, "../Ferrite/ferrite_assembly.jl"))

# ---------------------------------------------------------------------------
# 1. Assemble K, M, C (copied from benchmark_ferrite.jl)
# ---------------------------------------------------------------------------
const _msh = joinpath(@__DIR__, "beam_h27.msh")
isfile(_msh) || error("Mesh not found: run generate_beam_mesh.jl first.")
println("Loading mesh …")
grid   = togrid(_msh)
ip     = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr     = QuadratureRule{RefHexahedron}(3)
cv     = CellValues(qr, ip, geo_ip)
dh = DofHandler(grid);
add!(dh, :u, ip);
close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch);
update!(ch, 0.0)

E = 160e3;
ν = 0.22;
ρ      = 2.32e-3
λ_lame = (E*ν) / ((1+ν)*(1-2ν))
μ_lame = E / (2(1+ν))
α_ray  = 0.5369754008568333/500.0
β_ray  = 0.0

K_full = allocate_matrix(dh);
M_full = allocate_matrix(dh)
assemble_KM!(K_full, M_full, dh, cv, λ_lame, μ_lame, ρ)
free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
K = K_full[free, free];
M = M_full[free, free];
C = α_ray * M
n_free = length(free)
println("Free DOFs: ", n_free)

# ---------------------------------------------------------------------------
# 2. Eigensolution via StructureModalDampingEigensolver
# ---------------------------------------------------------------------------
println("\nSolving eigenproblem …")
solver_eig = StructureModalDampingEigensolver(4, α_ray, β_ray)
eigenproblem = spectrum(K, M, solver_eig; sorter! = (args...) -> nothing)
(eigenvalues, Y, X) = (eigenproblem.eigenvalues, eigenproblem.eigenmodes, eigenproblem.left_eigenmodes)

λ1 = eigenvalues[1];
λ2 = eigenvalues[2]
println("λ1 = ", λ1, "   (expect ≈ -0.000537 + 0.5374im)")
println("λ2 = ", λ2)

φ1 = Y[:, 1, 1]   # mass-normalised position eigenvector, mode 1
φ2 = Y[:, 1, 2]   # mode 2 = conj(mode 1)

# left eigenmodes as set by StructureModalDampingEigensolver: Y[:, 1, :] = φ (position block)
ℓ1_current = X[:, 1]   # = φ1 currently

println("\nMass-norm check:  φ1ᵀMφ1 = ", real(dot(φ1, M * φ1)))
println("Eucl-norm:        φ1ᵀφ1   = ", real(dot(φ1, φ1)))

# ---------------------------------------------------------------------------
# 3. Operators for monomial [2,1,0,0] — σ = 2λ1 + λ2
# ---------------------------------------------------------------------------
σ = 2λ1 + λ2
println("\n─── Monomial [2,1,0,0], σ = ", σ)

# MORFE.jl C_r(σ) = (C + (λ1+σ)*M)*φ1   [precompute_master_column_polynomials, ORD=2]
C_r_morfe = (C + (λ1 + σ)*M) * φ1

# MORFE.jl J_r(σ) = ((λ1+σ)*M + C)*ℓ1   [ℓ1 = φ1 currently → same as C_r]
J_r_morfe = C_r_morfe   # identical since ℓ1 = φ1

# Legacy: C_col = J_row = (σ - conj(λ1)) * φ1
C_r_legacy = (σ - conj(λ1)) * φ1

println("\n‖C_r_morfe‖  = ", norm(C_r_morfe))
println("‖C_r_legacy‖ = ", norm(C_r_legacy))
println("‖morfe‖/‖legacy‖ = ", norm(C_r_morfe) / norm(C_r_legacy))

# Are they proportional?  C_r_morfe ≈ γ * C_r_legacy  ?
# If C ≈ 0: C_r_morfe ≈ (λ1+σ)*M*φ1 and C_r_legacy ≈ (σ-λ̄1)*φ1
# Ratio component-wise: [(λ1+σ)*M*φ1]_i / [(σ-λ̄1)*φ1_i]  = (λ1+σ)*[M*φ1]_i / [(σ-λ̄1)*φ1_i]
# NOT a scalar unless M*φ1 ∝ φ1.
println("\nFirst entries of C_r_morfe  (re): ", real.(C_r_morfe[1:4]))
println("First entries of C_r_legacy (re): ", real.(C_r_legacy[1:4]))
println("Component ratios [morfe/legacy]:   ",
	[C_r_morfe[i]/C_r_legacy[i] for i in 1:4])

# ---------------------------------------------------------------------------
# 4. Schur complement denominators
# ---------------------------------------------------------------------------
println("\nComputing L(σ) factorisation (dense) …")
L_mat = Matrix(K + σ*C + σ^2*M)   # FOM×FOM dynamic stiffness
L_fac = lu(L_mat)

C_prime_morfe  = L_fac \ Vector(complex.(C_r_morfe))
C_prime_legacy = L_fac \ Vector(complex.(C_r_legacy))

# S = J_r^T * C_prime - Ĉ  (Ĉ = 1 for both)
S_morfe  = dot(J_r_morfe, C_prime_morfe) - 1.0
S_legacy = dot(C_r_legacy, C_prime_legacy) - 1.0   # J_r = C_r in both cases

println("\nSchur denominator  S_morfe  = ", S_morfe)
println("Schur denominator  S_legacy = ", S_legacy)
println("S_morfe / S_legacy          = ", S_morfe / S_legacy)

# ---------------------------------------------------------------------------
# 5. Test with a surrogate f_nl direction (φ1 → W_prime → J.W_prime)
#    In reality f_nl contains the actual nonlinear force, but the R ratio
#    equals (J_morfe · L⁻¹ · f) / (J_legacy · L⁻¹ · f) * S_legacy/S_morfe
#    for any f, provided g ≈ 0 (true for lowest-order nonlinear monomials).
# ---------------------------------------------------------------------------
println("\n─── Ratio analysis (g≈0 limit) ───")
println("R_morfe/R_legacy = [J_morfe · L⁻¹f / J_legacy · L⁻¹f] × [S_legacy / S_morfe]")

# Use a generic test direction for L⁻¹f:
test_rhs = randn(ComplexF64, n_free)
W_prime_test = L_fac \ test_rhs

num_morfe  = dot(J_r_morfe, W_prime_test)
num_legacy = dot(C_r_legacy, W_prime_test)

R_ratio_num   = num_morfe / num_legacy
R_ratio_denom = S_legacy / S_morfe
R_ratio_total = R_ratio_num * R_ratio_denom

println("  Numerator ratio  J_morfe/J_legacy at random rhs = ", R_ratio_num)
println("  Denominator ratio  S_legacy/S_morfe              = ", R_ratio_denom)
println("  Effective R_morfe/R_legacy                       = ", R_ratio_total)
println("  Observed from txt files                          ≈ ", -7.583)

# ---------------------------------------------------------------------------
# 6. What left_eigenmodes gives J_r = (σ-λ̄1)*φ1?
#    J_r(σ) = (λ1+σ)*M*ℓ1 + C*ℓ1 = (C + (λ1+σ)M)*ℓ1
#    We want this = (σ-λ̄1)*φ1.
#    Try ℓ1 = M^{-1}*φ1:
# ---------------------------------------------------------------------------
println("\n─── Could left_eigenmodes = M⁻¹φ give legacy J_r? ───")
Minv_φ1 = M \ φ1
J_r_Minv = (C + (λ1 + σ)*M) * Minv_φ1
println("  J_r with ℓ=M⁻¹φ first entries: ", J_r_Minv[1:4])
println("  (σ-λ̄1)*φ   first entries:      ", C_r_legacy[1:4])
println("  ‖diff‖/‖legacy‖ = ", norm(J_r_Minv - C_r_legacy)/norm(C_r_legacy))

# C_r comes from master_modes; if master_modes = M⁻¹φ:
C_r_Minv = (C + (λ1 + σ)*M) * Minv_φ1
println("\n  C_r with master=M⁻¹φ: ", C_r_Minv[1:4])
println("  Legacy C_r:            ", C_r_legacy[1:4])
println("  Match? ‖diff‖/‖legacy‖ = ", norm(C_r_Minv - C_r_legacy)/norm(C_r_legacy))

# But then W initialisation W[:,1,e1] = M⁻¹φ ≠ φ → wrong eigenvectors in W!
# So this approach breaks the W initialisation.

# ---------------------------------------------------------------------------
# 7. Correct fix: the bordered system should match legacy.
#    Legacy uses (σ-λ̄1)*φ1 for both C and J, with Ĉ=1 and g = -φ1^T*Wf_pos.
#    In MORFE.jl, the C_r and J_r come from the ORD=2 system:
#      B[3] = M, B[2] = C, B[1] = K
#    The correct approach for the HALF formulation:
#      C_r_correct(σ) = (σ - conj(λ_r))*φ_r
#    This requires changing precompute_master_column_polynomials to produce this.
#
#    Alternatively: change what's passed as master_modes to produce the right C_r.
# ---------------------------------------------------------------------------
println("\n─── Summary ───")
println("MORFE.jl C_r = (C + (λ+σ)M)φ ≈ 2iω*Mφ — has mass matrix factor!")
println("Legacy   C_r = (σ - λ̄)φ     ≈ 2iω*φ  — no mass matrix!")
println()
println("The factor of M (mass matrix) between the two conventions")
println("explains the non-trivial S and numerator difference → -7.583×.")
println()
println("Fix: the C_r column in the cohomological bordered system should be")
println("computed without the mass-matrix weighting, matching legacy formula.")
