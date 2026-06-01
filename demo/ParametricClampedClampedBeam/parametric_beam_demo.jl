"""
	parametric_beam_demo.jl

Demo: two-parameter parametric ROM for a 3D continuum beam:

  θ₁ — uniform axial stretch   (J₁ = e₁⊗e₁, constant)
  θ₂ — arch pre-deformation    (J₂(x₀) = ∇φ₁(x₀), first bending mode)

Reference map:  x(θ₁,θ₂,x₀) = x₀ + θ₁ J₁ x₀ + θ₂ φ₁(x₀)
Full Jacobian:  J(θ₁,θ₂,x₀) = I + θ₁ J₁ + θ₂ J₂(x₀)

Since J₂ varies per quadrature point, all polynomial series (det, adj,
1/det) are computed per QP using bivariate polynomials in (θ₁,θ₂)
via bivariate_geometry.jl / bivariate_polynomials.jl.

The script proceeds in six steps:

  (1) load the reference mesh and Ferrite FE space
  (2) solve the eigenproblem at θ₁=θ₂=0 to get master modes and
	  the arch mode φ₁ (first bending eigenmode)
  (3) assemble bivariate K_{k₁,k₂}, M_{k₁,k₂} coefficient matrices
	  via assemble_K_M_bivariate!
  (4) instantiate the bivariate quadratic/cubic nonlinear maps
	  ParametricGeometricNonlinearity2D{2|3}
  (5) augment the system with two external states (θ₁, θ₂), each with
	  eigenvalue 0 (frozen dynamics), and build the NVAR=4 multiindex set
  (6) call solve_cohomological_problem on the (z₁, z₂, θ₁, θ₂) variables

Output: the reduced dynamics in (z₁, z₂, θ₁, θ₂) — a two-parameter ROM.
Truncation order N_θ applies to both parameters equally.
"""

# ------------------------------------------------------------------
# Bootstrap: activate a demo-local environment so that Ferrite and
# other FEM packages are not added to MORFE's root Project.toml.
# On first run this installs everything (~2–5 min); subsequent runs
# skip straight to Pkg.instantiate (seconds).
# ------------------------------------------------------------------
using Pkg: Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
	Pkg.add([
		"Ferrite", "FerriteGmsh",
		"Arpack", "LinearMaps",
		"Tensors", "StaticArrays",
	])
end
Pkg.instantiate()

using MORFE
using Ferrite
using FerriteGmsh
using SparseArrays
using LinearAlgebra
using Arpack, LinearMaps   # LinearMaps triggers MORFEArpackExt
using Printf
using Tensors
using StaticArrays

# ------------------------------------------------------------------
# Load supporting code (order matters)
# ------------------------------------------------------------------
include(joinpath(@__DIR__, "theta_polynomials.jl"))
include(joinpath(@__DIR__, "parametric_geometry.jl"))
include(joinpath(@__DIR__, "bivariate_polynomials.jl"))
include(joinpath(@__DIR__, "bivariate_geometry.jl"))
include(joinpath(@__DIR__, "parametric_assembly.jl"))

# ==================================================================
# 1.  Mesh and FE setup  (identical to benchmark_ferrite.jl)
# ==================================================================
const _msh = joinpath(@__DIR__, "..", "BenchmarkFerrite", "beam_h27.msh")
isfile(_msh) || error("Mesh not found at $_msh.  Run generate_beam_mesh.jl in BenchmarkFerrite/ first.")

println("Loading mesh …")
grid   = togrid(_msh)
ip     = Lagrange{RefHexahedron, 2}()^3      # vector field, 3 components
geo_ip = Lagrange{RefHexahedron, 2}()        # quadratic isoparametric geometry
qr     = QuadratureRule{RefHexahedron}(3)    # 3³ Gauss points
cv     = CellValues(qr, ip, geo_ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)

println("Total DOFs : ", ndofs(dh))

# ==================================================================
# 2.  Material constants (same as the benchmark)
# ==================================================================
E = 160e3
ν = 0.22
ρ = 2.32e-3
λ_lame = (E * ν) / ((1 + ν) * (1 - 2ν))
μ_lame = E / (2(1 + ν))
α = 0.5369754008568333 / 500.0      # mass-proportional damping
β = 0.0                             # stiffness-proportional damping

# ==================================================================
# 3.  Parametric geometry
# ==================================================================
#
# θ₁: Uniform axial stretch along x₁:    J₁ = ∇₀φ = e₁ ⊗ e₁
# θ₂: Arch via first bending eigenmode   J₂(x₀) = ∇₀φ₁(x₀)   [computed per QP]
#
# The reference map is:
#   x(θ₁, θ₂, x₀) = x₀ + θ₁ J₁ x₀ + θ₂ φ₁(x₀)
#   J(θ₁, θ₂, x₀) = I + θ₁ J₁ + θ₂ J₂(x₀)

const J₀_mat = one(Tensor{2, 3, Float64})
const J₁ = Tensor{2, 3, Float64}((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0)    # e₁ ⊗ e₁

const N_θ = 4 # truncation order (both θ₁ and θ₂)

# Apply Dirichlet BCs — needed before eigenproblem and before bivariate assembly.
free          = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free        = length(free)

# ==================================================================
# 5.  Multiindex set on (z₁, z₂, θ₁, θ₂)
# ==================================================================
const ROM = 2
const N_EXT = 2                       # two real external states θ₁, θ₂
const NVAR = ROM + N_EXT              # = 4
const max_degree = 5
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
_max_uniq = length(mset)

# ==================================================================
# §1  Eigenproblem on the reference (θ₁=θ₂=0) configuration
# ==================================================================
#
# At the reference configuration J = I, so K₀ and M₀ are the standard
# isotropic FEM stiffness and mass assembled with the identity Jacobian.
# We use the existing univariate assembly (adjJ = [I], detJ = [1]) to
# get K₀ and M₀ before the full bivariate assembly.
println("\n§1 Computing K₀, M₀ for eigenproblem …")
_adj0 = [one(Tens3)]                    # adj(I) = I (constant, degree-0)
_det0 = [1.0, 0.0, 0.0, 0.0]           # det(I) = 1 (degree-0)
_invdet0 = reciprocal_series(_det0, 0)  # 1/1 = [1.0] (degree-0)

K0_full = allocate_matrix(dh)
M0_full = allocate_matrix(dh)
M0_aux  = allocate_matrix(dh)   # placeholder for M degree-1 slot
assemble_K_M_polynomial!([K0_full], [M0_full, M0_aux],
	dh, cv, λ_lame, μ_lame, ρ, _adj0, _det0, _invdet0)

FOM = n_free
K_ref = K0_full[free, free]
M_ref = M0_full[free, free]

println("Solving eigenproblem (K₀, M₀) …")
solver_eig = StructureModalDampingEigensolver(10, α, β)
r1 = @timed solve_eigenproblem(K_ref, M_ref, solver_eig; sorter! = (args...) -> nothing)
eigenproblem = r1.value
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)

println("  First eigenvalues (reference):")
for (i, λi) in enumerate(eigenvalues)
	println("    mode $i:  λ = $λi   |λ| = $(abs(λi))")
end
select_master_modes_by_sorting(eigenproblem, ROM)
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes       = Y[:, 1, 1:ROM]
left_eigenmodes    = X[:, 1:ROM]

ORD_model = size(eigenproblem.eigenmodes, 2)
master_modes_derivatives = zeros(ComplexF64, FOM, 2, ROM)
for r in 1:ROM
	master_modes_derivatives[:, 1, r] .= Y[:, 2, r]                           # λ·φ
	master_modes_derivatives[:, 2, r] .= master_eigenvalues[r] .* Y[:, 2, r]  # λ²·φ
end

# ==================================================================
# 6.  Extract arch mode  (first bending eigenmode in free-DOF space)
# ==================================================================
#
# θ₂ pre-deforms the beam in the shape φ₁ of the first master mode.
# We take the real part (the mode is complex only due to damping;
# for mass-proportional damping the imaginary part is negligible).
# Normalise so max(|u|) = 1, keeping θ₂ dimensionally sensible.
arch_mode_free = real(master_modes[:, 1])
arch_mode_free ./= maximum(abs, arch_mode_free)

# Quick sanity: check J · adj(J) = det(J) · I at a representative
# J₂ (the mean gradient of the arch mode over the reference domain).
# We skip the per-QP check here for brevity; it is verified implicitly
# by the assembly (wrong adjugate → wrong K₀ → wrong eigenfrequency).

# ==================================================================
# 7.  Bivariate K/M assembly
# ==================================================================
#
# K_b[k₁+1, k₂+1] and M_b[k₁+1, k₂+1] are the coefficient matrices
# for the (θ₁^k₁ θ₂^k₂) term of the linear stiffness/mass forms.

println("\nAllocating bivariate K/M coefficient matrices …")
K_b_full = [allocate_matrix(dh) for k1 in 0:N_θ, k2 in 0:N_θ]
M_b_full = [allocate_matrix(dh) for k1 in 0:N_θ, k2 in 0:N_θ]

println("Assembling K_b and M_b (bivariate, N_θ = $N_θ) …")
@time assemble_K_M_bivariate!(K_b_full, M_b_full,
	dh, cv, λ_lame, μ_lame, ρ, J₁, arch_mode_free,
	free_to_local, N_θ)

# Restrict to free DOFs
K_b = [K_b_full[k1+1, k2+1][free, free] for k1 in 0:N_θ, k2 in 0:N_θ]
M_b = [M_b_full[k1+1, k2+1][free, free] for k1 in 0:N_θ, k2 in 0:N_θ]

K = K_b[1, 1]    # reference stiffness (θ₁⁰ θ₂⁰)
M = M_b[1, 1]    # reference mass
C = α * M + β * K
println("Free DOFs : ", n_free)

# Quick sanity: det_and_adj_bseries convention check
let J₀_test = one(Tens3), J₂_zero = zero(Tens3)
	d1, _ = det_and_adj_bseries(J₀_test, J₁, J₂_zero, 1)
	d2, _ = det_and_adj_bseries(J₀_test, J₂_zero, J₁, 1)
	@printf "det_b[2,1] when B=J₁, C=0: %.4f  (expect 1.0)\n" d1[2, 1]
	@printf "det_b[1,2] when B=J₁, C=0: %.4f  (expect 0.0)\n" d1[1, 2]
	@printf "det_b[2,1] when B=0,  C=J₁: %.4f (expect 0.0)\n" d2[2, 1]
	@printf "det_b[1,2] when B=0,  C=J₁: %.4f (expect 1.0)\n" d2[1, 2]
end

# ==================================================================
# Diagnostic: modal stiffness/mass corrections — verifies the
#             bivariate assembly produces physically correct K_b.
#
# For mass-normalised φ₁ (φ₁ᵀ M₀ φ₁ = 1):
#   ∂ω/∂θ₁  =  (φ₁ᵀ K_b[2,1] φ₁ − ω₀² φ₁ᵀ M_b[2,1] φ₁) / (2ω₀)
#   ∂ω/∂θ₂  =  (φ₁ᵀ K_b[1,2] φ₁ − ω₀² φ₁ᵀ M_b[1,2] φ₁) / (2ω₀)
#
# Expected: ∂ω/∂θ₁ ≈ −2ω₀ (axial stretch)  |  ∂ω/∂θ₂ ≈ 0 (arch)
# ==================================================================
let
	φ = arch_mode_free                          # max-normalised; need mass norm
	ω₀_val = abs(master_eigenvalues[1])

	φᵀM₀φ = dot(φ, M * φ)                     # = 1 if mass-normalised

	K10_modal = dot(φ, K_b[2, 1] * φ) / φᵀM₀φ  # θ₁ stiffness correction
	M10_modal = dot(φ, M_b[2, 1] * φ) / φᵀM₀φ  # θ₁ mass correction
	K01_modal = dot(φ, K_b[1, 2] * φ) / φᵀM₀φ  # θ₂ stiffness correction
	M01_modal = dot(φ, M_b[1, 2] * φ) / φᵀM₀φ  # θ₂ mass correction

	dω_dθ1 = (K10_modal - ω₀_val^2 * M10_modal) / (2ω₀_val)
	dω_dθ2 = (K01_modal - ω₀_val^2 * M01_modal) / (2ω₀_val)

	println("\n── Bivariate assembly diagnostic ──")
	@printf "  ω₀                     = %+.6f rad/ms\n" ω₀_val
	@printf "  φᵀ M₀ φ                = %+.6f  (1.0 if mass-normalised)\n" φᵀM₀φ
	println()
	@printf "  K_b[2,1] modal (θ₁)    = %+.6f\n" K10_modal
	@printf "  M_b[2,1] modal (θ₁)    = %+.6f  (expected ≈ 1)\n" M10_modal
	@printf "  ∂ω/∂θ₁  (FEM)          = %+.6f   expected ≈ %+.6f  (−2ω₀)\n" dω_dθ1 (-2ω₀_val)
	println()
	@printf "  K_b[1,2] modal (θ₂)    = %+.6f\n" K01_modal
	@printf "  M_b[1,2] modal (θ₂)    = %+.6f  (expected ≈ 0)\n" M01_modal
	@printf "  ∂ω/∂θ₂  (FEM)          = %+.6f   expected ≈ 0\n" dω_dθ2
	println("────────────────────────────────────")
end

# ==================================================================
# 8.  Bivariate nonlinear maps and model assembly
# ==================================================================
#
# Sign convention: every internal-force MultilinearMap is negated
# (MORFE writes nonlinear terms on the RHS of M ẍ + C ẋ + K x = …).

println("\nBuilding ParametricGeometricNonlinearity2D{2} and {3} …")
pgn_quad = ParametricGeometricNonlinearity2D{2}(
	dh, cv, λ_lame, μ_lame, J₁, arch_mode_free, free_to_local, n_free, N_θ)
pgn_cube = ParametricGeometricNonlinearity2D{3}(
	dh, cv, λ_lame, μ_lame, J₁, arch_mode_free, free_to_local, n_free, N_θ)

quad_maps = multilinear_maps(pgn_quad)
cube_maps = multilinear_maps(pgn_cube)

println("Building bivariate linear corrections …")
linear_K_corrections = build_bivariate_K_corrections(K_b, N_θ)
linear_C_corrections = build_bivariate_C_corrections(K_b, M_b, α, β, N_θ)
linear_M_corrections = build_bivariate_M_corrections(M_b, N_θ)
println("  K-corrections : ", length(linear_K_corrections),
	"  C-corrections : ", length(linear_C_corrections),
	"  M-corrections : ", length(linear_M_corrections))

# Two external states (θ₁, θ₂), both frozen (ṙ = 0 → λ_ext = 0).
ext_sys = ExternalSystem((complex(0.0, 0.0), complex(0.0, 0.0)))

ZERO = spzeros(eltype(K), n_free, n_free)

model = NDOrderModel(
	(K, C, M, ZERO),
	(quad_maps...,
		cube_maps...,
		linear_K_corrections...,
		linear_C_corrections...,
		linear_M_corrections...),
	ext_sys,
)

# Two external eigenvalues (both 0) appended to master eigenvalues.
super_eigenvalues  = Vector{ComplexF64}([master_eigenvalues..., 0.0 + 0im, 0.0 + 0im])
target_eigenvalues = Vector{ComplexF64}(master_eigenvalues)

resonance_set = resonance_set_from_complex_normal_form_style(
	ROM, mset, super_eigenvalues, target_eigenvalues, 0.05)

# ==================================================================
# §2  Cohomological solve on the augmented (z₁, z₂, θ₁, θ₂) system
# ==================================================================
println("\n§2 Cohomological solve  (NVAR = $NVAR, max_degree = $max_degree, N_θ = $N_θ) …")
r2 = @timed solve_cohomological_problem(
	model, mset,
	master_eigenvalues,
	master_modes, left_eigenmodes,
	resonance_set;
	master_modes_derivatives = master_modes_derivatives,
	conjugate_permutation    = [2, 1, 3, 4],  # z₁↔z₂; θ₁,θ₂ real → self-conjugate
)
(W, R) = r2.value

println("\nReduced dynamics coefficients (parametric ROM in (z₁, z₂, θ₁, θ₂)):")
for m in 1:length(R.poly.multiindex_set.exponents)
	mi = R.poly.multiindex_set.exponents[m]
	c  = R.poly.coefficients[:, m]
	any(abs.(c) .> 1e-12) && println("  $mi   $(c[1])\t$(c[2])")
end

# ==================================================================
# 9.  Analytical scaling check
# ==================================================================
#
# At θ₂ = 0, the uniform-stretch scaling ω(θ₁) ∝ (1+θ₁)⁻² gives
# ∂ω/∂θ₁|_{θ=0} = −2 ω₀ for bending modes.
# The (0,0,1,0) monomial coefficient in R should reflect this.
println("\nAnalytical scaling (bending mode, uniform axial stretch θ₁ at θ₂=0):")
for i in 1:ROM
	ωi_0 = abs(master_eigenvalues[i])
	@printf "  mode %d : ω(0,0) = %.6f      ∂ω/∂θ₁ ≈ −2ω₀ = %.6f\n" i ωi_0 (-2*ωi_0)
end
println("  Check the (0,0,1,0) monomial diagonal entry in R above.")

# ==================================================================
# Summary
# ==================================================================
to_gb(b) = round(b / 1024^3; digits = 2)
sep = "=" ^ 72
println()
println(sep)
println("MORFE.jl — Two-Parameter Parametric Ferrite Demo (axial stretch + arch)")
println(@sprintf("  Mesh    : H27 quadratic Lagrange hex (40×2×2)   FOM = %d", FOM))
println(@sprintf("  ROM     : %d modes + %d parameters  max_degree = %d  N_θ = %d",
	ROM, N_EXT, max_degree, N_θ))
println("-" ^ 72)
println(@sprintf("  %-32s %9s %11s %7s", "Phase", "Time (s)", "Memory (GB)", "GC (s)"))
println("-" ^ 72)
println(@sprintf("  §1 %-28s %9.3f %11.2f %7.3f",
	"Eigenproblem", r1.time, to_gb(r1.bytes), r1.gctime))
println(@sprintf("  §2 %-28s %9.3f %11.2f %7.3f",
	"Cohomological solve", r2.time, to_gb(r2.bytes), r2.gctime))
println("-" ^ 72)
println(@sprintf("  %-32s %9.3f %11.2f %7.3f",
	"Σ Cumulative (§1+§2)",
	r1.time + r2.time,
	to_gb(r1.bytes) + to_gb(r2.bytes),
	r1.gctime + r2.gctime))
println(@sprintf("  Monomials (max_degree=%d, NVAR=%d) = %d",
	max_degree, NVAR, length(mset.exponents)))
println(sep)

println("\nDemo finished successfully.")

# ==================================================================
# 10.  Save ROM to results/
# ==================================================================
using Serialization

const _results_dir = joinpath(@__DIR__, "results")
mkpath(_results_dir)

serialize(joinpath(_results_dir, "W.jls"), W)
serialize(joinpath(_results_dir, "R.jls"), R)

open(joinpath(_results_dir, "summary.txt"), "w") do io
	println(io, "MORFE.jl — Parametric Clamped-Clamped Beam ROM")
	println(io, "FOM          = $FOM")
	println(io, "ROM          = $ROM")
	println(io, "N_EXT        = $N_EXT")
	println(io, "N_theta      = $N_θ")
	println(io, "max_degree   = $max_degree")
	println(io, "master_eigenvalues = $(collect(master_eigenvalues))")
	println(io, "eigenproblem_time_s    = $(r1.time)")
	println(io, "cohomological_time_s   = $(r2.time)")
end

open(joinpath(_results_dir, "R_coefficients.csv"), "w") do io
	exps = R.poly.multiindex_set.exponents
	NVAR_R = size(R.poly.coefficients, 1)
	header = join(["exp_$i" for i in 1:length(exps[1])], ",") * "," *
			 join(["R$(i)_re,R$(i)_im" for i in 1:NVAR_R], ",")
	println(io, header)
	for (m, ex) in enumerate(exps)
		c = R.poly.coefficients[:, m]
		any(abs.(c) .> 1e-14) || continue
		row = join(string.(Int.(ex)), ",") * "," *
			  join(["$(real(c[i])),$(imag(c[i]))" for i in 1:NVAR_R], ",")
		println(io, row)
	end
end

println("ROM saved to $(_results_dir)/")
println("  W.jls, R.jls, summary.txt, R_coefficients.csv")
