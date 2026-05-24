"""
	parametric_beam_demo.jl

Demo: parametric ROM for a 3D continuum beam under *uniform axial
stretch*.  Companion to `demo/BenchmarkFerrite/benchmark_ferrite.jl`.

Run with the same Project.toml as the benchmark; the mesh
`beam_h27.msh` is reused from BenchmarkFerrite (generate it once with
`generate_beam_mesh.jl` inside that folder).

The script proceeds in five steps:

  (1) load the reference mesh and Ferrite FE space
  (2) build the parametric geometry: J₁, det_series, adj_series, and
	  the scalar reciprocal series 1/det(J) — no `J⁻¹` series is ever
	  materialised; everywhere the weak form would need `inv(J)` we
	  substitute `adj(J) / det(J)` and let one `1/det(J)` cancel the
	  `det(J)` in `dV = det(J) · dV₀`
  (3) assemble the polynomial coefficient matrices K_k, M_k via
	  `assemble_K_M_polynomial!`  (linear K carries `(1/det)¹`, linear
	  M carries `det¹`)
  (4) instantiate the quadratic and cubic parametric internal-force
	  maps, ParametricGeometricNonlinearity{2|3} (which precompute
	  `(1/det)²` and `(1/det)³` respectively), and wrap each θ-power
	  as a MORFE.MultilinearMap of external arity k
  (5) solve the eigenproblem at θ = 0, augment the system with the
	  external state θ (eigenvalue 0, trivial dynamics), and call
	  `solve_cohomological_problem` on the (z₁, z₂, θ) variables

Output: the reduced dynamics in (z₁, z₂, θ) — a *parametric* ROM that
can be evaluated at any θ in the trust region without re-solving the
cohomological problem.

The truncation order in θ is a single knob, `N_θ`.
"""

using MORFE
using Ferrite
using FerriteGmsh
using SparseArrays
using LinearAlgebra
using Arpack
using Printf

# ------------------------------------------------------------------
# Ensure registry packages used by this demo are installed before the
# corresponding `using` statements.  Both of these are transitive
# dependencies of Ferrite, so they are likely already present in any
# environment that can load Ferrite — but `Pkg.add` is idempotent and
# cheap when the package is already registered, so this block is safe
# to keep as a one-shot bootstrap.
using Pkg: Pkg
Pkg.add("Tensors")           # 3D tensor algebra (det, inv, ⋅, ⊡, …)
Pkg.add("StaticArrays")      # SVector for MORFE's master_eigenvalues
using Tensors
using StaticArrays

# ------------------------------------------------------------------
# Load supporting code (order matters)
# ------------------------------------------------------------------
include(joinpath(@__DIR__, "theta_polynomials.jl"))
include(joinpath(@__DIR__, "parametric_geometry.jl"))
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
E       = 160e3
ν      = 0.22
ρ      = 2.32e-3
λ_lame = (E * ν) / ((1 + ν) * (1 - 2ν))
μ_lame = E / (2(1 + ν))
α      = 0.5369754008568333 / 500.0      # mass-proportional damping
β      = 0.0                              # stiffness-proportional damping

# ==================================================================
# 3.  Parametric geometry
# ==================================================================
#
# Uniform axial stretch along x₁:    φ(x₀) = (x_{0,1}, 0, 0),
# so the spatial gradient of φ is the constant rank-1 tensor
#
#     J₁ = ∇₀φ = e₁ ⊗ e₁ ,
#
# and the reference map is x(θ, x₀) = (I + θ J₁) x₀, with J₀ = I.

const J₀ = one(Tensor{2, 3, Float64})
const J₁ = Tensor{2, 3, Float64}((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0)    # e₁ ⊗ e₁

const N_θ = 5                                          # series truncation order

# Joint computation of det and adj: shares the matrix products
# (A², B², AB, BA) and traces between the two series, and avoids
# any inversion of J₀ (so the routine is correct even for J₀ ≠ I).
detJ_coeffs, adjJ_coeffs = det_and_adj_series(J₀, J₁)        # lengths 4 and 3
inv_detJ_coeffs          = reciprocal_series(detJ_coeffs, N_θ)  # length N_θ + 1

println("\nParametric geometry:")
println("  det J(θ) coefficients : ", detJ_coeffs)
println("  adj J(θ) at θ⁰        : ", adjJ_coeffs[1])
println("  adj J(θ) at θ¹        : ", adjJ_coeffs[2])
println("  1/det J(θ) coefficients (first few): ", inv_detJ_coeffs[1:min(end, 4)])

# Recurrence sanity check.  We verify the polynomial identity
# J(θ) · adj(J(θ)) ≡ det(J(θ)) · I  exactly (no reciprocal involved),
# which is the defining property of the adjugate.
adj_resid = check_adj_det_identity(J₀, J₁, adjJ_coeffs, detJ_coeffs)
@printf "  Identity residual ‖J·adj(J) − det(J)·I‖:\n"
for (k, r) in pairs(adj_resid)
	@printf "    θ^%d :  %.3e\n" (k - 1) r
end
@assert all(<(1e-12), adj_resid) "adj/det polynomial identity failed."

# ==================================================================
# 4.  Linear assembly:  K(θ) = Σ θ^k K_k,  M(θ) = Σ θ^k M_k
# ==================================================================
const N_K_used = N_θ                                       # match DPIM order
const N_M_used = length(detJ_coeffs) - 1                    # M's exact degree

K_full_coeffs = [allocate_matrix(dh) for _ in 0:N_K_used]
M_full_coeffs = [allocate_matrix(dh) for _ in 0:N_M_used]

println("\nAssembling K_k (k = 0…$N_K_used) and M_k (k = 0…$N_M_used) …")
@time assemble_K_M_polynomial!(K_full_coeffs, M_full_coeffs,
	dh, cv, λ_lame, μ_lame, ρ,
	adjJ_coeffs, detJ_coeffs, inv_detJ_coeffs)

# Apply Dirichlet BCs by restricting to free DOFs
free          = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free        = length(free)

K_k = [Kf[free, free] for Kf in K_full_coeffs]              # [K₀, K₁, …, K_{N_θ}]
M_k = [Mf[free, free] for Mf in M_full_coeffs]              # [M₀, M₁, …]

K = K_k[1]                                                  # θ⁰ — passed to NDOrderModel
M = M_k[1]
C = α * M + β * K
println("Free DOFs : ", n_free)

# ==================================================================
# 5.  Multiindex set on (z₁, z₂, θ)
# ==================================================================
const ROM        = 2
const N_EXT      = 1                       # θ is one real external state
const NVAR       = ROM + N_EXT
const max_degree = 5
mset             = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
_max_uniq        = length(mset)

# ==================================================================
# §1  Eigenproblem on the reference (θ = 0) configuration
# ==================================================================
println("\n§1 Eigenproblem on (K₀, M₀) …")
solver_eig = StructureModalDampingEigensolver(10, α, β)
r1 = @timed solve_eigenproblem(K, M, solver_eig; sorter! = (args...) -> nothing)
eigenproblem = r1.value
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)

println("  First eigenvalues (reference):")
for (i, λi) in enumerate(eigenvalues)
	println("    mode $i:  λ = $λi   |λ| = $(abs(λi))")
end

FOM = n_free
select_master_modes_by_sorting(eigenproblem, ROM)
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes       = Y[:, 1, 1:ROM]
left_eigenmodes    = X[:, 1:ROM]

ORD_model = size(eigenproblem.eigenmodes, 2)
master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
for r in 1:ROM, k in 1:(ORD_model-1)
	master_modes_derivatives[:, k, r] .= Y[:, k+1, r]
end

# ==================================================================
# 6.  Wrap K_k, C_k corrections as MultilinearMaps
# ==================================================================
#
# Sign convention (mirrors `term_cubic` in the benchmark): every
# *internal-force* `MultilinearMap` is added with a minus sign because
# MORFE writes them on the right-hand side of
#
#     M ẍ + C ẋ + K x  =  (sum of all multilinear terms).
#
# So a linear stiffness correction `θ^k K_k u` enters as
# `-θ^k K_k u`, and likewise for damping.
#
# NOTE on parametric mass.  MORFE's `MultilinearMap` supports modal
# arity `(a_pos, a_vel)`, but has no acceleration slot, so a term of
# the form `θ^k M_k ẍ` cannot be expressed.  We therefore *omit* the
# M-corrections at this level; the inertial part of the mass change is
# not captured here.  The *damping* part of the mass change, however,
# IS captured through the Rayleigh formula `C(θ) = α M(θ) + β K(θ)`,
# whose θ-corrections enter via `build_linear_C_corrections` below.
# For our problem β = 0, so only `C_1 = α M_1` is non-zero (higher
# corrections vanish because det J(θ) is degree 1).
linear_K_corrections = build_linear_K_corrections(K_k, N_K_used)
linear_C_corrections = build_linear_C_corrections(K_k, M_k, α, β, N_K_used, N_M_used)
println("  linear K-corrections : ", length(linear_K_corrections),
	"  linear C-corrections : ", length(linear_C_corrections))

# ==================================================================
# 7.  Parametric quadratic and cubic geometric nonlinearities
# ==================================================================
println("\nBuilding ParametricGeometricNonlinearity{2} and {3} …")
pgn_quad = ParametricGeometricNonlinearity{2}(
	dh, cv, free_to_local, n_free, λ_lame, μ_lame;
	adjJ_coeffs     = adjJ_coeffs,
	inv_detJ_coeffs = inv_detJ_coeffs,
	N_θ            = N_θ,
)
pgn_cube = ParametricGeometricNonlinearity{3}(
	dh, cv, free_to_local, n_free, λ_lame, μ_lame;
	adjJ_coeffs     = adjJ_coeffs,
	inv_detJ_coeffs = inv_detJ_coeffs,
	N_θ            = N_θ,
)

quad_maps = multilinear_maps(pgn_quad)   # N_θ + 1 maps with external arity 0…N_θ
cube_maps = multilinear_maps(pgn_cube)

# ==================================================================
# 8.  External system (θ̇ = 0)  and  model assembly
# ==================================================================
ext_sys = ExternalSystem((complex(0.0, 0.0),))

model = NDOrderModel(
	(K, C, M),
	(quad_maps...,
		cube_maps...,
		linear_K_corrections...,
		linear_C_corrections...),
	ext_sys,
)

super_eigenvalues  = Vector{ComplexF64}([master_eigenvalues..., complex(0.0, 0.0)])
target_eigenvalues = Vector{ComplexF64}(master_eigenvalues)

resonance_set = resonance_set_from_complex_normal_form_style(
	ROM, mset, super_eigenvalues, target_eigenvalues, 0.05)

# ==================================================================
# §2  Cohomological solve on the augmented (z₁, z₂, θ) system
# ==================================================================
println("\n§2 Cohomological solve  (NVAR = $NVAR, max_degree = $max_degree, N_θ = $N_θ) …")
r2 = @timed solve_cohomological_problem(
	model, mset,
	master_eigenvalues,
	master_modes, left_eigenmodes,
	resonance_set;
	master_modes_derivatives = master_modes_derivatives,
	conjugate_permutation    = [2, 1, 3],     # z₁↔z₂; θ is real, self-conjugate
)
(W, R) = r2.value

println("\nReduced dynamics coefficients (parametric ROM in (z₁, z₂, θ)):")
for m in 1:length(R.poly.multiindex_set.exponents)
	mi = R.poly.multiindex_set.exponents[m]
	c  = R.poly.coefficients[:, m]
	any(abs.(c) .> 1e-12) && println("  $mi   $(c[1])\t$(c[2])")
end

# ==================================================================
# 9.  Analytical scaling check  (bending-mode case)
# ==================================================================
#
# For an Euler-Bernoulli clamped-clamped beam:  ω_b(θ) ∝ (1+θ)⁻²
#     ⇒  d/dθ ω_b |_{θ=0} = −2 ω_b(0).
#
# The (z̄_i + θ × z_i) coefficient of the ROM diagonal entry for the
# bending master mode should approximate this slope.
println("\nAnalytical scaling (uniform axial stretch):")
for i in 1:ROM
	ωi_0 = abs(master_eigenvalues[i])
	@printf "  mode %d : ω(0) = %.6f      −2 ω(0) = %.6f    (bending dω/dθ)\n" i ωi_0 (-2*ωi_0)
end
println("  Compare with the θ-coefficient of the diagonal z_i entry in the ROM above.")
println("  (Axial modes would show dω/dθ = −ω(0).)")

# ==================================================================
# Summary
# ==================================================================
to_gb(b) = round(b / 1024^3; digits = 2)
sep = "=" ^ 72
println()
println(sep)
println("MORFE.jl — Parametric Ferrite Demo (uniform axial stretch)")
println(@sprintf("  Mesh    : H27 quadratic Lagrange hex (40×2×2)   FOM = %d", FOM))
println(@sprintf("  ROM     : %d modes + %d parameter  max_degree = %d  N_θ = %d",
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
