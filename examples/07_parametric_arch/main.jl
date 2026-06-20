"""
	main.jl

Demo: single-parameter parametric ROM for a 3D clamped-clamped arch beam.

The base configuration is a sinusoidal arch of height h₀ = h₀_L_ratio · L.
A single external parameter θ controls the deviation from this base arch:

	x(θ, x₀) = x₀ + (1 + θ) · w(x₀)
	J(θ, x₀) = J₀(x₀) + θ · J₁(x₀)

	J₀(x₀) = I + J_arch(x₀)     (base arch Jacobian)
	J₁(x₀) = J_arch(x₀)         (perturbation Jacobian)
	J_arch(x₀) = (π h₀ / L) cos(π x₁ / L) · (e₂ ⊗ e₁)

Parameter scaling:
  θ = −1 → straight beam (J = I everywhere)
  θ = 0  → base arch    (J = J₀)
  θ = +1 → doubled arch  (height = 2 h₀)

The DPIM is run on the reduced system (z₁, z₂, θ) — NVAR = 3.

Steps:
  (1) Load mesh and FE space.
  (2) Assemble all K_k, M_k coefficient matrices via assemble_K_M_arch!.
  (3) Solve eigenproblem on the base-arch (K₀, M₀).
  (4) Build ArchGeometricNonlinearity{2,3} nonlinear maps.
  (5) Build K/C/M corrections as MultilinearMaps (θ-powers k ≥ 1).
  (6) Construct NDOrderModel with one frozen external state (θ).
  (7) Build anisotropic multiindex set on (z₁, z₂, θ).
  (8) solve_cohomological_problem → (W, R).
  (9) Save ROM to results/data/.
"""

# -----------------------------------------------------------------------
# Bootstrap: activate demo-local environment
# -----------------------------------------------------------------------
using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
	Pkg.add([
		"Ferrite", "FerriteGmsh",
		"Arpack", "LinearMaps",
		"Tensors", "StaticArrays",
		"Serialization",
	])
end
Pkg.instantiate()

using MORFE
using Ferrite
using FerriteGmsh
using SparseArrays
using LinearAlgebra
using Arpack, LinearMaps
using Printf
using Tensors
using StaticArrays
using Serialization

# -----------------------------------------------------------------------
# Load supporting code (order matters)
# -----------------------------------------------------------------------
include(joinpath(@__DIR__, "fem", "theta_polynomials.jl"))
include(joinpath(@__DIR__, "fem", "parametric_geometry.jl"))
include(joinpath(@__DIR__, "fem", "arch_geometry.jl"))
include(joinpath(@__DIR__, "fem", "arch_assembly.jl"))

# =======================================================================
# §0  User parameters
# =======================================================================

const h₀_L_ratio = 0.10           # arch rise / span  (change to explore different arches)
const N_θ = 4                      # polynomial truncation order in θ
const max_degree_z = 9             # DPIM expansion order in normal coordinates (z₁, z₂)
const max_degree_θ = N_θ           # truncation in the parameter θ (anisotropic)
const max_degree_total = max_degree_z  # cap on total degree

const ROM = 2                      # number of master modes (first bending pair)
const N_EXT = 1                    # one external parameter: θ
const NVAR = ROM + N_EXT           # = 3

# =======================================================================
# §1  Mesh and FE setup
# =======================================================================

const _msh = joinpath(@__DIR__, "..", "..", "benchmark", "ferrite", "beam_h27_10x2x2.msh")
isfile(_msh) || error("Mesh not found at $_msh.  Run generate_beam_mesh.jl in benchmark/ferrite/ first.")

println("Loading mesh …")
grid = togrid(_msh)
ip = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr = QuadratureRule{RefHexahedron}(3)
cv = CellValues(qr, ip, geo_ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)

println("Total DOFs : ", ndofs(dh))

free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free = length(free)
FOM = n_free

# =======================================================================
# §2  Material constants
# =======================================================================

E = 160e3
ν = 0.22
ρ = 2.32e-3
λ_lame = (E * ν) / ((1 + ν) * (1 - 2ν))
μ_lame = E / (2(1 + ν))
α = 0.0
β = 0.0

# =======================================================================
# §3  Arch geometry
# =======================================================================

const L = 1000.0
const h₀ = h₀_L_ratio * L

@printf "\nArch parameters: h₀ = %.1f mm,  L = %.1f mm,  h₀/L = %.4f\n" h₀ L h₀_L_ratio

# Quick geometry sanity: arch Jacobian pair at the beam midpoint
let x_mid = Vec{3, Float64}((L / 2, 0.0, 0.0))
	J₀_mid, J₁_mid = arch_jacobian_pair(x_mid, h₀, L)
	det_ser, _ = det_and_adj_series(J₀_mid, J₁_mid)
	@printf "det(J₀) at x₁=L/2: %.8f  (expected 1.0 for sinusoidal arch)\n" det_ser[1]
end

# =======================================================================
# §4  Assemble K_k and M_k coefficient matrices (k = 0 … N_θ)
# =======================================================================

println("\nAllocating K/M coefficient matrices (N_θ = $N_θ) …")
K_arr_full = [allocate_matrix(dh) for _ in 0:N_θ]
M_arr_full = [allocate_matrix(dh) for _ in 0:N_θ]

println("Assembling K_arr, M_arr via assemble_K_M_arch! …")
@time assemble_K_M_arch!(K_arr_full, M_arr_full,
	dh, cv, λ_lame, μ_lame, ρ, h₀, L, free_to_local, N_θ)

K_arr = [K_arr_full[k+1][free, free] for k in 0:N_θ]
M_arr = [M_arr_full[k+1][free, free] for k in 0:N_θ]

K = K_arr[1]     # reference stiffness (base arch, θ = 0)
M = M_arr[1]     # reference mass
C = α * M + β * K

println("Free DOFs : ", n_free)

# =======================================================================
# §5  Eigenproblem on (K₀, M₀) — base arch configuration
# =======================================================================

println("\nSolving eigenproblem on base arch (K₀, M₀) …")
solver_eig = StructureModalDampingEigensolver(10, α, β)
r1 = @timed solve_eigenproblem(K, M, solver_eig; sorter! = (args...) -> nothing)
eigenproblem = r1.value
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)

println("  First eigenvalues (base arch, θ = 0):")
for (i, λi) in enumerate(eigenvalues)
	println("    mode $i:  λ = $λi   |λ| = $(abs(λi))")
end

select_master_modes_by_sorting(eigenproblem, ROM)
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes = Y[:, 1, 1:ROM]
left_eigenmodes = X[:, 1:ROM]

master_modes_derivatives = zeros(ComplexF64, FOM, 2, ROM)
for r in 1:ROM
	master_modes_derivatives[:, 1, r] .= Y[:, 2, r]
	master_modes_derivatives[:, 2, r] .= master_eigenvalues[r] .* Y[:, 2, r]
end

# =======================================================================
# §6  Bivariate assembly diagnostic
# =======================================================================
#
# For mass-normalised φ₁:  ∂ω/∂θ|_{θ=0} = (φ₁ᵀ K₁ φ₁ − ω₀² φ₁ᵀ M₁ φ₁) / (2ω₀)
# For the sinusoidal arch, K₁ = K_arr[2] and M₁ = M_arr[2].

let
	φ = real(master_modes[:, 1])
	ω₀_val = abs(master_eigenvalues[1])
	φᵀMφ = dot(φ, M * φ)

	K1_modal = dot(φ, K_arr[2] * φ) / φᵀMφ
	M1_modal = dot(φ, M_arr[2] * φ) / φᵀMφ
	dω_dθ = (K1_modal - ω₀_val^2 * M1_modal) / (2ω₀_val)

	println("\n── Arch assembly diagnostic ──")
	@printf "  ω₀ (base arch) = %+.6f rad/ms\n" ω₀_val
	@printf "  φᵀ M₀ φ = %+.6f  (1.0 if mass-normalised)\n" φᵀMφ
	@printf "  K_arr[2] modal (θ¹ coeff) = %+.6f\n" K1_modal
	@printf "  M_arr[2] modal (θ¹ coeff) = %+.6f  (expected ≈ 1)\n" M1_modal
	@printf "  ∂ω/∂θ  (FEM, first-order) = %+.6f\n" dω_dθ
	println("────────────────────────────────────")
end

# =======================================================================
# §7  Nonlinear maps and corrections
# =======================================================================

println("\nBuilding ArchGeometricNonlinearity{2} and {3} …")
pgn_quad = ArchGeometricNonlinearity{2}(dh, cv, λ_lame, μ_lame, h₀, L,
	free_to_local, n_free, N_θ)
pgn_cube = ArchGeometricNonlinearity{3}(dh, cv, λ_lame, μ_lame, h₀, L,
	free_to_local, n_free, N_θ)

quad_maps = multilinear_maps(pgn_quad)
cube_maps = multilinear_maps(pgn_cube)

println("Building K/C/M corrections …")
K_corrections = build_arch_K_corrections(K_arr, N_θ)
C_corrections = build_arch_C_corrections(K_arr, M_arr, α, β, N_θ)
M_corrections = build_arch_M_corrections(M_arr, N_θ)
println("  K: ", length(K_corrections),
	"  C: ", length(C_corrections),
	"  M: ", length(M_corrections))

# =======================================================================
# §8  Augmented system and multiindex set
# =======================================================================

ext_sys = ExternalSystem((complex(0.0, 0.0),))     # single frozen θ

ZERO = spzeros(eltype(K), n_free, n_free)

model = NDOrderModel(
	(K, C, M, ZERO),
	(quad_maps...,
		cube_maps...,
		K_corrections...,
		C_corrections...,
		M_corrections...),
	ext_sys,
)

println("\nBuilding multiindex set (NVAR=$NVAR, deg_z ≤ $max_degree_z, deg_θ ≤ $max_degree_θ) …")
mset = MultiindexSet([
	SVector{NVAR, Int}(a, b, c)
	for a in 0:max_degree_z for b in 0:max_degree_z
	for c in 0:max_degree_θ
	if a + b ≤ max_degree_z && c ≤ max_degree_θ && 1 ≤ a + b + c ≤ max_degree_total
])
println("  Monomials: ", length(mset))

resonance_set = resonance_set_from_complex_normal_form_style(
	mset, Vector{ComplexF64}(master_eigenvalues), 0.05;
	external_eigenvalues = zeros(ComplexF64, N_EXT))

# =======================================================================
# §9  Cohomological solve
# =======================================================================

println("\n§9 Cohomological solve (NVAR=$NVAR, deg_z ≤ $max_degree_z, deg_θ ≤ $max_degree_θ) …")
r2 = @timed solve_cohomological_problem(
	model, mset,
	master_eigenvalues,
	master_modes, left_eigenmodes,
	resonance_set;
	master_modes_derivatives = master_modes_derivatives,
	conjugate_permutation = [2, 1, 3],   # z₁ ↔ z₂; θ real → self-conjugate
)
(W, R) = r2.value

# =======================================================================
# §10  Output
# =======================================================================

println("\nReduced dynamics coefficients R  (z₁, z₂, θ):")
for m in 1:length(R.poly.multiindex_set.exponents)
	mi = R.poly.multiindex_set.exponents[m]
	c = R.poly.coefficients[:, m]
	any(abs.(c) .> 1e-12) && println("  $mi   $(c[1])\t$(c[2])")
end

println("\nFrequency sensitivity check (mode 1 at θ=0):")
ω₀ = abs(master_eigenvalues[1])
@printf "  ω₀ = %.6f rad/ms     ∂ω/∂θ|_{θ=0} ≈ %.6f  (ROM coefficient Im(R₁₁₀₁)/ω₀)\n" ω₀ ω₀

to_gb(b) = round(b / 1024^3; digits = 2)
sep = "=" ^ 72
println()
println(sep)
println("MORFE.jl — Parametric Arch ROM  (h₀/L = $h₀_L_ratio)")
println(@sprintf "  FOM = %d   ROM = %d   N_EXT = %d   N_θ = %d" FOM ROM N_EXT N_θ)
println(@sprintf "  deg_z ≤ %d   deg_θ ≤ %d   monomials = %d" max_degree_z max_degree_θ length(mset))
println("-" ^ 72)
println(@sprintf "  %-32s %9s %11s %7s" "Phase" "Time (s)" "Memory (GB)" "GC (s)")
println("-" ^ 72)
println(@sprintf "  §5 %-28s %9.3f %11.2f %7.3f" "Eigenproblem" r1.time to_gb(r1.bytes) r1.gctime)
println(@sprintf "  §9 %-28s %9.3f %11.2f %7.3f" "Cohomological solve" r2.time to_gb(r2.bytes) r2.gctime)
println("-" ^ 72)
println(@sprintf "  %-32s %9.3f %11.2f %7.3f" "Σ Cumulative" (r1.time + r2.time) (to_gb(r1.bytes) + to_gb(r2.bytes)) (r1.gctime + r2.gctime))
println(sep)

# =======================================================================
# §11  Save ROM
# =======================================================================

const _results_dir = joinpath(@__DIR__, "results", "data",
	@sprintf("arch_h%.3f", h₀_L_ratio))
mkpath(_results_dir)

serialize(joinpath(_results_dir, "W.jls"), W)
serialize(joinpath(_results_dir, "R.jls"), R)

open(joinpath(_results_dir, "summary.txt"), "w") do io
	println(io, "MORFE.jl — Parametric Arch ROM")
	println(io, "h0_L_ratio = $h₀_L_ratio")
	println(io, "h0_mm = $h₀")
	println(io, "L_mm = $L")
	println(io, "FOM = $FOM")
	println(io, "ROM = $ROM")
	println(io, "N_EXT = $N_EXT")
	println(io, "N_theta = $N_θ")
	println(io, "max_degree_z = $max_degree_z")
	println(io, "max_degree_θ = $max_degree_θ")
	println(io, "master_eigenvalues = $(collect(master_eigenvalues))")
	println(io, "eigenproblem_time_s = $(r1.time)")
	println(io, "cohomological_time_s = $(r2.time)")
	println(io, "julia_version: $(VERSION)")
	commit = try
		;
		readchomp(`git rev-parse --short HEAD`);
	catch
		;
		"unknown";
	end
	println(io, "morfe_commit: $commit")
	println(io, "timestamp: $(time())")
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

println("\nROM saved to $(_results_dir)/")
println("  W.jls, R.jls, summary.txt, R_coefficients.csv")
println("\nDemo finished successfully.")
