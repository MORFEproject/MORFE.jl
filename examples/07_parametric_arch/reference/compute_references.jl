"""
	compute_references.jl

Non-parametric DPIM baselines for the clamped-clamped arch beam.

For each arch height ratio in `H_RATIOS`, this script:
  1. Loads the flat beam mesh.
  2. Assembles K₀, M₀ on the flat mesh using the analytical arch Jacobian
     series at h = h_ratio · L via assemble_K_M_arch! (same code as main.jl).
  3. Runs a DPIM solve directly (NVAR=2, no θ) using ArchGeometricNonlinearity
     for the nonlinear maps — again the same code as main.jl at k=0.
  4. Saves R_coefficients.csv and summary.txt under
         results/reference/arch_h_<h_mm>mm/

These reference ROMs serve two purposes:
  a) Cross-validation: compare the parametric ROM (main.jl) at θ=0 against
     the reference at the same h₀/L — they should match to FEM precision
     because the computation is IDENTICAL (same mesh, same assembly, k=0 maps).
  b) Documentation of how the SSM shape and reduced dynamics evolve across
     arch heights, independently of the θ parametrisation.

The flat beam dimensions are 1000 × 10 × 24 mm (standard MORFE benchmark).

Usage (from the repository root):
	julia --project=examples/07_parametric_arch \\
		  examples/07_parametric_arch/reference/compute_references.jl
"""

# -----------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
if !isfile(joinpath(@__DIR__, "..", "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../../..")))
	Pkg.add([
		"Ferrite", "FerriteGmsh", "Arpack", "LinearMaps",
		"Tensors", "StaticArrays", "SparseArrays", "NPZ",
	])
end
Pkg.instantiate()

using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps
using LinearAlgebra, SparseArrays, Printf, Serialization
using Tensors, StaticArrays, NPZ

# -----------------------------------------------------------------------
# Same fem helpers as main.jl
# -----------------------------------------------------------------------
_fem = joinpath(@__DIR__, "..", "fem")
include(joinpath(_fem, "theta_polynomials.jl"))
include(joinpath(_fem, "parametric_geometry.jl"))
include(joinpath(_fem, "arch_geometry.jl"))
include(joinpath(_fem, "arch_assembly.jl"))

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

const _MESH = joinpath(@__DIR__, "..", "..", "..", "benchmark", "ferrite",
	"beam_h27_10x2x2.msh")
isfile(_MESH) || error("Mesh not found at $_MESH.  Run generate_beam_mesh.jl first.")

include(joinpath(@__DIR__, "..", "config.jl"))   # h0_L_ratio, N_INCREMENTS

const L = 1000.0
const ORDER = 11        # DPIM expansion order (same as main.jl's max_degree_z)

# Material constants — identical to main.jl §2
const E = 160e3
const ν_mat = 0.22
const ρ_mat = 2.32e-3
const λ_lame = (E * ν_mat) / ((1 + ν_mat) * (1 - 2ν_mat))
const μ_lame = E / (2 * (1 + ν_mat))
const α_damp = 0.0
const β_damp = 0.0

# Exact polynomial degrees for the sinusoidal arch (J_arch² = 0, det J = 1)
const N_θ_G_ref = 3   # G(u₁,u₂;θ) is exactly degree 3 in θ
const N_θ_H_ref = 4   # H(u₁,u₂,u₃;θ) is exactly degree 4 in θ

const ROM_REF = 2

# Reference arch heights: same range as main.jl's parametric sweep
const H_RATIOS = collect(range(0.0, 2 * h0_L_ratio; length = N_INCREMENTS + 1))

const _REF_DIR = joinpath(@__DIR__, "..", "results", "reference")
mkpath(_REF_DIR)

# -----------------------------------------------------------------------
# §1  Mesh and FE space — built ONCE from the flat grid (same as main.jl)
# -----------------------------------------------------------------------

println("Loading flat beam mesh …")
flat_grid = togrid(_MESH)
println("  Cells : ", getncells(flat_grid), "   Nodes : ", getnnodes(flat_grid))

ip     = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr     = QuadratureRule{RefHexahedron}(3)
cv     = CellValues(qr, ip, geo_ip)

dh = DofHandler(flat_grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(flat_grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)

println("Total DOFs : ", ndofs(dh))

free          = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free        = length(free)

println("Free DOFs  : ", n_free)

# -----------------------------------------------------------------------
# §2  Main loop over arch heights
# -----------------------------------------------------------------------

for h_ratio in H_RATIOS
	h   = h_ratio * L
	tag = @sprintf("arch_h_%.1fmm", h)
	dir = joinpath(_REF_DIR, tag)
	mkpath(dir)

	println("\n" * "=" ^ 60)
	println("Reference ROM  h/L = $h_ratio   (h = $h mm)")
	println("=" ^ 60)

	# a) Assemble K₀, M₀ on the flat mesh for this arch height
	#    N_θ = 0 → only the k=0 coefficient matrix (no parametric expansion)
	K_arr_full = [allocate_matrix(dh)]
	M_arr_full = [allocate_matrix(dh)]
	@printf "  Assembling K₀, M₀ …"
	t_assem = @elapsed assemble_K_M_arch!(K_arr_full, M_arr_full,
		dh, cv, λ_lame, μ_lame, ρ_mat, h, L, free_to_local, 0)
	@printf " %.2f s\n" t_assem
	K₀ = K_arr_full[1][free, free]
	M₀ = M_arr_full[1][free, free]
	C₀ = α_damp * M₀ + β_damp * K₀

	# b) Eigenproblem — same as main.jl §5
	solver_eig  = StructureModalDampingEigensolver(10, α_damp, β_damp)
	eigenproblem = solve_eigenproblem(K₀, M₀, solver_eig; sorter! = (args...) -> nothing)
	eigenvalues, Y, X = get_eigenpairs(eigenproblem)

	select_master_modes_by_sorting(eigenproblem, ROM_REF)
	master_eigenvalues = SVector{ROM_REF, ComplexF64}(eigenvalues[1:ROM_REF])
	master_modes       = Y[:, 1, 1:ROM_REF]
	left_eigenmodes    = X[:, 1:ROM_REF]

	ω₀ = abs(master_eigenvalues[1])
	@printf "  ω₀ = %.6f rad/ms\n" ω₀

	# c) master_modes_derivatives — ORD=3 structure (same as main.jl lines 188–191)
	#    Block 1: λ φ  (from eigensolver)
	#    Block 2: λ² φ (analytical)
	master_modes_derivatives = zeros(ComplexF64, n_free, 2, ROM_REF)
	for r in 1:ROM_REF
		master_modes_derivatives[:, 1, r] .= Y[:, 2, r]
		master_modes_derivatives[:, 2, r] .= master_eigenvalues[r] .* Y[:, 2, r]
	end

	# d) Nonlinear maps — k=0 coefficient only (same as pROM at θ=0)
	pgn_quad      = ArchGeometricNonlinearity{2}(dh, cv, λ_lame, μ_lame, h, L,
		free_to_local, n_free, N_θ_G_ref)
	pgn_cube      = ArchGeometricNonlinearity{3}(dh, cv, λ_lame, μ_lame, h, L,
		free_to_local, n_free, N_θ_H_ref)
	quad_maps_ref = multilinear_maps(pgn_quad)
	cube_maps_ref = multilinear_maps(pgn_cube)

	# e) NDOrderModel: ORD=3, NVAR=2; k=0 maps only (no θ series)
	ZERO  = spzeros(eltype(K₀), n_free, n_free)
	model = NDOrderModel(
		(K₀, C₀, M₀, ZERO),
		(quad_maps_ref[1], cube_maps_ref[1]),
	)

	# f) Multiindex set: NVAR=2, same degree cap as main.jl
	mset = all_multiindices_up_to(2, ORDER; min_degree = 1)

	resonance_set = resonance_set_from_complex_normal_form_style(
		mset, Vector{ComplexF64}(master_eigenvalues), 0.05)

	# g) Cohomological solve
	@printf "  Solving cohomological equations (%d monomials) …" length(mset)
	t_solve = @elapsed W_ref, R_ref = solve_cohomological_problem(
		model, mset, master_eigenvalues,
		master_modes, left_eigenmodes, resonance_set;
		master_modes_derivatives = master_modes_derivatives,
		conjugate_permutation    = [2, 1])
	@printf " %.2f s\n" t_solve

	# h) Save W.jls, R.jls, rom.npz, R_coefficients.csv
	data_dir = joinpath(dir, "data")
	mkpath(data_dir)
	serialize(joinpath(data_dir, "W.jls"), W_ref)
	serialize(joinpath(data_dir, "R.jls"), R_ref)
	let exps = W_ref.poly.multiindex_set.exponents
		npzwrite(joinpath(data_dir, "rom.npz"), Dict(
			"W"         => W_ref.poly.coefficients,
			"R"         => R_ref.poly.coefficients,
			"exponents" => Int32.(hcat([collect(e) for e in exps]...)),
		))
	end
	open(joinpath(data_dir, "R_coefficients.csv"), "w") do io
		exps   = R_ref.poly.multiindex_set.exponents
		coeffs = R_ref.poly.coefficients
		println(io, "exp_1,exp_2,R1_re,R1_im,R2_re,R2_im")
		for (m, ex) in enumerate(exps)
			c = coeffs[:, m]
			any(abs.(c) .> 1e-14) || continue
			println(io,
				"$(Int(ex[1])),$(Int(ex[2]))," *
				"$(real(c[1])),$(imag(c[1]))," *
				"$(real(c[2])),$(imag(c[2]))")
		end
	end

	open(joinpath(dir, "summary.txt"), "w") do io
		println(io, "model: ArchGeometricNonlinearity on flat mesh (same as main.jl)")
		println(io, "h_ratio: $h_ratio")
		println(io, "h_mm: $h")
		println(io, "L_mm: $L")
		println(io, "n_dofs_free: $n_free")
		println(io, "ROM: $ROM_REF")
		println(io, "master_eigenvalues: $(collect(master_eigenvalues))")
		println(io, "parametrisation_order: $ORDER")
		println(io, "n_monomials: $(length(mset))")
		println(io, "assemble_time_s: $t_assem")
		println(io, "cohomological_solve_time_s: $t_solve")
		println(io, "julia_version: $(VERSION)")
		commit = try
			readchomp(`git rev-parse --short HEAD`)
		catch
			"unknown"
		end
		println(io, "morfe_commit: $commit")
		println(io, "timestamp: $(time())")
	end

	println("  Saved → $dir/")
end

println("\n" * "=" ^ 60)
println("All references written to $_REF_DIR/")
println("  Subdirectories: ",
	join(map(r -> @sprintf("arch_h_%.1fmm", r * L), H_RATIOS), ", "))
println()
println("Cross-validation: compare arch_h_$(@sprintf("%.1f", h0_L_ratio * L))mm/data/R_coefficients.csv")
println("  against results/data/$(@sprintf("arch_h%.3f", h0_L_ratio))/R_coefficients.csv (pROM, exp_3=0 rows).")
println("  Expect ≲ 1e-12 relative error — computation is identical.")
