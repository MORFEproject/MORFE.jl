"""
Multi-mesh benchmark suite for MORFE.jl — Ferrite path.

Sweeps five denser mesh sizes of the clamped-clamped H27 beam at degree 5:

   80× 4×2  → ~23 625 free DoFs
   80× 4×4  → ~47 025 free DoFs
  160× 4×4  → ~94 125 free DoFs
  160× 8×4  → ~188 025 free DoFs
  160× 8×8  → ~375 825 free DoFs

For each mesh the script measures:
  §1  Spectrum
  §2  Cohomological solve (ROM=2, N_EXT=2, max_degree=5)

Results are written to timestamped subfolders:
  benchmark_results/{mesh_name}_degree{max_degree}_{YYYYMMDDTHHMMSS}/
	benchmark_per_monomial.csv   — per-monomial timing from cohomological solver
	benchmark_per_order.csv      — aggregated per polynomial degree
	summary.txt                  — high-level timing/memory numbers

A consolidated table is printed at the end.

Prerequisites:
  Run generate_beam_meshes.jl once to create the .msh files.

Usage:
  julia --project benchmark/ferrite/benchmark_suite.jl
"""

using Pkg: Pkg
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
using Dates
using Printf

include(joinpath(@__DIR__, "../Ferrite/ferrite_assembly.jl"))

# -----------------------------------------------------------------------
# Suite parameters (fixed across all mesh sizes)
# -----------------------------------------------------------------------

const SUITE_MESHES = [
(20, 4, 4),
(40, 8, 8)# (640, 2, 2)
]
const SUITE_ROM    = 2
const SUITE_N_EXT  = 2
const SUITE_NVAR   = SUITE_ROM + SUITE_N_EXT
const SUITE_DEGREE = 5

const E = 160e3
const ν_ratio = 0.22
const ρ_val = 2.32e-3
const α_damp = 0.5369754008568333 / 500.0
const β_damp = 0.0
const λ_lame = (E * ν_ratio) / ((1 + ν_ratio) * (1 - 2ν_ratio))
const μ_lame = E / (2(1 + ν_ratio))

const BENCH_BASE = joinpath(@__DIR__, "benchmark_results")

# -----------------------------------------------------------------------
# Per-mesh benchmark function
# -----------------------------------------------------------------------

"""
	benchmark_mesh(nx, ny, nz; max_degree=SUITE_DEGREE) -> NamedTuple

Load the mesh `beam_h27_{nx}x{ny}x{nz}.msh`, assemble linear matrices,
solve the eigenproblem and cohomological problem (ROM=$(SUITE_ROM)), and
write timed results to a timestamped subdirectory of `benchmark_results/`.

Returns `(; mesh_name, n_free, r1, r2, bench_dir)`.
"""
function benchmark_mesh(nx::Int, ny::Int, nz::Int; max_degree::Int = SUITE_DEGREE)
	mesh_name = "beam_h27_$(nx)x$(ny)x$(nz)"
	msh_path = joinpath(@__DIR__, "$(mesh_name).msh")
	isfile(msh_path) ||
		error("Mesh not found: $(msh_path)\nRun generate_beam_meshes.jl first.")

	timestamp = Dates.format(now(), "yyyymmddTHHMMSS")
	bench_dir = joinpath(BENCH_BASE, "$(mesh_name)_degree$(max_degree)_$(timestamp)")
	mkpath(bench_dir)

	sep = "=" ^ 70
	println()
	println(sep)
	println("Mesh: $(mesh_name)")
	println(sep)

	# -------------------------------------------------------------------
	# 1. Mesh and FE setup
	# -------------------------------------------------------------------
	println("Loading mesh …")
	grid = togrid(msh_path)
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

	println("  Total DOFs : ", ndofs(dh))

	# -------------------------------------------------------------------
	# 2. Linear matrices
	# -------------------------------------------------------------------
	K_full = allocate_matrix(dh)
	M_full = allocate_matrix(dh)
	assemble_KM!(K_full, M_full, dh, cv, λ_lame, μ_lame, ρ_val)

	free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
	free_to_local = Dict(d => i for (i, d) in enumerate(free))
	n_free = length(free)

	K = K_full[free, free]
	M = M_full[free, free]
	C = α_damp * M + β_damp * K

	println("  Free DOFs  : ", n_free)

	# -------------------------------------------------------------------
	# 3. Multiindex set
	# -------------------------------------------------------------------
	mset = all_multiindices_up_to(SUITE_NVAR, max_degree; min_degree = 1)
	max_uniq = length(mset)

	# -------------------------------------------------------------------
	# §1 — Spectrum
	# -------------------------------------------------------------------
	println("\n§1  Spectrum …")
	solver_eig = StructureModalDampingEigensolver(10, α_damp, β_damp)

	r1 = @timed spectrum(K, M, solver_eig; sorter! = (args...) -> nothing)
	eigenproblem = r1.value
	(eigenvalues, Y, X) = (eigenproblem.eigenvalues, eigenproblem.eigenmodes, eigenproblem.left_eigenmodes)

	println("  First eigenvalues:")
	for (i, λi) in enumerate(eigenvalues)
		println("    mode $i: λ = $λi")
	end

	FOM = n_free

	master_eigenvalues = SVector{SUITE_ROM, ComplexF64}(eigenvalues[1:SUITE_ROM])
	master_modes = Y[:, 1, 1:SUITE_ROM]
	left_eigenmodes = X[:, 1:SUITE_ROM]

	ORD_model = size(eigenproblem.eigenmodes, 2)
	master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, SUITE_ROM)
	for r in 1:SUITE_ROM, k in 1:(ORD_model-1)
		master_modes_derivatives[:, k, r] .= Y[:, k+1, r]
	end

	# -------------------------------------------------------------------
	# 4. Force terms
	# -------------------------------------------------------------------
	term_quad = FerriteGeometricNonlinearity{2}(
		dh, cv, free_to_local, n_free, λ_lame, μ_lame; max_unique_cols = max_uniq)
	term_cubic = FerriteGeometricNonlinearity{3}(
		dh, cv, free_to_local, n_free, λ_lame, μ_lame; max_unique_cols = max_uniq)

	Ω_force = abs(eigenvalues[1])
	f_vec = real(2.5 .* (M * master_modes[:, 1]))
	term_forcing = MultilinearMap(
		(res, r) -> (res .+= f_vec * sum(r)),
		(0, 0), 1,
	)

	ext_sys = ExternalSystem((complex(0.0, Ω_force), complex(0.0, -Ω_force)))
	model   = NthOrderModel(
	(K, C, M),
	(term_quad, term_cubic, term_forcing),
	ext_sys
)

	resonance_set = resonance_set_from_complex_normal_form_style(
		mset, Vector{ComplexF64}(master_eigenvalues), 0.05;
		external_eigenvalues = ComplexF64[im * Ω_force, -im * Ω_force])

	# -------------------------------------------------------------------
	# §2 — Cohomological solve
	# -------------------------------------------------------------------
	println("\n§2  Cohomological solve (max_degree = $(max_degree)) …")
	left_modes_derivatives = left_eigenmode_orders_from_slice(
		model.linear_terms, left_eigenmodes, collect(master_eigenvalues))[:, 1:(end - 1), :]
	spectral = SpectralData(; eigenvalues = master_eigenvalues,
		right_modes = master_modes, right_derivatives = master_modes_derivatives,
		left_modes = left_eigenmodes, left_blocks = Array(left_modes_derivatives))
	r2 = @timed solve_cohomological_problem(
		model, mset, spectral, resonance_set;
		conjugate_permutation = [2, 1, 4, 3],
		benchmark_dir = bench_dir,
	)
	(_, R) = r2.value

	# -------------------------------------------------------------------
	# Per-run summary file
	# -------------------------------------------------------------------
	to_gb(b) = b / 1024^3

	open(joinpath(bench_dir, "summary.txt"), "w") do io
		println(io, "mesh        = $(mesh_name)")
		println(io, "FOM         = $(n_free)")
		println(io, "ROM         = $(SUITE_ROM)")
		println(io, "N_EXT       = $(SUITE_N_EXT)")
		println(io, "max_degree  = $(max_degree)")
		println(io, "monomials   = $(length(mset.exponents))")
		println(io, "timestamp   = $(timestamp)")
		println(io)
		println(io, "eig_time_s        = $(r1.time)")
		println(io, "eig_bytes         = $(r1.bytes)")
		println(io, "eig_gctime_s      = $(r1.gctime)")
		println(io, "solve_time_s      = $(r2.time)")
		println(io, "solve_bytes       = $(r2.bytes)")
		println(io, "solve_gctime_s    = $(r2.gctime)")
		println(io, "total_time_s      = $(r1.time + r2.time)")
	end
	println("  Results → ", bench_dir)

	# Print reduced dynamics summary
	println("\n  Reduced dynamics coefficients (non-trivial):")
	for m in 1:length(R.poly.multiindex_set.exponents)
		mi = R.poly.multiindex_set.exponents[m]
		c  = R.poly.coefficients[:, m]
		any(abs.(c) .> 1e-12) && println("  $mi  $(c[1])  $(c[2])")
	end

	return (; mesh_name, n_free, r1, r2, bench_dir)
end

# -----------------------------------------------------------------------
# Run the suite
# -----------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__

	println("=" ^ 70)
	println("MORFE.jl — Multi-Mesh Benchmark Suite (Ferrite path)")
	println("  ROM=$(SUITE_ROM)  N_EXT=$(SUITE_N_EXT)  max_degree=$(SUITE_DEGREE)")
	println("  Meshes: ", join(["beam_h27_$(nx)x$(ny)x$(nz)" for (nx, ny, nz) in SUITE_MESHES], ", "))
	println("=" ^ 70)

	suite_results = []
	for (nx, ny, nz) in SUITE_MESHES
		push!(suite_results, benchmark_mesh(nx, ny, nz))
	end

	# -----------------------------------------------------------------------
	# Consolidated summary table
	# -----------------------------------------------------------------------

	to_gb(b) = b / 1024^3

	println()
	println("=" ^ 85)
	println("MORFE.jl — Benchmark Suite Results  (Ferrite, H27 beam, degree=$(SUITE_DEGREE))")
	println("-" ^ 85)
	@printf("%-24s  %6s  %9s  %9s  %9s  %9s\n", "Mesh", "FOM", "eig (s)", "solve (s)", "eig (GB)", "solve (GB)")
	println("-" ^ 85)
	for r in suite_results
		@printf("%-24s  %6d  %9.3f  %9.3f  %9.3f  %9.3f\n",
			r.mesh_name, r.n_free, r.r1.time, r.r2.time, to_gb(r.r1.bytes), to_gb(r.r2.bytes))
	end
	println("-" ^ 85)
	total_t = sum(r.r1.time + r.r2.time for r in suite_results)
	@printf("%-24s  %6s  %9s  %9s  %9s  %9s\n", "Total wall time", "", "", "", "", "")
	@printf("  %.1f s\n", total_t)
	println("=" ^ 85)
	println()
	println("Per-run results in:")
	for r in suite_results
		println("  ", r.bench_dir)
	end
	println()
	println("Suite finished successfully.")

end # if abspath(PROGRAM_FILE) == @__FILE__
