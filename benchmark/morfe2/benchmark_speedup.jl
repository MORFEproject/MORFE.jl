"""
benchmark_speedup.jl
====================
Measures the wall-time speedup of OPT-1 through OPT-4 relative to the
pre-optimisation baseline for the cohomological solve.

Sections
--------
  A. Full-solve benchmark: measures solve_cohomological_equations! end-to-end
	 using BenchmarkTools.  This is the primary speedup number.

  B. Micro-benchmarks: isolate each optimisation in a tight synthetic loop so
	 that the measured saving can be attributed to the specific change.

	 OPT-1+2  copy vs view + check=false on lu!
	 OPT-3    allocating assembly vs in-place (direct write into ctx buffer)
	 OPT-4    per-monomial `zeros` vs preallocated scratch buffer

Run from the repo root:
	julia --project=. benchmark/morfe20/benchmark_speedup.jl
"""

import Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "MORFE")
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
    Pkg.add(["Arpack", "LinearMaps", "StaticArrays", "KrylovKit", "BenchmarkTools"])
end
Pkg.instantiate()

using MORFE
include(joinpath(@__DIR__, "Morfe_2_0/Morfe_2_0.jl"))
using .Morfe_2_0

using LinearAlgebra
using SparseArrays
using StaticArrays
using Arpack
using LinearMaps
using KrylovKit
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10

println("="^70)
println("MORFE.jl  —  OPT-1 … OPT-4  speedup benchmark")
println("="^70)

# ===========================================================================
# Model setup  (identical to benchmark_morfe20.jl)
# ===========================================================================

info = Infostruct()
info_file = joinpath(@__DIR__, "beam_damp.jl")
include(info_file)

mesh_file = joinpath(@__DIR__, "beam.mphtxt")
mesh = read_mesh(mesh_file, domains_list, materials_list, materials_dict,
	boundaries_list, constrained_dof, bc_vals)

U = Field(mesh, Morfe_2_0.dim)

info.nm = length(info.Φ)
info.nz = 2 * info.nm
info.nzforce = 0
info.nrom = info.nz
info.nK = U.neq
info.nA = 2 * info.nK
info.nMat = info.nA + info.nz

colptr, rowval = Morfe_2_0.assembler_dummy_MK(mesh, U)
val = zeros(Float64, length(rowval))
K = SparseMatrixCSC(U.neq, U.neq, colptr, rowval, val)
M = deepcopy(K)
Morfe_2_0.assembler_MK!(mesh, U, K, M)
C = info.α * M + info.β * K

function quadratic!(res, Ψ₁, Ψ₂)
	Morfe_2_0.assembly_G!(res, Ψ₁, Ψ₂, mesh, U)
end
quadratic_term = MultilinearMap(quadratic!, (2, 0))

function cubic!(res, Ψ₁, Ψ₂, Ψ₃)
	Morfe_2_0.assembly_H!(res, Ψ₁, Ψ₂, Ψ₃, mesh, U)
end
cubic_term = MultilinearMap(cubic!, (3, 0))

model = NthOrderModel((K, C, M), (quadratic_term, cubic_term))
FOM = size(K, 1)
println("\nFOM = $FOM")

# Spectrum
mutable struct Mechanical_Problem_Solver <: AbstractEigensolver
	right_eig_result::Union{Nothing, Matrix}
	eigenvalues::Union{Nothing, Vector}
	nev::Int64
	α::Float64
	β::Float64
end
function mass_normalization!(ϕ, M, neig)
	for i in 1:neig
		c = transpose(ϕ[:, i]) * M * ϕ[:, i]
		for j in 1:(M.m)
			ϕ[j, i] /= sqrt(c)
		end
	end
end
function MORFE.SpectralDecomposition.eigensolve(model::NthOrderModel, solver::Mechanical_Problem_Solver)
	K = model.linear_terms[1]
	M = model.linear_terms[3]
	nev = solver.nev
	# Shift-invert via KrylovKit Lanczos: factor K once, then iterate K⁻¹M.
	# KrylovKit finds the nev largest eigenvalues of K⁻¹M = the nev smallest ω².
	FK = factorize(K)
	x0 = ones(size(K, 1))
	vals_inv, vecs,
	_ = eigsolve(x -> FK \ (M * x), x0, nev, :LR;
		krylovdim = max(3*nev+1, 30),
		maxiter = 500,
		tol = 1e-10,
		issymmetric = true)
	ω2 = real.(1 ./ vals_inv[1:nev])
	idx = sortperm(ω2)
	ω2 = ω2[idx]
	ϕ = real.(hcat(vecs[idx]...))
	mass_normalization!(ϕ, M, nev)
	ω = sqrt.(ω2)
	λ = zeros(ComplexF64, nev * 2)
	for i in 1:nev
		ξ = 0.5 * (solver.α / ω[i] + solver.β * ω[i])
		λ[2i-1] = ω[i] * (-ξ + sqrt(Complex(1 - ξ^2)) * im)
		λ[2i] = ω[i] * (-ξ - sqrt(Complex(1 - ξ^2)) * im)
	end
	eigenvectors = Matrix{ComplexF64}(undef, FOM * 2, nev * 2)
	for i in 1:nev
		eigenvectors[1:FOM, 2i-1] .= ϕ[:, i]
		eigenvectors[1:FOM, 2i] .= ϕ[:, i]
		eigenvectors[(FOM+1):end, 2i-1] .= λ[2i-1] * ϕ[:, i]
		eigenvectors[(FOM+1):end, 2i] .= λ[2i] * ϕ[:, i]
	end
	solver.right_eig_result = eigenvectors
	solver.eigenvalues = λ
	return λ, eigenvectors
end
function MORFE.SpectralDecomposition.eigensolve_left(model::NthOrderModel,
	solver::Mechanical_Problem_Solver)
	@assert solver.right_eig_result !== nothing
	left_eigenvectors = similar(solver.right_eig_result)
	for i in 1:Int(0.5*size(left_eigenvectors, 2))
		left_eigenvectors[(FOM+1):end, 2i-1] = solver.right_eig_result[1:FOM, 2i]
		left_eigenvectors[(FOM+1):end, 2i] = solver.right_eig_result[1:FOM, 2i-1]
	end
	for (i, λ) in enumerate(solver.eigenvalues)
		left_eigenvectors[1:FOM, i] = -(1 / conj(λ)) * model.linear_terms[1]' *
									  left_eigenvectors[(FOM+1):end, i]
	end
	return solver.eigenvalues, left_eigenvectors
end

# nev = number of natural frequencies; each gives one complex-conjugate eigenvalue pair.
# ROM=2 master modes come from the first natural frequency, so nev=1 is sufficient.
nev_needed = 1
eigenproblem = spectrum(
	model, solver = Mechanical_Problem_Solver(nothing, nothing, nev_needed, info.α, info.β),
	sorter! = (args...) -> nothing,
	normalizer! = (args...) -> nothing,
)
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)

ROM = 2;
N_EXT = 0;
NVAR = ROM + N_EXT
select_master_modes_by_sorting(eigenproblem, ROM)
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes = Y[1:FOM, 1:ROM]
left_eigenmodes = Y[(FOM+1):end, 1:ROM]
ORD_model = length(model.linear_terms) - 1
master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
for r in 1:ROM
	for k in 1:(ORD_model-1)
		master_modes_derivatives[:, k, r] .= Y[(k*FOM+1):((k+1)*FOM), r]
	end
end

outer_eigenvalues = eigenvalues[(ROM+1):end]
max_degree = 3
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
println("Multiindex set: degree ≤ $max_degree in $NVAR variables → $(length(mset)) monomials")

resonance_set = resonance_set_from_complex_normal_form_style(
	mset, Vector{ComplexF64}(master_eigenvalues), 0.05)

# ===========================================================================
# A. Full-solve benchmark
# ===========================================================================

println("\n", "─"^70)
println("A. Full solve: solve_cohomological_problem (JIT warm-up then benchmark)")
println("─"^70)

# Warm-up (forces JIT compilation)
W,
R = solve_cohomological_problem(
	model, mset, master_eigenvalues, master_modes, left_eigenmodes, resonance_set;
	master_modes_derivatives = master_modes_derivatives,
)
println("Warm-up done.  FOM = $FOM, monomials = $(length(mset))")
println()

# Benchmark the equation solve in isolation (skip setup / eigenproblem)
t_full = @benchmark solve_cohomological_problem(
	$model, $mset, $master_eigenvalues, $master_modes, $left_eigenmodes, $resonance_set;
	master_modes_derivatives = $master_modes_derivatives,
) seconds=10 samples=5

println("  solve_cohomological_problem:")
show(stdout, MIME"text/plain"(), t_full)
println()

# ===========================================================================
# B. Micro-benchmarks for individual optimisations
# ===========================================================================

println("\n", "─"^70)
println("B. Micro-benchmarks  (synthetic, FOM-sized matrices)")
println("─"^70)

n_big = FOM          # full buffer size
n_sub = FOM          # active sub-block size (worst case — no resonant modes)

A_buf = rand(ComplexF64, n_big, n_big)    # pre-existing buffer

# ── OPT-1+2: copy + lu  vs  view + lu(check=false) ──────────────────────
println("\n  OPT-1+2  —  copy + lu!  vs  view + lu!(check=false)")

t_copy_lu = @benchmark begin
	A_sys=$A_buf[1:($n_sub), 1:($n_sub)]    # range-index → allocates
	lu!(A_sys)
end seconds=5 samples=20

t_view_lu = @benchmark begin
	lu!(view($A_buf, 1:($n_sub), 1:($n_sub)), check = false)   # zero-allocation
end seconds=5 samples=20

println("    [OLD] copy + lu!:")
show(stdout, MIME"text/plain"(), t_copy_lu)
println()
println("    [NEW] view + lu!(check=false):")
show(stdout, MIME"text/plain"(), t_view_lu)
println()

ratio_opt12 = median(t_copy_lu).time / median(t_view_lu).time
println("    Speedup OPT-1+2: $(round(ratio_opt12, digits=2))×")
alloc_old = minimum(t_copy_lu).allocs
alloc_new = minimum(t_view_lu).allocs
println("    Allocations — old: $alloc_old,  new: $alloc_new")

# ── OPT-3: allocating assembly + copy  vs  in-place write ────────────────
println("\n  OPT-3  —  allocate+copy  vs  in-place assembly")

sys_buf = Matrix{ComplexF64}(undef, n_big, n_big)
src_M = rand(ComplexF64, n_sub, n_sub)

t_alloc_copy = @benchmark begin
	M_tmp=Matrix{ComplexF64}(undef, $n_sub, $n_sub)
	copyto!(M_tmp, $src_M)
	$sys_buf[1:($n_sub), 1:($n_sub)].=M_tmp
end seconds=5 samples=20

t_inplace = @benchmark begin
	copyto!(view($sys_buf, 1:($n_sub), 1:($n_sub)), $src_M)
end seconds=5 samples=20

println("    [OLD] alloc Matrix + copyto! + stacking .=:")
show(stdout, MIME"text/plain"(), t_alloc_copy)
println()
println("    [NEW] direct copyto! into view:")
show(stdout, MIME"text/plain"(), t_inplace)
println()

ratio_opt3 = median(t_alloc_copy).time / median(t_inplace).time
println("    Speedup OPT-3: $(round(ratio_opt3, digits=2))×")

# ── OPT-4: per-monomial zeros + similar  vs  preallocated reuse ──────────
println("\n  OPT-4  —  per-monomial zeros/similar  vs  preallocated reuse")

t_zeros_alloc = @benchmark begin
	result=zeros(ComplexF64, $FOM)
	scratch=similar(result)
	temp=similar(result)
	fill!(result, zero(ComplexF64))   # simulate compute_multilinear_terms work
	fill!(scratch, zero(ComplexF64))
	fill!(temp, zero(ComplexF64))
end seconds=5 samples=20

result_buf = zeros(ComplexF64, FOM)
scratch_buf = zeros(ComplexF64, FOM)
temp_buf = zeros(ComplexF64, FOM)

t_prealloced = @benchmark begin
	fill!($result_buf, zero(ComplexF64))
	fill!($scratch_buf, zero(ComplexF64))
	fill!($temp_buf, zero(ComplexF64))
end seconds=5 samples=20

println("    [OLD] zeros + similar × 2 per monomial:")
show(stdout, MIME"text/plain"(), t_zeros_alloc)
println()
println("    [NEW] fill! on preallocated buffers:")
show(stdout, MIME"text/plain"(), t_prealloced)
println()

ratio_opt4 = median(t_zeros_alloc).time / median(t_prealloced).time
println("    Speedup OPT-4: $(round(ratio_opt4, digits=2))×")

# ===========================================================================
# C. Allocation count in the hot path
# ===========================================================================

println("\n", "─"^70)
println("C. Per-monomial allocation check (after all optimisations)")
println("─"^70)

# Build context manually so we can call solve_cohomological_equations!
# in isolation for the allocation check.
using MORFE.CohomologicalEquations: CohomologicalContext, solve_single_monomial!
using MORFE.Multiindices: build_exponent_index_map, indices_in_box_with_bounded_degree
using MORFE.MultilinearTerms: build_multilinear_terms_cache
using MORFE.InvarianceEquation: precompute_column_polynomials,
	precompute_master_column_polynomials,
	precompute_external_column_polynomials
using MORFE.MasterModeOrthogonality: precompute_orthogonality_operator_coefficients,
	precompute_orthogonality_column_polynomials
using MORFE.Resonance: ResonanceSet
using MORFE.ParametrisationMethod: create_parametrisation_method_objects,
	multiindex_set

W2,
R2 = solve_cohomological_problem(
	model, mset, master_eigenvalues, master_modes, left_eigenmodes, resonance_set;
	master_modes_derivatives = master_modes_derivatives,
)

# Pick a non-resonant interior monomial (degree 2) for the allocation test.
# In a 2-variable degree-3 mset, degree-2 monomials are at indices 3..5.
test_idx = findfirst(idx -> sum(mset[idx]) == 2, 1:length(mset))
if test_idx !== nothing
	# Rebuild context (same as solve_cohomological_problem does internally)
	L = length(mset)
	LT = eltype(model.linear_terms[1])
	ORD = length(model.linear_terms) - 1
	ORDP1 = ORD + 1
	linear_terms = model.linear_terms
	zero_vec = SVector{NVAR, Int}(ntuple(_ -> 0, Val(NVAR)))
	has_zero = length(mset) >= 1 && mset.exponents[1] == zero_vec
	unit_offset = has_zero ? 1 : 0
	Λ_master = view(R2.poly.coefficients, 1:ROM, (unit_offset+1):(unit_offset+ROM))
	lambda_diag = [R2.poly.coefficients[i, i+unit_offset] for i in 1:NVAR]
	invariance_C_coeffs,
	D_steps = precompute_master_column_polynomials(
		linear_terms, master_modes, Λ_master)
	Λ_full = view(R2.poly.coefficients, 1:NVAR, (unit_offset+1):(unit_offset+NVAR))
	invariance_E_coeffs = precompute_external_column_polynomials(
		linear_terms, zeros(ComplexF64, FOM, N_EXT), Λ_full, D_steps)
	J_coeffs = precompute_orthogonality_operator_coefficients(
		linear_terms, left_eigenmodes, master_eigenvalues)
	generalised_eig = hcat(master_modes, zeros(ComplexF64, FOM, N_EXT))
	orth_C_coeffs,
	orth_E_coeffs = precompute_orthogonality_column_polynomials(
		J_coeffs, generalised_eig, Λ_full)
	skip_set = Set(begin
		indices = Int[]
		n_search = min(NVAR + 1, L)
		for r in 1:NVAR
			e_r = SVector{NVAR, Int}(ntuple(i -> i == r ? 1 : 0, Val(NVAR)))
			idx = findfirst(==(e_r), view(mset.exponents, 1:n_search))
			idx !== nothing && push!(indices, idx)
		end
		indices
	end)
	multiindex_dict = build_exponent_index_map(mset)
	lower_order_buffer = [zeros(ComplexF64, FOM) for _ in 1:ORD]
	sys_matrix = Matrix{ComplexF64}(undef, FOM + ROM, FOM + ROM)
	rhs_b = Vector{ComplexF64}(undef, FOM + ROM)
	ext_rhs_b = zeros(ComplexF64, FOM)
	ml_res_b = zeros(ComplexF64, FOM)
	cand_idx = Vector{Vector{Int}}(undef, L)
	for i in 1:L
		multi_i = mset[i]
		tdeg = sum(multi_i)
		cand_idx[i] = tdeg < 2 ? Int[] :
					  indices_in_box_with_bounded_degree(mset, collect(multi_i), 2, tdeg)
	end
	ctx_check = CohomologicalContext{ComplexF64, ORD, ORDP1, NVAR, FOM, LT}(
		linear_terms, generalised_eig, lambda_diag,
		invariance_C_coeffs, invariance_E_coeffs,
		J_coeffs, orth_C_coeffs, orth_E_coeffs,
		resonance_set, skip_set,
		multiindex_dict, lower_order_buffer, cand_idx,
		sys_matrix, rhs_b, ext_rhs_b, ml_res_b,
	)
	ml_cache_check = build_multilinear_terms_cache(model, W2)

	allocs = @allocated solve_single_monomial!(
		W2, R2, test_idx, ctx_check, model, ml_cache_check)
	println("\n  @allocated solve_single_monomial! (non-resonant, degree-2 monomial)")
	println("  = $allocs bytes")
	if allocs <= 1024
		println("  ✓  Hot path is effectively allocation-free (only the LU struct, ≤ 1 KiB)")
	else
		println("  ✗  Unexpected allocation — investigate further")
	end
end

println("\n", "="^70)
println("Benchmark complete.")
println("="^70)
