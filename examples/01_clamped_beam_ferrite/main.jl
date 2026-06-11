"""
Structural mechanical problem (clamped-clamped beam) using Ferrite.jl as FEM backend.

Demonstrates the MORFE.jl pipeline with the RHS-C batched FEM path:
  1. Assemble K, M via Ferrite element loops.
  2. Wrap geometric nonlinearity as FerriteGeometricNonlinearity (FEMMultilinearMap).
  3. Build NDOrderModel and solve cohomological equations.
  4. Realify reduced dynamics.

Material: St. Venant-Kirchhoff (nonlinearity up to cubic order in u).
Geometry: clamped-clamped beam, 1000 × 10 × 24, 40×3×1 Hex8 elements, quadratic order-2 Lagrange.
Matches the Gridap demo (FOM=4977) — use this to benchmark the RHS-C FEM batched path.
"""

include(joinpath(@__DIR__, "..", "common", "results_io.jl"))

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "MORFE")
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
	Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays"])
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

# -----------------------------------------------------------------------
# 1. Mesh and FE setup
# -----------------------------------------------------------------------

# Load the same GMSH mesh used by the Gridap demo (40×3×1 Hex8, L=1000×b=10×h=24).
# FerriteGmsh transfers the "Dirichlet" physical group directly as a facetset.
msh_path = joinpath(@__DIR__, "..", "02_clamped_beam_gridap", "clamped_clamped_beam.msh")
grid = togrid(msh_path)

ip = Lagrange{RefHexahedron, 2}()^3      # quadratic, 27-node Lagrange hex (= Gridap order 2)
qr = QuadratureRule{RefHexahedron}(3)    # 3 pts/dir → 27 QPs, integrates degree 5 (≥ Gridap degree=4)
cv = CellValues(qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)

println("Total DOFs : ", ndofs(dh))

# -----------------------------------------------------------------------
# 2. Linear matrices
# -----------------------------------------------------------------------

E = 160e3
ν = 0.22
ρ = 2.32e-3
λ = (E * ν) / ((1 + ν) * (1 - 2ν))
μ = E / (2(1 + ν))
α = 0.5370828278264171 / 100.0      # Rayleigh mass coefficient
β = 1.0 / (0.5370828278264171 * 100.0)  # Rayleigh stiffness coefficient

K_full = allocate_matrix(dh)
M_full = allocate_matrix(dh)

ferrite_assemble_KM!(K_full, M_full, dh, cv, λ, μ, ρ)

# Free DOFs = all DOFs minus the prescribed (Dirichlet) ones.
free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free = length(free)

K = K_full[free, free]
M = M_full[free, free]
C = α * M + β * K

println("Free DOFs  : ", n_free)

# -----------------------------------------------------------------------
# 3. Multiindex set (needed to size FEM buffers correctly)
# -----------------------------------------------------------------------

ROM = 2
N_EXT = 0
NVAR = ROM + N_EXT
max_degree = 9
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
_max_uniq = length(mset)

# -----------------------------------------------------------------------
# 4. Nonlinear terms (FEMMultilinearMap — RHS-C batched path)
# -----------------------------------------------------------------------

term_quad  = ferrite_nonlinearity(2, dh, cv, free_to_local, n_free, λ, μ; max_unique_cols = _max_uniq)
term_cubic = ferrite_nonlinearity(3, dh, cv, free_to_local, n_free, λ, μ; max_unique_cols = _max_uniq)

# -----------------------------------------------------------------------
# 4. NDOrderModel:  M ü + C u̇ + K u = F(u)
#    linear_terms = (K, C, M) = (B0, B1, B2)
# -----------------------------------------------------------------------

model = NDOrderModel(
	(K, C, M),
	(term_quad, term_cubic),
)

# -----------------------------------------------------------------------
# 5. Eigenproblem
# -----------------------------------------------------------------------

"""
	Mechanical_Problem_Solver <: AbstractEigensolver

Solves the undamped eigenproblem K ϕ = ω² M ϕ and recovers the damped
eigenvalues λ = ω(-ξ ± i√(1-ξ²)) using Rayleigh damping ξ = ½(α/ω + βω).
"""
mutable struct Mechanical_Problem_Solver <: AbstractEigensolver
	right_eig_result::Union{Nothing, Matrix}
	eigenvalues::Union{Nothing, Vector}
	nev::Int
	α::Float64
	β::Float64
end

function MORFE.Eigenproblems.solve(model::NDOrderModel, solver::Mechanical_Problem_Solver)
	ω2,
	ϕ = eigs(model.linear_terms[1], model.linear_terms[3];
		nev = solver.nev, which = :SM)
	idx = sortperm(real(ω2))[1:solver.nev]
	ω2 = real.(ω2[idx])
	ϕ = real.(ϕ[:, idx])
	ω = sqrt.(ω2)
	FOM = size(ϕ, 1)

	λ_all = zeros(ComplexF64, 2 * solver.nev)
	for i in 1:solver.nev
		ξ = 0.5 * (solver.α / ω[i] + solver.β * ω[i])
		λ_all[2i-1] = ω[i] * (-ξ + sqrt(Complex(1.0 - ξ^2)) * im)
		λ_all[2i] = ω[i] * (-ξ - sqrt(Complex(1.0 - ξ^2)) * im)
	end

	evecs = Matrix{ComplexF64}(undef, 2FOM, 2 * solver.nev)
	for i in 1:solver.nev
		evecs[1:FOM, 2i-1] .= ϕ[:, i]
		evecs[1:FOM, 2i] .= ϕ[:, i]
		evecs[(FOM+1):end, 2i-1] .= λ_all[2i-1] .* ϕ[:, i]
		evecs[(FOM+1):end, 2i] .= λ_all[2i] .* ϕ[:, i]
	end
	solver.right_eig_result = evecs
	solver.eigenvalues = λ_all
	return λ_all, reshape(evecs, FOM, 2, 2 * solver.nev)
end

function MORFE.Eigenproblems.solve_left(model::NDOrderModel, solver::Mechanical_Problem_Solver)
	@assert solver.right_eig_result !== nothing "Run solve() first"
	R = solver.right_eig_result
	FOM = size(R, 1) ÷ 2
	L = similar(R)

	for i in 1:(length(solver.eigenvalues)÷2)
		L[(FOM+1):end, 2i-1] = R[1:FOM, 2i]
		L[(FOM+1):end, 2i] = R[1:FOM, 2i-1]
	end
	for (i, λ) in enumerate(solver.eigenvalues)
		L[1:FOM, i] = -(1 / conj(λ)) * model.linear_terms[1]' * L[(FOM+1):end, i]
	end
	return solver.eigenvalues, reshape(L, FOM, 2, size(L, 2))
end

eigenproblem = solve_eigenproblem(
	model,
	solver = Mechanical_Problem_Solver(nothing, nothing, 10, α, β),
	sorter! = (args...) -> nothing,
)
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)

println("\nFirst eigenvalues:")
for (i, λ) in enumerate(eigenvalues[1:min(6, end)])
	println("  mode $i: λ = $λ")
end

# -----------------------------------------------------------------------
# 6. Master modes and reduced-variable structure
# -----------------------------------------------------------------------

FOM = n_free

select_master_modes_by_sorting(eigenproblem, ROM)

master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes = Y[:, 1, 1:ROM]
left_eigenmodes = Y[:, 2, 1:ROM]

ORD_model = length(model.linear_terms) - 1   # = 2
master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
for r in 1:ROM
	for k in 1:(ORD_model-1)
		master_modes_derivatives[:, k, r] .= Y[:, k+1, r]
	end
end

# -----------------------------------------------------------------------
# 7. Multiindex set and resonance set
# -----------------------------------------------------------------------

println("\nMultiindex set: degree ≤ $max_degree in $NVAR variables → $(length(mset)) monomials")

resonance_set = resonance_set_from_complex_normal_form_style(
	mset, Vector{ComplexF64}(master_eigenvalues), 0.05)

#println("\nResonance set:")
#for (idx, mi) in enumerate(mset.exponents)
#	res_str = join(findall(resonance_set.resonances[:, idx]), ", ")
#	isempty(res_str) && (res_str = "none")
#	println("  $mi → [$res_str]")
#end

# -----------------------------------------------------------------------
# 8. Solve cohomological equations
# -----------------------------------------------------------------------

print("\nCohomological solve: ")
_t_solve = @elapsed W, R = solve_cohomological_problem(
	model, mset,
	master_eigenvalues,
	master_modes, left_eigenmodes,
	resonance_set;
	master_modes_derivatives = master_modes_derivatives,
	conjugate_permutation = [2, 1],
)

# -----------------------------------------------------------------------
# 9. Realify reduced dynamics
# -----------------------------------------------------------------------

conj_map = zeros(Int, NVAR)
for i in 1:NVAR
	conj_map[i] = isodd(i) ? i + 1 : i - 1
end
Rr = ReducedDynamics(realify(R.poly, conj_map), R.external_system_size)

println("\nReduced dynamics coefficients:")
for m in 1:length(Rr.poly.multiindex_set.exponents)
	mi = Rr.poly.multiindex_set.exponents[m]
	c = Rr.poly.coefficients[:, m]
	any(abs.(c) .> 1e-12) && println("  $mi : $(real.(c))")
end

# -----------------------------------------------------------------------
# 10. Save results
# -----------------------------------------------------------------------

dirs = results_dirs(@__DIR__)
save_rom(dirs, W, R)
write_summary(dirs, [
	"example: 01_clamped_beam_ferrite",
	"model: clamped-clamped beam, St. Venant-Kirchhoff, Ferrite backend",
	"mesh: 40x3x1 Hex8 elements, quadratic Lagrange (order 2)",
	"n_dofs: $n_free",
	"master_modes: $ROM",
	"master_eigenvalues: $(collect(master_eigenvalues))",
	"parametrisation_order: $max_degree",
	"n_monomials: $(length(mset))",
	"cohomological_solve_time_s: $(_t_solve)",
])
println("\nResults written to $(dirs.base)")
