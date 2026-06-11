"""
Structural mechanical problem without forcing using Gridap.jl as FEM

This script demonstrates the formulation and discretization of a structural mechanics 
problem using finite element methods, followed by conversion to a first-order system.
"""

import Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "MORFE")
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
    Pkg.add(["Gridap", "GridapGmsh", "WriteVTK", "Gmsh", "Arpack", "LinearMaps",
        "StaticArrays"])
end
Pkg.instantiate()

using MORFE

using Gridap
using GridapGmsh
using WriteVTK
using Gmsh
using SparseArrays
using Arpack
using LinearMaps
using LinearAlgebra
using StaticArrays

# print test results to confirm correct relations
test = false

# ------------------------------------------------------------------------------
# 1. Define NDOrderModel
# Structural mechanical problem without forcing.
# First build second order ODE:
# M*d_t² U + C*d_tU + K*U = F(U)
# ------------------------------------------------------------------------------

# Load GMSH-Mesh
gmsh.initialize()
# gmsh.option.setNumber("General.Verbosity", 0)
model = GmshDiscreteModel(joinpath(@__DIR__, "clamped_clamped_beam.msh"))

# Define FEM Space
order = 2
degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Γ = BoundaryTriangulation(model, tags = "Neumann")
Γ_D = BoundaryTriangulation(model, tags = "Dirichlet")
dΓ = Measure(Γ, degree)
reffe = ReferenceFE(lagrangian, VectorValue{3, Float64}, order)
V = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = ["Dirichlet"],
	dirichlet_masks = [(true, true, true)])
g(x) = VectorValue(0.0, 0.0, 0.0)
U = TrialFESpace(V, g)

# Material properties and constitutive relations
sym(u) = 1 / 2 * (u + u')
E = 160e3
ν = 0.22
ρ = 2.32e-3
λ = (E * ν) / ((1 + ν) * (1 - 2 * ν))
μ = E / (2 * (1 + ν))
σ(ε) = λ * tr(ε) * one(ε) + 2 * μ * ε

#Linear forms
a(u, v) = ∫(ε(v) ⊙ (σ ∘ ε(u)))dΩ
m(dt2u, v) = ∫(ρ * dt2u ⋅ v)dΩ
E_nl(u1, u2) = 0.25 * ((∇(u1)') ⋅ ∇(u2) + (∇(u2)') ⋅ ∇(u1))
E_nl_grad(∇u1, ∇u2) = 0.25 * ((∇u1') ⋅ ∇u2 + (∇u2') ⋅ ∇u1)
σ_nln(ε) = λ * tr(ε) * one(TensorValue{3, 3, Float64}) + 2 * μ * ε

#Quadratic nonlinear terms
function g_quad(u1, u2, v)
	∫(ε(v) ⊙ (σ_nln(E_nl(u1, u2))) +
	  0.5 * (sym(∇(u1)' ⋅ ∇(v)) ⊙ σ_nln(ε(u2))
			 +
			 sym(∇(u2)' ⋅ ∇(v)) ⊙ σ_nln(ε(u1))))dΩ
end

#Cubic nonlinear terms
function h_cube(u1, u2, u3, v)
	1 / 3 *
	∫(sym(∇(u1)' ⋅ ∇(v)) ⊙ (σ_nln(E_nl(u2, u3))) +
	  sym(∇(u2)' ⋅ ∇(v)) ⊙ (σ_nln(E_nl(u1, u3))) +
	  sym(∇(u3)' ⋅ ∇(v)) ⊙ (σ_nln(E_nl(u1, u2))))dΩ
end

# Assemble matrices of second order system
stiffness_matrix = assemble_matrix((u, v) -> a(u, v), U, V)
mass_matrix = assemble_matrix((u, v) -> m(u, v), U, V)
α = 0.5370828278264171 / (100.0)
β = 1.0 / (0.5370828278264171 * 100.0)
damping_matrix = α * mass_matrix + β * stiffness_matrix

FOM = size(stiffness_matrix, 1)
println("FOM:", FOM)

# Define nonlinear terms for FullOrderModel of First Order
# Note: Current implementation needs fe space to be available in scope and is not
# efficiently implemented for complex values
function quadratic_nonlinearity!(res, vec1, vec2)
	u1r = FEFunction(V, real(vec1))
	u1i = FEFunction(V, imag(vec1))
	u2r = FEFunction(V, real(vec2))
	u2i = FEFunction(V, imag(vec2))
	res .-= assemble_vector(v -> (g_quad(u1r, u2r, v) - g_quad(u1i, u2i, v)), V)
	res .-= assemble_vector(v -> (g_quad(u1r, u2i, v) + g_quad(u1i, u2r, v)), V) * im
end
term_quad = MultilinearMap(quadratic_nonlinearity!, (2, 0))
function cubic_nonlinearity!(res, vec1, vec2, vec3)
	u1r = FEFunction(V, real(vec1))
	u2r = FEFunction(V, real(vec2))
	u3r = FEFunction(V, real(vec3))
	u1i = FEFunction(V, imag(vec1))
	u2i = FEFunction(V, imag(vec2))
	u3i = FEFunction(V, imag(vec3))
	res .-= assemble_vector(
		v -> (h_cube(u1r, u2r, u3r, v) - h_cube(u1r, u2i, u3i, v) -
			  h_cube(u1i, u2r, u3i, v) - h_cube(u1i, u2i, u3r, v)), V)
	res .-= assemble_vector(
		v -> (h_cube(u1i, u2r, u3r, v) + h_cube(u1r, u2i, u3r, v) +
			  h_cube(u1r, u2r, u3i, v) - h_cube(u1i, u2i, u3i, v)), V)
end
term_cubic = MultilinearMap(cubic_nonlinearity!, (3, 0))

model = NDOrderModel(
	(stiffness_matrix, damping_matrix, mass_matrix),# B0, B1, B2
	(term_quad, term_cubic),
)

# ------------------------------------------------------------------------------
# 2. Eigenproblem
# ------------------------------------------------------------------------------

"""
	Mechanical_Problem_Solver <: AbstractEigensolver

solves eigenproblem of the mechanical 2nd order problem:
	M*d_t² U + C*d_tU + K*U + n(U) = 0
with C = α*M + β*K under the assumption that M and K are spd matrices.
 
1) calculate eigenpairs of: (K-ωₖ^2)ϕₖ=0
2) calculate: ξₖ=0.5(α/ωₖ +β*ωₖ)
3) calculate eigenvalues: λₖ=-ξₖ*ωₖ ± i*ωₖ√(1-ξₖ^2) 

Attributes:
- right_eig_result: saves right eigenvectors in solve() to use in solve_left
- eigenvalues: saves eigenvalues in solve() to use in solve_left
- nev: number of eigenvalues/vectors to compute
- α: damping coefficient
- β: damping coefficient

"""
mutable struct Mechanical_Problem_Solver <: AbstractEigensolver
	right_eig_result::Union{Nothing, Matrix}
	eigenvalues::Union{Nothing, Vector}
	nev::Int64
	α::Float64
	β::Float64
end
function MORFE.Eigenproblems.solve(model::NDOrderModel, solver::Mechanical_Problem_Solver)
	ω2,
	ϕ = eigs(
		model.linear_terms[1], model.linear_terms[3], nev = solver.nev, which = :SM)
	# @time ω2, ϕ = eigen(Matrix(model.linear_terms[1]), Matrix(model.linear_terms[3]))
	idx = sortperm(real(ω2))[1:(solver.nev)]
	ω2 = ω2[idx]
	ϕ = ϕ[:, idx]
	FOM = length(ϕ[:, 1])
	ω = sqrt.(real(ω2))
	ϕ = real(ϕ)
	λ = zeros(ComplexF64, solver.nev * 2)
	for i in 1:(solver.nev)
		ξ = 0.5 * (solver.α / ω[i] + solver.β * ω[i])
		λ[2*i-1] = ω[i] * (-ξ + sqrt(Complex(1.0 - ξ^2)) * im)
		λ[2*i] = ω[i] * (-ξ - sqrt(Complex(1.0 - ξ^2)) * im)
	end
	eigenvectors = Matrix{ComplexF64}(undef, FOM * 2, solver.nev * 2)
	for i in 1:(solver.nev)
		eigenvectors[1:FOM, (i*2-1)] .= ϕ[:, i]
		eigenvectors[1:FOM, (i*2)] .= ϕ[:, i]
		#velocity part
		eigenvectors[(FOM+1):end, (i*2-1)] .= λ[(i*2-1)] * ϕ[:, i]
		eigenvectors[(FOM+1):end, (i*2)] .= λ[(i*2)] * ϕ[:, i]
	end
	# store results in solver for left eigen_vectors
	solver.right_eig_result = eigenvectors
	solver.eigenvalues = λ
	return λ, eigenvectors
end

function MORFE.Eigenproblems.solve_left(
	model::NDOrderModel, solver::Mechanical_Problem_Solver)
	@assert solver.right_eig_result!==nothing "First run solve()"
	left_eigenvectors = similar(solver.right_eig_result)
	FOM = Int(0.5 * length(left_eigenvectors[:, 1]))
	for i in 1:Int(0.5*size(left_eigenvectors, 2))
		left_eigenvectors[(FOM+1):end, (i*2-1)] = solver.right_eig_result[
			1:FOM, (i*2)]
		left_eigenvectors[(FOM+1):end, (i*2)] = solver.right_eig_result[
			1:FOM, (i*2-1)]
	end
	for (i, λ) in enumerate(solver.eigenvalues)
		# left_eigenvectors[(FOM + 1):end, i] = solver.right_eig_result[1:FOM, i]
		left_eigenvectors[1:FOM, i] = -(1 / conj(λ)) * model.linear_terms[1]' *
									  left_eigenvectors[(FOM+1):end, i]
	end
	return solver.eigenvalues, left_eigenvectors
end

# Compute left and right eigenpairs using the default solver and store it in Eigenproblem
eigenproblem = solve_eigenproblem(
	model, solver = Mechanical_Problem_Solver(nothing, nothing, 10, α, β),
	sorter! = (args...) -> nothing)
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)
for (i, λ) in enumerate(eigenvalues)
	println("  mode $i →   λ = $λ\n")
end

# test residuual of left and right eigenvectors
if test == true
	A, B = linear_first_order_matrices(model)
	FOM2 = Int(size(Y, 1) / 2)
	for (i, λ) in enumerate(eigenvalues)
		res_y = norm(A * Y[:, i] - λ * B * Y[:, i])
		res_x = norm(A' * X[:, i] - conj(λ) * B' * X[:, i])
		println("mode $i:", res_x, " ", res_y)
	end
end
# ------------------------------------------------------------------------------
# 3. Select master modes and build the reduced-variable structure
# ------------------------------------------------------------------------------
ROM = 2             # number of master (dominant) modes
N_EXT = 0           # number of external forcing modes (for future use)
NVAR = ROM + N_EXT

# mark eigenmodes in eigenproblem
# TODO !Usage of eigenproblems is not yet implemented in parametrisation!
select_master_modes_by_sorting(eigenproblem, ROM)

master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes = Y[1:FOM, 1:ROM]            # size: FOM × ROM
left_eigenmodes = Y[(FOM+1):end, 1:ROM]  # size: FOM × ROM

# Higher-order master mode derivatives W^(k)[e_r], k = 2 … ORD.
# For the companion-form eigenproblem with state ẑ = [x; ẋ; …], the k-th
# block of ẑ (rows (k-1)*FOM+1 : k*FOM) gives W^(k)[e_r] directly.
ORD_model = length(model.linear_terms) - 1   # = 2 for this second-order system
master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
for r in 1:ROM
	for k in 1:(ORD_model-1)   # k = 1 only for ORD = 2
		master_modes_derivatives[:, k, r] .= Y[(k*FOM+1):((k+1)*FOM), r]
	end
end

# ------------------------------------------------------------------------------
# 4. Build multiindex set and resonance set
# ------------------------------------------------------------------------------

outer_eigenvalues = eigenvalues[(ROM+1):end] # ROM...2*10
# no external system — NVAR = ROM
max_degree = 3
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
println("\nMultiindex set: degree ≤ $max_degree in $NVAR variables → $(length(mset)) monomials")

resonance_set = resonance_set_from_complex_normal_form_style(
	mset, Vector{ComplexF64}(master_eigenvalues), 0.05)
# resonance_set = resonance_set_from_graph_style(
#     mset, Vector{ComplexF64}(master_eigenvalues), ComplexF64[], outer_eigenvalues, 0.05
# )

println("\nResonance set:")
for (idx, mi) in enumerate(mset.exponents)
	res_str = join(findall(resonant_targets(resonance_set, idx)), ", ")
	isempty(res_str) && (res_str = "none")
	println("  $mi → [$res_str]")
end

# ------------------------------------------------------------------------------
# 5. Solve cohomological equations
#    External eigenvalues are read from model.external_system automatically.
# ------------------------------------------------------------------------------
@time W,
R = solve_cohomological_problem(
	model, mset,
	master_eigenvalues,
	master_modes, left_eigenmodes,
	resonance_set;
	master_modes_derivatives = master_modes_derivatives,
)

# ------------------------------------------------------------------------------
# 6. Realify
# ------------------------------------------------------------------------------
conj_map = zeros(Int64, NVAR)
for i in 1:NVAR
	if i % 2 == 1
		conj_map[i] = i + 1
	else
		conj_map[i] = i - 1
	end
end
Rr = ReducedDynamics(realify(R.poly, conj_map), R.external_system_size)

# ------------------------------------------------------------------------------
# 7. Write real dynamics to compare to Morfe 2.0
# ------------------------------------------------------------------------------
function write_rdyn(R::ReducedDynamics{ROM, NVAR, T}) where {ROM, NVAR, T}
	rdyn = ["" for i in 1:NVAR]
	for i in 1:NVAR
		rdyn[i] = "a" * string(i) * "' = "
	end
	coeff = R.poly.coefficients
	mset = R.poly.multiindex_set

	for m in 1:length(mset.exponents)
		multiindex = mset.exponents[m]
		monomial = ""
		for d in 1:NVAR
			if (multiindex[d] != 0)
				monomial *= "*a" * string(d) * "^" * string(multiindex[d])
			end
		end
		for d in 1:Int(NVAR*0.5)
			rcoeff = real(coeff[2*d-1, m])
			icoeff = -1 * imag(coeff[2*d-1, m])
			if abs(rcoeff) > 1e-20
				rdyn[2*d-1] *= " + " * string(rcoeff) * monomial
			end
			if abs(icoeff) > 1e-20
				rdyn[2*d] *= " + " * string(icoeff) * monomial
			end
		end
	end

	ofile = open("./equations.txt", "w")
	for i in 1:NVAR
		write(ofile, rdyn[i] * ";\n")
	end
	close(ofile)
end
write_rdyn(Rr)
