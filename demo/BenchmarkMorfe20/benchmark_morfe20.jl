"""
Implementation of the MORFE module to reproduce and compare the system from Morfe2.0
Different Morfe2.0 examples can be used by exchanging 'info_file' and 'mesh_file'.
"""

using MORFE
include("Morfe_2_0/Morfe_2_0.jl")
using .Morfe_2_0

using LinearAlgebra
using SparseArrays
using StaticArrays
using Arpack
using Profile
# using ProfileView

#Make info
info = Infostruct()
info_file = "beam_damp.jl"
include(info_file)

#Import mesh
mesh_file = "./demo/BenchmarkMorfe20/beam.mphtxt"
mesh = read_mesh(mesh_file, domains_list, materials_list, materials_dict,
    boundaries_list, constrained_dof, bc_vals)

# initialise a dummy field to store dofs ordering and static solutions
U = Field(mesh, Morfe_2_0.dim)

info.nm = length(info.Φ)   # master modes
info.nz = 2 * info.nm
info.nzforce = 2  # imposes only two nonautonomous
if info.Ffreq == 0
    info.nzforce = 0
end
info.nrom = info.nz + info.nzforce
info.nK = U.neq   # dim of FEM problem
info.nA = 2 * info.nK  # dim of first order sys
info.nMat = info.nA + info.nz  # dim of system to be solved

colptr, rowval = Morfe_2_0.assembler_dummy_MK(mesh, U)
val = zeros(Float64, length(rowval))
K = SparseMatrixCSC(U.neq, U.neq, colptr, rowval, val)
M = deepcopy(K)
Morfe_2_0.assembler_MK!(mesh, U, K, M)
C = info.α * M + info.β * K

# ------------------------------------------------------------------------------
# 1. Define NDOrderModel
# Structural mechanical problem without forcing.
# First build second order ODE:
# M*d_t² U + C*d_tU + K*U = F(U)
# ------------------------------------------------------------------------------

function quadratic!(res, Ψ₁, Ψ₂)
    Morfe_2_0.assembly_G!(res, Ψ₁, Ψ₂, mesh, U)
end
quadratic_term = MultilinearMap(quadratic!, (2, 0))
function cubic!(res, Ψ₁, Ψ₂, Ψ₃)
    Morfe_2_0.assembly_H!(res, Ψ₁, Ψ₂, Ψ₃, mesh, U)
end
cubic_term = MultilinearMap(cubic!, (3, 0))
model = NDOrderModel((K, C, M), (quadratic_term, cubic_term))

A, B = FullOrderModel.linear_first_order_matrices(model)
FOM = size(K, 1)
println("FOM:", FOM)

# ------------------------------------------------------------------------------
# 2. EigenProblem
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
function mass_normalization!(ϕ, M, neig)
    #
    for i in 1:neig
        c = transpose(ϕ[:, i]) * M * ϕ[:, i]
        for j in 1:(M.m)
            ϕ[j, i] /= sqrt(c)
        end
    end
    #
    return nothing
end
function MORFE.Eigenproblems.solve(model::NDOrderModel, solver::Mechanical_Problem_Solver)
    @time ω2,
    ϕ = eigs(
        model.linear_terms[1], model.linear_terms[3], nev = solver.nev,
        which = :LM, sigma = 1e-6, maxiter = 500, ncv = 20)
    # @time ω2, ϕ = eigen(Matrix(model.linear_terms[1]), Matrix(model.linear_terms[3]))
    # idx = sortperm(real(ω2))[1:(solver.nev)]
    # ω2 = ω2[idx]
    # ϕ = ϕ[:, idx]
    mass_normalization!(ϕ, model.linear_terms[3], solver.nev)
    FOM = length(ϕ[:, 1])
    ω = sqrt.(real(ω2))
    ϕ = real(ϕ)
    λ = zeros(ComplexF64, solver.nev * 2)
    for i in 1:(solver.nev)
        ξ = 0.5 * (solver.α / ω[i] + solver.β * ω[i])
        λ[2 * i - 1] = ω[i] * (-ξ + sqrt(Complex(1.0 - ξ^2)) * im)
        λ[2 * i] = ω[i] * (-ξ - sqrt(Complex(1.0 - ξ^2)) * im)
    end
    eigenvectors = Matrix{ComplexF64}(undef, FOM * 2, solver.nev * 2)
    for i in 1:(solver.nev)
        eigenvectors[1:FOM, (i * 2 - 1)] .= ϕ[:, i]
        eigenvectors[1:FOM, (i * 2)] .= ϕ[:, i]
        #velocity part
        eigenvectors[(FOM + 1):end, (i * 2 - 1)] .= λ[(i * 2 - 1)] * ϕ[:, i]
        eigenvectors[(FOM + 1):end, (i * 2)] .= λ[(i * 2)] * ϕ[:, i]
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
    for i in 1:Int(0.5 * size(left_eigenvectors, 2))
        left_eigenvectors[(FOM + 1):end, (i * 2 - 1)] = solver.right_eig_result[
        1:FOM, (i * 2)]
        left_eigenvectors[(FOM + 1):end, (i * 2)] = solver.right_eig_result[
        1:FOM, (i * 2 - 1)]
    end
    for (i, λ) in enumerate(solver.eigenvalues)
        # left_eigenvectors[(FOM + 1):end, i] = solver.right_eig_result[1:FOM, i]
        left_eigenvectors[1:FOM, i] = -(1 / conj(λ)) * model.linear_terms[1]' *
                                      left_eigenvectors[(FOM + 1):end, i]
    end
    return solver.eigenvalues, left_eigenvectors
end

# Compute left and right eigenpairs using the default solver and store it in EigenProblem
eigenproblem = compute_eigenproblem(
    model, solver = Mechanical_Problem_Solver(nothing, nothing, 10, info.α, info.β),
    sorter! = (args...) -> nothing,
    normalizer! = (args...) -> nothing)
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)
for (i, λ) in enumerate(eigenvalues)
    println("  mode $i →   λ = $λ\n")
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
left_eigenmodes = Y[(FOM + 1):end, 1:ROM]  # size: FOM × ROM

# Higher-order master mode derivatives W^(k)[e_r], k = 2 … ORD.
# For the companion-form eigenproblem with state ẑ = [x; ẋ; …], the k-th
# block of ẑ (rows (k-1)*FOM+1 : k*FOM) gives W^(k)[e_r] directly.
ORD_model = length(model.linear_terms) - 1   # = 2 for this second-order system
master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
for r in 1:ROM
    for k in 1:(ORD_model - 1)   # k = 1 only for ORD = 2
        master_modes_derivatives[:, k, r] .= Y[(k * FOM + 1):((k + 1) * FOM), r]
    end
end

# ------------------------------------------------------------------------------
# 4. Build multiindex set and resonance set
# ------------------------------------------------------------------------------

outer_eigenvalues = eigenvalues[(ROM + 1):end] # 2*10-FOM
# no external system 
super_eigenvalues = Vector{ComplexF64}(master_eigenvalues)
target_eigenvalues = Vector{ComplexF64}(master_eigenvalues)

max_degree = 3
mset = all_multiindices_up_to(NVAR, max_degree; min_degree = 1)
println("\nMultiindex set: degree ≤ $max_degree in $NVAR variables → $(length(mset)) monomials")

resonance_set = resonance_set_from_complex_normal_form_style(
    ROM, mset, super_eigenvalues, target_eigenvalues, 0.05
)
# resonance_set = resonance_set_from_graph_style(
#     ROM, mset, super_eigenvalues, outer_eigenvalues, 0.05
# )

println("\nResonance set:")
for (idx, mi) in enumerate(mset.exponents)
    res_str = join(findall(resonance_set.resonances[:, idx]), ", ")
    isempty(res_str) && (res_str = "none")
    println("  $mi → [$res_str]")
end

# ------------------------------------------------------------------------------
# 5. Solve cohomological equationseigs
#    External eigenvalues are read from model.external_system automatically.
# ------------------------------------------------------------------------------
Profile.clear()
W,
R = solve_cohomological_problem(
    model, mset,
    master_eigenvalues,
    master_modes, left_eigenmodes,
    resonance_set;
    master_modes_derivatives = master_modes_derivatives
)

@profview W,
R = solve_cohomological_problem(
    model, mset,
    master_eigenvalues,
    master_modes, left_eigenmodes,
    resonance_set;
    master_modes_derivatives = master_modes_derivatives
)
# # ------------------------------------------------------------------------------
# # 6. Realify
# # ------------------------------------------------------------------------------
# conj_map = zeros(Int64, NVAR)
# for i in 1:NVAR
#     if i % 2 == 1
#         conj_map[i] = i + 1
#     else
#         conj_map[i] = i - 1
#     end
# end
# Rr = ReducedDynamics(realify(R.poly, conj_map), R.external_system_size)

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
        for d in 1:Int(NVAR * 0.5)
            rcoeff = real(coeff[2 * d - 1, m])
            icoeff = -imag(coeff[2 * d - 1, m])
            if abs(rcoeff) > 1e-20
                rdyn[2 * d - 1] *= " + " * string(rcoeff) * monomial
            end
            if abs(icoeff) > 1e-20
                rdyn[2 * d] *= " + " * string(icoeff) * monomial
            end
        end
    end

    ofile = open("./equations.txt", "w")
    for i in 1:NVAR
        write(ofile, rdyn[i] * ";\n")
    end
    close(ofile)
end
# write_rdyn(Rr)

# # print values
# mset = R.poly.multiindex_set
# coefficients = R.poly.coefficients
# for m in 1:length(mset.exponents)
#     multiindex = mset.exponents[m]
#     coeff = coefficients[:, m]
#     println(multiindex, " ", coeff)
# end
