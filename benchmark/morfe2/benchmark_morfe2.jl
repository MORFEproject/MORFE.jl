"""
Implementation of the MORFE module to reproduce and compare the system from Morfe2.0
Different Morfe2.0 examples can be used by exchanging 'info_file' and 'mesh_file'.
"""

import Pkg
Pkg.activate(@__DIR__)
if !haskey(Pkg.project().dependencies, "MORFE")
    Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
    Pkg.add(["Arpack", "LinearMaps", "StaticArrays"])
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
using Profile
# using ProfileView

#Make info
info = Infostruct()
info_file = joinpath(@__DIR__, "beam_damp.jl")
include(info_file)

#Import mesh
mesh_file = joinpath(@__DIR__, "beam.mphtxt")
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
# 1. Define NthOrderModel
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
model = NthOrderModel((K, C, M), (quadratic_term, cubic_term))

A, B = FullOrderModel.linear_first_order_matrices(model)
FOM = size(K, 1)
println("FOM:", FOM)

# ------------------------------------------------------------------------------
# 2. EigenProblem
# ------------------------------------------------------------------------------
# Compute left and right eigenpairs using the special StructureModalDampingEigensolver and store it in EigenProblem
eigenproblem = spectrum(
	model, StructureModalDampingEigensolver(10, info.α, info.β);
	sorter! = (args...) -> nothing)
(eigenvalues, Y, X) = (eigenproblem.eigenvalues, eigenproblem.eigenmodes, eigenproblem.left_eigenmodes)
for (i, λ) in enumerate(eigenvalues)
	println("  mode $i →   λ = $λ\n")
end
# ------------------------------------------------------------------------------
# 3. Select master modes and build the reduced-variable structure
# ------------------------------------------------------------------------------
ROM = 2             # number of master (dominant) modes
N_EXT = 0           # number of external forcing modes (for future use)
NVAR = ROM + N_EXT


master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes = Y[:, 1, 1:ROM]            # size: FOM × ROM
# left_eigenmodes = Y[(FOM + 1):end, 1:ROM]  # size: FOM × ROM
left_eigenmodes = X[:, 1:ROM]      # size: FOM × ROM — X is the FOM × n_eigs left-eigenmode slice

# Higher-order master mode derivatives W^(k)[e_r], k = 2 … ORD.
# For the companion-form eigenproblem with state ẑ = [x; ẋ; …], the k-th
# block of ẑ (rows (k-1)*FOM+1 : k*FOM) gives W^(k)[e_r] directly.
ORD_model = length(model.linear_terms) - 1   # = 2 for this second-order system
# master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
# for r in 1:ROM
#     for k in 1:(ORD_model - 1)   # k = 1 only for ORD = 2
#         master_modes_derivatives[:, k, r] .= Y[(k * FOM + 1):((k + 1) * FOM), r]
#     end
# end
master_modes_derivatives = Y

# ------------------------------------------------------------------------------
# 4. Build multiindex set and resonance set
# ------------------------------------------------------------------------------

outer_eigenvalues = eigenvalues[(ROM+1):end] # 2*10-FOM
# no external system — NVAR = ROM
max_degree = 7
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
# 5. Solve cohomological equationseigs
#    External eigenvalues are read from model.external_system automatically.
# ------------------------------------------------------------------------------
# Profile.clear()
# One spectral object in place of five hand-sliced arrays; `SpectralData` applies the
# mirrored right/left block convention.
left_modes_derivatives = left_eigenmode_orders_from_slice(
    model.linear_terms, left_eigenmodes, collect(master_eigenvalues))[:, 1:(end - 1), :]
spectral = SpectralData(; eigenvalues = master_eigenvalues,
    right_modes = master_modes, right_derivatives = master_modes_derivatives,
    left_modes = left_eigenmodes, left_blocks = Array(left_modes_derivatives))

@time W, R = solve_cohomological_problem(model, mset, spectral, resonance_set)
@time W, R = solve_cohomological_problem(model, mset, spectral, resonance_set)

println("finished")

# @profview W,
# R = solve_cohomological_problem(
#     model, mset,
#     master_eigenvalues,
#     master_modes, left_eigenmodes,
#     resonance_set;
#     master_modes_derivatives = master_modes_derivatives
# )
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

# # ------------------------------------------------------------------------------
# # 7. Write real dynamics to compare to Morfe 2.0
# # ------------------------------------------------------------------------------
# function write_rdyn(R::ReducedDynamics{ROM, NVAR, T}) where {ROM, NVAR, T}
#     rdyn = ["" for i in 1:NVAR]
#     for i in 1:NVAR
#         rdyn[i] = "a" * string(i) * "' = "
#     end
#     coeff = R.poly.coefficients
#     mset = R.poly.multiindex_set

#     for m in 1:length(mset.exponents)
#         multiindex = mset.exponents[m]
#         monomial = ""
#         for d in 1:NVAR
#             if (multiindex[d] != 0)
#                 monomial *= "*a" * string(d) * "^" * string(multiindex[d])
#             end
#         end
#         for d in 1:Int(NVAR * 0.5)
#             rcoeff = real(coeff[2 * d - 1, m])
#             icoeff = -imag(coeff[2 * d - 1, m])
#             if abs(rcoeff) > 1e-20
#                 rdyn[2 * d - 1] *= " + " * string(rcoeff) * monomial
#             end
#             if abs(icoeff) > 1e-20
#                 rdyn[2 * d] *= " + " * string(icoeff) * monomial
#             end
#         end
#     end

#     ofile = open("./equations.txt", "w")
#     for i in 1:NVAR
#         write(ofile, rdyn[i] * ";\n")
#     end
#     close(ofile)
# end
# write_rdyn(Rr)

# print values
mset = R.poly.multiindex_set
coefficients = R.poly.coefficients
for m in 1:length(mset.exponents)
	multiindex = mset.exponents[m]
	coeff = coefficients[:, m]
	println(multiindex, " ", coeff)
end
