# demo_eigenproblem.jl
# ==============================================================================
# Shows usage of the module Eigenproblems in 
#   "src/SpectralDecomposition/Eigenproblems.jl"
# ==============================================================================

using MORFE
using LinearAlgebra

# ------------------------------------------------------------------------------
# 1. Setup 
#    Define NDOrderModel
#    Second-order ODE:  M ẍ + C ẋ + K x = 0
# ------------------------------------------------------------------------------
B0 = [2.0 -1.0; -1.0 2.0]   # stiffness
B1 = [0.01 0.0; 0.0 0.01]   # light damping
B2 = [1.0 0.0; 0.0 1.0]   # mass (highest-order coefficient)

model = NDOrderModel((B0, B1, B2))

# # ------------------------------------------------------------------------------
# # 2.1. Default 
# # Default solver uses eigen from LinearAlgebra,
# # sorts by magnitude, and normlizes biorthogonal
# # ------------------------------------------------------------------------------

# Compute left and right eigenpairs using the default solver and store it in Eigenproblem
eigenproblem = solve_eigenproblem(model)# , normaliser! = (args...) -> nothing)
(eigs, Y, X) = get_eigenpairs(eigenproblem)
for (i, λ) in enumerate(eigs)
    println("  mode $i →   λ = $λ\n\t     y = $(Y[:, i])\n\t     x = $(X[:, i])\n")
end

# Test left and right eigenvectors
A, B = linear_first_order_matrices(model)
for (i, λ) in enumerate(eigs)
    res_y = norm(A * Y[:, i] - λ * B * Y[:, i])
    res_x = norm(X[:, i]' * A - λ * X[:, i]' * B)
    @assert res_y<1e-8 "eigenvectors doesnt match: i=$i, res_y = $res_y"
    @assert res_x<1e-8 "eigenvectors doesnt match: i=$i, res_x = $res_x"
end

# Choose master_modes
select_master_modes_by_hand(eigenproblem, [false, true, false, true])
println("by_hand: ", eigenproblem.master_modes)
select_master_modes_by_sorting(eigenproblem, 2)
println("by_sorting: ", eigenproblem.master_modes)
select_master_modes_by_target_frequency(eigenproblem, [(-0.005 - 1 * im)], 1e-4)
println("by_target_frequency: ", eigenproblem.master_modes)

println("\n" * "="^80 * "\n")
# ------------------------------------------------------------------------------
# 2.2. Spezialized
# Assumes the first order system matrices are symmetric 
# ------------------------------------------------------------------------------
B0 = [-1.0 0.0; 0.0 -1.0]   # stiffness
B1 = [0.01 0.0; 0.0 0.01]   # light damping
B2 = [2.0 0.0; 0.0 2.0]   # mass (highest-order coefficient)

model = NDOrderModel((B0, B1, B2))

#Add new solver that uses that X'=Y
mutable struct My_Own_Solver <: AbstractEigenSolver
    right_eig_result::Any
end
# need to implement left and right solve functions
function MORFE.Eigenproblems.solve(model::NDOrderModel, solver::My_Own_Solver)
    A, B = linear_first_order_matrices(model)
    eig_result = eigen(A, B)
    solver.right_eig_result = eig_result
    println(solver.right_eig_result)
    return eig_result.values, eig_result.vectors
end

function MORFE.Eigenproblems.solve_left(model::NDOrderModel, solver::My_Own_Solver)
    @assert solver.right_eig_result!==nothing "First run solve()"
    eig_result = solver.right_eig_result
    return eig_result.values, (eig_result.vectors)
end

# Compute left and right eigenpairs using the default solver and store it in Eigenproblem
eigenproblem = solve_eigenproblem(
    model, solver = My_Own_Solver(nothing), normaliser! = (args...) -> nothing)
(eigs, Y, X) = get_eigenpairs(eigenproblem)
for (i, λ) in enumerate(eigs)
    println("  mode $i →   λ = $λ\n\t     y = $(Y[:, i])\n\t     x = $(X[:, i])\n")
end

# Test left and right eigenvectors
A, B = linear_first_order_matrices(model)
for (i, λ) in enumerate(eigs)
    res_y = norm(A * Y[:, i] - λ * B * Y[:, i])
    res_x = norm(X[:, i]' * A - λ * X[:, i]' * B)
    @assert res_y<1e-8 "eigenvectors doesnt match: i=$i, res_y = $res_y"
    @assert res_x<1e-8 "eigenvectors doesnt match: i=$i, res_x = $res_x"
end
