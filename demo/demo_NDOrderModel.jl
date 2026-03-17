"""
Demonstration of the usage of NDOrderModel
"""

include(joinpath(@__DIR__, "../src/FullOrderModel.jl"))

using LinearAlgebra

"""
    B_2 x'' + B_1 x' + B_0 x = N(x, x')
    Second order system
"""
N = 2
# Define linear matrices
B_2 = Matrix{Float64}(I, 2, 2)
B_1 = Matrix{Float64}(I, 2, 2)
B_0 = Matrix{Float64}(I, 2, 2)

# Define nonlinear terms
function gyroscopic_force!(res, x, xdot)  # assymetric
    @. res += x * xdot[end:-1:1]  # elementwise product
end

function fluid_drag!(res, xdot1, xdot2)  # fully symmetric
    @. res += xdot1 * xdot2 * 0.5  # elementwise product
end

function nonlinear_damping!(res, x1, x2, xdot)  # fully symmetric
    @. res += x1 * x2 * xdot * 0.5  # elementwise product
end

terms = (PolynomialTerm{N}(gyroscopic_force!, (0, 1); symmetry = (1, 2)),
    PolynomialTerm{N}(fluid_drag!, (1, 1); symmetry = (1, 1)),
    PolynomialTerm{N}(nonlinear_damping!, (0, 0, 1); symmetry = (1, 1, 1)))

#Define 
model = NDOrderModel(
    (B_0, B_1, B_2),
    terms
)