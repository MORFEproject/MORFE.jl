"""
Demonstration of the usage of NDOrderModel
"""

include(joinpath(@__DIR__, "../src/FullOrderModel.jl"))

using LinearAlgebra

"""
    B_2 x^(2) + B_1 x^(1) + B_0 x = N(x,x_1)
    Dimension = 2
"""
N = 3
# Define linear matrices
B_3 = Matrix{Float64}(I, 2, 2)
B_2 = Matrix{Float64}(I, 2, 2)
B_1 = Matrix{Float64}(I, 2, 2)
B_0 = Matrix{Float64}(I, 2, 2)

# Define nonlinear terms
function quad_x_xdot!(res, x, xdot)
    @. res += x * xdot   # elementwise product
end
function quad_xdot_xdot!(res, xdot, xdot2)
    @. res += xdot * xdot2 * 0.5  # elementwise product
end
function cubic_xdot_xdot2_xddot!(res, xdot, xdot2, xddot)
    @. res += xdot * xdot2 * xddot * 0.5  # elementwise product
end

terms = (PolynomialTerm{N}(quad_x_xdot!, (0, 1)),
    PolynomialTerm{N}(quad_xdot_xdot!, (1, 1); symmetry = (1, 1)),
    PolynomialTerm{N}(cubic_xdot_xdot2_xddot!, (1, 1, 2); symmetry = (1, 1, 2)))

#Define 
model = NDOrderModel(
    (B_0, B_1, B_2, B_3),
    terms
)