# demo_external_system.jl
# Demonstrates the usage of ExternalSystem for representing polynomial dynamical systems.

using StaticArrays, LinearAlgebra
include(joinpath(@__DIR__, "../../src/MORFE.jl"))  # adjust path as needed

using .MORFE.Multiindices: MultiindexSet, all_multiindices_up_to
using .MORFE.Polynomials: DensePolynomial, evaluate
using .MORFE.ExternalSystem: ExternalSystem

# -------------------------------------------------------------------
# 1. Build a purely linear, decoupled system from eigenvalues
# -------------------------------------------------------------------
eigenvalues_real = (-1.0, -2.0)               # real eigenvalues
ext_sys1 = ExternalSystem(eigenvalues_real)

println("\n=== System from real eigenvalues (promoted to complex) ===\n")
println("Type: ", typeof(ext_sys1))
println("Eigenvalues: ", ext_sys1.eigenvalues)
println("Linear matrix:\n", repr("text/plain", ext_sys1.linear_matrix))
println("Polynomial: ")
for (i, exp) in enumerate(ext_sys1.first_order_dynamics.multiindex_set.exponents)
	println("\tindex = $i:\texponent = $exp\tcoefficient=$(ext_sys1.first_order_dynamics.coefficients[i])")
end

# Evaluate the dynamics at a state
r = SVector(3.0, 4.0)
println("\nEvaluate at r = ", r)
println("linear_matrix * r =\t", ext_sys1.linear_matrix * r)
println("polynomial evaluate =\t", evaluate(ext_sys1.first_order_dynamics, r))

# -------------------------------------------------------------------
# 2. Build a nonlinear system from a polynomial and eigenvalues
# -------------------------------------------------------------------
# Define a 2‑D polynomial with linear and quadratic terms:
#   dx1/dt = -x1 + x2 + 0.5*x1*x2
#   dx2/dt = -x2 - x1*x2
#
# We need to represent this as a DensePolynomial with coefficients
# given for each multiindex.

n = 2
# Multiindices up to degree 2 (excluding constant term)
multi_set = all_multiindices_up_to(n, 2)
deleteat!(multi_set.exponents, 1)  # remove zero exponent (constant term)

# Get list of multiindices (as tuples)
exponents = multi_set.exponents

# Create mapping from exponent tuple to coefficient vector
coeff_dict = Dict{NTuple{2, Int}, SVector{2, Float64}}()
coeff_dict[(1, 0)] = SVector(-1.0, 0.0)   # dx1/dt: -x1; dx2/dt: 0
coeff_dict[(0, 1)] = SVector(1.0, -1.0)  # dx1/dt: +x2; dx2/dt: -x2
coeff_dict[(1, 1)] = SVector(0.5, -1.0)   # dx1/dt: 0.5*x1*x2; dx2/dt: -x1*x2

# Build coefficient list in the same order as exponents, (0,1), (2,0)
coefficients = [
	SVector(-1.0, 0.0), # (1,0) # dx1/dt: -x1       ;   dx2/dt: 0
	SVector(1.0, -1.0), # (0,1) # dx1/dt: 0         ;   dx2/dt: -1 x2
	SVector(5.0, 10.0), # (2,0) # dx1/dt: 5 x1^2    ;   dx2/dt: 10 x1^2
	SVector(-3.0, 0.0), # (1,1) # dx1/dt: -3 x1x2   ;   dx2/dt: 0
	SVector(0.1, -0.1), # (0,2) # dx1/dt: 0.1 x2^2  ;   dx2/dt: -0.1 x2^2
]

# Construct polynomial
poly = DensePolynomial(coefficients, multi_set)

# Build ExternalSystem from polynomial and eigenvalues (computes linear matrix automatically)
ext_sys2 = ExternalSystem(poly)

println("\n=== System with quadratic nonlinearities from DensePolynomial ===\n")
println("Type: ", typeof(ext_sys2))
println("Linear matrix:\n", repr("text/plain", ext_sys2.linear_matrix))
println("Eigenvalues: ", ext_sys2.eigenvalues)
println("Polynomial: ")
for (i, exp) in enumerate(ext_sys2.first_order_dynamics.multiindex_set.exponents)
	println("\tindex = $i:\texponent = $exp\tcoefficient=$(ext_sys2.first_order_dynamics.coefficients[i])")
end

# Evaluate the dynamics at a state
r = SVector(3.0, 4.0)
println("\nEvaluate at r = ", r)
println("linear_matrix * r =\t", ext_sys2.linear_matrix * r)
println("polynomial evaluate =\t", evaluate(ext_sys2.first_order_dynamics, r))

# -------------------------------------------------------------------
# 3. Linear decoupled system with complex eigenvalues
# -------------------------------------------------------------------
eigenvalues_complex = (-0.5 + 1.0im, -0.5 - 1.0im, -1.0 + 0.0im)
ext_sys3 = ExternalSystem(eigenvalues_complex)

println("\n=== Linear decoupled system from complex eigenvalues ===\n")
println("Eigenvalues: ", ext_sys3.eigenvalues)
println("Linear matrix:\n", repr("text/plain", ext_sys3.linear_matrix))

# Evaluate the dynamics at a state
r = SVector(3.0+4.0im, 4.0-3.0im, 0.5)
println("\nEvaluate at r = ", r)
println("linear_matrix * r =\t", ext_sys3.linear_matrix * r)
println("polynomial evaluate =\t", evaluate(ext_sys3.first_order_dynamics, r))

# -------------------------------------------------------------------
println("\n" * "="^80 * "\n")
println("Demo finished successfully.")