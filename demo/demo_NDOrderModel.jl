"""
Demonstration of the usage of NDOrderModel and FirstOrderModel
"""

include(joinpath(@__DIR__, "../src/MORFE.jl"))
using .MORFE.Multiindices: MultiindexSet
using .MORFE.Polynomials: DensePolynomial
using .MORFE.FullOrderModel: NDOrderModel, FirstOrderModel,
	MultilinearMap, evaluate_term!,
	linear_first_order_matrices, evaluate_nonlinear_terms!
using LinearAlgebra
using StaticArrays: SVector

n = 2 # dimension of system

# -------------------------------------------------------------------
# 1. FirstOrderModel: first‑order system
#    B₁ ẋ + B₀ x = F(x)
# -------------------------------------------------------------------

# Linear matrices (diagonal for simplicity)
B₁ = 0.5 * Matrix{Float64}(I, n, n)
B₀ = 3.0 * Matrix{Float64}(I, n, n)

function bilinear_term!(res, x1, x2)
	@. res += x1 * x2
end
f_term1 = MultilinearMap(bilinear_term!)   # degree 2

function trilinear_term!(res, x1, x2, x3)
	@. res += 3.0 * x1 * x2 * x3
end
f_term2 = MultilinearMap(trilinear_term!)  # degree 3

model_fo = FirstOrderModel((B₀, B₁), (f_term1, f_term2))
A_fo, B_fo = linear_first_order_matrices(model_fo)
println("=== FirstOrderModel ===")
println("\nA (linear part):\n", repr("text/plain", A_fo))
println("\nB (mass matrix):\n", repr("text/plain", B_fo))

# Evaluate nonlinear terms – note the state is passed as a 1‑tuple (x,)
x_fo = [1.0, 2.0]
res_fo = zeros(n)

for deg in 1:4
	res_fo .= 0
	evaluate_nonlinear_terms!(res_fo, model_fo, deg, x_fo)
	println("\nDegree $deg contribution: ", res_fo)
end

# -------------------------------------------------------------------
# 2. NDOrderModel: second‑order system
#    B₂ x'' + B₁ x' + B₀ x = F(x, x', r)
#    where r is an external forcing variable satisfying r' = 3*r
# -------------------------------------------------------------------

# Linear matrices (diagonal for simplicity)
B₂ = 3.0 * Matrix{Float64}(I, n, n)
B₁ = 2.0 * Matrix{Float64}(I, n, n)
B₀ = 1.0 * Matrix{Float64}(I, n, n)

# Nonlinear terms are defined as multilinear maps.
# Each map corresponds to a monomial in the derivatives.
# The multiindex (i₀, i₁) tells how many times x (derivative 0) and x' (derivative 1) appear.

# Term 1: x * x'  (asymmetric: linear in x and in x')
function asymmetric_force!(res, x, xdot)
	@. res += x * xdot        # elementwise product
end
term1 = MultilinearMap(asymmetric_force!, (1, 1))   # one x, one x'

# Term 2: 0.5 * x' * x'  (symmetric quadratic in x')
function fluid_drag!(res, xdot1, xdot2)
	@. res += 0.5 * xdot1 * xdot2
end
term2 = MultilinearMap(fluid_drag!, (0, 2))         # two copies of x'

# Term 3: 0.5 * x * x * x'  (cubic: two x, one x')
function nonlinear_damping!(res, x1, x2, xdot)
	@. res += 0.5 * x1 * x2 * xdot
end
term3 = MultilinearMap(nonlinear_damping!, (2, 1))  # two x, one x'

# External forcing: term = [1; 0] * r, with r' = 3*r
function forcing!(res, r)
	@. res += [1.0, 0.0] * r
end
term_forcing = MultilinearMap(forcing!, (0, 0), 1)   # no state derivatives, one external variable

# External dynamics: r' = 3*r (polynomial of degree 1 in r)
external_dynamics = DensePolynomial([SVector(3.0)], MultiindexSet([SVector(1)]))

# Collect all nonlinear terms
nonlinear_terms = (term1, term2, term3, term_forcing)

# Build the second‑order model, including external dynamics
model_nd = NDOrderModel((B₀, B₁, B₂), nonlinear_terms, external_dynamics)

# Generate the equivalent first‑order matrices for the linear part:
#   B Ẋ = A X   with X = [x; x']
A_nd, B_nd = linear_first_order_matrices(model_nd)

println("\n\n=== NDOrderModel ===")
println("\nA matrix (linear part in first‑order form):\n", repr("text/plain", A_nd))
println("\nB matrix (mass matrix in first‑order form):\n", repr("text/plain", B_nd))

# Evaluate nonlinear terms for a given state, its derivative, and external state
x = [1.0, 2.0]        # x
xdot = [0.1, -0.2]    # x'
r = 50.0              # external forcing variable
state_vectors = (x, xdot)   # tuple expected by evaluate_nonlinear_terms!

res_nd = zeros(n)

for term in model_nd.nonlinear_terms
	res_nd .= 0
	evaluate_term!(res_nd, term, state_vectors, r)
	println("\nContribution of $(term.f!): ", res_nd)
end

# Evaluate all terms of degree 1, 2, 3, 4. The forcing term is degree 1 (external only)
for deg in 1:4
	res_nd .= 0
	evaluate_nonlinear_terms!(res_nd, model_nd, deg, state_vectors, r)
	println("\nDegree $deg contribution: ", res_nd)
end

println("\n" * "="^80 * "\n")
println("Demo finished successfully.")
