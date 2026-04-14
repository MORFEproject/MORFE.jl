include(joinpath(@__DIR__, "../../src/MORFE.jl"))

using LinearAlgebra
using StaticArrays: SVector

using .MORFE
using .MORFE.EigenModesPropagation
using .MORFE.ParametrisationMethod: Parametrisation

# -------------------------------------------------------------------
# 1. Left and right eigenvectors
# -------------------------------------------------------------------

# Setup simple n-th order model without MultilinearMap's
FOM = 5
B₃ = 4.0 * Matrix{Float64}(I, FOM, FOM)
B₂ = 3.0 * Matrix{Float64}(I, FOM, FOM)
B₁ = 2.0 * Matrix{Float64}(I, FOM, FOM)
B₀ = 1.0 * Matrix{Float64}(I, FOM, FOM)
linear_terms = (B₀, B₁, B₂, B₃)
ORD = length(linear_terms) - 1
model = NDOrderModel(linear_terms)

#First order matrices
A, B = linear_first_order_matrices(model)

#Left and right eigenvectors from First order system
# It may be that it is neccesary to sort the left and right eigenvalues!
eigen_right = eigen(B \ A)
eigen_left = eigen(B' \ A')
μ = eigen_left.values
X = eigen_left.vectors
λ = eigen_right.values
Y = eigen_right.vectors
# Caution: λ and μ are not sorted: λ != conj(μ)

# Propagation
# Right eigenvectors are stored in Parametrisation
mset = all_multiindices_up_to(ORD * FOM, 1)
# param_coeff = zeros(ComplexF64, FOM, ORD, length(mset))
# poly = DensePolynomial(param_coeff, mset)
(param, r) = create_parametrisation_method_objects(mset, ORD, FOM)
param_coeff = param.poly.coefficients
left_eigenvectors = zeros(ComplexF64, FOM, ORD, ORD * FOM)

for i in 1:(ORD * FOM)
    println("Eigenvectors: $i: ")
    propagate_right_eigenvector_form_first(param, Y[1:FOM, i], λ[i], i)
    propagate_left_eigenvector_from_last(
        model, left_eigenvectors, X[(FOM * (ORD - 1) + 1):end, i], conj(μ[i]), i)
    tmp = norm(Y[:, i] - vec(param_coeff)[(FOM * ORD * (i - 1) + 1):(FOM * ORD * i)])
    println("   left: $tmp")
    tmp = norm(X[:, i] - vec(left_eigenvectors)[(FOM * ORD * (i - 1) + 1):(FOM * ORD * i)])
    println("   right: $tmp")
end

println("="^80)
println("Demo finished successfully.")