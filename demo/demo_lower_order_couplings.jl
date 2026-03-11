# demo_lower_order_couplings.jl
# Demo for compute_lower_order_couplings function

using LinearAlgebra

# Only include the module that defines the function – it already brings in its own Polynomials
include(joinpath(@__DIR__, "../src/ParametrisationMethod/RightHandSide/LowerOrderCouplings.jl"))
using .LowerOrderCouplings

# Bring needed names from its Polynomials submodule into scope
using .LowerOrderCouplings.Polynomials: DensePolynomial, MultiindexSet, all_multiindices_up_to, each_term

# Helper to construct a polynomial from a dictionary of coefficients (like in test_polynomials.jl)
function construct_polynomial(::Type{DensePolynomial}, dict::Dict{Vector{Int}, T}) where T
    if isempty(dict)
        mset = MultiindexSet(Matrix{Int}(undef, 0, 0))
        return DensePolynomial(T[], mset)
    else
        exps = collect(keys(dict))
        mset = MultiindexSet(exps)
        exp_to_idx = Dict{Tuple{Vararg{Int}}, Int}()
        for (j, col) in enumerate(eachcol(mset.exponents))
            exp_to_idx[Tuple(col)] = j
        end
        sample_val = first(values(dict))
        coeffs = [zero(sample_val) for _ in 1:length(mset)]
        for (exp, val) in dict
            coeffs[exp_to_idx[Tuple(exp)]] = val
        end
        return DensePolynomial(coeffs, mset)
    end
end

# -------------------------------------------------------------------
println("Demo: compute_lower_order_couplings \n")

# Define the multi‑index set covering all monomials up to total degree 2
# This includes exponents: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
param_dict = Dict{Vector{Int}, Vector{Float64}}(
    [0,0] => [1.0, 2.0, 0.0],
    [1,0] => [2.0, 4.0, -1.0],
    [0,1] => [3.0, 8.0, -3.0],
    [2,0] => [4.0, 16.0, -6.0],
    [1,1] => [5.0, 32.0, -9.0],
    [0,2] => [6.0, 64.0, -12.0]
)

# Reduced dynamics: each coefficient is a 2‑element vector (same length as number of variables)
red_dict = Dict{Vector{Int}, Vector{Float64}}(
    [0,0] => [0.1, 0.2],
    [1,0] => [1.1, 1.2],
    [0,1] => [2.1, 2.2],
    [2,0] => [3.1, 3.2],
    [1,1] => [4.1, 4.2],
    [0,2] => [5.1, 5.2]
)

param = construct_polynomial(DensePolynomial, param_dict)
red   = construct_polynomial(DensePolynomial, red_dict)

upper_bound = (1, 1)

println("\nParametrisation coefficients: ")
for (exp, coeff) in each_term(param)
    println("  exp $exp → $coeff")
end
println("\nReduced dynamics coefficients (vectors): ")
for (exp, coeff) in each_term(red)
    println("  exp $exp → $coeff")
end



# -------------------------------------------------------------------
println("\nExample 1: upper_bound = (1,1)")

result = compute_lower_order_couplings(upper_bound, param, red)
println("\nResult: $result")

# for upper_bound = (1,1) we have:
# i = 1 and e_1 = (1,0)
#   red_multiindex = (0,1)
#   param_multiindex = upper_bound - red_multiindex + e_1 = (2,0)
#   factor = param_multiindex[1] = 2
# i = 2 and e_2 = (0,1)
#   nothing
# result = param_(2,0) * 2 * red_multiindex_(0,1)_1 = [4.0, 16.0, -6.0] * 2 * 2.1 = [16.8, 67.2, -25.2]
expected_manual_computation = [16.8, 67.2, -25.2]
println("Expected = $expected_manual_computation")



# -------------------------------------------------------------------
println("\nExample 2: upper_bound = (2,1)")

upper_bound2 = (2, 1) # outside set of multiindices
result2 = compute_lower_order_couplings(upper_bound2, param, red)
println("Result: $result2")

# for upper_bound = (2,1) we have:
# i = 1 and e_1 = (1,0)
#   red_multiindex = (0,1)
#       param_multiindex = upper_bound - red_multiindex + e_1 = (3,0)
#       factor = param_multiindex[1] = 3
#       -> subtotal = param_(3,0) * 3 * red_multiindex_(0,1)_1 = nothing * 3 * 2.1 = 0.0
#   red_multiindex = (2,0)
#       param_multiindex = upper_bound - red_multiindex + e_1 = (1,1)
#       factor = param_multiindex[1] = 1
#       -> subtotal = param_(1,1) * 1 * red_multiindex_(2,0)_1 = [5.0, 32.0, -9.0] * 1 * 3.1 = [15.5, 99.2, -27.9]
#   red_multiindex = (1,1)
#       param_multiindex = upper_bound - red_multiindex + e_1 = (2,0)
#       factor = param_multiindex[1] = 2
#       -> subtotal = param_(2,0) * 2 * red_multiindex_(1,1)_1 = [4.0, 16.0, -6.0] * 2 * 4.1 = [32.8, 131.2, -49.2]
# i = 2 and e_2 = (0,1)
#   red_multiindex = (2,0)
#       param_multiindex = upper_bound - red_multiindex + e_2 = (0,2)
#       factor = param_multiindex[2] = 2
#       -> subtotal = param_(0,2) * 2 * red_multiindex_(2,0)_2 = [6.0, 64.0, -12.0] * 2 * 3.2 = [38.4, 409.6, -76.8]
#   red_multiindex = (1,1)
#       param_multiindex = upper_bound - red_multiindex + e_2 = (1,1)
#       factor = param_multiindex[2] = 1
#       -> subtotal = param_(1,1) * 1 * red_multiindex_(1,1)_2 = [5.0, 32.0, -9.0] * 1 * 4.2 = [21, 134.4, -37.8]
# result = 0.0 + [15.5, 99.2, -27.9] + [32.8, 131.2, -49.2] + [38.4, 409.6, -76.8] + [21, 134.4, -37.8] = [107.7, 774.4, -191.7]
expected_manual_computation = [107.7, 774.4, -191.7]
println("Expected = $expected_manual_computation")



# -------------------------------------------------------------------
println("\n--- Example 3: 3 variables, random polynomials ---")

# Random polynomials in 3 variables, total degree up to 2
nvars3 = 3
maxdeg = 5
mset3 = all_multiindices_up_to(nvars3, maxdeg)  # from LowerOrderCouplings.Polynomials
nterms = length(mset3)

# Random scalar coefficients for parametrisation (Float64)
param_coeffs3 = randn(nterms)
param3 = DensePolynomial(param_coeffs3, mset3)

# Random vector coefficients for reduced dynamics (each coefficient a 3‑vector)
red_coeffs3 = [randn(3) for _ in 1:nterms]
red3 = DensePolynomial(red_coeffs3, mset3)

upper_bound3 = (1, 4, 1)
result3 = compute_lower_order_couplings(upper_bound3, param3, red3)
println("Result for random 3‑variable case: $result3")



# -------------------------------------------------------------------
println("\nExample 4: zero polynomials (should return zero)")

mset4 = all_multiindices_up_to(2, 2)
zero_param = zero(DensePolynomial{Float64}, mset4)
zero_red   = zero(DensePolynomial{Vector{Float64}}, mset4, 2)  # vector length 2
upper_bound4 = (1, 1)
result4 = compute_lower_order_couplings(upper_bound4, zero_param, zero_red)
println("Result for zero polynomials: $result4")
@assert iszero(result4)



# -------------------------------------------------------------------
println("Demo finished successfully.")