# demo_lower_order_couplings.jl
# Demo for compute_lower_order_couplings function

using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "../src/MORFE.jl"))
using .MORFE.Multiindices: MultiindexSet
using .MORFE.Polynomials
using .MORFE.LowerOrderCouplings

using .LowerOrderCouplings.ParametrisationMethod: Parametrisation, ReducedDynamics

# Helper to create a MultiindexSet from a list of exponent vectors
function make_multiindex_set(exps::Vector{Vector{Int}})
    N = isempty(exps) ? 0 : length(exps[1])
    sv_exps = [SVector{N,Int}(exp) for exp in exps]
    return MultiindexSet(sv_exps)  # sorts in Grlex
end

# Helper to construct a Parametrisation (first‑order system, ORD = 1) from a dictionary
function construct_parametrisation(dict::Dict{Vector{Int}, Vector{Float64}})
    if isempty(dict)
        N = 0
        FOM = 0
        mset = MultiindexSet(Vector{SVector{0,Int}}())
        return DensePolynomial(SVector{1,SVector{FOM,Float64}}[], mset)
    end
    exps = collect(keys(dict))
    mset = make_multiindex_set(exps)
    N = length(exps[1])  # NVAR
    FOM = length(first(values(dict)))  # FOM

    # Map exponent -> index in mset
    exp_to_idx = Dict{SVector{N,Int}, Int}()
    for (idx, exp) in enumerate(mset.exponents)
        exp_to_idx[exp] = idx
    end

    # Build coefficient vector: each element is SVector{1, SVector{FOM,Float64}}
    coeffs = [SVector{1,SVector{FOM,Float64}}(SVector{FOM,Float64}(zeros(FOM))) for _ in 1:length(mset)]
    for (exp_vec, val) in dict
        exp_sv = SVector{N,Int}(exp_vec)
        idx = exp_to_idx[exp_sv]
        coeffs[idx] = SVector{1,SVector{FOM,Float64}}(SVector{FOM,Float64}(val))
    end
    return Parametrisation{1,FOM,N,Float64}(coeffs, mset)
end

# Helper to construct a ReducedDynamics from a dictionary
function construct_reduced_dynamics(dict::Dict{Vector{Int}, Vector{Float64}})
    if isempty(dict)
        N = 0
        ROM = 0
        mset = MultiindexSet(Vector{SVector{0,Int}}())
        return DensePolynomial(SVector{ROM,Float64}[], mset)
    end
    exps = collect(keys(dict))
    mset = make_multiindex_set(exps)
    N = length(exps[1])  # NVAR
    ROM = length(first(values(dict)))  # ROM

    exp_to_idx = Dict{SVector{N,Int}, Int}()
    for (idx, exp) in enumerate(mset.exponents)
        exp_to_idx[exp] = idx
    end

    coeffs = [SVector{ROM,Float64}(zeros(ROM)) for _ in 1:length(mset)]
    for (exp_vec, val) in dict
        exp_sv = SVector{N,Int}(exp_vec)
        idx = exp_to_idx[exp_sv]
        coeffs[idx] = SVector{ROM,Float64}(val)
    end
    return ReducedDynamics{ROM,N,Float64}(coeffs, mset)
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

param = construct_parametrisation(param_dict)
red = construct_reduced_dynamics(red_dict)

println("\nParametrisation coefficients: ")
for (idx, exp) in enumerate(multiindex_set(param).exponents)
    println("  exp $exp → $(param.coeffs[idx])")
end
println("\nReduced dynamics coefficients: ")
for (idx, exp) in enumerate(multiindex_set(red).exponents)
    println("  exp $exp → $(red.coeffs[idx])")
end

# -------------------------------------------------------------------
println("\nExample 1: upper_bound = (1,1)")
upper_bound = SVector{2,Int}(1,1)

result = compute_lower_order_couplings(upper_bound, param, red)
println("Result: $result")

# Expected result: param_(2,0) * 2 * red_(0,1)[1] = [4.0, 16.0, -6.0] * 2 * 2.1 = [16.8, 67.2, -25.2]
expected_manual_computation = [16.8, 67.2, -25.2]
println("Expected = $expected_manual_computation")
println("Difference: $(result - SVector{1,SVector{3,Float64}}(expected_manual_computation))")

# -------------------------------------------------------------------
println("\nExample 2: upper_bound = (2,1)")
upper_bound2 = SVector{2,Int}(2,1)
result2 = compute_lower_order_couplings(upper_bound2, param, red)
println("Result: $result2")

expected_manual_computation2 = [107.7, 774.4, -191.7]
println("Expected = $expected_manual_computation2")
println("Difference: $(result2 - SVector{1,SVector{3,Float64}}(expected_manual_computation2))")

# -------------------------------------------------------------------
println("\n--- Example 3: 3 variables, random polynomials ---")

# Random polynomials in 3 variables, total degree up to 5
nvars3 = 3
maxdeg = 5
mset3 = all_multiindices_up_to(nvars3, maxdeg)
nterms = length(mset3)

# Random coefficients for parametrisation: FOM = 3, ORD = 1
param_coeffs3 = [SVector{1,SVector{3,Float64}}(randn(SVector{3,Float64})) for _ in 1:nterms]
param3 = DensePolynomial(param_coeffs3, mset3)  # will be Parametrisation{1,3,3,Float64}

# Random coefficients for reduced dynamics: ROM = 3 (since 3 variables)
red_coeffs3 = [randn(SVector{3,Float64}) for _ in 1:nterms]
red3 = DensePolynomial(red_coeffs3, mset3)  # ReducedDynamics{3,3,Float64}

upper_bound3 = SVector{3,Int}(1,4,1)
result3 = compute_lower_order_couplings(upper_bound3, param3, red3)
println("Result for random 3‑variable case: $result3")

# -------------------------------------------------------------------
println("\nExample 4: zero polynomials (should return zero)")

mset4 = all_multiindices_up_to(2, 2)  # up to total degree 2 in 2 variables
# Zero parametrisation: FOM = 3, ORD = 1
zero_param = DensePolynomial([SVector{1,SVector{3,Float64}}(zeros(SVector{3,Float64})) for _ in 1:length(mset4)], mset4)
# Zero reduced dynamics: ROM = 2
zero_red = DensePolynomial([SVector{2,Float64}(zeros(2)) for _ in 1:length(mset4)], mset4)
upper_bound4 = SVector{2,Int}(1,1)
result4 = compute_lower_order_couplings(upper_bound4, zero_param, zero_red)
println("Result for zero polynomials: $result4")
@assert iszero(result4)

# -------------------------------------------------------------------
println("\n" * "="^80 * "\n")
println("Demo finished successfully.")