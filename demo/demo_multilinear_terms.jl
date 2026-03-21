# demo_multilinear_terms.jl
# Run this script from the directory that contains MultilinearTerms.jl

using LinearAlgebra

# ----------------------------------------------------------------------
# Include required modules (adjust paths if necessary)
# ----------------------------------------------------------------------
# include(joinpath(@__DIR__, "../../Polynomials.jl"))
# include(joinpath(@__DIR__, "../../FullOrderModel.jl"))
# using .Polynomials, .FullOrderModel

include("../src/ParametrisationMethod/RightHandSide/MultilinearTerms.jl")
using .MultilinearTerms: compute_multilinear_terms,
        Parametrisation, DensePolynomial,
        MultilinearMap, FirstOrderModel, NDOrderModel,
        all_multiindices_up_to, find_term

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
n = 2                               # dimension of the state space
nvars = 2                            # number of variables in the multi‑indices
max_degree = 2                        # maximum total degree of the polynomial expansion

# Create a multi‑index set containing all exponents up to degree 2
ms = all_multiindices_up_to(nvars, max_degree)

# Choose a specific exponent for which we want the nonlinear contribution
exp = [1, 1]   # corresponds to x₁·x₂

# Candidate indices: all indices in the multi‑index set (for simplicity)
candidate_indices = 1:length(ms)

# ----------------------------------------------------------------------
# 1. First‑order model :  B₁ ẋ + B₀ x = F(x)
# ----------------------------------------------------------------------

# Linear matrices (diagonal for simplicity)
B₁_fo = 0.5 * Matrix{Float64}(I, n, n)
B₀_fo = 3.0 * Matrix{Float64}(I, n, n)

# Nonlinear terms (multilinear maps)

# Quadratic term: (x, x) -> result
function bilinear_fo!(res, x1, x2)
    @. res += x1 * x2          # elementwise product
end
term1_fo = MultilinearMap(bilinear_fo!)   # degree 2, only x appears

# Cubic term: (x, x, x) -> result
function trilinear_fo!(res, x1, x2, x3)
    @. res += 3.0 * x1 * x2 * x3
end
term2_fo = MultilinearMap(trilinear_fo!)  # degree 3, only x appears

# Build the first‑order model
model_fo = FirstOrderModel((B₀_fo, B₁_fo), (term1_fo, term2_fo))

# Create a DensePolynomial whose coefficients are 2×2 matrices
poly_fo = DensePolynomial{Matrix{Float64}}(ms)

# Coefficient for exponent [1,0] (x₁)
poly_fo.coeffs[1] = [1.0; 2.0; 3.0]
# Coefficient for exponent [0,1] (x₂)
poly_fo.coeffs[2] = [4.0; 5.0; 6.0]
# Coefficient for exponent [2,0] (x₁²)
poly_fo.coeffs[3] = [7.0; 8.0; 9.0]
# Coefficient for exponent [1,1] (x₁·x₂) – this one will be used in factorisations for exp = [1,1]
poly_fo.coeffs[4] = [10; 11.0; 12.0]
# Coefficient for exponent [0,2] (x₂²)
poly_fo.coeffs[5] = [13.0; 14.0; 15.0]

println("\n=== First‑order model ===")
println("Coefficients of the parametrisation polynomial:")
for (idx, exp_) in enumerate(eachcol(ms.exponents))
    println("  exp = $exp_  →  ", poly_fo.coeffs[idx])
end

# Compute the total nonlinear contribution for exponent [1,1]
result_fo = compute_multilinear_terms(model_fo, exp, poly_fo, ms, candidate_indices)

println("\nResult for exponent $exp (first‑order model):")
println(result_fo)

# ----------------------------------------------------------------------
# 2. Second‑order (ND) model :  B₂ x'' + B₁ x' + B₀ x = F(x, x')
# ----------------------------------------------------------------------

# Linear matrices
B₂_nd = 3.0 * Matrix{Float64}(I, n, n)
B₁_nd = 2.0 * Matrix{Float64}(I, n, n)
B₀_nd = 1.0 * Matrix{Float64}(I, n, n)

# Nonlinear terms with their derivative‑order multiplicities

# Term 1: x * x'  (multiindex (1,1) → one x, one x')
function term1_nd!(res, x, xdot)
    @. res += x * xdot
end
nd1 = MultilinearMap(term1_nd!, (1,1))

# Term 2: 0.5 * x' * x'  (multiindex (0,2) → two copies of x')
function term2_nd!(res, xdot1, xdot2)
    @. res += 0.5 * xdot1 * xdot2
end
nd2 = MultilinearMap(term2_nd!, (0,2))

# Term 3: 0.5 * x * x * x'  (multiindex (2,1) → two x, one x')
function term3_nd!(res, x1, x2, xdot)
    @. res += 0.5 * x1 * x2 * xdot
end
nd3 = MultilinearMap(term3_nd!, (2,1))

# Build the second‑order model
model_nd = NDOrderModel((B₀_nd, B₁_nd, B₂_nd), (nd1, nd2, nd3))

# Create two polynomials: one for x (derivative order 0) and one for x' (derivative order 1)
poly_x   = DensePolynomial{Matrix{Float64}}(ms)   # coefficients for x
poly_xdot = DensePolynomial{Matrix{Float64}}(ms)  # coefficients for x'

# Set some coefficients (same as before, but could be different)
poly_x.coeffs[find_term(poly_x, [1,0])]   = [1.0 2.0; 3.0 4.0]
poly_x.coeffs[find_term(poly_x, [0,1])]   = [5.0 6.0; 7.0 8.0]
poly_x.coeffs[find_term(poly_x, [2,0])]   = [9.0 10.0; 11.0 12.0]
poly_x.coeffs[find_term(poly_x, [1,1])]   = [17.0 18.0; 19.0 20.0]

poly_xdot.coeffs[find_term(poly_xdot, [1,0])] = [0.1 0.2; 0.3 0.4]
poly_xdot.coeffs[find_term(poly_xdot, [0,1])] = [0.5 0.6; 0.7 0.8]
poly_xdot.coeffs[find_term(poly_xdot, [0,2])] = [1.3 1.4; 1.5 1.6]
poly_xdot.coeffs[find_term(poly_xdot, [1,1])] = [2.1 2.2; 2.3 2.4]

parametrisation_nd = (poly_x, poly_xdot)

println("\n\n=== Second‑order (ND) model ===")
println("Coefficients of poly_x (x):")
for (idx, exp_) in enumerate(eachcol(ms.exponents))
    println("  exp = $exp_  →  ", poly_x.coeffs[idx])
end
println("Coefficients of poly_xdot (x'):")
for (idx, exp_) in enumerate(eachcol(ms.exponents))
    println("  exp = $exp_  →  ", poly_xdot.coeffs[idx])
end

# Compute the total nonlinear contribution for the same exponent [1,1]
result_nd = compute_multilinear_terms(model_nd, exp, parametrisation_nd, ms, candidate_indices)

println("\nResult for exponent $exp (second‑order model):")
println(result_nd)

# -------------------------------------------------------------------
println("\n" * "="^80 * "\n")
println("Demo finished successfully.")