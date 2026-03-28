# demo_lower_order_couplings.jl
# Demo for compute_lower_order_couplings function

using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "../src/MORFE.jl"))
using .MORFE.Multiindices: MultiindexSet, all_multiindices_up_to
using .MORFE.Polynomials
using .MORFE.LowerOrderCouplings
using .MORFE.ParametrisationMethod

# -------------------------------------------------------------------
println("Demo: compute_lower_order_couplings \n")

# --- Example 1: 2 variables, total degree ≤ 2, known coefficients ---
NVAR = 2
maxdeg = 2
mset = all_multiindices_up_to(NVAR, maxdeg)   # Grlex order: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)

# Parametrisation: second‑order system (ORD=2), FOM=3
ORD = 2
FOM = 3

# Coefficient vectors in the same order as mset.exponents
# Each coefficient is an SVector{ORD, Vector{ComplexF64}}
param_coefficients = [
	#       position part              velocity part
	SVector(ComplexF64[1.0, 2.0, 0.0], ComplexF64[1.0im, 2.0im, 0.0]),          # (0,0)
	SVector(ComplexF64[2.0, 4.0, -1.0], ComplexF64[2.0im, 4.0im, -1.0im]),      # (1,0)
	SVector(ComplexF64[3.0, 8.0, -3.0], ComplexF64[3.0im, 8.0im, -3.0im]),      # (0,1)
	SVector(ComplexF64[4.0, 16.0, -6.0], ComplexF64[4.0im, 16.0im, -6.0im]),    # (2,0)
	SVector(ComplexF64[5.0, 32.0, -9.0], ComplexF64[5.0im, 32.0im, -9.0im]),    # (1,1)
	SVector(ComplexF64[6.0, 64.0, -12.0], ComplexF64[6.0im, 64.0im, -12.0im]),  # (0,2)
]

red_coefficients = [
	SVector{NVAR, ComplexF64}(0.0, 0.0),        # (0,0)
	SVector{NVAR, ComplexF64}(1.0im, 0.0),      # (1,0)
	SVector{NVAR, ComplexF64}(1.0, 1.0im),      # (0,1)
	SVector{NVAR, ComplexF64}(3.0, 3.0im),      # (2,0)
	SVector{NVAR, ComplexF64}(4.0, 4.0im),      # (1,1)
	SVector{NVAR, ComplexF64}(5.0, 5.0im),      # (0,2)
]

# Construct the polynomials
W = DensePolynomial(param_coefficients, mset)   # Parametrisation{ORD=2,FOM=3,NVAR=2,Complex}
R = DensePolynomial(red_coefficients, mset)     # ReducedDynamics{ROM=2,NVAR=2,Complex}

println("\nParametrisation coefficients: ")
for (idx, exp) in enumerate(mset.exponents)
	println("  exp $exp → $(W.coefficients[idx])")
end
println("\nReduced dynamics coefficients: ")
for (idx, exp) in enumerate(mset.exponents)
	println("  exp $exp → $(R.coefficients[idx])")
end


# ----------------------------------------------------------------------
# 1. Monomial (1,1)
# ----------------------------------------------------------------------

println("\nExample 1: upper_bound = (1,1)")
upper_bound = mset.exponents[5] # SVector(1,1)
result = compute_lower_order_couplings(upper_bound, W, R)

println("Result: $result")

# for upper_bound = (1,1) we have:
# i = 1 and e_1 = (1,0)
#   red_multiindex = (0,1)
#   param_multiindex = upper_bound - red_multiindex + e_1 = (2,0)
#   factor = param_multiindex[1] = 2
# i = 2 and e_2 = (0,1)
#   nothing
# result = param_(2,0) * 2 * red_multiindex_(0,1)_1 
# = ([4.0, 16.0, -6.0], [4.0, 16.0, -6.0])*im) * 2 * 1.0 = ([8.0, 32.0, -12.0], [8.0im, 32.0im, -12.0im])
#
expected1 = SVector{2, Vector{ComplexF64}}(
	ComplexF64[8.0, 32.0, -12.0],          # position part
	ComplexF64[8.0im, 32.0im, -12.0im],     # velocity part
)
println("Expected = $expected1")
println("Relative error: ", norm(result - expected1)/norm(expected1))

# ----------------------------------------------------------------------
# 2. Monomial (2,1)
# ----------------------------------------------------------------------

println("\nExample 2: upper_bound = (2,1)")
upper_bound2 = SVector{2, Int}(2, 1)
result2 = compute_lower_order_couplings(upper_bound2, W, R)
println("Result: $result2")

# for upper_bound = (2,1) we have:
#
# i = 1 and e_1 = (1,0) ---------------------------------------------
#
#   red_multiindex = (0,1)
#       param_multiindex = upper_bound - red_multiindex + e_1 = (3,0)
#       factor = param_multiindex[1] = 3
#       -> subtotal = param_(3,0) * 3 * red_multiindex_(0,1)_1 = nothing * 3 * 2.1 = 0.0
#
#   red_multiindex = (2,0)
#       param_multiindex = upper_bound - red_multiindex + e_1 = (1,1)
#       factor = param_multiindex[1] = 1
#       -> subtotal = param_(1,1) * 1 * red_multiindex_(2,0)_1 = 
#          = ([5.0, 32.0, -9.0], [5.0, 32.0, -9.0]*im) * 1 * 3 = ([15, 96, -27], [15im, 96im, -27im])
#
#   red_multiindex = (1,1)
#       param_multiindex = upper_bound - red_multiindex + e_1 = (2,0)
#       factor = param_multiindex[1] = 2
#       -> subtotal = param_(2,0) * 2 * red_multiindex_(1,1)_1 = 
#          = ([4.0, 16.0, -6.0], [4.0, 16.0, -6.0]*im) * 2 * 4 = ([32, 128, -48], [32im, 128im, -48im])
#
# i = 2 and e_2 = (0,1) ---------------------------------------------
#
#   red_multiindex = (2,0)
#       param_multiindex = upper_bound - red_multiindex + e_2 = (0,2)
#       factor = param_multiindex[2] = 2
#       -> subtotal = param_(0,2) * 2 * red_multiindex_(2,0)_2 = 
#          = ([6.0, 64.0, -12.0], [6.0, 64.0, -12.0]*im) * 2 * 3im = ([36im, 384im, -72im], [-36, -384, 72])
#
#   red_multiindex = (1,1)
#       param_multiindex = upper_bound - red_multiindex + e_2 = (1,1)
#       factor = param_multiindex[2] = 1
#       -> subtotal = param_(1,1) * 1 * red_multiindex_(1,1)_2 = 
#          = ([5.0, 32.0, -9.0], [5.0, 32.0, -9.0]*im) * 1 * 4im = ([20im, 128im, -36im], [-20, -128, 36])
#
# result = 0.0 + 
#        + ([15,      96,        -27      ], [ 15im,     96im,     -27im    ]) 
#        + ([32,      128,       -48      ], [ 32im,     128im,    -48im    ]) 
#        + ([36im,    384im,     -72im    ], [-36,      -384,       72      ])
#        + ([20im,    128im,     -36im    ], [-20,      -128,       36      ])
#        = ([47+56im, 224+512im, -75-108im], [-56+47im, -512+224im, 108-75im])
#
expected2 = SVector{2, Vector{ComplexF64}}(
	ComplexF64[47.0+56.0im, 224.0+512.0im, -75.0-108.0im],
	ComplexF64[-56.0+47.0im, -512.0+224.0im, 108.0-75.0im],
)
println("Expected = $expected2")
println("Relative error: ", norm(result2 - expected2)/norm(expected2))

# ----------------------------------------------------------------------
# 3. Random polynomials
# ----------------------------------------------------------------------

println("\nExample 3: 3 variables, random polynomials")

NVAR3 = 3
maxdeg3 = 5
mset3 = all_multiindices_up_to(NVAR3, maxdeg3)
nterms = length(mset3)

# Random parametrisation (first‑order, FOM=3)
ORD3 = 1
FOM3 = 3
W3, R3 = create_parametrisation_method_objects(mset3, ORD3, FOM3, NVAR3, Complex)

# Fill with random coefficients
for idx in 1:nterms
	W3.coefficients[idx] = SVector{1, SVector{3}}(randn(SVector{3, ComplexF64}))
	R3.coefficients[idx] = randn(SVector{NVAR3, ComplexF64})
end

upper_bound3 = SVector{3, Int}(1, 4, 1)
result3 = compute_lower_order_couplings(upper_bound3, W3, R3)
println("Result for random 3‑variable case: $result3")

# ----------------------------------------------------------------------
# 4. Zero polynomials
# ----------------------------------------------------------------------

println("\nExample 4: zero polynomials (should return zero)")

mset4 = all_multiindices_up_to(2, 2)   # up to total degree 2 in 2 variables
W_zero, R_zero = create_parametrisation_method_objects(mset4, 1, 3, 2, Complex)
upper_bound4 = SVector{2, Int}(1, 1)
result4 = compute_lower_order_couplings(upper_bound4, W_zero, R_zero)
println("Result for zero polynomials: $result4")
@assert iszero(result4)

# -------------------------------------------------------------------
println("\n" * "="^80 * "\n")
println("Demo finished successfully.")