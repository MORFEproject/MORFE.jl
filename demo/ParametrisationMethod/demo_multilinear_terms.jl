# demo_multilinear_terms_with_forcing.jl
# ========================================
# 2‑DOF second‑order system (FOM=2, ORD=2)
# Reduced coordinates: internal z₁, z₂ (ROM=2) + external u (FORCING_SIZE=1)
# Multiindex set over NVAR = ROM+FORCING_SIZE = 3 variables, total degree ≤3.
#
# Nonlinear terms (multilinear maps):
#   term1: F₁(x, ẋ) = x ⊙ ẋ               multiindex (1,1), me=0
#   term2: F₂(ẋ)   = 0.5 ẋ ⊙ ẋ            multiindex (0,2), me=0
#   term3: F₃(x,ẋ) = 0.5 x ⊙ x ⊙ ẋ        multiindex (2,1), me=0
#   term4: F₄(x,u) = 2.0 x ⊙ r            multiindex (1,0), me=1
#
# Parametrisation W: maps monomials in (z₁, z₂, r) → (x, ẋ)

# ENV["JULIA_DEBUG"] = "MultilinearTerms" # enable debug for the module

include(joinpath(@__DIR__, "../../src/MORFE.jl"))
using .MORFE.Multiindices: all_multiindices_up_to
using .MORFE.FullOrderModel: NDOrderModel, MultilinearMap
using .MORFE.ParametrisationMethod: create_parametrisation_method_objects
using .MORFE.MultilinearTerms: compute_multilinear_terms
using StaticArrays: SVector
using LinearAlgebra

# Dimensions
const FOM = 2
const ORD = 2
const ROM = 2
const FORCING_SIZE = 1
const NVAR = ROM + FORCING_SIZE   # = 3

# Multilinear terms
term1 = MultilinearMap((res, x, xdot)->(@. res += x*xdot), (1, 1)) # me=0
term2 = MultilinearMap((res, xdot1, xdot2)->(@. res += 0.5*xdot1*xdot2), (0, 2)) # me=0
term3 = MultilinearMap((res, x1, x2, xdot)->(@. res += 0.5*x1*x2*xdot), (2, 1)) # me=0
term4 = MultilinearMap((res, x, r)->(@. res += 2.0*x*r), (1, 0), 1) # me=1
term5 = MultilinearMap((res, r)->(@. res += [100.0, 200.0]*r), (0, 0), 1) # me=1

Id = Matrix{Float64}(I, FOM, FOM)
model = NDOrderModel((Id, Id, Id), (term1, term2, term3, term4, term5))

# Multiindex set over NVAR = 3 variables, total degree ≤ 3
mset = all_multiindices_up_to(NVAR, 3)

# Parametrisation (forcing_size = 0 because u is a coordinate)
W, _ = create_parametrisation_method_objects(mset, ORD, FOM, ROM, FORCING_SIZE, ComplexF64)

# Linear part
λ₁ = -0.1 + 10.0im
λ₂ = conj(λ₁)
φ₁ = ComplexF64[2.0, 3.0]
φ₂ = ComplexF64[4.0, 5.0]
b = ComplexF64[6.0, 7.0]

# Multiindex order for 3 variables (graded lex):
# (0,0,0) idx1
# (1,0,0) idx2   ← z₁
# (0,1,0) idx3   ← z₂
# (0,0,1) idx4   ← r
# (2,0,0) idx5
# (1,1,0) idx6
# (1,0,1) idx7
# (0,2,0) idx8
# (0,1,1) idx9
# (0,0,2) idx10
# ...

W.poly.coefficients[2] = SVector{ORD, Vector{ComplexF64}}(copy(φ₁), λ₁*φ₁)   # (1,0,0)
W.poly.coefficients[3] = SVector{ORD, Vector{ComplexF64}}(copy(φ₂), λ₂*φ₂)   # (0,1,0)
W.poly.coefficients[4] = SVector{ORD, Vector{ComplexF64}}(b, b*1.0im)        # (0,0,1)

# Quadratic coefficients
W.poly.coefficients[5] = SVector{ORD, Vector{ComplexF64}}(0.1*φ₁, b .+ φ₂)   # (2,0,0)
W.poly.coefficients[8] = SVector{ORD, Vector{ComplexF64}}(b .+ φ₂, 0.2*φ₂)   # (0,2,0)
W.poly.coefficients[6] = SVector{ORD, Vector{ComplexF64}}(0.05*φ₁, 0.05*φ₂)     # (1,1,0)
W.poly.coefficients[7] = SVector{ORD, Vector{ComplexF64}}(0.1*φ₁, -0.3*φ₂)       # (1,0,1)

# Cubic coefficients
W.poly.coefficients[11] = SVector{ORD, Vector{ComplexF64}}(500.0*φ₁, -500im*φ₂)  # (3,0,0)

println("\nParametrisation coefficients (first few):")
for idx in 1:10
	println("  $(mset.exponents[idx]) → $(W.poly.coefficients[idx])")
end

# -----------------------------------------------------------------------
# Compute contributions for exponent vectors (length NVAR = 3)
# -----------------------------------------------------------------------
println("\n=== MultilinearTerms demo with NVAR=3 ===\n")

exp100 = SVector(1, 0, 0) # only z₁
println("\nexp = $exp100  (linear in z₁)")
println("result = ", compute_multilinear_terms(model, exp100, W))
# Nonlinear terms cannot have linear contribution (1,0,0)
println("manual: $(ComplexF64[0.0, 0.0])\n")

exp001 = SVector(0, 0, 1) # only forcing contribution r
println("\nexp = $exp001  (linear in r)")
println("result = ", compute_multilinear_terms(model, exp001, W))
# Nonlinear terms cannot have linear contribution (1,0,0)
println("manual: $(ComplexF64[100.0, 200.0])\t(term5)\n")

exp200 = SVector(2, 0, 0) # quadratic in z₁
# decomposition for bilinear term
# = (1,0,0) + (1,0,0)
println("\nexp = $exp200  (quadratic in z₁)")
println("result = ", compute_multilinear_terms(model, exp200, W))
manual200_term1 = λ₁ .* φ₁ .* φ₁
manual200_term2 = 0.5 * λ₁^2 .* φ₁ .* φ₁
manual200 = manual200_term1 + manual200_term2
println("manual = $manual200\t(term1+term2)\n")

exp101 = SVector(1, 0, 1) # mixed z₁ · r
# decomposition for bilinear term
# = (1,0,0) + (0,0,1)
println("exp = $exp101  (mixed z₁·u)")
println("result = ", compute_multilinear_terms(model, exp101, W))
manual101_term1 = (1.0im + λ₁) .* φ₁ .* b
manual101_term2 = (0.5 * 1.0im * λ₁ .* φ₁ .* b) * 2
manual101_term4 = 2.0 * φ₁ * 1.0
manual101 = manual101_term1 + manual101_term2 + manual101_term4
println("manual = $manual101\t(term1+term2+term4)\n")

# exp = (3,0,1) : cubic in z₁, linear on r
exp301 = SVector(3, 0, 1)
# decompositions for bilinear term
# = (3,0,0) + (0,0,1)
# = (2,0,0) + (1,0,1)
# = (1,0,0) + (2,0,1) but (2,0,1) was not defined, so it is zero
# decompositions for trilinear term
# = (2,0,0) + (1,0,0) + (0,0,1)
# = (1,0,0) + (1,0,0) + (1,0,1)
println("exp = $exp301  (cubic in z₁, linear in r)")
println("result = ", compute_multilinear_terms(model, exp301, W))
manual300_term1 =
	((500.0*φ₁) .* (b*1.0im)) + (b .* (-500im*φ₂)) + # (3,0,0) + (0,0,1) and (0,0,1) + (3,0,0)
	((0.1*φ₁) .* (-0.3*φ₂)) + ((0.1*φ₁) .* (b .+ φ₂)) # (2,0,0) + (1,0,1) and (1,0,1) + (2,0,0)
manual300_term2 =
	0.5 * ((-500im*φ₂) .* (b * 1.0im)) * 2 + # (3,0,0) + (0,0,1) with two permutations
	0.5 * ((b .+ φ₂) .* (-0.3*φ₂)) * 2 # (2,0,0) + (1,0,1) with two permutations
manual300_term3 =
	0.5 * ((0.1*φ₁) .* (φ₁) .* (b * 1.0im)) * 2 + # ((2,0,0) + (1,0,0)) + (0,0,1) with two permutations
	0.5 * ((0.1*φ₁) .* (b) .* (λ₁*φ₁)) * 2 + # ((2,0,0) + (0,0,1)) + (1,0,0) with two permutations
	0.5 * ((b) .* (φ₁) .* (b .+ φ₂)) * 2 + # ((0,0,1) + (1,0,0)) + (2,0,0) with two permutations
	0.5 * ((φ₁) .* (φ₁) .* (-0.3*φ₂)) + # ((1,0,0) + (1,0,0)) + (1,0,1) without nontrivial permutations
	0.5 * ((φ₁) .* (0.1*φ₁) .* (λ₁*φ₁)) * 2 # ((1,0,0) + (1,0,1)) + (1,0,0) with two permutations
manual300_term4 =
	2.0 * (500.0*φ₁) .* 1.0 # (3,0,0) + (0,0,1) is the only contribition due to forcing
manual300 = manual300_term1 + manual300_term2 + manual300_term3 + manual300_term4
println("manual = $manual300\t(term1+term2+term3+term4)\n")

println("="^80)
println("Demo finished successfully.")
