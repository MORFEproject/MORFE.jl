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

include(joinpath(@__DIR__, "../src/MORFE.jl"))
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
term2 = MultilinearMap((res, x1, x2)->(@. res += 0.5*x1*x2), (0, 2)) # me=0
term3 = MultilinearMap((res, x1, x2, xdot)->(@. res += 0.5*x1*x2*xdot), (2, 1)) # me=0
term4 = MultilinearMap((res, x, u)->(@. res += 2.0*x*u), (1, 0), 1) # me=1

Id = Matrix{Float64}(I, FOM, FOM)
model = NDOrderModel((Id, Id, Id), (term1, term2, term3, term4))

# Multiindex set over NVAR = 3 variables, total degree ≤ 3
mset = all_multiindices_up_to(NVAR, 3)

# Parametrisation (forcing_size = 0 because u is a coordinate)
W, _ = create_parametrisation_method_objects(mset, ORD, FOM, NVAR, 0, ComplexF64)

# Linear part
λ₁ = -0.1 + 10.0im
λ₂ = conj(λ₁)
φ₁ = ComplexF64[1.0, 1.0]
φ₂ = ComplexF64[2.0, 2.0]
b = ComplexF64[3.0, 3.0]

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
W.poly.coefficients[4] = SVector{ORD, Vector{ComplexF64}}(b, b * 1.0im)      # (0,0,1)

# Quadratic coefficients
W.poly.coefficients[5] = SVector{ORD, Vector{ComplexF64}}(0.1*φ₁, zeros(FOM))   # (2,0,0)
W.poly.coefficients[8] = SVector{ORD, Vector{ComplexF64}}(zeros(FOM), 0.2*φ₂)   # (0,2,0)
W.poly.coefficients[6] = SVector{ORD, Vector{ComplexF64}}(0.05*φ₁, 0.05*φ₂)     # (1,1,0)
W.poly.coefficients[7] = SVector{ORD, Vector{ComplexF64}}(0.1*φ₁, 0.0*φ₂)       # (1,0,1)

println("\nParametrisation coefficients (first few):")
for idx in 1:10
	println("  $(mset.exponents[idx]) → $(W.poly.coefficients[idx])")
end

# -----------------------------------------------------------------------
# Compute contributions for exponent vectors (length NVAR = 3)
# -----------------------------------------------------------------------
println("\n=== MultilinearTerms demo with NVAR=3 ===\n")

# exp = (1,0,0) : only z₁
exp100 = SVector(1, 0, 0)
r100 = compute_multilinear_terms(model, exp100, W)
println("exp = (1,0,0)  (linear in z₁)")
println("  result = $r100")
println("  manual: W₁₀₀ = (φ₁, λ₁φ₁)  →  (φ₁, λ₁φ₁)\n")

# exp = (0,0,1) : only u
exp001 = SVector(0, 0, 1)
r001 = compute_multilinear_terms(model, exp001, W)
println("exp = (0,0,1)  (linear in u)")
println("  result = $r001")
println("  manual: W₀₀₁ = (b, 0)  →  ([1,0], [0,0])\n")

# exp = (2,0,0) : quadratic in z₁
exp200 = SVector(2, 0, 0)
r200 = compute_multilinear_terms(model, exp200, W)
println("exp = (2,0,0)  (quadratic in z₁)")
println("  result = $r200")
manual200 = (λ₁ + 0.5*λ₁^2) .* φ₁
println("  manual (term1+term2) = $manual200\n")

# exp = (1,0,1) : mixed z₁·u
exp101 = SVector(1, 0, 1)
r101 = compute_multilinear_terms(model, exp101, W)
println("exp = (1,0,1)  (mixed z₁·u)")
println("  result = $r101")
# term4: 2.0 * W₁₀₀[1] ⊙ W₀₀₁[1] = 2.0 * φ₁ ⊙ b = 2.0 φ₁
manual101 = 2.0 .* φ₁
println("  manual (term4) = $manual101\n")

# exp = (3,0,0) : cubic in z₁
exp300 = SVector(3, 0, 0)
r300 = compute_multilinear_terms(model, exp300, W)
println("exp = (3,0,0)  (cubic in z₁)")
println("  result = $r300")
manual300 = λ₁ .* φ₁
println("  manual (term3) = $manual300\n")

println("="^80)
println("Demo finished successfully.")
