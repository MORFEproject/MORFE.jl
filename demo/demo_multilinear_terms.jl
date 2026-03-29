# demo_multilinear_terms.jl
# Exercises MultilinearTerms.compute_multilinear_terms on a 2-DOF
# second-order system with a linear (eigenmode) parametrisation.

include(joinpath(@__DIR__, "../src/MORFE.jl"))
using .MORFE.Multiindices: MultiindexSet
using .MORFE.FullOrderModel: NDOrderModel, MultilinearMap
using .MORFE.ParametrisationMethod: create_parametrisation_method_objects
using .MORFE.MultilinearTerms: compute_multilinear_terms
using StaticArrays: SVector
using LinearAlgebra

# -----------------------------------------------------------------------
# Dimensions
# -----------------------------------------------------------------------
const FOM = 2   # full-order model dimension
const ROM = 2   # reduced-manifold dimension (two reduced coordinates, no forcing)
const ORD = 2   # native ODE order:  B₂ẍ + B₁ẋ + B₀x = F(x, ẋ)

# -----------------------------------------------------------------------
# Nonlinear terms of the full-order model
# -----------------------------------------------------------------------

# Term 1 — FullyAsymmetric (multiindex (1,1)):  F₁(x, ẋ) = x .* ẋ
function mixed_bilinear!(res, x, xdot)
	@. res += x * xdot
end
term1 = MultilinearMap(mixed_bilinear!, (1, 1))

# Term 2 — FullySymmetric  (multiindex (0,2)):  F₂(ẋ₁, ẋ₂) = 0.5 ẋ₁ .* ẋ₂
function vel_squared!(res, xdot1, xdot2)
	@. res += 0.5 * xdot1 * xdot2
end
term2 = MultilinearMap(vel_squared!, (0, 2))

# Term 3 — GroupwiseSymmetric (multiindex (2,1)):  F₃(x₁, x₂, ẋ) = 0.5 x₁ .* x₂ .* ẋ
function cubic_damping!(res, x1, x2, xdot)
	@. res += 0.5 * x1 * x2 * xdot
end
term3 = MultilinearMap(cubic_damping!, (2, 1))

# Build the second-order model (linear matrices are trivial here — not used by MultilinearTerms)
Id = Matrix{Float64}(I, FOM, FOM)
model = NDOrderModel((Id, Id, Id), (term1, term2, term3))

# -----------------------------------------------------------------------
# Multiindex set: all monomials z₁ᵃ z₂ᵇ with a + b ≤ 3
# -----------------------------------------------------------------------
monomials = [SVector(a, b) for d in 0:3 for a in d:-1:0 for b in (d - a,)]
# (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)
mset = MultiindexSet(monomials)

# -----------------------------------------------------------------------
# Create a zero-initialised parametrisation W and fill in the linear part.
#
#   Two eigenmodes with eigenvalues  λ₁ = -0.1 + 2i,  λ₂ = -0.1 - 2i
#   (complex-conjugate pair of a lightly damped oscillator).
#   Mode shapes: φ₁ = e₁ = [1,0],  φ₂ = e₂ = [0,1]  (identity basis).
#
#   The parametrisation maps (z₁, z₂) ↦ (x, ẋ) where, at linear order:
#       W[(1,0)] = ( φ₁,  λ₁ φ₁ )   [x-part, ẋ-part for mode 1]
#       W[(0,1)] = ( φ₂,  λ₂ φ₂ )
# -----------------------------------------------------------------------
W, _ = create_parametrisation_method_objects(mset, ORD, FOM, ROM, 0, ComplexF64)

λ₁ = -0.1 + 2.0im
λ₂ = conj(λ₁)
φ₁ = ComplexF64[1.0, 0.0]
φ₂ = ComplexF64[0.0, 1.0]

for (i, mono) in enumerate(monomials)
	if mono == SVector(1, 0)
		W.poly.coefficients[i] = SVector{ORD, Vector{ComplexF64}}(copy(φ₁), λ₁ .* φ₁)
	elseif mono == SVector(0, 1)
		W.poly.coefficients[i] = SVector{ORD, Vector{ComplexF64}}(copy(φ₂), λ₂ .* φ₂)
	end
end

# -----------------------------------------------------------------------
# Compute multilinear-term contributions at selected exponents
# -----------------------------------------------------------------------
println("=== MultilinearTerms demo ===\n")

# exp = (1,0): linear exponent — all terms have degree ≥ 2, so result is zero.
exp10 = SVector(1, 0)
r10 = compute_multilinear_terms(model, exp10, W)
println("exp = (1,0)  [linear level, no nonlinear contribution]")
println("  result = ", r10, "\n")

# exp = (2,0): quadratic — term1 and term2 contribute; term3 has degree 3.
exp20 = SVector(2, 0)
r20 = compute_multilinear_terms(model, exp20, W)
println("exp = (2,0)  [quadratic level, term1 + term2]")
println("  result = ", r20)

# ---- Manual verification at exp = (2,0) --------------------------------
# With a linear W only W[(1,0)] and W[(0,1)] are non-zero.
# The only factorisation of (2,0) into two non-zero reduced-coord multiindices
# is (1,0)+(1,0).
#
# term1 (asymmetric, orders = (1,2) → slot 1 from x-part, slot 2 from ẋ-part):
#   F₁( W[(1,0)][1], W[(1,0)][2] ) = φ₁ .* (λ₁ φ₁) = λ₁ * [1, 0]
#
# term2 (symmetric, deriv_idx=2 → both slots from ẋ-part, sym_count=1):
#   0.5 * 1 * F₂( W[(1,0)][2], W[(1,0)][2] )
#     = 0.5 * (λ₁ φ₁) .* (λ₁ φ₁) = 0.5 λ₁² * [1, 0]
#
# total = (λ₁ + 0.5 λ₁²) * [1, 0]
manual20 = (λ₁ + 0.5 * λ₁^2) .* φ₁
println("  manual check (term1 + term2) = ", manual20, "\n")

# exp = (1,1): mixed quadratic — term1 and term2 contribute.
exp11 = SVector(1, 1)
r11 = compute_multilinear_terms(model, exp11, W)
println("exp = (1,1)  [mixed quadratic, term1 + term2]")
println("  result = ", r11, "\n")

# exp = (3,0): cubic — term3 (GroupwiseSymmetric) contributes.
exp30 = SVector(3, 0)
r30 = compute_multilinear_terms(model, exp30, W)
println("exp = (3,0)  [cubic level, term3]")
println("  result = ", r30)

# Manual for term3 at (3,0):
# Factorisation of (3,0) into (x₁, x₂, ẋ) with groupwise symmetry (multiindex (2,1)):
#   two x-slots from mode 1, one ẋ-slot from mode 1
#   total_count accounts for the within-group permutation of the two x-slots
#   result ≈ 0.5 * total_count * F₃(φ₁, φ₁, λ₁φ₁)
#           = 0.5 * total_count * φ₁ .* φ₁ .* (λ₁φ₁)
#           = 0.5 * total_count * λ₁ * [1, 0]
println("\n  (term3 manual: 0.5 * total_count * λ₁ * [1,0],")
println("   where total_count from factorisations_groupwise_symmetric)")

println("\n" * "="^80 * "\n")
println("Demo finished successfully.")
