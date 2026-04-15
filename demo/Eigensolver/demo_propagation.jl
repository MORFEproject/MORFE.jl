include(joinpath(@__DIR__, "../../src/MORFE.jl"))

using LinearAlgebra
using Printf
using Random
using Statistics
using StaticArrays: SVector

using .MORFE
using .MORFE.EigenModesPropagation
using .MORFE.Eigensolvers: generalised_eigenpairs
using .MORFE.ParametrisationMethod: Parametrisation

# -------------------------------------------------------------------
# 1. Setup: random matrices of non-trivial sise
# -------------------------------------------------------------------
FOM = 20
ORD = 3
n_total = ORD * FOM
println("System: FOM=$FOM, ORD=$ORD  →  first-order sise $(n_total)×$(n_total)")

# Random SPD linear matrices for a well-conditioned problem
rng = MersenneTwister(42)
function random_spd(n, rng)
	G = randn(rng, n, n)
	return G' * G + n * I
end

linear_terms = ntuple(_ -> random_spd(FOM, rng), ORD + 1)
model = NDOrderModel(linear_terms)

# First-order companion matrices  B Ẋ = A X
A, B = linear_first_order_matrices(model)

# -------------------------------------------------------------------
# 2. Eigenpairs via Eigensolvers module (ARPACK-based)
# -------------------------------------------------------------------
nev = n_total ÷ 3   # compute a third of the spectrum (ncv >> 2*nev is satisfied)

println("\nRight eigenpairs  A y = λ B y  (nev=$nev) ...")
right_eig = generalised_eigenpairs(A, B; nev = nev, which = :LR)
λ = right_eig.values
Y = right_eig.vectors
println("  converged: $(right_eig.nconv)/$(nev)")

println("Left  eigenpairs  A' x = μ B' x  (nev=$nev) ...")
left_eig = generalised_eigenpairs(A', B'; nev = nev, which = :LR)
μ = left_eig.values
X = left_eig.vectors
println("  converged: $(left_eig.nconv)/$(nev)")

# -------------------------------------------------------------------
# 3. Propagate: fill the full ND-order eigenvectors
#
#   Right: given y[1:FOM], use  y[l+1] = λ · y[l]  to recover the
#          full companion-space vector y ∈ ℂ^{ORD·FOM}.
#
#   Left:  given x[end-FOM+1:end], use the adjoint recurrence
#          x[i-1]^H = λ·x[i]^H + x_last^H · B[i+1]  to recover x.
# -------------------------------------------------------------------
mset = all_multiindices_up_to(nev, 1)
(param, _) = create_parametrisation_method_objects(mset, ORD, FOM)
param_coeff = param.poly.coefficients
left_eigenvectors = zeros(ComplexF64, FOM, ORD, nev)

for i in 1:nev
	propagate_right_eigenvector_form_first(param, Y[1:FOM, i], λ[i], i)
	propagate_left_eigenvector_from_last(
		model, left_eigenvectors, X[(FOM*(ORD-1)+1):end, i], conj(μ[i]), i)
end

# -------------------------------------------------------------------
# 4. Validation: eigenproblem residuals on the propagated vectors
#
#   Right:  B Y λ = A Y   ⟺   A y_i - λ_i B y_i = 0
#   Left:   λ X^† B = X^† A   ⟺   x_i^† A - conj(μ_i) x_i^† B = 0
#
# Note: generalised_eigenpairs(A', B') solves  A^T x = μ B^T x.
# Taking complex conjugate of both sides gives x' * A = conj(μ) * x' * B.
# -------------------------------------------------------------------
right_res = Vector{Float64}(undef, nev)
left_res  = Vector{Float64}(undef, nev)

for i in 1:nev
	# Right eigenproblem residual  ||A y_i - λ_i B y_i|| / ||y_i||
	y_i = vec(param_coeff[:, :, i])
	right_res[i] = norm(A * y_i - λ[i] * (B * y_i)) / norm(y_i)

	# Left eigenproblem residual  ||x_i' A - conj(μ_i) x_i' B|| / ||x_i||
	x_i = vec(left_eigenvectors[:, :, i])
	left_res[i] = norm(x_i' * A - conj(μ[i]) * (x_i' * B)) / norm(x_i)
end

println("\n" * "="^70)
println("Right eigenproblem  B Y λ = A Y")
println("  relative residual  ||A y_i - λ_i B y_i|| / ||y_i||")
@printf "    max  = %.2e\n" maximum(right_res)
@printf "    mean = %.2e\n" mean(right_res)

println("\nLeft eigenproblem  λ X^† B = X^† A")
println("  relative residual  ||x_i' A - conj(μ_i) x_i' B|| / ||x_i||")
@printf "    max  = %.2e\n" maximum(left_res)
@printf "    mean = %.2e\n" mean(left_res)
println("="^70)
println("Demo finished successfully.")
