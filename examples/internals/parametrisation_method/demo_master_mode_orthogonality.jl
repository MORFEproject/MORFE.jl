using MORFE.MasterModeOrthogonality:
	precompute_orthogonality_operator_coefficients,
	precompute_orthogonality_column_polynomials,
	evaluate_orthogonality_row_and_lower_order_rhs!,
	evaluate_orthogonality_column_row!,
	evaluate_orthogonality_external_rhs,
	assemble_orthogonality_matrix_and_rhs!
using LinearAlgebra
using Random
using StaticArrays: SVector

# -------------------------------------------------------------------
# 1.  A genuine complex second-order pencil and its eigenvectors
# -------------------------------------------------------------------
# The orthogonality condition at a monomial with superharmonic s is the
# sesquilinear B-orthogonality of the companion LEFT eigenvector φ against
# the first-order state 𝒲:
#
#     φᴴ B 𝒲 = 0,   𝒲 = [W₁; …; W_ORD],  W₁ = W,  W_{j+1} = s W_j + Y_j f + ξ_j
#
# Everything below is checked against that identity — no eigenvalue enters
# the precomputation; the row coefficients are the (conjugated) left
# eigenvector order-blocks.
Random.seed!(42)

FOM = 5          # full-order model dimension
ROM = 2          # number of master modes
N_EXT = 1          # number of external forcing modes
NVAR = ROM + N_EXT

ORD = 2          # ODE order (B matrices: B₀, B₁, B₂)
ORDP1 = ORD + 1

# COMPLEX matrices: a real system would hide the sesquilinear conjugation.
B0 = randn(ComplexF64, FOM, FOM)
B1 = randn(ComplexF64, FOM, FOM)
B2 = randn(ComplexF64, FOM, FOM)
fom_matrices = (B0, B1, B2)   # fom_matrices[k+1] = Bₖ

# Companion pencil (A, B) as in linear_first_order_matrices.
A_comp = [zeros(ComplexF64, FOM, FOM) I; -B0 -B1]
B_comp = [Matrix{ComplexF64}(I, FOM, FOM) zeros(ComplexF64, FOM, FOM);
		  zeros(ComplexF64, FOM, FOM) B2]

# Right eigenpairs (λ, ψ):  (λB − A) ψ = 0.
FR = eigen(A_comp, B_comp)
# Left eigenvectors (solve_left convention): eigen of (Aᴴ, Bᴴ) has pencil
# eigenvalue ν with the reported eigenvalue conj(ν); the eigenvector φ solves
# φᴴ (λB − A) = 0 for λ = conj(ν).
FL = eigen(A_comp', B_comp')

# Two master modes: pick two well-separated finite eigenvalues.
finite = findall(x -> isfinite(x) && abs(x) < 1e6, FR.values)
master_idx = finite[1:ROM]
master_eigenvalues = SVector{ROM, ComplexF64}(FR.values[master_idx]...)

right_blocks = zeros(ComplexF64, FOM, ORD, ROM)   # Y_k blocks (position, velocity)
left_blocks = zeros(ComplexF64, FOM, ORD, ROM)    # φ_j blocks
for (r, i) in enumerate(master_idx)
	right_blocks[:, :, r] .= reshape(FR.vectors[:, i], FOM, ORD)
	j = argmin(abs.(conj.(FL.values) .- FR.values[i]))
	left_blocks[:, :, r] .= reshape(FL.vectors[:, j], FOM, ORD)
end
master_modes = right_blocks[:, 1, :]              # physical right slices ψ
left_eigenmodes = left_blocks[:, ORD, :]          # physical left slices ℓ = φ_ORD

println("Master eigenvalues: ", collect(master_eigenvalues))

# External forcing mode: physical direction Φ_ext, eigenvalue iΩ, coupled to
# master mode 1 through Λ.
Φ_ext = randn(ComplexF64, FOM, N_EXT)
λ_ext = 0.0 + 1.0im
reduced_dynamics_linear = ComplexF64[
	master_eigenvalues[1] 0 1;   # forcing excites master mode 1
	0 master_eigenvalues[2] 0;
	0 0 λ_ext]

# -------------------------------------------------------------------
# 2.  J_coeffs: row coefficients read off the left eigenvector blocks
# -------------------------------------------------------------------
#   J_r[j, :]   = conj(φ_{r,j})          j = 1 … ORD-1
#   J_r[ORD, :] = conj(B_ORDᴴ φ_{r,ORD})
# No eigenvalue is used.
J_coeffs = precompute_orthogonality_operator_coefficients(
	fom_matrices, left_eigenmodes, left_blocks[:, 1:(ORD-1), :],
)

println("\n=== J_coeffs from left eigenvector order-blocks ===")
for r in 1:ROM
	println("\nJ_coeffs[$r] (ORD × FOM):")
	display(J_coeffs[r])
end

# Ground truth: the assembled bilinear row J_r(s)·W must equal φ_rᴴ B 𝒲(W)
# for the pure-W lift 𝒲 = [W; sW].
println("\nGround-truth check  J_r(s)ᵀ W  =  φ_rᴴ B 𝒲(W):")
for r in 1:ROM
	φ = vec(left_blocks[:, :, r])
	for s in (0.3 + 0.5im, master_eigenvalues[r], -1.2 + 0.1im)
		W = randn(ComplexF64, FOM)
		row = J_coeffs[r][1, :] .+ J_coeffs[r][2, :] .* s
		lift = vcat(W, s .* W)
		err = abs(transpose(row) * W - φ' * B_comp * lift)
		println("  mode $r, s = $(round(s, digits = 3)):  error = $err")
	end
end

# -------------------------------------------------------------------
# 3.  C_coeffs / E_coeffs: couplings from the RIGHT eigenvector blocks
# -------------------------------------------------------------------
#   C_r(s) = Σ_k G_{r,k}(s) · Y_k^m      (right master blocks)
#   E_r(s) = Σ_k G_{r,k}(s) · Y_k^e      (external blocks via Λ recurrence)
C_coeffs, E_coeffs = precompute_orthogonality_column_polynomials(
	J_coeffs, right_blocks, Φ_ext, reduced_dynamics_linear,
)

println("\n=== C_coeffs / E_coeffs from right eigenvector blocks ===")
for r in 1:ROM
	println("\nC_coeffs[$r] ((ORD-1) × ROM):")
	display(C_coeffs[r])
	println("E_coeffs[$r] ((ORD-1) × N_EXT):")
	display(E_coeffs[r])
end

# Manual check (ORD = 2 → constants): C_r[1, m] = J_r[2, :] · Y₁^m  (bilinear).
println("\nManual verification (bilinear contraction):")
for r in 1:ROM
	C_manual = [transpose(J_coeffs[r][2, :]) * right_blocks[:, 1, m] for m in 1:ROM]
	E_manual = [transpose(J_coeffs[r][2, :]) * Φ_ext[:, e] for e in 1:N_EXT]
	println("  mode $r  |  C error = ", norm(C_coeffs[r][1, :] - C_manual),
		"  |  E error = ", norm(E_coeffs[r][1, :] - E_manual))
end

# -------------------------------------------------------------------
# 4.  Full orthogonality identity  φᴴ B 𝒲 = J·W + C·f_m + E·f_e + Σ G_k·ξ_k
# -------------------------------------------------------------------
s = master_eigenvalues[1]                     # exact resonance
resonance = SVector{ROM, Bool}(true, true)
nR = count(resonance)

W = randn(ComplexF64, FOM)
f = randn(ComplexF64, NVAR)                   # f = [f_m; f_e]
ξ = [randn(ComplexF64, FOM) for _ in 1:(ORD-1)]
lower_order_couplings = SVector{ORD, Vector{ComplexF64}}(ξ[1], zeros(ComplexF64, FOM))

println("\n=== Full identity check at exact resonance s = λ₁ ===")
for r in 1:ROM
	φ = vec(left_blocks[:, :, r])
	# True first-order state from the recurrence W_{j+1} = s W_j + Y_j f + ξ_j.
	Y1 = hcat(right_blocks[:, 1, :], Φ_ext)   # physical blocks of all NVAR modes
	W2 = s .* W .+ Y1 * f .+ ξ[1]
	truth = φ' * B_comp * vcat(W, W2)

	row = J_coeffs[r][1, :] .+ J_coeffs[r][2, :] .* s
	Cf = sum(C_coeffs[r][1, m] * f[m] for m in 1:ROM)
	Ef = sum(E_coeffs[r][1, e] * f[ROM+e] for e in 1:N_EXT)
	Gξ = transpose(J_coeffs[r][2, :]) * ξ[1]
	recon = transpose(row) * W + Cf + Ef + Gξ
	println("  mode $r:  |φᴴB𝒲 − (J·W + C·f_m + E·f_e + G·ξ)| = ", abs(truth - recon))
end

# -------------------------------------------------------------------
# 5.  Assembled system and low-level evaluators
# -------------------------------------------------------------------
external_dynamics = ComplexF64[f[ROM+e] for e in 1:N_EXT]

M = Matrix{ComplexF64}(undef, nR, FOM + nR)
rhs = zeros(ComplexF64, nR)
assemble_orthogonality_matrix_and_rhs!(
	M, rhs, s, J_coeffs, C_coeffs, E_coeffs,
	resonance, lower_order_couplings, external_dynamics,
)

println("\n=== Assembled orthogonality system ===")
println("System matrix M ($nR × $(FOM + nR)):")
display(M)
println("\nRight-hand side (length $nR):")
display(rhs)

# The assembled equation  M · [W; f_res] = rhs  must be equivalent to the
# identity of section 4 with φᴴB𝒲 = 0 moved around:
#   J·W + C·f_res = −E·f_e − Σ G_k·ξ_k
println("\nAssembly consistency (row equation vs identity):")
for (row_i, r) in enumerate(findall(collect(resonance)))
	lhs = transpose(M[row_i, 1:FOM]) * W +
		  sum(M[row_i, FOM+m] * f[m] for m in 1:nR)
	φ = vec(left_blocks[:, :, r])
	Y1 = hcat(right_blocks[:, 1, :], Φ_ext)
	W2 = s .* W .+ Y1 * f .+ ξ[1]
	truth = φ' * B_comp * vcat(W, W2)
	# lhs − rhs must equal φᴴB𝒲 (both express the same functional)
	println("  row $row_i:  |(M·[W;f] − rhs) − φᴴB𝒲| = ", abs(lhs - rhs[row_i] - truth))
end

# Low-level evaluators.
row1 = zeros(ComplexF64, FOM)
rhs_lower_1 = evaluate_orthogonality_row_and_lower_order_rhs!(
	row1, s, lower_order_couplings, J_coeffs[1],
)
println("\nL₁(s) row (fused Horner):")
display(row1)
println("Lower-order RHS for mode 1: ", rhs_lower_1)
println("Bilinear check: ", abs(rhs_lower_1 -
								(-transpose(J_coeffs[1][2, :]) * ξ[1])))

c1 = zeros(ComplexF64, nR)
evaluate_orthogonality_column_row!(c1, s, 1, C_coeffs, resonance)
println("\nC₁(s) resonant block: ", c1)

rhs_ext_1 = evaluate_orthogonality_external_rhs(s, 1, external_dynamics, E_coeffs)
println("External RHS for mode 1: ", rhs_ext_1)

println("\n" * "="^80)

# ===================================================================
# 7.  Random parametrisation and reduced dynamics
# ===================================================================
println("\n=== Random parametrisation and reduced dynamics, full check ===")

using MORFE.Multiindices: all_multiindices_up_to
using MORFE.ParametrisationMethod: create_parametrisation_method_objects,
	compute_higher_derivative_coefficients!
using MORFE.LowerOrderCouplings: compute_lower_order_couplings

NVAR7 = 3
maxdeg7 = 9
mset7 = all_multiindices_up_to(NVAR7, maxdeg7)
nterms7 = length(mset7)
FOM7 = 3
ORD7 = 2      # second-order system so compute_higher_derivative_coefficients! does real work
N_EXT7 = 0      # no external forcing

W7, R7 = create_parametrisation_method_objects(mset7, ORD7, FOM7, NVAR7, N_EXT7, ComplexF64)

# Fill with random coefficients (both derivative orders)
for idx in 1:nterms7
	for ord in 1:ORD7
		W7.poly.coefficients[:, ord, idx] = randn(ComplexF64, FOM7)
	end
	R7.poly.coefficients[:, idx] = randn(ComplexF64, NVAR7)
end

# Pick the first monomial with total degree ≥ 2 so lower-order couplings are non-trivial
idx7 = rand((NVAR7+2):length(mset7))
upper_bound7 = mset7[idx7]
superharmonic7 = rand(ComplexF64)

low_order_couplings7 = compute_lower_order_couplings(upper_bound7, W7, R7)
println("Lower-order couplings for monomial $upper_bound7:")
for (k, v) in enumerate(low_order_couplings7)
	println("  order $k: $v")
end

# compute_higher_derivative_coefficients! updates W7 in-place
generalised_eigenmodes7 = Matrix{ComplexF64}(I, FOM7, NVAR7)
external_dynamics7 = zeros(ComplexF64, N_EXT7)

compute_higher_derivative_coefficients!(
	W7.poly.coefficients, R7.poly.coefficients,
	external_dynamics7, superharmonic7, idx7,
	generalised_eigenmodes7, low_order_couplings7,
)
println("compute_higher_derivative_coefficients! completed without error.")

println("\n" * "="^80)
println("Demo finished successfully.")
