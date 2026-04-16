include(joinpath(@__DIR__, "../../src/MORFE.jl"))
using .MORFE.MasterModeOrthogonality:
	precompute_orthogonality_operator_coefficients,
	precompute_orthogonality_column_polynomials,
	evaluate_orthogonality_row_and_lower_order_rhs!,
	evaluate_orthogonality_column_row!,
	evaluate_orthogonality_external_rhs,
	assemble_orthogonality_matrix_and_rhs
using LinearAlgebra
using StaticArrays: SVector

# -------------------------------------------------------------------
# 1.  Define problem dimensions and parameters
# -------------------------------------------------------------------
FOM   = 5          # full-order model dimension
ROM   = 2          # number of master modes (reduced coordinates)
N_EXT = 1          # number of external forcing modes (NVAR = ROM + N_EXT)
NVAR  = ROM + N_EXT

ORD   = 2          # polynomial order of the ODE (B matrices: B₀, B₁, B₂)
ORDP1 = ORD + 1   # number of linear matrices

# Linear matrices B₀, B₁, B₂ (each FOM × FOM).
# For a second-order system: B₂ x'' + B₁ x' + B₀ x = ...
B0 = 2.0 * Matrix{ComplexF64}(I, FOM, FOM)
B1 = 3.0 * Matrix{ComplexF64}(I, FOM, FOM)
B2 = 5.0 * Matrix{ComplexF64}(I, FOM, FOM)

fom_matrices = (B0, B1, B2)   # NTuple{ORDP1}: fom_matrices[k+1] = Bₖ

# Eigenvalues of the master modes.
λ₁ = -0.1 + 1.0im
λ₂ = -0.1 - 1.0im
master_eigenvalues = SVector{ROM, ComplexF64}(λ₁, λ₂)

# Left eigenmodes Xℓ_r (each a length-FOM vector).
# For simplicity, use the first two standard basis vectors.
Xℓ₁ = [1.0+0.0im, 0.0, 0.0, 0.0, 0.0]
Xℓ₂ = [0.0+0.0im, 1.0, 0.0, 0.0, 0.0]
left_eigenmodes = hcat(Xℓ₁, Xℓ₂)   # FOM × ROM matrix

# Generalised right eigenvectors Y (FOM × NVAR).
# Columns 1:ROM = master modes, ROM+1:NVAR = external forcing modes.
generalised_right_eigenmodes = Matrix{ComplexF64}(I, FOM, NVAR)

# Reduced dynamics linear part Λ (NVAR × NVAR).
# Eigenvalues: λ₁, λ₂ for master modes; λ_ext = 0+1im for the external mode.
reduced_dynamics_linear = ComplexF64[
	-0.1+1.0im  0.0+0.0im  1.0+0.0im;   # coupling: forcing excites master mode 1
	 0.0+0.0im -0.1-1.0im  0.0+0.0im;   # no coupling to master mode 2
	 0.0+0.0im  0.0+0.0im  0.0+1.0im    # external forcing frequency
]

println("Reduced dynamics Λ (NVAR × NVAR):")
display(reduced_dynamics_linear)

# -------------------------------------------------------------------
# 2.  Precompute J_coeffs: row operator coefficients L_r(s)
# -------------------------------------------------------------------
# J_coeffs is Vector{Matrix{ComplexF64}}.
# J_coeffs[r] is ORD × FOM; row j stores the degree-(j-1) coefficient of L_r(s).
#
# Recurrence (ORD = 2):
#   J_r[2, :] = B₂ᵀ · Xℓ_r
#   J_r[1, :] = λ_r · J_r[2, :] + B₁ᵀ · Xℓ_r
#
# => L_r(s) = J_r[1,:] + J_r[2,:] · s
#           = Xℓ_rᵀ · (B₂·(s + λ_r) + B₁)   (row vector)
J_coeffs = precompute_orthogonality_operator_coefficients(
	fom_matrices, left_eigenmodes, master_eigenvalues,
)

println("\n=== Precomputed row operator coefficients J_coeffs ===")
for r in 1:ROM
	println("\nJ_coeffs[$r] (ORD × FOM):")
	display(J_coeffs[r])
end

# Manual verification for J_coeffs.
println("\nManual J_coeffs verification")
for r in 1:ROM
	Xℓ = left_eigenmodes[:, r]
	λ = master_eigenvalues[r]
	J2_manual = B2' * Xℓ                    # degree-1 coefficient
	J1_manual = B2' * Xℓ .* λ .+ B1' * Xℓ   # degree-0 coefficient
	println("Mode $r  |  J_$r[2,:] error = ", norm(J_coeffs[r][2, :] - J2_manual))
	println("Mode $r  |  J_$r[1,:] error = ", norm(J_coeffs[r][1, :] - J1_manual))
end

# -------------------------------------------------------------------
# 3.  Precompute C_coeffs and E_coeffs: joint operator D_r(s) columns
# -------------------------------------------------------------------
# C_coeffs[r] is (ORD-1) × ROM;   C_r(s) = Σ_{j=1}^{ORD-1} C_coeffs[r][j,:] · s^{j-1}
# E_coeffs[r] is (ORD-1) × N_EXT; E_r(s) = Σ_{j=1}^{ORD-1} E_coeffs[r][j,:] · s^{j-1}
#
# For ORD = 2, ORD-1 = 1: both polynomials are constants (degree 0):
#   Q_r[1] = Yᵀ · J_r[2, :] = Yᵀ · B₂ᵀ · Xℓ_r
#   C_coeffs[r][1, :] = Q_r[1][1:ROM]
#   E_coeffs[r][1, :] = Q_r[1][ROM+1:NVAR]
C_coeffs, E_coeffs = precompute_orthogonality_column_polynomials(
	J_coeffs, generalised_right_eigenmodes, reduced_dynamics_linear,
)

println("\n=== Precomputed joint operator coefficients ===")
for r in 1:ROM
	println("\nC_coeffs[$r] ((ORD-1) × ROM):")
	display(C_coeffs[r])
	println("\nE_coeffs[$r] ((ORD-1) × N_EXT):")
	display(E_coeffs[r])
end

# Manual verification.
println("\n--- Manual verification ---")
Y = generalised_right_eigenmodes
for r in 1:ROM
	Xℓ = left_eigenmodes[:, r]
	Q1_manual = Y' * (B2' * Xℓ)    # = (Xℓᵀ · B₂ · Y)ᵀ, i.e. stored as column
	C1_manual = Q1_manual[1:ROM]
	E1_manual = Q1_manual[(ROM+1):NVAR]
	println("Mode $r  |  C_coeffs[r][1,:] error = ", norm(C_coeffs[r][1, :] - C1_manual))
	println("Mode $r  |  E_coeffs[r][1,:] error = ", norm(E_coeffs[r][1, :] - E1_manual))
end

# -------------------------------------------------------------------
# 4.  Data for a specific multi-index
# -------------------------------------------------------------------
# Superharmonic s = Σᵢ kᵢ λᵢ
s = -2.0 + 0.0im

# Resonance pattern: only master mode 1 is resonant.
resonance = SVector{ROM, Bool}(true, true)
nR = count(resonance)

# Lower‑order couplings ξ_j (j = 1…ORD) – each is a FOM‑vector
# In a real simulation these come from lower‑order multiindices.
lower_order_couplings = SVector{ORD, Vector{ComplexF64}}(
	[0.1, 0.1, 0.1, 0.1, 0.1],       # position
	[10.0, 10.0, 10.0, 10.0, 10.0],  # velocity
)

# External dynamics amplitudes (length N_EXT).
external_dynamics = ComplexF64[-1000.0]

# -------------------------------------------------------------------
# 5.  Assemble the orthogonality matrix and right-hand side
# -------------------------------------------------------------------
# M   : nR × (FOM + nR)  with  M = [ L | C ]
# rhs : length nR
M, rhs = assemble_orthogonality_matrix_and_rhs(
	s, J_coeffs, C_coeffs, E_coeffs,
	resonance, lower_order_couplings, external_dynamics,
)

println("\n=== Assembled orthogonality system ===")
println("\nSystem matrix M (nR × (FOM + nR)) = ($nR × $(FOM + nR)):")
display(M)
println("\nRight-hand side rhs (length nR = $nR):")
display(rhs)

println("\n--- Manual verification ---")
M_manual = zeros(ComplexF64, nR, FOM + nR)
rhs_manual = zeros(ComplexF64, nR)
row_count = 1
for r in 1:ROM
	if resonance[r]
		aux = B2' * left_eigenmodes[:, r]
		M_manual[row_count, 1:FOM] = aux * (s + master_eigenvalues[r]) .+ (B1' * left_eigenmodes[:, r])

		for rr in 1:ROM
			if resonance[rr]
				M_manual[row_count, FOM+rr] = dot(generalised_right_eigenmodes[:, rr], aux)
			end
		end

		rhs_manual[row_count] = - lower_order_couplings[1]' * (B2' * left_eigenmodes[:, r])   # only j=1 coupling for ORD=2
		global row_count += 1
	end
end
println("System matrix error:", norm(M - M_manual))
println("RHS error:", norm(rhs - rhs_manual))

# -------------------------------------------------------------------
# 6.  Low-level function demonstrations
# -------------------------------------------------------------------
println("\n=== Low-level function demonstrations ===")

# 6a.  Fused L_r(s) row + scalar lower-order RHS for mode r = 1.
row1 = zeros(ComplexF64, FOM)
rhs_lower_1 = evaluate_orthogonality_row_and_lower_order_rhs!(
	row1, s, lower_order_couplings, J_coeffs[1],
)
println("\nL₁($s) row vector:")
display(row1)
println("Lower-order RHS contribution for mode 1: ", rhs_lower_1)

# 6b.  Resonant C_r(s) block for mode r = 1.
c1 = zeros(ComplexF64, nR)
evaluate_orthogonality_column_row!(c1, s, 1, C_coeffs, resonance)
println("\nC₁($s) restricted to resonant modes: ", c1)

# 6c.  Scalar external-forcing RHS for mode r = 1.
rhs_ext_1 = evaluate_orthogonality_external_rhs(s, 1, external_dynamics, E_coeffs)
println("External RHS contribution for mode 1: ", rhs_ext_1)

println("\n--- Manual verification ---")

# For ORD = 2:
#   L_r(s) = Xℓ_rᵀ · (B₂·(s + λ_r) + B₁)   (row vector, length FOM)
for r in 1:ROM
	Xℓ = left_eigenmodes[:, r]
	λ = master_eigenvalues[r]
	L_r_manual = (B2 .* (s + λ) .+ B1)' * Xℓ   # Xℓ_rᵀ · (B₂(s+λ_r) + B₁)
	println("L_$r($s) error = ", norm(J_coeffs[r]' * [1.0; s] - L_r_manual))
	#   row * [1; s] evaluates J_r[1,:] + J_r[2,:]*s = L_r(s)
end

# L_1(s) row reconstructed from J_coeffs.
L1_from_J = J_coeffs[1][1, :] .+ J_coeffs[1][2, :] .* s
println("\nL₁($s) from J_coeffs vs. row buffer error = ", norm(row1 - L1_from_J))

# Lower-order RHS for mode 1 (ORD-1 = 1 coupling only):
#   RHS_lower_1 = -J_r[2,:] · ξ₁ = -(B₂ᵀ·Xℓ₁)ᵀ · ξ₁
rhs_lower_1_manual = -dot(J_coeffs[1][2, :], lower_order_couplings[1])
println("\nRHS_lower_1 error = ", abs(rhs_lower_1 - rhs_lower_1_manual))

# Resonant C_r(s) for mode 1: constant (ORD-1 = 1), picks resonant columns.
# C_coeffs[1][1, :] = Q_1[1][1:ROM]; resonance = [true, false] → take column 1.
c1_manual = ComplexF64[C_coeffs[1][1, j] for j in eachindex(resonance) if resonance[j]]
println("\nC₁($s) resonant error = ", norm(c1 - c1_manual))

# External RHS for mode 1 (constant polynomial, ORD-1 = 1):
#   RHS_ext_1 = -E_coeffs[1][1, e] · external_dynamics[e]  for e = 1
rhs_ext_1_manual = -sum(E_coeffs[1][1, e] * external_dynamics[e] for e in 1:N_EXT)
println("\nRHS_ext_1 error = ", abs(rhs_ext_1 - rhs_ext_1_manual))

# Full RHS for mode 1 vs. assembled rhs[1].
rhs_1_manual = rhs_lower_1_manual + rhs_ext_1_manual
println("\nrhs[1] vs. manual error = ", abs(rhs[1] - rhs_1_manual))

# First FOM columns of M[1, :] must equal L₁(s).
println("M[1, 1:FOM] vs. L₁($s) error = ", norm(M[1, 1:FOM] - row1))

# Last nR columns of M[1, :] must equal the resonant C₁(s) block.
println("M[1, FOM+1:end] vs. C₁($s) error = ", norm(M[1, (FOM+1):end] - c1))

println("\n" * "="^80)

# ===================================================================
# 7.  Random parametrisation and reduced dynamics
# ===================================================================
println("\n=== Random parametrisation and reduced dynamics, full check ===")

using .MORFE.Multiindices: all_multiindices_up_to
using .MORFE.ParametrisationMethod: create_parametrisation_method_objects, compute_higher_derivative_coefficients!
using .MORFE.LowerOrderCouplings: compute_lower_order_couplings

NVAR7 = 3
maxdeg7 = 5
mset7 = all_multiindices_up_to(NVAR7, NVAR7*maxdeg7)
nterms = length(mset7)

# Random parametrisation (first‑order, FOM=3, ROM=NVAR7)
ORD3 = 1
FOM3 = 3
W7, R7 = create_parametrisation_method_objects(mset7, ORD3, FOM3, NVAR7, 0, ComplexF64)

# Fill with random coefficients
for idx in 1:nterms
	W7.poly.coefficients[:, 1, idx] = randn(ComplexF64, FOM3)
	R7.poly.coefficients[:, idx]    = randn(ComplexF64, NVAR7)
end

exp = rand(0:maxdeg7, NVAR7)
superharmonic = rand() + im*rand()
low_order_couplings = compute_lower_order_couplings(exp, W7, R7)
global_index = rand(1:nterms)
compute_higher_derivative_coefficients!(
	W7.poly.coefficients, R7.poly.coefficients, superharmonic, global_index,
	generalised_eigenmodes, low_order_couplings,
)




println("Result for random 3‑variable case: $result3")


println("\n" * "="^80)
println("Demo finished successfully.")
