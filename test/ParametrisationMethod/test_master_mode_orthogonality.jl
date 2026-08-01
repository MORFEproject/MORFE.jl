using Test
using LinearAlgebra
using Random
using StaticArrays

using MORFE.MasterModeOrthogonality:
	precompute_orthogonality_operator_coefficients,
	precompute_orthogonality_column_polynomials,
	evaluate_orthogonality_row_and_lower_order_rhs!,
	evaluate_orthogonality_column_row!,
	evaluate_orthogonality_external_rhs,
	assemble_orthogonality_matrix_and_rhs!
using MORFE.Eigenproblems: left_eigenmode_orders_from_slice

# =============================================================================
# Fixture: complex ORD = 2 companion pencil with genuine left/right eigenblocks
# =============================================================================
# The orthogonality condition is the sesquilinear B-orthogonality
#     φᴴ B 𝒲 = 0,  𝒲 = [W₁; W₂],  W₁ = W,  W₂ = s W + Y₁ f + ξ₁
# of the companion left eigenvector φ (solving φᴴ(λB − A) = 0). A COMPLEX
# system is essential: real matrices hide the conjugation (transpose == adjoint).
function _complex_ord2_fixture(; FOM = 5, N_EXT = 1, seed = 42)
	Random.seed!(seed)
	ROM = 2
	ORD = 2

	B0 = randn(ComplexF64, FOM, FOM)
	B1 = randn(ComplexF64, FOM, FOM)
	B2 = randn(ComplexF64, FOM, FOM)
	fom_matrices = (B0, B1, B2)

	A_comp = [zeros(ComplexF64, FOM, FOM) I; -B0 -B1]
	B_comp = [Matrix{ComplexF64}(I, FOM, FOM) zeros(ComplexF64, FOM, FOM);
			  zeros(ComplexF64, FOM, FOM) B2]

	FR = eigen(A_comp, B_comp)
	FL = eigen(A_comp', B_comp')   # solve_left convention: reported λ = conj(pencil ν)

	finite = findall(x -> isfinite(x) && abs(x) < 1e6, FR.values)
	master_idx = finite[1:ROM]
	master_eigenvalues = SVector{ROM, ComplexF64}(FR.values[master_idx]...)

	right_blocks = zeros(ComplexF64, FOM, ORD, ROM)
	left_blocks = zeros(ComplexF64, FOM, ORD, ROM)
	for (r, i) in enumerate(master_idx)
		right_blocks[:, :, r] .= reshape(FR.vectors[:, i], FOM, ORD)
		j = argmin(abs.(conj.(FL.values) .- FR.values[i]))
		left_blocks[:, :, r] .= reshape(FL.vectors[:, j], FOM, ORD)
	end

	Φ_ext = randn(ComplexF64, FOM, N_EXT)
	λ_ext = 0.0 + 1.0im
	Λ = ComplexF64[
		master_eigenvalues[1] 0 1;
		0 master_eigenvalues[2] 0;
		0 0 λ_ext]

	return (; FOM, ROM, ORD, N_EXT, fom_matrices, A_comp, B_comp,
		master_eigenvalues, right_blocks, left_blocks, Φ_ext, Λ)
end

@testset "MasterModeOrthogonality — sesquilinear ground truth" begin
	f = _complex_ord2_fixture()
	(; FOM, ROM, ORD, N_EXT, fom_matrices, B_comp,
		master_eigenvalues, right_blocks, left_blocks, Φ_ext, Λ) = f

	left_slice = left_blocks[:, ORD, :]
	J_coeffs = precompute_orthogonality_operator_coefficients(
		fom_matrices, left_slice, left_blocks[:, 1:(ORD-1), :])
	C_coeffs, E_coeffs = precompute_orthogonality_column_polynomials(
		J_coeffs, right_blocks, Φ_ext, Λ)

	@testset "row coefficients are conjugated eigenvector blocks" begin
		for r in 1:ROM
			@test J_coeffs[r][1, :] ≈ conj.(left_blocks[:, 1, r])
			@test J_coeffs[r][2, :] ≈ conj.(fom_matrices[3]' * left_blocks[:, 2, r])
		end
	end

	@testset "J(s)ᵀ W = φᴴ B 𝒲(W), incl. exact resonance s = λ" begin
		for r in 1:ROM
			φ = vec(left_blocks[:, :, r])
			for s in (0.3 + 0.5im, master_eigenvalues[r], -1.2 + 0.1im)
				W = randn(ComplexF64, FOM)
				row = J_coeffs[r][1, :] .+ J_coeffs[r][2, :] .* s
				truth = φ' * B_comp * vcat(W, s .* W)
				@test transpose(row) * W ≈ truth rtol = 1e-10
			end
		end
	end

	@testset "full identity: φᴴB𝒲 = J·W + C·f_m + E·f_e + G·ξ" begin
		NVAR = ROM + N_EXT
		Y1 = hcat(right_blocks[:, 1, :], Φ_ext)   # physical blocks, all NVAR modes
		for r in 1:ROM
			φ = vec(left_blocks[:, :, r])
			for s in (master_eigenvalues[r], 0.7 - 0.2im)
				W = randn(ComplexF64, FOM)
				fc = randn(ComplexF64, NVAR)
				ξ1 = randn(ComplexF64, FOM)
				W2 = s .* W .+ Y1 * fc .+ ξ1
				truth = φ' * B_comp * vcat(W, W2)

				row = J_coeffs[r][1, :] .+ J_coeffs[r][2, :] .* s
				recon = transpose(row) * W +
						sum(C_coeffs[r][1, m] * fc[m] for m in 1:ROM) +
						sum(E_coeffs[r][1, e] * fc[ROM+e] for e in 1:N_EXT) +
						transpose(J_coeffs[r][2, :]) * ξ1
				@test recon ≈ truth rtol = 1e-10
			end
		end
	end

	@testset "lower-order RHS is bilinear (guards conjugating dot)" begin
		s = master_eigenvalues[1] + 0.3
		ξ1 = randn(ComplexF64, FOM)
		couplings = SVector{ORD, Vector{ComplexF64}}(ξ1, zeros(ComplexF64, FOM))
		row = zeros(ComplexF64, FOM)
		rhs_lower = evaluate_orthogonality_row_and_lower_order_rhs!(
			row, s, couplings, J_coeffs[1])
		# ORD = 2: single tail G₁(s) = J[2, :]; contraction must NOT conjugate.
		expected = -transpose(J_coeffs[1][2, :]) * ξ1
		@test rhs_lower ≈ expected rtol = 1e-12
		# a conjugating dot() would give a different value for complex rows
		@test !(rhs_lower ≈ -dot(J_coeffs[1][2, :], ξ1))
		# the fused pass must also leave row = J(s)
		@test row ≈ J_coeffs[1][1, :] .+ J_coeffs[1][2, :] .* s rtol = 1e-12
	end

	@testset "assembled system realises the identity" begin
		NVAR = ROM + N_EXT
		s = master_eigenvalues[1]
		resonance = SVector{ROM, Bool}(true, true)
		W = randn(ComplexF64, FOM)
		fc = randn(ComplexF64, NVAR)
		ξ1 = randn(ComplexF64, FOM)
		couplings = SVector{ORD, Vector{ComplexF64}}(ξ1, zeros(ComplexF64, FOM))
		external_dynamics = ComplexF64[fc[ROM+e] for e in 1:N_EXT]

		# Constant size: ROM rows and a ROM-wide border, independent of nR.
		M = Matrix{ComplexF64}(undef, ROM, FOM + ROM)
		rhs = zeros(ComplexF64, ROM)
		assemble_orthogonality_matrix_and_rhs!(
			M, rhs, s, J_coeffs, C_coeffs, E_coeffs,
			resonance, couplings, external_dynamics)

		Y1 = hcat(right_blocks[:, 1, :], Φ_ext)
		W2 = s .* W .+ Y1 * fc .+ ξ1
		for r in 1:ROM
			φ = vec(left_blocks[:, :, r])
			truth = φ' * B_comp * vcat(W, W2)
			lhs = transpose(M[r, 1:FOM]) * W +
				  sum(M[r, FOM+m] * fc[m] for m in 1:ROM)
			# M·[W; f] − rhs = φᴴB𝒲  (rhs carries −E·f_e − G·ξ)
			@test lhs - rhs[r] ≈ truth rtol = 1e-10
		end
	end

	@testset "non-resonant rows become the trivial equation R[r] = 0" begin
		NVAR = ROM + N_EXT
		s = master_eigenvalues[1]
		resonance = SVector{ROM, Bool}(true, false)   # mode 2 non-resonant
		W = randn(ComplexF64, FOM)
		fc = randn(ComplexF64, NVAR)
		ξ1 = randn(ComplexF64, FOM)
		couplings = SVector{ORD, Vector{ComplexF64}}(ξ1, zeros(ComplexF64, FOM))
		external_dynamics = ComplexF64[fc[ROM+e] for e in 1:N_EXT]

		# Garbage-fill: every entry must be overwritten, masked ones with hard zeros.
		M = fill(ComplexF64(9, 4), ROM, FOM + ROM)
		rhs = fill(ComplexF64(9, 4), ROM)
		assemble_orthogonality_matrix_and_rhs!(
			M, rhs, s, J_coeffs, C_coeffs, E_coeffs,
			resonance, couplings, external_dynamics)

		# Row 2: all zeros except the τ = 1 pinning R[2] = 0, with zero RHS.
		@test all(iszero, M[2, 1:FOM])
		@test M[2, FOM+1] == 0
		@test M[2, FOM+2] == 1
		@test rhs[2] == 0

		# Row 1 keeps the true orthogonality condition, but its corner entry for the
		# masked mode 2 is dropped — lossless, since row 2 pins that coefficient to 0.
		@test M[1, FOM+2] == 0
		Y1 = hcat(right_blocks[:, 1, :], Φ_ext)
		fc_masked = copy(fc)
		fc_masked[2] = 0                       # what the trivial row enforces
		W2 = s .* W .+ Y1 * fc_masked .+ ξ1
		φ = vec(left_blocks[:, :, 1])
		truth = φ' * B_comp * vcat(W, W2)
		lhs = transpose(M[1, 1:FOM]) * W + sum(M[1, FOM+m] * fc_masked[m] for m in 1:ROM)
		@test lhs - rhs[1] ≈ truth rtol = 1e-10
	end
end

@testset "left_eigenmode_orders_from_slice reproduces eigen blocks" begin
	f = _complex_ord2_fixture(; seed = 7)
	(; ORD, ROM, fom_matrices, master_eigenvalues, left_blocks) = f

	slice = left_blocks[:, ORD, :]
	blocks = left_eigenmode_orders_from_slice(
		fom_matrices, slice, collect(master_eigenvalues))
	for r in 1:ROM
		@test blocks[:, :, r] ≈ left_blocks[:, :, r] rtol = 1e-8
	end
end

@testset "ORD = 1 pass-through (no blocks, no eigenvalues)" begin
	Random.seed!(3)
	FOM = 4
	B0 = randn(ComplexF64, FOM, FOM)
	B1 = randn(ComplexF64, FOM, FOM)
	ℓ = randn(ComplexF64, FOM, 2)

	J_coeffs = precompute_orthogonality_operator_coefficients((B0, B1), ℓ)
	for r in 1:2
		@test size(J_coeffs[r]) == (1, FOM)
		@test J_coeffs[r][1, :] ≈ conj.(B1' * ℓ[:, r])
	end

	blocks = zeros(ComplexF64, FOM, 1, 2)
	blocks[:, 1, :] .= randn(ComplexF64, FOM, 2)
	C_coeffs, E_coeffs = precompute_orthogonality_column_polynomials(
		J_coeffs, blocks, zeros(ComplexF64, FOM, 0), zeros(ComplexF64, 2, 2))
	for r in 1:2
		@test size(C_coeffs[r]) == (0, 2)
		@test size(E_coeffs[r]) == (0, 0)
	end
end
