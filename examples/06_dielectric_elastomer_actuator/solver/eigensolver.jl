"""
solver/eigensolver.jl — cubic-pencil eigenanalysis and master-pair extraction.

Solves the companion-form generalised eigenproblem of the third-order model with the
dense `DefaultEigensolver` and extracts everything `solve_cohomological_problem` needs:

- `master_eigenvalues :: SVector{2, ComplexF64}` — (λ₁, conj λ₁), slowest oscillatory pair
- `master_modes :: Matrix` (FOM × 2) — conjugate by construction, tip-normalised
- `master_modes_derivatives :: Array` (FOM × 2 × 2) — blocks (λ₁φ, λ₁²φ)
- `left_eigenmodes :: Matrix` (FOM × 2) — pencil left vectors ℓ with ℓᵀP(λ₁) = 0
  (highest-order block of the companion left eigenvector), selected among
  matching/conjugation candidates by normalised pencil residual

Why not `solve_eigenproblem`: its global left/right matching and biorthogonal
normalisation are ill-posed on the ~n-fold quasi-degenerate RC relaxation cluster of
this model (eigenvalues clustered at −ĉ/R), producing warnings and unstable scalings.
Only the well-separated master pair is needed, so `solve` / `solve_left` are called
directly and the pairing is done here for that pair alone.

Spectrum structure to expect: a handful of underdamped bending pairs, the RC relaxation
cluster near −ĉ/R, and (high-frequency) overdamped Kelvin–Voigt modes as real pairs.
"""

using Printf

function dea_eigenanalysis(model, B0, B1, B2, B3, idx_wtip, rc_target;
	report::Bool = true)
	solver = DefaultEigensolver()

	# Right eigenpairs, sorted by |λ| (we sort ourselves; no global left matching)
	λs, Y = solve(model, solver)
	perm = sortperm(λs; by = abs)
	λs = λs[perm]
	Y = Y[:, :, perm]

	# Left eigenpairs of the transposed companion pencil (unsorted — matched locally)
	λLs, XL = solve_left(model, solver)

	# ── spectrum classification ───────────────────────────────────────────────
	@assert all(real(λ) < 0 for λ in λs) "A4: unstable eigenvalue found"
	osc = [k for k in eachindex(λs) if imag(λs[k]) > 0.1]
	real_eigs = sort([real(λs[k]) for k in eachindex(λs) if abs(imag(λs[k])) ≤ 0.1])
	@assert !isempty(osc) "no oscillatory mode found"
	λ_rc = real_eigs[argmin(abs.(real_eigs .+ rc_target))]
	@assert abs(λ_rc + rc_target) < 0.05 * rc_target "A5: RC branch not found near −ĉ/R"

	# ── master pair: slowest oscillatory mode, tip-normalised ─────────────────
	i1 = osc[1]
	λ1 = λs[i1]
	φ = Y[:, 1, i1]
	φ = φ ./ φ[idx_wtip]              # z₁ ≈ complex tip amplitude
	φd1 = λ1 .* φ                     # exact companion structure (checked below)
	φd2 = λ1^2 .* φ

	n = length(φ)
	master_eigenvalues = SVector{2, ComplexF64}(λ1, conj(λ1))
	master_modes = Matrix{ComplexF64}(hcat(φ, conj.(φ)))
	mmd = zeros(ComplexF64, n, 2, 2)
	mmd[:, 1, 1] .= φd1
	mmd[:, 2, 1] .= φd2
	mmd[:, 1, 2] .= conj.(φd1)
	mmd[:, 2, 2] .= conj.(φd2)

	# ── pencil left vector (highest-order companion block), residual-selected ─
	P(λ) = λ^3 .* B3 .+ λ^2 .* B2 .+ λ .* B1 .+ B0
	Pt(λ) = λ^3 .* transpose(B3) .+ λ^2 .* transpose(B2) .+ λ .* transpose(B1) .+
			transpose(B0)
	pencil_scale(λ) = abs(λ)^3 * norm(B3) + abs(λ)^2 * norm(B2) + abs(λ) * norm(B1) +
					  norm(B0)
	j1 = argmin([abs(λLs[k] - λ1) for k in eachindex(λLs)])
	j2 = argmin([abs(λLs[k] - conj(λ1)) for k in eachindex(λLs)])
	cands = (XL[:, 3, j1], conj.(XL[:, 3, j1]), XL[:, 3, j2], conj.(XL[:, 3, j2]))
	resids = [norm(Pt(λ1) * ℓc) / (pencil_scale(λ1) * norm(ℓc)) for ℓc in cands]
	ℓ = collect(cands[argmin(resids)])
	left_eigenmodes = Matrix{ComplexF64}(hcat(ℓ, conj.(ℓ)))

	# ── acceptance (Phase 6) ──────────────────────────────────────────────────
	resR = norm(P(λ1) * φ) / (pencil_scale(λ1) * norm(φ))
	resD1 = norm(Y[:, 2, i1] ./ Y[idx_wtip, 1, i1] .- φd1) / norm(φd1)
	resD2 = norm(Y[:, 3, i1] ./ Y[idx_wtip, 1, i1] .- φd2) / norm(φd2)
	@assert resR < 1e-10 "A1: right pencil residual $resR"
	@assert minimum(resids) < 1e-10 "A2: left pencil residual $(minimum(resids))"
	@assert resD1 < 1e-8 "A3: companion block 2 ≠ λ₁·φ ($resD1)"
	@assert resD2 < 1e-8 "A3: companion block 3 ≠ λ₁²·φ ($resD2)"

	# ── report ────────────────────────────────────────────────────────────────
	if report
		println("Spectrum of the cubic pencil (3n = $(3n) eigenvalues):")
		@printf "  oscillatory pairs: %d;  real eigenvalues: %d in [%.3e, %.3e]\n" length(osc) length(real_eigs) real_eigs[1] real_eigs[end]
		@printf "  RC relaxation cluster at %.6f  (−ĉ/R = %.6f)\n" λ_rc (-rc_target)
		println("  slowest oscillatory pairs (master first):")
		for k in osc[1:min(5, end)]
			ξk = -real(λs[k]) / abs(λs[k])
			@printf "    λ = %+.6f %+.6fim   |λ| = %8.4f   ξ = %.4f\n" real(λs[k]) imag(λs[k]) abs(λs[k]) ξk
		end
		@printf "Phase 6 checks passed: λ₁ = %+.6f %+.6fim  (residuals R %.1e | L %.1e)\n" real(λ1) imag(λ1) resR minimum(resids)
	end

	return (; λs, λ1, master_eigenvalues, master_modes, mmd, left_eigenmodes)
end
