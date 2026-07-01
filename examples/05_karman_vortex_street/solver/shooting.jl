"""
	shooting.jl — Picard iteration for FOM periodic orbits.

Starting from a ROM-predicted initial condition (s_guess, T), drives the
trajectory onto the true FOM limit cycle by:

  1. n_damp orbits with artificial damping  −γ B₁ s  to suppress transients
	 from the ROM initial-condition error.
  2. A 2-orbit undamped probe (if lift_weights provided) to estimate the true
	 FOM period T_fom from upward zero-crossings of the lift signal.
  3. Undamped Picard orbits  s ← Φ(s, T_fom)  until the mass-norm periodicity
	 residual is below `tol · X_max`.

At convergence, s_po = s(0) satisfies  Φ(s_po, T_fom) ≈ s_po  to the requested
tolerance and is the FOM periodic-orbit representative for that period.
"""

function _estimate_period_zerocross(signal::Vector{Float64}, dt::Float64)
	crossings = Float64[]
	for i in 2:length(signal)
		if signal[i-1] < 0.0 && signal[i] >= 0.0
			frac = -signal[i-1] / (signal[i] - signal[i-1])
			push!(crossings, dt * (i - 1 + frac))
		end
	end
	length(crossings) < 2 && return NaN
	return sum(diff(crossings)) / (length(crossings) - 1)
end

"""
	find_periodic_orbit(s_guess, η_prime, T, fom, B₀, B₁, K_visc, h₀_vec; kwargs...)
	→ (s_po, E_norm, X_max, F_L_max, F_L_min, n_orbits, converged)

Find the FOM periodic orbit near the ROM prediction `(s_guess, T)` by Picard
iteration (integrate-and-map) with an initial damped phase to absorb transients,
optionally followed by a period-estimation probe from the lift signal.

## Keyword arguments
- `Δt`          : time-step size (default `1e-3`)
- `θ`           : implicit weight (0.5 = Crank-Nicolson, 1 = backward Euler)
- `n_damp`      : number of damped spin-up orbits (default 10)
- `γ_damp`      : artificial damping coefficient; adds `−γ B₁ s` to shift all
				  eigenvalues left by γ (default 5.0)
- `max_picard`  : maximum undamped Picard orbits before giving up (default 200)
- `tol`         : convergence tolerance on `‖E‖_M / X_max` (default 1e-4)
- `lift_weights`: if provided (`Vector{Float64}` of length n_free_dpim), enables
				  (a) a 2-orbit undamped probe after spin-up to estimate T_fom
				  from upward zero-crossings of `F_L(t) = dot(lift_weights, s)`,
				  (b) tracking of F_L_max and F_L_min over the converged orbit.
				  Falls back to the ROM period T if fewer than 2 crossings are found.
- `verbose`     : print residual every orbit if `true`

## Returns
- `s_po`      : periodic-orbit initial condition
- `E_norm`    : `sqrt((Φ(s_po,T)−s_po)ᵀ B₁ (Φ(s_po,T)−s_po))` at convergence
- `X_max`     : `max_t sqrt(s(t)ᵀ B₁ s(t))` over the last orbit
- `F_L_max`   : maximum lift over the converged orbit (NaN if lift_weights===nothing)
- `F_L_min`   : minimum lift over the converged orbit (NaN if lift_weights===nothing)
- `n_orbits`  : total number of periods integrated (damp + probe + Picard)
- `converged` : `true` if tolerance was met before `max_picard` was reached
"""
function find_periodic_orbit(
	s_guess::Vector{Float64},
	η_prime::Float64,
	T::Float64,
	fom,
	B₀, B₁,
	K_visc,
	h₀_vec::Vector{Float64};
	Δt::Float64 = 1e-3,
	θ::Float64 = 0.5,
	n_damp::Int = 10,
	γ_damp::Float64 = 5.0,
	max_picard::Int = 200,
	tol::Float64 = 1e-4,
	lift_weights::Union{Vector{Float64}, Nothing} = nothing,
	verbose::Bool = false,
)
	n_steps = max(1, round(Int, T / Δt))
	Δt_exact = T / n_steps
	inv_dt = 1.0 / Δt_exact
	A_imp = B₀ .- η_prime .* K_visc

	s   = copy(s_guess)
	rhs = similar(s)
	f2  = similar(s)
	tmp = similar(s)

	total_orbits = 0

	# ── Phase 1: damped spin-up ───────────────────────────────────────────────
	if n_damp > 0
		A_damp     = A_imp .+ γ_damp .* B₁
		LHS_damp   = inv_dt .* B₁ .+ θ .* A_damp
		RHS_M_damp = inv_dt .* B₁ .- (1.0 - θ) .* A_damp
		L_klu_damp = klu(LHS_damp)

		for orbit in 1:n_damp
			for step in 1:n_steps
				eval_perturbation_convection!(f2, s, fom)
				mul!(rhs, RHS_M_damp, s)
				axpy!(1.0, f2, rhs)
				axpy!(η_prime, h₀_vec, rhs)
				ldiv!(s, L_klu_damp, rhs)
				if !isfinite(dot(s, s))
					@warn "find_periodic_orbit: blow-up during damped orbit $orbit/$n_damp at " *
						  "step $step — increase γ_damp or decrease Δt"
					return s, NaN, NaN, NaN, NaN, total_orbits + orbit, false
				end
			end
			verbose && @printf("  damp %3d/%3d  ‖s‖ = %.3e\n",
				orbit, n_damp, sqrt(max(dot(s, s), 0.0)))
		end
		total_orbits += n_damp
	end

	# ── Build undamped clean matrices (used for probe + Picard) ───────────────
	LHS_clean   = inv_dt .* B₁ .+ θ .* A_imp
	RHS_M_clean = inv_dt .* B₁ .- (1.0 - θ) .* A_imp
	L_klu_clean = klu(LHS_clean)

	# ── Phase 2: period-estimation probe (2×T_ROM, undamped) ─────────────────
	if lift_weights !== nothing
		n_probe   = 2 * n_steps
		F_L_probe = Vector{Float64}(undef, n_probe)
		for step in 1:n_probe
			eval_perturbation_convection!(f2, s, fom)
			mul!(rhs, RHS_M_clean, s)
			axpy!(1.0, f2, rhs)
			axpy!(η_prime, h₀_vec, rhs)
			ldiv!(s, L_klu_clean, rhs)
			if !isfinite(dot(s, s))
				@warn "find_periodic_orbit: blow-up during period-estimation probe at step $step"
				return s, NaN, NaN, NaN, NaN, total_orbits, false
			end
			F_L_probe[step] = dot(lift_weights, s)
		end
		total_orbits += 2

		T_fom = _estimate_period_zerocross(F_L_probe, Δt_exact)
		if isfinite(T_fom) && T_fom > 0.0
			verbose && @printf("  period probe: T_ROM=%.5f → T_fom=%.5f\n", T, T_fom)
			n_steps     = max(1, round(Int, T_fom / Δt_exact))
			Δt_exact   = T_fom / n_steps
			inv_dt      = 1.0 / Δt_exact
			LHS_clean   = inv_dt .* B₁ .+ θ .* A_imp
			RHS_M_clean = inv_dt .* B₁ .- (1.0 - θ) .* A_imp
			L_klu_clean = klu(LHS_clean)
		else
			verbose && @printf("  period probe: insufficient zero-crossings, keeping T_ROM=%.5f\n", T)
		end
	end

	# ── Phase 3: undamped Picard until ‖E‖_M / X_max < tol ──────────────────
	E_norm = NaN
	X_max = NaN
	F_L_max = lift_weights !== nothing ? dot(lift_weights, s) : NaN
	F_L_min = F_L_max

	for orbit in 1:max_picard
		s_start = copy(s)

		mul!(tmp, B₁, s)
		X_max   = sqrt(max(dot(s, tmp), 0.0))
		F_L_max = lift_weights !== nothing ? dot(lift_weights, s) : NaN
		F_L_min = F_L_max

		for step in 1:n_steps
			eval_perturbation_convection!(f2, s, fom)
			mul!(rhs, RHS_M_clean, s)
			axpy!(1.0, f2, rhs)
			axpy!(η_prime, h₀_vec, rhs)
			ldiv!(s, L_klu_clean, rhs)
			if !isfinite(dot(s, s))
				@warn "find_periodic_orbit: blow-up during Picard orbit $orbit at step $step"
				return s_start, NaN, NaN, NaN, NaN, total_orbits + orbit, false
			end
			mul!(tmp, B₁, s)
			X_max = max(X_max, sqrt(max(dot(s, tmp), 0.0)))
			if lift_weights !== nothing
				F_L     = dot(lift_weights, s)
				F_L_max = max(F_L_max, F_L)
				F_L_min = min(F_L_min, F_L)
			end
		end

		E = s .- s_start
		mul!(tmp, B₁, E)
		E_norm  = sqrt(max(dot(E, tmp), 0.0))
		rel_err = X_max > 0.0 ? E_norm / X_max : NaN

		verbose && @printf("  Picard %3d  ‖E‖_M = %.3e  X_max = %.3e  rel = %.3e\n",
			total_orbits + orbit, E_norm, X_max, rel_err)

		if isfinite(rel_err) && rel_err < tol
			return s_start, E_norm, X_max, F_L_max, F_L_min, total_orbits + orbit, true
		end
	end

	@warn "find_periodic_orbit: did not converge in $(total_orbits + max_picard) orbits " *
		  "(rel = $(X_max > 0 ? E_norm/X_max : NaN))"
	return s, E_norm, X_max, F_L_max, F_L_min, total_orbits + max_picard, false
end

# ─────────────────────────────────────────────────────────────────────────────
# Newton-Krylov (GMRES) shooting for FOM periodic orbits
# ─────────────────────────────────────────────────────────────────────────────

"""
Integrate one full orbit (no period subdivision) and return s(T).
Returns `nothing` on blow-up.
"""
function _integrate_one_orbit(
	s₀::Vector{Float64},
	η_prime::Float64,
	T::Float64,
	fom,
	L_klu,                       # pre-factored LHS for this η, T, Δt
	RHS_M::AbstractMatrix,
	h₀_vec::Vector{Float64},
	Δt::Float64,
	lift_w::Union{Vector{Float64}, Nothing} = nothing,
)
	n_steps = max(1, round(Int, T / Δt))
	s = copy(s₀)
	rhs = similar(s)
	f2 = similar(s)
	FL_hist = lift_w === nothing ? nothing : Vector{Float64}(undef, n_steps)

	for k in 1:n_steps
		if dot(s, s) > 1e-28
			eval_perturbation_convection!(f2, s, fom)
		else
			fill!(f2, 0.0)
		end
		mul!(rhs, RHS_M, s)
		axpy!(1.0, f2, rhs)
		axpy!(η_prime, h₀_vec, rhs)
		ldiv!(s, L_klu, rhs)
		isfinite(dot(s, s)) || return nothing, nothing
		lift_w !== nothing && (FL_hist[k] = dot(lift_w, s))
	end
	return s, FL_hist
end

"""
	_gmres_monodromy(b, s₀, Φ_s₀, matvec_args...; krylov_dim, tol, v0) → (x, converged)

Restart-less GMRES to solve (M − I) x = b where the matrix-vector product
(M − I)v = (Φ(s₀ + ε v) − Φ(s₀)) / ε − v  is matrix-free (one FOM integration per step).

`matvec_args` = (η_prime, T, fom, L_klu, RHS_M, h₀_vec, Δt).
"""
function _gmres_monodromy(
	b::Vector{Float64},
	s₀::Vector{Float64},
	Φ_s₀::Vector{Float64},
	η_prime::Float64,
	T::Float64,
	fom,
	L_klu,
	RHS_M::AbstractMatrix,
	h₀_vec::Vector{Float64},
	Δt::Float64;
	krylov_dim::Int = 20,
	tol::Float64 = 1e-6,
	v0::Union{Vector{Float64}, Nothing} = nothing,
)
	n = length(b)
	β = norm(b)
	β < 1e-300 && return zeros(n), true, 0

	# ε for finite-difference monodromy-vector product
	s_nrm = max(1.0, norm(s₀))

	# Allocate
	V = zeros(n, krylov_dim + 1)
	H = zeros(krylov_dim + 1, krylov_dim)
	cs = zeros(krylov_dim)
	sn = zeros(krylov_dim)
	g = zeros(krylov_dim + 1)
	g[1] = β

	# First Krylov vector (optionally warm-started with v0)
	if v0 !== nothing
		nv = norm(v0)
		V[:, 1] = nv > 1e-300 ? v0 ./ nv : b ./ β
	else
		V[:, 1] = b ./ β
	end

	n_mv = 0
	j_end = 1
	rel_res = 1.0

	for j in 1:krylov_dim
		j_end = j
		vj    = @view V[:, j]

		# (M−I) vⱼ via finite difference
		v_nrm = norm(vj)
		ε = 1e-6 * s_nrm / max(v_nrm, 1e-300)
		sp = s₀ .+ ε .* vj
		Φ_sp, _ = _integrate_one_orbit(sp, η_prime, T, fom, L_klu, RHS_M, h₀_vec, Δt)
		n_mv += 1
		if Φ_sp === nothing
			return zeros(n), false, n_mv
		end
		w = (Φ_sp .- Φ_s₀) ./ ε .- vj

		# Modified Gram-Schmidt
		for i in 1:j
			H[i, j] = dot(@view(V[:, i]), w)
			axpy!(-H[i, j], @view(V[:, i]), w)
		end
		H[j+1, j] = norm(w)
		H[j+1, j] < 1e-300 && break   # lucky breakdown

		V[:, j+1] = w ./ H[j+1, j]

		# Apply previous Givens rotations
		for i in 1:(j-1)
			t         = cs[i] * H[i, j] + sn[i] * H[i+1, j]
			H[i+1, j] = -sn[i] * H[i, j] + cs[i] * H[i+1, j]
			H[i, j]   = t
		end

		# New Givens rotation
		r = hypot(H[j, j], H[j+1, j])
		cs[j] = H[j, j] / r
		sn[j] = H[j+1, j] / r
		H[j, j] = r;
		H[j+1, j] = 0.0
		g[j+1] = -sn[j] * g[j];
		g[j] = cs[j] * g[j]

		rel_res = abs(g[j+1]) / β
		rel_res < tol && break
	end

	# Back-substitution
	y = zeros(j_end)
	for i in j_end:-1:1
		y[i] = g[i]
		for k in (i+1):j_end
			y[i] -= H[i, k] * y[k]
		end
		y[i] /= H[i, i]
	end

	x = V[:, 1:j_end] * y
	return x, rel_res < tol, n_mv
end

"""
	find_periodic_orbit_newton(s_init, η_prime, T_init, fom, B₀, B₁, K_visc, h₀_vec,
							   φ₁_re; kwargs...)
	→ (s_po, T_po, n_integrations, converged)

Newton-Krylov shooting for a FOM periodic orbit.

Starting from `s_init` (typically the real-part of the ROM parametrisation W(z₀)), solve

	F(s₀, T) = Φ(s₀, T, η') − s₀ = 0

using Newton-GMRES.  The monodromy matrix-vector products are computed by finite
differences of the flow map.

Phase condition:  dot(φ₁_re, s₀) = 0  (pins the Poincaré phase; φ₁_re = Re(ψ₁)
the real part of the dominant FOM eigenvector, extracted from the W parametrisation
via `evaluate(W.poly, [ε,0,0]) / ε`).

The period T is updated every Newton step from the lift zero-crossings (if
`lift_weights` is provided) or kept fixed.

## Keyword arguments
- `Δt`          : time step (default `1e-3`)
- `θ`           : implicit weight (default `0.5`)
- `max_newton`  : outer Newton iterations (default `6`)
- `tol`         : convergence tolerance on ‖F‖ / ‖s₀‖ (default `1e-3`)
- `krylov_dim`  : max Krylov subspace dimension for GMRES (default `20`)
- `lift_weights`: if provided, used to update T from zero-crossings each Newton step
- `verbose`     : print Newton residuals

## Returns
- `s_po`          : periodic-orbit initial condition at convergence
- `T_po`          : period at convergence
- `n_integrations`: total FOM integrations used (1 per Newton + Krylov steps)
- `converged`     : `true` if ‖F‖/‖s₀‖ < tol

"""
function find_periodic_orbit_newton(
	s_init::Vector{Float64},
	η_prime::Float64,
	T_init::Float64,
	fom,
	B₀, B₁,
	K_visc,
	h₀_vec::Vector{Float64},
	φ₁_re::Vector{Float64};   # phase condition direction
	Δt::Float64 = 1e-3,
	θ::Float64 = 0.5,
	max_newton::Int = 6,
	tol::Float64 = 1e-3,
	krylov_dim::Int = 20,
	lift_weights::Union{Vector{Float64}, Nothing} = nothing,
	verbose::Bool = false,
)
	T         = T_init
	s₀      = copy(s_init)
	n_integ   = 0
	converged = false

	# Pre-build LHS / RHS matrices (rebuilt when T changes)
	function _build_matrices(T_val)
		n_steps = max(1, round(Int, T_val / Δt))
		Δt_ex  = T_val / n_steps
		inv_dt  = 1.0 / Δt_ex
		A_imp   = B₀ .- η_prime .* K_visc
		LHS     = inv_dt .* B₁ .+ θ .* A_imp
		RHS_M   = inv_dt .* B₁ .- (1.0 - θ) .* A_imp
		L_klu   = klu(LHS)
		return L_klu, RHS_M, Δt_ex
	end

	L_klu, RHS_M, Δt_ex = _build_matrices(T)

	# Warm-start vector for GMRES: the Im part of the eigenvector direction
	# (dominant Floquet direction near the limit cycle)
	v0_warm = nothing

	for newton_iter in 1:max_newton
		# ── Integrate one orbit ───────────────────────────────────────────
		Φ_s₀, FL_hist = _integrate_one_orbit(
			s₀, η_prime, T, fom, L_klu, RHS_M, h₀_vec, Δt_ex, lift_weights)
		n_integ += 1
		if Φ_s₀ === nothing
			@warn "find_periodic_orbit_newton: blow-up at Newton iter $newton_iter"
			return s₀, T, n_integ, false
		end

		# ── Shooting residual ─────────────────────────────────────────────
		F     = Φ_s₀ .- s₀
		F_nrm = norm(F)
		s_nrm = max(norm(s₀), 1.0)
		rel   = F_nrm / s_nrm

		verbose && @printf("  Newton %2d  ‖F‖/‖s‖ = %.3e  T = %.5f\n",
			newton_iter, rel, T)

		if rel < tol
			converged = true
			break
		end

		# ── Update period from lift zero-crossings ────────────────────────
		if lift_weights !== nothing && FL_hist !== nothing
			T_new = _estimate_period_zerocross(FL_hist, Δt_ex)
			if isfinite(T_new) && T_new > 0.5 * T && T_new < 2.0 * T
				T = T_new
				L_klu, RHS_M, Δt_ex = _build_matrices(T)
			end
		end

		# ── Phase-condition residual g = dot(φ₁_re, s₀) ──────────────────
		g = dot(φ₁_re, s₀)

		# ── GMRES solve:  (M−I) p = −F ───────────────────────────────────
		p, gm_conv, n_mv_p = _gmres_monodromy(
			-F, s₀, Φ_s₀, η_prime, T, fom, L_klu, RHS_M, h₀_vec, Δt_ex;
			krylov_dim = krylov_dim, tol = 1e-4, v0 = v0_warm)
		n_integ += n_mv_p

		# ── GMRES solve:  (M−I) q = φ₁_re  (period/phase direction) ─────
		# used to remove the phase ambiguity from p
		q, _, n_mv_q = _gmres_monodromy(
			φ₁_re, s₀, Φ_s₀, η_prime, T, fom, L_klu, RHS_M, h₀_vec, Δt_ex;
			krylov_dim = min(krylov_dim, 10), tol = 1e-4)
		n_integ += n_mv_q

		# ── Bordered correction: enforce dot(φ₁_re, δs₀) = −g ────────────
		denom = dot(φ₁_re, q)
		if abs(denom) > 1e-300
			α = (-g - dot(φ₁_re, p)) / denom
			δs₀ = p .+ α .* q
		else
			δs₀ = p   # skip phase correction if degenerate
		end

		# ── Update state; keep warm-start for next GMRES ──────────────────
		s₀ .+= δs₀
		v0_warm = δs₀ ./ max(norm(δs₀), 1e-300)
	end

	# Final periodicity check
	if !converged
		Φ_s₀, _ = _integrate_one_orbit(
			s₀, η_prime, T, fom, L_klu, RHS_M, h₀_vec, Δt_ex, nothing)
		n_integ += 1
		if Φ_s₀ !== nothing
			F_nrm = norm(Φ_s₀ .- s₀)
			converged = F_nrm / max(norm(s₀), 1.0) < tol
		end
	end

	return s₀, T, n_integ, converged
end

# ─────────────────────────────────────────────────────────────────────────────
# FOM pseudo-arclength continuation (PALC) corrector
# ─────────────────────────────────────────────────────────────────────────────

"""
	find_periodic_orbit_newton_palc(
		s_pred, T_pred, η_pred, ŝ, T̂, η̂,
		τ_s_eff, τ_T_eff, τ_η_eff, Δs,
		fom, B₀, B₁, K_visc, h₀_vec, φ₁_re; kwargs...)
	→ (s_new, T_new, η_new, n_newton, n_integ, converged)

FOM pseudo-arclength continuation corrector for the Hopf limit-cycle branch.

Starting from the predictor `(s_pred, T_pred, η_pred)`, Newton-GMRES corrects
`(s₀, η')` (with `T` updated from lift zero-crossings) so that

	F(s₀, T, η') = Φ(s₀, T, η') − s₀ = 0     (shooting)
	g(s₀)        = dot(φ₁_re, s₀)    = 0     (phase condition)
	N(s₀, T, η') = τ_s_eff·(s₀−ŝ) + τ_T_eff·(T−T̂) + τ_η_eff·(η′−η̂) − Δs = 0  (arclength)

`η'` is free (unlike `find_periodic_orbit_newton`, which fixes it). Each Newton step needs
one extra FOM integration to estimate `∂Φ/∂η'` by finite differences, and three GMRES solves
against the same `(M−I)` operator (for the shooting, phase, and η′ directions). The resulting
2×2 bordered system gives the phase-direction scale `α` and the η′ correction `δη′`.

## Returns
- `s_new`, `T_new`, `η_new` : corrected branch point
- `n_newton`  : number of outer Newton iterations used (for Δs adaptation)
- `n_integ`   : total FOM integrations used (Newton + GMRES + ∂Φ/∂η′)
- `converged` : `true` if ‖F‖/max(‖s₀‖,1) < tol
"""
function find_periodic_orbit_newton_palc(
	s_pred::Vector{Float64},
	T_pred::Float64,
	η_pred::Float64,
	ŝ::Vector{Float64},
	T̂::Float64,
	η̂::Float64,
	τ_s_eff::Vector{Float64},
	τ_T_eff::Float64,
	τ_η_eff::Float64,
	Δs::Float64,
	fom,
	B₀, B₁,
	K_visc,
	h₀_vec::Vector{Float64},
	φ₁_re::Vector{Float64};
	Δt::Float64 = 1e-2,
	θ::Float64 = 0.5,
	max_newton::Int = 6,
	tol::Float64 = 1e-3,
	krylov_dim::Int = 20,
	lift_weights::Union{Vector{Float64}, Nothing} = nothing,
	verbose::Bool = false,
	max_dη::Float64 = Inf,
)
	s₀ = copy(s_pred)
	T = T_pred
	η = η_pred
	n_integ = 0
	n_newton = 0
	converged = false

	function _build_matrices(T_val, η_val)
		n_steps = max(1, round(Int, T_val / Δt))
		Δt_ex = T_val / n_steps
		inv_dt = 1.0 / Δt_ex
		A_imp = B₀ .- η_val .* K_visc
		LHS = inv_dt .* B₁ .+ θ .* A_imp
		RHS_M = inv_dt .* B₁ .- (1.0 - θ) .* A_imp
		L_klu = klu(LHS)
		return L_klu, RHS_M, Δt_ex
	end

	L_klu, RHS_M, Δt_ex = _build_matrices(T, η)
	v0_warm = nothing

	for newton_iter in 1:max_newton
		n_newton = newton_iter

		# ── Integrate one orbit ───────────────────────────────────────────
		Φ_s₀, FL_hist = _integrate_one_orbit(
			s₀, η, T, fom, L_klu, RHS_M, h₀_vec, Δt_ex, lift_weights)
		n_integ += 1
		if Φ_s₀ === nothing
			@warn "find_periodic_orbit_newton_palc: blow-up at Newton iter $newton_iter"
			return s₀, T, η, n_newton, n_integ, false
		end

		# ── Update period from lift zero-crossings ────────────────────────
		if lift_weights !== nothing && FL_hist !== nothing
			T_new = _estimate_period_zerocross(FL_hist, Δt_ex)
			if isfinite(T_new) && T_new > 0.5 * T && T_new < 2.0 * T
				T = T_new
				L_klu, RHS_M, Δt_ex = _build_matrices(T, η)
			end
		end

		# ── Shooting and phase residuals ──────────────────────────────────
		F = Φ_s₀ .- s₀
		F_nrm = norm(F)
		s_nrm = max(norm(s₀), 1e-300)
		rel = F_nrm / s_nrm
		g = dot(φ₁_re, s₀)

		verbose && @printf("  PALC Newton %2d  ‖F‖/‖s‖ = %.3e  η′ = %+.5e  T = %.5f\n",
			newton_iter, rel, η, T)

		if rel < tol
			converged = true
			break
		end

		# ── ∂Φ/∂η′ via finite differences ─────────────────────────────────
		ε_η = 1e-6 * max(1.0, abs(η))
		L_klu_η, RHS_M_η, Δt_ex_η = _build_matrices(T, η + ε_η)
		Φ_η, _ = _integrate_one_orbit(
			s₀, η + ε_η, T, fom, L_klu_η, RHS_M_η, h₀_vec, Δt_ex_η, nothing)
		n_integ += 1
		if Φ_η === nothing
			@warn "find_periodic_orbit_newton_palc: blow-up in ∂Φ/∂η′ at iter $newton_iter"
			return s₀, T, η, n_newton, n_integ, false
		end
		dΦ_dη = (Φ_η .- Φ_s₀) ./ ε_η

		# ── GMRES: p = (M−I)⁻¹(−F) ────────────────────────────────────────
		p, _, n_mv_p = _gmres_monodromy(
			-F, s₀, Φ_s₀, η, T, fom, L_klu, RHS_M, h₀_vec, Δt_ex;
			krylov_dim = krylov_dim, tol = 1e-4, v0 = v0_warm)
		n_integ += n_mv_p

		# ── GMRES: q_φ = (M−I)⁻¹ φ₁_re  (phase direction) ────────────────
		q_φ, _, n_mv_qφ = _gmres_monodromy(
			φ₁_re, s₀, Φ_s₀, η, T, fom, L_klu, RHS_M, h₀_vec, Δt_ex;
			krylov_dim = min(krylov_dim, 10), tol = 1e-4)
		n_integ += n_mv_qφ

		# ── GMRES: q_η = (M−I)⁻¹(−∂Φ/∂η′)  (η′ direction) ────────────────
		q_η, _, n_mv_qη = _gmres_monodromy(
			-dΦ_dη, s₀, Φ_s₀, η, T, fom, L_klu, RHS_M, h₀_vec, Δt_ex;
			krylov_dim = min(krylov_dim, 10), tol = 1e-4)
		n_integ += n_mv_qη

		# ── Arclength residual ────────────────────────────────────────────
		N = dot(τ_s_eff, s₀ .- ŝ) + τ_T_eff * (T - T̂) + τ_η_eff * (η - η̂) - Δs

		# ── 2×2 bordered system for (α, δη′) ──────────────────────────────
		φp = dot(φ₁_re, p)
		φqφ = dot(φ₁_re, q_φ)
		φqη = dot(φ₁_re, q_η)
		τp = dot(τ_s_eff, p)
		τqφ = dot(τ_s_eff, q_φ)
		τqη = dot(τ_s_eff, q_η)

		A22 = @SMatrix [φqφ φqη
		                τqφ (τ_η_eff+τqη)]
		b2 = SVector(-g - φp, -N - τp)
		det22 = A22[1, 1] * A22[2, 2] - A22[1, 2] * A22[2, 1]

		if abs(det22) > 1e-300
			sol = A22 \ b2
			α = sol[1]
			δη = sol[2]
		else
			α = abs(φqφ) > 1e-300 ? (-g - φp) / φqφ : 0.0
			δη = 0.0
		end

		# ── Clamp η step: when τ_η_eff≈0 the 2×2 system is unconstrained in η
		# and can give wildly large δη.  Cap it independently of α so the
		# phase correction is not damped.
		if abs(δη) > max_dη
			δη = copysign(max_dη, δη)
		end

		# ── Apply correction ───────────────────────────────────────────────
		δs₀ = p .+ α .* q_φ .+ δη .* q_η
		s₀ .+= δs₀
		η += δη

		abs(δη) > 1e-9 && (L_klu, RHS_M, Δt_ex = _build_matrices(T, η))

		v0_warm = δs₀ ./ max(norm(δs₀), 1e-300)
	end

	if !converged
		Φ_s₀, _ = _integrate_one_orbit(s₀, η, T, fom, L_klu, RHS_M, h₀_vec, Δt_ex, nothing)
		n_integ += 1
		if Φ_s₀ !== nothing
			F_nrm = norm(Φ_s₀ .- s₀)
			converged = F_nrm / max(norm(s₀), 1e-300) < tol
		end
	end

	return s₀, T, η, n_newton, n_integ, converged
end
