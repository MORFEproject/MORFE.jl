"""
Module `InvarianceError` — a posteriori validation of the computed invariant manifold.

After `solve_cohomological_problem` produces `(W, R)`, this module measures how
well the parametrisation satisfies the invariance equation

    ∂W/∂z · R(z) = F(W(z))

by evaluating both sides on random points `z` on the manifold and computing the
relative residual.  Results are plotted as convergence curves against the polynomial
order and returned as structured data for further analysis.
"""
module InvarianceError

using LinearAlgebra: mul!, dot, norm, lu, ldiv!
using Plots
using Random
using Statistics: median

using ..Polynomials: DensePolynomial, evaluate
using ..ParametrisationMethod: Parametrisation, ReducedDynamics
using ..FullOrderModel: NDOrderModel, evaluate_nonlinear_terms!

export invariance_error_norms, invariance_error_convergence, plot_invariance_convergence

# ─────────────────────────────────────────────────────────────────────────────
# Analytic JVP of the last derivative block W[:,ORD,:]
# ─────────────────────────────────────────────────────────────────────────────

"""
	_jvp_last_block!(result, W_poly, z, v, pw_buf)

Accumulate `J_{W[:,ORD,:]}(z) · v` into `result` using the analytic Jacobian
of the last stored derivative block.  `pw_buf` is a pre-allocated
`(NVAR, max_exp+1)` matrix reused across calls; on entry it is filled with
`pw_buf[j, e+1] = z[j]^e`.
"""
function _jvp_last_block!(
	result::AbstractVector,
	W_poly::DensePolynomial{T, NVAR},
	z::AbstractVector,
	v::AbstractVector,
	pw_buf::AbstractMatrix,       # (NVAR, max_exp+1), pre-allocated
) where {T, NVAR}
	coeffs   = W_poly.coefficients   # (FOM, ORD, L)
	exps     = W_poly.multiindex_set.exponents
	max_exps = W_poly.max_exponents  # SVector{NVAR,Int}
	L        = length(exps)
	FOM      = size(coeffs, 1)
	ORD      = size(coeffs, 2)
	Tv       = eltype(pw_buf)

	# Fill power table: pw_buf[j, e+1] = z[j]^e
	@inbounds for j in 1:NVAR
		pw_buf[j, 1] = one(Tv)
		zj = Tv(z[j])
		for e in 1:max_exps[j]
			pw_buf[j, e+1] = pw_buf[j, e] * zj
		end
	end

	@inbounds for l in 1:L
		exp_l = exps[l]
		for j in 1:NVAR
			α_j = exp_l[j]
			α_j == 0 && continue
			vj = Tv(v[j])
			iszero(vj) && continue

			# ∂_j m_l(z) · v_j = α_j · z_j^(α_j-1) · ∏_{k≠j} z_k^α_k · v_j
			dm = Tv(α_j) * pw_buf[j, α_j]    # α_j * z_j^(α_j-1)
			for k in 1:NVAR
				k == j && continue
				dm *= pw_buf[k, exp_l[k]+1]  # z_k^α_k
			end
			c = vj * dm
			for i in 1:FOM
				result[i] += c * coeffs[i, ORD, l]
			end
		end
	end
	return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Single-point invariance error (in-place)
# Returns rz = R(z) for reuse in the s_eff computation.
# ─────────────────────────────────────────────────────────────────────────────

function _invariance_error_at!(
	E::AbstractVector,
	buf_nl::AbstractVector,
	buf_fom::AbstractVector,
	pw_buf::AbstractMatrix,
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT},
	max_deg::Int,
	W::Parametrisation,
	R::ReducedDynamics,
	z::AbstractVector,
	r_external = nothing,
) where {ORD, ORDP1, N_NL, N_EXT}
	T     = eltype(E)
	one_T = one(T)

	# evaluate still allocates; in-place variants would require Polynomials changes
	X_vals = evaluate(W.poly, z)    # (FOM, ORD)
	rz     = evaluate(R.poly, z)    # NVAR vector

	# x^{(ORD)} = J_{W[:,ORD,:]}(z) · R(z)
	fill!(E, zero(T))
	_jvp_last_block!(E, W.poly, z, rz, pw_buf)

	mul!(buf_fom, model.linear_terms[end], E)
	copyto!(E, buf_fom)

	for k in 0:(ORD-1)
		mul!(E, model.linear_terms[k+1], view(X_vals, :, k + 1), one_T, one_T)
	end

	if max_deg >= 1
		fill!(buf_nl, zero(T))
		state_vectors = ntuple(k -> view(X_vals, :, k), Val(ORD))
		r = if !isnothing(r_external)
			r_external
		elseif N_EXT > 0
			z[(length(z)-N_EXT+1):end]
		else
			nothing
		end
		for deg in 1:max_deg
			evaluate_nonlinear_terms!(buf_nl, model, deg, state_vectors, r)
		end
		E .-= buf_nl
	end

	return rz
end

# ─────────────────────────────────────────────────────────────────────────────
# Sample reduced coordinates (in-place)
# ─────────────────────────────────────────────────────────────────────────────

function _sample_z!(z::AbstractVector, ROM, N_EXT, r_magnitude, rng)
	fill!(z, zero(eltype(z)))
	for j in 1:ROM
		z[j] = complex(randn(rng), randn(rng))
	end
	if N_EXT > 0 && !iszero(r_magnitude)
		NVAR = length(z)
		for j in (ROM+1):NVAR
			z[j] = complex(randn(rng), randn(rng))
		end
		ext_norm = norm(view(z, (ROM+1):NVAR))
		if !iszero(ext_norm)
			z[(ROM+1):NVAR] .*= r_magnitude / ext_norm
		end
	end
	return z
end

# ─────────────────────────────────────────────────────────────────────────────
# Log-log OLS regression with online left-trim to discard saturated points
# ─────────────────────────────────────────────────────────────────────────────

"""
	_log_log_regression(radii, errors; min_points=5) → (slope, intercept)

OLS fit of log(error) ~ slope·log(radius) + intercept.  Points are sorted by
radius and removed one at a time from the left (smallest radius first) as long
as doing so increases the slope.  This adaptively discards points saturated
near machine precision without a hard threshold.

Running Σ-updates make each step O(1); total cost O(N).
Returns `(NaN, NaN)` when fewer than `min_points` points remain.
"""
function _log_log_regression(radii, errors; min_points::Int = 5)
	n_total = length(radii)
	n_total < min_points && return (NaN, NaN)

	perm  = sortperm(radii)
	log_r = log.(radii[perm])
	log_e = log.(errors[perm])

	n    = Float64(n_total)
	Σx  = sum(log_r)
	Σy  = sum(log_e)
	Σxy = dot(log_r, log_e)
	Σxx = dot(log_r, log_r)

	d      = n * Σxx - Σx^2
	best_s = iszero(d) ? NaN : (n * Σxy - Σx * Σy) / d
	best_b = isnan(best_s) ? NaN : (Σy - best_s * Σx) / n

	for i in 1:(n_total-min_points)
		xi = log_r[i];
		yi = log_e[i]
		n -= 1;
		Σx -= xi;
		Σy -= yi;
		Σxy -= xi * yi;
		Σxx -= xi^2
		d = n * Σxx - Σx^2
		s = iszero(d) ? NaN : (n * Σxy - Σx * Σy) / d
		(isnan(s) || s <= best_s) && break
		best_s = s
		best_b = (Σy - s * Σx) / n
	end

	return (best_s, best_b)
end

# ─────────────────────────────────────────────────────────────────────────────
# Public API: invariance_error_norms
# ─────────────────────────────────────────────────────────────────────────────

"""
	invariance_error_norms(model, W, R;
						   n_samples  = 1000,
						   amplitude  = 1.0,
						   r_external = nothing,
						   rng        = Random.default_rng())
	→ NamedTuple{(:max, :mean, :rms, :pointwise)}

Evaluate the invariance-equation residual ‖E(z)‖₂ over a Gaussian point cloud
in reduced coordinates.

`amplitude` is the standard deviation of each complex master-mode component
(real and imaginary parts drawn i.i.d. from `N(0, amplitude²/2)`).
External coordinates are fixed to zero unless `r_external` is provided.
"""
function invariance_error_norms(
	model::NDOrderModel{ORD, ORDP1, N_NL},
	W::Parametrisation{ORD, NVAR},
	R::ReducedDynamics;
	n_samples::Int = 1000,
	amplitude::Real = 1.0,
	r_external = nothing,
	rng::AbstractRNG = Random.default_rng(),
) where {ORD, ORDP1, N_NL, NVAR}
	FOM     = model.n_fom
	ROM     = Base.size(R)
	Tc      = ComplexF64
	σ      = Float64(amplitude) / sqrt(2.0)
	max_deg = maximum(t.deg for t in model.nonlinear_terms; init = 0)
	max_exp = maximum(W.poly.max_exponents)

	E       = zeros(Tc, FOM)
	buf_nl  = zeros(Tc, FOM)
	buf_fom = zeros(Tc, FOM)
	pw_buf  = zeros(Tc, NVAR, max_exp + 1)
	z       = zeros(Tc, NVAR)

	pointwise = Vector{Float64}(undef, n_samples)

	for s in 1:n_samples
		fill!(z, zero(Tc))
		for j in 1:ROM
			z[j] = complex(σ * randn(rng), σ * randn(rng))
		end
		_invariance_error_at!(E, buf_nl, buf_fom, pw_buf, model, max_deg, W, R, z, r_external)
		pointwise[s] = norm(E)
	end

	return (
		max       = maximum(pointwise),
		mean      = sum(pointwise) / n_samples,
		rms       = sqrt(sum(x^2 for x in pointwise) / n_samples),
		pointwise = pointwise,
	)
end

# ─────────────────────────────────────────────────────────────────────────────
# Public API: invariance_error_convergence
# ─────────────────────────────────────────────────────────────────────────────

"""
	invariance_error_convergence(model, W, R;
								 n_samples    = 1000,
								 r_magnitudes = [0.0],
								 rng          = Random.default_rng())
	→ Vector of NamedTuples, one per entry in `r_magnitudes`

For each forcing magnitude `|r|` in `r_magnitudes`, draw `n_samples` points with
master-mode coordinates from `N(0,1)` and external coordinates sampled uniformly
on a sphere of radius `|r|`.  Each result NamedTuple contains:

- `radii`            — `‖z_k‖` for each sample
- `radii_master`     — `‖(z_k)_master‖` for each sample
- `force_errors`     — `‖E(z_k)‖₂` (force residual)
- `state_errors`     — `‖L(s̄)⁻¹ E(z_k)‖₂` (state error estimate)
- `s_bar`            — median local superharmonic `s̄ = median(⟨z,R(z)⟩/⟨z,z⟩)`
- `max_order`        — total degree of the highest monomial in W
- `r_magnitude`      — the `|r|` value for this level
- `convergence_rate` — OLS log-log slope of `force_errors` vs `radii_master`
					   (saturated points auto-trimmed)

Use `plot_invariance_convergence` to visualise the result.
"""
function invariance_error_convergence(
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT},
	W::Parametrisation{ORD, NVAR},
	R::ReducedDynamics;
	n_samples::Int = 1000,
	r_magnitudes::AbstractVector{<:Real} = [0.0],
	rng::AbstractRNG = Random.default_rng(),
) where {ORD, ORDP1, N_NL, N_EXT, NVAR}
	return map(r_magnitudes) do r_mag
		_convergence_one_level(model, W, R, n_samples, Float64(r_mag), rng)
	end
end

function _convergence_one_level(
	model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT},
	W::Parametrisation{ORD, NVAR},
	R::ReducedDynamics,
	n_samples::Int,
	r_mag::Float64,
	rng::AbstractRNG,
) where {ORD, ORDP1, N_NL, N_EXT, NVAR}
	FOM       = model.n_fom
	ROM       = Base.size(R)
	Tc        = ComplexF64
	max_order = sum(W.poly.multiindex_set.exponents[end])
	max_deg   = maximum(t.deg for t in model.nonlinear_terms; init = 0)
	max_exp   = maximum(W.poly.max_exponents)

	# Pre-allocate all working buffers once
	E       = zeros(Tc, FOM)
	buf_nl  = zeros(Tc, FOM)
	buf_fom = zeros(Tc, FOM)
	δx     = zeros(Tc, FOM)
	pw_buf  = zeros(Tc, NVAR, max_exp + 1)

	# Store samples as a (NVAR × n_samples) matrix for contiguous memory layout
	z_samples = Matrix{Tc}(undef, NVAR, n_samples)
	for s in 1:n_samples
		_sample_z!(view(z_samples, :, s), ROM, N_EXT, r_mag, rng)
	end

	# Phase 1: evaluate R(z_k) → s̄ via median Rayleigh quotient
	# Fill pre-allocated Float64 buffers to avoid real.(s_vals) / imag.(s_vals) allocs
	re_s = Vector{Float64}(undef, n_samples)
	im_s = Vector{Float64}(undef, n_samples)
	for s in 1:n_samples
		z_k = view(z_samples, :, s)
		rz_k = evaluate(R.poly, z_k)
		dz2 = real(dot(z_k, z_k))
		sk = iszero(dz2) ? zero(Tc) : Tc(dot(z_k, rz_k) / dz2)
		re_s[s] = real(sk)
		im_s[s] = imag(sk)
	end
	s_bar = complex(median(re_s), median(im_s))

	# Assemble and factor dynamic stiffness L(s̄) = ∑_k s̄^{k-1} Bₖ
	L_bar = Tc(s_bar^0) * model.linear_terms[1]
	for k in 2:length(model.linear_terms)
		L_bar = L_bar + Tc(s_bar^(k - 1)) * model.linear_terms[k]
	end
	lu_Lbar = lu(L_bar)

	# Phase 2: force residual and state error for each sample
	radii        = Vector{Float64}(undef, n_samples)
	radii_master = Vector{Float64}(undef, n_samples)
	force_errors = Vector{Float64}(undef, n_samples)
	state_errors = Vector{Float64}(undef, n_samples)

	for s in 1:n_samples
		z_k             = view(z_samples, :, s)
		radii[s]        = norm(z_k)
		radii_master[s] = norm(view(z_k, 1:ROM))
		_invariance_error_at!(E, buf_nl, buf_fom, pw_buf, model, max_deg, W, R, z_k)
		force_errors[s] = norm(E)
		ldiv!(δx, lu_Lbar, E)
		state_errors[s] = norm(δx)
	end

	convergence_rate, _ = _log_log_regression(radii_master, force_errors)

	return (
		radii            = radii,
		radii_master     = radii_master,
		force_errors     = force_errors,
		state_errors     = state_errors,
		s_bar            = s_bar,
		max_order        = max_order,
		r_magnitude      = r_mag,
		convergence_rate = convergence_rate,
	)
end

# ─────────────────────────────────────────────────────────────────────────────
# Public API: plot_invariance_convergence
# ─────────────────────────────────────────────────────────────────────────────

"""
	plot_invariance_convergence(results;
								x_axis            = :both,
								show_state_errors = true,
								show_regression   = false,
								title             = "Invariance error convergence")

Log-log scatter of force residuals (and optionally state error estimates) for
each forcing level in `results` (output of `invariance_error_convergence`).

Each forcing level gets a distinct colour.  A dashed O(‖·‖^{max_order+1})
reference line and a dotted machine-precision floor are added automatically.

## Arguments
- `results`: output of `invariance_error_convergence`.
- `x_axis`:
  - `:both` (default) — return `(full = plot_‖z‖, master = plot_‖z_master‖)`
  - `:full`   — single plot with full norm `‖z‖` on the x-axis
  - `:master` — single plot with `‖z_master‖` on the x-axis
- `show_state_errors`: overlay state error estimates (diamond markers).
- `show_regression`: overlay per-level OLS regression lines with estimated
  slope in the legend (master plot) or as unlabelled lines (full plot).
  Saturated points are trimmed automatically before fitting.
- `title`: plot title.
"""
function plot_invariance_convergence(
	results;
	x_axis::Symbol = :both,
	show_state_errors::Bool = true,
	show_regression::Bool = false,
	title::AbstractString = "Invariance error convergence",
)
	isempty(results) && error("results is empty")
	x_axis in (:both, :full, :master) ||
		error("x_axis must be :both, :full, or :master")

	if x_axis == :both
		return (
			full   = _plot_convergence(results, :full, show_state_errors, show_regression, title),
			master = _plot_convergence(results, :master, show_state_errors, show_regression, title),
		)
	else
		return _plot_convergence(results, x_axis, show_state_errors, show_regression, title)
	end
end

# ─────────────────────────────────────────────────────────────────────────────
# Plot internals
# ─────────────────────────────────────────────────────────────────────────────

function _reference_line_params(results, get_radii, max_order)
	r0      = results[1]
	rv0     = get_radii(r0)
	idx_med = sortperm(rv0)[length(rv0)÷2]
	r_ref   = rv0[idx_med]
	e_ref   = r0.force_errors[idx_med]
	C_ref   = e_ref / r_ref^(max_order + 1)
	all_r   = reduce(vcat, get_radii(res) for res in results)
	r_line  = exp.(LinRange(log(minimum(all_r)), log(maximum(all_r)), 300))
	return r_line, C_ref
end

function _plot_convergence(results, x_axis, show_state_errors, show_regression, title)
	max_order  = results[1].max_order
	get_radii  = x_axis == :full ? (res -> res.radii) : (res -> res.radii_master)
	xlabel_str = x_axis == :full ? "‖z‖" : "‖z_master‖"

	r_line, C_ref = _reference_line_params(results, get_radii, max_order)
	colors = palette(:tab10)

	p = plot(;
		xscale = :log10, yscale = :log10,
		xlabel = xlabel_str, ylabel = "error norm",
		title  = title,
		legend = :topleft,
		grid   = true)

	if x_axis == :master
		# Shape legend entries first (dummy series, legend only)
		scatter!(p, Float64[], Float64[];
			marker = :circle, color = :grey50,
			markersize = 4, markerstrokewidth = 0, label = "○  ‖E‖")
		show_state_errors && scatter!(p, Float64[], Float64[];
			marker = :diamond, color = :grey50,
			markersize = 4, markerstrokewidth = 0, label = "◇  ‖L⁻¹E‖")

		plot!(p, r_line, C_ref .* r_line .^ (max_order + 1);
			label = "O($(xlabel_str)^$(max_order + 1))",
			lw = 2, linestyle = :dash, color = :black)

		for (i, res) in enumerate(results)
			col     = colors[mod1(i, length(colors))]
			rv      = get_radii(res)
			r_label = iszero(res.r_magnitude) ? "|r| = 0" :
			"|r| = $(round(res.r_magnitude, sigdigits=2))"

			scatter!(p, rv, res.force_errors;
				label = "", color = col,
				markersize = 3, markerstrokewidth = 0, alpha = 0.5)
			show_state_errors && scatter!(p, rv, res.state_errors;
				label = "", color = col, marker = :diamond,
				markersize = 3, markerstrokewidth = 0, alpha = 0.35)

			if show_regression
				slope, intercept = _log_log_regression(rv, res.force_errors)
				if !isnan(slope)
					rate_str = string(round(slope; digits = 1))
					plot!(p, r_line, exp.(intercept .+ slope .* log.(r_line));
						label = "$r_label  (p ≈ $rate_str)",
						color = col, lw = 2, linestyle = :dashdot)
				else
					plot!(p, Float64[], Float64[]; label = r_label, color = col, lw = 2)
				end
			else
				plot!(p, Float64[], Float64[]; label = r_label, color = col, lw = 2)
			end
		end

	else  # :full
		for (i, res) in enumerate(results)
			col   = colors[mod1(i, length(colors))]
			rv    = get_radii(res)
			label = iszero(res.r_magnitude) ? "|r| = 0  (unforced)" :
			"|r| = $(round(res.r_magnitude, sigdigits=2))"

			scatter!(p, rv, res.force_errors;
				label = label * "  ‖E‖", color = col,
				markersize = 3, markerstrokewidth = 0, alpha = 0.5)
			show_state_errors && scatter!(p, rv, res.state_errors;
				label = label * "  ‖L⁻¹E‖", color = col, marker = :diamond,
				markersize = 3, markerstrokewidth = 0, alpha = 0.35)

			if show_regression
				slope, intercept = _log_log_regression(rv, res.force_errors)
				if !isnan(slope)
					plot!(p, r_line, exp.(intercept .+ slope .* log.(r_line));
						label = "", color = col, lw = 2, linestyle = :dashdot)
				end
			end
		end

		plot!(p, r_line, C_ref .* r_line .^ (max_order + 1);
			label = "O($(xlabel_str)^$(max_order + 1))",
			lw = 2, linestyle = :dash, color = :black)
	end

	hline!(p, [eps(Float64)];
		label = "machine precision", lw = 1, linestyle = :dot, color = :grey40)

	return p
end

end # module
