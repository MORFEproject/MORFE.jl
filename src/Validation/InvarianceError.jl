module InvarianceError

using LinearAlgebra: mul!, dot, norm, lu
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
    _jvp_last_block!(result, W_poly, z, v)

Accumulate `J_{W[:,ORD,:]}(z) · v` into `result` using the analytic Jacobian
of the last stored derivative block.  No ForwardDiff needed.
"""
function _jvp_last_block!(
        result::AbstractVector,
        W_poly::DensePolynomial{T, NVAR},
        z::AbstractVector,
        v::AbstractVector
) where {T, NVAR}
    coeffs   = W_poly.coefficients          # (FOM, ORD, L)
    exps     = W_poly.multiindex_set.exponents
    max_exps = W_poly.max_exponents         # SVector{NVAR,Int}
    L        = length(exps)
    FOM      = size(coeffs, 1)
    ORD      = size(coeffs, 2)
    Tv       = promote_type(T, eltype(z), eltype(v))

    # pw[j] is a vector where pw[j][e+1] = z[j]^e
    pw = ntuple(Val(NVAR)) do j
        me = max_exps[j]
        pj = Vector{Tv}(undef, me + 1)
        pj[1] = one(Tv)
        zj = Tv(z[j])
        @inbounds for e in 1:me
            pj[e + 1] = pj[e] * zj
        end
        pj
    end

    @inbounds for l in 1:L
        exp_l = exps[l]
        for j in 1:NVAR
            α_j = exp_l[j]
            α_j == 0 && continue
            vj = Tv(v[j])
            iszero(vj) && continue

            # ∂_j m_l(z) · v_j  =  α_j · z_j^(α_j-1) · ∏_{k≠j} z_k^α_k · v_j
            # pw[j][α_j] = z_j^(α_j-1)  (safe: no division needed)
            dm = Tv(α_j) * pw[j][α_j]
            for k in 1:NVAR
                k == j && continue
                dm *= pw[k][exp_l[k] + 1]   # z_k^α_k
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
        model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT},
        W::Parametrisation,
        R::ReducedDynamics,
        z::AbstractVector,
        r_external = nothing
) where {ORD, ORDP1, N_NL, N_EXT}
    T     = eltype(E)
    one_T = one(T)

    # Evaluate parametrisation (FOM × ORD) and reduced dynamics (NVAR)
    X_vals = evaluate(W.poly, z)    # Matrix{T}, shape (FOM, ORD)
    rz     = evaluate(R.poly, z)    # Vector{T}, length NVAR

    # x^{(ORD)} = J_{W[:,ORD,:]}(z) · R(z), written into E
    fill!(E, zero(T))
    _jvp_last_block!(E, W.poly, z, rz)

    # E ← B_ORD · x^{(ORD)}
    mul!(buf_fom, model.linear_terms[end], E)
    copyto!(E, buf_fom)

    # E += ∑_{k=0}^{ORD-1} B_k · x^{(k)}
    for k in 0:(ORD - 1)
        mul!(E, model.linear_terms[k + 1], view(X_vals, :, k + 1), one_T, one_T)
    end

    # E -= F(x, ẋ, …, x^{(ORD-1)}, r)
    if N_NL > 0
        max_deg = maximum(t.deg for t in model.nonlinear_terms; init = 0)
        if max_deg >= 1
            fill!(buf_nl, zero(T))
            state_vectors = ntuple(k -> view(X_vals, :, k), Val(ORD))
            # When r_external is not given, read the external state from the tail of z.
            # z[(ROM+1):end] = z_ext equals the external system state r that enters F.
            # This is correct for all degrees, including degree-1 forcing terms (linear in r).
            r = if !isnothing(r_external)
                r_external
            elseif N_EXT > 0
                ROM_inferred = length(z) - N_EXT
                z[(ROM_inferred + 1):end]
            else
                nothing
            end
            for deg in 1:max_deg
                evaluate_nonlinear_terms!(buf_nl, model, deg, state_vectors, r)
            end
            E .-= buf_nl
        end
    end

    return rz
end

# ─────────────────────────────────────────────────────────────────────────────
# Sample reduced coordinates
# ─────────────────────────────────────────────────────────────────────────────

function _sample_z(ROM, NVAR, N_EXT, r_magnitude, rng)
    z = zeros(ComplexF64, NVAR)
    # Master-mode coordinates: i.i.d. N(0,1) complex
    for j in 1:ROM
        z[j] = complex(randn(rng), randn(rng))
    end
    # External coordinates: random direction on sphere of radius r_magnitude
    if N_EXT > 0 && !iszero(r_magnitude)
        for j in (ROM + 1):NVAR
            z[j] = complex(randn(rng), randn(rng))
        end
        ext_norm = norm(view(z, (ROM + 1):NVAR))
        if !iszero(ext_norm)
            z[(ROM + 1):NVAR] .*= r_magnitude / ext_norm
        end
    end
    return z
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
(real and imaginary parts are drawn i.i.d. from `N(0, amplitude²/2)`).
External coordinates are fixed to zero unless `r_external` is provided.
"""
function invariance_error_norms(
        model::NDOrderModel{ORD, ORDP1, N_NL},
        W::Parametrisation{ORD, NVAR},
        R::ReducedDynamics;
        n_samples::Int = 1000,
        amplitude::Real = 1.0,
        r_external = nothing,
        rng::AbstractRNG = Random.default_rng()
) where {ORD, ORDP1, N_NL, NVAR}
    FOM = model.n_fom
    ROM = Base.size(R)
    Tc  = ComplexF64
    σ   = Float64(amplitude) / sqrt(2.0)

    E       = zeros(Tc, FOM)
    buf_nl  = zeros(Tc, FOM)
    buf_fom = zeros(Tc, FOM)
    z       = zeros(Tc, NVAR)

    pointwise = Vector{Float64}(undef, n_samples)

    for s in 1:n_samples
        fill!(z, zero(Tc))
        for j in 1:ROM
            z[j] = complex(σ * randn(rng), σ * randn(rng))
        end
        _invariance_error_at!(E, buf_nl, buf_fom, model, W, R, z, r_external)
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
on a sphere of radius `|r|`.  For each point record:

- `radii[k]`        = `‖z_k‖`
- `force_errors[k]` = `‖E(z_k)‖₂` (force residual)
- `state_errors[k]` = `‖L(s̄)⁻¹ E(z_k)‖₂` (state error estimate)

The representative frequency `s̄ = median(s_eff_k)` where
`s_eff(z) = ⟨z, R(z)⟩ / ⟨z, z⟩` is computed from the sample cloud
— no eigenvalue needs to be passed.  Both error series have expected
log-log slope `max_order + 1`.

Use `plot_invariance_convergence` to visualise the result.
"""
function invariance_error_convergence(
        model::NDOrderModel{ORD, ORDP1, N_NL, N_EXT},
        W::Parametrisation{ORD, NVAR},
        R::ReducedDynamics;
        n_samples::Int = 1000,
        r_magnitudes::AbstractVector{<:Real} = [0.0],
        rng::AbstractRNG = Random.default_rng()
) where {ORD, ORDP1, N_NL, N_EXT, NVAR}
    FOM  = model.n_fom
    ROM  = Base.size(R)
    Tc   = ComplexF64
    mset = W.poly.multiindex_set
    max_order = sum(mset.exponents[end])

    results = map(r_magnitudes) do r_mag
        _convergence_one_level(model, W, R, FOM, ROM, NVAR, N_EXT, Tc,
            max_order, n_samples, Float64(r_mag), rng)
    end

    return results
end

function _convergence_one_level(model, W, R, FOM, ROM, NVAR, N_EXT, Tc,
        max_order, n_samples, r_mag, rng)
    # Pre-sample all reduced coordinates for this forcing level
    z_samples = [_sample_z(ROM, NVAR, N_EXT, r_mag, rng) for _ in 1:n_samples]

    # Phase 1: evaluate R(z_k) to determine local superharmonics → s̄
    s_vals = map(z_samples) do z_k
        rz_k = evaluate(R.poly, z_k)
        dz2  = real(dot(z_k, z_k))
        iszero(dz2) ? zero(Tc) : Tc(dot(z_k, rz_k) / dz2)
    end
    s_bar = complex(median(real.(s_vals)), median(imag.(s_vals)))

    # Assemble dynamic stiffness L(s̄) = ∑_k s̄^{k-1} Bₖ and factor once
    L_bar = Tc(s_bar^0) * model.linear_terms[1]
    for k in 2:length(model.linear_terms)
        L_bar = L_bar + Tc(s_bar^(k - 1)) * model.linear_terms[k]
    end
    lu_Lbar = lu(L_bar)

    # Phase 2: compute force residual and state error for each sample
    radii        = Vector{Float64}(undef, n_samples)
    radii_master = Vector{Float64}(undef, n_samples)
    force_errors = Vector{Float64}(undef, n_samples)
    state_errors = Vector{Float64}(undef, n_samples)

    E       = zeros(Tc, FOM)
    buf_nl  = zeros(Tc, FOM)
    buf_fom = zeros(Tc, FOM)

    for s in 1:n_samples
        z_k = z_samples[s]
        radii[s]        = norm(z_k)
        radii_master[s] = norm(view(z_k, 1:ROM))
        _invariance_error_at!(E, buf_nl, buf_fom, model, W, R, z_k)
        force_errors[s] = norm(E)
        δx = lu_Lbar \ E
        state_errors[s] = norm(δx)
    end

    return (
        radii        = radii,
        radii_master = radii_master,
        force_errors = force_errors,
        state_errors = state_errors,
        s_bar        = s_bar,
        max_order    = max_order,
        r_magnitude  = r_mag,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Public API: plot_invariance_convergence
# ─────────────────────────────────────────────────────────────────────────────

"""
    plot_invariance_convergence(results;
                                x_axis            = :both,
                                show_state_errors = true,
                                title             = "Invariance error convergence")

Log-log scatter of force residuals (and optionally state error estimates) for
each forcing level in `results` (the output of `invariance_error_convergence`).

Each forcing level gets a distinct colour.  A dashed reference line with slope
`max_order + 1` and a dotted machine-precision floor are added automatically.

## Arguments
- `results`: output of `invariance_error_convergence` — a `Vector` of NamedTuples.
- `x_axis`: which amplitude measure to use on the x-axis.
  - `:both` (default) — return `(full = plot_with_‖z‖, master = plot_with_‖z_master‖)`
  - `:full`   — single plot using the full norm `‖z‖` (master + external coords)
  - `:master` — single plot using `‖z_master‖` (master coordinates only)
- `show_state_errors`: overlay state error estimates with diamond markers.
- `title`: plot title (the x-axis label is appended automatically).
"""
function plot_invariance_convergence(
        results;
        x_axis::Symbol = :both,
        show_state_errors::Bool = true,
        title::AbstractString = "Invariance error convergence"
)
    isempty(results) && error("results is empty")
    x_axis in (:both, :full, :master) ||
        error("x_axis must be :both, :full, or :master")

    if x_axis == :both
        return (
            full   = _plot_convergence(results, :full,   show_state_errors, title),
            master = _plot_convergence(results, :master, show_state_errors, title),
        )
    else
        return _plot_convergence(results, x_axis, show_state_errors, title)
    end
end

function _plot_convergence(results, x_axis, show_state_errors, title)
    max_order  = results[1].max_order
    get_radii  = x_axis == :full ? (res -> res.radii) : (res -> res.radii_master)
    xlabel_str = x_axis == :full ? "‖z‖" : "‖z_master‖"

    # Reference line anchored at the median of the first level's force errors
    r0      = results[1]
    rv0     = get_radii(r0)
    idx_med = sortperm(rv0)[length(rv0) ÷ 2]
    r_ref   = rv0[idx_med]
    e_ref   = r0.force_errors[idx_med]
    C_ref   = e_ref / r_ref^(max_order + 1)
    all_r   = reduce(vcat, get_radii(res) for res in results)
    r_line  = exp.(LinRange(log(minimum(all_r)), log(maximum(all_r)), 300))

    colors = palette(:tab10)

    p = plot(;
        xscale = :log10, yscale = :log10,
        xlabel = xlabel_str, ylabel = "error norm",
        title  = title,
        legend = :topleft,
        grid   = true)

    for (i, res) in enumerate(results)
        col   = colors[mod1(i, length(colors))]
        label = iszero(res.r_magnitude) ? "|r| = 0  (unforced)" :
                "|r| = $(round(res.r_magnitude, sigdigits=2))"
        rv    = get_radii(res)

        scatter!(p, rv, res.force_errors;
            label      = label * "  ‖E‖",
            color      = col,
            markersize = 3, markerstrokewidth = 0, alpha = 0.5)

        if show_state_errors
            scatter!(p, rv, res.state_errors;
                label      = label * "  ‖L⁻¹E‖",
                color      = col,
                marker     = :diamond,
                markersize = 3, markerstrokewidth = 0, alpha = 0.35)
        end
    end

    plot!(p, r_line, C_ref .* r_line .^ (max_order + 1);
        label     = "O($(xlabel_str)^$(max_order + 1))",
        lw        = 2,
        linestyle = :dash,
        color     = :black)

    hline!(p, [eps(Float64)];
        label     = "machine precision",
        lw        = 1,
        linestyle = :dot,
        color     = :grey40)

    return p
end

end # module
