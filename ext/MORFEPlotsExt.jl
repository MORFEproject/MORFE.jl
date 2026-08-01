module MORFEPlotsExt

using MORFE
using MORFE.InvarianceError: _log_log_regression
using Plots
using Statistics: median

# ─────────────────────────────────────────────────────────────────────────────
# Plot internals
# ─────────────────────────────────────────────────────────────────────────────

function _reference_line_params(results, get_radii, max_order)
    r0 = results[1]
    rv0 = get_radii(r0)
    idx_med = sortperm(rv0)[length(rv0) ÷ 2]
    r_ref = rv0[idx_med]
    e_ref = r0.force_errors[idx_med]
    C_ref = e_ref / r_ref^(max_order + 1)
    all_r = reduce(vcat, get_radii(res) for res in results)
    r_line = exp.(LinRange(log(minimum(all_r)), log(maximum(all_r)), 300))
    return r_line, C_ref
end

function _plot_convergence(results, x_axis, show_state_errors, show_regression, title)
    max_order = results[1].max_order
    get_radii = x_axis == :full ? (res -> res.radii) : (res -> res.radii_master)
    xlabel_str = x_axis == :full ? "‖z‖" : "‖z_master‖"

    r_line, C_ref = _reference_line_params(results, get_radii, max_order)
    colors = palette(:tab10)

    p = plot(;
        xscale = :log10, yscale = :log10,
        xlabel = xlabel_str, ylabel = "error norm",
        title = title,
        legend = :topleft,
        grid = true)

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
            col = colors[mod1(i, length(colors))]
            rv = get_radii(res)
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
            col = colors[mod1(i, length(colors))]
            rv = get_radii(res)
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

# ─────────────────────────────────────────────────────────────────────────────
# Public API override
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
function MORFE.InvarianceError.plot_invariance_convergence(
        results;
        x_axis::Symbol = :both,
        show_state_errors::Bool = true,
        show_regression::Bool = false,
        title::AbstractString = "Invariance error convergence"
)
    isempty(results) && error("results is empty")
    x_axis in (:both, :full, :master) ||
        error("x_axis must be :both, :full, or :master")

    if x_axis == :both
        return (
            full = _plot_convergence(
                results, :full, show_state_errors, show_regression, title),
            master = _plot_convergence(
                results, :master, show_state_errors, show_regression, title)
        )
    else
        return _plot_convergence(results, x_axis, show_state_errors, show_regression, title)
    end
end

end # module MORFEPlotsExt
