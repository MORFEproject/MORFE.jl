function plot_backbone_absolute(r_range, Ω_curves, branches, labels;
                                Ω_curves_ref = nothing,
                                colors       = nothing,
                                xlabel = "Backbone frequency  Ω  (rad/s)",
                                ylabel = "Modal amplitude  |z₁|",
                                title  = "Backbone curves",
                                r_max  = maximum(r_range),
                                legend = :best,
                                size   = (800, 600), dpi = 150)
    clrs = isnothing(colors) ? palette(:tab10) : colors
    plt  = plot(; xlabel, ylabel, title, ylims = (0, r_max * 1.05), size, dpi, legend)
    for (k, (Ωcurve, br, lbl)) in enumerate(zip(Ω_curves, branches, labels))
        plot!(plt, Ωcurve, collect(r_range); lw = 2.0, color = clrs[k], label = lbl)
        if !isnothing(Ω_curves_ref)
            plot!(plt, Ω_curves_ref[k], collect(r_range);
                  lw = 1.2, ls = :dash, color = clrs[k], label = nothing)
        end
        if !isnothing(br) && length(br.branch) > 1
            r_bk = [s.r for s in br.branch]
            Ω_bk = [s.Ω for s in br.branch]
            mask = r_bk .<= r_max
            scatter!(plt, Ω_bk[mask], r_bk[mask];
                     color = clrs[k], ms = 3, markerstrokewidth = 0, label = nothing)
        end
    end
    return plt
end

function plot_backbone_shift(r_range, Ω_curves, ω₀_vals, branches, labels;
                             Ω_curves_ref = nothing,
                             ω₀_vals_ref  = nothing,
                             colors       = nothing,
                             xlabel = "Nonlinear frequency shift  Ω − ω₀  (rad/s)",
                             ylabel = "Modal amplitude  |z₁|",
                             title  = "Backbone shift",
                             r_max  = maximum(r_range),
                             legend = :best,
                             size   = (800, 600), dpi = 150)
    clrs = isnothing(colors) ? palette(:tab10) : colors
    plt  = plot(; xlabel, ylabel, title, ylims = (0, r_max * 1.05), size, dpi, legend)
    for (k, (Ωcurve, ω₀, br, lbl)) in enumerate(zip(Ω_curves, ω₀_vals, branches, labels))
        plot!(plt, Ωcurve .- ω₀, collect(r_range); lw = 2.0, color = clrs[k], label = lbl)
        if !isnothing(Ω_curves_ref) && !isnothing(ω₀_vals_ref)
            plot!(plt, Ω_curves_ref[k] .- ω₀_vals_ref[k], collect(r_range);
                  lw = 1.2, ls = :dash, color = clrs[k], label = nothing)
        end
        if !isnothing(br) && length(br.branch) > 1
            r_bk = [s.r for s in br.branch]
            Ω_bk = [s.Ω for s in br.branch]
            mask = r_bk .<= r_max
            scatter!(plt, Ω_bk[mask] .- ω₀, r_bk[mask];
                     color = clrs[k], ms = 3, markerstrokewidth = 0, label = nothing)
        end
    end
    return plt
end

function plot_eigenfrequency_vs_parameter(
    param_main, ω_main, label_main;
    param_ref   = nothing,
    ω_ref       = nothing,
    label_ref   = nothing,
    ω₀_ref_val  = nothing,
    hline_label = nothing,
    vline_at    = nothing,
    xlabel      = "Parameter",
    ylabel      = "Linear eigenfrequency  ω₀  (rad/ms)",
    title       = "Eigenfrequency vs parameter",
    main_color  = :royalblue, main_lw = 2.5,
    ref_color   = :darkorange, ref_ls  = :dashdot,
    size = (700, 480), dpi = 150, legend = :top)

    plt = plot(param_main, ω_main;
               xlabel, ylabel, title,
               lw = main_lw, color = main_color, label = label_main,
               legend, size, dpi)
    if !isnothing(ω_ref)
        pv = isnothing(param_ref) ? param_main : param_ref
        plot!(plt, pv, ω_ref; lw = 2.0, ls = ref_ls, color = ref_color, label = label_ref)
    end
    if !isnothing(ω₀_ref_val)
        hline!(plt, [ω₀_ref_val]; ls = :dash, color = :gray, lw = 1.2, label = hline_label)
    end
    if !isnothing(vline_at)
        vline!(plt, [vline_at]; ls = :dot, color = :black, lw = 0.8, label = nothing)
    end
    return plt
end
