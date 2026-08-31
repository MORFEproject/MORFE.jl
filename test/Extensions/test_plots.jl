using Plots

@testset "invariance convergence plots" begin
    radii = 10.0 .^ range(-4, -1; length = 6)
    results = [(max_order = 3, radii = radii .* (1 + level),
                   radii_master = radii, force_errors = (1 + level) .* radii .^ 4,
                   state_errors = (2 + level) .* radii .^ 4,
                   r_magnitude = level == 0 ? 0.0 : 0.2)
               for level in 0:1]

    full = plot_invariance_convergence(results; x_axis = :full,
        show_state_errors = false, show_regression = true)
    master = plot_invariance_convergence(results; x_axis = :master,
        show_state_errors = true, show_regression = true)
    both = plot_invariance_convergence(results; x_axis = :both,
        show_state_errors = false, show_regression = false, title = "coverage")
    @test full isa Plots.Plot
    @test master isa Plots.Plot
    @test both.full isa Plots.Plot
    @test both.master isa Plots.Plot
    @test_throws ErrorException plot_invariance_convergence([])
    @test_throws ErrorException plot_invariance_convergence(results; x_axis = :radius)
end
