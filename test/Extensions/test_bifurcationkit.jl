# The BifurcationKit extension poses the reduced dynamics ż = R(z, p) in the real form
# x = [Re z; Im z], which every supported BifurcationKit version can continue.
using BifurcationKit
using MORFE.Polynomials: evaluate

@testset "make_bk_problem: the real form of the reduced dynamics" begin
    # Two coupled oscillators with a cubic term and a stiffness that depends on a frozen
    # parameter. Every term of R carries a master coordinate, so z = 0 is an equilibrium
    # for every parameter value: the branch is known exactly.
    K = [2.0 -1.0; -1.0 2.0]
    C = 0.001 .* Matrix(1.0I, 2, 2)
    M = Matrix(1.0I, 2, 2)
    K1 = [0.3 0.1; 0.1 0.2]
    cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -x1 * x2 * x3), (3, 0))
    stiffness = MultilinearMap((res, x, r) -> (res .-= r[1] .* (K1 * x)), (1, 0), 1)
    model = NthOrderModel((K, C, M), (cubic, stiffness), ExternalSystem((0.0,)))
    ep = spectrum(NthOrderModel((K, C, M)); solver = DefaultEigensolver())
    spectral = SpectralData(model, ep; master = master_by_sorting(2))
    _, R = parametrise(model, spectral, all_multiindices_up_to(3, 3; min_degree = 1);
        resonance = ResonanceConfig(
            style = :complex_normal_form, tol = 0.05, warn_outer = false),
        options = ParametrisationOptions(show_progress = false, verbose = false))
    ROM = 2

    prob = make_bk_problem(R)
    @test prob.u0 == zeros(2ROM)
    z0 = ComplexF64[1e-3 + 2e-3im, 1e-3 - 2e-3im]
    @test make_bk_problem(R; z0).u0 == [1e-3, 1e-3, 2e-3, -2e-3]

    # The real form evaluates to the real and imaginary parts of R.
    z, p = ComplexF64[0.05 + 0.02im, 0.05 - 0.02im], [0.1]
    rz = evaluate(R.poly, vcat(z, p))[1:ROM]
    @test BifurcationKit.residual(prob, vcat(real.(z), imag.(z)), p) ≈
          vcat(real.(rz), imag.(rz))

    branch = continuation(prob, PALC(),
        ContinuationPar(p_min = -0.2, p_max = 0.2, ds = 0.01, dsmax = 0.02,
            max_steps = 30, detect_bifurcation = 0))
    @test length(branch) > 2
    @test maximum(branch.param) ≈ 0.2
end
