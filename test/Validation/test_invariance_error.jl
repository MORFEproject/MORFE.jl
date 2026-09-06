using LinearAlgebra
using Random: MersenneTwister
using Test
using MORFE

function _unit_index(mset, variable)
    return only(i
    for (i, exponent) in enumerate(mset.exponents)
    if exponent[variable] == 1 && sum(exponent) == 1)
end

@testset "Invariance-error validation" begin
    @testset "point-cloud slope and machine-floor trimming" begin
        # Two conjugate linear modes with an intentionally omitted quadratic
        # manifold term. The exact residual of W(z)=[z1,z2] is -W.*W and hence
        # scales quadratically over the complete point cloud.
        B0 = ComplexF64[-im 0; 0 im]
        B1 = Matrix{ComplexF64}(I, 2, 2)
        quadratic = MultilinearMap(
            (result, x1, x2) -> (result .+= x1 .* x2), (2,);
            fully_asymmetric = false)
        model = NthOrderModel((B0, B1), (quadratic,))
        mset = all_multiindices_up_to(2, 1; min_degree = 1)
        W, R = create_parametrisation_method_objects(mset, 1, 2, 2, 0, ComplexF64)
        i1, i2 = _unit_index(mset, 1), _unit_index(mset, 2)
        W.poly.coefficients[1, 1, i1] = 1
        W.poly.coefficients[2, 1, i2] = 1
        R.poly.coefficients[1, i1] = im
        R.poly.coefficients[2, i2] = -im

        result = only(invariance_error_convergence(
            model, W, R; n_samples = 32, rng = MersenneTwister(91)))
        # A Gaussian cloud changes both radius and direction, hence also the
        # direction-dependent quadratic prefactor.  The fitted rate should be
        # close to two, but it is not algebraically equal to two.
        @test result.convergence_rate ≈ 2 atol=0.1
        @test all(isfinite, result.force_errors)

        # The production estimator removes low-radius samples from the left
        # while that improves the slope. Saturated points must therefore not
        # deflate an otherwise exact eighth-order cloud.
        radii = 10.0 .^ range(-8, -1; length = 40)
        errors = radii .^ 8
        errors[1:10] .= eps(Float64)
        slope, _ = MORFE.InvarianceError._log_log_regression(radii, errors)
        @test slope ≈ 8 atol=1e-12
    end

    @testset "fixed external target reaches W, R, and the FOM" begin
        # xdot+x=r, zdot=-z, rdot=0 has the exact invariant map W(z,r)=z+r.
        # This test fails sharply if the FOM sees r_target while W/R see r=0.
        forcing = MultilinearMap(
            (result, r) -> (result[1] += r[1]), (0,), 1;
            fully_asymmetric = false)
        external = ExternalSystem((0.0 + 0im,))
        model = NthOrderModel((ones(1, 1), ones(1, 1)), (forcing,), external)
        mset = all_multiindices_up_to(2, 1; min_degree = 1)
        W, R = create_parametrisation_method_objects(mset, 1, 1, 1, 1, ComplexF64)
        iz, ir = _unit_index(mset, 1), _unit_index(mset, 2)
        W.poly.coefficients[1, 1, iz] = 1
        W.poly.coefficients[1, 1, ir] = 1
        R.poly.coefficients[1, iz] = -1

        target = ComplexF64[0.3]
        workspace = InvarianceErrorWorkspace(model, W)
        defect = zeros(ComplexF64, 1)
        z = ComplexF64[0.2, target[1]]
        invariance_error_residual!(
            defect, workspace, model, W, R, z; r_external = target)
        @test norm(defect) ≤ 10eps(Float64)

        cloud = invariance_error_norms(
            model, W, R; n_samples = 16, amplitude = 0.2,
            r_external = target, rng = MersenneTwister(7))
        @test cloud.max ≤ 100eps(Float64)
        @test_throws ArgumentError invariance_error_residual!(
            defect, workspace, model, W, R, ComplexF64[0.2, 0.0];
            r_external = target)
    end
end
