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

        spectral = SpectralData(; eigenvalues = ComplexF64[im, -im],
            right_modes = Matrix{ComplexF64}(I, 2, 2),
            left_modes = Matrix{ComplexF64}(I, 2, 2), conjugate_permutation = [2, 1])
        result = only(invariance_error_convergence(
            model, W, R; spectral, n_samples = 32, rng = MersenneTwister(91)))
        # The cloud is realified, z₂ = conj(z₁), so ‖E‖ = √2|z₁|² = ‖z‖²/√2 at every
        # point and the fitted rate is two exactly.
        @test result.convergence_rate ≈ 2 atol=1e-10
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

    @testset "frequency-matched state error" begin
        # A conservative conjugate pair on dofs 1 and 2 (eigenvalues ±i) and a slave dof 3
        # (eigenvalue -2). With W(z) = (z₁, z₂, 0), a quadratic force x₁x₂ on `dof` leaves
        # the residual E = -z₁z₂ e_dof. The orbit frequency is s̄ = i exactly, whereas both
        # members of the pair together would cancel to s̄ ≈ 0.
        mset = all_multiindices_up_to(2, 1; min_degree = 1)
        i1, i2 = _unit_index(mset, 1), _unit_index(mset, 2)
        modes = Matrix{ComplexF64}(I, 3, 2)
        spectral = SpectralData(; eigenvalues = ComplexF64[im, -im], right_modes = modes,
            left_modes = modes, conjugate_permutation = [2, 1])
        function state_error_of(dof; spectral = spectral)
            force = MultilinearMap(
                (result, x, y) -> (result[dof] += (x[1] * y[2] + y[1] * x[2]) / 2), (2,);
                fully_asymmetric = false)
            model = NthOrderModel(
                (ComplexF64[-im 0 0; 0 im 0; 0 0 2], Matrix{ComplexF64}(I, 3, 3)), (force,))
            W, R = create_parametrisation_method_objects(mset, 1, 3, 2, 0, ComplexF64)
            W.poly.coefficients[1, 1, i1] = 1
            W.poly.coefficients[2, 1, i2] = 1
            R.poly.coefficients[1, i1] = im
            R.poly.coefficients[2, i2] = -im
            return only(invariance_error_convergence(
                model, W, R; spectral, n_samples = 64, rng = MersenneTwister(5)))
        end

        # On the slave dof, the state error is E₃ / L₃₃(s̄) = E₃ / (i + 2): matched to the
        # orbit, not the quasi-static E₃ / 2. L(s̄) is singular on dof 1 at s̄ = i, which
        # the border keeps out of the solve.
        slave = state_error_of(3)
        @test slave.s_bar == im
        @test all(isfinite, slave.state_errors)
        @test slave.state_errors ≈ slave.force_errors ./ sqrt(5) rtol=1e-12

        # Without a conjugate permutation, the pair is found by its eigenvalues.
        unpaired = SpectralData(; eigenvalues = ComplexF64[im, -im], right_modes = modes,
            left_modes = modes)
        @test state_error_of(3; spectral = unpaired).s_bar == im

        # Along a master mode, the residual is absorbed by the reduced dynamics.
        master = state_error_of(1)
        @test maximum(master.force_errors) > 0.1
        @test maximum(master.state_errors) ≤ 1e-14 * maximum(master.force_errors)
    end
end
