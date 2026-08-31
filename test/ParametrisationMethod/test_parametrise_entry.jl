using StaticArrays

@testset "parametrise entry validation" begin
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = Matrix{Float64}(I, 2, 2)
    B1 = 0.001 .* B2
    cubic = MultilinearMap(
        (res, x1, x2, x3) -> (@. res += -x1 * x2 * x3), (3, 0))
    model = NthOrderModel((B0, B1, B2), (cubic,))
    ep = spectrum(model; solver = DefaultEigensolver())
    spectral = SpectralData(model, ep; master = master_by_sorting(2))
    resonance = ResonanceConfig(
        style = :complex_normal_form, tol = 0.05, warn_outer = false)
    options = ParametrisationOptions(show_progress = false, verbose = false)

    W0, R0 = parametrise(model, spectral, 3; resonance, options)
    full3 = all_multiindices_up_to(2, 3; min_degree = 1)
    W1, R1 = parametrise(model, spectral, full3; resonance, options)
    @test W1.poly.coefficients == W0.poly.coefficients
    @test R1.poly.coefficients == R0.poly.coefficients

    @test build_multiindex_set(full3, 2) === full3
    @test_throws ArgumentError build_multiindex_set(:cubic, 2)
    @test_throws AssertionError build_multiindex_set(0, 2)

    bad_nvar = all_multiindices_up_to(3, 3; min_degree = 1)
    with_zero = all_multiindices_up_to(2, 3; min_degree = 0)
    no_unit = MultiindexSet([e for e in full3.exponents if Tuple(e) != (0, 1)])
    not_closed = delete_multiindices(full3, [[2, 0]])
    not_conj = delete_multiindices(full3, [[1, 2]])
    @test_throws ArgumentError parametrise(model, spectral, bad_nvar; resonance, options)
    @test_throws ArgumentError parametrise(model, spectral, with_zero; resonance, options)
    @test_throws ArgumentError parametrise(model, spectral, no_unit; resonance, options)
    @test !is_downward_closed(not_closed)
    @test_throws ArgumentError parametrise(model, spectral, not_closed; resonance, options)
    @test is_downward_closed(not_conj)
    @test !is_conjugate_closed(not_conj, [2, 1])
    @test_throws ArgumentError parametrise(model, spectral, not_conj;
        resonance, conjugate_permutation = [2, 1], options)

    Wc, Rc = parametrise(model, spectral, 3;
        resonance, conjugate_permutation = [2, 1], options)
    @test Wc.poly.coefficients ≈ W0.poly.coefficients
    @test Rc.poly.coefficients ≈ R0.poly.coefficients
    @test_throws ArgumentError parametrise(model, spectral, full3;
        resonance, conjugate_permutation = [1, 1], options)

    unchecked = ParametrisationOptions(
        validate_mset = false, show_progress = false, verbose = false)
    @test unchecked.validate_mset === false
end
