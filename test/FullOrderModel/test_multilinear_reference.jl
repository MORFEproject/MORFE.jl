using MORFE.MultilinearTerms: build_multilinear_terms_cache, compute_multilinear_terms!
using MORFE.ParametrisationMethod: create_parametrisation_method_objects
using MORFE.Polynomials: find_in_multiindex_set

function _reference_linear_external(A)
    mset = all_multiindices_up_to(size(A, 1), 1)
    coefficients = zeros(eltype(A), size(A, 1), length(mset))
    polynomial = DensePolynomial(coefficients, mset)
    for j in axes(A, 2)
        exponent = zeros(Int, size(A, 1))
        exponent[j] = 1
        coefficients[:, find_in_multiindex_set(polynomial, exponent)] .= A[:, j]
    end
    return ExternalSystem(DensePolynomial(coefficients, mset))
end

function _check_reference_evaluator(external_system)
    qsym = MultilinearMap(
        (r, x, y) -> (r .+= x .* y), (2, 0); fully_asymmetric = false)
    qasym = MultilinearMap(
        (r, x, y) -> (r .+= 2 .* x .+ 3 .* y), (2, 0); fully_asymmetric = true)
    mixed = MultilinearMap(
        (r, x, xd) -> (r .+= x .* xd), (1, 1))
    external = MultilinearMap(
        (r, x, e) -> (r .+= x .* sum(e)), (1, 0), 1;
        fully_asymmetric = false)
    K = zeros(ComplexF64, 2, 2)
    model = NthOrderModel((K, K, K), (qsym, qasym, mixed, external), external_system)
    ROM, N_EXT = 2, 2
    mset = all_multiindices_up_to(ROM + N_EXT, 3; min_degree = 1)
    W, _ = create_parametrisation_method_objects(
        mset, 2, 2, ROM, N_EXT, ComplexF64)
    W.poly.coefficients .= reshape(
        ComplexF64[0.01i + 0.003im * i for i in eachindex(W.poly.coefficients)],
        size(W.poly.coefficients))
    cache = build_multilinear_terms_cache(model, W)
    for index in eachindex(mset.exponents)
        cached = zeros(ComplexF64, 2)
        compute_multilinear_terms!(cached, model, index, W, cache)
        reference = compute_multilinear_terms(model, mset[index], W)
        @test cached ≈ reference atol=1e-12
    end
end

@testset "non-cached multilinear reference evaluator" begin
    _check_reference_evaluator(ExternalSystem((1im, -1im)))
    rebased = _reference_linear_external(ComplexF64[0 -1; 1 0])
    @test external_basis(rebased) !== nothing
    _check_reference_evaluator(rebased)
end
