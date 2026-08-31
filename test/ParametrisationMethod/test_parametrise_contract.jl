# The `mset` contract enforced by `parametrise(model, spectral, expansion_order)` and by
# the solve it delegates to.
#
# The two closure clauses are the ones worth pinning, because they fail differently: a
# missing divisor is read as zero and silently corrupts the right-hand side, while a missing
# conjugate partner only loses the pairing optimisation. Both were documented as required
# long before they were enforced.

using Test
using LinearAlgebra
using StaticArrays

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, SpectralData
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.Resonance: ResonanceConfig, build_resonance_set
using MORFE.Multiindices: MultiindexSet, find_in_set

@testset "parametrise contract" begin
    # ── minimal 2-DOF Duffing model (mirrors test_conjugate_symmetry.jl) ──
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = [1.0 0.0; 0.0 1.0]
    B1 = 0.001 * B2
    term_cubic = MultilinearMap(
        (res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3),
        (3, 0)
    )
    model = NthOrderModel((B0, B1, B2), (term_cubic,))
    ROM = 2
    order = 5

    ep = spectrum(model; solver = DefaultEigensolver())
    sd = SpectralData(model, ep; master = master_by_sorting(ROM))
    cnf = ResonanceConfig(style = :complex_normal_form, tol = 0.05, warn_outer = false)

    W0, R0 = parametrise(model, sd, order; resonance = cnf)

    @testset "parametrise ≡ its own steps, run by hand" begin
        # A genuine restriction of order 5, so the set is doing work.
        mset = all_multiindices_up_to(ROM, 3; min_degree = 1)
        W1, R1 = parametrise(model, sd, mset; resonance = cnf)

        rset = build_resonance_set(model, mset, sd, cnf)
        W2, R2 = solve_cohomological_problem(model, mset, sd, rset)
        @test R1.poly.coefficients == R2.poly.coefficients
        @test W1.poly.coefficients == W2.poly.coefficients
    end

    @testset "conjugate_permutation pass-through" begin
        Wp, Rp = parametrise(model, sd, order; resonance = cnf,
            conjugate_permutation = [2, 1])
        # Symmetry enforcement must agree with the plain solve on this
        # conjugate-symmetric model.
        @test isapprox(Rp.poly.coefficients, R0.poly.coefficients;
            rtol = 1e-8, atol = 1e-10)
    end

    @testset "malformed msets throw" begin
        # wrong number of variables
        bad_nvar = all_multiindices_up_to(3, 3; min_degree = 1)
        @test_throws ArgumentError parametrise(model, sd, bad_nvar; resonance = cnf)
        # contains the zero multiindex
        with_zero = all_multiindices_up_to(ROM, 3; min_degree = 0)
        @test_throws ArgumentError parametrise(model, sd, with_zero; resonance = cnf)
        # missing a unit multiindex
        no_unit = MultiindexSet([e
                                 for e in all_multiindices_up_to(ROM, 3; min_degree = 1).exponents
                                 if Tuple(e) != (0, 1)])
        @test_throws ArgumentError parametrise(model, sd, no_unit; resonance = cnf)
    end

    full3 = all_multiindices_up_to(ROM, 3; min_degree = 1)

    @testset "mset must be downward closed" begin
        # Dropping [2,0] leaves [3,0] with a missing divisor; every unit survives, so
        # this isolates the closure clause from the earlier ones.
        not_closed = delete_multiindices(full3, [[2, 0]])
        @test !is_downward_closed(not_closed)
        @test find_in_set(not_closed, [1, 0]) !== nothing   # units still present
        @test_throws ArgumentError parametrise(model, sd, not_closed; resonance = cnf)

        # The solve is where every path lands, so it must reject the set too — that is
        # what `validate_mset = true` on the low-level entry is for.
        rset = build_resonance_set(model, not_closed, sd, cnf)
        @test_throws ArgumentError solve_cohomological_problem(
            model, not_closed, sd, rset)
    end

    @testset "mset must be closed under conjugate_permutation" begin
        # Drop [1,2] but keep [2,1]: still downward closed (nothing retained divides
        # into [1,2]), so only the conjugate clause can fire.
        not_conj = delete_multiindices(full3, [[1, 2]])
        @test is_downward_closed(not_conj)
        @test !is_conjugate_closed(not_conj, [2, 1])

        @test_throws ArgumentError parametrise(model, sd, not_conj; resonance = cnf,
            conjugate_permutation = [2, 1])

        # Without a permutation the same set is perfectly legal.
        @test parametrise(model, sd, not_conj; resonance = cnf) isa Tuple
    end

    @testset "malformed conjugate_permutation throws" begin
        @test_throws ArgumentError parametrise(model, sd, full3; resonance = cnf,
            conjugate_permutation = [2, 1, 3])          # wrong length
        @test_throws ArgumentError parametrise(model, sd, full3; resonance = cnf,
            conjugate_permutation = [1, 1])             # not a permutation
    end

    @testset "validate_mset = false skips the checks" begin
        not_closed = delete_multiindices(full3, [[2, 0]])
        # The answer is meaningless by construction — we assert only that the escape
        # hatch reaches the solver instead of throwing.
        @test parametrise(model, sd, not_closed; resonance = cnf,
            options = ParametrisationOptions(validate_mset = false)) isa Tuple
    end
end
