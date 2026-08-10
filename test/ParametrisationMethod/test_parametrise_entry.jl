# Tests for the high-level parametrise(model, order, eigenproblem; …) entry:
# the new mset / conjugate_permutation / external_eigenvalues kwargs must
# default to the old behaviour and, with a custom mset, reproduce the
# hand-rolled solve_cohomological_problem pipeline exactly.

using Test
using LinearAlgebra
using StaticArrays

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver,
                                   select_master_modes_by_sorting
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.Multiindices: MultiindexSet, find_in_set

@testset "parametrise entry kwargs" begin
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
    select_master_modes_by_sorting(ep, ROM)

    W0, R0 = parametrise(model, order, ep;
        resonance = :complex_normal_form, resonance_tol = 0.05)

    @testset "explicit default mset ≡ defaulted call" begin
        mset = all_multiindices_up_to(ROM, order; min_degree = 1)
        W1, R1 = parametrise(model, order, ep;
            resonance = :complex_normal_form, resonance_tol = 0.05, mset = mset)
        @test R1.poly.coefficients == R0.poly.coefficients
        @test W1.poly.coefficients == W0.poly.coefficients
    end

    @testset "custom mset ≡ hand-rolled solve" begin
        # anisotropic subset: total degree ≤ 3 (a genuine restriction of order 5)
        mset = all_multiindices_up_to(ROM, 3; min_degree = 1)
        W1, R1 = parametrise(model, 3, ep;
            resonance = :complex_normal_form, resonance_tol = 0.05, mset = mset)

        # replicate what the entry point assembles
        master_mask = ep.master_modes
        master_eigs = SVector{ROM, ComplexF64}(ep.eigenvalues[master_mask])
        master_modes = ep.eigenmodes[:, 1, master_mask]
        left_modes = ep.left_eigenmodes[:, master_mask]
        mmd = @view(ep.eigenmodes[:, 2:end, master_mask])
        lmd = @view(ep.left_eigenmodes_orders[:, 1:1, master_mask])
        rset = MORFE.Resonance.build_resonance_set(model, :complex_normal_form,
            mset, ep, 0.05, nothing)
        W2, R2 = solve_cohomological_problem(model, mset, master_eigs,
            master_modes, left_modes, rset;
            master_modes_derivatives = mmd, left_modes_derivatives = lmd)
        @test R1.poly.coefficients == R2.poly.coefficients
        @test W1.poly.coefficients == W2.poly.coefficients
    end

    @testset "conjugate_permutation pass-through" begin
        Wp, Rp = parametrise(model, order, ep;
            resonance = :complex_normal_form, resonance_tol = 0.05,
            conjugate_permutation = [2, 1])
        # symmetry enforcement must agree with the plain solve on this
        # conjugate-symmetric model
        @test isapprox(Rp.poly.coefficients, R0.poly.coefficients;
            rtol = 1e-8, atol = 1e-10)
    end

    @testset "malformed msets throw" begin
        # wrong number of variables
        bad_nvar = all_multiindices_up_to(3, 3; min_degree = 1)
        @test_throws ArgumentError parametrise(model, 3, ep; mset = bad_nvar)
        # contains the zero multiindex
        with_zero = all_multiindices_up_to(ROM, 3; min_degree = 0)
        @test_throws ArgumentError parametrise(model, 3, ep; mset = with_zero)
        # missing a unit multiindex
        no_unit = MultiindexSet([e
                                 for e in all_multiindices_up_to(ROM, 3; min_degree = 1).exponents
                                 if Tuple(e) != (0, 1)])
        @test_throws ArgumentError parametrise(model, 3, ep; mset = no_unit)
    end

    # The two closure clauses were documented as required long before they were
    # enforced; these pin the enforcement.
    full3 = all_multiindices_up_to(ROM, 3; min_degree = 1)

    @testset "mset must be downward closed" begin
        # Dropping [2,0] leaves [3,0] with a missing divisor; every unit survives, so
        # this isolates the closure clause from the earlier ones.
        not_closed = delete_multiindices(full3, [[2, 0]])
        @test !is_downward_closed(not_closed)
        @test find_in_set(not_closed, [1, 0]) !== nothing   # units still present
        @test_throws ArgumentError parametrise(model, 3, ep; mset = not_closed)

        # The low-level entry is the path MORFEFerrite uses — it must reject too.
        master_mask = ep.master_modes
        master_eigs = SVector{ROM, ComplexF64}(ep.eigenvalues[master_mask])
        master_modes = ep.eigenmodes[:, 1, master_mask]
        left_modes = ep.left_eigenmodes[:, master_mask]
        mmd = @view(ep.eigenmodes[:, 2:end, master_mask])
        lmd = @view(ep.left_eigenmodes_orders[:, 1:1, master_mask])
        rset = MORFE.Resonance.build_resonance_set(model, :complex_normal_form,
            not_closed, ep, 0.05, nothing)
        @test_throws ArgumentError solve_cohomological_problem(model, not_closed,
            master_eigs, master_modes, left_modes, rset;
            master_modes_derivatives = mmd, left_modes_derivatives = lmd)
    end

    @testset "mset must be closed under conjugate_permutation" begin
        # Drop [1,2] but keep [2,1]: still downward closed (nothing retained divides
        # into [1,2]), so only the conjugate clause can fire.
        not_conj = delete_multiindices(full3, [[1, 2]])
        @test is_downward_closed(not_conj)
        @test !is_conjugate_closed(not_conj, [2, 1])

        @test_throws ArgumentError parametrise(model, 3, ep;
            resonance = :complex_normal_form, resonance_tol = 0.05,
            mset = not_conj, conjugate_permutation = [2, 1])

        # Without a permutation the same set is perfectly legal.
        @test parametrise(model, 3, ep; resonance = :complex_normal_form,
            resonance_tol = 0.05, mset = not_conj) isa Tuple
    end

    @testset "malformed conjugate_permutation throws" begin
        @test_throws ArgumentError parametrise(model, 3, ep;
            mset = full3, conjugate_permutation = [2, 1, 3])          # wrong length
        @test_throws ArgumentError parametrise(model, 3, ep;
            mset = full3, conjugate_permutation = [1, 1])             # not a permutation
    end

    @testset "validate_mset = false skips the checks" begin
        not_closed = delete_multiindices(full3, [[2, 0]])
        # The answer is meaningless by construction — we assert only that the escape
        # hatch reaches the solver instead of throwing.
        @test parametrise(model, 3, ep; resonance = :complex_normal_form,
            resonance_tol = 0.05, mset = not_closed,
            validate_mset = false) isa Tuple
    end
end
