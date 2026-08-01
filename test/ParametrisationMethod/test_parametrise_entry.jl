# Tests for the high-level parametrise(model, order, eigenproblem; …) entry:
# the new mset / conjugate_permutation / external_eigenvalues kwargs must
# default to the old behaviour and, with a custom mset, reproduce the
# hand-rolled solve_cohomological_problem pipeline exactly.

using Test
using LinearAlgebra
using StaticArrays

using MORFE
using MORFE.FullOrderModel: NDOrderModel, MultilinearMap
using MORFE.Eigenproblems: solve_eigenproblem, DefaultEigensolver,
                           select_master_modes_by_sorting
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.Multiindices: MultiindexSet

@testset "parametrise entry kwargs" begin
    # ── minimal 2-DOF Duffing model (mirrors test_conjugate_symmetry.jl) ──
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = [1.0 0.0; 0.0 1.0]
    B1 = 0.001 * B2
    term_cubic = MultilinearMap(
        (res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3),
        (3, 0)
    )
    model = NDOrderModel((B0, B1, B2), (term_cubic,))
    ROM = 2
    order = 5

    ep = solve_eigenproblem(model; solver = DefaultEigensolver())
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
end
