# `SpectralData` must reproduce the hand-assembled spectral arguments exactly.
#
# These are equality tests, not `≈` tests, and deliberately so: the failure modes this
# guards against are index swaps and slice mix-ups (the right blocks index from the
# front, the left blocks from the back), which produce plausible-looking but wrong
# numbers. Only bit-equality catches them reliably.

using Test
using LinearAlgebra
using StaticArrays

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver,
                                   select_master_modes_by_sorting,
                                   left_eigenmode_orders_from_slice
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.Resonance: build_resonance_set
using MORFE.SpectralDecomposition: SpectralData, ModeBundle, check_biorthogonality,
                                   right_modes, left_modes,
                                   right_mode_derivatives, left_mode_blocks,
                                   master_eigenvalues, outer_eigenvalues,
                                   master_bundle, outer_bundle, indices,
                                   master_conjugate_permutation,
                                   outer_conjugate_permutation,
                                   physical_mode, spectrum_entries

function _cubic(nd)
    MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3),
        ntuple(i -> i == 1 ? 3 : 0, nd))
end

@testset "SpectralData" begin
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = [1.0 0.0; 0.0 1.0]
    B1 = 0.001 * B2
    ROM = 2

    model = NthOrderModel((B0, B1, B2), (_cubic(2),))
    ep = spectrum(model; solver = DefaultEigensolver())
    select_master_modes_by_sorting(ep, ROM)
    mask = ep.master_modes
    idx = findall(mask)

    λ = SVector{ROM, ComplexF64}(ep.eigenvalues[mask])
    Ψ = ep.eigenmodes[:, 1, mask]
    ℓ = ep.left_eigenmodes[:, mask]
    mmd = Array(ep.eigenmodes[:, 2:end, mask])
    lmd = Array(ep.left_eigenmodes_orders[:, 1:(end - 1), mask])

    @testset "reconciles to the hand-assembled arguments" begin
        sd = SpectralData(model, ep; master = idx)
        @test master_eigenvalues(sd) == λ
        @test right_modes(sd) == Ψ
        @test left_modes(sd) == ℓ
        @test right_mode_derivatives(sd) == mmd
        @test left_mode_blocks(sd) == lmd
        # Non-master eigenvalues land in the outer bundle, in spectrum order.
        @test outer_eigenvalues(sd) ==
              ep.eigenvalues[setdiff(1:length(ep.eigenvalues), idx)]
    end

    @testset "mirrored index convention" begin
        sd = SpectralData(model, ep; master = idx)
        # Right physical is the FIRST order-block; left physical is the LAST. Getting
        # these the same way round is the bug this asserts against.
        @test right_modes(sd) == sd.master.right_blocks[:, 1, :]
        @test left_modes(sd) == sd.master.left_blocks[:, end, :]
        @test right_modes(sd) != sd.master.right_blocks[:, end, :]

        # Biorthogonality is the numerical guard: any swap destroys G ≈ I.
        @test isapprox(check_biorthogonality(sd, model), I; atol = 1e-10)
    end

    @testset "master selection: mask ≡ indices, order preserved" begin
        by_mask = SpectralData(model, ep; master = mask)
        by_idx = SpectralData(model, ep; master = idx)
        @test master_eigenvalues(by_mask) == master_eigenvalues(by_idx)

        # Index ORDER is the reduced-coordinate order and must not be sorted away.
        rev = SpectralData(model, ep; master = reverse(idx))
        @test master_eigenvalues(rev) == reverse(master_eigenvalues(by_idx))
        @test right_modes(rev) == right_modes(by_idx)[:, [2, 1]]
    end

    @testset "conjugate permutation" begin
        @test SpectralData(model, ep; master = idx).conjugate_permutation === nothing
        @test master_conjugate_permutation(SpectralData(model, ep; master = idx)) ===
              nothing

        # The STORED involution spans the whole spectrum; the master block is derived.
        # This 2-DOF model has four eigenvalues in two conjugate pairs.
        det = SpectralData(model, ep; master = idx, conjugate_permutation = :detect)
        @test det.conjugate_permutation == [2, 1, 4, 3]      # σ over 1:n_eigs
        @test master_conjugate_permutation(det) == [2, 1]    # restricted to 1:ROM
        @test outer_conjugate_permutation(det) == [2, 1]     # restricted to 1:n_outer

        # An explicit master block is still accepted in ROM-length form and widened; the
        # outer entries stay self-paired because the caller stated only the master pairing.
        exp = SpectralData(model, ep; master = idx, conjugate_permutation = [2, 1])
        @test master_conjugate_permutation(exp) == [2, 1]
        @test exp.conjugate_permutation == [2, 1, 3, 4]

        # Master block only — a full NVAR-length vector is the wrong length here.
        @test_throws ArgumentError SpectralData(model, ep; master = idx,
            conjugate_permutation = [2, 1, 4, 3])
        @test_throws ArgumentError SpectralData(model, ep; master = idx,
            conjugate_permutation = [1, 1])

        # Selecting half a conjugate pair leaves no symmetry to restrict, and must be
        # rejected at construction where the offending entry can be named — not later,
        # inside the solve.
        @test_throws ArgumentError SpectralData(model, ep; master = [1, 3],
            conjugate_permutation = :detect)
    end

    @testset "physical mode numbers survive non-adjacent pairs" begin
        # The case that motivates deriving mode numbers from σ's orbits rather than
        # computing ⌈i/2⌉: a spectrum whose conjugate partners are NOT adjacent, which a
        # shift-invert or filtered eigensolver can return.
        λna = ComplexF64[1 + 2im, 3 + 4im, 1 - 2im, 3 - 4im]   # pairs {1,3} and {2,4}
        Ψna = ComplexF64[1 0 1 0; 0 1 0 1; 1 1 1 1]
        Ψna[:, 3] .= conj.(Ψna[:, 1])
        Ψna[:, 4] .= conj.(Ψna[:, 2])
        sd = SpectralData(; eigenvalues = λna[1:2], right_modes = Ψna[:, 1:2],
            left_modes = Ψna[:, 1:2], outer_eigenvalues = λna[3:4],
            conjugate_permutation = nothing)

        # Raw-array construction numbers masters 1:ROM then the outer entries after them,
        # so here the spectrum order is [λ₁, λ₂, λ₃, λ₄] as given.
        @test indices(master_bundle(sd)) == [1, 2]
        @test indices(outer_bundle(sd)) == [3, 4]

        # With σ pairing {1,3} and {2,4}, mode numbering by first appearance gives
        # entry 1 → mode 1, entry 2 → mode 2, entry 3 → mode 1, entry 4 → mode 2.
        # ⌈i/2⌉ would wrongly give 1,1,2,2.
        σ = [3, 4, 1, 2]
        @test MORFE.SpectralDecomposition._mode_numbers(σ, 4) == [1, 2, 1, 2]
        @test MORFE.SpectralDecomposition._mode_numbers(σ, 4) != [1, 1, 2, 2]

        # Adjacent pairs must still give the conventional numbering.
        @test MORFE.SpectralDecomposition._mode_numbers([2, 1, 4, 3], 4) == [1, 1, 2, 2]
        # A real (self-paired) eigenvalue is its own mode.
        @test MORFE.SpectralDecomposition._mode_numbers([2, 1, 3], 3) == [1, 1, 2]
        # No conjugate structure at all: every entry is its own mode.
        @test MORFE.SpectralDecomposition._mode_numbers(nothing, 3) == [1, 2, 3]
    end

    @testset "solve ≡ positional path" begin
        order = 5
        mset = all_multiindices_up_to(ROM, order; min_degree = 1)
        rset = build_resonance_set(model, :complex_normal_form, mset, ep, 0.05, nothing)
        for perm in (nothing, [2, 1])
            Wa, Ra = solve_cohomological_problem(model, mset, λ, Ψ, ℓ, rset;
                master_modes_derivatives = mmd, left_modes_derivatives = lmd,
                conjugate_permutation = perm, show_progress = false)
            sd = SpectralData(model, ep; master = idx, conjugate_permutation = perm)
            Wb, Rb = solve_cohomological_problem(model, mset, sd, rset;
                show_progress = false)
            @test Wa.poly.coefficients == Wb.poly.coefficients
            @test Ra.poly.coefficients == Rb.poly.coefficients
        end
    end

    @testset "ORD mismatch: ORD-3 model fed by an ORD-2 eigenproblem" begin
        # The MORFEFerrite examples 04/07 case. Right blocks are extended by multiplying
        # the LAST AVAILABLE block by λ — not by forming a fresh λ^{k-1}ψ — and left
        # blocks are rebuilt against the augmented linear_terms.
        Z = zeros(size(B0))
        aug = NthOrderModel((B0, B1, B2, Z), (_cubic(3),))

        Y2 = ep.eigenmodes[:, 2, mask]
        mmd3 = zeros(ComplexF64, size(Ψ, 1), 2, ROM)
        for r in 1:ROM
            mmd3[:, 1, r] .= Y2[:, r]
            mmd3[:, 2, r] .= λ[r] .* Y2[:, r]
        end
        lmd3 = left_eigenmode_orders_from_slice(aug.linear_terms, ℓ, collect(λ))[:,
            1:(end - 1), :]

        sd = SpectralData(aug, ep; master = idx, conjugate_permutation = [2, 1])
        @test right_mode_derivatives(sd) == mmd3
        @test left_mode_blocks(sd) == Array(lmd3)
        @test right_modes(sd) == Ψ
        @test left_modes(sd) == ℓ

        mset = all_multiindices_up_to(ROM, 5; min_degree = 1)
        rset = build_resonance_set(aug, :complex_normal_form, mset, ep, 0.05, nothing)
        Wa, Ra = solve_cohomological_problem(aug, mset, λ, Ψ, ℓ, rset;
            master_modes_derivatives = mmd3, left_modes_derivatives = lmd3,
            conjugate_permutation = [2, 1], show_progress = false)
        Wb, Rb = solve_cohomological_problem(aug, mset, sd, rset; show_progress = false)
        @test Wa.poly.coefficients == Wb.poly.coefficients
        @test Ra.poly.coefficients == Rb.poly.coefficients
    end

    @testset "external system: permutation derived, not hand-written" begin
        Ω = 1.3
        fvec = [1.0, 0.5]
        force = MultilinearMap(
            (res, r) -> begin
                @inbounds for j in 1:2
                    iszero(r[j]) || (res .+= r[j] .* fvec)
                end
                res
            end, (0, 0), 1)
        forced = NthOrderModel((B0, B1, B2), (_cubic(2), force),
            ExternalSystem((im * Ω, -im * Ω)))
        mset = all_multiindices_up_to(ROM + 2, 5; min_degree = 1)
        rset = build_resonance_set(forced, :complex_normal_form, mset, ep, 0.05, nothing)

        sd = SpectralData(forced, ep; master = idx, conjugate_permutation = [2, 1])
        Wa, Ra = solve_cohomological_problem(forced, mset, λ, Ψ, ℓ, rset;
            master_modes_derivatives = mmd, left_modes_derivatives = lmd,
            conjugate_permutation = [2, 1, 4, 3], show_progress = false)
        Wb, Rb = solve_cohomological_problem(forced, mset, sd, rset; show_progress = false)
        # The ROM-length master block is extended over the external variables from the
        # external system, reproducing the literal [2, 1, 4, 3] exactly.
        @test Wa.poly.coefficients == Wb.poly.coefficients
        @test Ra.poly.coefficients == Rb.poly.coefficients
    end

    @testset "direct construction from raw arrays (ORD = 1)" begin
        n = 3
        Ψ1 = ComplexF64[1 0; 0 1; 1 1]
        ℓ1 = ComplexF64[1 0; 0 1; 0 0]
        sd = SpectralData(; eigenvalues = ComplexF64[1 + 2im, 1 - 2im],
            right_modes = Ψ1, left_modes = ℓ1,
            outer_eigenvalues = ComplexF64[-3.0],
            conjugate_permutation = [2, 1])
        @test right_modes(sd) == Ψ1
        @test left_modes(sd) == ℓ1
        # ORD == 1 has no derivative blocks at all.
        @test right_mode_derivatives(sd) === nothing
        @test left_mode_blocks(sd) === nothing
        @test outer_eigenvalues(sd) == ComplexF64[-3.0]
    end
end
