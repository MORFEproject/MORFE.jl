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
                                   left_eigenmode_orders_from_slice
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.Resonance: build_resonance_set, ResonanceConfig
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
    idx = master_by_sorting(ROM)
    mask = [i in idx for i in eachindex(ep.eigenvalues)]

    cnf = ResonanceConfig(style = :complex_normal_form, tol = 0.05, warn_outer = false)

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

        # A SPECTRUM-WIDE involution (one entry per eigenvalue) is taken verbatim — it is
        # exactly what `:detect` derives above, and stating it spares the eigenvector
        # verification. The master restriction is identical to the ROM-length form, so a
        # call site moving between them leaves the solve bit-identical.
        wide = SpectralData(model, ep; master = idx, conjugate_permutation = [2, 1, 4, 3])
        @test wide.conjugate_permutation == [2, 1, 4, 3]
        @test master_conjugate_permutation(wide) == master_conjugate_permutation(det)
        @test outer_conjugate_permutation(wide) == outer_conjugate_permutation(det)

        # Neither ROM-length nor spectrum-length is still a length error.
        @test_throws ArgumentError SpectralData(model, ep; master = idx,
            conjugate_permutation = [2, 1, 3])
        # Right length, but not a permutation / not an involution.
        @test_throws ArgumentError SpectralData(model, ep; master = idx,
            conjugate_permutation = [1, 1])
        @test_throws ArgumentError SpectralData(model, ep; master = idx,
            conjugate_permutation = [2, 1, 4, 4])
        @test_throws ArgumentError SpectralData(model, ep; master = idx,
            conjugate_permutation = [2, 3, 4, 1])

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

    @testset "the solve reads the involution off the bundle" begin
        order = 5
        mset = all_multiindices_up_to(ROM, order; min_degree = 1)
        plain = SpectralData(model, ep; master = idx)
        rset = build_resonance_set(model, mset, plain, cnf)
        for perm in (nothing, [2, 1])
            # Carried by the bundle ...
            sd = SpectralData(model, ep; master = idx, conjugate_permutation = perm)
            Wa, Ra = solve_cohomological_problem(model, mset, sd, rset;
                options = ParametrisationOptions(show_progress = false))
            # ... must be the same solve as stating it at the call site.
            Wb, Rb = solve_cohomological_problem(model, mset, plain, rset;
                conjugate_permutation = perm,
                options = ParametrisationOptions(show_progress = false))
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

        # The reconciled blocks above are the whole content of this case; solving with
        # them simply has to work at the augmented order.
        mset = all_multiindices_up_to(ROM, 5; min_degree = 1)
        rset = build_resonance_set(aug, mset, sd, cnf)
        W, R = solve_cohomological_problem(aug, mset, sd, rset;
            options = ParametrisationOptions(show_progress = false))
        @test size(W.poly.coefficients, 2) == 3     # ORD = 3 order-blocks
        @test size(R.poly.coefficients, 2) == length(mset)
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
        sd = SpectralData(forced, ep; master = idx, conjugate_permutation = [2, 1])
        mset = all_multiindices_up_to(ROM + 2, 5; min_degree = 1)
        rset = build_resonance_set(forced, mset, sd, cnf)

        # The ROM-length master block is extended over the external variables from the
        # external system, and must reproduce the literal [2, 1, 4, 3] exactly.
        Wa, Ra = solve_cohomological_problem(forced, mset, sd, rset;
            conjugate_permutation = [2, 1, 4, 3],
            options = ParametrisationOptions(show_progress = false))
        Wb, Rb = solve_cohomological_problem(forced, mset, sd, rset;
            options = ParametrisationOptions(show_progress = false))
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
