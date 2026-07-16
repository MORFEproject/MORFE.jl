using Test
using MORFE
using MORFE.RomIO: write_rom_coefficients_csv
using Random

@testset "RomComparison" begin
    Random.seed!(7)

    @testset ":exact over shared monomials (graded truncation)" begin
        mset_full = all_multiindices_up_to(2, 5; min_degree = 1)
        exps_full = mset_full.exponents
        L = length(exps_full)
        coeffs = randn(ComplexF64, 2, L)

        dir = mktempdir()
        full_csv = joinpath(dir, "full.csv")
        write_rom_coefficients_csv(full_csv, exps_full, coeffs)

        # truncated "FAST" run = the degree ≤ 3 subset of the same coefficients
        keep = [m for m in 1:L if sum(exps_full[m]) ≤ 3]
        trunc_csv = joinpath(dir, "trunc.csv")
        write_rom_coefficients_csv(trunc_csv, exps_full[keep], coeffs[:, keep])

        pass, dev, report = compare_rom_coefficients(trunc_csv, full_csv)
        @test pass
        @test dev < 1e-12
        @test occursin("shared rows", report)

        # perturb one coefficient beyond rtol → fail
        bad = copy(coeffs[:, keep])
        bad[1, 2] *= (1 + 1e-3)
        bad_csv = joinpath(dir, "bad.csv")
        write_rom_coefficients_csv(bad_csv, exps_full[keep], bad)
        pass_bad, dev_bad, _ = compare_rom_coefficients(bad_csv, full_csv)
        @test !pass_bad
        @test dev_bad > 1e-4
    end

    @testset ":gauge_invariant under a real modal gauge" begin
        # NVAR = 3: two modal coordinates (z₁, z̄₁) + one external (η).
        exps = [1 0 0; 0 1 0; 1 0 1; 0 1 1; 2 1 0; 1 2 0]
        Lg = size(exps, 1)
        ref = randn(ComplexF64, 2, Lg)

        # gauge z → a·z with real a: row coefficients scale by a^(modal_deg − 1)
        a = 1.7
        new = copy(ref)
        for r in 1:Lg
            modal_deg = exps[r, 1] + exps[r, 2]
            new[:, r] .*= a^(modal_deg - 1)
        end

        dir = mktempdir()
        ref_csv = joinpath(dir, "ref.csv")
        new_csv = joinpath(dir, "new.csv")
        write_rom_coefficients_csv(ref_csv, [exps[r, :] for r in 1:Lg], ref)
        write_rom_coefficients_csv(new_csv, [exps[r, :] for r in 1:Lg], new)

        # raw comparison must fail (the cubic rows scaled by a² = 2.89)…
        pass_exact, _, _ = compare_rom_coefficients(new_csv, ref_csv; mode = :exact)
        @test !pass_exact
        # …but the gauge-invariant comparison passes.
        pass_gi, dev_gi, report = compare_rom_coefficients(new_csv, ref_csv;
            mode = :gauge_invariant, n_master = 2)
        @test pass_gi
        @test dev_gi < 1e-12
        @test occursin("modal-degree-1", report)
    end

    @testset "error paths" begin
        dir = mktempdir()
        exps = [1 0; 0 1]
        c = randn(ComplexF64, 2, 2)
        csv = joinpath(dir, "x.csv")
        write_rom_coefficients_csv(csv, [exps[r, :] for r in 1:2], c)
        @test_throws ArgumentError compare_rom_coefficients(csv, csv; mode = :bogus)
    end
end
