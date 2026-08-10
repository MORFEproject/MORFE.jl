# `ResonanceConfig` must reproduce the loose-keyword path exactly, and the separated
# inner/outer tolerances must fix the combination that used to be a bounds error.
#
# `ResonanceSet` is plain data (BitMatrix), so these are exact comparisons that need no
# cohomological solve at all.

using Test
using LinearAlgebra
using StaticArrays

using MORFE
using MORFE.FullOrderModel: NDOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver,
                                   select_master_modes_by_sorting
using MORFE.Resonance: ResonanceConfig, resolve_tolerances, build_resonance_set,
                       resonance_set_from_complex_normal_form_style,
                       resonance_set_from_graph_style,
                       resonant_multiindices
using MORFE.SpectralDecomposition: SpectralData

@testset "ResonanceConfig" begin
    B0 = [2.0 -1.0; -1.0 2.0]
    B2 = [1.0 0.0; 0.0 1.0]
    B1 = 0.001 * B2
    cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0))
    model = NDOrderModel((B0, B1, B2), (cubic,))
    ROM = 2
    ep = spectrum(model; solver = DefaultEigensolver())
    select_master_modes_by_sorting(ep, ROM)
    idx = findall(ep.master_modes)
    sd = SpectralData(model, ep; master = idx)
    mset = all_multiindices_up_to(ROM, 5; min_degree = 1)

    master_eigs = collect(ComplexF64, ep.eigenvalues[ep.master_modes])

    @testset "config path ≡ loose-keyword path" begin
        for style in (:graph, :complex_normal_form)
            old = build_resonance_set(model, style, mset, ep, 0.05, nothing)
            new = build_resonance_set(model, mset, sd,
                ResonanceConfig(style = style, tol = 0.05, warn_outer = false))
            @test new.inner_resonances == old.inner_resonances
            # Outer targets are opt-in now; the old path always passed them, so compare
            # the inner block (which is the only thing the solve reads) and check the
            # opt-in reproduces the outer block too.
            withouter = build_resonance_set(model, mset, sd,
                ResonanceConfig(style = style, tol = 0.05,
                    outer_targets = true, warn_outer = false))
            @test withouter.outer_resonances == old.outer_resonances
        end
    end

    @testset "structural validation happens at the config site" begin
        @test_throws ArgumentError ResonanceConfig(style = :nonsense)
        # :real_normal_form pairs conjugate targets, so it needs the map.
        @test_throws ArgumentError ResonanceConfig(style = :real_normal_form)
        # A conjugacy_map with any other style would be silently ignored — rejected.
        @test_throws ArgumentError ResonanceConfig(
            style = :complex_normal_form, conjugacy_map = [2, 1])
        @test_throws ArgumentError ResonanceConfig(tol = -1.0)
        @test_throws ArgumentError ResonanceConfig(tol_relative = 0.0)
        @test ResonanceConfig() isa ResonanceConfig
        @test ResonanceConfig(style = :real_normal_form,
            conjugacy_map = [2, 1]) isa ResonanceConfig
    end

    @testset "tol defaults to nothing so guards fire only when set" begin
        c = ResonanceConfig()
        @test c.tol === nothing
        @test c.style === :graph
        # A default-constructed config must not emit the "tolerance unused" notice; if
        # `tol` had a numeric default there would be no way to tell "unset" from "set".
        inner, outer = @test_logs resolve_tolerances(c, master_eigs, ComplexF64[], length(mset))
        @test inner == 0.0
    end

    @testset "tol_relative sizes each target family separately" begin
        outer_eigs = ComplexF64[5.0 + 1.0im, 5.0 - 1.0im, -2.0 + 0.0im]
        c = ResonanceConfig(style = :complex_normal_form, tol_relative = 0.05,
            outer_targets = true, warn_outer = false)
        inner, outer = resolve_tolerances(c, master_eigs, outer_eigs, length(mset))
        @test length(inner) == length(mset)
        @test length(outer) == length(mset)
        # Each family is sized for its OWN target count — this is what makes
        # tol_relative combinable with outer targets at all.
        @test length(inner[1]) == length(master_eigs)
        @test length(outer[1]) == length(outer_eigs)
        @test inner[1] ≈ [0.05 * abs(λ) for λ in master_eigs]
        @test outer[1] ≈ [0.05 * abs(λ) for λ in outer_eigs]
    end

    @testset "per-target tol with more outer than inner targets" begin
        # Previously a bounds error: one tol vector sized for n_int was handed to the
        # n_out-sized outer condition. n_out = 3 > n_int = 2 is exactly the failing case.
        outer_eigs = ComplexF64[5.0 + 1.0im, 5.0 - 1.0im, -2.0 + 0.0im]
        NMON = length(mset)
        inner_tol = [[0.05 * abs(λ) for λ in master_eigs] for _ in 1:NMON]
        outer_tol = [[0.05 * abs(λ) for λ in outer_eigs] for _ in 1:NMON]

        rs = resonance_set_from_complex_normal_form_style(
            mset, master_eigs, inner_tol;
            outer_eigenvalues = outer_eigs, outer_tol = outer_tol)
        @test size(rs.outer_resonances, 1) == 3

        # Omitting outer_tol with a per-target inner tol is now a clear error rather
        # than an out-of-bounds access deep inside is_resonant.
        @test_throws ArgumentError resonance_set_from_complex_normal_form_style(
            mset, master_eigs, inner_tol; outer_eigenvalues = outer_eigs)

        # A scalar tol still serves both families, exactly as before.
        @test resonance_set_from_complex_normal_form_style(
            mset, master_eigs, 0.05; outer_eigenvalues = outer_eigs) isa MORFE.ResonanceSet
    end

    @testset "scalar tol unchanged through the config path" begin
        direct = resonance_set_from_complex_normal_form_style(mset, master_eigs, 0.05)
        viacfg = build_resonance_set(model, mset, sd,
            ResonanceConfig(style = :complex_normal_form, tol = 0.05, warn_outer = false))
        @test viacfg.inner_resonances == direct.inner_resonances
    end
end
