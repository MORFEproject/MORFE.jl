# `ResonanceConfig` must reproduce what the low-level constructors do when called
# directly, and the separated inner/outer tolerances must fix the combination that used to
# be a bounds error.
#
# `ResonanceSet` is plain data (BitMatrix), so these are exact comparisons that need no
# cohomological solve at all.

using Test
using LinearAlgebra
using StaticArrays

using MORFE
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver
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
    model = NthOrderModel((B0, B1, B2), (cubic,))
    ROM = 2
    ep = spectrum(model; solver = DefaultEigensolver())
    idx = master_by_sorting(ROM)
    sd = SpectralData(model, ep; master = idx)
    mset = all_multiindices_up_to(ROM, 5; min_degree = 1)

    master_eigs = collect(ComplexF64, ep.eigenvalues[idx])
    outer_eigs = collect(ComplexF64, ep.eigenvalues[setdiff(1:length(ep.eigenvalues), idx)])

    @testset "config path ≡ the constructor called directly" begin
        direct = Dict(
            :graph => resonance_set_from_graph_style(
                mset, master_eigs, ComplexF64[], outer_eigs, 0.05),
            :complex_normal_form => resonance_set_from_complex_normal_form_style(
                mset, master_eigs, 0.05; outer_eigenvalues = outer_eigs))
        for style in (:graph, :complex_normal_form)
            new = build_resonance_set(model, mset, sd,
                ResonanceConfig(style = style, tol = 0.05, warn_outer = false))
            @test new.inner_resonances == direct[style].inner_resonances
            # Outer targets are opt-in: the inner block is the only thing the solve reads,
            # so the outer one is built only when it was asked for.
            @test new.outer_resonances === nothing
            withouter = build_resonance_set(model, mset, sd,
                ResonanceConfig(style = style, tol = 0.05,
                    outer_targets = true, warn_outer = false))
            @test withouter.outer_resonances == direct[style].outer_resonances
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

@testset "off-manifold near-resonance warning" begin
    # ω₂ = 3ω₁ puts the cubic monomial z₁³ exactly on the OUTER pair's eigenvalue: s = 3iω₁
    # meets λ₂ = 3iω₁. Nothing at degree 1 is resonant, so only a scan over the whole
    # monomial set can see it — and it must report the PAIR once, not each conjugate.
    M = [1.0 0.0; 0.0 1.0]
    C = 1.0e-6 * M
    cubic = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0))
    beam(ω2) = NthOrderModel(([1.0 0.0; 0.0 ω2^2], C, M), (cubic,))

    function run(model, config, mset)
        logs, rset = Test.collect_test_logs() do
            ep = spectrum(model; solver = DefaultEigensolver())
            spec = SpectralData(model, ep; master = [1, 2],
                conjugate_permutation = :detect)
            build_resonance_set(model, mset, spec, config)
        end
        return filter(r -> r.level >= Base.CoreLogging.Warn, logs), rset
    end

    cnf(tol) = ResonanceConfig(style = :complex_normal_form, tol = tol, warn_outer = true)
    ms = all_multiindices_up_to(2, 3; min_degree = 1)

    @testset "one warning per conjugate pair, naming mode and entries" begin
        warns, _ = run(beam(3.0), cnf(0.05), ms)
        @test length(warns) == 1
        msg = string(warns[1].message)
        # The physical mode number is what a user adds to `master`; the spectrum entries
        # are what they index in their own spectrum. Both are reported because they
        # coincide only under an adjacency assumption that need not hold.
        @test occursin("outer physical mode pair 2", msg)
        @test occursin("spectrum entries 3, 4", msg)
        @test occursin("(3, 0)", msg)      # z₁³  ⇒ s = +3iω₁
        @test occursin("(0, 3)", msg)      # z̄₁³ ⇒ s = -3iω₁
        @test occursin("outer", msg) && !occursin("non-master", msg)
    end

    @testset "detuned ⇒ silent" begin
        # ω₂ = 2.5 ω₁ leaves every superharmonic at least 0.5 away from the outer pair.
        warns, _ = run(beam(2.5), cnf(0.05), ms)
        @test isempty(warns)
    end

    @testset "silent by default: :graph carries tol = 0" begin
        # A default config has no tolerance, and `_default_tol(:graph)` is 0.0 — the
        # warning must not start firing merely because someone omitted `tol`.
        warns, _ = run(beam(3.0), ResonanceConfig(), ms)
        @test isempty(warns)
    end

    @testset "per-target tol vector skips the scan instead of throwing" begin
        # The vector is sized for the INNER targets, so it cannot index an outer one.
        # This used to reach `_resolve_outer_tol` and abort the whole solve.
        vtol = [[0.05, 0.05] for _ in 1:length(ms)]
        warns, rset = run(beam(3.0), cnf(vtol), ms)
        @test rset isa MORFE.ResonanceSet
        @test isempty(warns)
    end
end
