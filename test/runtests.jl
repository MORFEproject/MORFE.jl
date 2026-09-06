using MORFE
using Test
using Random
using LinearAlgebra
using SparseArrays
using Symbolics

const GROUP = get(ENV, "GROUP", "all")

should_run(group) = GROUP == "all" || GROUP == group

@testset "MORFE Tests" begin
    if should_run("full_order_model")
        @testset "FullOrderModel" begin
            include("FullOrderModel/test_full_order_model.jl")
            include("FullOrderModel/test_external_system.jl")
            include("FullOrderModel/test_multilinear_maps.jl")
            include("FullOrderModel/test_multilinear_reference.jl")
            include("FullOrderModel/test_fem_external_arguments.jl")
        end
    end
    if should_run("parametrisation_method")
        @testset "ParametrisationMethod" begin
            include("ParametrisationMethod/test_parametrisation_method.jl")
            include("ParametrisationMethod/test_invariance_equation.jl")
            include("ParametrisationMethod/test_master_mode_orthogonality.jl")
            @testset "Resonances" begin
                include("ParametrisationMethod/test_resonances.jl")
                include("ParametrisationMethod/test_resonance_config.jl")
            end
            @testset "ConjugateSymmetry" begin
                include("ParametrisationMethod/test_conjugate_symmetry.jl")
            end
            @testset "Parametrise" begin
                include("ParametrisationMethod/test_parametrise_contract.jl")
                include("ParametrisationMethod/test_parametrise_unified.jl")
                include("ParametrisationMethod/test_parametrise_entry.jl")
            end
            include("ParametrisationMethod/test_bordered_solver.jl")
            include("ParametrisationMethod/test_cohomological_backends.jl")
            include("ParametrisationMethod/test_cohomological_documentation.jl")
            include("ParametrisationMethod/test_cohomological_uk_english.jl")
            if get(ENV, "MORFE_RUN_PERFORMANCE_REGRESSION", "0") == "1"
                include("ParametrisationMethod/benchmark_default_solver.jl")
            end
            include("ParametrisationMethod/test_external_coupling.jl")
            include("ParametrisationMethod/test_noconj_debug.jl")
        end
    end
    if should_run("rom_io")
        @testset "RomIO" begin
            include("Export/test_rom_io.jl")
            include("Export/test_observables.jl")
            include("Export/test_normal_form_branch.jl")
        end
        @testset "RomComparison" begin
            include("Validation/test_rom_comparison.jl")
        end
        include("Validation/test_invariance_error.jl")
    end
    if should_run("spectral_decomposition")
        @testset "SpectralDecomposition" begin
            include("SpectralDecomposition/test_eigenproblems.jl")
            include("SpectralDecomposition/test_spectral_data.jl")
        end
    end
    if should_run("utils")
        @testset "Utils" begin
            @testset "Multiindices" begin
                include("Utils/test_multiindices.jl")
            end
            @testset "Polynomials" begin
                include("Utils/test_polynomials.jl")
            end
            @testset "Realification" begin
                include("Utils/test_realification.jl")
            end
        end
    end
    if should_run("extensions")
        @testset "MORFESymbolicsExt" begin
            include("Extensions/test_morfe_symbolics.jl")
        end
        @testset "MORFEArpackExt" begin
            include("Extensions/test_arpack.jl")
        end
        @testset "MORFEPlotsExt" begin
            include("Extensions/test_plots.jl")
        end
    end
    if should_run("aqua")
        include("Quality/aqua.jl")
    end
end

if GROUP == "examples"
    # `examples/` is no longer tracked: the examples the website publishes live in the
    # MORFEExamples repository, and the rest are untracked working copies. These smoke
    # tests therefore run only where a working copy is present, and skip otherwise --
    # the same treatment MORFEFerrite gives its untracked Joukowski sources.
    const _internals = normpath(joinpath(@__DIR__, "..", "examples", "internals"))

    @testset "Examples smoke tests" begin
        @testset "internals" begin
            for demo in ("demo_polynomials.jl", "demo_multiindices_factorisations.jl")
                path = joinpath(_internals, demo)
                if isfile(path)
                    include(path)
                else
                    @info "Skipping $demo: no examples/ working copy in this checkout"
                    @test_skip isfile(path)
                end
            end
            # Both drivers write their figures to a scratch directory rather than the
            # example's own results/, so the test run leaves no artefacts behind.
            for (driver, out_var, n_figures) in (
                ("multiindex_sets", "MORFE_LATTICE_OUT", 4),
            # four figures plus the website card's thumbnail
                ("full_order_model", "MORFE_FOM_OUT", 5))
                path = joinpath(_internals, driver, "main.jl")
                if !isfile(path)
                    @info "Skipping $driver/main.jl: no examples/ working copy in this checkout"
                    @test_skip isfile(path)
                    continue
                end
                mktempdir() do tmp
                    withenv(out_var => tmp) do
                        include(path)
                    end
                    @test length(readdir(tmp)) == n_figures
                end
            end
            @test true
        end
        # The remaining examples (02 Gridap, 06 dielectric) manage their own Pkg
        # environment (Pkg.activate + Pkg.instantiate in main.jl). Run them standalone.
        # Ferrite-backed examples and the SVK/Fluid UIs live in MORFEFerrite.jl.
    end
end
