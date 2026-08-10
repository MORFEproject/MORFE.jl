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
            @testset "ParametriseEntry" begin
                include("ParametrisationMethod/test_parametrise_entry.jl")
            end
            include("ParametrisationMethod/test_bordered_solver.jl")
            include("ParametrisationMethod/test_external_coupling.jl")
        end
    end
    if should_run("rom_io")
        @testset "RomIO" begin
            include("Export/test_rom_io.jl")
        end
        @testset "RomComparison" begin
            include("Validation/test_rom_comparison.jl")
        end
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
    if should_run("end_to_end")
        @testset "EndToEnd" begin
            #
        end
    end
    if should_run("extensions")
        @testset "MORFESymbolicsExt" begin
            include("Extensions/test_morfe_symbolics.jl")
        end
    end
end

if GROUP == "examples"
    @testset "Examples smoke tests" begin
        @testset "internals" begin
            include(joinpath(
                @__DIR__, "..", "examples", "internals", "demo_polynomials.jl"))
            include(joinpath(@__DIR__, "..", "examples", "internals",
                "demo_multiindices_factorisations.jl"))
            # Writes its lattice figures to a scratch directory rather than the
            # example's own results/, so the test run leaves no artefacts behind.
            mktempdir() do tmp
                withenv("MORFE_LATTICE_OUT" => tmp) do
                    include(joinpath(@__DIR__, "..", "examples", "internals",
                        "multiindex_sets", "main.jl"))
                end
                @test length(readdir(tmp)) == 4
            end
            mktempdir() do tmp
                withenv("MORFE_FOM_OUT" => tmp) do
                    include(joinpath(@__DIR__, "..", "examples", "internals",
                        "full_order_model", "main.jl"))
                end
                # four figures plus the website card's thumbnail
                @test length(readdir(tmp)) == 5
            end
            @test true
        end
        # The remaining examples (02 Gridap, 06 dielectric) manage their own Pkg
        # environment (Pkg.activate + Pkg.instantiate in main.jl). Run them standalone.
        # Ferrite-backed examples and the SVK/Fluid UIs live in MORFEFerrite.jl.
    end
end
