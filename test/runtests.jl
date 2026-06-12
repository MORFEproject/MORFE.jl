using MORFE
using Test
using Random
using LinearAlgebra
using SparseArrays

const GROUP = get(ENV, "GROUP", "all")

should_run(group) = GROUP == "all" || GROUP == group

@testset "MORFE Tests" begin
    if should_run("full_order_model")
        @testset "FullOrderModel" begin
            include("FullOrderModel/test_full_order_model.jl")
            include("FullOrderModel/test_external_system.jl")
            include("FullOrderModel/test_multilinear_maps.jl")
        end
    end
    if should_run("parametrisation_method")
        @testset "ParametrisationMethod" begin
            include("ParametrisationMethod/test_parametrisation_method.jl")
            include("ParametrisationMethod/test_invariance_equation.jl")
            @testset "Resonances" begin
                include("ParametrisationMethod/test_resonances.jl")
            end
            @testset "ConjugateSymmetry" begin
                include("ParametrisationMethod/test_conjugate_symmetry.jl")
            end
        end
    end
    if should_run("spectral_decomposition")
        @testset "SpectralDecomposition" begin
            include("SpectralDecomposition/test_eigenproblems.jl")
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
end

if GROUP == "examples"
    @testset "Examples smoke tests" begin
        @testset "internals" begin
            include(joinpath(@__DIR__, "..", "examples", "internals", "demo_polynomials.jl"))
            include(joinpath(@__DIR__, "..", "examples", "internals", "demo_multiindices_factorisations.jl"))
            @test true
        end
        # Examples 01–05 each manage their own Pkg environment (Pkg.activate + Pkg.instantiate
        # in main.jl) and require FEM backends (Ferrite, Gridap) not present in the MORFE
        # test environment. Run them standalone:
        #   julia --project=examples/01_clamped_beam_ferrite examples/01_clamped_beam_ferrite/main.jl
        #   julia --project=examples/01_clamped_beam_ferrite examples/01_clamped_beam_ferrite/validate.jl
        # See examples/README.md for details.
    end
end

if GROUP == "structural_svk"
    # High-level SVK + Ferrite UI (MORFEStructuralSVK extension). Needs the
    # Ferrite/FerriteGmsh/Arpack/LinearMaps test extras, so run via:
    #   GROUP=structural_svk julia --project -e 'using Pkg; Pkg.test()'
    @testset "MORFEStructuralSVK" begin
        include("StructuralSVK/test_structural_svk.jl")
    end
end