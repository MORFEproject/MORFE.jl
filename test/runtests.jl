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
    end
end