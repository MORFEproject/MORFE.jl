using MORFE
using Test

const GROUP = get(ENV, "GROUP", "all")
# Run tests
@testset "MORFE Tests" begin
    if GROUP in ("all", "tests")
        @testset "FullOrderModel" begin
            include("FullOrderModel/test_full_order_model.jl")
        end
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
    if GROUP in ("all", "parametrisation")
    end
end

#Run demos 
@testset "Morfe Demos" begin
    if GROUP in ("all", "demos")
        @testset "FullOrderModel" begin
            redirect_stdout(devnull) do
                include(joinpath(
                    @__DIR__, "..", "demo/FullOrderModel/demo_NDOrderModel.jl"))
            end
        end
        @testset "ParametrisationMethod" begin
            # ignore println
            redirect_stdout(devnull) do
                include(joinpath(
                    @__DIR__, "..", "demo/ParametrisationMethod/demo_invariance_equation.jl"))
                include(joinpath(
                    @__DIR__, "..", "demo/ParametrisationMethod/demo_lower_order_couplings.jl"))
                include(joinpath(
                    @__DIR__, "..",
                    "demo/ParametrisationMethod/demo_master_mode_orthogonality.jl"))
                include(joinpath(
                    @__DIR__, "..", "demo/ParametrisationMethod/demo_multilinear_terms.jl"))
                include(joinpath(
                    @__DIR__, "..", "demo/ParametrisationMethod/demo_parametrisation_method.jl"))
                include(joinpath(
                    @__DIR__, "..", "demo/ParametrisationMethod/demo_resonances.jl"))
            end
        end
    end
end