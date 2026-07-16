using Test
using MORFE
using MORFE.ParametrisationMethod: create_parametrisation_method_objects
using Random

@testset "RomIO round-trip" begin
    Random.seed!(42)
    mset = all_multiindices_up_to(2, 3; min_degree = 1)
    FOM = 4
    W, R = create_parametrisation_method_objects(mset, 2, FOM, ComplexF64)
    L = length(mset)
    R.poly.coefficients .= randn(ComplexF64, 2, L)
    R.poly.coefficients[:, end] .= 1e-16 + 0im   # below drop threshold → omitted
    W.poly.coefficients .= randn(ComplexF64, FOM, 2, L)

    dir = mktempdir()
    save_rom(dir, W, R; metadata = Pair{String, Any}["model" => "rom-io-test"])

    @test isfile(joinpath(dir, "data", "W.jls"))
    @test isfile(joinpath(dir, "data", "R.jls"))
    @test isfile(joinpath(dir, "data", "R_coefficients.csv"))
    @test isdir(joinpath(dir, "figures"))
    summary = read(joinpath(dir, "summary.txt"), String)
    @test occursin("model: rom-io-test", summary)
    @test occursin("julia_version:", summary)

    exps, coeffs = read_rom_coefficients(joinpath(dir, "data", "R_coefficients.csv"))
    @test size(exps, 1) == L - 1            # the tiny row was dropped
    @test size(exps, 2) == 2
    @test size(coeffs, 2) == 2
    # every surviving row matches the source coefficients exactly (text round-trip)
    lookup = Dict(Tuple(mset.exponents[m]) => m for m in 1:L)
    for r in 1:size(exps, 1)
        m = lookup[Tuple(exps[r, :])]
        for i in 1:2
            @test coeffs[r, i] ≈ R.poly.coefficients[i, m] rtol = 1e-14
        end
    end
end
