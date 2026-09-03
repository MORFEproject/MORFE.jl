using MORFE
using Aqua
using Test

@testset "Aqua.jl" begin
    Aqua.test_all(
        MORFE;
        ambiguities = true,
        unbound_args = false
    )
    Aqua.test_ambiguities(MORFE; recursive = false)
end