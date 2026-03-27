include(joinpath(@__DIR__, "../src/MORFE.jl"))
using .MORFE
using StaticArrays

N = 5
# Define MultiindexSet
multiindexset = all_multiindices_up_to(N, 3)
println(multiindexset)
println("=== Using SingleResonance ===")
res1 = SingleResonance(1, SVector{N, Int}(1, 0, 0, 0, 0))
res2 = SingleResonance(2, SVector{N, Int}(0, 1, 0, 0, 0))
res3 = SingleResonance(3, SVector{N, Int}(0, 0, 1, 0, 0))
res4 = SingleResonance(4, SVector{N, Int}(0, 0, 0, 1, 0))
res5 = SingleResonance(5, SVector{N, Int}(0, 0, 0, 0, 1))
# This is not in multiindexset so it is skipped but there is printed a warning
res6 = SingleResonance(1, SVector{N, Int}(1, 0, 1, 2, 3))

#build resonance set
res_set1 = resonance_set(multiindexset, (res1, res2, res3, res4, res5, res6))
println(res_set1.resonances)

println("=== Using Tolerance ===")
eigs = SVector{5, ComplexF64}(1.0, 2.0, 3.0 + 3.0im, 4.0, 5.0)
res_set_2 = resonance_set_from_eigenvalues(multiindexset, eigs, 1e-9)
# println(res_set_2.resonances)
# 2:1 resonance between λ1 and λ2 
index = MORFE.Multiindices.find_in_set(multiindexset, [2, 0, 0, 0, 0])
println("Resonances for (2,0,0,0,0): $(res_set_2.resonances[index])")
println("\n" * "="^80 * "\n")
println("Demo finished successfully.")