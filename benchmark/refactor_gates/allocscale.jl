# Does a per-solve cost stay O(1), or does it scale with the problem?
#
# The question this answers is the one worth asking before deciding a cost matters: a
# constant few hundred bytes on a 500 kB solve is noise, whereas anything tracking |mset|
# or n is a real regression however small it looks at one size.
#
# It reports the setup overhead — everything the solve does before the graded loop —
# by differencing against a solve that reuses already-initialised W and R, so the loop
# itself contributes to both and cancels.

using MORFE, LinearAlgebra, SparseArrays, StaticArrays, Printf
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, SpectralData
using MORFE.CohomologicalEquations: solve_cohomological_problem
using MORFE.Resonance: build_resonance_set, ResonanceConfig

cub = MultilinearMap((res, x1, x2, x3) -> (@. res += -1.0 * x1 * x2 * x3), (3, 0))
println("  n   order   |mset|      no conj          conj     conj delta")
for (n, order) in ((6, 3), (6, 7), (30, 5), (30, 7))
    B0 = Matrix(SymTridiagonal(fill(2.0, n), fill(-1.0, n - 1)))
    B2 = Matrix(1.0I, n, n)
    B1 = 0.002 * B2
    dense = NthOrderModel((B0, B1, B2), (cub,))
    model = NthOrderModel((sparse(B0), sparse(B1), sparse(B2)), (cub,))
    ep = spectrum(dense; solver = DefaultEigensolver())
    sd = SpectralData(model, ep; master = master_by_sorting(2))
    mset = all_multiindices_up_to(2, order; min_degree = 1)
    rset = build_resonance_set(model, mset, sd,
        ResonanceConfig(style = :complex_normal_form, tol = 0.05,
            outer_targets = true, warn_outer = false))

    plain() = solve_cohomological_problem(model, mset, sd, rset;
        conjugate_permutation = nothing, show_progress = false)
    conj() = solve_cohomological_problem(model, mset, sd, rset;
        conjugate_permutation = [2, 1], show_progress = false)
    plain()
    conj()
    a = @allocated plain()
    b = @allocated conj()
    @printf("%4d %5d %8d %14d %13d %+12d\n", n, order, length(mset), a, b, b - a)
end
