# Opt-in performance gate for the restored sparse default path. Run with
# MORFE_RUN_PERFORMANCE_REGRESSION=1. The legacy harness selects the exact
# direct-GrLex/no-check policy used before operational solver features existed.

using MORFE
using Test
using LinearAlgebra
using SparseArrays
using Statistics
using MORFE.FullOrderModel: NthOrderModel, MultilinearMap
using MORFE.SpectralDecomposition: spectrum, DefaultEigensolver, SpectralData
using MORFE.Resonance: ResonanceConfig

function _median_time(f; samples=7)
    f()
    GC.gc()
    return median(@elapsed(f()) for _ in 1:samples)
end

function _median_allocations(f; samples=5)
    f()
    GC.gc()
    return median(@allocated(f()) for _ in 1:samples)
end

function benchmark_default_against_legacy(solve_with_options, system_dimension)
    legacy = ParametrisationOptions(; backend=:klu, grouping=:off,
        residual_check=:off, show_progress=false, verbose=false)
    current = ParametrisationOptions(; backend=:klu, grouping=:auto,
        residual_check=:off, show_progress=false, verbose=false)
    legacy_solve() = solve_with_options(legacy)
    current_solve() = solve_with_options(current)

    legacy_time = _median_time(legacy_solve)
    current_time = _median_time(current_solve)
    time_regression = current_time / legacy_time - 1
    legacy_allocations = _median_allocations(legacy_solve)
    current_allocations = _median_allocations(current_solve)
    allowed_allocation_growth = 4096

    @test time_regression < 0.0
    @test current_allocations - legacy_allocations <= allowed_allocation_growth
    return (; legacy_time, current_time, time_regression,
        legacy_allocations, current_allocations, allowed_allocation_growth)
end

function _run_default_solver_benchmark()
    side = parse(Int, get(ENV, "MORFE_PERFORMANCE_GRID_SIDE", "30"))
    n = side^2
    line = spdiagm(-1=>fill(-1.0,side-1), 0=>fill(4.2,side),
        1=>fill(-1.0,side-1))
    stiffness = kron(sparse(I,side,side),line) +
        kron(spdiagm(-1=>fill(-1.0,side-1),1=>fill(-1.0,side-1)),
            sparse(I,side,side))
    B0 = Matrix(Symmetric(stiffness))
    B2 = Matrix{Float64}(I,n,n)
    B1 = 0.002 .* B2
    cubic = MultilinearMap(
        (result,x1,x2,x3)->(@. result += -0.02*x1*x2*x3),(3,0))
    frozen = ExternalSystem((0.0,0.0))
    dense = NthOrderModel((B0,B1,B2),(cubic,),frozen)
    sparse_model = NthOrderModel(map(sparse,(B0,B1,B2)),(cubic,),frozen)
    eigenpairs = spectrum(dense;solver=DefaultEigensolver())
    spectral = SpectralData(dense,eigenpairs;master=master_by_sorting(2))
    resonance = ResonanceConfig(style=:complex_normal_form,tol=0.05,
        warn_outer=false)
    solve_with_options(options) = parametrise(sparse_model,spectral,3;
        resonance,options)
    evidence = benchmark_default_against_legacy(solve_with_options,n+2)
    @info "MORFE default-path performance gate" evidence
end

@testset "default solver performance regression" begin
    _run_default_solver_benchmark()
end
