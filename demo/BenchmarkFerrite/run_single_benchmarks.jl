"""
Single-run script: beam_h27_10x2x2, max_degree = 13.
Usage: julia --project demo/BenchmarkFerrite/run_single_benchmarks.jl
"""

include(joinpath(@__DIR__, "benchmark_suite.jl"))

benchmark_mesh(10, 2, 2; max_degree = 10)
