# Benchmark Environment

## Hardware

| Property | Value |
|----------|-------|
| CPU | Apple M2 Pro |
| Physical cores | 12 |
| Logical cores | 12 |
| RAM | 16 GB |

## Software

| Property | Value |
|----------|-------|
| OS | macOS 26.4.1 (Build 25E253) |
| Julia | 1.12.5 |
| MORFE.jl | v3.0.0 |
| Ferrite.jl | v1.4.0 |

## Benchmark settings

All runs were executed on 2026-05-22 with the following fixed parameters:

| Parameter | Value |
|-----------|-------|
| ROM dimension (`ROM`) | 2 |
| External directions (`N_EXT`) | 2 |
| Expansion degree (`max_degree`) | 11 |
| Total monomials | 1364 |
| Element type | H27 (27-node hexahedron) |
| Mesh topology | `Nx × 2 × 2` beam |

The four meshes vary only in the number of elements along the beam axis (`Nx`):

| Run folder | Mesh | FOM (DOFs) | Solve time (s) | Solve allocs (GB) |
|------------|------|------------|---------------|-------------------|
| `beam_h27_10x2x2_degree11_20260522T000835` | 10×2×2 | 1 425 | 354.2 | 82.3 |
| `beam_h27_20x2x2_degree11_20260522T001432` | 20×2×2 | 2 925 | 698.3 | 161.5 |
| `beam_h27_40x2x2_degree11_20260522T002611` | 40×2×2 | 5 925 | 1 398.6 | 322.9 |
| `beam_h27_80x2x2_degree11_20260522T004930` | 80×2×2 | 11 925 | 2 814.0 | 646.5 |

Each run was launched sequentially (one at a time) from `benchmark_suite.jl`.
Times are wall-clock seconds as reported by `@timed`; allocations are the total
heap bytes reported by Julia for the `solve_cohomological_problem` call.
