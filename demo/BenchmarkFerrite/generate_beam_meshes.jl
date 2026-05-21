"""
Generate all four beam mesh densities for the multi-mesh benchmark suite.

Produces the following files for each nx ∈ {10, 20, 40, 80}:

  demo/BenchmarkFerrite/
	beam_h27_{nx}x2x2.msh          — GMSH format (Ferrite path)

  legacy_morfe/MORFE2.0/input/
	beam_h27_{nx}x2x2.mphtxt       — COMSOL format (legacy MORFE2.0 path)
	beam_h27_{nx}x2x2.jl           — legacy MORFE2.0 input configuration

Target DoF scaling (formula: 75 × (2·NX − 1) free DoFs):
  NX=10  → ~1 425 free DoFs  (≈ 1/4× current)
  NX=20  → ~2 925 free DoFs  (≈ 1/2× current)
  NX=40  → ~5 925 free DoFs  (= 1× current, beam_h27.msh)
  NX=80  → ~11 925 free DoFs (≈ 2× current)

Also copies the NX=40 output to the canonical `beam_h27.{msh,mphtxt}` so that
`benchmark_ferrite.jl` and the existing legacy benchmark are unaffected.

Run once before `benchmark_suite.jl`:
  julia --project demo/BenchmarkFerrite/generate_beam_meshes.jl
"""

using Printf

include(joinpath(@__DIR__, "generate_beam_mesh.jl"))

const NX_SUITE = [10, 20, 40, 80]
const NY_SUITE = 2
const NZ_SUITE = 2

println("=" ^ 70)
println("Beam mesh suite generation  (NY=$(NY_SUITE), NZ=$(NZ_SUITE), H27 elements)")
println("=" ^ 70)
println()

results = []
for nx in NX_SUITE
	println("-" ^ 50)
	println("Generating NX=$(nx) …")
	meta = generate_beam_mesh(nx, NY_SUITE, NZ_SUITE, max_order = 11)
	push!(results, meta)
	println()
end

# Also copy the NX=40 mesh to the canonical names used by benchmark_ferrite.jl
# and the existing legacy benchmark (beam_h27.msh / beam_h27.mphtxt).
idx40 = findfirst(r -> occursin("40x", r.msh_path), results)
if !isnothing(idx40)
	canonical_msh    = joinpath(@__DIR__, "beam_h27.msh")
	canonical_mphtxt = joinpath(
	@__DIR__, "../../legacy_morfe/MORFE2.0/input/beam_h27.mphtxt")
	cp(results[idx40].msh_path, canonical_msh; force = true)
	cp(results[idx40].mphtxt_path, canonical_mphtxt; force = true)
	println("Canonical files updated:")
	println("  ", canonical_msh)
	println("  ", canonical_mphtxt)
	println()
end

# Summary table
println("=" ^ 70)
println("Summary")
println("=" ^ 70)
@printf("%-22s  %6s  %5s  %5s  %-30s  %s\n", "Mesh", "FreeDOF", "#Q9", "#H27", "Dirichlet Q9 indices", "H27 range")
println("-" ^ 70)
for (nx, meta) in zip(NX_SUITE, results)
	dq9     = meta.dirichlet_q9_indices
	hr      = meta.h27_range
	n_q9    = hr.start - 1
	n_h27   = hr.stop - hr.start + 1
	dq9_str = length(dq9) <= 4 ? string(dq9) : "[$(dq9[1])…$(dq9[end])]"
	@printf("%-22s  %6d  %5d  %5d  %-30s  %d:%d\n",
		"beam_h27_$(nx)x2x2", meta.free_dofs, n_q9, n_h27, dq9_str, hr.start, hr.stop)
end
println("=" ^ 70)
println()
println("Done. Run benchmark_suite.jl to execute the benchmark sweep.")
