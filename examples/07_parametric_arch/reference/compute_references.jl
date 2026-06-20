"""
	compute_references.jl

Non-parametric DPIM baselines for the clamped-clamped arch beam.

For each arch height ratio in `H_RATIOS`, this script:
  1. Loads the flat beam mesh.
  2. Physically shifts node y-coordinates to match the sinusoidal arch:
		 y_new = y_old + h · sin(π x / L)
	 producing an arched Ferrite.Grid with all facetsets intact.
  3. Runs the high-level SVK DPIM pipeline (MORFEStructuralSVK extension)
	 on the arched mesh — this gives a standard, non-parametric ROM that
	 captures vibrations around the corresponding arched equilibrium.
  4. Saves W.jls, R.jls, summary.txt, R_coefficients.csv under
		 results/reference/arch_h_<ratio>/

These reference ROMs serve two purposes:
  a) Cross-validation: compare the parametric ROM (main.jl) at θ=0 against
	 the reference at the same h₀/L — they should match to FEM precision.
  b) Documentation of how the SSM shape and reduced dynamics evolve across
	 arch heights, independently of the θ parametrisation.

The flat beam dimensions are 1000 × 10 × 24 mm (standard MORFE benchmark).

Usage (from the repository root):
	julia --project=examples/07_parametric_arch \\
		  examples/07_parametric_arch/reference/compute_references.jl
"""

# -----------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
if !isfile(joinpath(@__DIR__, "..", "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../../..")))
	Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays"])
end
Pkg.instantiate()

using MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps, Printf

SVK = Base.get_extension(MORFE, :MORFEStructuralSVK)
SVK === nothing && error(
	"MORFEStructuralSVK extension not loaded.  Ensure MORFE + Ferrite + " *
	"FerriteGmsh + Arpack + LinearMaps are all `using`-d before calling " *
	"Base.get_extension.")

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

const _MESH = joinpath(@__DIR__, "..", "..", "..", "benchmark", "ferrite",
	"beam_h27_10x2x2.msh")
isfile(_MESH) || error("Mesh not found at $_MESH.  Run generate_beam_mesh.jl first.")

const L = 1000.0      # beam span (mm)
const ORDER = 9       # DPIM expansion order (same as main.jl's max_degree_z)

# Arch height ratios to compute references for.
# Include 0.0 (straight beam) and the value used in main.jl (0.10) plus
# several bracketing values so validation spans θ ∈ [−1, +1].
const H_RATIOS = [0.00, 0.05, 0.10, 0.15, 0.20]

material = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3)
damping = SVK.RayleighDamping(
	α = 0.5369754008568333 / 500.0,
	β = 0.0,
)

const _REF_DIR = joinpath(@__DIR__, "..", "results", "reference")
mkpath(_REF_DIR)

# -----------------------------------------------------------------------
# Helper: build an arched Ferrite.Grid from a flat grid
# -----------------------------------------------------------------------

"""
	make_arch_grid(flat_grid, h, L) -> Ferrite.Grid

Return a new Ferrite.Grid whose nodes are shifted by the sinusoidal arch
displacement  Δy(x₁) = h · sin(π x₁ / L),  keeping all cell connectivity,
facetsets, nodesets, and cellsets intact.

h = 0.0 returns a copy of the flat grid with identical node coordinates.
"""
function make_arch_grid(flat_grid::Ferrite.Grid, h::Float64, L::Float64)
	shifted_nodes = [
		Ferrite.Node(n.x + Ferrite.Vec{3, Float64}((0.0, h * sin(π * n.x[1] / L), 0.0)))
		for n in flat_grid.nodes
	]
	return Ferrite.Grid(
		flat_grid.cells,
		shifted_nodes;
		facetsets = flat_grid.facetsets,
		nodesets = flat_grid.nodesets,
		cellsets = flat_grid.cellsets,
	)
end

# -----------------------------------------------------------------------
# Main loop over arch heights
# -----------------------------------------------------------------------

println("Loading flat beam mesh …")
flat_grid = togrid(_MESH)
println("  Cells : ", getncells(flat_grid), "   Nodes : ", getnnodes(flat_grid))

for h_ratio in H_RATIOS
	h = h_ratio * L
	tag = @sprintf("arch_h_%.3f", h_ratio)
	dir = joinpath(_REF_DIR, tag)
	mkpath(dir)

	println("\n" * "="^60)
	println("Reference ROM  h₀/L = $h_ratio   (h₀ = $h mm)")
	println("="^60)

	println("  Building arched grid …")
	agrid = make_arch_grid(flat_grid, h, L)

	println("  Assembling mechanical model (SVK, order 2) …")
	beam_model = SVK.mechanical_model(
		agrid;
		material = material,
		damping = damping,
		dirichlet = "Dirichlet",
		fe_order = 2,
		quad_order = 3,
	)

	println("  Running DPIM  (master = [1], order = $ORDER) …")
	t_rom = @elapsed rom = SVK.parametrise(beam_model; master = [1], order = ORDER)

	println(@sprintf "  Done in %.2f s." t_rom)

	SVK.save_rom(rom, dir)
	println("  Saved to $dir/")

	# Print first eigenvalue for quick sanity
	eigs = rom.eigenvalues
	ω₀ = abs(eigs[1])
	@printf "  ω₀ = %.6f rad/ms   (f₀ = %.4f kHz)\n" ω₀ (ω₀ / (2π) * 1e-3)
end

println("\n" * "="^60)
println("All references written to $_REF_DIR/")
println("  Subdirectories: ", join(map(r -> @sprintf("arch_h_%.3f", r), H_RATIOS), ", "))
println()
println("Validation: compare each reference's R_coefficients.csv against")
println("  the parametric ROM (main.jl) evaluated at θ = 0 for the")
println("  matching h₀/L ratio.")
